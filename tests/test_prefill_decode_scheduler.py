from __future__ import annotations

import torch

from optiz_qwen.scheduling import (
    PREFILL_LOGITS_TO_KEEP_ENV,
    prefill_last_logit_only_enabled,
    run_greedy_prefill_decode,
)


class FakeOutputs:
    def __init__(self, logits, past_key_values=None) -> None:
        self.logits = logits
        self.past_key_values = past_key_values


class FakeGreedyModel:
    def __init__(self) -> None:
        self.calls = []
        self.vocab_size = 8

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        step = input_ids.shape[-1]
        logits = torch.full((1, step, self.vocab_size), -1000.0)
        logits[0, -1, step % self.vocab_size] = 1000.0
        return FakeOutputs(logits=logits, past_key_values={"step": step})


def test_run_greedy_prefill_decode_uses_past_key_values() -> None:
    model = FakeGreedyModel()
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
    }

    generated_ids, stats = run_greedy_prefill_decode(
        model,
        inputs,
        max_new_tokens=3,
        tokenizer=object(),
    )

    assert generated_ids.shape == (1, 3)
    assert stats.prompt_tokens == 3
    assert stats.generated_tokens == 3
    assert len(model.calls) == 3
    assert model.calls[0]["use_cache"] is True
    assert model.calls[1]["past_key_values"] == {"step": 3}
    assert model.calls[2]["past_key_values"] == {"step": 1}


def test_run_greedy_prefill_decode_extends_attention_mask_before_first_decode() -> None:
    model = FakeGreedyModel()
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
    }

    run_greedy_prefill_decode(
        model,
        inputs,
        max_new_tokens=2,
        tokenizer=object(),
    )

    assert torch.equal(model.calls[1]["attention_mask"], torch.tensor([[1, 1, 1, 1]], dtype=torch.long))


class FakePrepareModel(FakeGreedyModel):
    def __init__(self) -> None:
        super().__init__()
        self.prepared_position_ids = None

    def _prepare_position_ids_for_generation(self, input_ids, model_kwargs):
        attention_mask = model_kwargs["attention_mask"]
        return attention_mask.long().cumsum(-1).unsqueeze(0)

    def prepare_inputs_for_generation(self, **kwargs):
        self.prepared_position_ids = kwargs.get("position_ids")
        return kwargs


def test_run_greedy_prefill_decode_passes_prepared_position_ids_to_decode() -> None:
    model = FakePrepareModel()
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
    }

    run_greedy_prefill_decode(
        model,
        inputs,
        max_new_tokens=2,
        tokenizer=object(),
    )

    assert model.prepared_position_ids is not None
    assert torch.equal(model.prepared_position_ids, torch.tensor([[[1, 2, 3, 4]]], dtype=torch.long))


class FakeDeferredCache:
    defer_prefill_cache_injection = True

    class Config:
        activation_threshold = 1

    qserve_config = Config()

    def __init__(self) -> None:
        self.adopted = None

    def adopt_native_prefill_cache(self, native_cache) -> None:
        self.adopted = native_cache


def test_run_greedy_prefill_decode_defers_custom_cache_until_after_native_prefill() -> None:
    model = FakeGreedyModel()
    cache = FakeDeferredCache()
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
    }

    run_greedy_prefill_decode(model, inputs, max_new_tokens=2, tokenizer=object(), kv_cache=cache)

    assert "past_key_values" not in model.calls[0]
    assert cache.adopted == {"step": 3}
    assert model.calls[1]["past_key_values"] is cache


def test_run_greedy_prefill_decode_keeps_native_cache_below_kv_threshold() -> None:
    class DelayedCache(FakeDeferredCache):
        class Config:
            activation_threshold = 99

        qserve_config = Config()

    model = FakeGreedyModel()
    cache = DelayedCache()
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
    }

    run_greedy_prefill_decode(model, inputs, max_new_tokens=2, tokenizer=object(), kv_cache=cache)

    assert cache.adopted is None
    assert model.calls[1]["past_key_values"] == {"step": 3}


def test_run_greedy_prefill_decode_keeps_answer_prefix_native_before_activation() -> None:
    class WarmupCache(FakeDeferredCache):
        class Config:
            activation_threshold = 1
            decode_warmup_tokens = 3

        qserve_config = Config()

    model = FakeGreedyModel()
    cache = WarmupCache()
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
    }

    run_greedy_prefill_decode(model, inputs, max_new_tokens=5, tokenizer=object(), kv_cache=cache)

    assert cache.adopted == {"step": 1}
    assert model.calls[1]["past_key_values"] == {"step": 3}
    assert model.calls[2]["past_key_values"] == {"step": 1}
    assert model.calls[3]["past_key_values"] == {"step": 1}


def test_run_greedy_prefill_decode_runs_post_prefill_callback_before_decode() -> None:
    model = FakeGreedyModel()
    callback_state = {"calls": 0}
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
    }

    run_greedy_prefill_decode(
        model,
        inputs,
        max_new_tokens=2,
        tokenizer=object(),
        post_prefill_callback=lambda: callback_state.__setitem__("calls", callback_state["calls"] + 1),
    )

    assert callback_state["calls"] == 1


def test_prefill_last_logit_only_defaults_on_and_respects_the_kill_switch(monkeypatch) -> None:
    monkeypatch.delenv(PREFILL_LOGITS_TO_KEEP_ENV, raising=False)
    assert prefill_last_logit_only_enabled() is True
    for value in ("1", "true", "YES", "on"):
        monkeypatch.setenv(PREFILL_LOGITS_TO_KEEP_ENV, value)
        assert prefill_last_logit_only_enabled() is True
    for value in ("0", "false", "no", "off"):
        monkeypatch.setenv(PREFILL_LOGITS_TO_KEEP_ENV, value)
        assert prefill_last_logit_only_enabled() is False


def test_prefill_requests_only_the_last_logit_row(monkeypatch) -> None:
    """Greedy prefill reads ``logits[:, -1, :]``, so the lm_head only needs one row.

    On PPU the full-sequence projection costs 3.04 ms of a ~52 ms prefill at
    vocab 248320; asking for one position removes that from TTFT.
    """

    monkeypatch.delenv(PREFILL_LOGITS_TO_KEEP_ENV, raising=False)
    model = FakeGreedyModel()
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
    }

    _, stats = run_greedy_prefill_decode(model, inputs, max_new_tokens=2, tokenizer=object())

    assert model.calls[0]["logits_to_keep"] == 1
    assert stats.prefill_logits_trimmed is True
    # Decode already runs one position at a time; trimming there would be noise.
    assert "logits_to_keep" not in model.calls[1]


def test_prefill_keeps_full_logits_when_the_switch_is_off(monkeypatch) -> None:
    monkeypatch.setenv(PREFILL_LOGITS_TO_KEEP_ENV, "0")
    model = FakeGreedyModel()
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
    }

    _, stats = run_greedy_prefill_decode(model, inputs, max_new_tokens=2, tokenizer=object())

    assert "logits_to_keep" not in model.calls[0]
    assert stats.prefill_logits_trimmed is False


class FakeStrictSignatureModel(FakeGreedyModel):
    """A model whose forward has no ``logits_to_keep`` and no ``**kwargs``."""

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=True):
        return FakeGreedyModel.__call__(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

    def __call__(self, **kwargs):
        return self.forward(**kwargs)


def test_prefill_skips_trimming_when_the_model_cannot_accept_it(monkeypatch) -> None:
    """The probe is by signature, so an unsupported model never raises mid-prefill."""

    monkeypatch.delenv(PREFILL_LOGITS_TO_KEEP_ENV, raising=False)
    model = FakeStrictSignatureModel()
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
    }

    _, stats = run_greedy_prefill_decode(model, inputs, max_new_tokens=2, tokenizer=object())

    assert "logits_to_keep" not in model.calls[0]
    assert stats.prefill_logits_trimmed is False


class FakeTrimmingModel(FakeGreedyModel):
    """Honors ``logits_to_keep`` the way ``modeling_qwen3_5`` does."""

    def __call__(self, **kwargs):
        outputs = FakeGreedyModel.__call__(self, **kwargs)
        keep = kwargs.get("logits_to_keep")
        if keep:
            outputs.logits = outputs.logits[:, -keep:, :]
        return outputs


def test_trimmed_prefill_yields_the_same_first_token(monkeypatch) -> None:
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
    }

    monkeypatch.setenv(PREFILL_LOGITS_TO_KEEP_ENV, "0")
    full_ids, full_stats = run_greedy_prefill_decode(
        FakeTrimmingModel(), inputs, max_new_tokens=4, tokenizer=object()
    )

    monkeypatch.setenv(PREFILL_LOGITS_TO_KEEP_ENV, "1")
    trimmed_ids, trimmed_stats = run_greedy_prefill_decode(
        FakeTrimmingModel(), inputs, max_new_tokens=4, tokenizer=object()
    )

    assert full_stats.prefill_logits_trimmed is False
    assert trimmed_stats.prefill_logits_trimmed is True
    assert torch.equal(full_ids, trimmed_ids)
