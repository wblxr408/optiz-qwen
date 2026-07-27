from __future__ import annotations

import torch

from optiz_qwen.scheduling import run_greedy_prefill_decode


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
