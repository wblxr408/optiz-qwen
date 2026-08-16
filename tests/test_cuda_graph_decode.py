"""Tests for the CUDA-graph decode scheduling layer.

Capture itself needs the target accelerator, so these tests cover everything
around it: the env-switch contract, the ``StaticCache`` construction guard, the
error paths that protect a graph from being replayed against the wrong state,
and the decode loop that replays it.  The hardware latency numbers live in
``test_ppu_hybrid_regression.py``.
"""

from __future__ import annotations

import pytest
import torch

from optiz_qwen.scheduling import (
    CudaGraphDecodeReport,
    CudaGraphDecoder,
    build_static_cache,
    cuda_graph_decode_enabled,
    resolved_max_cache_len,
    resolved_warmup_steps,
    run_greedy_prefill_decode,
)


class FakeOutputs:
    def __init__(self, logits, past_key_values=None) -> None:
        self.logits = logits
        self.past_key_values = past_key_values


class FakeGreedyModel:
    def __init__(self) -> None:
        self.calls = []
        self.call_inference_mode: list[bool] = []
        self.vocab_size = 8

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        self.call_inference_mode.append(torch.is_inference_mode_enabled())
        input_ids = kwargs["input_ids"]
        step = input_ids.shape[-1]
        logits = torch.full((1, step, self.vocab_size), -1000.0)
        logits[0, -1, step % self.vocab_size] = 1000.0
        return FakeOutputs(logits=logits, past_key_values={"step": step})


class FakeStaticCache:
    """Stands in for ``StaticCache``: fixed identity, resettable."""

    def __init__(self) -> None:
        self.max_cache_len = 2048
        self.resets = 0
        self.reset_under_inference_mode: list[bool] = []

    def reset(self) -> None:
        self.resets += 1
        self.reset_under_inference_mode.append(torch.is_inference_mode_enabled())


class FakeGraphDecoder:
    """Replays a deterministic token sequence, recording replay arguments."""

    def __init__(self, cache, *, tokens=(4, 5, 6)) -> None:
        self._cache = cache
        self._tokens = list(tokens)
        self.advances: list[tuple[int, int]] = []
        self.vocab_size = 8

    @property
    def cache(self):
        return self._cache

    def advance(self, *, token_id: int, position: int):
        self.advances.append((token_id, position))
        emitted = self._tokens[min(len(self.advances) - 1, len(self._tokens) - 1)]
        logits = torch.full((1, self.vocab_size), -1000.0)
        logits[0, emitted] = 1000.0
        return logits


def _inputs(prompt_len: int = 3) -> dict:
    return {
        "input_ids": torch.arange(1, prompt_len + 1, dtype=torch.long).view(1, -1),
        "attention_mask": torch.ones((1, prompt_len), dtype=torch.long),
    }


def test_switch_is_off_by_default_and_accepts_truthy_spellings(monkeypatch) -> None:
    monkeypatch.delenv("OPTIZ_QWEN_CUDA_GRAPH_DECODE", raising=False)
    assert cuda_graph_decode_enabled() is False

    for value in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("OPTIZ_QWEN_CUDA_GRAPH_DECODE", value)
        assert cuda_graph_decode_enabled() is True

    for value in ("0", "false", "off", ""):
        monkeypatch.setenv("OPTIZ_QWEN_CUDA_GRAPH_DECODE", value)
        assert cuda_graph_decode_enabled() is False


def test_cache_len_and_warmup_resolvers(monkeypatch) -> None:
    monkeypatch.delenv("OPTIZ_QWEN_CUDA_GRAPH_MAX_CACHE_LEN", raising=False)
    monkeypatch.delenv("OPTIZ_QWEN_CUDA_GRAPH_WARMUP_STEPS", raising=False)
    assert resolved_max_cache_len() == 2048
    assert resolved_warmup_steps() == 3

    monkeypatch.setenv("OPTIZ_QWEN_CUDA_GRAPH_MAX_CACHE_LEN", "4096")
    monkeypatch.setenv("OPTIZ_QWEN_CUDA_GRAPH_WARMUP_STEPS", "1")
    assert resolved_max_cache_len() == 4096
    assert resolved_warmup_steps() == 1

    monkeypatch.setenv("OPTIZ_QWEN_CUDA_GRAPH_MAX_CACHE_LEN", "0")
    with pytest.raises(ValueError):
        resolved_max_cache_len()

    monkeypatch.setenv("OPTIZ_QWEN_CUDA_GRAPH_WARMUP_STEPS", "0")
    with pytest.raises(ValueError):
        resolved_warmup_steps()


def test_build_static_cache_requires_text_config() -> None:
    class NoTextConfig:
        config = object()

    with pytest.raises(ValueError, match="text_config"):
        build_static_cache(NoTextConfig(), max_cache_len=128, device="cpu", dtype=torch.float32)


def test_decoder_refuses_to_replay_before_capture() -> None:
    decoder = CudaGraphDecoder(FakeGreedyModel(), FakeStaticCache(), capture_backend="flash_attention_2")

    assert decoder.captured is False
    assert decoder.capture_prompt_tokens is None
    with pytest.raises(RuntimeError, match="not been captured"):
        decoder.advance(token_id=1, position=3)


def test_decoder_report_records_both_backends() -> None:
    cache = FakeStaticCache()
    decoder = CudaGraphDecoder(FakeGreedyModel(), cache, capture_backend="flash_attention_2", warmup_steps=5)

    report = decoder.report(prefill_backend="sdpa")

    assert isinstance(report, CudaGraphDecodeReport)
    assert report.capture_backend == "flash_attention_2"
    assert report.prefill_backend == "sdpa"
    assert report.captured is False
    assert report.max_cache_len == 2048
    assert report.warmup_steps == 5
    assert report.to_dict()["capture_backend"] == "flash_attention_2"


def test_graph_decode_resets_cache_and_pins_prefill_cache_position() -> None:
    model = FakeGreedyModel()
    cache = FakeStaticCache()
    decoder = FakeGraphDecoder(cache)

    run_greedy_prefill_decode(
        model,
        _inputs(3),
        max_new_tokens=3,
        tokenizer=object(),
        kv_cache=cache,
        graph_decoder=decoder,
    )

    # Warmup and capture wrote decode entries into this cache; the reset is what
    # makes prefill land at positions 0..n-1 again.
    assert cache.resets == 1
    assert torch.equal(model.calls[0]["cache_position"], torch.arange(3))
    assert model.calls[0]["past_key_values"] is cache


def test_graph_decode_replays_once_per_token_at_advancing_positions() -> None:
    model = FakeGreedyModel()
    cache = FakeStaticCache()
    decoder = FakeGraphDecoder(cache, tokens=(4, 5, 6))

    generated_ids, stats = run_greedy_prefill_decode(
        model,
        _inputs(3),
        max_new_tokens=4,
        tokenizer=object(),
        kv_cache=cache,
        graph_decoder=decoder,
    )

    # Exactly one model call: prefill.  Every decode step is a graph replay.
    assert len(model.calls) == 1
    assert len(decoder.advances) == 3
    positions = [position for _, position in decoder.advances]
    assert positions == [3, 4, 5]
    # The first replay consumes the token argmaxed from prefill logits.
    assert decoder.advances[0][0] == generated_ids[0, 0].item()
    assert stats.generated_tokens == 4
    assert generated_ids.shape == (1, 4)


def test_graph_decode_stops_at_eos() -> None:
    cache = FakeStaticCache()
    decoder = FakeGraphDecoder(cache, tokens=(4, 7, 7))

    generated_ids, stats = run_greedy_prefill_decode(
        FakeGreedyModel(),
        _inputs(3),
        max_new_tokens=8,
        tokenizer=object(),
        kv_cache=cache,
        eos_token_id=7,
        graph_decoder=decoder,
    )

    assert generated_ids[0, -1].item() == 7
    assert stats.generated_tokens == 3
    assert len(decoder.advances) == 2


def test_graph_decode_fires_the_post_decode_callback_per_token() -> None:
    cache = FakeStaticCache()
    decoder = FakeGraphDecoder(cache)
    seen: list[int] = []

    run_greedy_prefill_decode(
        FakeGreedyModel(),
        _inputs(3),
        max_new_tokens=3,
        tokenizer=object(),
        kv_cache=cache,
        graph_decoder=decoder,
        post_decode_callback=lambda: seen.append(1),
    )

    assert len(seen) == 2


def test_graph_decode_resets_and_prefills_under_inference_mode() -> None:
    """Regression: PPU raised on ``reset()`` outside ``inference_mode``.

    The cache's conv/recurrent state tensors are allocated inside
    ``inference_mode`` during graph capture, which makes them inference tensors.
    ``zero_()`` on an inference tensor outside ``inference_mode`` is a hard
    RuntimeError, and prefill writes into those same buffers, so both the reset
    and the prefill forward must run under ``inference_mode`` on this path.
    """

    model = FakeGreedyModel()
    cache = FakeStaticCache()
    decoder = FakeGraphDecoder(cache)

    run_greedy_prefill_decode(
        model,
        _inputs(3),
        max_new_tokens=2,
        tokenizer=object(),
        kv_cache=cache,
        graph_decoder=decoder,
    )

    assert cache.reset_under_inference_mode == [True]
    assert model.call_inference_mode == [True]


def test_non_graph_path_keeps_the_no_grad_guard() -> None:
    """The baseline path must not be silently switched to inference_mode.

    Inference tensors cannot be mutated later, so widening the guard for the
    non-hybrid path would constrain callers that legitimately post-process the
    returned cache.
    """

    model = FakeGreedyModel()

    run_greedy_prefill_decode(
        model,
        _inputs(3),
        max_new_tokens=2,
        tokenizer=object(),
    )

    assert model.call_inference_mode[0] is False


def test_graph_decoder_requires_its_own_cache() -> None:
    decoder = FakeGraphDecoder(FakeStaticCache())

    with pytest.raises(ValueError, match="requires the StaticCache"):
        run_greedy_prefill_decode(
            FakeGreedyModel(),
            _inputs(3),
            max_new_tokens=2,
            tokenizer=object(),
            kv_cache=None,
            graph_decoder=decoder,
        )

    with pytest.raises(ValueError, match="different KV cache"):
        run_greedy_prefill_decode(
            FakeGreedyModel(),
            _inputs(3),
            max_new_tokens=2,
            tokenizer=object(),
            kv_cache=FakeStaticCache(),
            graph_decoder=decoder,
        )


class FakeDeferredCache(FakeStaticCache):
    defer_prefill_cache_injection = True


def test_graph_decoder_is_mutually_exclusive_with_the_packed_kv_chain() -> None:
    cache = FakeDeferredCache()
    decoder = FakeGraphDecoder(cache)

    with pytest.raises(ValueError, match="deferred packed-KV chain"):
        run_greedy_prefill_decode(
            FakeGreedyModel(),
            _inputs(3),
            max_new_tokens=2,
            tokenizer=object(),
            kv_cache=cache,
            graph_decoder=decoder,
        )
