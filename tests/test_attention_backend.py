"""Tests for the split prefill/decode attention backend selection.

These are host-side tests: they cover the config mutation contract that makes
the hybrid possible.  The latency claims that motivate the split are hardware
facts and live in ``test_ppu_hybrid_regression.py`` against the saved artifact.
"""

from __future__ import annotations

import pytest

from optiz_qwen.kernels import (
    DEFAULT_DECODE_BACKEND,
    DEFAULT_PREFILL_BACKEND,
    GRAPH_SAFE_BACKENDS,
    attention_backend,
    current_attention_backend,
    resolved_decode_backend,
    resolved_prefill_backend,
    set_attention_backend,
    validate_graph_backend,
)


class FakeConfig:
    def __init__(self, backend: str = "sdpa", *, with_text_config: bool = True) -> None:
        self._attn_implementation = backend
        self.text_config = FakeConfig(backend, with_text_config=False) if with_text_config else None
        self.vision_config = FakeConfig(backend, with_text_config=False) if with_text_config else None


class FakeModel:
    def __init__(self, backend: str = "sdpa") -> None:
        self.config = FakeConfig(backend)


def test_defaults_encode_the_measured_split() -> None:
    # prefill sdpa 50.68 ms vs FA2 56.41 ms; decode FA2 6.24 ms vs sdpa 8.91 ms.
    assert DEFAULT_PREFILL_BACKEND == "sdpa"
    assert DEFAULT_DECODE_BACKEND == "flash_attention_2"


def test_set_attention_backend_updates_text_config_and_returns_previous() -> None:
    model = FakeModel("sdpa")

    previous = set_attention_backend(model, "flash_attention_2")

    assert previous == "sdpa"
    assert model.config._attn_implementation == "flash_attention_2"
    assert model.config.text_config._attn_implementation == "flash_attention_2"


def test_set_attention_backend_leaves_the_vision_tower_alone() -> None:
    model = FakeModel("sdpa")

    set_attention_backend(model, "flash_attention_2")

    # The vision tower is 12.2% of prefill and is not part of this optimization.
    assert model.config.vision_config._attn_implementation == "sdpa"


def test_set_attention_backend_normalizes_and_rejects_empty() -> None:
    model = FakeModel("sdpa")

    set_attention_backend(model, "  FLASH_ATTENTION_2 ")
    assert current_attention_backend(model) == "flash_attention_2"

    with pytest.raises(ValueError):
        set_attention_backend(model, "   ")


def test_attention_backend_restores_on_exit_and_on_exception() -> None:
    model = FakeModel("sdpa")

    with attention_backend(model, "flash_attention_2") as previous:
        assert previous == "sdpa"
        assert current_attention_backend(model) == "flash_attention_2"
    assert current_attention_backend(model) == "sdpa"

    with pytest.raises(RuntimeError):
        with attention_backend(model, "flash_attention_2"):
            raise RuntimeError("boom")
    assert current_attention_backend(model) == "sdpa"


def test_attention_backend_none_is_a_noop() -> None:
    model = FakeModel("sdpa")

    with attention_backend(model, None) as previous:
        assert previous == "sdpa"
        assert current_attention_backend(model) == "sdpa"


def test_validate_graph_backend_rejects_eager() -> None:
    # eager invalidates CUDA graph capture on PPU-ZW810E: capture aborts.
    with pytest.raises(ValueError, match="graph-safe"):
        validate_graph_backend("eager")

    for backend in GRAPH_SAFE_BACKENDS:
        assert validate_graph_backend(backend) == backend


def test_env_overrides_are_honored(monkeypatch) -> None:
    monkeypatch.setenv("OPTIZ_QWEN_ATTN_PREFILL", "EAGER")
    monkeypatch.setenv("OPTIZ_QWEN_ATTN_DECODE", "sdpa")

    assert resolved_prefill_backend() == "eager"
    assert resolved_decode_backend() == "sdpa"

    monkeypatch.delenv("OPTIZ_QWEN_ATTN_PREFILL")
    monkeypatch.delenv("OPTIZ_QWEN_ATTN_DECODE")

    assert resolved_prefill_backend() == DEFAULT_PREFILL_BACKEND
    assert resolved_decode_backend() == DEFAULT_DECODE_BACKEND
