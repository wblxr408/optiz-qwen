"""Attention backend selection for the split prefill/decode attention strategy.

Measured on PPU-ZW810E (see docs/ppu_optimization_design.md): the two phases of
generation want *different* attention kernels.

    prefill   sdpa                50.68 ms  vs flash_attention_2  56.41 ms
    decode    flash_attention_2    6.24 ms  vs sdpa                8.91 ms

Using one backend for both phases therefore gives up one metric to win the other.
The split is possible because ``modeling_qwen3_5.py`` resolves the kernel *per
forward call*::

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward)

and because a captured CUDA graph freezes whatever kernels were live at capture
time.  So capturing the decode graph while the config says ``flash_attention_2``
and then restoring ``sdpa`` yields FA2 decode replays with sdpa prefill dispatch.

This module owns only the config mutation.  Graph capture lives in
``optiz_qwen.scheduling.cuda_graph_decode``.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

#: Backends that are known to work on the PPU target for the decode graph.
#: ``eager`` is deliberately excluded: CUDA graph capture is invalidated under
#: it on PPU (capture aborts), which is a hardware/runtime constraint rather
#: than a harness problem.
GRAPH_SAFE_BACKENDS = ("sdpa", "flash_attention_2")

#: Backends accepted for the (uncaptured) prefill pass.
PREFILL_BACKENDS = ("sdpa", "flash_attention_2", "eager")

DEFAULT_PREFILL_BACKEND = "sdpa"
DEFAULT_DECODE_BACKEND = "flash_attention_2"

ATTENTION_BACKEND_ENV_KEYS = (
    "OPTIZ_QWEN_ATTN_PREFILL",
    "OPTIZ_QWEN_ATTN_DECODE",
)


def resolved_prefill_backend() -> str:
    """Prefill backend requested through the environment, or the default."""

    value = os.environ.get("OPTIZ_QWEN_ATTN_PREFILL", "").strip().lower()
    return value or DEFAULT_PREFILL_BACKEND


def resolved_decode_backend() -> str:
    """Decode backend requested through the environment, or the default."""

    value = os.environ.get("OPTIZ_QWEN_ATTN_DECODE", "").strip().lower()
    return value or DEFAULT_DECODE_BACKEND


def current_attention_backend(model: Any) -> str | None:
    """Report the backend the model would dispatch on the next forward call."""

    config = getattr(model, "config", None)
    if config is None:
        return None
    return getattr(config, "_attn_implementation", None)


def set_attention_backend(model: Any, backend: str) -> str | None:
    """Point the model at ``backend`` for subsequent forward calls.

    Returns the previously configured backend so callers can restore it.  Both
    the top-level config and ``text_config`` are updated: the decoder layers
    read the text config, while some shared helpers read the top-level one.
    The vision config is intentionally left alone -- the vision tower is 12.2%
    of prefill and is not part of this optimization.
    """

    normalized = str(backend).strip().lower()
    if not normalized:
        raise ValueError("attention backend must be a non-empty string")
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("model has no config; cannot select an attention backend")
    previous = getattr(config, "_attn_implementation", None)
    config._attn_implementation = normalized
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        text_config._attn_implementation = normalized
    return previous


@contextmanager
def attention_backend(model: Any, backend: str | None) -> Iterator[str | None]:
    """Temporarily select ``backend``, restoring the previous one on exit.

    A ``None`` backend is a no-op, so callers can pass an optional override
    without branching.
    """

    if backend is None:
        yield current_attention_backend(model)
        return
    previous = set_attention_backend(model, backend)
    try:
        yield previous
    finally:
        if previous is not None:
            set_attention_backend(model, previous)


def validate_graph_backend(backend: str) -> str:
    """Reject decode backends that cannot survive CUDA graph capture on PPU."""

    normalized = str(backend).strip().lower()
    if normalized not in GRAPH_SAFE_BACKENDS:
        raise ValueError(
            f"attention backend {normalized!r} is not graph-safe on the PPU target; "
            f"choose one of {GRAPH_SAFE_BACKENDS}. "
            "eager invalidates CUDA graph capture on PPU-ZW810E."
        )
    return normalized
