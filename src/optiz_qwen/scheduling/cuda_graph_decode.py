"""CUDA-Graph decode for Qwen3.5-2B on the PPU target.

Why this exists
---------------
Decode on PPU-ZW810E is **dispatch-bound, not memory-bound**.  Measured on the
target with the real checkpoint (docs/ppu_optimization_design.md):

    baseline decode step            19.3 - 21.1 ms   (47 - 51 tok/s)
    HBM floor for 4.426 GB weights   2.20 ms         (454 tok/s)
    overhead factor                  8.9x
    CUDA kernels per decode step     5778
    CPU launch time, 20 queued steps 20.656 ms/step vs 20.678 ms wall

The CPU cannot issue work fast enough to keep the device busy, so the fix is to
stop issuing per-step: capture one decode step as a CUDA graph and replay it.

Two preconditions had to be discovered on this hardware:

1. ``position_ids`` and ``cache_position`` must be passed explicitly.  Letting
   the model derive them raises ``CUDA error: operation not permitted when
   stream is capturing`` (``hggcErrorStreamCaptureUnsupported``) from
   ``compute_3d_position_ids``, which does a host-to-device copy via
   ``position_ids.view(1, 1, -1).expand(3, batch, -1).to(device)``.
2. The KV cache must be a ``StaticCache``.  ``DynamicCache`` concatenates into a
   fresh allocation each step, so addresses baked into the graph go stale and
   replay emits degenerate output with a frozen ``seq_length``.

Verified on target: replay is bit-faithful against eager decode over the same
static cache, and decode cost is flat across a 256-token generation
(quartile means ``[6.33, 6.39, 6.33, 6.32]`` ms).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

DEFAULT_MAX_CACHE_LEN = 2048
DEFAULT_WARMUP_STEPS = 3

CUDA_GRAPH_ENV_KEYS = (
    "OPTIZ_QWEN_CUDA_GRAPH_DECODE",
    "OPTIZ_QWEN_CUDA_GRAPH_MAX_CACHE_LEN",
    "OPTIZ_QWEN_CUDA_GRAPH_WARMUP_STEPS",
)


@dataclass(frozen=True)
class CudaGraphDecodeReport:
    """What was actually captured, for artifact and provenance recording."""

    enabled: bool
    captured: bool
    max_cache_len: int
    warmup_steps: int
    capture_backend: str | None
    prefill_backend: str | None
    capture_prompt_tokens: int | None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def cuda_graph_decode_enabled() -> bool:
    """Whether the hybrid CUDA-graph decode path is switched on."""

    value = os.environ.get("OPTIZ_QWEN_CUDA_GRAPH_DECODE", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def resolved_max_cache_len() -> int:
    raw = os.environ.get("OPTIZ_QWEN_CUDA_GRAPH_MAX_CACHE_LEN", "").strip()
    if not raw:
        return DEFAULT_MAX_CACHE_LEN
    value = int(raw)
    if value <= 0:
        raise ValueError("OPTIZ_QWEN_CUDA_GRAPH_MAX_CACHE_LEN must be positive")
    return value


def resolved_warmup_steps() -> int:
    raw = os.environ.get("OPTIZ_QWEN_CUDA_GRAPH_WARMUP_STEPS", "").strip()
    if not raw:
        return DEFAULT_WARMUP_STEPS
    value = int(raw)
    if value < 1:
        raise ValueError("OPTIZ_QWEN_CUDA_GRAPH_WARMUP_STEPS must be >= 1")
    return value


def build_static_cache(model: Any, *, max_cache_len: int, device: Any, dtype: Any) -> Any:
    """Allocate the fixed-address KV cache the graph will be captured against.

    ``StaticCache`` is built from ``config.text_config`` so the hybrid layer
    stack (18 linear-attention + 6 full-attention layers) materializes the right
    mix of ``StaticLayer`` and ``LinearAttentionLayer`` entries.
    """

    from transformers.cache_utils import StaticCache

    text_config = getattr(getattr(model, "config", None), "text_config", None)
    if text_config is None:
        raise ValueError("model config has no text_config; cannot size a StaticCache")
    return StaticCache(
        config=text_config,
        max_cache_len=int(max_cache_len),
        max_batch_size=1,
        device=device,
        dtype=dtype,
    )


class CudaGraphDecoder:
    """Owns one captured decode step and replays it per generated token.

    The static input tensors are allocated once and mutated in place; the graph
    reads exactly those addresses, so ``advance`` only has to ``fill_`` them and
    call ``replay``.  ``logits`` reads back from the captured output tensor.

    The instance is single-cache by construction: the graph is bound to the
    ``StaticCache`` it was captured against.  A different cache means a new
    decoder.
    """

    def __init__(
        self,
        model: Any,
        cache: Any,
        *,
        capture_backend: str,
        warmup_steps: int = DEFAULT_WARMUP_STEPS,
    ) -> None:
        self._model = model
        self._cache = cache
        self._capture_backend = capture_backend
        self._warmup_steps = int(warmup_steps)
        self._graph = None
        self._outputs = None
        self._input_ids = None
        self._position_ids = None
        self._cache_position = None
        self._capture_prompt_tokens: int | None = None

    @property
    def captured(self) -> bool:
        return self._graph is not None

    @property
    def cache(self) -> Any:
        return self._cache

    @property
    def capture_prompt_tokens(self) -> int | None:
        return self._capture_prompt_tokens

    def capture(self, *, token_id: int, position: int, device: Any) -> None:
        """Capture a single decode step under the decode attention backend.

        ``position`` only seeds the static tensors; replay overwrites them, so
        the captured graph is not bound to this particular prompt length.  The
        caller is responsible for having prefilled ``self._cache`` first, and for
        discarding those writes afterwards (``cache.reset()`` then re-prefill)
        because warmup and capture both write decode entries into the cache.
        """

        import torch

        from optiz_qwen.kernels.attention_backend import (
            attention_backend,
            validate_graph_backend,
        )

        if self.captured:
            raise RuntimeError("decode graph already captured")
        backend = validate_graph_backend(self._capture_backend)

        self._input_ids = torch.full((1, 1), int(token_id), dtype=torch.long, device=device)
        # M-RoPE wants (3, batch, seq); passing it explicitly avoids the
        # in-forward host-to-device copy that aborts capture on PPU.
        self._position_ids = torch.full((3, 1, 1), int(position), dtype=torch.long, device=device)
        self._cache_position = torch.full((1,), int(position), dtype=torch.long, device=device)

        step_kwargs = {
            "input_ids": self._input_ids,
            "past_key_values": self._cache,
            "position_ids": self._position_ids,
            "cache_position": self._cache_position,
            "use_cache": True,
        }

        with attention_backend(self._model, backend), torch.inference_mode():
            # Warm up on a side stream so allocator and autotuning work happens
            # before capture rather than inside it.
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(self._warmup_steps):
                    self._model(**step_kwargs)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._outputs = self._model(**step_kwargs)
            torch.cuda.synchronize()

        self._graph = graph
        self._capture_prompt_tokens = int(position)

    def advance(self, *, token_id: int, position: int) -> Any:
        """Replay one decode step for ``token_id`` at ``position``.

        Returns the last-position logits tensor, which aliases the captured
        output buffer and is overwritten by the next ``advance``.
        """

        if not self.captured:
            raise RuntimeError("decode graph has not been captured")
        self._input_ids.fill_(int(token_id))
        self._position_ids.fill_(int(position))
        self._cache_position.fill_(int(position))
        self._graph.replay()
        return self._outputs.logits[:, -1, :]

    def report(self, *, prefill_backend: str | None, enabled: bool = True) -> CudaGraphDecodeReport:
        return CudaGraphDecodeReport(
            enabled=enabled,
            captured=self.captured,
            max_cache_len=int(getattr(self._cache, "max_cache_len", 0) or 0),
            warmup_steps=self._warmup_steps,
            capture_backend=self._capture_backend,
            prefill_backend=prefill_backend,
            capture_prompt_tokens=self._capture_prompt_tokens,
        )
