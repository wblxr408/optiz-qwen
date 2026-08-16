"""Kernel fusion and hot-path operator optimization layer."""

from .attention_backend import (
    ATTENTION_BACKEND_ENV_KEYS,
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
from .qserve_int4_attention import (
    qserve_int4_decode_attention,
    qserve_int8_decode_attention,
    qserve_int4_split_decode_attention,
    triton_int4_decode_available,
)
from .qwen35_fused_attention import install_qwen35_fused_attention
from .vision_prefill_sync import (
    VISION_SYNC_ELISION_ENV,
    chunk_lengths_from_grid,
    elide_vision_attention_host_sync,
    vision_sync_elision_available,
    vision_sync_elision_enabled,
)

__all__ = [
    "ATTENTION_BACKEND_ENV_KEYS",
    "DEFAULT_DECODE_BACKEND",
    "DEFAULT_PREFILL_BACKEND",
    "GRAPH_SAFE_BACKENDS",
    "VISION_SYNC_ELISION_ENV",
    "attention_backend",
    "chunk_lengths_from_grid",
    "current_attention_backend",
    "elide_vision_attention_host_sync",
    "install_qwen35_fused_attention",
    "qserve_int4_decode_attention",
    "qserve_int8_decode_attention",
    "qserve_int4_split_decode_attention",
    "resolved_decode_backend",
    "resolved_prefill_backend",
    "set_attention_backend",
    "triton_int4_decode_available",
    "validate_graph_backend",
    "vision_sync_elision_available",
    "vision_sync_elision_enabled",
]
