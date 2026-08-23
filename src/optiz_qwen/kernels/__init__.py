"""Kernel fusion and hot-path operator optimization layer."""

from .qserve_int4_attention import (
    qserve_int4_decode_attention,
    qserve_int4_split_decode_attention,
    triton_int4_decode_available,
)
from .qwen35_chunk_delta import (
    ChunkDeltaComparison,
    ChunkDeltaKernelReport,
    compare_chunk_delta_kernel,
    install_qwen35_chunk_delta_kernel,
    qwen35_chunk_delta_reference,
)
from .qwen35_fused_attention import install_qwen35_fused_attention

__all__ = [
    "ChunkDeltaComparison",
    "ChunkDeltaKernelReport",
    "compare_chunk_delta_kernel",
    "install_qwen35_chunk_delta_kernel",
    "install_qwen35_fused_attention",
    "qserve_int4_decode_attention",
    "qserve_int4_split_decode_attention",
    "qwen35_chunk_delta_reference",
    "triton_int4_decode_available",
]

from .attention_backend import (
    attention_backend,
    resolved_decode_backend,
    resolved_prefill_backend,
    set_attention_backend,
)
__all__ += ["attention_backend", "resolved_decode_backend", "resolved_prefill_backend", "set_attention_backend"]
