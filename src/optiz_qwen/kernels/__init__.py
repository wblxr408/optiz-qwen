"""Kernel fusion and hot-path operator optimization layer."""

from .qserve_int4_attention import qserve_int4_decode_attention, triton_int4_decode_available
from .qwen35_fused_attention import install_qwen35_fused_attention

__all__ = ["install_qwen35_fused_attention", "qserve_int4_decode_attention", "triton_int4_decode_available"]
