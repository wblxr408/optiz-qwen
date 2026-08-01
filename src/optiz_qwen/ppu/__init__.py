"""PPU-specific adaptation layer."""

from .compatibility import PpuCompatibilityStatus, inspect_ppu_compatibility
from .qwen35_delta import (
    Qwen35PpuDeltaConfig,
    Qwen35PpuDeltaReport,
    get_qwen35_ppu_delta_runtime,
    install_qwen35_ppu_delta_kernel,
    load_qwen35_ppu_delta_extension,
)
from .qwen35_gdn_projection import (
    GdnDecodeProjectionFusionReport,
    get_qwen35_gdn_decode_projection_runtime,
    install_qwen35_gdn_decode_projection_fusion,
)

__all__ = [
    "GdnDecodeProjectionFusionReport",
    "PpuCompatibilityStatus",
    "Qwen35PpuDeltaConfig",
    "Qwen35PpuDeltaReport",
    "get_qwen35_gdn_decode_projection_runtime",
    "get_qwen35_ppu_delta_runtime",
    "inspect_ppu_compatibility",
    "install_qwen35_gdn_decode_projection_fusion",
    "install_qwen35_ppu_delta_kernel",
    "load_qwen35_ppu_delta_extension",
]
