"""Algorithm-side compression layer from the competition architecture."""

from optiz_qwen.compression.kivi_external import (
    KiviConfig,
    KiviIntegrationError,
    KiviQbMatmulStatus,
    KiviSourceStatus,
    KiviUnsupportedModelError,
    apply_kivi_config_to_transformers_config,
    infer_upstream_model_family,
    inspect_kivi_source,
    inspect_qb_matmul_kernel,
    load_upstream_kivi_model_class,
    load_upstream_quant_module,
)
from optiz_qwen.compression.qwen35_kivi_cache import (
    KiviAttentionLayer,
    Qwen35KiviCache,
    Qwen35KiviCacheReport,
    build_qwen35_kivi_cache,
)

__all__ = [
    "KiviAttentionLayer",
    "KiviConfig",
    "KiviIntegrationError",
    "KiviQbMatmulStatus",
    "KiviSourceStatus",
    "KiviUnsupportedModelError",
    "Qwen35KiviCache",
    "Qwen35KiviCacheReport",
    "apply_kivi_config_to_transformers_config",
    "build_qwen35_kivi_cache",
    "infer_upstream_model_family",
    "inspect_kivi_source",
    "inspect_qb_matmul_kernel",
    "load_upstream_kivi_model_class",
    "load_upstream_quant_module",
]
