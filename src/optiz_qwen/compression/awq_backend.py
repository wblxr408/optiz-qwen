"""Lightweight AWQ backend readiness probes.

This module uses ``importlib.util.find_spec`` only. It does not import torch,
transformers, AutoAWQ, awq, or autoawq modules, and it does not load models or
write artifacts.
"""

from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass

SUPPORTED_BACKENDS = {"autoawq"}


@dataclass(frozen=True)
class AWQBackendReadiness:
    """Dry-run readiness metadata for a planned AWQ execution backend."""

    backend_name: str
    package_available: bool
    can_quantize: bool
    reason: str
    recommended_environment: str

    def to_dict(self) -> dict[str, str | bool]:
        return asdict(self)


def _package_available(module_names: tuple[str, ...]) -> bool:
    return any(importlib.util.find_spec(name) is not None for name in module_names)


def probe_awq_backend(backend_name: str) -> AWQBackendReadiness:
    """Return readiness metadata for a planned AWQ backend without importing it."""

    normalized = backend_name.strip().lower()
    if normalized not in SUPPORTED_BACKENDS:
        supported = ", ".join(sorted(SUPPORTED_BACKENDS))
        raise ValueError(f"unsupported AWQ backend: {backend_name}; supported backends: {supported}")

    if normalized == "autoawq":
        package_available = _package_available(("awq", "autoawq"))
        if package_available:
            reason = (
                "AutoAWQ import target was found, but Phase 5 only performs a "
                "readiness probe; real quantization is intentionally disabled."
            )
        else:
            reason = (
                "AutoAWQ import target was not found. This is expected on local "
                "developer machines unless the real quantization environment has "
                "been prepared."
            )
        return AWQBackendReadiness(
            backend_name="autoawq",
            package_available=package_available,
            can_quantize=False,
            reason=reason,
            recommended_environment=(
                "Use a dedicated Linux quantization host with a suitable NVIDIA GPU, "
                "matching CUDA stack, AutoAWQ-compatible packages, Qwen3.5-2B VLM "
                "weights, and a small calibration set. Validate BF16 baseline first."
            ),
        )

    raise AssertionError(f"unhandled supported backend: {normalized}")
