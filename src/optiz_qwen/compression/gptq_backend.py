"""Lightweight llmcompressor GPTQ backend readiness probes.

This module only uses :func:`importlib.util.find_spec`. It does not import
llmcompressor, compressed-tensors, transformers, torch, or any model class.
"""

from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass

SUPPORTED_BACKENDS = {"llmcompressor"}


@dataclass(frozen=True)
class GPTQBackendReadiness:
    """Dependency readiness for the llmcompressor GPTQ execution path."""

    backend_name: str
    llmcompressor_available: bool
    compressed_tensors_available: bool
    transformers_available: bool
    package_available: bool
    can_quantize: bool
    reason: str
    recommended_environment: str

    def to_dict(self) -> dict[str, str | bool]:
        return asdict(self)


def _package_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def probe_gptq_backend(backend_name: str) -> GPTQBackendReadiness:
    """Probe GPTQ dependencies without importing them or loading a model."""

    normalized = backend_name.strip().lower()
    if normalized not in SUPPORTED_BACKENDS:
        supported = ", ".join(sorted(SUPPORTED_BACKENDS))
        raise ValueError(
            f"unsupported GPTQ backend: {backend_name}; supported backends: {supported}"
        )

    llmcompressor_available = _package_available("llmcompressor")
    compressed_tensors_available = _package_available("compressed_tensors")
    transformers_available = _package_available("transformers")
    missing = [
        name
        for name, available in (
            ("llmcompressor", llmcompressor_available),
            ("compressed_tensors", compressed_tensors_available),
            ("transformers", transformers_available),
        )
        if not available
    ]
    package_available = not missing

    if package_available:
        reason = (
            "llmcompressor, compressed_tensors, and transformers were found by "
            "find_spec. This lightweight probe does not validate CUDA readiness, "
            "model compatibility, or available GPU memory."
        )
    else:
        reason = "Missing required GPTQ import targets: " + ", ".join(missing)

    return GPTQBackendReadiness(
        backend_name="llmcompressor",
        llmcompressor_available=llmcompressor_available,
        compressed_tensors_available=compressed_tensors_available,
        transformers_available=transformers_available,
        package_available=package_available,
        can_quantize=package_available,
        reason=reason,
        recommended_environment=(
            "Use the verified WSL .venv-awq-linux environment with a compatible "
            "NVIDIA CUDA stack, local Qwen3.5-2B weights, and local MMBench "
            "calibration data."
        ),
    )
