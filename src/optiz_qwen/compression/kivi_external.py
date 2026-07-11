"""Adapter for using the upstream jy-yuan/KIVI implementation.

This module intentionally does not reimplement KIVI quantization.  It
locates a checked-out upstream KIVI repository, validates the expected
source layout, and exposes a narrow loader for the model classes that the
upstream project currently ships.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


UPSTREAM_REPO_URL = "https://github.com/jy-yuan/KIVI"
DEFAULT_RELATIVE_SOURCE_PATH = Path("artifacts") / "third_party" / "KIVI"

SUPPORTED_UPSTREAM_MODEL_FAMILIES: dict[str, tuple[str, str]] = {
    "llama": ("models.llama_kivi", "LlamaForCausalLM_KIVI"),
    "mistral": ("models.mistral_kivi", "MistralForCausalLM_KIVI"),
}

REQUIRED_UPSTREAM_FILES = (
    "README.md",
    "LICENSE",
    "models/llama_kivi.py",
    "models/mistral_kivi.py",
    "quant/new_pack.py",
    "quant/matmul.py",
)


class KiviIntegrationError(RuntimeError):
    """Raised when the upstream KIVI source cannot be used."""


class KiviUnsupportedModelError(KiviIntegrationError):
    """Raised when upstream KIVI has no matching model implementation."""


@dataclass(frozen=True)
class KiviConfig:
    """Runtime preset for an upstream KIVI KV-cache experiment."""

    k_bits: int = 2
    v_bits: int = 2
    group_size: int = 32
    residual_length: int = 32
    model_family: str = "auto"
    source_path: str | None = None
    use_flash: bool = True

    def validate(self) -> None:
        if self.k_bits not in {2, 4, 8}:
            raise ValueError("KIVI k_bits must be one of {2, 4, 8}.")
        if self.v_bits not in {2, 4, 8}:
            raise ValueError("KIVI v_bits must be one of {2, 4, 8}.")
        if self.group_size <= 0:
            raise ValueError("KIVI group_size must be positive.")
        if self.residual_length <= 0:
            raise ValueError("KIVI residual_length must be positive.")
        if self.residual_length % self.group_size != 0:
            raise ValueError("KIVI residual_length must be divisible by group_size.")


@dataclass(frozen=True)
class KiviSourceStatus:
    """Current availability report for the external upstream source tree."""

    path: Path
    exists: bool
    missing_files: tuple[str, ...]
    commit: str | None
    license_name: str | None
    supported_model_families: tuple[str, ...]

    @property
    def usable_source_tree(self) -> bool:
        return self.exists and not self.missing_files


@dataclass(frozen=True)
class KiviQbMatmulStatus:
    """Availability report for upstream KIVI qB MatMul kernels."""

    path: Path
    source_tree_ready: bool
    importable: bool
    has_triton_outer_kernel: bool
    has_cuda_outer_kernel: bool
    requires_cuda_extension: bool
    import_error: str | None
    integration_status: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_source_path(root: Path | None = None) -> Path:
    base = root if root is not None else project_root()
    return base / DEFAULT_RELATIVE_SOURCE_PATH


def resolve_source_path(source_path: str | Path | None = None) -> Path:
    if source_path is None:
        return default_source_path()
    return Path(source_path).expanduser().resolve()


def inspect_kivi_source(source_path: str | Path | None = None) -> KiviSourceStatus:
    path = resolve_source_path(source_path)
    exists = path.exists()
    missing = tuple(
        relative for relative in REQUIRED_UPSTREAM_FILES if not (path / relative).exists()
    )
    commit = _git_commit(path) if exists else None
    license_name = _detect_license(path / "LICENSE") if exists else None
    return KiviSourceStatus(
        path=path,
        exists=exists,
        missing_files=missing,
        commit=commit,
        license_name=license_name,
        supported_model_families=tuple(sorted(SUPPORTED_UPSTREAM_MODEL_FAMILIES)),
    )


def inspect_qb_matmul_kernel(source_path: str | Path | None = None) -> KiviQbMatmulStatus:
    """Inspect whether upstream KIVI qB MatMul kernels can be imported.

    This is a capability probe only.  A positive import check is not enough
    to claim the Qwen3.5 VLM adapter has replaced native attention.
    """

    status = inspect_kivi_source(source_path)
    import_error = None
    module = None
    if status.usable_source_tree:
        previous_modules = {
            name: sys.modules.get(name)
            for name in ("quant", "quant.matmul", "kivi_gemv")
        }
        for name in previous_modules:
            sys.modules.pop(name, None)
        with _temporary_sys_path(status.path):
            try:
                module = importlib.import_module("quant.matmul")
            except Exception as exc:  # pragma: no cover - depends on optional CUDA deps
                import_error = repr(exc)
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    else:
        import_error = f"source tree is not ready; missing={list(status.missing_files)}"

    has_triton = bool(module is not None and hasattr(module, "triton_bmm_fA_qB_outer"))
    has_cuda = bool(module is not None and hasattr(module, "cuda_bmm_fA_qB_outer"))
    requires_cuda_extension = bool(import_error and "kivi_gemv" in import_error)
    return KiviQbMatmulStatus(
        path=status.path,
        source_tree_ready=status.usable_source_tree,
        importable=module is not None,
        has_triton_outer_kernel=has_triton,
        has_cuda_outer_kernel=has_cuda,
        requires_cuda_extension=requires_cuda_extension,
        import_error=import_error,
        integration_status="not_integrated_for_qwen3_5_vlm_attention",
    )


def infer_upstream_model_family(model_config: Any) -> str:
    """Infer the upstream KIVI model family from a HF-style config or dict."""

    model_type = _config_value(model_config, "model_type")
    architectures = _config_value(model_config, "architectures") or ()
    text_config = _config_value(model_config, "text_config")
    text_model_type = _config_value(text_config, "model_type") if text_config else None

    haystack = " ".join(
        str(item).lower()
        for item in (model_type, text_model_type, *architectures)
        if item is not None
    )
    if "mistral" in haystack:
        return "mistral"
    if "llama" in haystack:
        return "llama"
    if "qwen" in haystack:
        raise KiviUnsupportedModelError(
            "Upstream jy-yuan/KIVI ships Llama and Mistral model classes, "
            "but no Qwen3.5 VLM attention/cache implementation. Keep this as "
            "a separate module until a Qwen adapter is added or upstream adds support."
        )
    raise KiviUnsupportedModelError(
        "Could not infer a KIVI-supported model family. Supported upstream "
        "families are: llama, mistral."
    )


def apply_kivi_config_to_transformers_config(
    transformers_config: Any,
    kivi_config: KiviConfig,
) -> Any:
    """Attach upstream KIVI config attributes to a transformers config object."""

    kivi_config.validate()
    setattr(transformers_config, "k_bits", kivi_config.k_bits)
    setattr(transformers_config, "v_bits", kivi_config.v_bits)
    setattr(transformers_config, "group_size", kivi_config.group_size)
    setattr(transformers_config, "residual_length", kivi_config.residual_length)
    if kivi_config.use_flash:
        setattr(transformers_config, "use_flash", True)
    return transformers_config


def load_upstream_kivi_model_class(
    model_family: str,
    source_path: str | Path | None = None,
) -> type[Any]:
    """Load a model class directly from the upstream KIVI source tree."""

    normalized_family = model_family.lower().strip()
    if normalized_family not in SUPPORTED_UPSTREAM_MODEL_FAMILIES:
        raise KiviUnsupportedModelError(
            f"Unsupported upstream KIVI model family: {model_family!r}. "
            f"Supported families: {', '.join(sorted(SUPPORTED_UPSTREAM_MODEL_FAMILIES))}."
        )

    status = inspect_kivi_source(source_path)
    if not status.usable_source_tree:
        raise KiviIntegrationError(
            "Upstream KIVI source tree is not ready. "
            f"path={status.path}; missing={list(status.missing_files)}. "
            "Run scripts/prepare_kivi_upstream.ps1 first."
        )

    module_name, class_name = SUPPORTED_UPSTREAM_MODEL_FAMILIES[normalized_family]
    previous_modules = {
        name: sys.modules.get(name)
        for name in ("models", module_name)
    }
    for name in previous_modules:
        sys.modules.pop(name, None)
    with _temporary_sys_path(status.path):
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - depends on optional CUDA deps
            raise KiviIntegrationError(
                "Found upstream KIVI source, but importing its model class failed. "
                "Install the upstream package and its quant CUDA extension before "
                "using real KIVI inference."
            ) from exc
    for name, previous in previous_modules.items():
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    return getattr(module, class_name)


def load_upstream_quant_module(source_path: str | Path | None = None) -> Any:
    """Load upstream KIVI's quant.new_pack module directly.

    The returned module contains the upstream KV-cache functions such as
    quant_and_pack_kcache, quant_and_pack_vcache, unpack_and_dequant_kcache,
    and unpack_and_dequant_vcache.
    """

    status = inspect_kivi_source(source_path)
    if not status.usable_source_tree:
        raise KiviIntegrationError(
            "Upstream KIVI source tree is not ready. "
            f"path={status.path}; missing={list(status.missing_files)}. "
            "Run scripts/prepare_kivi_upstream.ps1 first."
        )
    previous_modules = {
        name: sys.modules.get(name)
        for name in ("quant", "quant.new_pack")
    }
    for name in previous_modules:
        sys.modules.pop(name, None)
    with _temporary_sys_path(status.path):
        try:
            module = importlib.import_module("quant.new_pack")
        except Exception as exc:  # pragma: no cover - depends on optional Triton deps
            raise KiviIntegrationError(
                "Found upstream KIVI source, but importing quant.new_pack failed. "
                "Install upstream KIVI dependencies such as triton before using "
                "real KV-cache pack/unpack functions."
            ) from exc
    for name, previous in previous_modules.items():
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    return module


def _config_value(config: Any, key: str) -> Any:
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


def _detect_license(path: Path) -> str | None:
    if not path.exists():
        return None
    first_chunk = path.read_text(encoding="utf-8", errors="ignore")[:256].lower()
    if "mit license" in first_chunk:
        return "MIT"
    if "apache license" in first_chunk:
        return "Apache-2.0"
    return "unknown"


def _git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


@contextmanager
def _temporary_sys_path(path: Path) -> Iterator[None]:
    as_text = str(path)
    inserted = as_text not in sys.path
    if inserted:
        sys.path.insert(0, as_text)
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(as_text)
            except ValueError:
                pass
