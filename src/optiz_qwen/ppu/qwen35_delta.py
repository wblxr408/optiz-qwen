"""PPU prefill kernel for the Qwen3.5 Gated Delta Rule."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.cpp_extension import CUDA_HOME, load


INSTALLATION_ATTRIBUTE = "_optiz_ppu_delta_kernel"


@dataclass(frozen=True)
class Qwen35PpuDeltaConfig:
    kernel_layers: int = 9
    position: str = "last"

    def __post_init__(self) -> None:
        if self.kernel_layers <= 0:
            raise ValueError("kernel_layers must be positive.")
        if self.position not in {"first", "last"}:
            raise ValueError("position must be 'first' or 'last'.")


@dataclass(frozen=True)
class Qwen35PpuDeltaReport:
    backend: str
    layer_indices: tuple[int, ...]
    kernel_layers: int
    position: str


@lru_cache(maxsize=1)
def load_qwen35_ppu_delta_extension() -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError("The Qwen3.5 PPU Delta kernel requires a CUDA-compatible PPU runtime.")
    if CUDA_HOME is None:
        raise RuntimeError("The PPU CUDA-compatible SDK was not found.")
    source_root = Path(__file__).with_name("csrc")
    sources = [source_root / "qwen35_delta.cpp", source_root / "qwen35_delta.cu"]
    missing = [str(source) for source in sources if not source.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing PPU Delta kernel sources: {missing}")
    return load(
        name="optiz_qwen_ppu_delta_v1",
        sources=[str(source) for source in sources],
        verbose=False,
    )


class _PpuDeltaKernel:
    def __init__(self, extension: Any, config: Qwen35PpuDeltaConfig) -> None:
        forward = getattr(extension, "forward", None)
        if not callable(forward):
            raise TypeError("The PPU Delta extension must expose a callable forward function.")
        self._forward = forward
        self.config = config
        self.calls = 0

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        chunk_size: int = 64,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if kwargs:
            raise TypeError(f"Unsupported PPU Delta kernel arguments: {sorted(kwargs)}")
        if initial_state is not None:
            raise NotImplementedError("The PPU Delta kernel does not support an initial state.")
        if cu_seqlens is not None:
            raise NotImplementedError("The PPU Delta kernel does not support packed sequences.")
        if not use_qk_l2norm_in_kernel:
            raise ValueError("The PPU Delta kernel requires Q/K L2 normalization.")
        query = query * torch.rsqrt(query.square().sum(dim=-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt(key.square().sum(dim=-1, keepdim=True) + 1e-6)
        output, final_state = self._forward(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            g.contiguous(),
            beta.contiguous(),
        )
        self.calls += 1
        return output, final_state if output_final_state else None


def install_qwen35_ppu_delta_kernel(
    model: nn.Module,
    config: Qwen35PpuDeltaConfig = Qwen35PpuDeltaConfig(),
    *,
    extension: Any | None = None,
) -> Qwen35PpuDeltaReport:
    """Install the PPU prefill kernel on a fixed leading or trailing layer subset."""

    if model.training:
        raise ValueError("The PPU Delta kernel is inference-only; call eval() first.")
    layers = [
        module
        for module in model.modules()
        if type(module).__name__ == "Qwen3_5GatedDeltaNet"
    ]
    if not layers:
        raise ValueError("Model does not contain Qwen3_5GatedDeltaNet layers.")
    if config.kernel_layers > len(layers):
        raise ValueError(
            f"kernel_layers={config.kernel_layers} exceeds the {len(layers)} GDN layers."
        )
    if any(hasattr(layer, INSTALLATION_ATTRIBUTE) for layer in layers):
        raise RuntimeError("The PPU Delta kernel is already installed.")
    selected = (
        layers[: config.kernel_layers]
        if config.position == "first"
        else layers[-config.kernel_layers :]
    )
    kernel = _PpuDeltaKernel(
        load_qwen35_ppu_delta_extension() if extension is None else extension,
        config,
    )
    for layer in selected:
        layer.chunk_gated_delta_rule = kernel
        setattr(layer, INSTALLATION_ATTRIBUTE, kernel)
    return Qwen35PpuDeltaReport(
        backend="ppu_hggc_recurrent_fp64_reduce_v1",
        layer_indices=tuple(int(getattr(layer, "layer_idx", -1)) for layer in selected),
        kernel_layers=len(selected),
        position=config.position,
    )


def get_qwen35_ppu_delta_runtime(model: nn.Module) -> dict[str, Any] | None:
    kernels = {
        id(getattr(module, INSTALLATION_ATTRIBUTE)): getattr(
            module,
            INSTALLATION_ATTRIBUTE,
        )
        for module in model.modules()
        if hasattr(module, INSTALLATION_ATTRIBUTE)
    }
    if not kernels:
        return None
    layer_indices = tuple(
        int(getattr(module, "layer_idx", -1))
        for module in model.modules()
        if hasattr(module, INSTALLATION_ATTRIBUTE)
    )
    kernel = next(iter(kernels.values()))
    return {
        "enabled": True,
        "layer_indices": layer_indices,
        "calls": sum(kernel.calls for kernel in kernels.values()),
        "config": asdict(kernel.config),
    }
