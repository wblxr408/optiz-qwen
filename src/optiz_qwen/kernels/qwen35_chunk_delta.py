"""Numerical contract for Qwen3.5 chunk Gated Delta Rule kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


ChunkDeltaKernel = Callable[..., tuple[torch.Tensor, torch.Tensor | None]]
INSTALLATION_ATTRIBUTE = "_optiz_chunk_delta_kernel"


@dataclass(frozen=True)
class ChunkDeltaKernelReport:
    layer_indices: tuple[int, ...]
    backend: str


@dataclass(frozen=True)
class ChunkDeltaComparison:
    output_max_abs_error: float
    state_max_abs_error: float


def _l2_normalize(value: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    return value * torch.rsqrt(value.square().sum(dim=-1, keepdim=True) + epsilon)


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> tuple[int, int, int, int, int]:
    if cu_seqlens is not None:
        raise NotImplementedError("The initial PPU chunk Delta contract does not support packed sequences.")
    if query.ndim != 4 or key.shape != query.shape:
        raise ValueError("query and key must have shape [batch, sequence, heads, key_dim].")
    if value.ndim != 4 or value.shape[:3] != query.shape[:3]:
        raise ValueError("value must match query batch, sequence, and head dimensions.")
    batch, sequence, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    if g.shape != (batch, sequence, heads) or beta.shape != g.shape:
        raise ValueError("g and beta must have shape [batch, sequence, heads].")
    if any(tensor.device != query.device for tensor in (key, value, g, beta)):
        raise ValueError("all chunk Delta inputs must be on the same device.")
    if any(not tensor.is_floating_point() for tensor in (query, key, value, g, beta)):
        raise TypeError("all chunk Delta inputs must use floating-point dtypes.")
    if initial_state is not None and initial_state.shape != (batch, heads, key_dim, value_dim):
        raise ValueError("initial_state must have shape [batch, heads, key_dim, value_dim].")
    return batch, sequence, heads, key_dim, value_dim


def qwen35_chunk_delta_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """FP32-state recurrence matching the Qwen3.5 Transformers fallback contract."""

    batch, sequence, heads, key_dim, value_dim = _validate_inputs(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        cu_seqlens,
    )
    output_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2_normalize(query)
        key = _l2_normalize(key)
    query, key, value, beta, g = [
        tensor.transpose(1, 2).contiguous().to(torch.float32)
        for tensor in (query, key, value, beta, g)
    ]
    query = query * (key_dim**-0.5)
    state = (
        torch.zeros(
            batch,
            heads,
            key_dim,
            value_dim,
            dtype=torch.float32,
            device=query.device,
        )
        if initial_state is None
        else initial_state.to(device=query.device, dtype=torch.float32)
    )
    output = torch.empty(
        batch,
        heads,
        sequence,
        value_dim,
        dtype=torch.float32,
        device=query.device,
    )
    for index in range(sequence):
        query_step = query[:, :, index]
        key_step = key[:, :, index]
        value_step = value[:, :, index]
        decay_step = g[:, :, index].exp().unsqueeze(-1).unsqueeze(-1)
        beta_step = beta[:, :, index].unsqueeze(-1)
        state = state * decay_step
        memory = (state * key_step.unsqueeze(-1)).sum(dim=-2)
        delta = (value_step - memory) * beta_step
        state = state + key_step.unsqueeze(-1) * delta.unsqueeze(-2)
        output[:, :, index] = (state * query_step.unsqueeze(-1)).sum(dim=-2)
    final_state = state if output_final_state else None
    return output.transpose(1, 2).contiguous().to(output_dtype), final_state


def compare_chunk_delta_kernel(
    kernel: ChunkDeltaKernel,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
) -> ChunkDeltaComparison:
    kwargs = {
        "g": g,
        "beta": beta,
        "initial_state": initial_state,
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": use_qk_l2norm_in_kernel,
        "cu_seqlens": None,
    }
    expected_output, expected_state = qwen35_chunk_delta_reference(
        query,
        key,
        value,
        **kwargs,
    )
    actual_output, actual_state = kernel(query, key, value, **kwargs)
    if actual_output.shape != expected_output.shape or actual_state is None or expected_state is None:
        raise ValueError("Chunk Delta kernel returned an invalid output or state shape.")
    if actual_state.shape != expected_state.shape:
        raise ValueError("Chunk Delta kernel returned an invalid final state shape.")
    return ChunkDeltaComparison(
        output_max_abs_error=float(
            (actual_output.float() - expected_output.float()).abs().max().item()
        ),
        state_max_abs_error=float(
            (actual_state.float() - expected_state.float()).abs().max().item()
        ),
    )


def install_qwen35_chunk_delta_kernel(
    model: nn.Module,
    kernel: ChunkDeltaKernel,
    *,
    backend: str,
) -> ChunkDeltaKernelReport:
    """Install one explicitly selected chunk-prefill kernel on every Qwen3.5 GDN layer."""

    if model.training:
        raise ValueError("Chunk Delta kernels are inference-only; call eval() first.")
    if not callable(kernel):
        raise TypeError("chunk Delta kernel must be callable.")
    if not backend.strip():
        raise ValueError("chunk Delta backend name must not be empty.")
    layers = [
        module
        for module in model.modules()
        if type(module).__name__ == "Qwen3_5GatedDeltaNet"
    ]
    if not layers:
        raise ValueError("Model does not contain Qwen3_5GatedDeltaNet layers.")
    if any(hasattr(layer, INSTALLATION_ATTRIBUTE) for layer in layers):
        raise RuntimeError("A chunk Delta kernel is already installed.")
    for layer in layers:
        layer.chunk_gated_delta_rule = kernel
        setattr(layer, INSTALLATION_ATTRIBUTE, backend)
    return ChunkDeltaKernelReport(
        layer_indices=tuple(int(getattr(layer, "layer_idx", -1)) for layer in layers),
        backend=backend,
    )
