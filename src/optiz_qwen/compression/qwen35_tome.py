"""Runtime ToMe adapter for the Qwen3.5 visual encoder."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb_vision

from optiz_qwen.compression.tome import merge_single_visual_sample, merge_visual_units


@dataclass(frozen=True)
class Qwen35TomeConfig:
    layer: int
    r: int
    unit_size: int = 4
    proportional_attention: bool = False

    def validate(self, depth: int) -> None:
        if not 0 <= self.layer < depth:
            raise ValueError(f"ToMe layer must be in [0, {depth - 1}].")
        if self.r <= 0:
            raise ValueError("ToMe r must be positive when the adapter is enabled.")
        if self.unit_size != 4:
            raise ValueError("Qwen3.5 ToMe requires unit_size=4 for its 2x2 PatchMerger.")


@dataclass(frozen=True)
class Qwen35TomeRuntime:
    enabled: bool
    layer: int
    r: int
    input_tokens: int
    compact_tokens: int
    restored_tokens: int
    merged_units: int
    merge_host_ms: float
    proportional_attention: bool


class _TomeContext:
    def __init__(self, config: Qwen35TomeConfig) -> None:
        self.config = config
        self.enabled = True
        self.token_sizes: torch.Tensor | None = None
        self.position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None
        self.cu_seqlens: torch.Tensor | None = None
        self.runtime: Qwen35TomeRuntime | None = None
        self.restore_indices: torch.Tensor | None = None

    def reset(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
    ) -> None:
        self.token_sizes = torch.ones(
            hidden_states.shape[0],
            1,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self.position_embeddings = position_embeddings
        self.cu_seqlens = cu_seqlens
        self.runtime = None
        self.restore_indices = None


class Qwen35TomeBlock(nn.Module):
    def __init__(self, block: nn.Module, layer_index: int, depth: int, context: _TomeContext) -> None:
        super().__init__()
        self.block = block
        self.layer_index = layer_index
        self.is_last_layer = layer_index == depth - 1
        self.context = context

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        if not self.context.enabled:
            return self.block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        if self.layer_index == 0:
            self.context.reset(hidden_states, position_embeddings, cu_seqlens)

        current_positions = self.context.position_embeddings
        current_cu_seqlens = self.context.cu_seqlens
        if current_positions is None or current_cu_seqlens is None:
            raise RuntimeError("ToMe visual context was not initialized by layer 0.")
        if hidden_states.shape[0] != current_positions[0].shape[0]:
            raise RuntimeError("ToMe hidden-state and position lengths diverged.")

        if self.layer_index != self.context.config.layer:
            if self.context.config.proportional_attention and self.layer_index > self.context.config.layer:
                token_sizes = self.context.token_sizes
                if token_sizes is None:
                    raise RuntimeError("ToMe token sizes were not initialized.")
                hidden_states = hidden_states + _proportional_attention_forward(
                    self.block.attn,
                    self.block.norm1(hidden_states),
                    token_sizes,
                    current_cu_seqlens,
                    current_positions,
                )
                hidden_states = hidden_states + self.block.mlp(self.block.norm2(hidden_states))
            else:
                hidden_states = self.block(
                    hidden_states,
                    cu_seqlens=current_cu_seqlens,
                    position_embeddings=current_positions,
                    **kwargs,
                )
            return self._restore_output_length(hidden_states)

        captured_qkv: list[torch.Tensor] = []
        handle = self.block.attn.qkv.register_forward_hook(
            lambda _module, _inputs, output: captured_qkv.append(output)
        )
        try:
            hidden_states = hidden_states + self.block.attn(
                self.block.norm1(hidden_states),
                cu_seqlens=current_cu_seqlens,
                position_embeddings=current_positions,
                **kwargs,
            )
        finally:
            handle.remove()
        if len(captured_qkv) != 1:
            raise RuntimeError("Qwen3.5 visual Attention did not expose exactly one QKV projection.")

        qkv = captured_qkv[0]
        heads = self.block.attn.num_heads
        metric = qkv.reshape(qkv.shape[0], 3, heads, -1)[:, 1].reshape(qkv.shape[0], -1)
        token_sizes = self.context.token_sizes
        if token_sizes is None:
            raise RuntimeError("ToMe token sizes were not initialized.")

        merge_started = time.perf_counter()
        if current_cu_seqlens.numel() == 2:
            result = merge_single_visual_sample(
                hidden_states,
                metric,
                token_sizes,
                r=self.context.config.r,
                unit_size=self.context.config.unit_size,
            )
        else:
            result = merge_visual_units(
                hidden_states,
                metric,
                token_sizes,
                current_cu_seqlens,
                r=self.context.config.r,
                unit_size=self.context.config.unit_size,
            )
        merge_host_ms = (time.perf_counter() - merge_started) * 1000.0

        self.context.token_sizes = result.token_sizes
        self.context.position_embeddings = tuple(
            position[result.retained_token_indices] for position in current_positions
        )
        self.context.cu_seqlens = result.cu_seqlens
        restore_indices = torch.empty(
            hidden_states.shape[0],
            dtype=torch.long,
            device=hidden_states.device,
        )
        restore_indices[result.retained_token_indices] = torch.arange(
            result.hidden_states.shape[0],
            device=hidden_states.device,
        )
        offsets = torch.arange(self.context.config.unit_size, device=hidden_states.device)
        source_tokens = (
            result.source_unit_indices[:, None] * self.context.config.unit_size + offsets[None, :]
        ).reshape(-1)
        destination_tokens = (
            result.destination_unit_indices[:, None] * self.context.config.unit_size + offsets[None, :]
        ).reshape(-1)
        restore_indices[source_tokens] = restore_indices[destination_tokens]
        self.context.restore_indices = restore_indices
        self.context.runtime = Qwen35TomeRuntime(
            enabled=True,
            layer=self.context.config.layer,
            r=self.context.config.r,
            input_tokens=hidden_states.shape[0],
            compact_tokens=result.hidden_states.shape[0],
            restored_tokens=hidden_states.shape[0],
            merged_units=result.source_unit_indices.numel(),
            merge_host_ms=merge_host_ms,
            proportional_attention=self.context.config.proportional_attention,
        )
        hidden_states = result.hidden_states
        hidden_states = hidden_states + self.block.mlp(self.block.norm2(hidden_states))
        return self._restore_output_length(hidden_states)

    def _restore_output_length(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.is_last_layer:
            return hidden_states
        restore_indices = self.context.restore_indices
        if restore_indices is None:
            raise RuntimeError("ToMe reached the final visual layer without a restore map.")
        return hidden_states[restore_indices]


def _proportional_attention_forward(
    attention: nn.Module,
    hidden_states: torch.Tensor,
    token_sizes: torch.Tensor,
    cu_seqlens: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    if not hasattr(attention, "qkv") or not hasattr(attention, "proj"):
        raise TypeError("Proportional attention requires Qwen3.5 vision qkv and proj modules.")
    if token_sizes.shape != (hidden_states.shape[0], 1):
        raise ValueError("Proportional attention token_sizes must have shape [tokens, 1].")

    token_count = hidden_states.shape[0]
    qkv = attention.qkv(hidden_states).reshape(token_count, 3, attention.num_heads, -1)
    query, key, value = qkv.permute(1, 0, 2, 3).unbind(0)
    query, key = apply_rotary_pos_emb_vision(query, key, *position_embeddings)
    query = query.transpose(0, 1).unsqueeze(0)
    key = key.transpose(0, 1).unsqueeze(0)
    value = value.transpose(0, 1).unsqueeze(0)

    if cu_seqlens.numel() == 2:
        attention_output = _scaled_dot_product_with_size(query, key, value, token_sizes, attention.scaling)
    else:
        boundaries = cu_seqlens.detach().cpu().tolist()
        outputs = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            outputs.append(
                _scaled_dot_product_with_size(
                    query[:, :, start:end],
                    key[:, :, start:end],
                    value[:, :, start:end],
                    token_sizes[start:end],
                    attention.scaling,
                )
            )
        attention_output = torch.cat(outputs, dim=2)

    attention_output = attention_output.transpose(1, 2).reshape(token_count, -1).contiguous()
    return attention.proj(attention_output)


def _scaled_dot_product_with_size(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    token_sizes: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    size_bias = token_sizes.log().transpose(0, 1).unsqueeze(0).unsqueeze(0)
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=size_bias,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )


def install_qwen35_tome(model: Any, config: Qwen35TomeConfig) -> None:
    visual = _find_visual_encoder(model)
    if getattr(visual, "_optiz_tome_context", None) is not None:
        raise RuntimeError("Qwen3.5 ToMe is already installed on this model.")
    config.validate(len(visual.blocks))

    context = _TomeContext(config)
    depth = len(visual.blocks)
    visual.blocks = nn.ModuleList(
        Qwen35TomeBlock(block, layer_index, depth, context)
        for layer_index, block in enumerate(visual.blocks)
    )
    visual._optiz_tome_context = context


def get_qwen35_tome_runtime(model: Any) -> dict[str, Any] | None:
    visual = _find_visual_encoder(model)
    context = getattr(visual, "_optiz_tome_context", None)
    if context is None or context.runtime is None:
        return None
    return asdict(context.runtime)


def set_qwen35_tome_enabled(model: Any, enabled: bool) -> None:
    visual = _find_visual_encoder(model)
    context = getattr(visual, "_optiz_tome_context", None)
    if context is None:
        raise RuntimeError("Qwen3.5 ToMe is not installed on this model.")
    context.enabled = enabled
    context.runtime = None


def _find_visual_encoder(model: Any) -> nn.Module:
    base_model = getattr(model, "model", None)
    visual = getattr(base_model, "visual", None)
    if visual is None or not hasattr(visual, "blocks"):
        raise TypeError("Expected a Qwen3.5 model exposing model.visual.blocks.")
    return visual
