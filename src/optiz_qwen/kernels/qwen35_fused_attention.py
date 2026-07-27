"""Qwen3.5 full-attention adapter for direct packed-KV decode."""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch

from optiz_qwen.kernels.qserve_int4_attention import (
    qserve_int4_decode_attention,
    qserve_int4_split_decode_attention,
    triton_int4_decode_available,
)


def _qserve_fused_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Any | None = None,
    **kwargs,
):
    # Packed-KV acceleration only applies to one-token decode.  Delegating
    # prefill to the original module avoids a second Python-level attention
    # implementation on the TTFT-critical path.
    if hidden_states.shape[1] != 1:
        original_forward = getattr(self, "_optiz_original_forward", None)
        if original_forward is None:
            raise RuntimeError("qserve fused attention lost the native Qwen3.5 forward method.")
        return original_forward(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, eager_attention_forward

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states, gate = torch.chunk(
        self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
    )
    gate = gate.reshape(*input_shape, -1)
    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    fused_decode = False
    cache_layer = None
    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        cache_layer = past_key_values.layers[self.layer_idx]
        fused_decode = (
            hidden_states.shape[1] == 1
            and triton_int4_decode_available(query_states)
            and getattr(cache_layer, "_key_code", None) is not None
            and bool(getattr(past_key_values, "_optiz_dense_decode_mask", False))
        )

    if fused_decode:
        if getattr(past_key_values, "attention_backend", "triton_int4_decode") == "triton_int4_split_decode":
            attn_output = qserve_int4_split_decode_attention(query_states, cache_layer, scaling=self.scaling)
        else:
            attn_output = qserve_int4_decode_attention(query_states, cache_layer, scaling=self.scaling)
        attn_weights = None
        past_key_values.kernel_calls += 1
    else:
        if past_key_values is not None and hidden_states.shape[1] == 1:
            key_states, value_states = cache_layer.materialize()
            past_key_values.fallback_calls += 1
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)
    return self.o_proj(attn_output), attn_weights


def install_qwen35_fused_attention(model: Any) -> tuple[int, ...]:
    """Patch only Qwen3.5 full-attention modules on one model instance."""

    language_model = model.model.language_model
    installed = []
    for layer_idx, decoder_layer in enumerate(language_model.layers):
        if language_model.config.layer_types[layer_idx] != "full_attention":
            continue
        attention = decoder_layer.self_attn
        if not hasattr(attention, "_optiz_original_forward"):
            attention._optiz_original_forward = attention.forward
            attention.forward = MethodType(_qserve_fused_forward, attention)
        installed.append(layer_idx)
    return tuple(installed)
