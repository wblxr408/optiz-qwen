"""Qwen3.5 VLM cache adapter backed by upstream KIVI pack/unpack code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers.cache_utils import CacheLayerMixin, DynamicCache

from optiz_qwen.compression.kivi_external import KiviConfig, load_upstream_quant_module


@dataclass(frozen=True)
class Qwen35KiviCacheReport:
    enabled: bool
    full_attention_layers: tuple[int, ...]
    k_bits: int
    v_bits: int
    group_size: int
    residual_length: int


class KiviAttentionLayer(CacheLayerMixin):
    """Single full-attention layer cache using upstream KIVI tensor packing."""

    is_compileable = False
    supports_early_init = True
    is_sliding = False

    def __init__(self, config: KiviConfig, quant_module: Any | None = None) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.quant_module = quant_module if quant_module is not None else load_upstream_quant_module(config.source_path)
        self.keys = None
        self.values = None
        self.device = None
        self.dtype = None
        self.is_initialized = False
        self._seq_length = 0
        self._key_code = None
        self._key_scale = None
        self._key_mn = None
        self._key_residual = None
        self._value_code = None
        self._value_scale = None
        self._value_mn = None
        self._value_residual = None

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.is_initialized = True

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        if self._seq_length:
            previous_keys, previous_values = self.materialize()
            key_states = torch.cat([previous_keys, key_states], dim=-2)
            value_states = torch.cat([previous_values, value_states], dim=-2)

        self._rebuild_from_full(key_states.contiguous(), value_states.contiguous())
        return self.materialize()

    def materialize(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized or self._seq_length == 0:
            raise ValueError("KIVI cache layer is empty and cannot be materialized.")

        keys = self._materialize_keys()
        values = self._materialize_values()
        self.keys = keys
        self.values = values
        return keys, values

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.get_seq_length() + query_length, 0

    def get_seq_length(self) -> int:
        return self._seq_length

    def get_max_length(self) -> int:
        return -1

    def get_max_cache_shape(self) -> int:
        return self.get_max_length()

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)
        if self.get_seq_length() <= max_length:
            return
        keys, values = self.materialize()
        self._rebuild_from_full(keys[..., :max_length, :].contiguous(), values[..., :max_length, :].contiguous())

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.get_seq_length() == 0:
            return
        keys, values = self.materialize()
        self._rebuild_from_full(
            keys.repeat_interleave(repeats, dim=0).contiguous(),
            values.repeat_interleave(repeats, dim=0).contiguous(),
        )

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.get_seq_length() == 0:
            return
        keys, values = self.materialize()
        self._rebuild_from_full(keys[indices, ...].contiguous(), values[indices, ...].contiguous())

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self.batch_select_indices(beam_idx.to(self.device))

    def reset(self) -> None:
        self._seq_length = 0
        self._key_code = self._key_scale = self._key_mn = self._key_residual = None
        self._value_code = self._value_scale = self._value_mn = self._value_residual = None
        self.keys = None
        self.values = None
        self.is_initialized = False

    def _rebuild_from_full(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        self.dtype = keys.dtype
        self.device = keys.device
        self.is_initialized = True
        self._seq_length = int(keys.shape[-2])
        self._pack_keys(keys)
        self._pack_values(values)
        self.keys = None
        self.values = None

    def _pack_keys(self, keys: torch.Tensor) -> None:
        seq_len = int(keys.shape[-2])
        quant_len = seq_len - (seq_len % self.config.residual_length)
        if quant_len == 0:
            self._key_code = self._key_scale = self._key_mn = None
            self._key_residual = keys
            return
        self._validate_pack_dim(quant_len, self.config.k_bits, "key token dimension")

        key_quant = keys[..., :quant_len, :].contiguous()
        self._key_residual = keys[..., quant_len:, :].contiguous()
        self._key_code, self._key_scale, self._key_mn = self.quant_module.triton_quantize_and_pack_along_last_dim(
            key_quant.transpose(2, 3).contiguous(),
            self.config.group_size,
            self.config.k_bits,
        )

    def _pack_values(self, values: torch.Tensor) -> None:
        seq_len = int(values.shape[-2])
        if seq_len <= self.config.residual_length:
            self._value_code = self._value_scale = self._value_mn = None
            self._value_residual = values
            return

        quant_len = seq_len - self.config.residual_length
        self._validate_pack_dim(int(values.shape[-1]), self.config.v_bits, "value head dimension")
        value_quant = values[..., :quant_len, :].contiguous()
        self._value_residual = values[..., quant_len:, :].contiguous()
        self._value_code, self._value_scale, self._value_mn = self.quant_module.triton_quantize_and_pack_along_last_dim(
            value_quant,
            self.config.group_size,
            self.config.v_bits,
        )

    def _materialize_keys(self) -> torch.Tensor:
        parts = []
        if self._key_code is not None:
            dequantized = self.quant_module.unpack_and_dequant_vcache(
                self._key_code,
                self._key_scale.unsqueeze(-1),
                self._key_mn.unsqueeze(-1),
                self.config.group_size,
                self.config.k_bits,
            ).transpose(2, 3)
            parts.append(dequantized)
        if self._key_residual is not None and self._key_residual.shape[-2] > 0:
            parts.append(self._key_residual)
        return torch.cat(parts, dim=-2) if len(parts) > 1 else parts[0]

    def _materialize_values(self) -> torch.Tensor:
        parts = []
        if self._value_code is not None:
            parts.append(
                self.quant_module.unpack_and_dequant_vcache(
                    self._value_code,
                    self._value_scale.unsqueeze(-1),
                    self._value_mn.unsqueeze(-1),
                    self.config.group_size,
                    self.config.v_bits,
                )
            )
        if self._value_residual is not None and self._value_residual.shape[-2] > 0:
            parts.append(self._value_residual)
        return torch.cat(parts, dim=-2) if len(parts) > 1 else parts[0]

    @staticmethod
    def _validate_pack_dim(size: int, bits: int, name: str) -> None:
        features_per_int = 32 // bits
        if size < features_per_int or size % features_per_int != 0:
            raise ValueError(
                f"KIVI {name}={size} must be divisible by {features_per_int} for {bits}-bit packing."
            )


class Qwen35KiviCache(DynamicCache):
    """DynamicCache-compatible container for Qwen3.5 mixed attention stacks."""

    def __init__(self, model_config: Any, kivi_config: KiviConfig | None = None, quant_module: Any | None = None) -> None:
        self.kivi_config = kivi_config or KiviConfig()
        self.kivi_config.validate()
        super().__init__(config=model_config)
        decoder_config = model_config.get_text_config(decoder=True) if hasattr(model_config, "get_text_config") else model_config
        self.layer_types = tuple(getattr(decoder_config, "layer_types", ()))
        self.full_attention_layers = tuple(
            idx for idx, layer_type in enumerate(self.layer_types) if layer_type == "full_attention"
        )
        self.quant_module = quant_module if quant_module is not None else load_upstream_quant_module(self.kivi_config.source_path)
        for layer_idx in self.full_attention_layers:
            self.layers[layer_idx] = KiviAttentionLayer(self.kivi_config, self.quant_module)

    def report(self) -> Qwen35KiviCacheReport:
        return Qwen35KiviCacheReport(
            enabled=True,
            full_attention_layers=self.full_attention_layers,
            k_bits=self.kivi_config.k_bits,
            v_bits=self.kivi_config.v_bits,
            group_size=self.kivi_config.group_size,
            residual_length=self.kivi_config.residual_length,
        )

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if (
            layer_idx == 0
            and self.layer_types
            and self.layer_types[0] == "linear_attention"
            and self.full_attention_layers
        ):
            layer_idx = self.full_attention_layers[0]
        return super().get_seq_length(layer_idx)

    def __repr__(self) -> str:
        return f"Qwen35KiviCache({self.report()})"


def build_qwen35_kivi_cache(model_config: Any, kivi_config: KiviConfig | None = None) -> Qwen35KiviCache:
    return Qwen35KiviCache(model_config=model_config, kivi_config=kivi_config)
