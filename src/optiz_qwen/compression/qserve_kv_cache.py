"""QServe-inspired Qwen3.5 KV-cache adapter.

This module keeps the competition scope constrained to KV-only changes, but
implements the retained deferred split packed-KV experiment:

- 4-bit groupwise KV quantization
- progressive quantization with a dense residual window
- no external Triton dependency

The goal is to provide a real second chain for experiments and benchmarking
while preserving the baseline path unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers.cache_utils import CacheLayerMixin, DynamicCache


@dataclass(frozen=True)
class QServeKvConfig:
    """Runtime preset for the QServe-style KV cache."""

    k_bits: int = 4
    v_bits: int = 4
    group_size: int = 32
    residual_length: int = 32
    activation_threshold: int = 1024
    decode_warmup_tokens: int = 4
    attention_backend: str = "triton_int4_split_decode"

    def validate(self) -> None:
        if self.k_bits not in {4, 8} or self.v_bits != self.k_bits:
            raise ValueError("qserve_kv supports matched 4-bit or 8-bit K/V only.")
        if self.group_size <= 0:
            raise ValueError("qserve_kv group_size must be positive.")
        if self.residual_length <= 0:
            raise ValueError("qserve_kv residual_length must be positive.")
        if self.residual_length % self.group_size != 0:
            raise ValueError("qserve_kv residual_length must be divisible by group_size.")
        if self.activation_threshold < self.residual_length:
            raise ValueError("qserve_kv activation_threshold must be at least residual_length.")
        if self.decode_warmup_tokens < 0:
            raise ValueError("qserve_kv decode_warmup_tokens must be non-negative.")
        valid_backends = {"triton_int4_decode", "triton_int4_split_decode", "triton_int8_decode"}
        if self.attention_backend not in valid_backends:
            raise ValueError("qserve_kv attention_backend must be a retained Triton decode backend.")
        if self.k_bits == 8 and self.attention_backend != "triton_int8_decode":
            raise ValueError("8-bit QServe KV requires triton_int8_decode.")
        if self.k_bits == 4 and self.attention_backend == "triton_int8_decode":
            raise ValueError("triton_int8_decode requires 8-bit QServe KV.")


@dataclass(frozen=True)
class QServeKvCacheReport:
    enabled: bool
    full_attention_layers: tuple[int, ...]
    k_bits: int
    v_bits: int
    group_size: int
    residual_length: int
    activation_threshold: int
    decode_warmup_tokens: int
    attention_backend: str
    implementation: str = "qserve_deferred_split_packed_kv"


class QServeAttentionLayer(CacheLayerMixin):
    """Full-attention layer cache with local 4-bit progressive KV quantization."""

    is_compileable = False
    supports_early_init = True
    is_sliding = False

    def __init__(self, config: QServeKvConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.keys = None
        self.values = None
        self.device = None
        self.dtype = None
        self.is_initialized = False
        self._seq_length = 0
        self._key_code = None
        self._key_scale = None
        self._key_min = None
        self._key_residual = None
        self._key_last_dim = None
        self._value_code = None
        self._value_scale = None
        self._value_min = None
        self._value_residual = None
        self._value_last_dim = None

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

        if self._seq_length == 0:
            self._rebuild_from_full(key_states, value_states)
            return key_states, value_states

        if key_states.shape[-2] == 1:
            self._append_decode_token(key_states, value_states)
            return self.materialize()

        if self._seq_length:
            previous_keys, previous_values = self.materialize()
            key_states = torch.cat([previous_keys, key_states], dim=-2)
            value_states = torch.cat([previous_values, value_states], dim=-2)

        self._rebuild_from_full(key_states, value_states)
        return key_states, value_states

    def materialize(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized or self._seq_length == 0:
            raise ValueError("qserve_kv cache layer is empty and cannot be materialized.")

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
        self._key_code = self._key_scale = self._key_min = self._key_residual = None
        self._value_code = self._value_scale = self._value_min = self._value_residual = None
        self._key_last_dim = self._value_last_dim = None
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

    def _append_decode_token(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.is_initialized = True
        self._seq_length += int(key_states.shape[-2])
        self._append_keys(key_states)
        self._append_values(value_states)
        self.keys = None
        self.values = None

    def _append_keys(self, key_states: torch.Tensor) -> None:
        if self._key_residual is None:
            self._key_residual = key_states
        else:
            self._key_residual = torch.cat([self._key_residual, key_states], dim=-2).contiguous()

        if self._key_residual.shape[-2] < self.config.residual_length:
            return

        self._quantize_and_append_keys(self._key_residual)
        self._key_residual = self._key_residual[..., 0:0, :].contiguous()

    def _append_values(self, value_states: torch.Tensor) -> None:
        if self._value_residual is None:
            self._value_residual = value_states
        else:
            self._value_residual = torch.cat([self._value_residual, value_states], dim=-2).contiguous()

        if self._value_residual.shape[-2] < self.config.residual_length:
            return

        self._quantize_and_append_values(self._value_residual)
        self._value_residual = self._value_residual[..., 0:0, :].contiguous()

    def _pack_keys(self, keys: torch.Tensor) -> None:
        seq_len = int(keys.shape[-2])
        quant_len = seq_len - (seq_len % self.config.residual_length)
        self._key_last_dim = int(keys.shape[-1])
        if quant_len == 0:
            self._key_code = self._key_scale = self._key_min = None
            self._key_residual = keys
            return
        self._quantize_and_append_keys(keys[..., :quant_len, :].contiguous())
        self._key_residual = keys[..., quant_len:, :].contiguous()

    def _pack_values(self, values: torch.Tensor) -> None:
        seq_len = int(values.shape[-2])
        quant_len = seq_len - (seq_len % self.config.residual_length)
        self._value_last_dim = int(values.shape[-1])
        if quant_len == 0:
            self._value_code = self._value_scale = self._value_min = None
            self._value_residual = values
            return
        self._quantize_and_append_values(values[..., :quant_len, :].contiguous())
        self._value_residual = values[..., quant_len:, :].contiguous()

    def _quantize_and_append_keys(self, key_states: torch.Tensor) -> None:
        packed, scale, mn = self._quantize_groupwise(
            key_states, self.config.group_size, self.config.k_bits
        )
        self._key_code = self._concat_or_init(self._key_code, packed, dim=-2)
        self._key_scale = self._concat_or_init(self._key_scale, scale, dim=-2)
        self._key_min = self._concat_or_init(self._key_min, mn, dim=-2)

    def _quantize_and_append_values(self, value_states: torch.Tensor) -> None:
        packed, scale, mn = self._quantize_groupwise(
            value_states, self.config.group_size, self.config.v_bits
        )
        self._value_code = self._concat_or_init(self._value_code, packed, dim=-2)
        self._value_scale = self._concat_or_init(self._value_scale, scale, dim=-2)
        self._value_min = self._concat_or_init(self._value_min, mn, dim=-2)

    def _materialize_keys(self) -> torch.Tensor:
        parts = []
        if self._key_code is not None:
            parts.append(
                self._dequantize_groupwise(
                    self._key_code,
                    self._key_scale,
                    self._key_min,
                    self.config.group_size,
                    self._key_last_dim,
                    self.dtype,
                    self.config.k_bits,
                )
            )
        if self._key_residual is not None and self._key_residual.shape[-2] > 0:
            parts.append(self._key_residual)
        if not parts:
            raise ValueError("qserve_kv key cache is empty.")
        return torch.cat(parts, dim=-2) if len(parts) > 1 else parts[0]

    def _materialize_values(self) -> torch.Tensor:
        parts = []
        if self._value_code is not None:
            parts.append(
                self._dequantize_groupwise(
                    self._value_code,
                    self._value_scale,
                    self._value_min,
                    self.config.group_size,
                    self._value_last_dim,
                    self.dtype,
                    self.config.v_bits,
                )
            )
        if self._value_residual is not None and self._value_residual.shape[-2] > 0:
            parts.append(self._value_residual)
        if not parts:
            raise ValueError("qserve_kv value cache is empty.")
        return torch.cat(parts, dim=-2) if len(parts) > 1 else parts[0]

    @staticmethod
    def _quantize_groupwise(
        tensor: torch.Tensor,
        group_size: int,
        bits: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tensor.shape[-1] % group_size != 0:
            raise ValueError(
                f"qserve_kv last dimension {tensor.shape[-1]} must be divisible by group_size={group_size}."
            )
        original_shape = tensor.shape
        last_dim = int(tensor.shape[-1])
        groups = last_dim // group_size
        tensor_f32 = tensor.to(dtype=torch.float32)
        grouped = tensor_f32.reshape(*original_shape[:-1], groups, group_size)
        group_min = grouped.amin(dim=-1, keepdim=True)
        group_max = grouped.amax(dim=-1, keepdim=True)
        levels = (1 << bits) - 1
        scale = (group_max - group_min) / float(levels)
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        codes = torch.round((grouped - group_min) / scale).clamp(0, levels).to(torch.uint8)
        flattened = codes.reshape(*original_shape[:-1], last_dim)
        packed = QServeAttentionLayer._pack_uint4(flattened) if bits == 4 else flattened.contiguous()
        return packed, scale.squeeze(-1), group_min.squeeze(-1)

    @staticmethod
    def _dequantize_groupwise(
        packed: torch.Tensor,
        scale: torch.Tensor,
        group_min: torch.Tensor,
        group_size: int,
        last_dim: int,
        dtype: torch.dtype,
        bits: int,
    ) -> torch.Tensor:
        codes = QServeAttentionLayer._unpack_uint4(packed, last_dim) if bits == 4 else packed
        groups = last_dim // group_size
        codes = codes.reshape(*packed.shape[:-1], groups, group_size).to(torch.float32)
        scale = scale.unsqueeze(-1)
        group_min = group_min.unsqueeze(-1)
        dequant = group_min + codes * scale
        return dequant.reshape(*packed.shape[:-1], last_dim).to(dtype=dtype)

    @staticmethod
    def _pack_uint4(values: torch.Tensor) -> torch.Tensor:
        pad = values.shape[-1] % 2
        if pad:
            values = torch.cat([values, values.new_zeros(values.shape[:-1] + (1,))], dim=-1)
        pairs = values.reshape(*values.shape[:-1], -1, 2)
        packed = (pairs[..., 0] & 0x0F) | ((pairs[..., 1] & 0x0F) << 4)
        return packed.contiguous().to(torch.uint8)

    @staticmethod
    def _unpack_uint4(packed: torch.Tensor, last_dim: int) -> torch.Tensor:
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        values = torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], -1)
        if values.shape[-1] > last_dim:
            values = values[..., :last_dim]
        return values.contiguous().to(torch.uint8)

    @staticmethod
    def _concat_or_init(existing: torch.Tensor | None, new: torch.Tensor, dim: int) -> torch.Tensor:
        if existing is None:
            return new
        return torch.cat([existing, new], dim=dim).contiguous()


class QServeKvCache(DynamicCache):
    """DynamicCache-compatible container for the local QServe-inspired chain."""

    def __init__(self, model_config: Any, qserve_config: QServeKvConfig | None = None) -> None:
        self.qserve_config = qserve_config or QServeKvConfig()
        self.qserve_config.validate()
        super().__init__(config=model_config)
        decoder_config = model_config.get_text_config(decoder=True) if hasattr(model_config, "get_text_config") else model_config
        self.layer_types = tuple(getattr(decoder_config, "layer_types", ()))
        self.full_attention_layers = tuple(
            idx for idx, layer_type in enumerate(self.layer_types) if layer_type == "full_attention"
        )
        for layer_idx in self.full_attention_layers:
            self.layers[layer_idx] = QServeAttentionLayer(self.qserve_config)

    def report(self) -> QServeKvCacheReport:
        return QServeKvCacheReport(
            enabled=True,
            full_attention_layers=self.full_attention_layers,
            k_bits=self.qserve_config.k_bits,
            v_bits=self.qserve_config.v_bits,
            group_size=self.qserve_config.group_size,
            residual_length=self.qserve_config.residual_length,
            activation_threshold=self.qserve_config.activation_threshold,
            decode_warmup_tokens=self.qserve_config.decode_warmup_tokens,
            attention_backend=self.qserve_config.attention_backend,
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
        return f"QServeKvCache({self.report()})"


class QServeFusedAttentionLayer(QServeAttentionLayer):
    """Cache layer that keeps decode KV packed for the fused attention path."""

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()
        if self._seq_length == 0:
            self._rebuild_from_full(key_states, value_states)
            return key_states, value_states
        if key_states.shape[-2] == 1:
            self._append_decode_token(key_states, value_states)
            return key_states, value_states
        previous_keys, previous_values = self.materialize()
        keys = torch.cat([previous_keys, key_states], dim=-2)
        values = torch.cat([previous_values, value_states], dim=-2)
        self._rebuild_from_full(keys, values)
        return keys, values


class QServeFusedKvCache(QServeKvCache):
    """QServe cache whose full-attention decode reads packed KV directly."""

    def __init__(self, model_config: Any, qserve_config: QServeKvConfig | None = None) -> None:
        super().__init__(model_config=model_config, qserve_config=qserve_config)
        self.attention_backend = self.qserve_config.attention_backend
        for layer_idx in self.full_attention_layers:
            self.layers[layer_idx] = QServeFusedAttentionLayer(self.qserve_config)
        self.kernel_calls = 0
        self.fallback_calls = 0

    def __repr__(self) -> str:
        return f"QServeFusedKvCache({self.report()})"


class QServeSplitFusedKvCache(QServeFusedKvCache):
    """Fused cache selecting the two-stage packed-KV decode backend."""

    attention_backend = "triton_int4_split_decode"

    def __repr__(self) -> str:
        return f"QServeSplitFusedKvCache({self.report()})"


class QServeDeferredSplitFusedAttentionLayer(QServeFusedAttentionLayer):
    """Keeps short-context KV dense and packs only after the measured threshold.

    The public VLM evaluation has short image-text prompts.  Packing them on
    the first decode both adds conversion cost and routes attention through a
    kernel whose launch overhead is not amortized.  This layer therefore uses
    the native dense cache until the real cache length reaches the configured
    crossover, then switches once to packed KV for following decode steps.
    """

    def __init__(self, config: QServeKvConfig) -> None:
        super().__init__(config)
        self._deferred_prefill = False

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

        if self._seq_length == 0:
            # The first output token is produced from prefill logits.  Keeping
            # this cache dense until the next model call removes quantization
            # from TTFT while preserving the packed decode representation.
            self.dtype = key_states.dtype
            self.device = key_states.device
            self._seq_length = int(key_states.shape[-2])
            self.keys = key_states
            self.values = value_states
            self._deferred_prefill = True
            return key_states, value_states

        if self._deferred_prefill:
            prefill_keys = self.keys
            prefill_values = self.values
            if prefill_keys is None or prefill_values is None:
                raise RuntimeError("deferred qserve prefill cache is missing dense KV tensors.")
            dense_keys = torch.cat([prefill_keys, key_states], dim=-2).contiguous()
            dense_values = torch.cat([prefill_values, value_states], dim=-2).contiguous()
            self._seq_length = int(dense_keys.shape[-2])
            if self._seq_length < self.config.activation_threshold:
                self.keys = dense_keys
                self.values = dense_values
                return dense_keys, dense_values
            self._deferred_prefill = False
            self._rebuild_from_full(dense_keys, dense_values)
            return key_states, value_states

        return super().update(key_states, value_states, *args, **kwargs)

    def adopt_prefill(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        """Adopt dense KV produced by the native prefill cache without repacking it."""

        self.lazy_initialization(key_states, value_states)
        self._seq_length = int(key_states.shape[-2])
        self.keys = key_states.contiguous()
        self.values = value_states.contiguous()
        self._deferred_prefill = True

    def materialize(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._deferred_prefill:
            if self.keys is None or self.values is None:
                raise RuntimeError("deferred qserve prefill cache is missing dense KV tensors.")
            return self.keys, self.values
        return super().materialize()

    def reset(self) -> None:
        super().reset()
        self._deferred_prefill = False


class QServeDeferredSplitFusedKvCache(QServeSplitFusedKvCache):
    """Split decode cache that shifts prefill quantization out of TTFT."""

    attention_backend = "triton_int4_split_decode"
    defer_prefill_cache_injection = True

    def __init__(self, model_config: Any, qserve_config: QServeKvConfig | None = None) -> None:
        super().__init__(model_config=model_config, qserve_config=qserve_config)
        for layer_idx in self.full_attention_layers:
            self.layers[layer_idx] = QServeDeferredSplitFusedAttentionLayer(self.qserve_config)

    def adopt_native_prefill_cache(self, native_cache: Any) -> None:
        """Transfer native prefill state, replacing only full-attention cache layers."""

        native_layers = getattr(native_cache, "layers", None)
        if native_layers is None:
            raise TypeError("native prefill cache does not expose cache layers.")
        if len(native_layers) != len(self.layers):
            raise ValueError("native prefill cache depth does not match the Qwen3.5 cache.")
        for layer_idx, native_layer in enumerate(native_layers):
            if layer_idx not in self.full_attention_layers:
                self.layers[layer_idx] = native_layer
                continue
            key_states = getattr(native_layer, "keys", None)
            value_states = getattr(native_layer, "values", None)
            if key_states is None or value_states is None:
                raise RuntimeError(f"native full-attention layer {layer_idx} has no dense KV to adopt.")
            qserve_layer = self.layers[layer_idx]
            if not isinstance(qserve_layer, QServeDeferredSplitFusedAttentionLayer):
                raise TypeError("deferred split cache lost its full-attention layer type.")
            qserve_layer.adopt_prefill(key_states, value_states)
        for attribute in ("_seen_tokens", "_max_batch_size", "_max_cache_len"):
            if hasattr(native_cache, attribute):
                setattr(self, attribute, getattr(native_cache, attribute))

    def packed_decode_active(self) -> bool:
        """Whether every adapted full-attention layer has crossed the threshold."""

        return bool(self.full_attention_layers) and all(
            getattr(self.layers[layer_idx], "_key_code", None) is not None
            for layer_idx in self.full_attention_layers
        )

    def __repr__(self) -> str:
        return f"QServeDeferredSplitFusedKvCache({self.report()})"


def build_qserve_deferred_split_fused_kv_cache(
    model_config: Any,
    qserve_config: QServeKvConfig | None = None,
) -> QServeDeferredSplitFusedKvCache:
    return QServeDeferredSplitFusedKvCache(model_config=model_config, qserve_config=qserve_config)
