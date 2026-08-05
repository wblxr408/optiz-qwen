from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from optiz_qwen.compression import (
    QServeDeferredSplitFusedAttentionLayer,
    QServeDeferredSplitFusedKvCache,
    QServeKvConfig,
    build_qserve_deferred_split_fused_kv_cache,
)
from optiz_qwen.compression.qserve_kv_cache import (
    QServeAttentionLayer,
)
from optiz_qwen.scheduling import build_kv_chain
from optiz_qwen.kernels.qwen35_fused_attention import _qserve_fused_forward
from optiz_qwen.kernels.qserve_int4_attention import (
    qserve_int4_split_decode_attention,
    qserve_int8_decode_attention,
)


class MinimalQwenConfig:
    layer_types = ("linear_attention", "full_attention", "linear_attention")
    num_hidden_layers = 3

    def get_text_config(self, decoder: bool = True):
        return self


def test_qserve_attention_layer_materializes_dense_cache() -> None:
    layer = QServeAttentionLayer(QServeKvConfig(k_bits=4, v_bits=4, group_size=16, residual_length=16))
    keys = torch.randn(1, 2, 16, 16)
    values = torch.randn(1, 2, 16, 16)

    out_keys, out_values = layer.update(keys, values)

    assert out_keys.shape == keys.shape
    assert out_values.shape == values.shape
    assert layer.get_seq_length() == 16
    assert layer.materialize()[0].shape == keys.shape


def test_qserve_attention_layer_streams_decode_and_preserves_dtype() -> None:
    layer = QServeAttentionLayer(QServeKvConfig(k_bits=4, v_bits=4, group_size=16, residual_length=16))
    prefill_keys = torch.randn(1, 2, 16, 16, dtype=torch.bfloat16)
    prefill_values = torch.randn(1, 2, 16, 16, dtype=torch.bfloat16)
    decode_key = torch.randn(1, 2, 1, 16, dtype=torch.bfloat16)
    decode_value = torch.randn(1, 2, 1, 16, dtype=torch.bfloat16)

    layer.update(prefill_keys, prefill_values)
    out_keys, out_values = layer.update(decode_key, decode_value)

    assert out_keys.dtype == torch.bfloat16
    assert out_values.dtype == torch.bfloat16
    assert out_keys.shape[-2] == 17
    assert out_values.shape[-2] == 17


def test_deferred_split_cache_preserves_prefill_dense_until_decode() -> None:
    layer = QServeDeferredSplitFusedAttentionLayer(
        QServeKvConfig(
            k_bits=4,
            v_bits=4,
            group_size=16,
            residual_length=16,
            activation_threshold=32,
        )
    )
    prefill_keys = torch.randn(1, 2, 16, 16)
    prefill_values = torch.randn_like(prefill_keys)
    layer.update(prefill_keys, prefill_values)

    assert layer._deferred_prefill is True
    assert layer._key_code is None
    assert torch.equal(layer.materialize()[0], prefill_keys)

    decode_key = torch.randn(1, 2, 1, 16)
    decode_value = torch.randn_like(decode_key)
    layer.update(decode_key, decode_value)

    assert layer._deferred_prefill is True
    assert layer._key_code is None
    assert torch.equal(layer.materialize()[0], torch.cat([prefill_keys, decode_key], dim=-2))
    assert layer.get_seq_length() == 17


def test_deferred_split_layer_packs_only_after_activation_threshold() -> None:
    layer = QServeDeferredSplitFusedAttentionLayer(
        QServeKvConfig(
            k_bits=4,
            v_bits=4,
            group_size=16,
            residual_length=16,
            activation_threshold=32,
        )
    )
    prefill_keys = torch.randn(1, 2, 31, 16)
    prefill_values = torch.randn_like(prefill_keys)
    layer.update(prefill_keys, prefill_values)
    layer.update(torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16))

    assert layer._key_code is not None
    assert layer._deferred_prefill is False


def test_deferred_split_layer_can_adopt_native_prefill_tensors() -> None:
    layer = QServeDeferredSplitFusedAttentionLayer(
        QServeKvConfig(k_bits=4, v_bits=4, group_size=16, residual_length=16)
    )
    prefill_keys = torch.randn(1, 2, 16, 16)
    prefill_values = torch.randn_like(prefill_keys)
    layer.adopt_prefill(prefill_keys, prefill_values)

    assert layer._deferred_prefill is True
    assert layer.get_seq_length() == 16
    assert torch.equal(layer.materialize()[1], prefill_values)


def test_deferred_split_chain_reports_its_runtime_implementation() -> None:
    cache, report = build_kv_chain(
        chain_name="qserve_deferred_split_fused_kv",
        model_config=MinimalQwenConfig(),
        enabled=True,
        qserve_config=QServeKvConfig(k_bits=4, v_bits=4, group_size=16, residual_length=16),
    )

    assert isinstance(cache, QServeDeferredSplitFusedKvCache)
    assert report.implementation == "qserve_triton_int4_deferred_split_decode"


def test_qserve_config_rejects_unknown_attention_backend() -> None:
    with pytest.raises(ValueError, match="attention_backend"):
        QServeKvConfig(attention_backend="unknown").validate()


def test_qserve_config_requires_int8_backend_for_8bit_kv() -> None:
    with pytest.raises(ValueError, match="requires triton_int8_decode"):
        QServeKvConfig(k_bits=8, v_bits=8).validate()


def test_removed_kv_chain_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="Only qserve_deferred_split_fused_kv is retained"):
        build_kv_chain(
            chain_name="qserve_fused_kv",
            model_config=MinimalQwenConfig(),
            enabled=True,
        )


def test_fused_attention_delegates_prefill_to_the_native_forward() -> None:
    class Attention:
        def __init__(self) -> None:
            self._optiz_original_forward = self.native_forward
            self.calls = 0

        def native_forward(self, hidden_states, **kwargs):
            self.calls += 1
            return hidden_states, kwargs

    attention = Attention()
    hidden_states = torch.randn(1, 2, 16)
    result = _qserve_fused_forward(
        attention,
        hidden_states,
        position_embeddings=(torch.empty(0), torch.empty(0)),
        attention_mask=None,
    )

    assert attention.calls == 1
    assert result[0] is hidden_states


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA Triton runtime")
def test_split_kernel_matches_eager_attention_for_the_same_quantized_kv() -> None:
    torch.manual_seed(31)
    layer = QServeAttentionLayer(
        QServeKvConfig(k_bits=4, v_bits=4, group_size=32, residual_length=32)
    )
    keys = torch.randn(1, 2, 32, 32, device="cuda", dtype=torch.float16)
    values = torch.randn_like(keys)
    query = torch.randn(1, 8, 1, 32, device="cuda", dtype=torch.float16)
    layer.update(keys, values)

    actual = qserve_int4_split_decode_attention(query, layer, scaling=1 / (32**0.5))
    dense_keys, dense_values = layer.materialize()
    expected = F.scaled_dot_product_attention(
        query,
        dense_keys.repeat_interleave(4, dim=1),
        dense_values.repeat_interleave(4, dim=1),
        is_causal=False,
    ).transpose(1, 2)

    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=5e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA Triton runtime")
def test_int8_kernel_matches_eager_attention_for_the_same_quantized_kv() -> None:
    torch.manual_seed(37)
    layer = QServeAttentionLayer(
        QServeKvConfig(
            k_bits=8,
            v_bits=8,
            group_size=32,
            residual_length=32,
            attention_backend="triton_int8_decode",
        )
    )
    keys = torch.randn(1, 2, 32, 32, device="cuda", dtype=torch.float16)
    values = torch.randn_like(keys)
    query = torch.randn(1, 8, 1, 32, device="cuda", dtype=torch.float16)
    layer.update(keys, values)

    actual = qserve_int8_decode_attention(query, layer, scaling=1 / (32**0.5))
    dense_keys, dense_values = layer.materialize()
    expected = F.scaled_dot_product_attention(
        query,
        dense_keys.repeat_interleave(4, dim=1),
        dense_values.repeat_interleave(4, dim=1),
        is_causal=False,
    ).transpose(1, 2)

    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=5e-4)
