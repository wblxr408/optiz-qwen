from __future__ import annotations

import torch

from optiz_qwen.compression import QServeAttentionLayer, QServeFusedAttentionLayer, QServeKvCache, QServeKvConfig


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


def test_qserve_cache_only_replaces_full_attention_layers() -> None:
    cache = QServeKvCache(MinimalQwenConfig(), QServeKvConfig(k_bits=4, v_bits=4, group_size=16, residual_length=16))

    assert cache.full_attention_layers == (1,)
    assert isinstance(cache.layers[1], QServeAttentionLayer)
    assert cache.layers[0].__class__.__name__ != "QServeAttentionLayer"
    assert cache.report().implementation == "qserve_kv4_local"


def test_fused_layer_keeps_decode_output_packed_until_attention() -> None:
    layer = QServeFusedAttentionLayer(
        QServeKvConfig(k_bits=4, v_bits=4, group_size=16, residual_length=16)
    )
    prefill_keys = torch.randn(1, 2, 16, 16)
    prefill_values = torch.randn_like(prefill_keys)
    decode_key = torch.randn(1, 2, 1, 16)
    decode_value = torch.randn_like(decode_key)
    layer.update(prefill_keys, prefill_values)

    returned_keys, returned_values = layer.update(decode_key, decode_value)

    assert returned_keys.shape[-2] == 1
    assert returned_values.shape[-2] == 1
    materialized_keys, materialized_values = layer.materialize()
    assert materialized_keys.shape[-2] == 17
    assert materialized_values.shape[-2] == 17
