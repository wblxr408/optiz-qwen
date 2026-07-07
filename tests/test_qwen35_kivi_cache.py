from __future__ import annotations

import torch

from optiz_qwen.compression import KiviConfig, KiviAttentionLayer, Qwen35KiviCache
from optiz_qwen.evaluation.dndx_wrapper import VLMModel


class FakeUpstreamKiviQuant:
    def __init__(self) -> None:
        self.pack_calls: list[tuple[tuple[int, ...], int, int]] = []

    def triton_quantize_and_pack_along_last_dim(self, data, group_size, bit):
        self.pack_calls.append((tuple(data.shape), group_size, bit))
        groups = data.shape[-1] // group_size
        scale = torch.ones(*data.shape[:-1], groups, dtype=data.dtype, device=data.device)
        mn = torch.zeros_like(scale)
        return data.clone(), scale, mn

    def unpack_and_dequant_vcache(self, code, scale, mn, group_size, bits):
        return code.clone()


class MinimalQwenConfig:
    layer_types = ("linear_attention", "full_attention")
    num_hidden_layers = 2

    def get_text_config(self, decoder: bool = True):
        return self


def test_kivi_attention_layer_uses_upstream_pack_unpack_shape() -> None:
    quant = FakeUpstreamKiviQuant()
    config = KiviConfig(k_bits=2, v_bits=2, group_size=16, residual_length=16)
    layer = KiviAttentionLayer(config, quant)
    keys = torch.randn(1, 2, 18, 16)
    values = torch.randn(1, 2, 18, 16)

    out_keys, out_values = layer.update(keys, values)

    assert torch.equal(out_keys, keys)
    assert torch.equal(out_values, values)
    assert layer.get_seq_length() == 18
    assert quant.pack_calls == [
        ((1, 2, 16, 16), 16, 2),
        ((1, 2, 2, 16), 16, 2),
    ]


def test_qwen35_kivi_cache_replaces_only_full_attention_layers() -> None:
    quant = FakeUpstreamKiviQuant()
    cache = Qwen35KiviCache(
        MinimalQwenConfig(),
        KiviConfig(k_bits=2, v_bits=2, group_size=16, residual_length=16),
        quant_module=quant,
    )

    assert cache.full_attention_layers == (1,)
    assert isinstance(cache.layers[1], KiviAttentionLayer)
    assert cache.layers[0].__class__.__name__ != "KiviAttentionLayer"


def test_qwen35_kivi_cache_updates_full_attention_layer() -> None:
    quant = FakeUpstreamKiviQuant()
    cache = Qwen35KiviCache(
        MinimalQwenConfig(),
        KiviConfig(k_bits=2, v_bits=2, group_size=16, residual_length=16),
        quant_module=quant,
    )
    keys = torch.randn(1, 2, 16, 16)
    values = torch.randn(1, 2, 16, 16)

    out_keys, out_values = cache.update(keys, values, 1)

    assert torch.equal(out_keys, keys)
    assert torch.equal(out_values, values)
    assert cache.get_seq_length() == 16
    assert cache.report().full_attention_layers == (1,)


def test_dndx_wrapper_builds_kivi_cache_when_enabled(monkeypatch) -> None:
    calls = {}
    sentinel = object()

    def fake_build(model_config, kivi_config):
        calls["model_config"] = model_config
        calls["kivi_config"] = kivi_config
        return sentinel

    import optiz_qwen.compression as compression

    monkeypatch.setattr(compression, "build_qwen35_kivi_cache", fake_build)
    monkeypatch.setenv("OPTIZ_QWEN_KIVI_KV_CACHE", "1")
    monkeypatch.setenv("OPTIZ_QWEN_KIVI_GROUP_SIZE", "16")
    monkeypatch.setenv("OPTIZ_QWEN_KIVI_RESIDUAL_LENGTH", "16")
    model = VLMModel("unused", backend="dummy")
    model._model = type("Model", (), {"config": MinimalQwenConfig()})()

    assert model._build_kivi_cache_if_enabled() is sentinel
    assert calls["kivi_config"].group_size == 16
    assert calls["kivi_config"].residual_length == 16
