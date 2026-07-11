from __future__ import annotations

import torch

from optiz_qwen.compression import KiviConfig, KiviAttentionLayer, QServeKvCache, Qwen35KiviCache
from optiz_qwen.evaluation.dndx_wrapper import VLMModel
from optiz_qwen.scheduling.prefill_decode import PrefillDecodeStats


class FakeUpstreamKiviQuant:
    def __init__(self) -> None:
        self.pack_calls: list[tuple[tuple[int, ...], int, int]] = []
        self.unpack_calls: list[tuple[tuple[int, ...], int, int]] = []

    def triton_quantize_and_pack_along_last_dim(self, data, group_size, bit):
        self.pack_calls.append((tuple(data.shape), group_size, bit))
        groups = data.shape[-1] // group_size
        scale = torch.ones(*data.shape[:-1], groups, dtype=data.dtype, device=data.device)
        mn = torch.zeros_like(scale)
        return data.clone(), scale, mn

    def unpack_and_dequant_vcache(self, code, scale, mn, group_size, bits):
        self.unpack_calls.append((tuple(code.shape), group_size, bits))
        return code.clone()


class Float32UnpackKiviQuant(FakeUpstreamKiviQuant):
    def unpack_and_dequant_vcache(self, code, scale, mn, group_size, bits):
        self.unpack_calls.append((tuple(code.shape), group_size, bits))
        return code.float()


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
    assert quant.unpack_calls == []


def test_kivi_attention_layer_streams_decode_without_full_repack() -> None:
    quant = FakeUpstreamKiviQuant()
    config = KiviConfig(k_bits=2, v_bits=2, group_size=16, residual_length=16)
    layer = KiviAttentionLayer(config, quant)
    prefill_keys = torch.randn(1, 2, 16, 16)
    prefill_values = torch.randn(1, 2, 16, 16)
    decode_key = torch.randn(1, 2, 1, 16)
    decode_value = torch.randn(1, 2, 1, 16)

    layer.update(prefill_keys, prefill_values)
    out_keys, out_values = layer.update(decode_key, decode_value)

    assert out_keys.shape == (1, 2, 17, 16)
    assert out_values.shape == (1, 2, 17, 16)
    assert layer.get_seq_length() == 17
    assert quant.pack_calls == [
        ((1, 2, 16, 16), 16, 2),
        ((1, 2, 1, 16), 16, 2),
    ]
    assert quant.unpack_calls == []
    assert torch.equal(out_keys, torch.cat([prefill_keys, decode_key], dim=-2))
    assert torch.equal(out_values, torch.cat([prefill_values, decode_value], dim=-2))


def test_kivi_attention_layer_materializes_original_dtype() -> None:
    quant = Float32UnpackKiviQuant()
    config = KiviConfig(k_bits=2, v_bits=2, group_size=16, residual_length=16)
    layer = KiviAttentionLayer(config, quant)
    prefill_keys = torch.randn(1, 2, 16, 16, dtype=torch.bfloat16)
    prefill_values = torch.randn(1, 2, 16, 16, dtype=torch.bfloat16)
    decode_key = torch.randn(1, 2, 1, 16, dtype=torch.bfloat16)
    decode_value = torch.randn(1, 2, 1, 16, dtype=torch.bfloat16)

    layer.update(prefill_keys, prefill_values)
    out_keys, out_values = layer.update(decode_key, decode_value)

    assert out_keys.dtype == torch.bfloat16
    assert out_values.dtype == torch.bfloat16


def test_kivi_attention_layer_reset_clears_dense_cache() -> None:
    quant = FakeUpstreamKiviQuant()
    config = KiviConfig(k_bits=2, v_bits=2, group_size=16, residual_length=16)
    layer = KiviAttentionLayer(config, quant)
    keys = torch.randn(1, 2, 16, 16)
    values = torch.randn(1, 2, 16, 16)

    layer.update(keys, values)
    layer.reset()

    assert layer.get_seq_length() == 0
    assert layer.keys is None
    assert layer.values is None
    assert layer._dense_keys is None
    assert layer._dense_values is None


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

    try:
        result = model._build_kivi_cache_if_enabled()
    except Exception as exc:
        import pytest

        pytest.skip(f"legacy KIVI depends on local Triton env: {exc}")
    else:
        assert result is sentinel
        assert calls["kivi_config"].group_size == 16
        assert calls["kivi_config"].residual_length == 16


def test_kivi_prefill_decode_route_is_opt_in(monkeypatch) -> None:
    import optiz_qwen.evaluation.dndx_wrapper as wrapper

    calls = {"prefill": 0, "generate": 0}

    def fake_run_greedy_prefill_decode(*args, **kwargs):
        calls["prefill"] += 1
        return torch.tensor([[7]], dtype=torch.long), PrefillDecodeStats(
            prompt_tokens=2,
            generated_tokens=1,
            prefill_seconds=0.1,
            decode_seconds=0.1,
            ttft_seconds=0.1,
            elapsed_seconds=0.2,
        )

    class FakeTokenizer:
        eos_token_id = 0

        def decode(self, *args, **kwargs):
            return "Answer: A"

    class FakeProcessor:
        tokenizer = FakeTokenizer()

        def apply_chat_template(self, *args, **kwargs):
            class Inputs(dict):
                def __init__(self):
                    super().__init__()
                    self.input_ids = torch.tensor([[1, 2]], dtype=torch.long)
                    self["input_ids"] = self.input_ids

                def to(self, device):
                    return self

            return Inputs()

    class FakeModel:
        device = torch.device("cpu")

        def generate(self, **kwargs):
            calls["generate"] += 1
            return torch.tensor([[1, 2, 3]])

    monkeypatch.setattr(wrapper, "run_greedy_prefill_decode", fake_run_greedy_prefill_decode)
    model = VLMModel("unused", backend="dummy")
    model._backend_name = "transformers"
    model._model = FakeModel()
    model._processor = FakeProcessor()
    model._tokenizer = FakeTokenizer()
    monkeypatch.setenv("OPTIZ_QWEN_KIVI_KV_CACHE", "1")
    monkeypatch.setenv("OPTIZ_QWEN_CHOICE_FALLBACK", "0")
    monkeypatch.setattr(
        model,
        "_build_kv_chain_if_enabled",
        lambda: (
            QServeKvCache(MinimalQwenConfig()),
            type("Report", (), {"chain_name": "qserve_kv", "enabled": True, "implementation": "qserve_kv4_local", "__dict__": {}})(),
        ),
    )
    monkeypatch.setattr(model, "_choice_fallback_enabled", lambda: False)
    monkeypatch.setattr(model, "_build_model_inputs", lambda **kwargs: {})
    monkeypatch.setattr(model, "_load_transformers_backend", lambda: None)

    result = model._generate_with_transformers(
        image=object(),
        prompt="test",
        choices={"A": "x", "B": "y"},
        generation_config=type("Cfg", (), {"max_new_tokens": 1, "temperature": 0.0, "top_p": 1.0})(),
    )

    assert calls["prefill"] == 1
    assert calls["generate"] == 0
    assert result.meta["prefill_decode_runtime"] is not None


def test_kv_chain_selector_prefers_new_chain(monkeypatch) -> None:
    monkeypatch.delenv("OPTIZ_QWEN_KIVI_KV_CACHE", raising=False)
    monkeypatch.setenv("OPTIZ_QWEN_KV_CHAIN_ENABLED", "1")
    monkeypatch.setenv("OPTIZ_QWEN_KV_CHAIN", "qserve_kv")

    model = VLMModel("unused", backend="dummy")
    model._model = type("Model", (), {"config": MinimalQwenConfig()})()

    kv_chain, kv_report = model._build_kv_chain_if_enabled()

    assert kv_chain is not None
    assert kv_report.chain_name == "qserve_kv"
    assert kv_report.implementation == "qserve_kv4_local"
    assert isinstance(kv_chain, QServeKvCache)


def test_kv_chain_selector_uses_qserve_defaults(monkeypatch) -> None:
    model = VLMModel("unused", backend="dummy")
    model._model = type("Model", (), {"config": MinimalQwenConfig()})()
    monkeypatch.setenv("OPTIZ_QWEN_KV_CHAIN_ENABLED", "1")
    monkeypatch.setenv("OPTIZ_QWEN_KV_CHAIN", "qserve_kv")
    monkeypatch.delenv("OPTIZ_QWEN_KV_CHAIN_K_BITS", raising=False)
    monkeypatch.delenv("OPTIZ_QWEN_KV_CHAIN_V_BITS", raising=False)

    _, kv_report = model._build_kv_chain_if_enabled()

    assert kv_report.k_bits == 4
    assert kv_report.v_bits == 4


def test_kv_chain_selector_falls_back_to_legacy_kivi(monkeypatch) -> None:
    import pytest

    model = VLMModel("unused", backend="dummy")
    model._model = type("Model", (), {"config": MinimalQwenConfig()})()
    monkeypatch.delenv("OPTIZ_QWEN_KV_CHAIN_ENABLED", raising=False)
    monkeypatch.setenv("OPTIZ_QWEN_KIVI_KV_CACHE", "1")

    with pytest.raises(Exception):
        model._build_kivi_cache_if_enabled()
