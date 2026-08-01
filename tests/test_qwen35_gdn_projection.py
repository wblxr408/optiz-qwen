from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from optiz_qwen.ppu import (
    get_qwen35_gdn_decode_projection_runtime,
    install_qwen35_gdn_decode_projection_fusion,
)


class FakeCache:
    def __init__(self, previous: bool) -> None:
        self.previous = previous

    def has_previous_state(self, layer_index: int) -> bool:
        assert layer_index == 0
        return self.previous


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_idx = 0
        self.in_proj_qkv = nn.Linear(8, 12, bias=False)
        self.in_proj_z = nn.Linear(8, 6, bias=False)
        self.in_proj_b = nn.Linear(8, 2, bias=False)
        self.in_proj_a = nn.Linear(8, 2, bias=False)

    def forward(self, hidden_states, cache_params=None, attention_mask=None):
        return (
            self.in_proj_qkv(hidden_states),
            self.in_proj_z(hidden_states),
            self.in_proj_b(hidden_states),
            self.in_proj_a(hidden_states),
        )


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gdn = Qwen3_5GatedDeltaNet()


def assert_outputs_equal(expected, actual) -> None:
    assert len(expected) == len(actual)
    for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
        assert torch.equal(expected_tensor, actual_tensor)


def make_models() -> tuple[FakeModel, FakeModel]:
    torch.manual_seed(20260801)
    baseline = FakeModel().eval()
    candidate = copy.deepcopy(baseline).eval()
    return baseline, candidate


def test_install_packs_weights_without_extra_steady_state_storage() -> None:
    _baseline, candidate = make_models()
    state_keys = set(candidate.state_dict())
    original_shapes = {
        name: tuple(getattr(candidate.gdn, name).weight.shape)
        for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
    }

    report = install_qwen35_gdn_decode_projection_fusion(candidate)

    weights = [
        getattr(candidate.gdn, name).weight
        for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
    ]
    assert report.layer_count == 1
    assert report.projection_count == 4
    assert report.original_weight_bytes == report.packed_weight_bytes
    assert report.extra_steady_state_weight_bytes == 0
    assert len({weight.untyped_storage().data_ptr() for weight in weights}) == 1
    assert set(candidate.state_dict()) == state_keys
    assert {
        name: tuple(getattr(candidate.gdn, name).weight.shape)
        for name in original_shapes
    } == original_shapes


def test_prefill_keeps_the_four_projection_path_exact() -> None:
    baseline, candidate = make_models()
    install_qwen35_gdn_decode_projection_fusion(candidate)
    hidden_states = torch.randn(1, 5, 8)

    expected = baseline.gdn(hidden_states, cache_params=FakeCache(previous=False))
    actual = candidate.gdn(hidden_states, cache_params=FakeCache(previous=False))

    assert_outputs_equal(expected, actual)
    runtime = get_qwen35_gdn_decode_projection_runtime(candidate)
    assert runtime is not None
    assert runtime["fused_decode_calls"] == 0
    assert runtime["baseline_calls"] == 1


def test_cached_single_token_decode_is_exact_and_fused() -> None:
    baseline, candidate = make_models()
    install_qwen35_gdn_decode_projection_fusion(candidate)
    hidden_states = torch.randn(1, 1, 8)

    expected = baseline.gdn(hidden_states, cache_params=FakeCache(previous=True))
    actual = candidate.gdn(hidden_states, cache_params=FakeCache(previous=True))

    assert_outputs_equal(expected, actual)
    runtime = get_qwen35_gdn_decode_projection_runtime(candidate)
    assert runtime is not None
    assert runtime["fused_decode_calls"] == 1
    assert runtime["baseline_calls"] == 0
    assert runtime["unique_packed_storage_bytes"] == runtime["report"]["packed_weight_bytes"]


def test_uncached_single_token_call_is_also_exactly_fused() -> None:
    _baseline, candidate = make_models()
    install_qwen35_gdn_decode_projection_fusion(candidate)

    candidate.gdn(torch.randn(1, 1, 8), cache_params=FakeCache(previous=False))

    runtime = get_qwen35_gdn_decode_projection_runtime(candidate)
    assert runtime is not None
    assert runtime["fused_decode_calls"] == 1
    assert runtime["baseline_calls"] == 0


def test_decode_rejects_batch_larger_than_one() -> None:
    _baseline, candidate = make_models()
    install_qwen35_gdn_decode_projection_fusion(candidate)

    with pytest.raises(ValueError, match="batch size 1"):
        candidate.gdn(torch.randn(2, 1, 8), cache_params=FakeCache(previous=True))


def test_storage_change_after_installation_fails_early() -> None:
    _baseline, candidate = make_models()
    install_qwen35_gdn_decode_projection_fusion(candidate)

    with pytest.raises(RuntimeError, match="cannot move device or dtype"):
        candidate.to(dtype=torch.float64)


def test_installation_requires_eval_and_is_not_repeatable() -> None:
    training_model = FakeModel()
    with pytest.raises(ValueError, match="eval"):
        install_qwen35_gdn_decode_projection_fusion(training_model)

    training_model.eval()
    install_qwen35_gdn_decode_projection_fusion(training_model)
    with pytest.raises(RuntimeError, match="already installed"):
        install_qwen35_gdn_decode_projection_fusion(training_model)
