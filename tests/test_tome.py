from __future__ import annotations

import pytest
import torch

from optiz_qwen.compression.tome import merge_single_visual_sample, merge_visual_units


def make_inputs(unit_values: list[float], boundaries: list[int] | None = None):
    hidden_states = torch.tensor(unit_values).repeat_interleave(4).reshape(-1, 1)
    metric = torch.stack((hidden_states[:, 0], 1 - hidden_states[:, 0]), dim=-1)
    token_sizes = torch.ones(hidden_states.shape[0], 1)
    cu_seqlens = torch.tensor(boundaries or [0, hidden_states.shape[0]], dtype=torch.int32)
    return hidden_states, metric, token_sizes, cu_seqlens


def test_r_zero_is_identity() -> None:
    inputs = make_inputs([0.0, 0.1, 0.9, 1.0])

    result = merge_visual_units(*inputs, r=0)

    assert torch.equal(result.hidden_states, inputs[0])
    assert torch.equal(result.token_sizes, inputs[2])
    assert torch.equal(result.retained_token_indices, torch.arange(16))
    assert result.source_unit_indices.numel() == 0
    assert result.destination_unit_indices.numel() == 0


def test_r_zero_accepts_a_single_unit() -> None:
    inputs = make_inputs([0.0])

    result = merge_visual_units(*inputs, r=0)

    assert torch.equal(result.hidden_states, inputs[0])
    assert result.cu_seqlens.tolist() == [0, 4]


def test_merges_most_similar_units_and_preserves_spatial_order() -> None:
    inputs = make_inputs([0.0, 0.1, 0.9, 1.0])

    result = merge_visual_units(*inputs, r=1)

    assert result.hidden_states.shape == (12, 1)
    assert result.source_unit_indices.tolist() == [0]
    assert result.destination_unit_indices.tolist() == [1]
    assert result.retained_token_indices.tolist() == list(range(4, 16))
    assert torch.allclose(result.hidden_states[:4], torch.full((4, 1), 0.05))
    assert torch.equal(result.hidden_states[4:], inputs[0][8:])


def test_weighted_merge_conserves_token_size() -> None:
    hidden_states, metric, token_sizes, cu_seqlens = make_inputs([0.0, 0.2])
    token_sizes[:4] = 3

    result = merge_visual_units(hidden_states, metric, token_sizes, cu_seqlens, r=1)

    assert torch.allclose(result.hidden_states, torch.full((4, 1), 0.05))
    assert torch.equal(result.token_sizes, torch.full((4, 1), 4.0))
    assert result.token_sizes.sum() == token_sizes.sum()


def test_packed_samples_are_matched_independently() -> None:
    inputs = make_inputs([0.0, 0.1, 0.9, 1.0], boundaries=[0, 8, 16])

    result = merge_visual_units(*inputs, r=1)

    assert result.cu_seqlens.tolist() == [0, 4, 8]
    assert result.source_unit_indices.tolist() == [0, 2]
    assert result.destination_unit_indices.tolist() == [1, 3]
    assert torch.allclose(result.hidden_states[:4], torch.full((4, 1), 0.05))
    assert torch.allclose(result.hidden_states[4:], torch.full((4, 1), 0.95))


def test_retained_indices_can_trim_position_embeddings() -> None:
    inputs = make_inputs([0.0, 0.1, 0.9, 1.0])
    positions = torch.arange(16)

    result = merge_visual_units(*inputs, r=1)

    assert torch.equal(positions[result.retained_token_indices], torch.arange(4, 16))


def test_single_sample_path_matches_packed_path() -> None:
    hidden_states, metric, token_sizes, cu_seqlens = make_inputs([0.0, 0.1, 0.9, 1.0])

    packed = merge_visual_units(hidden_states, metric, token_sizes, cu_seqlens, r=1)
    single = merge_single_visual_sample(hidden_states, metric, token_sizes, r=1)

    assert torch.equal(single.hidden_states, packed.hidden_states)
    assert torch.equal(single.token_sizes, packed.token_sizes)
    assert torch.equal(single.cu_seqlens, packed.cu_seqlens)
    assert torch.equal(single.retained_token_indices, packed.retained_token_indices)


def test_profile_reports_each_merge_stage() -> None:
    hidden_states, metric, token_sizes, _ = make_inputs([0.0, 0.1, 0.9, 1.0])

    result = merge_single_visual_sample(hidden_states, metric, token_sizes, r=1, profile=True)

    assert result.timings_ms is not None
    assert set(result.timings_ms) == {
        "metric_preparation",
        "bipartite_matching",
        "weighted_aggregation",
        "output_compaction",
        "result_assembly",
        "total",
    }
    assert result.timings_ms["total"] >= 0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_cpu_and_mps_produce_the_same_merge() -> None:
    cpu_inputs = make_inputs([0.0, 0.2, 0.7, 1.0])
    mps_inputs = tuple(tensor.to("mps") for tensor in cpu_inputs)

    cpu_result = merge_visual_units(*cpu_inputs, r=1)
    mps_result = merge_visual_units(*mps_inputs, r=1)

    assert torch.allclose(mps_result.hidden_states.cpu(), cpu_result.hidden_states)
    assert torch.equal(mps_result.token_sizes.cpu(), cpu_result.token_sizes)
    assert torch.equal(mps_result.retained_token_indices.cpu(), cpu_result.retained_token_indices)
    assert torch.equal(mps_result.source_unit_indices.cpu(), cpu_result.source_unit_indices)
    assert torch.equal(mps_result.destination_unit_indices.cpu(), cpu_result.destination_unit_indices)


@pytest.mark.parametrize(
    ("input_index", "replacement", "message"),
    [
        (0, torch.zeros(8), "hidden_states"),
        (1, torch.zeros(7, 2), "metric"),
        (2, torch.ones(8), "token_sizes"),
        (3, torch.tensor([0, 2, 8], dtype=torch.int32), "divisible"),
    ],
)
def test_rejects_invalid_shapes(input_index, replacement, message) -> None:
    inputs = list(make_inputs([0.0, 0.1]))
    inputs[input_index] = replacement

    with pytest.raises(ValueError, match=message):
        merge_visual_units(*inputs, r=1)


def test_rejects_r_above_matching_capacity() -> None:
    inputs = make_inputs([0.0, 0.1])

    with pytest.raises(ValueError, match="capacity"):
        merge_visual_units(*inputs, r=2)
