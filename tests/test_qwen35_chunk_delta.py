from __future__ import annotations

import pytest
import torch
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

from optiz_qwen.kernels import (
    compare_chunk_delta_kernel,
    install_qwen35_chunk_delta_kernel,
    qwen35_chunk_delta_reference,
)


def make_inputs(sequence: int = 4):
    torch.manual_seed(20260801)
    query = torch.randn(1, sequence, 2, 3, dtype=torch.bfloat16)
    key = torch.randn(1, sequence, 2, 3, dtype=torch.bfloat16)
    value = torch.randn(1, sequence, 2, 4, dtype=torch.bfloat16)
    g = -torch.rand(1, sequence, 2, dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(1, sequence, 2, dtype=torch.bfloat16))
    return query, key, value, g, beta


def test_scalar_recurrence_has_known_result() -> None:
    query = torch.ones(1, 2, 1, 1)
    key = torch.ones_like(query)
    value = torch.tensor([[[[2.0]], [[4.0]]]])
    g = torch.full((1, 2, 1), torch.log(torch.tensor(0.5)))
    beta = torch.ones(1, 2, 1)

    output, state = qwen35_chunk_delta_reference(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=True,
    )

    assert torch.equal(output, torch.tensor([[[[2.0]], [[4.0]]]]))
    assert state is not None
    assert torch.equal(state, torch.tensor([[[[4.0]]]]))


def test_reference_preserves_output_dtype_and_uses_fp32_state() -> None:
    query, key, value, g, beta = make_inputs()

    output, state = qwen35_chunk_delta_reference(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    assert output.shape == (1, 4, 2, 4)
    assert output.dtype == torch.bfloat16
    assert state is not None
    assert state.shape == (1, 2, 3, 4)
    assert state.dtype == torch.float32


def test_comparison_reports_zero_for_reference_backend() -> None:
    query, key, value, g, beta = make_inputs()

    comparison = compare_chunk_delta_kernel(
        qwen35_chunk_delta_reference,
        query,
        key,
        value,
        g=g,
        beta=beta,
    )

    assert comparison.output_max_abs_error == 0.0
    assert comparison.state_max_abs_error == 0.0


def test_reference_matches_transformers_fallback_across_chunk_boundary() -> None:
    query, key, value, g, beta = make_inputs(sequence=65)
    expected_output, expected_state = torch_chunk_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    actual_output, actual_state = qwen35_chunk_delta_reference(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    assert torch.equal(actual_output, expected_output)
    assert actual_state is not None
    assert expected_state is not None
    assert torch.allclose(actual_state, expected_state, atol=1e-6, rtol=0.0)


def test_contract_rejects_packed_or_invalid_inputs() -> None:
    query, key, value, g, beta = make_inputs()
    kwargs = {
        "g": g,
        "beta": beta,
        "initial_state": None,
        "output_final_state": True,
    }

    with pytest.raises(NotImplementedError, match="packed"):
        qwen35_chunk_delta_reference(
            query,
            key,
            value,
            cu_seqlens=torch.tensor([0, 4]),
            **kwargs,
        )
    with pytest.raises(ValueError, match="value"):
        qwen35_chunk_delta_reference(query, key, value[:, :-1], **kwargs)


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.chunk_gated_delta_rule = lambda *args, **kwargs: (args[0], None)


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Qwen3_5GatedDeltaNet(0), Qwen3_5GatedDeltaNet(1)])


def test_installer_replaces_only_chunk_kernel_and_is_explicit() -> None:
    model = FakeModel().eval()

    report = install_qwen35_chunk_delta_kernel(
        model,
        qwen35_chunk_delta_reference,
        backend="ppu_hggc_v1",
    )

    assert report.layer_indices == (0, 1)
    assert report.backend == "ppu_hggc_v1"
    assert all(
        layer.chunk_gated_delta_rule is qwen35_chunk_delta_reference
        for layer in model.layers
    )
    with pytest.raises(RuntimeError, match="already installed"):
        install_qwen35_chunk_delta_kernel(
            model,
            qwen35_chunk_delta_reference,
            backend="ppu_hggc_v1",
        )
