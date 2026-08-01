from __future__ import annotations

import pytest
import torch
from torch import nn

from optiz_qwen.ppu import (
    Qwen35PpuDeltaConfig,
    get_qwen35_ppu_delta_runtime,
    install_qwen35_ppu_delta_kernel,
)


class FakeExtension:
    def __init__(self) -> None:
        self.inputs = None

    def forward(self, query, key, value, g, beta):
        self.inputs = (query, key, value, g, beta)
        state = torch.zeros(
            query.shape[0],
            query.shape[2],
            query.shape[3],
            value.shape[3],
            dtype=torch.float32,
        )
        return value, state


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.chunk_gated_delta_rule = lambda *args, **kwargs: (args[0], None)


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(Qwen3_5GatedDeltaNet(index) for index in range(18))


def test_installer_selects_trailing_layers_and_reports_runtime() -> None:
    model = FakeModel().eval()
    extension = FakeExtension()
    config = Qwen35PpuDeltaConfig(kernel_layers=9, position="last")

    report = install_qwen35_ppu_delta_kernel(
        model,
        config,
        extension=extension,
    )

    assert report.layer_indices == tuple(range(9, 18))
    assert report.kernel_layers == 9
    query = torch.randn(1, 3, 2, 4)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    g = torch.randn(1, 3, 2)
    beta = torch.randn(1, 3, 2)
    output, state = model.layers[-1].chunk_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    assert torch.equal(output, value)
    assert state is not None
    normalized_query, normalized_key, *_ = extension.inputs
    assert torch.allclose(normalized_query.square().sum(dim=-1), torch.ones(1, 3, 2))
    assert torch.allclose(normalized_key.square().sum(dim=-1), torch.ones(1, 3, 2))
    runtime = get_qwen35_ppu_delta_runtime(model)
    assert runtime is not None
    assert runtime["calls"] == 1
    assert runtime["config"] == {"kernel_layers": 9, "position": "last"}


def test_kernel_rejects_unsupported_state_and_packed_inputs() -> None:
    model = FakeModel().eval()
    install_qwen35_ppu_delta_kernel(
        model,
        Qwen35PpuDeltaConfig(kernel_layers=1),
        extension=FakeExtension(),
    )
    kernel = model.layers[-1].chunk_gated_delta_rule
    query = torch.randn(1, 2, 1, 4)
    kwargs = {
        "g": torch.randn(1, 2, 1),
        "beta": torch.randn(1, 2, 1),
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
    }

    with pytest.raises(NotImplementedError, match="initial state"):
        kernel(query, query, query, initial_state=torch.zeros(1), **kwargs)
    with pytest.raises(NotImplementedError, match="packed"):
        kernel(
            query,
            query,
            query,
            initial_state=None,
            cu_seqlens=torch.tensor([0, 2]),
            **kwargs,
        )
    with pytest.raises(TypeError, match="unsupported_option"):
        kernel(
            query,
            query,
            query,
            initial_state=None,
            unsupported_option=True,
            **kwargs,
        )


def test_installation_is_explicit_and_validates_configuration() -> None:
    with pytest.raises(ValueError, match="positive"):
        Qwen35PpuDeltaConfig(kernel_layers=0)
    with pytest.raises(ValueError, match="position"):
        Qwen35PpuDeltaConfig(position="middle")

    model = FakeModel()
    with pytest.raises(ValueError, match="eval"):
        install_qwen35_ppu_delta_kernel(model, extension=FakeExtension())

    model.eval()
    with pytest.raises(ValueError, match="exceeds"):
        install_qwen35_ppu_delta_kernel(
            model,
            Qwen35PpuDeltaConfig(kernel_layers=19),
            extension=FakeExtension(),
        )
