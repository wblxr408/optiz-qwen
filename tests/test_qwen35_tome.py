from __future__ import annotations

import pytest
import torch
from torch import nn

from optiz_qwen.compression.qwen35_tome import (
    Qwen35TomeConfig,
    get_qwen35_tome_runtime,
    install_qwen35_tome,
    set_qwen35_tome_enabled,
)


class FakeAttention(nn.Module):
    def __init__(self, hidden_size: int = 2) -> None:
        super().__init__()
        self.num_heads = 1
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        nn.init.eye_(self.qkv.weight[:hidden_size])
        nn.init.eye_(self.qkv.weight[hidden_size : hidden_size * 2])
        nn.init.eye_(self.qkv.weight[hidden_size * 2 :])
        self.seen_lengths: list[int] = []

    def forward(self, hidden_states, cu_seqlens, position_embeddings, **kwargs):
        self.seen_lengths.append(hidden_states.shape[0])
        self.qkv(hidden_states)
        return torch.zeros_like(hidden_states)


class FakeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.Identity()
        self.norm2 = nn.Identity()
        self.attn = FakeAttention()
        self.mlp = nn.Identity()

    def forward(self, hidden_states, cu_seqlens, position_embeddings, **kwargs):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class FakeVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([FakeBlock(), FakeBlock()])


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.visual = FakeVisual()


def test_single_layer_adapter_updates_following_layer_context() -> None:
    model = FakeModel()
    install_qwen35_tome(model, Qwen35TomeConfig(layer=0, r=1))
    hidden_states = torch.tensor([[0.0, 1.0]] * 4 + [[0.1, 0.9]] * 4)
    positions = (torch.arange(16).reshape(8, 2), torch.arange(16).reshape(8, 2))
    cu_seqlens = torch.tensor([0, 8], dtype=torch.int32)

    hidden_states = model.model.visual.blocks[0](
        hidden_states,
        cu_seqlens=cu_seqlens,
        position_embeddings=positions,
    )
    hidden_states = model.model.visual.blocks[1](
        hidden_states,
        cu_seqlens=cu_seqlens,
        position_embeddings=positions,
    )

    runtime = get_qwen35_tome_runtime(model)
    assert hidden_states.shape == (8, 2)
    assert runtime is not None
    assert runtime["input_tokens"] == 8
    assert runtime["compact_tokens"] == 4
    assert runtime["restored_tokens"] == 8
    assert runtime["merged_units"] == 1
    assert model.model.visual.blocks[1].block.attn.seen_lengths == [4]


def test_default_model_is_not_modified_without_installation() -> None:
    model = FakeModel()

    assert get_qwen35_tome_runtime(model) is None
    assert isinstance(model.model.visual.blocks[0], FakeBlock)


def test_installed_adapter_can_be_disabled_for_paired_measurement() -> None:
    model = FakeModel()
    install_qwen35_tome(model, Qwen35TomeConfig(layer=0, r=1))
    set_qwen35_tome_enabled(model, False)
    hidden_states = torch.tensor([[0.0, 1.0]] * 4 + [[0.1, 0.9]] * 4)
    positions = (torch.arange(16).reshape(8, 2), torch.arange(16).reshape(8, 2))
    cu_seqlens = torch.tensor([0, 8], dtype=torch.int32)

    for block in model.model.visual.blocks:
        hidden_states = block(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=positions,
        )

    assert hidden_states.shape == (8, 2)
    assert get_qwen35_tome_runtime(model) is None
    assert model.model.visual.blocks[1].block.attn.seen_lengths == [8]


@pytest.mark.parametrize(
    "config",
    [
        Qwen35TomeConfig(layer=-1, r=1),
        Qwen35TomeConfig(layer=2, r=1),
        Qwen35TomeConfig(layer=0, r=0),
        Qwen35TomeConfig(layer=0, r=1, unit_size=2),
    ],
)
def test_rejects_invalid_adapter_config(config) -> None:
    with pytest.raises(ValueError):
        install_qwen35_tome(FakeModel(), config)


def test_rejects_duplicate_installation() -> None:
    model = FakeModel()
    config = Qwen35TomeConfig(layer=0, r=1)
    install_qwen35_tome(model, config)

    with pytest.raises(RuntimeError, match="already installed"):
        install_qwen35_tome(model, config)
