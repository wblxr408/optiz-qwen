"""Decode-only packed input projections for Qwen3.5 Gated DeltaNet."""

from __future__ import annotations

import threading
import types
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


PROJECTION_NAMES = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
INSTALLATION_ATTRIBUTE = "_optiz_gdn_decode_projection_fusion"


@dataclass(frozen=True)
class GdnDecodeProjectionFusionReport:
    layer_count: int
    projection_count: int
    original_weight_bytes: int
    packed_weight_bytes: int
    extra_steady_state_weight_bytes: int


class _LayerProjectionFusion:
    def __init__(self, layer: nn.Module) -> None:
        self.layer = layer
        self.layer_index = int(getattr(layer, "layer_idx", -1))
        self.projections = tuple(self._require_projection(name) for name in PROJECTION_NAMES)
        self.split_sizes = tuple(int(projection.weight.shape[0]) for projection in self.projections)
        self.input_features = int(self.projections[0].weight.shape[1])
        self.original_weight_bytes = sum(
            projection.weight.numel() * projection.weight.element_size()
            for projection in self.projections
        )
        self._validate_projection_contract()

        with torch.no_grad():
            packed_weight = torch.cat(
                [projection.weight.detach() for projection in self.projections],
                dim=0,
            ).contiguous()
        self.packed_weight = packed_weight
        self._bind_weight_views()
        self._state = threading.local()
        self.fused_decode_calls = 0
        self.baseline_calls = 0

    @property
    def packed_weight_bytes(self) -> int:
        return self.packed_weight.numel() * self.packed_weight.element_size()

    def _require_projection(self, name: str) -> nn.Linear:
        projection = getattr(self.layer, name, None)
        if not isinstance(projection, nn.Linear):
            raise TypeError(f"GDN layer {self.layer_index} requires nn.Linear {name}.")
        return projection

    def _validate_projection_contract(self) -> None:
        weights = [projection.weight for projection in self.projections]
        if any(projection.bias is not None for projection in self.projections):
            raise ValueError("GDN decode projection fusion requires bias-free projections.")
        if any(weight.ndim != 2 for weight in weights):
            raise ValueError("GDN projection weights must be two-dimensional.")
        if any(weight.shape[1] != self.input_features for weight in weights):
            raise ValueError("GDN projection input dimensions must match.")
        if any(weight.device != weights[0].device for weight in weights):
            raise ValueError("GDN projection weights must be on one device before installation.")
        if any(weight.dtype != weights[0].dtype for weight in weights):
            raise ValueError("GDN projection weights must use one dtype before installation.")
        if len({weight.untyped_storage().data_ptr() for weight in weights}) != len(weights):
            raise ValueError("GDN projection weights unexpectedly share storage before packing.")

    def _bind_weight_views(self) -> None:
        offset = 0
        for projection, output_features in zip(
            self.projections,
            self.split_sizes,
            strict=True,
        ):
            view = self.packed_weight.narrow(0, offset, output_features)
            projection.weight = nn.Parameter(
                view,
                requires_grad=projection.weight.requires_grad,
            )
            def reject_apply(this, function, recurse=True):
                raise RuntimeError(
                    "Packed GDN projections cannot move device or dtype after installation."
                )

            projection._apply = types.MethodType(reject_apply, projection)
            offset += output_features
        self._validate_packed_storage()

    def _validate_packed_storage(self) -> None:
        packed_storage = self.packed_weight.untyped_storage()
        packed_pointer = packed_storage.data_ptr()
        for projection in self.projections:
            weight = projection.weight
            if weight.untyped_storage().data_ptr() != packed_pointer:
                raise RuntimeError(
                    "Packed GDN projection storage was invalidated. Install fusion only after "
                    "the model reaches its final device and dtype."
                )
        if packed_storage.nbytes() != self.packed_weight_bytes:
            raise RuntimeError("Packed GDN projection storage contains unexpected padding.")

    def _validate_hidden_states(self, hidden_states: torch.Tensor) -> None:
        if hidden_states.ndim != 3:
            raise ValueError("GDN hidden_states must have shape [batch, sequence, hidden].")
        if hidden_states.shape[-1] != self.input_features:
            raise ValueError("GDN hidden size does not match the packed projection input size.")
        if hidden_states.shape[1] == 1 and hidden_states.shape[0] != 1:
            raise ValueError("GDN decode projection fusion currently requires batch size 1.")

    def _clear_state(self) -> None:
        self._state.next_projection = 0
        self._state.outputs = None

    def project(self, projection_index: int, hidden_states: torch.Tensor) -> torch.Tensor:
        outputs = getattr(self._state, "outputs", None)
        if projection_index == 0:
            self._validate_hidden_states(hidden_states)
            if outputs is not None:
                raise RuntimeError("Previous fused GDN projection outputs were not fully consumed.")
            if hidden_states.shape[1] != 1:
                self.baseline_calls += 1
                return F.linear(hidden_states, self.projections[0].weight)
            packed_output = F.linear(hidden_states, self.packed_weight)
            self._state.outputs = torch.split(packed_output, self.split_sizes, dim=-1)
            self._state.next_projection = 1
            return self._state.outputs[0]
        if outputs is None:
            return F.linear(hidden_states, self.projections[projection_index].weight)
        if projection_index != self._state.next_projection:
            raise RuntimeError(
                "Unexpected GDN projection order: "
                f"expected {PROJECTION_NAMES[self._state.next_projection]}, "
                f"received {PROJECTION_NAMES[projection_index]}."
            )
        output = outputs[projection_index]
        self._state.next_projection += 1
        if self._state.next_projection == len(PROJECTION_NAMES):
            self.fused_decode_calls += 1
            self._clear_state()
        return output


def _install_layer(layer: nn.Module) -> _LayerProjectionFusion:
    if hasattr(layer, INSTALLATION_ATTRIBUTE):
        raise RuntimeError(f"GDN layer {getattr(layer, 'layer_idx', -1)} is already fused.")
    fusion = _LayerProjectionFusion(layer)
    for projection_index, projection in enumerate(fusion.projections):
        def projection_forward(this, hidden_states, __index=projection_index):
            return fusion.project(__index, hidden_states)

        projection.forward = types.MethodType(projection_forward, projection)
    setattr(layer, INSTALLATION_ATTRIBUTE, fusion)
    return fusion


def install_qwen35_gdn_decode_projection_fusion(
    model: nn.Module,
) -> GdnDecodeProjectionFusionReport:
    """Pack and fuse Qwen3.5 GDN input projections for batch-1 cached decode."""

    if model.training:
        raise ValueError("GDN decode projection fusion is inference-only; call eval() first.")
    layers = [
        module
        for module in model.modules()
        if type(module).__name__ == "Qwen3_5GatedDeltaNet"
    ]
    if not layers:
        raise ValueError("Model does not contain Qwen3_5GatedDeltaNet layers.")
    if any(hasattr(layer, INSTALLATION_ATTRIBUTE) for layer in layers):
        raise RuntimeError("GDN decode projection fusion is already installed.")

    fusions = [_install_layer(layer) for layer in layers]
    original_weight_bytes = sum(fusion.original_weight_bytes for fusion in fusions)
    packed_weight_bytes = sum(fusion.packed_weight_bytes for fusion in fusions)
    if packed_weight_bytes != original_weight_bytes:
        raise RuntimeError("Packed GDN weights changed the steady-state weight byte count.")
    return GdnDecodeProjectionFusionReport(
        layer_count=len(fusions),
        projection_count=len(fusions) * len(PROJECTION_NAMES),
        original_weight_bytes=original_weight_bytes,
        packed_weight_bytes=packed_weight_bytes,
        extra_steady_state_weight_bytes=packed_weight_bytes - original_weight_bytes,
    )


def get_qwen35_gdn_decode_projection_runtime(model: nn.Module) -> dict[str, Any] | None:
    fusions = [
        getattr(module, INSTALLATION_ATTRIBUTE)
        for module in model.modules()
        if hasattr(module, INSTALLATION_ATTRIBUTE)
    ]
    if not fusions:
        return None
    unique_storage_bytes = 0
    storage_pointers = set()
    for fusion in fusions:
        fusion._validate_packed_storage()
        pointer = fusion.packed_weight.untyped_storage().data_ptr()
        if pointer not in storage_pointers:
            storage_pointers.add(pointer)
            unique_storage_bytes += fusion.packed_weight.untyped_storage().nbytes()
    return {
        "enabled": True,
        "layer_count": len(fusions),
        "fused_decode_calls": sum(fusion.fused_decode_calls for fusion in fusions),
        "baseline_calls": sum(fusion.baseline_calls for fusion in fusions),
        "unique_packed_storage_bytes": unique_storage_bytes,
        "report": asdict(
            GdnDecodeProjectionFusionReport(
                layer_count=len(fusions),
                projection_count=len(fusions) * len(PROJECTION_NAMES),
                original_weight_bytes=sum(fusion.original_weight_bytes for fusion in fusions),
                packed_weight_bytes=sum(fusion.packed_weight_bytes for fusion in fusions),
                extra_steady_state_weight_bytes=0,
            )
        ),
    }
