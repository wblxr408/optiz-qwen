"""Token Merging primitives adapted to Qwen3.5 visual merge units."""

from __future__ import annotations

from dataclasses import dataclass
import time

import torch


@dataclass(frozen=True)
class TomeMergeResult:
    hidden_states: torch.Tensor
    token_sizes: torch.Tensor
    cu_seqlens: torch.Tensor
    retained_token_indices: torch.Tensor
    source_unit_indices: torch.Tensor
    destination_unit_indices: torch.Tensor
    timings_ms: dict[str, float] | None = None


def merge_visual_units(
    hidden_states: torch.Tensor,
    metric: torch.Tensor,
    token_sizes: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    r: int,
    unit_size: int = 4,
) -> TomeMergeResult:
    """Merge ``r`` similar visual units per packed sample.

    Each unit contains the patches consumed together by Qwen3.5's final
    PatchMerger. Matching follows ToMe's alternating bipartite partition, while
    retained units stay in their original spatial order.
    """

    _validate_tensor_shapes(hidden_states, metric, token_sizes)
    boundaries = _validated_packed_boundaries(cu_seqlens, hidden_states.shape[0], unit_size)
    return _merge_visual_units(
        hidden_states,
        metric,
        token_sizes,
        cu_seqlens,
        boundaries=boundaries,
        r=r,
        unit_size=unit_size,
    )


def merge_single_visual_sample(
    hidden_states: torch.Tensor,
    metric: torch.Tensor,
    token_sizes: torch.Tensor,
    *,
    r: int,
    unit_size: int = 4,
    profile: bool = False,
) -> TomeMergeResult:
    """Merge one visual sample without reading sequence metadata from its device."""

    _validate_tensor_shapes(hidden_states, metric, token_sizes)
    if unit_size <= 0:
        raise ValueError("unit_size must be positive.")
    if hidden_states.shape[0] % unit_size != 0:
        raise ValueError("the visual sample length must be divisible by unit_size.")
    if r < 0:
        raise ValueError("r must be non-negative.")
    cu_seqlens = torch.tensor(
        [0, hidden_states.shape[0]],
        dtype=torch.int32,
        device=hidden_states.device,
    )
    return _merge_visual_units(
        hidden_states,
        metric,
        token_sizes,
        cu_seqlens,
        boundaries=[0, hidden_states.shape[0]],
        r=r,
        unit_size=unit_size,
        profile=profile,
    )


def _merge_visual_units(
    hidden_states: torch.Tensor,
    metric: torch.Tensor,
    token_sizes: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    boundaries: list[int],
    r: int,
    unit_size: int,
    profile: bool = False,
) -> TomeMergeResult:
    if r < 0:
        raise ValueError("r must be non-negative.")

    output_states: list[torch.Tensor] = []
    output_sizes: list[torch.Tensor] = []
    retained_indices: list[torch.Tensor] = []
    source_indices: list[torch.Tensor] = []
    destination_indices: list[torch.Tensor] = []
    output_lengths = [0]
    timings_ms: dict[str, float] | None = {} if profile else None

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        sample_states = hidden_states[start:end]
        sample_metric = metric[start:end]
        sample_sizes = token_sizes[start:end]
        unit_count = (end - start) // unit_size

        if r == 0:
            sample_indices = torch.arange(start, end, device=hidden_states.device)
            output_states.append(sample_states)
            output_sizes.append(sample_sizes)
            retained_indices.append(sample_indices)
            source_indices.append(torch.empty(0, dtype=torch.long, device=hidden_states.device))
            destination_indices.append(torch.empty(0, dtype=torch.long, device=hidden_states.device))
            output_lengths.append(output_lengths[-1] + end - start)
            continue

        if r > min((unit_count + 1) // 2, unit_count // 2):
            raise ValueError(
                f"r={r} exceeds the bipartite matching capacity for a sample "
                f"with {unit_count} visual units."
            )

        _synchronize(hidden_states.device, profile)
        stage_started = time.perf_counter()
        units = sample_states.reshape(unit_count, unit_size, -1)
        sizes = sample_sizes.reshape(unit_count, unit_size, 1)
        unit_metric = sample_metric.reshape(unit_count, unit_size, -1).mean(dim=1)
        unit_metric = torch.nn.functional.normalize(unit_metric, dim=-1)
        _record_stage(timings_ms, "metric_preparation", stage_started, hidden_states.device)

        _synchronize(hidden_states.device, profile)
        stage_started = time.perf_counter()
        even_units = torch.arange(0, unit_count, 2, device=hidden_states.device)
        odd_units = torch.arange(1, unit_count, 2, device=hidden_states.device)
        scores = unit_metric[even_units] @ unit_metric[odd_units].transpose(0, 1)
        best_scores, best_destinations = scores.max(dim=1)
        selected_even = best_scores.topk(r, largest=True, sorted=True).indices
        sources = even_units[selected_even]
        destinations = odd_units[best_destinations[selected_even]]
        _record_stage(timings_ms, "bipartite_matching", stage_started, hidden_states.device)

        _synchronize(hidden_states.device, profile)
        stage_started = time.perf_counter()
        weighted_units = units * sizes
        merged_weighted = weighted_units.index_add(0, destinations, weighted_units[sources])
        merged_sizes = sizes.index_add(0, destinations, sizes[sources])
        merged_units = merged_weighted / merged_sizes
        _record_stage(timings_ms, "weighted_aggregation", stage_started, hidden_states.device)

        _synchronize(hidden_states.device, profile)
        stage_started = time.perf_counter()
        keep_units = torch.ones(unit_count, dtype=torch.bool, device=hidden_states.device)
        keep_units[sources] = False
        kept_unit_indices = torch.arange(unit_count, device=hidden_states.device)[keep_units]
        sample_token_indices = (
            kept_unit_indices[:, None] * unit_size
            + torch.arange(unit_size, device=hidden_states.device)[None, :]
        ).reshape(-1)

        output_states.append(merged_units[keep_units].reshape(-1, hidden_states.shape[-1]))
        output_sizes.append(merged_sizes[keep_units].reshape(-1, 1))
        retained_indices.append(sample_token_indices + start)
        source_indices.append(sources + start // unit_size)
        destination_indices.append(destinations + start // unit_size)
        output_lengths.append(output_lengths[-1] + sample_token_indices.numel())
        _record_stage(timings_ms, "output_compaction", stage_started, hidden_states.device)

    _synchronize(hidden_states.device, profile)
    stage_started = time.perf_counter()
    return TomeMergeResult(
        hidden_states=torch.cat(output_states, dim=0),
        token_sizes=torch.cat(output_sizes, dim=0),
        cu_seqlens=cu_seqlens.new_tensor(output_lengths),
        retained_token_indices=torch.cat(retained_indices),
        source_unit_indices=torch.cat(source_indices),
        destination_unit_indices=torch.cat(destination_indices),
        timings_ms=_finish_result_timing(timings_ms, stage_started, hidden_states.device),
    )


def _record_stage(
    timings_ms: dict[str, float] | None,
    name: str,
    started: float,
    device: torch.device,
) -> None:
    if timings_ms is None:
        return
    _synchronize(device, True)
    timings_ms[name] = timings_ms.get(name, 0.0) + (time.perf_counter() - started) * 1000.0


def _finish_result_timing(
    timings_ms: dict[str, float] | None,
    started: float,
    device: torch.device,
) -> dict[str, float] | None:
    if timings_ms is None:
        return None
    _synchronize(device, True)
    timings_ms["result_assembly"] = (time.perf_counter() - started) * 1000.0
    timings_ms["total"] = sum(timings_ms.values())
    return timings_ms


def _synchronize(device: torch.device, enabled: bool) -> None:
    if not enabled:
        return
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _validate_tensor_shapes(
    hidden_states: torch.Tensor,
    metric: torch.Tensor,
    token_sizes: torch.Tensor,
) -> None:
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [tokens, hidden_size].")
    if metric.ndim != 2 or metric.shape[0] != hidden_states.shape[0]:
        raise ValueError("metric must have shape [tokens, metric_size].")
    if token_sizes.shape != (hidden_states.shape[0], 1):
        raise ValueError("token_sizes must have shape [tokens, 1].")
    if hidden_states.device != metric.device or hidden_states.device != token_sizes.device:
        raise ValueError("hidden_states, metric, and token_sizes must share a device.")
    if not hidden_states.is_floating_point() or not metric.is_floating_point():
        raise ValueError("hidden_states and metric must use floating-point dtypes.")
    if not token_sizes.is_floating_point():
        raise ValueError("token_sizes must use a floating-point dtype.")


def _validated_packed_boundaries(
    cu_seqlens: torch.Tensor,
    token_count: int,
    unit_size: int,
) -> list[int]:
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must contain at least one packed sample.")
    boundaries = cu_seqlens.detach().cpu().tolist()
    if boundaries[0] != 0 or boundaries[-1] != token_count:
        raise ValueError("cu_seqlens must span all input tokens.")
    if any(end <= start for start, end in zip(boundaries[:-1], boundaries[1:])):
        raise ValueError("cu_seqlens must be strictly increasing.")
    if unit_size <= 0:
        raise ValueError("unit_size must be positive.")
    if any((end - start) % unit_size != 0 for start, end in zip(boundaries[:-1], boundaries[1:])):
        raise ValueError("every packed sample length must be divisible by unit_size.")
    return boundaries
