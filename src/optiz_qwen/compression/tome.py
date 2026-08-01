"""Token Merging primitives adapted to Qwen3.5 visual merge units."""

from __future__ import annotations

from dataclasses import dataclass
import time

import torch

DToMeThreshold = float | tuple[tuple[int, float], ...]


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
    matching: str = "tome",
    threshold: DToMeThreshold | None = None,
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
        matching=matching,
        threshold=threshold,
    )


def merge_single_visual_sample(
    hidden_states: torch.Tensor,
    metric: torch.Tensor,
    token_sizes: torch.Tensor,
    *,
    r: int,
    unit_size: int = 4,
    matching: str = "tome",
    threshold: DToMeThreshold | None = None,
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
        matching=matching,
        threshold=threshold,
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
    matching: str,
    threshold: DToMeThreshold | None,
    profile: bool = False,
) -> TomeMergeResult:
    if r < 0:
        raise ValueError("r must be non-negative.")
    _validate_matching(matching, threshold)

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

        if matching != "dtome" and r > min((unit_count + 1) // 2, unit_count // 2):
            raise ValueError(
                f"r={r} exceeds the bipartite matching capacity for a sample "
                f"with {unit_count} visual units."
            )

        _synchronize(hidden_states.device, profile)
        stage_started = time.perf_counter()
        units = sample_states.reshape(unit_count, unit_size, -1)
        sizes = sample_sizes.reshape(unit_count, unit_size, 1)
        unit_metric = sample_metric.reshape(unit_count, unit_size, -1).mean(dim=1)
        if matching == "dtome":
            unit_metric = unit_metric.float()
        unit_metric = torch.nn.functional.normalize(unit_metric, dim=-1)
        _record_stage(timings_ms, "metric_preparation", stage_started, hidden_states.device)

        _synchronize(hidden_states.device, profile)
        stage_started = time.perf_counter()
        sources, destinations = _select_merge_pairs(
            unit_metric,
            r,
            matching,
            _resolve_dtome_threshold(threshold, (unit_count + 1) // 2)
            if matching == "dtome"
            else threshold,
        )
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


def _select_merge_pairs(
    unit_metric: torch.Tensor,
    r: int,
    matching: str,
    threshold: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if matching in {"tome", "dtome"}:
        even_units = torch.arange(0, unit_metric.shape[0], 2, device=unit_metric.device)
        odd_units = torch.arange(1, unit_metric.shape[0], 2, device=unit_metric.device)
        scores = unit_metric[even_units] @ unit_metric[odd_units].transpose(0, 1)
        best_scores, best_destinations = scores.max(dim=1)
        if matching == "dtome":
            selected_even = torch.nonzero(best_scores > threshold, as_tuple=False).flatten()
            if selected_even.numel() == 0:
                empty = torch.empty(0, dtype=torch.long, device=unit_metric.device)
                return empty, empty
            selected_even = selected_even[
                best_scores[selected_even].argsort(descending=True)
            ]
            return even_units[selected_even], odd_units[best_destinations[selected_even]]
        selected_even = best_scores.topk(r, largest=True, sorted=True).indices
        return even_units[selected_even], odd_units[best_destinations[selected_even]]

    similarities = unit_metric @ unit_metric.transpose(0, 1)
    energy = torch.nn.functional.elu(similarities - 0.5).mean(dim=-1)
    mergeable = energy.argsort(descending=True)[: 2 * r]
    sources = mergeable[::2]
    destination_candidates = mergeable[1::2]
    scores = similarities[sources][:, destination_candidates]
    destinations = destination_candidates[scores.argmax(dim=-1)]
    return sources, destinations


def visual_unit_matching_scores(
    metric: torch.Tensor,
    *,
    unit_size: int = 4,
) -> torch.Tensor:
    """Return each source unit's best ToMe edge score for threshold calibration."""

    if metric.ndim != 2 or not metric.is_floating_point():
        raise ValueError("metric must be a floating-point tensor with shape [tokens, features].")
    if unit_size <= 0 or metric.shape[0] % unit_size != 0:
        raise ValueError("metric length must be divisible by a positive unit_size.")
    unit_count = metric.shape[0] // unit_size
    if unit_count < 2:
        raise ValueError("threshold calibration requires at least two visual units.")
    unit_metric = metric.float().reshape(unit_count, unit_size, -1).mean(dim=1)
    unit_metric = torch.nn.functional.normalize(unit_metric, dim=-1)
    even_units = unit_metric[::2]
    odd_units = unit_metric[1::2]
    return (even_units @ odd_units.transpose(0, 1)).max(dim=-1).values


def _validate_matching(matching: str, threshold: DToMeThreshold | None) -> None:
    if matching not in {"tome", "pitome", "dtome"}:
        raise ValueError("matching must be 'tome', 'pitome', or 'dtome'.")
    if matching == "dtome":
        if threshold is None:
            raise ValueError("dtome matching requires a threshold.")
        if isinstance(threshold, tuple):
            if not threshold:
                raise ValueError("dtome threshold schedule must not be empty.")
            previous_limit = 0
            for source_limit, value in threshold:
                if source_limit <= previous_limit or not -1.0 <= value <= 1.0:
                    raise ValueError(
                        "dtome threshold schedule requires increasing positive limits "
                        "and thresholds in [-1, 1]."
                    )
                previous_limit = source_limit
        elif not -1.0 <= threshold <= 1.0:
            raise ValueError("dtome matching requires a threshold in [-1, 1].")
    elif threshold is not None:
        raise ValueError("threshold is only valid with dtome matching.")


def _resolve_dtome_threshold(
    threshold: DToMeThreshold | None,
    source_count: int,
) -> float:
    if threshold is None:
        raise RuntimeError("validated DToMe threshold is missing.")
    if isinstance(threshold, float):
        return threshold
    for source_limit, value in threshold:
        if source_count <= source_limit:
            return value
    return threshold[-1][1]


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
