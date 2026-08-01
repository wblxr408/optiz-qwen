"""Plot per-sample DToMe allocation and timing behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_result_spec(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"Expected LABEL=PATH, got {raw!r}.")
    label, path = raw.split("=", 1)
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="append", type=parse_result_spec, required=True)
    parser.add_argument("--candidate", action="append", type=parse_result_spec, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_paths = dict(args.baseline)
    candidate_paths = dict(args.candidate)
    if baseline_paths.keys() != candidate_paths.keys():
        raise ValueError("Baseline and candidate labels must match.")

    records = []
    for label in baseline_paths:
        baseline = json.loads(baseline_paths[label].read_text(encoding="utf-8"))
        candidate = json.loads(candidate_paths[label].read_text(encoding="utf-8"))
        if len(baseline["answers"]) != len(candidate["answers"]):
            raise ValueError(f"Answer count differs for {label}.")
        for before, after in zip(baseline["answers"], candidate["answers"]):
            if str(before["question_id"]) != str(after["question_id"]):
                raise ValueError(f"Question order differs for {label}.")
            runtime = after["meta"]["tome"]
            records.append(
                {
                    "label": label,
                    "question_id": str(after["question_id"]),
                    "input_tokens": runtime["input_tokens"],
                    "merged_units": runtime["merged_units"],
                    "ttft_delta_ms": after["ttft_ms"] - before["ttft_ms"],
                }
            )

    labels = list(baseline_paths)
    colors = dict(zip(labels, ("#4C78A8", "#F58518", "#54A24B")))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.patch.set_facecolor("white")

    allocation_ax, length_ax, timing_ax, histogram_ax = axes.flat
    for label in labels:
        indexed_subset = [
            (index, record)
            for index, record in enumerate(records)
            if record["label"] == label
        ]
        indices = [index for index, _record in indexed_subset]
        subset = [record for _index, record in indexed_subset]
        allocation_ax.scatter(
            indices,
            [record["merged_units"] for record in subset],
            color=colors[label],
            label=label,
            s=22,
            alpha=0.8,
        )
        length_ax.scatter(
            [record["input_tokens"] for record in subset],
            [record["merged_units"] for record in subset],
            color=colors[label],
            label=label,
            s=25,
            alpha=0.75,
        )
        timing_ax.scatter(
            [record["merged_units"] for record in subset],
            [record["ttft_delta_ms"] for record in subset],
            color=colors[label],
            label=label,
            s=25,
            alpha=0.75,
        )
        histogram_ax.hist(
            [record["merged_units"] for record in subset],
            bins=range(0, 58, 4),
            color=colors[label],
            label=label,
            alpha=0.5,
        )

    allocation_ax.axhline(32, color="#333333", linestyle="--", linewidth=1)
    allocation_ax.set_title("Per-sample dynamic merge allocation")
    allocation_ax.set_xlabel("sample index")
    allocation_ax.set_ylabel("merged visual units")

    length_ax.axhline(32, color="#333333", linestyle="--", linewidth=1)
    length_ax.set_title("Allocation after visual-length calibration")
    length_ax.set_xlabel("input visual tokens")
    length_ax.set_ylabel("merged visual units")

    timing_ax.axhline(0, color="#333333", linewidth=1)
    timing_ax.set_title("Merge allocation vs paired TTFT delta")
    timing_ax.set_xlabel("merged visual units")
    timing_ax.set_ylabel("DToMe - baseline (ms)")

    histogram_ax.axvline(32, color="#333333", linestyle="--", linewidth=1)
    histogram_ax.set_title("Dynamic merge-count distribution")
    histogram_ax.set_xlabel("merged visual units")
    histogram_ax.set_ylabel("sample count")

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors[label],
            markeredgecolor=colors[label],
            label=label,
        )
        for label in labels
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=len(labels),
        frameon=False,
    )
    fig.suptitle("Qwen3.5-2B DToMe Dynamic Allocation, 150 MMBench Samples", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
