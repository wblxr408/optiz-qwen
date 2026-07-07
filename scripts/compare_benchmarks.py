#!/usr/bin/env python3
"""Compare DNDX benchmark JSON outputs and render a compact PNG chart."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.patches as mpatches

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINES = {
    "cn": REPO_ROOT / "benchmarks/output/result_dev_cn_20_mps.json",
    "en": REPO_ROOT / "benchmarks/output/result_dev_en_20_mps.json",
}
DEFAULT_DATASET_ORDER = ("cn", "en")


@dataclass(frozen=True)
class Benchmark:
    dataset: str
    path: Path
    sample_count: int
    accuracy: float
    correct: int
    total: int
    avg_ttft_ms: float | None
    throughput: float
    validation_failed: int
    elapsed_seconds: float
    avg_seconds_per_sample: float
    answer_order: list[str]
    answers: dict[str, dict]


def parse_result_spec(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            f"Expected DATASET=PATH, got {raw!r}"
        )
    dataset, path = raw.split("=", 1)
    dataset = dataset.strip()
    if not dataset:
        raise argparse.ArgumentTypeError("Dataset label cannot be empty")
    return dataset, Path(path).expanduser()


def load_benchmark(dataset: str, path: Path) -> Benchmark:
    payload = json.loads(path.read_text(encoding="utf-8"))
    answer_rows = payload.get("answers", [])
    answers = {
        str(row["question_id"]): row
        for row in answer_rows
    }
    return Benchmark(
        dataset=dataset,
        path=path,
        sample_count=int(payload["sample_count"]),
        accuracy=float(payload["accuracy"]["score"]),
        correct=int(payload["accuracy"]["correct"]),
        total=int(payload["accuracy"]["total"]),
        avg_ttft_ms=payload["performance"].get("avg_ttft_ms"),
        throughput=float(payload["performance"]["avg_throughput_tokens_per_sec"]),
        validation_failed=int(payload["public_validation"]["failed_samples"]),
        elapsed_seconds=float(payload["timing"]["benchmark_elapsed_seconds"]),
        avg_seconds_per_sample=float(payload["timing"]["avg_seconds_per_sample"]),
        answer_order=[str(row["question_id"]) for row in answer_rows],
        answers=answers,
    )


def load_group(specs: Iterable[tuple[str, Path]]) -> dict[str, Benchmark]:
    group: dict[str, Benchmark] = {}
    for dataset, path in specs:
        if dataset in group:
            raise ValueError(f"Duplicate dataset label: {dataset}")
        group[dataset] = load_benchmark(dataset, path)
    return group


def default_baseline_specs(
    candidate_specs: Iterable[tuple[str, Path]],
) -> list[tuple[str, Path]]:
    candidate_labels = {dataset for dataset, _ in candidate_specs}
    missing_defaults = sorted(candidate_labels - set(DEFAULT_BASELINES))
    if missing_defaults:
        raise ValueError(
            "No hard-coded baseline path for dataset labels: "
            f"{missing_defaults}. Please pass --baseline DATASET=PATH."
        )

    specs = [
        (dataset, DEFAULT_BASELINES[dataset])
        for dataset in DEFAULT_DATASET_ORDER
        if dataset in candidate_labels
    ]
    missing_files = [str(path) for _, path in specs if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Default baseline file does not exist: "
            f"{missing_files}. Please run the baseline or pass --baseline."
        )
    return specs


def fmt_delta(value: float, suffix: str = "") -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.3f}{suffix}"


def answer_sets(benchmark: Benchmark) -> tuple[set[str], set[str]]:
    wrong = {
        qid
        for qid, row in benchmark.answers.items()
        if not bool(row.get("correct"))
    }
    invalid = {
        qid
        for qid, row in benchmark.answers.items()
        if row.get("validation_errors")
    }
    return wrong, invalid


def print_comparison(
    baseline: dict[str, Benchmark],
    candidate: dict[str, Benchmark],
    baseline_name: str,
    candidate_name: str,
) -> None:
    datasets = sorted(set(baseline) & set(candidate))
    missing_baseline = sorted(set(candidate) - set(baseline))
    missing_candidate = sorted(set(baseline) - set(candidate))
    if missing_baseline or missing_candidate:
        raise ValueError(
            "Dataset labels must match. "
            f"Missing in baseline: {missing_baseline}; "
            f"missing in candidate: {missing_candidate}"
        )

    print("\nBenchmark comparison")
    print(f"baseline:  {baseline_name}")
    print(f"candidate: {candidate_name}\n")
    for dataset in datasets:
        before = baseline[dataset]
        after = candidate[dataset]
        ttft_before = before.avg_ttft_ms or 0.0
        ttft_after = after.avg_ttft_ms or 0.0
        rows = (
            ("samples", before.sample_count, after.sample_count, after.sample_count - before.sample_count),
            ("accuracy", f"{before.accuracy:.3f}", f"{after.accuracy:.3f}", fmt_delta(after.accuracy - before.accuracy)),
            ("ttft_ms", f"{ttft_before:.1f}", f"{ttft_after:.1f}", fmt_delta(ttft_after - ttft_before)),
            ("tok/s", f"{before.throughput:.2f}", f"{after.throughput:.2f}", fmt_delta(after.throughput - before.throughput)),
            ("invalid", before.validation_failed, after.validation_failed, fmt_delta(after.validation_failed - before.validation_failed)),
        )
        header = ("metric", baseline_name, candidate_name, "delta")
        widths = (12, 14, 14, 12)
        print(f"[{dataset}]")
        print("".join(str(cell).ljust(width) for cell, width in zip(header, widths)))
        print("-" * sum(widths))
        for row in rows:
            print("".join(str(cell).ljust(width) for cell, width in zip(row, widths)))
        print()

    print("\nPer-sample changes")
    for dataset in datasets:
        before_wrong, before_invalid = answer_sets(baseline[dataset])
        after_wrong, after_invalid = answer_sets(candidate[dataset])
        fixed = sorted(before_wrong - after_wrong)
        regressed = sorted(after_wrong - before_wrong)
        validation_fixed = sorted(before_invalid - after_invalid)
        validation_regressed = sorted(after_invalid - before_invalid)
        print(f"- {dataset}:")
        print(f"  fixed wrong answers: {fixed or 'none'}")
        print(f"  new wrong answers: {regressed or 'none'}")
        print(f"  fixed validation errors: {validation_fixed or 'none'}")
        print(f"  new validation errors: {validation_regressed or 'none'}")


def render_plot(
    baseline: dict[str, Benchmark],
    candidate: dict[str, Benchmark],
    baseline_name: str,
    candidate_name: str,
    output: Path,
) -> None:
    datasets = sorted(baseline)
    status_colors = {
        "both_correct": "#4c78a8",
        "fixed": "#54a24b",
        "regressed": "#e45756",
        "both_wrong": "#f58518",
    }
    fig, axes = plt.subplots(
        4,
        len(datasets),
        figsize=(max(10, 6.5 * len(datasets)), 10),
        squeeze=False,
    )

    for col_index, dataset in enumerate(datasets):
        before = baseline[dataset]
        after = candidate[dataset]
        status_ax = axes[0][col_index]
        ttft_ax = axes[1][col_index]
        throughput_ax = axes[2][col_index]
        token_ax = axes[3][col_index]

        qids = [qid for qid in after.answer_order if qid in before.answers]
        qids.extend(qid for qid in before.answer_order if qid in after.answers and qid not in qids)
        x = list(range(len(qids)))
        labels = qids

        statuses: list[str] = []
        ttft_delta: list[float] = []
        throughput_delta: list[float] = []
        token_delta: list[int] = []
        invalid_after_x: list[int] = []
        invalid_before_x: list[int] = []

        for idx, qid in enumerate(qids):
            before_row = before.answers[qid]
            after_row = after.answers[qid]
            before_correct = bool(before_row.get("correct"))
            after_correct = bool(after_row.get("correct"))
            if before_correct and after_correct:
                statuses.append("both_correct")
            elif not before_correct and after_correct:
                statuses.append("fixed")
            elif before_correct and not after_correct:
                statuses.append("regressed")
            else:
                statuses.append("both_wrong")

            ttft_delta.append(float(after_row.get("ttft_ms", 0.0)) - float(before_row.get("ttft_ms", 0.0)))
            throughput_delta.append(
                float(after_row.get("throughput_tokens_per_sec", 0.0))
                - float(before_row.get("throughput_tokens_per_sec", 0.0))
            )
            token_delta.append(int(after_row.get("token_count", 0)) - int(before_row.get("token_count", 0)))
            if before_row.get("validation_errors"):
                invalid_before_x.append(idx)
            if after_row.get("validation_errors"):
                invalid_after_x.append(idx)

        status_ax.bar(x, [1] * len(x), color=[status_colors[s] for s in statuses], width=0.9)
        if invalid_before_x:
            status_ax.scatter(invalid_before_x, [0.25] * len(invalid_before_x), color="black", marker="x", s=28, label=f"{baseline_name} invalid")
        if invalid_after_x:
            status_ax.scatter(invalid_after_x, [0.75] * len(invalid_after_x), color="black", marker="o", s=18, label=f"{candidate_name} invalid")
        status_ax.set_title(
            f"{dataset}: answer status "
            f"({baseline_name} acc {before.accuracy:.3f}, {candidate_name} acc {after.accuracy:.3f})"
        )
        status_ax.set_yticks([])

        ttft_colors = ["#54a24b" if value < 0 else "#e45756" if value > 0 else "#9d9d9d" for value in ttft_delta]
        ttft_ax.bar(x, ttft_delta, color=ttft_colors, width=0.85)
        ttft_ax.axhline(0, color="black", linewidth=0.8)
        ttft_ax.set_title("TTFT delta ms\nnegative is faster")
        ttft_ax.grid(axis="y", alpha=0.25)

        throughput_colors = ["#54a24b" if value > 0 else "#e45756" if value < 0 else "#9d9d9d" for value in throughput_delta]
        throughput_ax.bar(x, throughput_delta, color=throughput_colors, width=0.85)
        throughput_ax.axhline(0, color="black", linewidth=0.8)
        throughput_ax.set_title("Throughput delta tok/s\npositive is better")
        throughput_ax.grid(axis="y", alpha=0.25)

        token_colors = ["#4c78a8" if value <= 0 else "#f58518" for value in token_delta]
        token_ax.bar(x, token_delta, color=token_colors, width=0.85)
        token_ax.axhline(0, color="black", linewidth=0.8)
        token_ax.set_title("Generated token delta\nlower often helps speed")
        token_ax.grid(axis="y", alpha=0.25)

        for ax in (status_ax, ttft_ax, throughput_ax, token_ax):
            ax.set_xticks(x, labels, rotation=90 if len(labels) > 12 else 45, fontsize=7)
            ax.set_xlabel("question_id")

    legend_handles = [
        mpatches.Patch(color=status_colors["both_correct"], label="both correct"),
        mpatches.Patch(color=status_colors["fixed"], label="fixed"),
        mpatches.Patch(color=status_colors["regressed"], label="regressed"),
        mpatches.Patch(color=status_colors["both_wrong"], label="both wrong"),
        plt.Line2D([0], [0], marker="x", color="black", linestyle="", label=f"{baseline_name} validation error"),
        plt.Line2D([0], [0], marker="o", color="black", linestyle="", label=f"{candidate_name} validation error"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=6,
        fontsize=9,
    )
    fig.suptitle("Per-sample Benchmark Comparison", fontsize=14, y=0.955)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare DNDX benchmark JSON files and generate a PNG chart."
    )
    parser.add_argument(
        "--baseline",
        action="append",
        type=parse_result_spec,
        help=(
            "Baseline result in DATASET=PATH format. Repeat for multiple datasets. "
            "If omitted, the script uses the hard-coded local baseline matching "
            "each candidate dataset label."
        ),
    )
    parser.add_argument(
        "--candidate",
        action="append",
        type=parse_result_spec,
        required=True,
        help="Candidate result in DATASET=PATH format. Repeat for multiple datasets.",
    )
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("benchmarks/output/benchmark_comparison.png"),
        help="PNG output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_specs = args.baseline or default_baseline_specs(args.candidate)
    baseline = load_group(baseline_specs)
    candidate = load_group(args.candidate)
    print_comparison(
        baseline,
        candidate,
        args.baseline_name,
        args.candidate_name,
    )
    render_plot(
        baseline,
        candidate,
        args.baseline_name,
        args.candidate_name,
        args.plot,
    )
    print(f"\nSaved plot: {args.plot}")


if __name__ == "__main__":
    main()
