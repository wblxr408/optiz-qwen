"""Plot accuracy status and paired TTFT deltas for a ToMe experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-throughput", action="store_true")
    parser.add_argument("--include-token-count", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    baseline_rows = {str(row["question_id"]): row for row in baseline["answers"]}
    candidate_rows = {str(row["question_id"]): row for row in candidate["answers"]}
    question_ids = [str(row["question_id"]) for row in baseline["answers"]]
    if question_ids != [str(row["question_id"]) for row in candidate["answers"]]:
        raise ValueError("Baseline and candidate question order must match.")

    status_colors = {
        (True, True): "#4C78A8",
        (False, False): "#F58518",
        (False, True): "#54A24B",
        (True, False): "#E45756",
    }
    statuses = [
        status_colors[(bool(baseline_rows[qid]["correct"]), bool(candidate_rows[qid]["correct"]))]
        for qid in question_ids
    ]
    deltas = [
        float(candidate_rows[qid]["ttft_ms"]) - float(baseline_rows[qid]["ttft_ms"])
        for qid in question_ids
    ]
    delta_colors = ["#54A24B" if value < 0 else "#E45756" for value in deltas]
    throughput_deltas = [
        float(candidate_rows[qid]["throughput_tokens_per_sec"])
        - float(baseline_rows[qid]["throughput_tokens_per_sec"])
        for qid in question_ids
    ]
    token_deltas = [
        int(candidate_rows[qid]["token_count"]) - int(baseline_rows[qid]["token_count"])
        for qid in question_ids
    ]
    baseline_ttft = float(baseline["performance"]["avg_ttft_ms"])
    candidate_ttft = float(candidate["performance"]["avg_ttft_ms"])
    speedup = (baseline_ttft - candidate_ttft) / baseline_ttft * 100
    speed_label = f"{speedup:.2f}% faster" if speedup >= 0 else f"{-speedup:.2f}% slower"

    x = range(len(question_ids))
    row_count = 2 + int(args.include_throughput) + int(args.include_token_count)
    height_ratios = [1, 2]
    if args.include_throughput:
        height_ratios.append(2)
    if args.include_token_count:
        height_ratios.append(1.5)
    fig, axes = plt.subplots(
        row_count,
        1,
        figsize=(12, 6.5 + 2 * (row_count - 2)),
        gridspec_kw={"height_ratios": height_ratios},
    )
    fig.patch.set_facecolor("white")
    status_ax, ttft_ax = axes[:2]
    status_ax.bar(x, [1] * len(question_ids), color=statuses, width=0.88)
    status_ax.set_yticks([])
    status_ax.set_title(
        f"Answer status: baseline {baseline['accuracy']['correct']}/{len(question_ids)}, "
        f"ToMe {candidate['accuracy']['correct']}/{len(question_ids)}"
    )
    status_ax.spines[["left", "right", "top"]].set_visible(False)
    status_ax.legend(
        handles=[
            mpatches.Patch(color="#4C78A8", label="both correct"),
            mpatches.Patch(color="#F58518", label="both wrong"),
            mpatches.Patch(color="#54A24B", label="fixed"),
            mpatches.Patch(color="#E45756", label="regressed"),
        ],
        ncol=4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.45),
    )

    ttft_ax.bar(x, deltas, color=delta_colors, width=0.75)
    ttft_ax.axhline(0, color="#333333", linewidth=1)
    ttft_ax.set_title(
        f"Paired TTFT delta: {baseline_ttft:.1f} -> {candidate_ttft:.1f} ms "
        f"({speed_label})"
    )
    ttft_ax.set_ylabel("ToMe - baseline (ms)\nnegative is faster")
    ttft_ax.grid(axis="y", alpha=0.25)

    plot_axes = [status_ax, ttft_ax]
    next_axis = 2
    if args.include_throughput:
        throughput_ax = axes[next_axis]
        next_axis += 1
        throughput_colors = ["#54A24B" if value > 0 else "#E45756" for value in throughput_deltas]
        throughput_ax.bar(x, throughput_deltas, color=throughput_colors, width=0.75)
        throughput_ax.axhline(0, color="#333333", linewidth=1)
        throughput_ax.set_title(
            "Paired decode throughput delta: "
            f"{baseline['performance']['avg_throughput_tokens_per_sec']:.3f} -> "
            f"{candidate['performance']['avg_throughput_tokens_per_sec']:.3f} tok/s"
        )
        throughput_ax.set_ylabel("ToMe - baseline (tok/s)\npositive is faster")
        throughput_ax.grid(axis="y", alpha=0.25)
        plot_axes.append(throughput_ax)

    if args.include_token_count:
        token_ax = axes[next_axis]
        token_colors = ["#54A24B" if value <= 0 else "#E45756" for value in token_deltas]
        token_ax.bar(x, token_deltas, color=token_colors, width=0.75)
        token_ax.axhline(0, color="#333333", linewidth=1)
        token_ax.set_title(f"Generated token delta (sum: {sum(token_deltas):+d})")
        token_ax.set_ylabel("ToMe - baseline")
        token_ax.grid(axis="y", alpha=0.25)
        plot_axes.append(token_ax)

    for axis in plot_axes:
        axis.set_xticks(list(x), question_ids, rotation=45, ha="right", fontsize=8)
        axis.set_xlabel("question_id")

    fig.suptitle("Qwen3.5-2B ToMe L12 R32, MMBench EN20 on MPS", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
