"""Run baseline and ToMe alternately on one loaded Qwen3.5 model."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from optiz_qwen.compression import Qwen35TomeConfig, install_qwen35_tome, set_qwen35_tome_enabled
from optiz_qwen.evaluation.answer_parsing import parse_choice_answer
from optiz_qwen.evaluation.dndx_public_benchmark import (
    build_prompt,
    compute_throughput,
    decode_image,
    fixed_generation_config,
    load_mmbench_tsv,
    settle_runtime,
    validate_public_result,
)
from optiz_qwen.evaluation.dndx_wrapper import VLMModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        default="resources/eval_dataset/raw/mmbench_public/mmbench_dev_en.tsv",
    )
    parser.add_argument("--model-path", default="resources/model_weights/raw/Qwen3.5-2B")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--warmup-samples", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--r", type=int, default=32)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def run_one(model: VLMModel, sample, max_new_tokens: int, tome_enabled: bool) -> dict:
    set_qwen35_tome_enabled(model._model, tome_enabled)
    settle_runtime(model)
    config = fixed_generation_config(max_new_tokens)
    result = model.generate_with_metrics(
        image=decode_image(sample.image_b64),
        prompt=build_prompt(sample),
        choices=sample.choices,
        generation_config=config,
        sample_id=sample.sample_id,
    )
    parsed_answer, answer_source = parse_choice_answer(result.text, sample.choices)
    throughput = compute_throughput(result.token_count, result.ttft_seconds, result.elapsed_seconds)
    return {
        "question_id": sample.sample_id,
        "text": result.text,
        "parsed_answer": parsed_answer,
        "answer_source": answer_source,
        "correct": parsed_answer == sample.answer,
        "ttft_ms": result.ttft_seconds * 1000.0,
        "throughput_tokens_per_sec": throughput,
        "token_count": result.token_count,
        "elapsed_seconds": result.elapsed_seconds,
        "validation_errors": validate_public_result(
            result.text,
            parsed_answer,
            result.token_count,
            config.max_new_tokens,
        ),
        "meta": result.meta,
    }


def build_payload(mode: str, rows: list[dict], args: argparse.Namespace) -> dict:
    valid_ttft = [row["ttft_ms"] for row in rows if math.isfinite(row["ttft_ms"])]
    valid_throughput = [
        row["throughput_tokens_per_sec"]
        for row in rows
        if math.isfinite(row["throughput_tokens_per_sec"])
    ]
    correct = sum(bool(row["correct"]) for row in rows)
    invalid = sum(bool(row["validation_errors"]) for row in rows)
    return {
        "benchmark_version": "tome_paired_v1",
        "timestamp": datetime.now().isoformat(),
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "sample_count": len(rows),
        "backend": "transformers",
        "optimization": {
            "mode": mode,
            "paired_alternating_order": True,
            "tome_layer": args.layer if mode == "tome" else None,
            "tome_r": args.r if mode == "tome" else None,
        },
        "performance": {
            "avg_ttft_ms": statistics.mean(valid_ttft),
            "avg_throughput_tokens_per_sec": statistics.mean(valid_throughput),
        },
        "timing": {
            "benchmark_elapsed_seconds": sum(row["elapsed_seconds"] for row in rows),
            "avg_seconds_per_sample": statistics.mean(row["elapsed_seconds"] for row in rows),
        },
        "accuracy": {"score": correct / len(rows), "correct": correct, "total": len(rows)},
        "public_validation": {
            "passed": invalid == 0,
            "failed_samples": invalid,
        },
        "answers": rows,
    }


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0 or args.warmup_samples < 0:
        raise ValueError("num-samples must be positive and warmup-samples non-negative.")
    samples = load_mmbench_tsv(Path(args.dataset_path), limit=args.num_samples)
    if len(samples) != args.num_samples:
        raise ValueError(f"Requested {args.num_samples} samples, found {len(samples)}.")

    model = VLMModel(args.model_path, backend="transformers", device=args.device)
    config = Qwen35TomeConfig(layer=args.layer, r=args.r)
    install_qwen35_tome(model._model, config)
    model._tome_config = config

    for sample in samples[: args.warmup_samples]:
        run_one(model, sample, args.max_new_tokens, False)
        run_one(model, sample, args.max_new_tokens, True)

    records = {"baseline": [], "tome": []}
    for index, sample in enumerate(samples):
        order = ("baseline", "tome") if index % 2 == 0 else ("tome", "baseline")
        for mode in order:
            records[mode].append(run_one(model, sample, args.max_new_tokens, mode == "tome"))
    for mode in records:
        records[mode].sort(key=lambda row: next(
            index for index, sample in enumerate(samples) if sample.sample_id == row["question_id"]
        ))

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for mode, rows in records.items():
        output = args.output_prefix.with_name(f"{args.output_prefix.name}_{mode}.json")
        output.write_text(json.dumps(build_payload(mode, rows, args), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{mode}: {output}")


if __name__ == "__main__":
    main()
