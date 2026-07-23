"""Compare synchronized Qwen3.5 visual-block times before and after ToMe."""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import statistics
import sys
import time
import types
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from optiz_qwen.compression.qwen35_tome import Qwen35TomeConfig, install_qwen35_tome


class BlockTimer:
    def __init__(self, block: Any, device: torch.device) -> None:
        self.block = block
        self.device = device
        self.original_forward = block.forward
        self.records: list[tuple[int, float]] = []

    def install(self) -> None:
        timer = self

        def timed_forward(_module, hidden_states, *args, **kwargs):
            synchronize(timer.device)
            started = time.perf_counter()
            output = timer.original_forward(hidden_states, *args, **kwargs)
            synchronize(timer.device)
            timer.records.append((hidden_states.shape[0], (time.perf_counter() - started) * 1000.0))
            return output

        self.block.forward = types.MethodType(timed_forward, self.block)

    def restore(self) -> None:
        self.block.forward = self.original_forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        default="resources/eval_dataset/raw/mmbench_public/mmbench_dev_en.tsv",
    )
    parser.add_argument("--model-path", default="resources/model_weights/raw/Qwen3.5-2B")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--warmup-samples", type=int, default=2)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--r", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def load_rows(path: Path, count: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) < count:
        raise ValueError(f"Requested {count} samples, found {len(rows)}.")
    return rows[:count]


def build_prompt(row: dict[str, str]) -> str:
    choices = "\n".join(
        f"{key}. {(row.get(key) or '').strip()}"
        for key in ("A", "B", "C", "D")
        if (row.get(key) or "").strip()
    )
    return (
        "Solve this single-choice question. Your response must make one final choice among A/B/C/D clearly. "
        "You may include one short reason.\n"
        f"Question: {(row.get('question') or '').strip()}\n{choices}\n"
    )


def prepare_inputs(processor: Any, row: dict[str, str], device: torch.device):
    image = Image.open(io.BytesIO(base64.b64decode(row["image"]))).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_prompt(row)},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)


def run_samples(model: Any, processor: Any, rows: list[dict[str, str]], device: torch.device) -> None:
    for row in rows:
        inputs = prepare_inputs(processor, row, device)
        with torch.inference_mode():
            outputs = model(**inputs, use_cache=False, return_dict=True, logits_to_keep=1)
        synchronize(device)
        del outputs, inputs


def timed_run(
    model: Any,
    processor: Any,
    rows: list[dict[str, str]],
    device: torch.device,
) -> tuple[list[BlockTimer], BlockTimer]:
    visual_timer = BlockTimer(model.model.visual, device)
    timers = [BlockTimer(block, device) for block in model.model.visual.blocks]
    visual_timer.install()
    for timer in timers:
        timer.install()
    try:
        run_samples(model, processor, rows, device)
    finally:
        for timer in timers:
            timer.restore()
        visual_timer.restore()
    return timers, visual_timer


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0 or args.warmup_samples < 0:
        raise ValueError("num-samples must be positive and warmup-samples non-negative.")
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")
    rows = load_rows(Path(args.dataset_path), max(args.num_samples, args.warmup_samples))
    sample_rows = rows[: args.num_samples]
    warmup_rows = rows[: args.warmup_samples]
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=dtype,
    ).to(device).eval()

    run_samples(model, processor, warmup_rows, device)
    baseline_timers, baseline_visual_timer = timed_run(model, processor, sample_rows, device)

    install_qwen35_tome(model, Qwen35TomeConfig(layer=args.layer, r=args.r))
    run_samples(model, processor, warmup_rows, device)
    tome_timers, tome_visual_timer = timed_run(model, processor, sample_rows, device)

    layers = []
    for layer, (baseline, tome) in enumerate(zip(baseline_timers, tome_timers)):
        baseline_times = [elapsed for _, elapsed in baseline.records]
        tome_times = [elapsed for _, elapsed in tome.records]
        layers.append(
            {
                "layer": layer,
                "baseline_input_tokens": [tokens for tokens, _ in baseline.records],
                "tome_input_tokens": [tokens for tokens, _ in tome.records],
                "baseline_mean_ms": statistics.mean(baseline_times),
                "tome_mean_ms": statistics.mean(tome_times),
                "delta_mean_ms": statistics.mean(tome_times) - statistics.mean(baseline_times),
                "baseline_median_ms": statistics.median(baseline_times),
                "tome_median_ms": statistics.median(tome_times),
            }
        )

    baseline_visual_times = [elapsed for _, elapsed in baseline_visual_timer.records]
    tome_visual_times = [elapsed for _, elapsed in tome_visual_timer.records]
    baseline_block_total = sum(row["baseline_mean_ms"] for row in layers)
    tome_block_total = sum(row["tome_mean_ms"] for row in layers)
    visual_summary = {
        "baseline_mean_ms": statistics.mean(baseline_visual_times),
        "tome_mean_ms": statistics.mean(tome_visual_times),
        "delta_mean_ms": statistics.mean(tome_visual_times) - statistics.mean(baseline_visual_times),
        "baseline_block_sum_ms": baseline_block_total,
        "tome_block_sum_ms": tome_block_total,
        "baseline_non_block_ms": statistics.mean(baseline_visual_times) - baseline_block_total,
        "tome_non_block_ms": statistics.mean(tome_visual_times) - tome_block_total,
    }
    payload = {
        "device": str(device),
        "dtype": str(dtype),
        "num_samples": args.num_samples,
        "warmup_samples": args.warmup_samples,
        "tome_layer": args.layer,
        "tome_r": args.r,
        "visual_summary": visual_summary,
        "layers": layers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("layer  baseline_tokens  tome_tokens  baseline_ms  tome_ms  delta_ms")
    for row in layers:
        print(
            f"{row['layer']:>5}  "
            f"{statistics.mean(row['baseline_input_tokens']):>15.1f}  "
            f"{statistics.mean(row['tome_input_tokens']):>11.1f}  "
            f"{row['baseline_mean_ms']:>11.3f}  "
            f"{row['tome_mean_ms']:>7.3f}  "
            f"{row['delta_mean_ms']:>8.3f}"
        )
    print("visual_summary:", json.dumps(visual_summary, indent=2))
    print(f"JSON: {args.output}")


if __name__ == "__main__":
    main()
