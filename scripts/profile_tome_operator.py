"""Profile isolated ToMe stages at representative Qwen3.5 visual lengths."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from optiz_qwen.compression.tome import merge_single_visual_sample


TOKEN_COUNTS = (256, 320, 480, 768)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--r", type=int, default=32)
    parser.add_argument("--matching", choices=["tome", "pitome", "dtome"], default="tome")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def make_inputs(token_count: int, device: torch.device):
    generator = torch.Generator(device="cpu").manual_seed(20260713 + token_count)
    hidden_states = torch.randn(token_count, 1024, generator=generator, dtype=torch.float32)
    metric = torch.randn(token_count, 1024, generator=generator, dtype=torch.float32)
    dtype = torch.bfloat16 if device.type != "cpu" else torch.float32
    return (
        hidden_states.to(device=device, dtype=dtype),
        metric.to(device=device, dtype=dtype),
        torch.ones(token_count, 1, device=device, dtype=dtype),
    )


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive.")
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")

    results = []
    for token_count in TOKEN_COUNTS:
        inputs = make_inputs(token_count, device)
        for _ in range(args.warmup):
            merge_single_visual_sample(
                *inputs,
                r=args.r,
                matching=args.matching,
                threshold=args.threshold,
            )
        samples = [
            merge_single_visual_sample(
                *inputs,
                r=args.r,
                matching=args.matching,
                threshold=args.threshold,
                profile=True,
            ).timings_ms
            for _ in range(args.repeats)
        ]
        if any(sample is None for sample in samples):
            raise RuntimeError("Profiling did not return stage timings.")
        stage_names = tuple(samples[0])
        summary = {
            name: {
                "mean_ms": statistics.mean(sample[name] for sample in samples),
                "median_ms": statistics.median(sample[name] for sample in samples),
            }
            for name in stage_names
        }
        results.append(
            {
                "input_tokens": token_count,
                "compact_tokens": token_count - args.r * 4,
                "r": args.r,
                "timings": summary,
            }
        )

    payload = {
        "device": str(device),
        "dtype": "bfloat16" if device.type != "cpu" else "float32",
        "warmup": args.warmup,
        "repeats": args.repeats,
        "matching": args.matching,
        "threshold": args.threshold,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("tokens  compact  metric_ms  match_ms  aggregate_ms  compact_ms  total_ms")
    for row in results:
        timing = row["timings"]
        print(
            f"{row['input_tokens']:>6}  {row['compact_tokens']:>7}  "
            f"{timing['metric_preparation']['median_ms']:>9.3f}  "
            f"{timing['bipartite_matching']['median_ms']:>8.3f}  "
            f"{timing['weighted_aggregation']['median_ms']:>12.3f}  "
            f"{timing['output_compaction']['median_ms']:>10.3f}  "
            f"{timing['total']['median_ms']:>8.3f}"
        )
    print(f"JSON: {args.output}")


if __name__ == "__main__":
    main()
