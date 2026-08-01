"""Calibrate a Qwen3.5 DToMe threshold on a generic image directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from optiz_qwen.compression import visual_unit_matching_scores


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--model-path", default="resources/model_weights/raw/Qwen3.5-2B")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--target-r", type=int, default=32)
    parser.add_argument("--num-images", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--source-edge-bin-limits",
        type=int,
        nargs="+",
        default=[48, 64, 80, 96, 128, 160, 200],
    )
    parser.add_argument(
        "--max-pixel-schedule",
        type=int,
        nargs="+",
        default=None,
        help="Cycle calibration images through these Qwen image pixel budgets.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def select_images(image_dir: Path, count: int, seed: int) -> list[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    paths = sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(paths) < count:
        raise ValueError(f"Requested {count} calibration images, found {len(paths)}.")
    return random.Random(seed).sample(paths, count)


def manifest_sha256(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.num_images <= 0 or args.target_r <= 0:
        raise ValueError("num-images and target-r must be positive.")
    if (
        any(limit <= 0 for limit in args.source_edge_bin_limits)
        or args.source_edge_bin_limits != sorted(set(args.source_edge_bin_limits))
    ):
        raise ValueError("source-edge-bin-limits must be unique increasing positive integers.")
    if args.max_pixel_schedule is not None and any(
        max_pixels < 65536 for max_pixels in args.max_pixel_schedule
    ):
        raise ValueError("max-pixel-schedule values must be at least Qwen's 65536-pixel minimum.")

    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")
    image_paths = select_images(args.image_dir, args.num_images, args.seed)

    pixel_budgets = args.max_pixel_schedule or [None]
    processors = {
        max_pixels: AutoProcessor.from_pretrained(
            args.model_path,
            local_files_only=True,
            trust_remote_code=True,
            **({"max_pixels": max_pixels} if max_pixels is not None else {}),
        )
        for max_pixels in set(pixel_budgets)
    }
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=dtype,
    ).to(device).eval()
    visual = model.model.visual
    if not 0 <= args.layer < len(visual.blocks):
        raise ValueError(f"layer must be in [0, {len(visual.blocks) - 1}].")

    captured_qkv: list[torch.Tensor] = []
    handle = visual.blocks[args.layer].attn.qkv.register_forward_hook(
        lambda _module, _inputs, output: captured_qkv.append(output)
    )
    per_image_scores: list[torch.Tensor] = []
    try:
        for index, path in enumerate(image_paths):
            image = Image.open(path).convert("RGB")
            max_pixels = pixel_budgets[index % len(pixel_budgets)]
            processor = processors[max_pixels]
            inputs = processor(images=image, return_tensors="pt").to(device)
            captured_qkv.clear()
            with torch.inference_mode():
                visual(
                    inputs["pixel_values"],
                    grid_thw=inputs["image_grid_thw"],
                )
            synchronize(device)
            if len(captured_qkv) != 1:
                raise RuntimeError("Expected exactly one QKV capture per calibration image.")
            qkv = captured_qkv[0]
            heads = visual.blocks[args.layer].attn.num_heads
            metric = qkv.reshape(qkv.shape[0], 3, heads, -1)[:, 1].reshape(qkv.shape[0], -1)
            per_image_scores.append(visual_unit_matching_scores(metric).float().cpu())
    finally:
        handle.remove()

    all_scores = torch.cat(per_image_scores)
    target_edges = args.target_r * len(per_image_scores)
    if target_edges >= all_scores.numel():
        raise ValueError(
            f"target-r={args.target_r} requests {target_edges} edges, "
            f"but calibration produced only {all_scores.numel()}."
        )
    sorted_scores = all_scores.sort(descending=True).values
    threshold = float(sorted_scores[target_edges])
    merge_counts = [int((scores > threshold).sum()) for scores in per_image_scores]
    candidate_edge_counts = [scores.numel() for scores in per_image_scores]
    if args.source_edge_bin_limits[-1] < max(candidate_edge_counts):
        raise ValueError(
            "The final source-edge-bin limit must cover every calibration image; "
            f"need at least {max(candidate_edge_counts)}."
        )

    threshold_schedule = []
    lower_limit = 0
    for upper_limit in args.source_edge_bin_limits:
        bucket_scores = [
            scores
            for scores in per_image_scores
            if lower_limit < scores.numel() <= upper_limit
        ]
        lower_limit = upper_limit
        if not bucket_scores:
            continue
        bucket_target_edges = args.target_r * len(bucket_scores)
        bucket_all_scores = torch.cat(bucket_scores)
        if bucket_target_edges >= bucket_all_scores.numel():
            raise ValueError(
                f"target-r={args.target_r} is too large for source-edge bin <= {upper_limit}."
            )
        bucket_threshold = float(
            bucket_all_scores.sort(descending=True).values[bucket_target_edges]
        )
        bucket_merge_counts = [
            int((scores > bucket_threshold).sum()) for scores in bucket_scores
        ]
        threshold_schedule.append(
            {
                "source_edge_limit": upper_limit,
                "image_count": len(bucket_scores),
                "threshold": bucket_threshold,
                "projected_merge_counts": {
                    "mean": statistics.mean(bucket_merge_counts),
                    "median": statistics.median(bucket_merge_counts),
                    "min": min(bucket_merge_counts),
                    "max": max(bucket_merge_counts),
                },
            }
        )

    payload = {
        "calibration_version": "dtome_threshold_v1",
        "timestamp": datetime.now().isoformat(),
        "model_path": str(Path(args.model_path).resolve()),
        "image_dir": str(args.image_dir.resolve()),
        "image_manifest_sha256": manifest_sha256(image_paths, args.image_dir),
        "image_count": len(image_paths),
        "seed": args.seed,
        "layer": args.layer,
        "target_r": args.target_r,
        "max_pixel_schedule": args.max_pixel_schedule,
        "threshold": threshold,
        "candidate_edge_counts": {
            "mean": statistics.mean(candidate_edge_counts),
            "median": statistics.median(candidate_edge_counts),
            "min": min(candidate_edge_counts),
            "max": max(candidate_edge_counts),
        },
        "projected_merge_counts": {
            "mean": statistics.mean(merge_counts),
            "median": statistics.median(merge_counts),
            "min": min(merge_counts),
            "max": max(merge_counts),
        },
        "threshold_schedule": threshold_schedule,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
