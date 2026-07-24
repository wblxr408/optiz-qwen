#!/usr/bin/env python3
"""Build a deterministic, image-validated AWQ calibration manifest."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_ROOT = REPO_ROOT / "scripts"
for import_root in (SRC_ROOT, SCRIPT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from optiz_qwen.evaluation.dndx_public_benchmark import (
    Sample,
    decode_image,
    load_mmbench_tsv,
)
from awq_contract import semantic_config_sha256
from experiment_utils import inspect_git, sha256_file, sha256_lines


DEFAULT_CONFIG = REPO_ROOT / "configs/experiments/d_awq_w4a16_cuda.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a deterministic AWQ calibration manifest without modifying weights."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        help="Override calibration.dataset when shared assets live outside the worktree.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def resolve_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def sample_sha256(sample: Sample) -> str:
    image_bytes = base64.b64decode(sample.image_b64, validate=True)
    payload = {
        "sample_id": sample.sample_id,
        "language": sample.language,
        "question": sample.question,
        "hint": sample.hint,
        "choices": sample.choices,
        "answer": sample.answer,
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "category": sample.category,
        "subcategory": sample.subcategory,
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("AWQ experiment config must be a JSON object.")
    calibration = raw.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("AWQ experiment config is missing calibration settings.")
    return raw


def build_manifest(
    *,
    repo_root: Path,
    config_path: Path,
    dataset_path_override: Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    calibration = config["calibration"]
    dataset_path = (
        dataset_path_override.expanduser().resolve()
        if dataset_path_override is not None
        else resolve_path(repo_root, calibration["dataset"])
    )
    dataset_sha256 = sha256_file(dataset_path)
    expected_dataset_sha256 = calibration.get("expected_dataset_sha256")
    if dataset_sha256 != expected_dataset_sha256:
        raise ValueError("Calibration dataset SHA-256 does not match the experiment contract.")

    samples = load_mmbench_tsv(dataset_path)
    num_samples = int(calibration["num_samples"])
    excluded_prefix_count = int(calibration["exclude_fixed_eval_prefix_count"])
    seed = int(calibration["seed"])
    if num_samples <= 0 or excluded_prefix_count < 0:
        raise ValueError("Calibration counts must be positive and non-negative.")
    if len(samples) < num_samples + excluded_prefix_count:
        raise ValueError("Dataset is too small for disjoint calibration and evaluation samples.")

    evaluation_samples = samples[:excluded_prefix_count]
    pool = samples[excluded_prefix_count:]
    selected = random.Random(seed).sample(pool, num_samples)
    evaluation_ids = [sample.sample_id for sample in evaluation_samples]
    selected_ids = [sample.sample_id for sample in selected]
    overlap = sorted(set(evaluation_ids) & set(selected_ids))
    if overlap:
        raise ValueError("Calibration and fixed evaluation sample IDs overlap.")

    selected_rows: list[dict[str, Any]] = []
    for sample in selected:
        with decode_image(sample.image_b64) as image:
            width, height = image.size
            mode = image.mode
        selected_rows.append(
            {
                "sample_id": sample.sample_id,
                "sample_sha256": sample_sha256(sample),
                "image_width": int(width),
                "image_height": int(height),
                "image_mode": mode,
            }
        )

    return {
        "schema_version": 1,
        "manifest_type": "awq_multimodal_calibration_selection",
        "read_only_weight_contract": True,
        "experiment_id": config.get("experiment_id"),
        "execution_enabled": bool(config.get("execution_enabled")),
        "git": inspect_git(repo_root),
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "semantic_sha256": semantic_config_sha256(config),
        },
        "dataset": {
            "path": str(dataset_path),
            "sha256": dataset_sha256,
            "sample_count": len(samples),
        },
        "selection": {
            "seed": seed,
            "requested_calibration_samples": num_samples,
            "selected_calibration_samples": len(selected_rows),
            "excluded_eval_prefix_count": excluded_prefix_count,
            "excluded_eval_sample_ids": evaluation_ids,
            "excluded_eval_sample_ids_sha256": sha256_lines(evaluation_ids),
            "selected_sample_ids_sha256": sha256_lines(selected_ids),
            "overlap_count": 0,
            "all_images_decoded": len(selected_rows) == num_samples,
            "samples": selected_rows,
        },
        "claim_boundary": "calibration_manifest_only_no_quantization_or_performance_claim",
    }


def write_manifest(path: Path, manifest: dict[str, Any], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    config_path = resolve_path(repo_root, args.config)
    manifest = build_manifest(
        repo_root=repo_root,
        config_path=config_path,
        dataset_path_override=args.dataset_path,
    )
    if args.output is None:
        print(json.dumps(manifest, indent=2, ensure_ascii=True))
    else:
        output = resolve_path(repo_root, args.output)
        write_manifest(output, manifest, overwrite=args.overwrite)
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
