"""Dry-run CLI for the planned AWQ W4A16 quantization workflow.

Phase 1 intentionally does not load models, import torch/transformers, install
AutoAWQ, download assets, or write quantized artifacts. It validates inputs and
prints the quantization plan that a later server-side implementation should run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = "resources/model_weights/raw/Qwen3.5-2B"
DEFAULT_CALIBRATION_TSV = "resources/eval_dataset/raw/mmbench_public/mmbench_dev_en.tsv"
DEFAULT_OUTPUT_DIR = "artifacts/quantized/qwen35_2b_awq_w4a16"
ALLOWED_OUTPUT_ROOT = PurePosixPath("artifacts/quantized/qwen35_2b_awq_w4a16")


def _is_absolute_path(value: str) -> bool:
    return PureWindowsPath(value).is_absolute() or value.startswith("/")


def _repo_relative_posix(value: str, *, field_name: str) -> PurePosixPath:
    if not value or value.strip() == "":
        raise ValueError(f"{field_name} must not be empty")
    windows_path = PureWindowsPath(value)
    if _is_absolute_path(value) or windows_path.drive:
        raise ValueError(f"{field_name} must be a repository-relative path: {value}")
    normalized = value.replace("\\", "/").strip("/")
    path = PurePosixPath(normalized)
    if ".." in path.parts:
        raise ValueError(f"{field_name} must not contain path traversal '..': {value}")
    return path


def _resolve_existing_path(value: str, *, field_name: str) -> Path:
    relative_path = _repo_relative_posix(value, field_name=field_name)
    resolved = REPO_ROOT / Path(relative_path.as_posix())
    if not resolved.exists():
        raise ValueError(f"{field_name} does not exist: {value}")
    return resolved


def validate_output_dir(value: str) -> PurePosixPath:
    output_dir = _repo_relative_posix(value, field_name="output_dir")
    if output_dir != ALLOWED_OUTPUT_ROOT and ALLOWED_OUTPUT_ROOT not in output_dir.parents:
        raise ValueError(
            "output_dir must be artifacts/quantized/qwen35_2b_awq_w4a16 "
            f"or one of its subdirectories: {value}"
        )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run AWQ W4A16 quantization plan for Qwen3.5-2B VLM",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--calibration-tsv", default=DEFAULT_CALIBRATION_TSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-calibration-samples", type=int, default=128)
    parser.add_argument("--weight-bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--activation-dtype", default="bf16")
    parser.add_argument("--zero-point", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_calibration_samples <= 0:
        raise ValueError("num_calibration_samples must be positive")
    if args.weight_bits != 4:
        raise ValueError("Phase 1 AWQ plan only supports W4A16, so weight_bits must be 4")
    if args.group_size <= 0:
        raise ValueError("group_size must be positive")
    if args.activation_dtype.lower() != "bf16":
        raise ValueError("Phase 1 AWQ plan requires activation_dtype=bf16")

    model_path = _resolve_existing_path(args.model_path, field_name="model_path")
    calibration_tsv = _resolve_existing_path(
        args.calibration_tsv,
        field_name="calibration_tsv",
    )
    output_dir = validate_output_dir(args.output_dir)

    return {
        "phase": "awq_w4a16_dry_run",
        "dry_run": True,
        "status": "planned_only",
        "model_path": args.model_path,
        "model_path_resolved": str(model_path),
        "calibration_tsv": args.calibration_tsv,
        "calibration_tsv_resolved": str(calibration_tsv),
        "output_dir": output_dir.as_posix(),
        "quantization": {
            "method": "awq",
            "scheme": "W4A16",
            "weight_bits": args.weight_bits,
            "activation_dtype": args.activation_dtype.lower(),
            "group_size": args.group_size,
            "zero_point": args.zero_point,
        },
        "calibration": {
            "source_format": "MMBench TSV",
            "sample_count": args.num_calibration_samples,
        },
        "metadata_preview": {
            "artifact_status": "not_generated",
            "performance_claim": "not_benchmarked",
            "writes_artifacts": False,
            "loads_model": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.dry_run:
        parser.error("current phase only supports --dry-run; real AWQ is not implemented")

    try:
        plan = build_plan(args)
    except ValueError as exc:
        parser.error(str(exc))

    print(json.dumps(plan, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
