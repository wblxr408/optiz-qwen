"""CLI for GPTQ W4A16 planning, preflight checks, and guarded execution.

Dry-run and preflight modes do not load models, import heavy ML dependencies,
download assets, or write quantized artifacts. Real execution is only reachable
via --execute plus --confirm-write-artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from optiz_qwen.compression.gptq_backend import probe_gptq_backend
from optiz_qwen.compression.gptq_execution import (
    execute_llmcompressor_gptq_quantization,
)

DEFAULT_MODEL_PATH = "resources/model_weights/raw/Qwen3.5-2B"
DEFAULT_CALIBRATION_TSV = "resources/eval_dataset/raw/mmbench_public/mmbench_dev_en.tsv"
DEFAULT_OUTPUT_DIR = "artifacts/quantized/qwen35_2b_gptq_w4a16"
ALLOWED_OUTPUT_ROOT = PurePosixPath("artifacts/quantized/qwen35_2b_gptq_w4a16")


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
            "output_dir must be artifacts/quantized/qwen35_2b_gptq_w4a16 "
            f"or one of its subdirectories: {value}"
        )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="llmcompressor GPTQ W4A16 quantization for Qwen3.5-2B VLM",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--calibration-tsv", default=DEFAULT_CALIBRATION_TSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-calibration-samples", type=int, default=128)
    parser.add_argument("--weight-bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--activation-dtype", default="bf16")
    parser.add_argument("--backend", default="llmcompressor")
    parser.add_argument("--confirm-write-artifacts", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def validate_quantization_args(args: argparse.Namespace) -> None:
    if args.num_calibration_samples <= 0:
        raise ValueError("num_calibration_samples must be positive")
    if args.weight_bits != 4:
        raise ValueError("GPTQ main path only supports W4A16, so weight_bits must be 4")
    if args.group_size != 128:
        raise ValueError("GPTQ W4A16 path requires group_size=128")
    if args.activation_dtype.lower() != "bf16":
        raise ValueError("GPTQ W4A16 path requires activation_dtype=bf16")


def build_preflight_payload(args: argparse.Namespace) -> dict[str, Any]:
    validate_quantization_args(args)

    model_path = _resolve_existing_path(args.model_path, field_name="model_path")
    calibration_tsv = _resolve_existing_path(
        args.calibration_tsv,
        field_name="calibration_tsv",
    )
    output_dir = validate_output_dir(args.output_dir)
    backend_readiness = probe_gptq_backend(args.backend)

    return {
        "mode": "preflight",
        "backend": backend_readiness.to_dict(),
        "model_path": args.model_path,
        "model_path_resolved": str(model_path),
        "calibration_tsv": args.calibration_tsv,
        "calibration_tsv_resolved": str(calibration_tsv),
        "output_dir": output_dir.as_posix(),
        "would_load_model": False,
        "would_write_artifacts": False,
        "can_execute": backend_readiness.can_quantize,
        "performance_claim": "not_benchmarked",
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    preflight = build_preflight_payload(args)

    return {
        "phase": "gptq_w4a16_dry_run",
        "mode": "dry_run",
        "dry_run": True,
        "status": "planned_only",
        "model_path": args.model_path,
        "model_path_resolved": preflight["model_path_resolved"],
        "calibration_tsv": args.calibration_tsv,
        "calibration_tsv_resolved": preflight["calibration_tsv_resolved"],
        "output_dir": preflight["output_dir"],
        "quantization": {
            "backend": "llmcompressor",
            "method": "gptq",
            "scheme": "W4A16",
            "weight_bits": args.weight_bits,
            "activation_dtype": args.activation_dtype.lower(),
            "group_size": args.group_size,
            "targets": "language model Linear layers",
            "ignore": ["lm_head", "re:.*visual.*"],
            "vision_dtype": "bf16",
            "serialization": "compressed-tensors",
        },
        "calibration": {
            "source_format": "MMBench TSV",
            "sample_count": args.num_calibration_samples,
            "multimodal": True,
        },
        "backend": preflight["backend"],
        "metadata_preview": {
            "artifact_status": "not_generated",
            "performance_claim": "not_benchmarked",
            "writes_artifacts": False,
            "loads_model": False,
        },
    }


def execute_plan(args: argparse.Namespace) -> dict[str, Any]:
    if not args.confirm_write_artifacts:
        raise ValueError(
            "--execute requires --confirm-write-artifacts before real GPTQ can run"
        )

    preflight = build_preflight_payload(args)
    backend_readiness = probe_gptq_backend(args.backend)
    if not backend_readiness.can_quantize:
        raise RuntimeError(
            "llmcompressor GPTQ backend is not available according to preflight; "
            "real GPTQ execution cannot continue"
        )

    output_relative = validate_output_dir(args.output_dir)
    return execute_llmcompressor_gptq_quantization(
        model_path=Path(preflight["model_path_resolved"]),
        calibration_tsv=Path(preflight["calibration_tsv_resolved"]),
        output_dir=REPO_ROOT / Path(output_relative.as_posix()),
        num_calibration_samples=args.num_calibration_samples,
        weight_bits=args.weight_bits,
        group_size=args.group_size,
        activation_dtype=args.activation_dtype,
        confirm_write_artifacts=args.confirm_write_artifacts,
        backend_readiness=backend_readiness,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.dry_run:
            plan = build_plan(args)
        elif args.preflight:
            plan = build_preflight_payload(args)
        else:
            plan = execute_plan(args)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    print(json.dumps(plan, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
