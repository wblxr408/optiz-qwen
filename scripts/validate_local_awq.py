"""Dry-run CLI for local AWQ smoke validation planning.

This script validates the local smoke config and optional JSONL sample metadata.
It does not load models, read images, run inference, quantize weights, or write
artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "configs/experiments/local_awq_smoke.yaml"
REQUIRED_CONFIG_PATH_KEYS = (
    "baseline_model_path",
    "awq_artifact_path",
    "local_validation_data",
)


def _is_absolute_path(value: str) -> bool:
    return PureWindowsPath(value).is_absolute() or value.startswith("/")


def repo_relative_path(value: str, *, field_name: str) -> PurePosixPath:
    if not value or value.strip() == "":
        raise ValueError(f"{field_name} must not be empty")
    if _is_absolute_path(value) or PureWindowsPath(value).drive:
        raise ValueError(f"{field_name} must be a repository-relative path: {value}")
    normalized = value.replace("\\", "/").strip("/")
    path = PurePosixPath(normalized)
    if ".." in path.parts:
        raise ValueError(f"{field_name} must not contain path traversal '..': {value}")
    return path


def resolve_repo_file(value: str, *, field_name: str, must_exist: bool = True) -> Path:
    relative_path = repo_relative_path(value, field_name=field_name)
    resolved = REPO_ROOT / Path(relative_path.as_posix())
    if must_exist and not resolved.exists():
        raise ValueError(f"{field_name} does not exist: {value}")
    if must_exist and not resolved.is_file():
        raise ValueError(f"{field_name} must be a file: {value}")
    return resolved


def parse_simple_yaml_scalars(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if line.startswith(" ") or line.startswith("-"):
            continue
        if ":" not in line:
            raise ValueError(f"config line {line_number} is not a key-value entry")
        key, raw_value = line.split(":", 1)
        values[key.strip()] = raw_value.strip().strip('"').strip("'")
    return values


def positive_int(value: str, *, field_name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive")
    return parsed


def validate_config(config: dict[str, str]) -> None:
    if config.get("validation_scope") != "local_smoke":
        raise ValueError("validation_scope must be local_smoke")
    if config.get("performance_claim") != "not_benchmarked":
        raise ValueError("performance_claim must be not_benchmarked")
    for key in REQUIRED_CONFIG_PATH_KEYS:
        if key not in config:
            raise ValueError(f"config must include {key}")
        repo_relative_path(config[key], field_name=key)
    if "max_samples" not in config:
        raise ValueError("config must include max_samples")
    positive_int(config["max_samples"], field_name="max_samples")


def validate_sample_record(record: Any, *, line_number: int) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"samples line {line_number} must be a JSON object")
    for key in ("sample_id", "image_path", "question"):
        value = record.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"samples line {line_number} must include non-empty {key}")
    repo_relative_path(record["image_path"], field_name=f"samples line {line_number} image_path")
    has_reference = isinstance(record.get("reference_answer"), str) and record[
        "reference_answer"
    ].strip()
    has_behavior = isinstance(record.get("expected_behavior"), str) and record[
        "expected_behavior"
    ].strip()
    if not has_reference and not has_behavior:
        raise ValueError(
            f"samples line {line_number} must include reference_answer or expected_behavior"
        )


def validate_samples(samples_path: Path, *, max_samples: int) -> int:
    valid_count = 0
    with samples_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if valid_count >= max_samples:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"samples line {line_number} is invalid JSON: {exc.msg}") from exc
            validate_sample_record(record, line_number=line_number)
            valid_count += 1
    return valid_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run local AWQ smoke validation plan for Qwen3.5-2B VLM",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--samples", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    if not args.dry_run:
        raise ValueError("current phase only supports dry-run; real local inference is not implemented")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max_samples must be positive")

    config_relative = repo_relative_path(args.config, field_name="config")
    config_path = resolve_repo_file(args.config, field_name="config")
    config = parse_simple_yaml_scalars(config_path)
    validate_config(config)

    samples_value = args.samples or config["local_validation_data"]
    samples_relative = repo_relative_path(samples_value, field_name="samples")
    samples_path = REPO_ROOT / Path(samples_relative.as_posix())
    planned_max_samples = args.max_samples or positive_int(
        config["max_samples"],
        field_name="max_samples",
    )

    samples_status = "present" if samples_path.exists() else "missing"
    validated_sample_count = 0
    if samples_status == "present":
        if not samples_path.is_file():
            raise ValueError(f"samples must be a file: {samples_value}")
        validated_sample_count = validate_samples(samples_path, max_samples=planned_max_samples)

    return {
        "validation_scope": config["validation_scope"],
        "dry_run": True,
        "config_path": config_relative.as_posix(),
        "samples_path": samples_relative.as_posix(),
        "samples_status": samples_status,
        "planned_status": f"samples_{samples_status}",
        "planned_max_samples": planned_max_samples,
        "validated_sample_count": validated_sample_count,
        "loads_model": False,
        "writes_artifacts": False,
        "uses_official_dataset": False,
        "performance_claim": config["performance_claim"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
