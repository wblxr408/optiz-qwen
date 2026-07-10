"""Dry-run local AWQ status summary CLI.

This script performs read-only checks for the local AWQ smoke workflow. It does
not load models, read images, run inference, benchmark, or write artifacts.
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
    windows_path = PureWindowsPath(value)
    if _is_absolute_path(value) or windows_path.drive:
        raise ValueError(f"{field_name} must be a repository-relative path: {value}")
    normalized = value.replace("\\", "/").strip("/")
    path = PurePosixPath(normalized)
    if ".." in path.parts:
        raise ValueError(f"{field_name} must not contain path traversal '..': {value}")
    return path


def resolve_repo_path(value: str, *, field_name: str) -> Path:
    relative_path = repo_relative_path(value, field_name=field_name)
    return REPO_ROOT / Path(relative_path.as_posix())


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


def positive_int(value: str | int, *, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
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


def directory_size_bytes(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def size_bytes_if_present(path: Path) -> int | None:
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return directory_size_bytes(path)
    return None


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


def inspect_samples(samples_path: Path, *, max_samples: int) -> tuple[str, int]:
    if not samples_path.exists():
        return "missing", 0
    if not samples_path.is_file():
        raise ValueError(f"samples must be a file: {samples_path.relative_to(REPO_ROOT).as_posix()}")

    inspected_count = 0
    with samples_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if inspected_count >= max_samples:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"samples line {line_number} is invalid JSON: {exc.msg}") from exc
            validate_sample_record(record, line_number=line_number)
            inspected_count += 1
    return "present", inspected_count


def load_preflight_json(value: str | None) -> tuple[str, dict[str, Any] | None]:
    if value is None:
        return "not_provided", None
    preflight_path = resolve_repo_path(value, field_name="preflight_json")
    if not preflight_path.exists():
        raise ValueError(f"preflight_json does not exist: {value}")
    if not preflight_path.is_file():
        raise ValueError(f"preflight_json must be a file: {value}")
    try:
        payload = json.loads(preflight_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"preflight_json is invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("preflight_json must contain a JSON object")
    return "loaded", payload


def build_next_steps(
    *,
    baseline_exists: bool,
    artifact_exists: bool,
    samples_status: str,
    inspected_sample_count: int,
    can_run_local_comparison: bool,
) -> list[str]:
    steps: list[str] = []
    if not baseline_exists:
        steps.append("Place the BF16 baseline model at the configured repository-relative path.")
    if not artifact_exists:
        steps.append("Run guarded AWQ execution on a prepared server to produce the local artifact.")
    if samples_status == "missing":
        steps.append("Add team-authored local smoke samples when ready; do not use official evaluation data.")
    elif inspected_sample_count == 0:
        steps.append("Add at least one valid local smoke sample before comparison.")
    if can_run_local_comparison:
        steps.append("A future local BF16 vs AWQ smoke comparison can be run manually; this script did not run it.")
    steps.append("Keep performance_claim as not_benchmarked until real comparison artifacts exist.")
    return steps


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    if not args.dry_run:
        raise ValueError("current phase only supports dry-run/status summary")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max_samples must be positive")

    config_relative = repo_relative_path(args.config, field_name="config")
    config_path = REPO_ROOT / Path(config_relative.as_posix())
    if not config_path.exists():
        raise ValueError(f"config does not exist: {args.config}")
    if not config_path.is_file():
        raise ValueError(f"config must be a file: {args.config}")

    config = parse_simple_yaml_scalars(config_path)
    validate_config(config)

    max_samples = args.max_samples or positive_int(config["max_samples"], field_name="max_samples")
    baseline_relative = repo_relative_path(config["baseline_model_path"], field_name="baseline_model_path")
    artifact_relative = repo_relative_path(config["awq_artifact_path"], field_name="awq_artifact_path")
    samples_relative = repo_relative_path(config["local_validation_data"], field_name="local_validation_data")

    baseline_path = REPO_ROOT / Path(baseline_relative.as_posix())
    artifact_path = REPO_ROOT / Path(artifact_relative.as_posix())
    samples_path = REPO_ROOT / Path(samples_relative.as_posix())

    samples_status, inspected_sample_count = inspect_samples(samples_path, max_samples=max_samples)
    preflight_status, preflight_payload = load_preflight_json(args.preflight_json)

    baseline_exists = baseline_path.exists()
    artifact_exists = artifact_path.exists()
    can_run_local_comparison = (
        baseline_exists
        and artifact_exists
        and samples_status == "present"
        and inspected_sample_count > 0
    )

    summary: dict[str, Any] = {
        "mode": "local_awq_status",
        "dry_run": True,
        "config_path": config_relative.as_posix(),
        "validation_scope": config["validation_scope"],
        "performance_claim": config["performance_claim"],
        "baseline_model_path": baseline_relative.as_posix(),
        "baseline_exists": baseline_exists,
        "baseline_size_bytes": size_bytes_if_present(baseline_path),
        "awq_artifact_path": artifact_relative.as_posix(),
        "awq_artifact_exists": artifact_exists,
        "awq_artifact_status": "present" if artifact_exists else "artifact_missing",
        "awq_artifact_size_bytes": size_bytes_if_present(artifact_path),
        "samples_path": samples_relative.as_posix(),
        "samples_status": samples_status,
        "planned_max_samples": max_samples,
        "inspected_sample_count": inspected_sample_count,
        "preflight_status": preflight_status,
        "can_run_local_comparison": can_run_local_comparison,
        "would_load_model": False,
        "would_run_inference": False,
        "would_write_artifacts": False,
        "uses_official_benchmark": False,
        "next_steps": build_next_steps(
            baseline_exists=baseline_exists,
            artifact_exists=artifact_exists,
            samples_status=samples_status,
            inspected_sample_count=inspected_sample_count,
            can_run_local_comparison=can_run_local_comparison,
        ),
    }
    if preflight_payload is not None:
        summary["preflight_summary"] = {
            "mode": preflight_payload.get("mode"),
            "can_execute": preflight_payload.get("can_execute"),
            "performance_claim": preflight_payload.get("performance_claim"),
        }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run local AWQ status summary for Qwen3.5-2B VLM",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-json", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = build_summary(args)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
