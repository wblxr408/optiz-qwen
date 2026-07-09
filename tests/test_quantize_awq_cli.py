from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT = ROOT_DIR / "scripts" / "quantize_awq.py"
OUTPUT_DIR = "artifacts/quantized/qwen35_2b_awq_w4a16"


@pytest.fixture()
def cli_inputs() -> Iterator[dict[str, str]]:
    root = ROOT_DIR / ".pytest-workspace-tmp" / uuid.uuid4().hex
    model_dir = root / "fake_model"
    calibration_tsv = root / "calibration.tsv"
    model_dir.mkdir(parents=True)
    calibration_tsv.write_text(
        "index\timage\tquestion\tA\tB\tanswer\n"
        "1\tfake-image\tWhich option is visible?\tLeft\tRight\tA\n",
        encoding="utf-8",
    )
    try:
        yield {
            "model_path": model_dir.relative_to(ROOT_DIR).as_posix(),
            "calibration_tsv": calibration_tsv.relative_to(ROOT_DIR).as_posix(),
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)
        try:
            root.parent.rmdir()
        except OSError:
            pass


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT_DIR,
        text=True,
        capture_output=True,
        check=False,
    )


def test_dry_run_outputs_plan_without_creating_artifact(
    cli_inputs: dict[str, str],
) -> None:
    result = run_cli(
        "--model-path",
        cli_inputs["model_path"],
        "--calibration-tsv",
        cli_inputs["calibration_tsv"],
        "--output-dir",
        OUTPUT_DIR,
        "--num-calibration-samples",
        "8",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert payload["output_dir"] == OUTPUT_DIR
    assert payload["quantization"]["method"] == "awq"
    assert payload["quantization"]["scheme"] == "W4A16"
    assert payload["quantization"]["weight_bits"] == 4
    assert payload["quantization"]["activation_dtype"] == "bf16"
    assert payload["backend"]["backend_name"] == "autoawq"
    assert isinstance(payload["backend"]["package_available"], bool)
    assert payload["backend"]["can_quantize"] == payload["backend"]["package_available"]
    assert payload["backend"]["reason"]
    assert payload["backend"]["recommended_environment"]
    assert payload["metadata_preview"]["performance_claim"] == "not_benchmarked"
    assert payload["metadata_preview"]["writes_artifacts"] is False
    assert payload["metadata_preview"]["loads_model"] is False


def test_dry_run_accepts_explicit_autoawq_backend(cli_inputs: dict[str, str]) -> None:
    result = run_cli(
        "--model-path",
        cli_inputs["model_path"],
        "--calibration-tsv",
        cli_inputs["calibration_tsv"],
        "--output-dir",
        OUTPUT_DIR,
        "--backend",
        "autoawq",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["backend"]["backend_name"] == "autoawq"
    assert isinstance(payload["backend"]["can_quantize"], bool)


def test_preflight_outputs_json_without_creating_artifacts(cli_inputs: dict[str, str]) -> None:
    result = run_cli(
        "--model-path",
        cli_inputs["model_path"],
        "--calibration-tsv",
        cli_inputs["calibration_tsv"],
        "--output-dir",
        OUTPUT_DIR,
        "--preflight",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["mode"] == "preflight"
    assert payload["model_path"] == cli_inputs["model_path"]
    assert payload["calibration_tsv"] == cli_inputs["calibration_tsv"]
    assert payload["output_dir"] == OUTPUT_DIR
    assert payload["would_load_model"] is False
    assert payload["would_write_artifacts"] is False
    assert payload["can_execute"] == payload["backend"]["package_available"]
    assert payload["performance_claim"] == "not_benchmarked"


def test_execute_without_confirm_is_rejected(cli_inputs: dict[str, str]) -> None:
    result = run_cli(
        "--model-path",
        cli_inputs["model_path"],
        "--calibration-tsv",
        cli_inputs["calibration_tsv"],
        "--output-dir",
        OUTPUT_DIR,
        "--execute",
    )

    assert result.returncode != 0
    assert "--confirm-write-artifacts" in result.stderr


def test_modes_are_mutually_exclusive(cli_inputs: dict[str, str]) -> None:
    result = run_cli(
        "--model-path",
        cli_inputs["model_path"],
        "--calibration-tsv",
        cli_inputs["calibration_tsv"],
        "--output-dir",
        OUTPUT_DIR,
        "--dry-run",
        "--preflight",
    )

    assert result.returncode != 0
    assert "not allowed with argument" in result.stderr


def test_mode_is_required(cli_inputs: dict[str, str]) -> None:
    result = run_cli(
        "--model-path",
        cli_inputs["model_path"],
        "--calibration-tsv",
        cli_inputs["calibration_tsv"],
        "--output-dir",
        OUTPUT_DIR,
    )

    assert result.returncode != 0
    assert "one of the arguments --dry-run --preflight --execute is required" in result.stderr


def test_unknown_backend_is_rejected(cli_inputs: dict[str, str]) -> None:
    result = run_cli(
        "--model-path",
        cli_inputs["model_path"],
        "--calibration-tsv",
        cli_inputs["calibration_tsv"],
        "--output-dir",
        OUTPUT_DIR,
        "--backend",
        "unknown_backend",
        "--preflight",
    )

    assert result.returncode != 0
    assert "unsupported AWQ backend" in result.stderr


def test_output_dir_must_stay_under_awq_artifact_root(
    cli_inputs: dict[str, str],
) -> None:
    result = run_cli(
        "--model-path",
        cli_inputs["model_path"],
        "--calibration-tsv",
        cli_inputs["calibration_tsv"],
        "--output-dir",
        "artifacts/quantized/other_model",
        "--dry-run",
    )

    assert result.returncode != 0
    assert "output_dir must be artifacts/quantized/qwen35_2b_awq_w4a16" in result.stderr


def test_output_dir_must_be_repository_relative(cli_inputs: dict[str, str]) -> None:
    result = run_cli(
        "--model-path",
        cli_inputs["model_path"],
        "--calibration-tsv",
        cli_inputs["calibration_tsv"],
        "--output-dir",
        "D:/tmp/qwen35_2b_awq_w4a16",
        "--dry-run",
    )

    assert result.returncode != 0
    assert "repository-relative" in result.stderr


def test_output_dir_rejects_path_traversal(cli_inputs: dict[str, str]) -> None:
    result = run_cli(
        "--model-path",
        cli_inputs["model_path"],
        "--calibration-tsv",
        cli_inputs["calibration_tsv"],
        "--output-dir",
        "artifacts/quantized/qwen35_2b_awq_w4a16/../escape",
        "--dry-run",
    )

    assert result.returncode != 0
    assert "path traversal" in result.stderr


def test_drive_relative_paths_are_rejected(cli_inputs: dict[str, str]) -> None:
    result = run_cli(
        "--model-path",
        cli_inputs["model_path"],
        "--calibration-tsv",
        cli_inputs["calibration_tsv"],
        "--output-dir",
        "D:tmp/qwen35_2b_awq_w4a16",
        "--dry-run",
    )

    assert result.returncode != 0
    assert "repository-relative" in result.stderr


def test_missing_calibration_tsv_is_rejected(cli_inputs: dict[str, str]) -> None:
    result = run_cli(
        "--model-path",
        cli_inputs["model_path"],
        "--calibration-tsv",
        ".pytest-workspace-tmp/missing.tsv",
        "--output-dir",
        OUTPUT_DIR,
        "--dry-run",
    )

    assert result.returncode != 0
    assert "calibration_tsv does not exist" in result.stderr
