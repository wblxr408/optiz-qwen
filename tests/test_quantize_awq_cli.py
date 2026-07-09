from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT = ROOT_DIR / "scripts" / "quantize_awq.py"
MODEL_PATH = "resources/model_weights/raw/Qwen3.5-2B"
CALIBRATION_TSV = "resources/eval_dataset/raw/mmbench_public/mmbench_dev_en.tsv"
OUTPUT_DIR = "artifacts/quantized/qwen35_2b_awq_w4a16"


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT_DIR,
        text=True,
        capture_output=True,
        check=False,
    )


def test_dry_run_outputs_plan_without_creating_artifact() -> None:
    result = run_cli(
        "--model-path",
        MODEL_PATH,
        "--calibration-tsv",
        CALIBRATION_TSV,
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
    assert payload["metadata_preview"]["performance_claim"] == "not_benchmarked"
    assert payload["metadata_preview"]["writes_artifacts"] is False


def test_non_dry_run_is_rejected() -> None:
    result = run_cli(
        "--model-path",
        MODEL_PATH,
        "--calibration-tsv",
        CALIBRATION_TSV,
        "--output-dir",
        OUTPUT_DIR,
    )

    assert result.returncode != 0
    assert "only supports --dry-run" in result.stderr
    assert "real AWQ is not implemented" in result.stderr


def test_output_dir_must_stay_under_awq_artifact_root() -> None:
    result = run_cli(
        "--model-path",
        MODEL_PATH,
        "--calibration-tsv",
        CALIBRATION_TSV,
        "--output-dir",
        "artifacts/quantized/other_model",
        "--dry-run",
    )

    assert result.returncode != 0
    assert "output_dir must be artifacts/quantized/qwen35_2b_awq_w4a16" in result.stderr


def test_output_dir_must_be_repository_relative() -> None:
    result = run_cli(
        "--model-path",
        MODEL_PATH,
        "--calibration-tsv",
        CALIBRATION_TSV,
        "--output-dir",
        "D:/tmp/qwen35_2b_awq_w4a16",
        "--dry-run",
    )

    assert result.returncode != 0
    assert "repository-relative" in result.stderr


def test_output_dir_rejects_path_traversal() -> None:
    result = run_cli(
        "--model-path",
        MODEL_PATH,
        "--calibration-tsv",
        CALIBRATION_TSV,
        "--output-dir",
        "artifacts/quantized/qwen35_2b_awq_w4a16/../escape",
        "--dry-run",
    )

    assert result.returncode != 0
    assert "path traversal" in result.stderr


def test_drive_relative_paths_are_rejected() -> None:
    result = run_cli(
        "--model-path",
        MODEL_PATH,
        "--calibration-tsv",
        CALIBRATION_TSV,
        "--output-dir",
        "D:tmp/qwen35_2b_awq_w4a16",
        "--dry-run",
    )

    assert result.returncode != 0
    assert "repository-relative" in result.stderr


def test_missing_calibration_tsv_is_rejected() -> None:
    result = run_cli(
        "--model-path",
        MODEL_PATH,
        "--calibration-tsv",
        "resources/eval_dataset/raw/mmbench_public/missing.tsv",
        "--output-dir",
        OUTPUT_DIR,
        "--dry-run",
    )

    assert result.returncode != 0
    assert "calibration_tsv does not exist" in result.stderr
