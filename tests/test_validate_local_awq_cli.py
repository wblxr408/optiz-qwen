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
SCRIPT = ROOT_DIR / "scripts" / "validate_local_awq.py"


@pytest.fixture()
def workspace_tmp_dir() -> Iterator[Path]:
    root = ROOT_DIR / ".pytest-workspace-tmp"
    path = root / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        try:
            root.rmdir()
        except OSError:
            pass


def repo_relative(path: Path) -> str:
    return path.relative_to(ROOT_DIR).as_posix()


def write_config(
    path: Path,
    *,
    samples_path: str,
    validation_scope: str = "local_smoke",
    performance_claim: str = "not_benchmarked",
    baseline_model_path: str = "resources/model_weights/raw/Qwen3.5-2B",
    awq_artifact_path: str = "artifacts/quantized/qwen35_2b_awq_w4a16",
    max_samples: int = 10,
) -> Path:
    path.write_text(
        "\n".join(
            [
                f"validation_scope: {validation_scope}",
                f"baseline_model_path: {baseline_model_path}",
                f"awq_artifact_path: {awq_artifact_path}",
                f"local_validation_data: {samples_path}",
                "quantization: awq_w4a16",
                f"performance_claim: {performance_claim}",
                f"max_samples: {max_samples}",
                "max_new_tokens: 64",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def write_jsonl(path: Path, records: list[dict[str, str]] | list[str]) -> Path:
    lines = [
        record if isinstance(record, str) else json.dumps(record, ensure_ascii=False)
        for record in records
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT_DIR,
        text=True,
        capture_output=True,
        check=False,
    )


def test_dry_run_succeeds_when_samples_are_missing(workspace_tmp_dir: Path) -> None:
    samples = workspace_tmp_dir / "missing_samples.jsonl"
    config = write_config(
        workspace_tmp_dir / "local_awq_smoke.yaml",
        samples_path=repo_relative(samples),
    )

    result = run_cli("--config", repo_relative(config), "--dry-run")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["validation_scope"] == "local_smoke"
    assert payload["dry_run"] is True
    assert payload["samples_status"] == "missing"
    assert payload["planned_status"] == "samples_missing"
    assert payload["planned_max_samples"] == 10
    assert payload["loads_model"] is False
    assert payload["writes_artifacts"] is False
    assert payload["uses_official_dataset"] is False
    assert payload["performance_claim"] == "not_benchmarked"


def test_dry_run_validates_present_samples_and_honors_max_override(
    workspace_tmp_dir: Path,
) -> None:
    samples = write_jsonl(
        workspace_tmp_dir / "samples.jsonl",
        [
            {
                "sample_id": "local-001",
                "image_path": "resources/local_validation/images/local-001.png",
                "question": "What text is visible?",
                "reference_answer": "STOP",
                "category": "ocr",
            },
            {
                "sample_id": "local-002",
                "image_path": "resources/local_validation/images/local-002.png",
                "question": "Where is the object?",
                "expected_behavior": "Mentions the upper-left area.",
            },
        ],
    )
    config = write_config(
        workspace_tmp_dir / "local_awq_smoke.yaml",
        samples_path=repo_relative(samples),
    )

    result = run_cli(
        "--config",
        repo_relative(config),
        "--max-samples",
        "1",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["samples_status"] == "present"
    assert payload["planned_status"] == "samples_present"
    assert payload["planned_max_samples"] == 1
    assert payload["validated_sample_count"] == 1


def test_non_dry_run_is_rejected() -> None:
    result = run_cli()

    assert result.returncode != 0
    assert "only supports dry-run" in result.stderr
    assert "real local inference is not implemented" in result.stderr


def test_validation_scope_must_be_local_smoke(workspace_tmp_dir: Path) -> None:
    config = write_config(
        workspace_tmp_dir / "bad_scope.yaml",
        samples_path=repo_relative(workspace_tmp_dir / "samples.jsonl"),
        validation_scope="official_benchmark",
    )

    result = run_cli("--config", repo_relative(config), "--dry-run")

    assert result.returncode != 0
    assert "validation_scope must be local_smoke" in result.stderr


def test_performance_claim_must_remain_not_benchmarked(workspace_tmp_dir: Path) -> None:
    config = write_config(
        workspace_tmp_dir / "bad_claim.yaml",
        samples_path=repo_relative(workspace_tmp_dir / "samples.jsonl"),
        performance_claim="faster_than_baseline",
    )

    result = run_cli("--config", repo_relative(config), "--dry-run")

    assert result.returncode != 0
    assert "performance_claim must be not_benchmarked" in result.stderr


def test_absolute_config_path_is_rejected(workspace_tmp_dir: Path) -> None:
    config = write_config(
        workspace_tmp_dir / "local_awq_smoke.yaml",
        samples_path=repo_relative(workspace_tmp_dir / "samples.jsonl"),
    )

    result = run_cli("--config", str(config.resolve()), "--dry-run")

    assert result.returncode != 0
    assert "repository-relative" in result.stderr


def test_config_path_traversal_is_rejected() -> None:
    result = run_cli("--config", "configs/experiments/../local_awq_smoke.yaml", "--dry-run")

    assert result.returncode != 0
    assert "path traversal" in result.stderr


def test_jsonl_invalid_json_has_clear_error(workspace_tmp_dir: Path) -> None:
    samples = write_jsonl(workspace_tmp_dir / "samples.jsonl", ["{bad json"])
    config = write_config(
        workspace_tmp_dir / "local_awq_smoke.yaml",
        samples_path=repo_relative(samples),
    )

    result = run_cli("--config", repo_relative(config), "--dry-run")

    assert result.returncode != 0
    assert "invalid JSON" in result.stderr


def test_jsonl_missing_question_has_clear_error(workspace_tmp_dir: Path) -> None:
    samples = write_jsonl(
        workspace_tmp_dir / "samples.jsonl",
        [
            {
                "sample_id": "local-001",
                "image_path": "resources/local_validation/images/local-001.png",
                "reference_answer": "A",
            }
        ],
    )
    config = write_config(
        workspace_tmp_dir / "local_awq_smoke.yaml",
        samples_path=repo_relative(samples),
    )

    result = run_cli("--config", repo_relative(config), "--dry-run")

    assert result.returncode != 0
    assert "question" in result.stderr


def test_jsonl_requires_reference_answer_or_expected_behavior(
    workspace_tmp_dir: Path,
) -> None:
    samples = write_jsonl(
        workspace_tmp_dir / "samples.jsonl",
        [
            {
                "sample_id": "local-001",
                "image_path": "resources/local_validation/images/local-001.png",
                "question": "What should happen?",
            }
        ],
    )
    config = write_config(
        workspace_tmp_dir / "local_awq_smoke.yaml",
        samples_path=repo_relative(samples),
    )

    result = run_cli("--config", repo_relative(config), "--dry-run")

    assert result.returncode != 0
    assert "reference_answer or expected_behavior" in result.stderr
