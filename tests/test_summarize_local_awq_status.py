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
SCRIPT = ROOT_DIR / "scripts" / "summarize_local_awq_status.py"
TEMPLATE = ROOT_DIR / "reports" / "local_awq_summary_template.md"


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
    baseline_model_path: str,
    awq_artifact_path: str,
    samples_path: str,
    validation_scope: str = "local_smoke",
    performance_claim: str = "not_benchmarked",
    max_samples: int = 10,
) -> Path:
    path.write_text(
        "\n".join(
            [
                f"validation_scope: {validation_scope}",
                "description: Local AWQ smoke validation only; not an official benchmark.",
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


def test_missing_artifact_and_samples_reports_not_run(
    workspace_tmp_dir: Path,
) -> None:
    baseline = workspace_tmp_dir / "fake_baseline"
    baseline.mkdir()
    (baseline / "model.marker").write_text("baseline", encoding="utf-8")
    artifact = workspace_tmp_dir / "missing_awq_artifact"
    samples = workspace_tmp_dir / "missing_samples.jsonl"
    config = write_config(
        workspace_tmp_dir / "local_awq_smoke.yaml",
        baseline_model_path=repo_relative(baseline),
        awq_artifact_path=repo_relative(artifact),
        samples_path=repo_relative(samples),
    )

    result = run_cli("--config", repo_relative(config), "--dry-run")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["mode"] == "local_awq_status"
    assert payload["dry_run"] is True
    assert payload["baseline_exists"] is True
    assert payload["baseline_size_bytes"] > 0
    assert payload["awq_artifact_exists"] is False
    assert payload["awq_artifact_status"] == "artifact_missing"
    assert payload["awq_artifact_size_bytes"] is None
    assert payload["samples_status"] == "missing"
    assert payload["inspected_sample_count"] == 0
    assert payload["preflight_status"] == "not_provided"
    assert payload["can_run_local_comparison"] is False
    assert payload["would_load_model"] is False
    assert payload["would_run_inference"] is False
    assert payload["would_write_artifacts"] is False
    assert payload["uses_official_benchmark"] is False
    assert payload["performance_claim"] == "not_benchmarked"


def test_present_fake_inputs_allow_future_local_comparison(
    workspace_tmp_dir: Path,
) -> None:
    baseline = workspace_tmp_dir / "fake_baseline"
    artifact = workspace_tmp_dir / "fake_awq_artifact"
    baseline.mkdir()
    artifact.mkdir()
    (baseline / "model.marker").write_text("baseline", encoding="utf-8")
    (artifact / "awq.marker").write_text("artifact", encoding="utf-8")
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
        baseline_model_path=repo_relative(baseline),
        awq_artifact_path=repo_relative(artifact),
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
    assert payload["baseline_exists"] is True
    assert payload["awq_artifact_exists"] is True
    assert payload["samples_status"] == "present"
    assert payload["planned_max_samples"] == 1
    assert payload["inspected_sample_count"] == 1
    assert payload["can_run_local_comparison"] is True
    assert "future local BF16 vs AWQ smoke comparison" in " ".join(payload["next_steps"])


def test_non_dry_run_is_rejected() -> None:
    result = run_cli()

    assert result.returncode != 0
    assert "only supports dry-run/status summary" in result.stderr


def test_performance_claim_must_remain_not_benchmarked(
    workspace_tmp_dir: Path,
) -> None:
    config = write_config(
        workspace_tmp_dir / "bad_claim.yaml",
        baseline_model_path=repo_relative(workspace_tmp_dir / "fake_baseline"),
        awq_artifact_path=repo_relative(workspace_tmp_dir / "fake_artifact"),
        samples_path=repo_relative(workspace_tmp_dir / "samples.jsonl"),
        performance_claim="speedup_claimed",
    )

    result = run_cli("--config", repo_relative(config), "--dry-run")

    assert result.returncode != 0
    assert "performance_claim must be not_benchmarked" in result.stderr


def test_validation_scope_must_be_local_smoke(workspace_tmp_dir: Path) -> None:
    config = write_config(
        workspace_tmp_dir / "bad_scope.yaml",
        baseline_model_path=repo_relative(workspace_tmp_dir / "fake_baseline"),
        awq_artifact_path=repo_relative(workspace_tmp_dir / "fake_artifact"),
        samples_path=repo_relative(workspace_tmp_dir / "samples.jsonl"),
        validation_scope="official_benchmark",
    )

    result = run_cli("--config", repo_relative(config), "--dry-run")

    assert result.returncode != 0
    assert "validation_scope must be local_smoke" in result.stderr


def test_absolute_path_is_rejected(workspace_tmp_dir: Path) -> None:
    config = write_config(
        workspace_tmp_dir / "absolute_path.yaml",
        baseline_model_path="D:/models/Qwen3.5-2B",
        awq_artifact_path=repo_relative(workspace_tmp_dir / "fake_artifact"),
        samples_path=repo_relative(workspace_tmp_dir / "samples.jsonl"),
    )

    result = run_cli("--config", repo_relative(config), "--dry-run")

    assert result.returncode != 0
    assert "repository-relative" in result.stderr


def test_path_traversal_is_rejected(workspace_tmp_dir: Path) -> None:
    config = write_config(
        workspace_tmp_dir / "traversal.yaml",
        baseline_model_path=repo_relative(workspace_tmp_dir / "fake_baseline"),
        awq_artifact_path="artifacts/quantized/qwen35_2b_awq_w4a16/../escape",
        samples_path=repo_relative(workspace_tmp_dir / "samples.jsonl"),
    )

    result = run_cli("--config", repo_relative(config), "--dry-run")

    assert result.returncode != 0
    assert "path traversal" in result.stderr


def test_invalid_samples_jsonl_has_clear_error(workspace_tmp_dir: Path) -> None:
    samples = write_jsonl(workspace_tmp_dir / "samples.jsonl", ["{bad json"])
    config = write_config(
        workspace_tmp_dir / "invalid_samples.yaml",
        baseline_model_path=repo_relative(workspace_tmp_dir / "fake_baseline"),
        awq_artifact_path=repo_relative(workspace_tmp_dir / "fake_artifact"),
        samples_path=repo_relative(samples),
    )

    result = run_cli("--config", repo_relative(config), "--dry-run")

    assert result.returncode != 0
    assert "invalid JSON" in result.stderr


def test_optional_preflight_json_is_loaded(workspace_tmp_dir: Path) -> None:
    preflight = workspace_tmp_dir / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "mode": "preflight",
                "can_execute": False,
                "performance_claim": "not_benchmarked",
            }
        ),
        encoding="utf-8",
    )
    config = write_config(
        workspace_tmp_dir / "local_awq_smoke.yaml",
        baseline_model_path=repo_relative(workspace_tmp_dir / "fake_baseline"),
        awq_artifact_path=repo_relative(workspace_tmp_dir / "fake_artifact"),
        samples_path=repo_relative(workspace_tmp_dir / "samples.jsonl"),
    )

    result = run_cli(
        "--config",
        repo_relative(config),
        "--preflight-json",
        repo_relative(preflight),
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["preflight_status"] == "loaded"
    assert payload["preflight_summary"]["mode"] == "preflight"
    assert payload["preflight_summary"]["can_execute"] is False
    assert payload["preflight_summary"]["performance_claim"] == "not_benchmarked"


def test_template_markdown_sets_reporting_boundaries() -> None:
    assert TEMPLATE.exists()

    text = TEMPLATE.read_text(encoding="utf-8").lower()

    assert "not an official benchmark" in text
    assert "must not claim performance gains" in text
    assert "only after a real local bf16 vs awq smoke run" in text
    assert "artifacts must not be committed to git" in text
    assert "official evaluation remains separate" in text
