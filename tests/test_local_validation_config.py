from __future__ import annotations

from pathlib import Path, PureWindowsPath

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "configs" / "experiments" / "local_awq_smoke.yaml"
README_PATH = ROOT_DIR / "resources" / "local_validation" / "README.md"


def parse_simple_yaml_scalars(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if not line or line.startswith(" ") or line.startswith("#") or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        value = raw_value.strip().strip('"').strip("'")
        values[key.strip()] = value
    return values


def is_repo_relative_path(value: str) -> bool:
    normalized = value.replace("\\", "/").strip("/")
    return (
        bool(normalized)
        and not value.startswith("/")
        and not PureWindowsPath(value).is_absolute()
        and not PureWindowsPath(value).drive
        and ".." not in Path(normalized).parts
    )


def test_local_awq_smoke_config_exists_and_has_required_values() -> None:
    assert CONFIG_PATH.exists()

    config = parse_simple_yaml_scalars(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["validation_scope"] == "local_smoke"
    assert config["baseline_model_path"] == "resources/model_weights/raw/Qwen3.5-2B"
    assert config["awq_artifact_path"] == "artifacts/quantized/qwen35_2b_awq_w4a16"
    assert config["local_validation_data"] == "resources/local_validation/samples.jsonl"
    assert config["quantization"] == "awq_w4a16"
    assert config["performance_claim"] == "not_benchmarked"
    assert config["max_samples"] == "10"
    assert config["max_new_tokens"] == "64"


def test_config_paths_are_repository_relative() -> None:
    config = parse_simple_yaml_scalars(CONFIG_PATH.read_text(encoding="utf-8"))

    for key in ["baseline_model_path", "awq_artifact_path", "local_validation_data"]:
        assert is_repo_relative_path(config[key]), key


def test_readme_exists_and_defines_samples_jsonl_format() -> None:
    assert README_PATH.exists()

    text = README_PATH.read_text(encoding="utf-8").lower()

    assert "samples.jsonl" in text
    for field in [
        "sample_id",
        "image_path",
        "question",
        "reference_answer",
        "expected_behavior",
        "category",
        "notes",
    ]:
        assert field in text


def test_readme_sets_local_smoke_boundaries() -> None:
    text = README_PATH.read_text(encoding="utf-8").lower()

    assert "official competition evaluation dataset" in text
    assert "not an official benchmark" in text
    assert "do not report local smoke results as an official benchmark" in text
    assert "smoke, regression, and sanity checks" in text


def test_config_and_readme_do_not_claim_performance_gains() -> None:
    combined = (
        CONFIG_PATH.read_text(encoding="utf-8")
        + "\n"
        + README_PATH.read_text(encoding="utf-8")
    ).lower()

    assert "performance_claim: not_benchmarked" in combined
    forbidden_claims = [
        "faster_than_baseline",
        "ttft gain",
        "throughput gain",
        "accuracy gain",
        "latency reduction",
        "official score",
        "speedup",
    ]
    for phrase in forbidden_claims:
        assert phrase not in combined
