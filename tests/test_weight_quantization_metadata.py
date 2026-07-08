from __future__ import annotations

import pytest

from optiz_qwen.compression.metadata import (
    DEFAULT_AWQ_ARTIFACT_PATH,
    QuantizedArtifactMetadata,
    WeightQuantizationConfig,
    awq_w4a16_config,
    baseline_bf16_config,
    planned_awq_artifact_metadata,
)


def test_baseline_bf16_config_keeps_quantization_disabled() -> None:
    config = baseline_bf16_config()

    assert config.quantization == "none"
    assert config.activation_dtype == "bf16"
    assert config.weight_bits is None
    assert config.group_size is None
    assert config.quantized_artifact_path is None
    assert config.performance_claim == "not_benchmarked"


def test_awq_w4a16_config_records_planned_weight_only_path() -> None:
    config = awq_w4a16_config()

    assert config.quantization == "awq"
    assert config.weight_bits == 4
    assert config.activation_dtype == "bf16"
    assert config.group_size == 128
    assert config.zero_point is True
    assert config.quantized_artifact_path == DEFAULT_AWQ_ARTIFACT_PATH
    assert config.performance_claim == "not_benchmarked"


def test_planned_awq_metadata_does_not_claim_benchmark_results() -> None:
    metadata = planned_awq_artifact_metadata()

    assert metadata.status == "planned"
    assert metadata.generated_by_command is None
    assert metadata.benchmark_summary_path is None
    assert metadata.performance_claim == "not_benchmarked"


def test_awq_config_rejects_non_w4_weight_bits() -> None:
    config = WeightQuantizationConfig(
        name="qwen35_2b_awq_w8a16_placeholder",
        quantization="awq",
        weight_bits=8,
        activation_dtype="bf16",
        group_size=128,
        zero_point=True,
        quantized_artifact_path="artifacts/quantized/qwen35_2b_awq_w8a16",
    )

    with pytest.raises(ValueError, match="W4A16"):
        config.validate()


def test_absolute_paths_are_rejected() -> None:
    config = WeightQuantizationConfig(
        name="bad_absolute_path",
        quantization="none",
        base_model_path="D:/models/Qwen3.5-2B",
    )

    with pytest.raises(ValueError, match="repository-relative"):
        config.validate()


def test_performance_claims_stay_disabled_without_real_benchmark() -> None:
    metadata = QuantizedArtifactMetadata(
        quantization_config=awq_w4a16_config(),
        performance_claim="faster_than_baseline",
    )

    with pytest.raises(ValueError, match="not_benchmarked"):
        metadata.validate()
