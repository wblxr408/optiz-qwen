"""Declarative metadata for weight quantization.

This module does not quantize, download, load, or benchmark a model. It only
records the BF16 control path and the planned AWQ W4A16 artifact contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PureWindowsPath
from typing import Literal

QuantizationMode = Literal["none", "awq"]
ArtifactStatus = Literal["planned", "generated", "validated"]

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-2B"
DEFAULT_BASE_MODEL_PATH = "resources/model_weights/raw/Qwen3.5-2B"
DEFAULT_AWQ_ARTIFACT_PATH = "artifacts/quantized/qwen35_2b_awq_w4a16"
DEFAULT_CALIBRATION_DATASET_PATH = "resources/eval_dataset/raw/mmbench_public/"


def _check_relative_path(value: str | None, *, field_name: str) -> None:
    if value is None:
        return
    if PureWindowsPath(value).is_absolute() or value.startswith("/"):
        raise ValueError(f"{field_name} must be repository-relative, got: {value}")


@dataclass(frozen=True)
class WeightQuantizationConfig:
    """Static config for either BF16 baseline or planned AWQ W4A16."""

    name: str
    quantization: QuantizationMode
    model_id: str = DEFAULT_MODEL_ID
    base_model_path: str = DEFAULT_BASE_MODEL_PATH
    activation_dtype: str = "bf16"
    weight_bits: int | None = None
    group_size: int | None = None
    zero_point: bool | None = None
    quantized_artifact_path: str | None = None
    calibration_dataset_path: str | None = None
    calibration_sample_count: int | None = None
    performance_claim: str = "not_benchmarked"

    def validate(self) -> None:
        """Validate static metadata without touching model files."""

        _check_relative_path(self.base_model_path, field_name="base_model_path")
        _check_relative_path(
            self.quantized_artifact_path,
            field_name="quantized_artifact_path",
        )
        _check_relative_path(
            self.calibration_dataset_path,
            field_name="calibration_dataset_path",
        )
        if self.performance_claim != "not_benchmarked":
            raise ValueError(
                "performance_claim must remain 'not_benchmarked' until a real "
                "baseline-vs-AWQ benchmark summary exists"
            )
        if self.quantization == "none":
            self._validate_baseline()
            return
        if self.quantization == "awq":
            self._validate_awq_w4a16()
            return
        raise ValueError(f"unsupported quantization mode: {self.quantization}")

    def _validate_baseline(self) -> None:
        if self.weight_bits is not None:
            raise ValueError("BF16 baseline must not set weight_bits")
        if self.group_size is not None:
            raise ValueError("BF16 baseline must not set group_size")
        if self.quantized_artifact_path is not None:
            raise ValueError("BF16 baseline must not set quantized_artifact_path")

    def _validate_awq_w4a16(self) -> None:
        if self.weight_bits != 4:
            raise ValueError("AWQ main path must use W4A16, so weight_bits must be 4")
        if self.activation_dtype.lower() not in {"bf16", "fp16", "float16"}:
            raise ValueError("AWQ W4A16 requires a 16-bit activation dtype")
        if self.group_size is None or self.group_size <= 0:
            raise ValueError("AWQ W4A16 requires a positive group_size")
        if self.zero_point is None:
            raise ValueError("AWQ W4A16 must explicitly record zero_point")
        if self.quantized_artifact_path is None:
            raise ValueError("AWQ W4A16 requires a planned quantized_artifact_path")


@dataclass(frozen=True)
class QuantizedArtifactMetadata:
    """Metadata contract for a future generated AWQ artifact."""

    quantization_config: WeightQuantizationConfig
    artifact_path: str = DEFAULT_AWQ_ARTIFACT_PATH
    source_model_path: str = DEFAULT_BASE_MODEL_PATH
    status: ArtifactStatus = "planned"
    generated_by_command: str | None = None
    generated_at: str | None = None
    benchmark_summary_path: str | None = None
    performance_claim: str = "not_benchmarked"

    def validate(self) -> None:
        """Validate artifact metadata without requiring the artifact to exist."""

        self.quantization_config.validate()
        _check_relative_path(self.artifact_path, field_name="artifact_path")
        _check_relative_path(self.source_model_path, field_name="source_model_path")
        _check_relative_path(
            self.benchmark_summary_path,
            field_name="benchmark_summary_path",
        )
        if self.quantization_config.quantization != "awq":
            raise ValueError("quantized artifact metadata requires quantization='awq'")
        if self.status == "planned" and self.generated_by_command is not None:
            raise ValueError("planned artifacts must not record generated_by_command")
        if self.performance_claim != "not_benchmarked":
            raise ValueError(
                "artifact performance_claim must remain 'not_benchmarked' until "
                "a real benchmark summary is available"
            )


def baseline_bf16_config() -> WeightQuantizationConfig:
    """Return the required BF16 control-path config."""

    config = WeightQuantizationConfig(
        name="qwen35_2b_baseline_bf16",
        quantization="none",
        activation_dtype="bf16",
    )
    config.validate()
    return config


def awq_w4a16_config() -> WeightQuantizationConfig:
    """Return the planned AWQ W4A16 weight-only config."""

    config = WeightQuantizationConfig(
        name="qwen35_2b_awq_w4a16",
        quantization="awq",
        activation_dtype="bf16",
        weight_bits=4,
        group_size=128,
        zero_point=True,
        quantized_artifact_path=DEFAULT_AWQ_ARTIFACT_PATH,
        calibration_dataset_path=DEFAULT_CALIBRATION_DATASET_PATH,
        calibration_sample_count=None,
    )
    config.validate()
    return config


def planned_awq_artifact_metadata() -> QuantizedArtifactMetadata:
    """Return metadata for the planned AWQ artifact before generation."""

    metadata = QuantizedArtifactMetadata(quantization_config=awq_w4a16_config())
    metadata.validate()
    return metadata
