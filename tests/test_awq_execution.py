from __future__ import annotations

import sys
import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from optiz_qwen.compression.awq_backend import AWQBackendReadiness
from optiz_qwen.compression import awq_execution

HEAVY_MODULES = {"torch", "transformers", "awq", "autoawq"}
ROOT_DIR = Path(__file__).resolve().parents[1]


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


def available_readiness() -> AWQBackendReadiness:
    return AWQBackendReadiness(
        backend_name="autoawq",
        package_available=True,
        can_quantize=True,
        reason="test backend",
        recommended_environment="test environment",
    )


def unavailable_readiness() -> AWQBackendReadiness:
    return AWQBackendReadiness(
        backend_name="autoawq",
        package_available=False,
        can_quantize=False,
        reason="missing backend",
        recommended_environment="test environment",
    )


def test_module_import_does_not_import_heavy_dependencies() -> None:
    before = {name for name in HEAVY_MODULES if name in sys.modules}

    __import__("optiz_qwen.compression.awq_execution")

    after = {name for name in HEAVY_MODULES if name in sys.modules}
    assert after == before


def test_execute_requires_artifact_write_confirmation(workspace_tmp_dir: Path) -> None:
    with pytest.raises(RuntimeError, match="confirm-write-artifacts"):
        awq_execution.execute_autoawq_quantization(
            model_path=workspace_tmp_dir / "model",
            calibration_tsv=workspace_tmp_dir / "calibration.tsv",
            output_dir=workspace_tmp_dir / "out",
            num_calibration_samples=1,
            weight_bits=4,
            group_size=128,
            activation_dtype="bf16",
            zero_point=True,
            confirm_write_artifacts=False,
            backend_readiness=available_readiness(),
        )


def test_execute_rejects_unavailable_backend_before_heavy_imports(
    workspace_tmp_dir: Path,
) -> None:
    before = {name for name in HEAVY_MODULES if name in sys.modules}

    with pytest.raises(RuntimeError, match="AutoAWQ backend is not available"):
        awq_execution.execute_autoawq_quantization(
            model_path=workspace_tmp_dir / "model",
            calibration_tsv=workspace_tmp_dir / "calibration.tsv",
            output_dir=workspace_tmp_dir / "out",
            num_calibration_samples=1,
            weight_bits=4,
            group_size=128,
            activation_dtype="bf16",
            zero_point=True,
            confirm_write_artifacts=True,
            backend_readiness=unavailable_readiness(),
        )

    after = {name for name in HEAVY_MODULES if name in sys.modules}
    assert after == before


def test_heavy_interface_loader_is_called_only_on_confirmed_available_execute(
    monkeypatch: pytest.MonkeyPatch,
    workspace_tmp_dir: Path,
) -> None:
    calls: list[str] = []

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, *_args: object, **_kwargs: object) -> "FakeTokenizer":
            calls.append("tokenizer.from_pretrained")
            return cls()

        def save_pretrained(self, _path: str) -> None:
            calls.append("tokenizer.save_pretrained")

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *_args: object, **_kwargs: object) -> "FakeModel":
            calls.append("model.from_pretrained")
            return cls()

        def quantize(self, *_args: object, **_kwargs: object) -> None:
            calls.append("model.quantize")

        def save_quantized(self, _path: str) -> None:
            calls.append("model.save_quantized")

    def fake_loader() -> tuple[type[FakeModel], type[FakeTokenizer]]:
        calls.append("heavy_loader")
        return FakeModel, FakeTokenizer

    monkeypatch.setattr(awq_execution, "_load_autoawq_interfaces", fake_loader)
    monkeypatch.setattr(
        awq_execution,
        "load_mmbench_calibration_records",
        lambda *_args, **_kwargs: [SimpleNamespace(prompt_text="calibration prompt")],
    )

    summary = awq_execution.execute_autoawq_quantization(
        model_path=workspace_tmp_dir / "model",
        calibration_tsv=workspace_tmp_dir / "calibration.tsv",
        output_dir=workspace_tmp_dir / "out",
        num_calibration_samples=1,
        weight_bits=4,
        group_size=128,
        activation_dtype="bf16",
        zero_point=True,
        confirm_write_artifacts=True,
        backend_readiness=available_readiness(),
    )

    assert calls == [
        "heavy_loader",
        "tokenizer.from_pretrained",
        "model.from_pretrained",
        "model.quantize",
        "model.save_quantized",
        "tokenizer.save_pretrained",
    ]
    assert summary["mode"] == "execute"
    assert summary["writes_artifacts"] is True
    assert summary["performance_claim"] == "not_benchmarked"
