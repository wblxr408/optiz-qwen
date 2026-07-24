import importlib.util
import json
import math
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_v11_cuda_matrix.py"
SPEC = importlib.util.spec_from_file_location("run_v11_cuda_matrix", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _result(ttft: float, throughput: float, accuracy: float) -> dict:
    return {
        "sample_count": 10,
        "performance": {
            "avg_ttft_ms": ttft,
            "avg_throughput_tokens_per_sec": throughput,
        },
        "accuracy": {"score": accuracy},
    }


def test_summary_reports_relative_changes() -> None:
    summary = MODULE.summarize_results(
        {
            "baseline": [
                _result(100.0, 50.0, 0.8),
                _result(100.0, 50.0, 0.8),
            ],
            "gdn-fast": [
                _result(50.0, 60.0, 0.8),
                _result(50.0, 60.0, 0.8),
            ],
        }
    )

    assert math.isclose(
        summary["gdn-fast"]["relative_to_baseline"]["ttft_percent"], -50.0
    )
    assert math.isclose(
        summary["gdn-fast"]["relative_to_baseline"]["throughput_percent"], 20.0
    )
    assert math.isclose(
        summary["gdn-fast"]["relative_to_baseline"]["accuracy_percentage_points"],
        0.0,
    )


def test_gdn_environment_does_not_leak_into_baseline(tmp_path: Path) -> None:
    code_root = tmp_path / "code"
    overlay = tmp_path / "overlay"
    (code_root / "src").mkdir(parents=True)
    overlay.mkdir()

    baseline = MODULE.CaseSpec("baseline", tmp_path / "model", False)
    fast = MODULE.CaseSpec("gdn-fast", tmp_path / "model", True)
    baseline_env = MODULE.build_environment(
        code_root=code_root,
        spec=baseline,
        gdn_overlay=overlay,
        seed=7,
        cuda_visible_devices="0",
    )
    fast_env = MODULE.build_environment(
        code_root=code_root,
        spec=fast,
        gdn_overlay=overlay,
        seed=7,
        cuda_visible_devices="0",
    )

    assert baseline_env["PYTHONPATH"] == str((code_root / "src").resolve())
    assert fast_env["PYTHONPATH"].split(MODULE.os.pathsep)[0] == str(overlay.resolve())


def test_python_launcher_symlink_is_preserved(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    target = tmp_path / "base-python"
    launcher = tmp_path / "venv-python"
    target.write_text("", encoding="utf-8")
    try:
        launcher.symlink_to(target)
    except OSError:
        return

    assert MODULE.executable_path(launcher) == launcher.absolute()
    assert MODULE.executable_path(launcher) != launcher.resolve()


def _switch_args(*extra: str):
    return MODULE.parse_args(
        [
            "--dataset-path",
            "dataset.tsv",
            "--baseline-model-path",
            "model",
            "--output-root",
            "output",
            *extra,
        ]
    )


def test_default_switches_select_only_baseline() -> None:
    args = _switch_args()

    names, mode = MODULE.resolve_case_names(args)
    [spec] = MODULE.resolve_cases(args)

    assert names == ["baseline"]
    assert mode == "default-off-switches"
    assert spec.name == "baseline"
    assert spec.enable_gdn_fastpath is False


def test_awq_and_gdn_switches_select_combined_case() -> None:
    args = _switch_args(
        "--awq-model-path",
        "awq-model",
        "--gdn-overlay",
        "gdn-overlay",
        "--enable-awq",
        "--enable-gdn-fastpath",
    )

    [spec] = MODULE.resolve_cases(args)

    assert spec.name == "awq-gdn"
    assert spec.model_path == Path("awq-model")
    assert spec.enable_gdn_fastpath is True


def test_explicit_cases_cannot_be_mixed_with_default_off_switches() -> None:
    import pytest

    args = _switch_args("--cases", "baseline", "--enable-awq")

    with pytest.raises(ValueError, match="cannot be combined"):
        MODULE.resolve_case_names(args)


def test_awq_artifact_requires_compressed_w4a16_metadata(tmp_path: Path) -> None:
    model = tmp_path / "awq"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"weights")
    (model / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "quantization_status": "compressed",
                    "format": "pack-quantized",
                    "config_groups": {
                        "group_0": {
                            "targets": ["model.layer.proj"],
                            "weights": {"num_bits": 4},
                            "input_activations": None,
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    report = MODULE.inspect_awq_artifact(model)

    assert report["ready"] is True
    assert report["valid_w4a16_group_count"] == 1
    assert report["target_count"] == 1


def test_gdn_probe_validation_rejects_contaminated_baseline() -> None:
    import pytest

    baseline = {
        "report": {
            "transformers_fastpath": {"is_fast_path_available": True}
        }
    }

    with pytest.raises(RuntimeError, match="baseline environment"):
        MODULE.validate_gdn_probes(
            baseline_probe=baseline,
            enabled_probe=None,
        )
