import importlib.util
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
