from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/check_awq_cuda_readiness.py"
SPEC = importlib.util.spec_from_file_location("check_awq_cuda_readiness", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
readiness = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = readiness
SPEC.loader.exec_module(readiness)


def config(*, execution_enabled: bool = False) -> dict:
    return {
        "execution_enabled": execution_enabled,
        "environment": {
            "python": "3.12",
            "torch": "2.10.0+cu128",
            "torchvision": "0.25.0+cu128",
            "transformers": "5.10.1",
            "llmcompressor": "0.12.0",
        },
    }


def packages() -> dict:
    return {
        "torch": {"installed": True, "version": "2.10.0+cu128"},
        "torchvision": {"installed": True, "version": "0.25.0+cu128"},
        "transformers": {"installed": True, "version": "5.10.1"},
        "llmcompressor": {"installed": True, "version": "0.12.0"},
        "compressed-tensors": {"installed": True, "version": "0.13.0"},
    }


class AWQCudaReadinessTests(unittest.TestCase):
    def test_local_version_requires_exact_build_when_expected_has_build(self) -> None:
        self.assertTrue(readiness.version_matches("2.10.0+cu128", "2.10.0+cu128"))
        self.assertFalse(readiness.version_matches("2.10.0+cu126", "2.10.0+cu128"))
        self.assertTrue(readiness.version_matches("5.10.1", "5.10.1"))

    @mock.patch.object(readiness.platform, "python_version", return_value="3.12.11")
    def test_ready_environment_is_not_authorized_while_switch_is_off(self, _mock) -> None:
        report = readiness.summarize_readiness(
            config=config(execution_enabled=False),
            packages=packages(),
            symbols={"awq": {"available": True}},
            cuda={"available": True, "bf16_supported": True},
            model_exists=True,
            dataset_exists=True,
        )

        self.assertTrue(report["environment_ready"])
        self.assertFalse(report["execution_enabled"])
        self.assertFalse(report["quantization_authorized"])

    @mock.patch.object(readiness.platform, "python_version", return_value="3.12.11")
    def test_missing_symbol_blocks_readiness(self, _mock) -> None:
        report = readiness.summarize_readiness(
            config=config(execution_enabled=True),
            packages=packages(),
            symbols={"awq": {"available": False}},
            cuda={"available": True, "bf16_supported": True},
            model_exists=True,
            dataset_exists=True,
        )

        self.assertFalse(report["checks"]["required_symbols"])
        self.assertFalse(report["environment_ready"])
        self.assertFalse(report["quantization_authorized"])


if __name__ == "__main__":
    unittest.main()
