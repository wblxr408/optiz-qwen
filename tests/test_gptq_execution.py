from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace


COMPRESSION_DIR = (
    Path(__file__).resolve().parents[1] / "src/optiz_qwen/compression"
)
TEST_PACKAGE = "_gptq_execution_test_package"
package = ModuleType(TEST_PACKAGE)
package.__path__ = [str(COMPRESSION_DIR)]
sys.modules[TEST_PACKAGE] = package

MODULE_NAME = f"{TEST_PACKAGE}.gptq_execution"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME,
    COMPRESSION_DIR / "gptq_execution.py",
)
assert SPEC is not None and SPEC.loader is not None
gptq_execution = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = gptq_execution
SPEC.loader.exec_module(gptq_execution)


def _fake_torch(
    *,
    cuda_available: bool,
    mem_get_info,
    get_memory_info=None,
) -> SimpleNamespace:
    accelerator = SimpleNamespace(
        current_accelerator=lambda: SimpleNamespace(type="cuda"),
    )
    if get_memory_info is not None:
        accelerator.get_memory_info = get_memory_info

    return SimpleNamespace(
        accelerator=accelerator,
        cuda=SimpleNamespace(
            is_available=lambda: cuda_available,
            mem_get_info=mem_get_info,
        ),
    )


class TorchAcceleratorMemoryInfoCompatTests(unittest.TestCase):
    def test_existing_accelerator_memory_info_is_not_overwritten(self) -> None:
        original = lambda device_index=None: (1, 2)
        torch_module = _fake_torch(
            cuda_available=True,
            mem_get_info=lambda device_index=None: (3, 4),
            get_memory_info=original,
        )

        installed = gptq_execution._install_torch_accelerator_memory_info_compat(
            torch_module
        )

        self.assertFalse(installed)
        self.assertIs(torch_module.accelerator.get_memory_info, original)

    def test_missing_accelerator_memory_info_installs_cuda_fallback(self) -> None:
        torch_module = _fake_torch(
            cuda_available=True,
            mem_get_info=lambda device_index=None: (5, 6),
        )

        installed = gptq_execution._install_torch_accelerator_memory_info_compat(
            torch_module
        )

        self.assertTrue(installed)
        self.assertTrue(callable(torch_module.accelerator.get_memory_info))

    def test_fallback_forwards_argument_and_result(self) -> None:
        calls = []

        def mem_get_info(device_index=None):
            calls.append(device_index)
            return (97_633_383_408, 102_676_561_920)

        torch_module = _fake_torch(
            cuda_available=True,
            mem_get_info=mem_get_info,
        )
        self.assertTrue(
            gptq_execution._install_torch_accelerator_memory_info_compat(torch_module)
        )

        expected = (97_633_383_408, 102_676_561_920)
        self.assertEqual(torch_module.accelerator.get_memory_info(0), expected)
        self.assertEqual(torch_module.accelerator.get_memory_info(None), expected)
        self.assertEqual(calls, [0, None])

    def test_unavailable_cuda_does_not_install_fallback(self) -> None:
        torch_module = _fake_torch(
            cuda_available=False,
            mem_get_info=lambda device_index=None: (7, 8),
        )

        installed = gptq_execution._install_torch_accelerator_memory_info_compat(
            torch_module
        )

        self.assertFalse(installed)
        self.assertFalse(hasattr(torch_module.accelerator, "get_memory_info"))


if __name__ == "__main__":
    unittest.main()
