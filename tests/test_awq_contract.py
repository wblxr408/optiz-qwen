from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/awq_contract.py"
SPEC = importlib.util.spec_from_file_location("awq_contract", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
contract = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = contract
SPEC.loader.exec_module(contract)


class AWQContractTests(unittest.TestCase):
    def test_execution_authorization_does_not_change_scientific_identity(self) -> None:
        disabled = {"execution_enabled": False, "algorithm": "AWQ", "seed": 7}
        enabled = {"execution_enabled": True, "algorithm": "AWQ", "seed": 7}

        self.assertEqual(
            contract.semantic_config_sha256(disabled),
            contract.semantic_config_sha256(enabled),
        )

    def test_scientific_parameter_change_changes_identity(self) -> None:
        first = {"execution_enabled": False, "algorithm": "AWQ", "seed": 7}
        second = {"execution_enabled": False, "algorithm": "AWQ", "seed": 8}

        self.assertNotEqual(
            contract.semantic_config_sha256(first),
            contract.semantic_config_sha256(second),
        )


if __name__ == "__main__":
    unittest.main()
