from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/inventory_awq_modules.py"
SPEC = importlib.util.spec_from_file_location("inventory_awq_modules", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
inventory = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = inventory
SPEC.loader.exec_module(inventory)


class AWQModuleInventoryTests(unittest.TestCase):
    def test_semantic_groups_cover_qwen35_linear_paths(self) -> None:
        cases = {
            "model.visual.blocks.0.attn.qkv": "vision_encoder",
            "model.visual.merger.linear_fc1": "multimodal_projector",
            "model.language_model.layers.0.linear_attn.in_proj_qkv": (
                "language_gated_deltanet"
            ),
            "model.language_model.layers.3.self_attn.q_proj": "language_attention",
            "model.language_model.layers.0.mlp.gate_proj": "language_mlp",
            "lm_head": "lm_head",
        }
        self.assertEqual(
            {name: inventory.classify_linear(name) for name in cases}, cases
        )

    def test_conservative_candidate_excludes_sensitive_groups(self) -> None:
        names = [
            "model.visual.blocks.0.attn.qkv",
            "model.visual.merger.linear_fc1",
            "model.language_model.layers.0.linear_attn.in_proj_qkv",
            "model.language_model.layers.3.self_attn.q_proj",
            "model.language_model.layers.0.mlp.gate_proj",
            "lm_head",
        ]
        result = inventory.partition_linear_names(names)
        candidate = result["conservative_candidate"]

        self.assertEqual(candidate["selected_count"], 2)
        self.assertEqual(candidate["ignored_count"], 4)
        self.assertIn(
            "model.language_model.layers.3.self_attn.q_proj",
            candidate["selected_names"],
        )
        self.assertIn(
            "model.language_model.layers.0.linear_attn.in_proj_qkv",
            candidate["ignored_names"],
        )

    def test_unclassified_linear_blocks_policy_resolution(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unclassified"):
            inventory.partition_linear_names(["unexpected.linear"])


if __name__ == "__main__":
    unittest.main()
