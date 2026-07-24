from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/quantize_awq_w4a16.py"
SPEC = importlib.util.spec_from_file_location("quantize_awq_w4a16", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
driver = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = driver
SPEC.loader.exec_module(driver)


class AWQDriverTests(unittest.TestCase):
    def contract_fixture(self, *, execution_enabled: bool = False):
        selected = ["model.language_model.layers.0.mlp.gate_proj"]
        config = {
            "experiment_id": "test",
            "execution_enabled": execution_enabled,
            "quantization": {"pipeline": "sequential"},
        }
        calibration = {
            "experiment_id": "test",
            "read_only_weight_contract": True,
            "config": {"semantic_sha256": "semantic-hash"},
            "selection": {
                "overlap_count": 0,
                "all_images_decoded": True,
                "selected_calibration_samples": 1,
                "samples": [{"sample_id": "1"}],
            },
        }
        inventory = {
            "experiment_id": "test",
            "checkpoint_weights_loaded": False,
            "experiment_config": {"semantic_sha256": "semantic-hash"},
            "linear_inventory": {
                "conservative_candidate": {
                    "selected_names": selected,
                    "selected_names_sha256": driver.sha256_lines(selected),
                    "selected_groups": ["language_attention", "language_mlp"],
                }
            },
        }
        return config, calibration, inventory

    def test_execute_is_an_explicit_cli_switch_even_when_default_is_off(self) -> None:
        config, calibration, inventory = self.contract_fixture()
        result = driver.validate_contract(
            config=config,
            config_semantic_sha256="semantic-hash",
            calibration=calibration,
            inventory=inventory,
            execute=True,
        )

        self.assertTrue(result["execution_requested"])
        self.assertTrue(result["default_execution_disabled"])

    def test_dry_run_accepts_bound_manifests_and_keeps_execution_off(self) -> None:
        config, calibration, inventory = self.contract_fixture()
        result = driver.validate_contract(
            config=config,
            config_semantic_sha256="semantic-hash",
            calibration=calibration,
            inventory=inventory,
            execute=False,
        )

        self.assertFalse(result["execution_enabled"])
        self.assertFalse(result["execution_requested"])
        self.assertEqual(result["selected_target_count"], 1)

    def test_custom_mappings_exclude_linear_attention_and_vision(self) -> None:
        layer_types = [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ]
        mappings = driver.build_mapping_specs(layer_types)
        serialized = str(mappings)

        self.assertEqual(len(mappings), 3)
        self.assertIn("self_attn.q_proj", serialized)
        self.assertIn("language_model", serialized)
        self.assertNotIn("linear_attn", serialized)
        self.assertNotIn("visual", serialized)

    def test_mapping_resolution_stays_inside_exact_targets(self) -> None:
        names = [
            "model.language_model.layers.3.input_layernorm",
            "model.language_model.layers.3.self_attn.q_proj",
            "model.language_model.layers.3.self_attn.k_proj",
            "model.language_model.layers.3.self_attn.v_proj",
        ]
        mappings = [
            {
                "smooth_layer": "re:.*input_layernorm$",
                "balance_layers": ["re:.*self_attn.[qkv]_proj$"],
            }
        ]
        report = driver.validate_mapping_resolution(
            mapping_specs=mappings,
            named_module_names=names,
            selected_names=names[1:],
        )

        self.assertEqual(report[0]["smooth_match_count"], 1)
        self.assertEqual(report[0]["balance_match_count"], 3)

    def test_mapping_resolution_rejects_non_target_balance_module(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside exact targets"):
            driver.validate_mapping_resolution(
                mapping_specs=[
                    {
                        "smooth_layer": "re:.*norm$",
                        "balance_layers": ["re:.*proj$"],
                    }
                ],
                named_module_names=["layer.norm", "layer.proj"],
                selected_names=[],
            )

    def test_baseline_payload_requires_fixed_unquantized_environment(self) -> None:
        config = {
            "calibration": {"expected_dataset_sha256": "dataset-hash"},
            "environment": {
                "torch": "2.10.0+cu128",
                "transformers": "5.10.1",
                "llmcompressor": "0.12.0",
                "compressed_tensors": "0.17.1",
            },
        }
        payload = {
            "run_mode": "benchmark",
            "backend": "transformers",
            "sample_count": 10,
            "seed": 20260625,
            "protocol": {
                "dtype_requested": "bfloat16",
                "warmup_samples": 2,
                "max_new_tokens": 64,
                "batch_size": 1,
                "use_cache": True,
                "choice_fallback_enabled": False,
            },
            "runtime": {
                "backend_resolved": "transformers",
                "device_resolved": "cuda:0",
                "load_dtype_resolved": "bfloat16",
                "runtime_quantization_evidence": False,
            },
            "performance": {
                "run_contract_valid": True,
                "avg_ttft_ms": 100.0,
                "avg_throughput_tokens_per_sec": 50.0,
                "avg_request_elapsed_ms": 400.0,
            },
            "public_validation": {"passed": True, "failed_samples": 0},
            "accuracy": {"total": 10, "score": 1.0},
            "reproducibility": {
                "dataset": {
                    "sha256": "dataset-hash",
                    "selected_sample_ids_sha256": "sample-hash",
                },
                "software": {
                    "packages": {
                        "torch": "2.10.0+cu128",
                        "transformers": "5.10.1",
                        "llmcompressor": "0.12.0",
                        "compressed-tensors": "0.17.1",
                    }
                },
                "source": {"git_available": True, "git_dirty": False},
            },
        }
        result = driver.validate_baseline_payload(
            payload=payload,
            config=config,
            expected_eval_sample_ids_sha256="sample-hash",
        )
        self.assertEqual(result["throughput_tokens_per_sec"], 50.0)

        payload["runtime"]["runtime_quantization_evidence"] = True
        with self.assertRaisesRegex(ValueError, "unquantized"):
            driver.validate_baseline_payload(
                payload=payload,
                config=config,
                expected_eval_sample_ids_sha256="sample-hash",
            )

    def test_baseline_evidence_is_optional_for_weight_generation(self) -> None:
        result = driver.validate_baselines(
            paths=[],
            config={},
            expected_eval_sample_ids_sha256="unused",
        )

        self.assertFalse(result["provided"])
        self.assertEqual(result["run_count"], 0)

    @unittest.skipIf(sys.platform == "win32", "local OpenMP runtimes conflict")
    def test_calibration_dataloader_returns_model_inputs_without_labels(self) -> None:
        import torch

        class FakeProcessor:
            def __call__(self, *, text, images, **_kwargs):
                self.last_text = text
                self.last_images = images
                return {
                    "input_ids": torch.tensor([[1, 2]]),
                    "attention_mask": torch.tensor([[1, 1]]),
                    "pixel_values": torch.ones((4, 3)),
                    "image_grid_thw": torch.tensor([[1, 2, 2]]),
                }

        processor = FakeProcessor()
        loader = driver.build_calibration_dataloader(
            dataset=[{"text": "prompt", "images": "real-image"}],
            processor=processor,
            batch_size=1,
            max_seq_length=2048,
        )
        batch = next(iter(loader))

        self.assertNotIn("labels", batch)
        self.assertEqual(processor.last_text, ["prompt"])
        self.assertEqual(processor.last_images, ["real-image"])
        self.assertEqual(tuple(batch["pixel_values"].shape), (4, 3))

    def test_complete_vlm_target_preflight_rejects_text_only_model(self) -> None:
        class Linear:
            pass

        TextOnly = type(
            "Qwen3_5ForCausalLM",
            (),
            {"named_modules": lambda self: iter([("target", Linear())])},
        )
        with self.assertRaisesRegex(ValueError, "complete VLM"):
            driver.validate_loaded_model_targets(
                model=TextOnly(),
                expected_architecture="Qwen3_5ForConditionalGeneration",
                selected_names=["target"],
            )

    def test_complete_vlm_target_preflight_requires_all_linear_targets(self) -> None:
        Linear = type("Linear", (), {})
        CompleteVLM = type(
            "Qwen3_5ForConditionalGeneration",
            (),
            {
                "named_modules": lambda self: iter(
                    [("", self), ("target", Linear())]
                )
            },
        )
        report = driver.validate_loaded_model_targets(
            model=CompleteVLM(),
            expected_architecture="Qwen3_5ForConditionalGeneration",
            selected_names=["target"],
        )

        self.assertEqual(report["matched_target_count"], 1)


if __name__ == "__main__":
    unittest.main()
