#!/usr/bin/env python3
"""Prepare or execute the gated conservative AWQ W4A16 experiment."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_ROOT = REPO_ROOT / "scripts"
for import_root in (SRC_ROOT, SCRIPT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from optiz_qwen.evaluation.dndx_public_benchmark import (  # noqa: E402
    build_prompt,
    decode_image,
    load_mmbench_tsv,
)
from awq_contract import semantic_config_sha256  # noqa: E402
from experiment_utils import inspect_git, sha256_file, sha256_lines  # noqa: E402
from prepare_awq_calibration import sample_sha256  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs/experiments/d_awq_w4a16_cuda.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run or execute conservative multimodal AWQ W4A16."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--module-inventory", type=Path, required=True)
    parser.add_argument(
        "--baseline-result",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional pre-quantization BF16 evidence. If supplied, repeat exactly "
            "three times. DNDX v1.1 comparison is performed after generation by "
            "run_v11_cuda_matrix.py."
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite-report", action="store_true")
    return parser.parse_args(argv)


def resolve_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def load_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return raw


def validate_contract(
    *,
    config: dict[str, Any],
    config_semantic_sha256: str,
    calibration: dict[str, Any],
    inventory: dict[str, Any],
    execute: bool,
) -> dict[str, Any]:
    experiment_id = config.get("experiment_id")
    if config.get("quantization", {}).get("pipeline") != "sequential":
        raise ValueError("AWQ transform and quantization require the sequential pipeline.")
    if calibration.get("experiment_id") != experiment_id:
        raise ValueError("Calibration manifest experiment_id does not match config.")
    if inventory.get("experiment_id") != experiment_id:
        raise ValueError("Module inventory experiment_id does not match config.")
    if calibration.get("config", {}).get("semantic_sha256") != config_semantic_sha256:
        raise ValueError("Calibration manifest is not bound to the current config.")
    if (
        inventory.get("experiment_config", {}).get("semantic_sha256")
        != config_semantic_sha256
    ):
        raise ValueError("Module inventory is not bound to the current config.")
    if not calibration.get("read_only_weight_contract"):
        raise ValueError("Calibration manifest lacks its read-only contract.")
    if inventory.get("checkpoint_weights_loaded") is not False:
        raise ValueError("Module inventory must prove checkpoint weights were not loaded.")

    selection = calibration.get("selection", {})
    candidate = inventory.get("linear_inventory", {}).get(
        "conservative_candidate", {}
    )
    selected_names = candidate.get("selected_names")
    if not isinstance(selected_names, list) or not selected_names:
        raise ValueError("Module inventory has no conservative AWQ targets.")
    if candidate.get("selected_names_sha256") != sha256_lines(selected_names):
        raise ValueError("Conservative AWQ target list hash is invalid.")
    if candidate.get("selected_groups") != ["language_attention", "language_mlp"]:
        raise ValueError("Conservative AWQ target groups are not the approved pair.")
    if selection.get("overlap_count") != 0 or not selection.get("all_images_decoded"):
        raise ValueError("Calibration selection is not disjoint and image-validated.")
    if selection.get("selected_calibration_samples") != len(selection.get("samples", [])):
        raise ValueError("Calibration selection count does not match its sample manifest.")

    execution_enabled = bool(config.get("execution_enabled"))
    return {
        "execution_enabled": execution_enabled,
        "execution_requested": execute,
        "default_execution_disabled": not execution_enabled,
        "selected_target_count": len(selected_names),
        "selected_target_names_sha256": sha256_lines(selected_names),
        "calibration_sample_count": len(selection["samples"]),
    }


def build_mapping_specs(layer_types: list[str]) -> list[dict[str, Any]]:
    full_indices = [
        index for index, layer_type in enumerate(layer_types)
        if layer_type == "full_attention"
    ]
    if not full_indices or len(full_indices) == len(layer_types):
        raise ValueError("Expected a hybrid model with full and linear attention layers.")
    full_pattern = "|".join(map(str, full_indices))
    layer_prefix = r".*language_model\.layers"
    return [
        {
            "smooth_layer": (
                rf"re:{layer_prefix}\.({full_pattern})\.input_layernorm$"
            ),
            "balance_layers": [
                rf"re:{layer_prefix}\.({full_pattern})\.self_attn.q_proj$",
                rf"re:{layer_prefix}\.({full_pattern})\.self_attn.k_proj$",
                rf"re:{layer_prefix}\.({full_pattern})\.self_attn.v_proj$",
            ],
        },
        {
            "smooth_layer": rf"re:{layer_prefix}\.[0-9]+\.post_attention_layernorm$",
            "balance_layers": [
                rf"re:{layer_prefix}\.[0-9]+\.mlp.gate_proj$",
                rf"re:{layer_prefix}\.[0-9]+\.mlp.up_proj$",
            ],
        },
        {
            "smooth_layer": rf"re:{layer_prefix}\.[0-9]+\.mlp.up_proj$",
            "balance_layers": [
                rf"re:{layer_prefix}\.[0-9]+\.mlp.down_proj$",
            ],
        },
    ]


def validate_baseline_payload(
    *,
    payload: dict[str, Any],
    config: dict[str, Any],
    expected_eval_sample_ids_sha256: str,
) -> dict[str, float]:
    if payload.get("run_mode") != "benchmark" or payload.get("backend") != "transformers":
        raise ValueError("AWQ baseline must use benchmark mode and Transformers.")
    if payload.get("sample_count") != 10 or payload.get("seed") != 20260625:
        raise ValueError("AWQ baseline must use the fixed ten-sample selection and seed.")
    protocol = payload.get("protocol", {})
    expected_protocol = {
        "dtype_requested": "bfloat16",
        "warmup_samples": 2,
        "max_new_tokens": 64,
        "batch_size": 1,
        "use_cache": True,
        "choice_fallback_enabled": False,
    }
    if any(protocol.get(key) != value for key, value in expected_protocol.items()):
        raise ValueError("AWQ baseline protocol does not match the fixed contract.")
    runtime = payload.get("runtime", {})
    if (
        runtime.get("backend_resolved") != "transformers"
        or runtime.get("device_resolved") != "cuda:0"
        or runtime.get("load_dtype_resolved") != "bfloat16"
        or runtime.get("runtime_quantization_evidence") is not False
    ):
        raise ValueError("AWQ baseline runtime is not an unquantized CUDA BF16 run.")
    performance = payload.get("performance", {})
    validation = payload.get("public_validation", {})
    accuracy = payload.get("accuracy", {})
    if not performance.get("run_contract_valid") or validation.get("passed") is not True:
        raise ValueError("AWQ baseline contract or public validation failed.")
    if validation.get("failed_samples") != 0 or accuracy.get("total") != 10:
        raise ValueError("AWQ baseline does not contain ten valid samples.")
    dataset = payload.get("reproducibility", {}).get("dataset", {})
    if (
        dataset.get("sha256") != config["calibration"]["expected_dataset_sha256"]
        or dataset.get("selected_sample_ids_sha256")
        != expected_eval_sample_ids_sha256
    ):
        raise ValueError("AWQ baseline dataset identity or sample IDs changed.")
    packages = payload.get("reproducibility", {}).get("software", {}).get(
        "packages", {}
    )
    expected_environment = config["environment"]
    expected_packages = {
        "torch": expected_environment["torch"],
        "transformers": expected_environment["transformers"],
        "llmcompressor": expected_environment["llmcompressor"],
        "compressed-tensors": expected_environment["compressed_tensors"],
    }
    if any(packages.get(name) != version for name, version in expected_packages.items()):
        raise ValueError("AWQ baseline package versions do not match the experiment.")
    source = payload.get("reproducibility", {}).get("source", {})
    if source.get("git_available") is not True or source.get("git_dirty") is not False:
        raise ValueError("AWQ baseline was not produced from a clean Git source.")
    return {
        "ttft_ms": float(performance["avg_ttft_ms"]),
        "throughput_tokens_per_sec": float(
            performance["avg_throughput_tokens_per_sec"]
        ),
        "request_elapsed_ms": float(performance["avg_request_elapsed_ms"]),
        "accuracy": float(accuracy["score"]),
    }


def validate_baselines(
    *,
    paths: list[Path],
    config: dict[str, Any],
    expected_eval_sample_ids_sha256: str,
) -> dict[str, Any]:
    if not paths:
        return {
            "provided": False,
            "run_count": 0,
            "claim_boundary": (
                "weight_generation_only_post_quantization_v11_benchmark_required"
            ),
        }
    if len(paths) != 3:
        raise ValueError("Provide either zero or exactly three BF16 baseline results.")
    file_hashes = [sha256_file(path) for path in paths]
    if len(set(file_hashes)) != len(file_hashes):
        raise ValueError("BF16 baseline result files must be three distinct runs.")
    runs = [
        validate_baseline_payload(
            payload=load_json(path),
            config=config,
            expected_eval_sample_ids_sha256=expected_eval_sample_ids_sha256,
        )
        for path in paths
    ]
    ttft = [run["ttft_ms"] for run in runs]
    throughput = [run["throughput_tokens_per_sec"] for run in runs]
    elapsed = [run["request_elapsed_ms"] for run in runs]
    return {
        "provided": True,
        "result_sha256": file_hashes,
        "run_count": 3,
        "all_accuracy": [run["accuracy"] for run in runs],
        "ttft_ms_mean": statistics.mean(ttft),
        "ttft_ms_sample_std": statistics.stdev(ttft),
        "throughput_tokens_per_sec_mean": statistics.mean(throughput),
        "throughput_tokens_per_sec_sample_std": statistics.stdev(throughput),
        "request_elapsed_ms_mean": statistics.mean(elapsed),
        "request_elapsed_ms_sample_std": statistics.stdev(elapsed),
    }


def validate_mapping_resolution(
    *,
    mapping_specs: list[dict[str, Any]],
    named_module_names: list[str],
    selected_names: list[str],
) -> list[dict[str, Any]]:
    selected = set(selected_names)
    report: list[dict[str, Any]] = []
    for specification in mapping_specs:
        smooth_pattern = specification["smooth_layer"]
        if not smooth_pattern.startswith("re:"):
            raise ValueError("AWQ smooth mappings must use explicit regular expressions.")
        smooth_matches = [
            name for name in named_module_names
            if re.fullmatch(smooth_pattern[3:], name)
        ]
        if not smooth_matches:
            raise ValueError(f"AWQ smooth mapping matched no modules: {smooth_pattern}")
        balance_matches: list[str] = []
        for balance_pattern in specification["balance_layers"]:
            if not balance_pattern.startswith("re:"):
                raise ValueError("AWQ balance mappings must use regular expressions.")
            matches = [
                name for name in named_module_names
                if re.fullmatch(balance_pattern[3:], name)
            ]
            if not matches:
                raise ValueError(f"AWQ balance mapping matched no modules: {balance_pattern}")
            balance_matches.extend(matches)
        unexpected = sorted(set(balance_matches) - selected)
        if unexpected:
            raise ValueError("AWQ mappings would modify modules outside exact targets.")
        report.append(
            {
                "smooth_match_count": len(smooth_matches),
                "smooth_matches_sha256": sha256_lines(smooth_matches),
                "balance_match_count": len(balance_matches),
                "balance_matches_sha256": sha256_lines(balance_matches),
            }
        )
    return report


def build_dataset(
    *,
    dataset_path: Path,
    calibration: dict[str, Any],
    processor: Any,
) -> tuple[Any, dict[str, Any]]:
    from datasets import Dataset

    sample_by_id = {
        sample.sample_id: sample for sample in load_mmbench_tsv(dataset_path)
    }
    texts: list[str] = []
    images: list[Any] = []
    selected_ids: list[str] = []
    image_shapes: list[str] = []
    for expected in calibration["selection"]["samples"]:
        sample_id = str(expected["sample_id"])
        sample = sample_by_id.get(sample_id)
        if sample is None:
            raise ValueError(f"Calibration sample is missing from dataset: {sample_id}")
        if sample_sha256(sample) != expected["sample_sha256"]:
            raise ValueError(f"Calibration sample content changed: {sample_id}")
        image = decode_image(sample.image_b64)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": build_prompt(sample)},
                ],
            }
        ]
        texts.append(
            processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        images.append(image)
        selected_ids.append(sample_id)
        image_shapes.append(f"{image.width}x{image.height}:{image.mode}")

    dataset = Dataset.from_dict({"text": texts, "images": images})
    probes = []
    for index in sorted({0, len(dataset) - 1}):
        row = dataset[index]
        encoded = processor(
            text=[row["text"]],
            images=[row["images"]],
            return_tensors="pt",
        )
        probes.append(
            {
                "index": index,
                "input_ids_shape": list(encoded["input_ids"].shape),
                "pixel_values_shape": list(encoded["pixel_values"].shape),
                "image_grid_thw": encoded["image_grid_thw"].tolist(),
            }
        )
    return dataset, {
        "row_count": len(dataset),
        "selected_sample_ids_sha256": sha256_lines(selected_ids),
        "image_shapes_sha256": sha256_lines(image_shapes),
        "processor_probes": probes,
    }


def create_recipe(
    *,
    mapping_specs: list[dict[str, Any]],
    selected_names: list[str],
    quantization: dict[str, Any],
) -> list[Any]:
    import torch
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQMapping, AWQModifier

    mappings = [AWQMapping(**specification) for specification in mapping_specs]
    return [
        AWQModifier(
            mappings=mappings,
            offload_device=torch.device(quantization["offload_device"]),
            duo_scaling=quantization["duo_scaling"],
            n_grid=int(quantization["n_grid"]),
        ),
        QuantizationModifier(
            targets=selected_names,
            scheme=quantization["scheme"],
        ),
    ]


def build_calibration_dataloader(
    *,
    dataset: Any,
    processor: Any,
    batch_size: int,
    max_seq_length: int,
) -> Any:
    from torch.utils.data import DataLoader

    if batch_size != 1:
        raise ValueError("The multimodal AWQ contract currently requires batch_size=1.")

    def collate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = processor(
            text=[row["text"] for row in rows],
            images=[row["images"] for row in rows],
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        return dict(encoded)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_rows,
    )


def validate_loaded_model_targets(
    *,
    model: Any,
    expected_architecture: str,
    selected_names: list[str],
) -> dict[str, Any]:
    architecture = type(model).__name__
    if architecture != expected_architecture:
        raise ValueError(
            f"Loaded {architecture}, expected complete VLM {expected_architecture}."
        )
    named_modules = {name: module for name, module in model.named_modules()}
    missing = sorted(set(selected_names) - set(named_modules))
    if missing:
        raise ValueError("Loaded complete VLM does not contain every exact AWQ target.")
    non_linear = [
        name for name in selected_names
        if type(named_modules[name]).__name__ != "Linear"
    ]
    if non_linear:
        raise ValueError("One or more exact AWQ targets are not Linear modules.")
    return {
        "architecture": architecture,
        "named_module_count": len(named_modules),
        "matched_target_count": len(selected_names),
        "matched_target_names_sha256": sha256_lines(selected_names),
    }


def execute_quantization(
    *,
    model_path: Path,
    processor: Any,
    dataset: Any,
    recipe: list[Any],
    selected_names: list[str],
    expected_architecture: str,
    quantization: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    import torch
    from llmcompressor import oneshot
    from transformers import AutoModelForMultimodalLM

    if output_dir.exists():
        raise FileExistsError(f"Quantized output already exists: {output_dir}")
    staging = output_dir.with_name(f".{output_dir.name}.staging-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"Quantization staging path already exists: {staging}")
    staging.parent.mkdir(parents=True, exist_ok=True)
    calibration_dataloader = build_calibration_dataloader(
        dataset=dataset,
        processor=processor,
        batch_size=int(quantization["batch_size"]),
        max_seq_length=int(quantization["max_seq_length"]),
    )
    try:
        model = AutoModelForMultimodalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        ).to("cuda:0")
        model.eval()
        model_preflight = validate_loaded_model_targets(
            model=model,
            expected_architecture=expected_architecture,
            selected_names=selected_names,
        )
        oneshot(
            model=model,
            processor=processor,
            dataset=calibration_dataloader,
            recipe=recipe,
            precision="bfloat16",
            save_compressed=True,
            output_dir=str(staging),
            num_calibration_samples=len(dataset),
            shuffle_calibration_samples=False,
            batch_size=int(quantization["batch_size"]),
            max_seq_length=int(quantization["max_seq_length"]),
            pipeline=quantization["pipeline"],
            sequential_offload_device=quantization["offload_device"],
        )
        processor.save_pretrained(staging)
        os.replace(staging, output_dir)
        return model_preflight
    except BaseException:
        # Preserve a non-empty staging directory for failure analysis.
        if staging.exists() and not any(staging.iterdir()):
            staging.rmdir()
        raise


def write_report(path: Path, report: dict[str, Any], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Report already exists: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    config_path = resolve_path(repo_root, args.config)
    calibration_path = resolve_path(repo_root, args.calibration_manifest)
    inventory_path = resolve_path(repo_root, args.module_inventory)
    baseline_paths = [resolve_path(repo_root, path) for path in args.baseline_result]
    config = load_json(config_path)
    calibration = load_json(calibration_path)
    inventory = load_json(inventory_path)
    contract = validate_contract(
        config=config,
        config_semantic_sha256=semantic_config_sha256(config),
        calibration=calibration,
        inventory=inventory,
        execute=args.execute,
    )
    baselines = validate_baselines(
        paths=baseline_paths,
        config=config,
        expected_eval_sample_ids_sha256=calibration["selection"][
            "excluded_eval_sample_ids_sha256"
        ],
    )

    git = inspect_git(repo_root)
    if not git.get("git_available"):
        raise ValueError(
            "Git metadata is unavailable. Run inside a Git clone with git on PATH."
        )
    if git.get("git_dirty"):
        raise ValueError("AWQ dry-run and execution require a clean Git worktree.")
    model_path = (
        args.model_path.expanduser().resolve()
        if args.model_path is not None
        else resolve_path(repo_root, config["model"]["path"])
    )
    dataset_path = (
        args.dataset_path.expanduser().resolve()
        if args.dataset_path is not None
        else resolve_path(repo_root, config["calibration"]["dataset"])
    )
    if sha256_file(dataset_path) != config["calibration"]["expected_dataset_sha256"]:
        raise ValueError("Dataset hash does not match the AWQ experiment config.")

    from accelerate import init_empty_weights
    from transformers import (
        AutoConfig,
        AutoProcessor,
        Qwen3_5ForConditionalGeneration,
    )

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    dataset, dataset_report = build_dataset(
        dataset_path=dataset_path,
        calibration=calibration,
        processor=processor,
    )
    model_config = load_json(model_path / "config.json")
    layer_types = model_config.get("text_config", {}).get("layer_types")
    if not isinstance(layer_types, list):
        raise ValueError("Model config is missing text_config.layer_types.")
    mapping_specs = build_mapping_specs(layer_types)
    selected_names = inventory["linear_inventory"]["conservative_candidate"][
        "selected_names"
    ]
    loaded_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    with init_empty_weights():
        empty_model = Qwen3_5ForConditionalGeneration(loaded_config)
    mapping_resolution = validate_mapping_resolution(
        mapping_specs=mapping_specs,
        named_module_names=[name for name, _module in empty_model.named_modules()],
        selected_names=selected_names,
    )
    recipe = create_recipe(
        mapping_specs=mapping_specs,
        selected_names=selected_names,
        quantization=config["quantization"],
    )

    report = {
        "schema_version": 1,
        "report_type": "awq_w4a16_driver",
        "experiment_id": config.get("experiment_id"),
        "candidate_id": config["quantization"]["candidate_id"],
        "git": git,
        "contract": contract,
        "inputs": {
            "config_sha256": sha256_file(config_path),
            "config_semantic_sha256": semantic_config_sha256(config),
            "calibration_manifest_sha256": sha256_file(calibration_path),
            "module_inventory_sha256": sha256_file(inventory_path),
            "dataset_sha256": sha256_file(dataset_path),
            "model_config_sha256": sha256_file(model_path / "config.json"),
        },
        "dataset": dataset_report,
        "bf16_baselines": baselines,
        "recipe": {
            "scheme": config["quantization"]["scheme"],
            "target_count": len(selected_names),
            "target_names_sha256": sha256_lines(selected_names),
            "mapping_specs": mapping_specs,
            "mapping_resolution": mapping_resolution,
            "duo_scaling": config["quantization"]["duo_scaling"],
            "n_grid": config["quantization"]["n_grid"],
            "batch_size": config["quantization"]["batch_size"],
            "max_seq_length": config["quantization"]["max_seq_length"],
            "pipeline": config["quantization"]["pipeline"],
            "calibration_input": (
                "preprocessed_dataloader_without_training_labels"
            ),
        },
        "execution_attempted": args.execute,
        "execution_completed": False,
        "claim_boundary": "dry_run_only_no_quantization_or_performance_claim",
    }
    if args.execute:
        if args.output_dir is None:
            raise ValueError("--execute requires --output-dir.")
        output_dir = resolve_path(repo_root, args.output_dir)
        model_preflight = execute_quantization(
            model_path=model_path,
            processor=processor,
            dataset=dataset,
            recipe=recipe,
            selected_names=selected_names,
            expected_architecture=config["model"]["expected_architecture"],
            quantization=config["quantization"],
            output_dir=output_dir,
        )
        report["execution_model"] = model_preflight
        report["execution_completed"] = True
        report["claim_boundary"] = (
            "quantization_completed_pending_artifact_reload_and_benchmark_validation"
        )

    if args.report is None:
        print(json.dumps(report, indent=2, ensure_ascii=True))
    else:
        report_path = resolve_path(repo_root, args.report)
        write_report(report_path, report, overwrite=args.overwrite_report)
        print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
