#!/usr/bin/env python3
"""Inventory Qwen3.5 AWQ targets on meta tensors without loading checkpoint weights."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_ROOT = REPO_ROOT / "scripts"
for import_root in (SRC_ROOT, SCRIPT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from awq_contract import semantic_config_sha256  # noqa: E402
from experiment_utils import inspect_git, sha256_file, sha256_lines  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs/experiments/d_awq_w4a16_cuda.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a read-only semantic inventory of Qwen3.5 Linear modules."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Override model.path when weights live outside the worktree.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def resolve_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def load_config(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("model"), dict):
        raise ValueError("AWQ experiment config is missing model settings.")
    if bool(raw.get("execution_enabled")):
        raise ValueError("Module inventory must run while execution_enabled is false.")
    return raw


def classify_linear(name: str) -> str:
    if name == "lm_head":
        return "lm_head"
    if name.startswith("model.visual.blocks."):
        return "vision_encoder"
    if name.startswith("model.visual.merger."):
        return "multimodal_projector"
    if name.startswith("model.language_model.layers.") and ".linear_attn." in name:
        return "language_gated_deltanet"
    if name.startswith("model.language_model.layers.") and ".self_attn." in name:
        return "language_attention"
    if name.startswith("model.language_model.layers.") and ".mlp." in name:
        return "language_mlp"
    return "unclassified"


def partition_linear_names(names: Iterable[str]) -> dict[str, Any]:
    ordered = list(names)
    if len(ordered) != len(set(ordered)):
        raise ValueError("Linear module names must be unique.")
    groups: dict[str, list[str]] = {}
    for name in ordered:
        groups.setdefault(classify_linear(name), []).append(name)
    if groups.get("unclassified"):
        raise ValueError("Unclassified Linear modules block AWQ policy resolution.")

    selected_groups = {"language_attention", "language_mlp"}
    selected = [name for name in ordered if classify_linear(name) in selected_groups]
    ignored = [name for name in ordered if classify_linear(name) not in selected_groups]
    if set(selected) & set(ignored) or set(selected) | set(ignored) != set(ordered):
        raise AssertionError("AWQ target and ignore partitions are inconsistent.")
    return {
        "group_counts": dict(sorted(Counter(map(classify_linear, ordered)).items())),
        "groups": {key: value for key, value in sorted(groups.items())},
        "conservative_candidate": {
            "selected_groups": sorted(selected_groups),
            "selected_count": len(selected),
            "selected_names": selected,
            "selected_names_sha256": sha256_lines(selected),
            "ignored_count": len(ignored),
            "ignored_names": ignored,
            "ignored_names_sha256": sha256_lines(ignored),
            "policy_status": "inventory_resolved_execution_still_disabled",
        },
    }


def config_architecture(config: Any) -> str | None:
    architectures = getattr(config, "architectures", None)
    if isinstance(architectures, list) and architectures:
        return str(architectures[0])
    return None


def build_inventory(
    *,
    repo_root: Path,
    config_path: Path,
    model_path_override: Path | None = None,
) -> dict[str, Any]:
    experiment = load_config(config_path)
    model_contract = experiment["model"]
    model_path = (
        model_path_override.expanduser().resolve()
        if model_path_override is not None
        else resolve_path(repo_root, model_contract["path"])
    )
    model_config_path = model_path / "config.json"
    if not model_config_path.is_file():
        raise FileNotFoundError(f"Model config is missing: {model_config_path}")

    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen3_5ForConditionalGeneration

    loaded_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    actual_model_type = str(getattr(loaded_config, "model_type", ""))
    if actual_model_type != model_contract["expected_model_type"]:
        raise ValueError("Loaded model_type does not match the AWQ experiment contract.")
    with init_empty_weights():
        model = Qwen3_5ForConditionalGeneration(loaded_config)
    actual_architecture = type(model).__name__
    if actual_architecture != model_contract["expected_architecture"]:
        raise ValueError("Loaded architecture does not match the AWQ experiment contract.")

    named_modules = list(model.named_modules())
    linear_names = [
        name for name, module in named_modules if type(module).__name__ == "Linear"
    ]
    partition = partition_linear_names(linear_names)
    parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})
    if parameter_devices != ["meta"]:
        raise ValueError("Inventory construction allocated non-meta parameters.")

    return {
        "schema_version": 1,
        "manifest_type": "awq_qwen35_module_inventory",
        "read_only_weight_contract": True,
        "checkpoint_weights_loaded": False,
        "experiment_id": experiment.get("experiment_id"),
        "execution_enabled": False,
        "git": inspect_git(repo_root),
        "experiment_config": {
            "sha256": sha256_file(config_path),
            "semantic_sha256": semantic_config_sha256(experiment),
        },
        "model": {
            "identifier": model_contract["identifier"],
            "model_type": actual_model_type,
            "config_architecture": config_architecture(loaded_config),
            "instantiated_architecture": actual_architecture,
            "config_sha256": sha256_file(model_config_path),
            "parameter_devices": parameter_devices,
            "named_module_count": len(named_modules),
            "linear_count": len(linear_names),
            "linear_names_sha256": sha256_lines(linear_names),
        },
        "linear_inventory": partition,
        "claim_boundary": (
            "module_inventory_only_no_checkpoint_load_quantization_or_performance_claim"
        ),
    }


def write_inventory(path: Path, inventory: dict[str, Any], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(inventory, indent=2, ensure_ascii=True) + "\n",
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
    inventory = build_inventory(
        repo_root=repo_root,
        config_path=config_path,
        model_path_override=args.model_path,
    )
    if args.output is None:
        print(json.dumps(inventory, indent=2, ensure_ascii=True))
    else:
        output = resolve_path(repo_root, args.output)
        write_inventory(output, inventory, overwrite=args.overwrite)
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
