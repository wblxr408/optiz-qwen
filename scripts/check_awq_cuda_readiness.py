#!/usr/bin/env python3
"""Probe the isolated CUDA AWQ environment without loading or changing weights."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_ROOT = REPO_ROOT / "scripts"
for import_root in (SRC_ROOT, SCRIPT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiment_utils import inspect_git  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs/experiments/d_awq_w4a16_cuda.json"
PACKAGE_NAMES = (
    "torch",
    "torchvision",
    "transformers",
    "llmcompressor",
    "compressed-tensors",
)
REQUIRED_SYMBOLS = {
    "awq_modifier": "llmcompressor.modifiers.transform.awq:AWQModifier",
    "quantization_modifier": (
        "llmcompressor.modifiers.quantization:QuantizationModifier"
    ),
    "auto_processor": "transformers:AutoProcessor",
    "qwen35_model": "transformers:Qwen3_5ForConditionalGeneration",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether the isolated CUDA AWQ environment is ready."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--dataset-path", type=Path)
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
    if not isinstance(raw, dict):
        raise ValueError("AWQ experiment config must be a JSON object.")
    for key in ("model", "environment", "calibration"):
        if not isinstance(raw.get(key), dict):
            raise ValueError(f"AWQ experiment config is missing {key} settings.")
    return raw


def version_matches(actual: str | None, expected: str) -> bool:
    if actual is None:
        return False
    if "+" in expected:
        return actual == expected
    return actual == expected or actual.startswith(f"{expected}+")


def probe_packages() -> dict[str, dict[str, Any]]:
    packages: dict[str, dict[str, Any]] = {}
    for name in PACKAGE_NAMES:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        packages[name] = {"installed": version is not None, "version": version}
    return packages


def probe_symbol(specification: str) -> dict[str, Any]:
    module_name, symbol_name = specification.split(":", maxsplit=1)
    try:
        module = importlib.import_module(module_name)
        symbol = getattr(module, symbol_name)
    except Exception as error:  # import failures are readiness evidence
        return {
            "available": False,
            "specification": specification,
            "error_type": type(error).__name__,
        }
    return {
        "available": True,
        "specification": specification,
        "resolved_type": type(symbol).__name__,
    }


def probe_cuda() -> dict[str, Any]:
    try:
        torch = importlib.import_module("torch")
    except Exception as error:
        return {
            "torch_imported": False,
            "available": False,
            "error_type": type(error).__name__,
        }

    available = bool(torch.cuda.is_available())
    result: dict[str, Any] = {
        "torch_imported": True,
        "available": available,
        "torch_cuda_version": torch.version.cuda,
        "device_count": int(torch.cuda.device_count()) if available else 0,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()) if available else False,
        "devices": [],
    }
    if available:
        for index in range(torch.cuda.device_count()):
            capability = torch.cuda.get_device_capability(index)
            result["devices"].append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "compute_capability": f"{capability[0]}.{capability[1]}",
                }
            )
    return result


def summarize_readiness(
    *,
    config: dict[str, Any],
    packages: dict[str, dict[str, Any]],
    symbols: dict[str, dict[str, Any]],
    cuda: dict[str, Any],
    model_exists: bool,
    dataset_exists: bool,
) -> dict[str, Any]:
    expected = config["environment"]
    checks = {
        "python_version": platform.python_version().startswith(
            f"{expected['python']}."
        ),
        "torch_version": version_matches(
            packages["torch"]["version"], expected["torch"]
        ),
        "torchvision_version": version_matches(
            packages["torchvision"]["version"], expected["torchvision"]
        ),
        "transformers_version": version_matches(
            packages["transformers"]["version"], expected["transformers"]
        ),
        "llmcompressor_version": version_matches(
            packages["llmcompressor"]["version"], expected["llmcompressor"]
        ),
        "compressed_tensors_installed": packages["compressed-tensors"]["installed"],
        "required_symbols": all(item["available"] for item in symbols.values()),
        "cuda_available": bool(cuda.get("available")),
        "cuda_bf16_supported": bool(cuda.get("bf16_supported")),
        "model_exists": model_exists,
        "dataset_exists": dataset_exists,
    }
    environment_ready = all(checks.values())
    execution_enabled = bool(config.get("execution_enabled"))
    return {
        "checks": checks,
        "environment_ready": environment_ready,
        "execution_enabled": execution_enabled,
        "quantization_authorized": environment_ready and execution_enabled,
    }


def build_report(
    *,
    repo_root: Path,
    config_path: Path,
    model_path_override: Path | None = None,
    dataset_path_override: Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    model_path = (
        model_path_override.expanduser().resolve()
        if model_path_override is not None
        else resolve_path(repo_root, config["model"]["path"])
    )
    dataset_path = (
        dataset_path_override.expanduser().resolve()
        if dataset_path_override is not None
        else resolve_path(repo_root, config["calibration"]["dataset"])
    )
    packages = probe_packages()
    symbols = {
        name: probe_symbol(specification)
        for name, specification in REQUIRED_SYMBOLS.items()
    }
    cuda = probe_cuda()
    readiness = summarize_readiness(
        config=config,
        packages=packages,
        symbols=symbols,
        cuda=cuda,
        model_exists=model_path.exists(),
        dataset_exists=dataset_path.exists(),
    )
    return {
        "schema_version": 1,
        "probe_type": "awq_cuda_environment_readiness",
        "read_only_weight_contract": True,
        "experiment_id": config.get("experiment_id"),
        "git": inspect_git(repo_root),
        "runtime": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
        },
        "packages": packages,
        "symbols": symbols,
        "cuda": cuda,
        "assets": {
            "model_exists": model_path.exists(),
            "dataset_exists": dataset_path.exists(),
        },
        "readiness": readiness,
        "claim_boundary": (
            "environment_probe_only_no_model_load_quantization_or_performance_claim"
        ),
    }


def write_report(path: Path, report: dict[str, Any], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
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
    report = build_report(
        repo_root=repo_root,
        config_path=config_path,
        model_path_override=args.model_path,
        dataset_path_override=args.dataset_path,
    )
    if args.output is None:
        print(json.dumps(report, indent=2, ensure_ascii=True))
    else:
        output = resolve_path(repo_root, args.output)
        write_report(output, report, overwrite=args.overwrite)
        print(output)
    return 0 if report["readiness"]["environment_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
