from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import traceback
from pathlib import Path


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def inspect_fastpath(*, require_cuda: bool) -> dict:
    result = {
        "packages": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "triton": package_version("triton"),
            "fla-core": package_version("fla-core"),
            "causal-conv1d": package_version("causal-conv1d"),
        },
        "cuda": {},
        "transformers_fastpath": {},
        "smoke": {},
        "ready": False,
        "errors": [],
    }
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        result["cuda"] = {
            "available": cuda_available,
            "torch_cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
            "compute_capability": (
                list(torch.cuda.get_device_capability(0)) if cuda_available else None
            ),
        }
        if require_cuda and not cuda_available:
            result["errors"].append("CUDA is required but unavailable.")

        modeling_qwen3_5 = importlib.import_module(
            "transformers.models.qwen3_5.modeling_qwen3_5"
        )
        from transformers.utils.import_utils import (
            is_causal_conv1d_available,
            is_flash_linear_attention_available,
        )

        function_names = (
            "causal_conv1d_fn",
            "causal_conv1d_update",
            "chunk_gated_delta_rule",
            "fused_recurrent_gated_delta_rule",
        )
        functions = {
            name: callable(getattr(modeling_qwen3_5, name, None))
            for name in function_names
        }
        result["transformers_fastpath"] = {
            "causal_conv1d_available": is_causal_conv1d_available(),
            "flash_linear_attention_available": is_flash_linear_attention_available(),
            "functions": functions,
            "is_fast_path_available": bool(
                getattr(modeling_qwen3_5, "is_fast_path_available", False)
            ),
        }

        if cuda_available and functions["causal_conv1d_fn"]:
            x = torch.randn(1, 32, 8, device="cuda", dtype=torch.bfloat16)
            weight = torch.randn(32, 4, device="cuda", dtype=torch.bfloat16)
            bias = torch.randn(32, device="cuda", dtype=torch.bfloat16)
            output = modeling_qwen3_5.causal_conv1d_fn(
                x=x,
                weight=weight,
                bias=bias,
                activation="silu",
            )
            torch.cuda.synchronize()
            result["smoke"] = {
                "causal_conv1d_shape": list(output.shape),
                "causal_conv1d_finite": bool(torch.isfinite(output).all().item()),
            }
        result["ready"] = (
            all(functions.values())
            and result["transformers_fastpath"]["is_fast_path_available"]
            and (cuda_available or not require_cuda)
            and not result["errors"]
        )
    except Exception as exc:
        result["errors"].append(repr(exc))
        result["traceback"] = traceback.format_exc()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe the Qwen3.5 GDN fast path.")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect_fastpath(require_cuda=args.require_cuda)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    raise SystemExit(0 if result["ready"] else 1)


if __name__ == "__main__":
    main()
