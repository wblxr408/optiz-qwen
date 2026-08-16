"""How much TTFT is actually recoverable by eliminating prefill dispatch?

Why this exists
---------------
``scripts/profile_ppu_prefill.py`` established that prefill has
``cpu_issue_fraction`` 0.988+, and that reading alone says "dispatch-bound, so
capture it like decode".  But the same artifact's top-12 device ops sum to
42.5 ms against a ~53 ms wall, which would mean the device is ~80% busy too.
Both cannot be true about *headroom*: decode was worth 8.9x because its device
work was 2.2 ms of a 20 ms step.

The 42.5 ms figure is suspect because ``key_averages()`` mixes two kinds of row:
CPU operator rows (``aten::mm``) whose ``self_device_time_total`` includes the
kernels they launched, and the kernel rows themselves (``gemm_ktype0_...``).
Summing both double counts.  This script separates them, so the ceiling on any
prefill dispatch work is a measured number rather than an inference.

The number that matters is ``device_busy_fraction`` = kernel device time / wall.
It bounds the win: a perfectly captured prefill still has to run the kernels, so
the best case is wall -> kernel device time.

Run on the target:
    PYTHONPATH=src:. python scripts/profile_ppu_prefill_headroom.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="./resources/model_weights/raw/Qwen3.5-2B")
    parser.add_argument(
        "--dataset-path",
        default="./resources/eval_dataset/raw/mmbench_public/mmbench_dev_en.tsv",
    )
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--output", default="benchmarks/output/ppu_prefill_headroom.json")
    return parser.parse_args()


def build_model_and_inputs(args: argparse.Namespace) -> tuple[Any, list[dict[str, Any]]]:
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    from optiz_qwen.evaluation.dndx_public_benchmark import (
        build_prompt,
        decode_image,
        load_mmbench_tsv,
    )

    processor = AutoProcessor.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).to("cuda:0")
    model.eval()
    model.config._attn_implementation = "sdpa"
    model.config.text_config._attn_implementation = "sdpa"

    samples = load_mmbench_tsv(Path(args.dataset_path), limit=args.samples)
    batches = []
    for sample in samples:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": decode_image(sample.image_b64)},
                {"type": "text", "text": build_prompt(sample)},
            ],
        }]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        batches.append(dict(inputs))
    return model, batches


def _is_kernel_row(row: Any) -> bool:
    """Is this ``key_averages()`` row a device kernel rather than a CPU operator?

    Kernel rows carry ``device_type`` CUDA; ``aten::*`` operator rows carry CPU
    and attribute their children's device time into ``self_device_time_total``,
    which is exactly the double count this script exists to remove.
    """

    from torch.autograd import DeviceType

    device_type = getattr(row, "device_type", None)
    if device_type is not None:
        try:
            return device_type == DeviceType.CUDA
        except Exception:  # pragma: no cover - enum mismatch across versions
            pass
    # Fallback: operator rows are the ones namespaced with "::".
    return "::" not in str(getattr(row, "key", ""))


def device_accounting(model: Any, inputs: dict[str, Any], *, reps: int) -> dict[str, Any]:
    """Split one prefill forward into kernel device time vs CPU operator time."""

    from torch.profiler import ProfilerActivity, profile

    with torch.inference_mode():
        model(**inputs)
        _sync()

        # Untimed-by-profiler wall clock, so profiler overhead cannot inflate it.
        walls = []
        for _ in range(reps):
            _sync()
            start = time.perf_counter()
            model(**inputs)
            _sync()
            walls.append((time.perf_counter() - start) * 1000.0)
        walls.sort()
        wall_ms = walls[len(walls) // 2]

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            model(**inputs)
            _sync()

    rows = list(prof.key_averages())
    kernel_rows = [r for r in rows if _is_kernel_row(r) and r.self_device_time_total > 0]
    operator_rows = [r for r in rows if not _is_kernel_row(r)]

    kernel_device_ms = sum(r.self_device_time_total for r in kernel_rows) / 1000.0
    kernel_launches = sum(int(r.count) for r in kernel_rows)
    operator_device_ms = sum(
        r.self_device_time_total for r in operator_rows if r.self_device_time_total > 0
    ) / 1000.0
    operator_cpu_ms = sum(r.self_cpu_time_total for r in operator_rows) / 1000.0
    operator_calls = sum(int(r.count) for r in operator_rows)

    top_kernels = sorted(kernel_rows, key=lambda r: -r.self_device_time_total)[:12]

    busy = kernel_device_ms / wall_ms if wall_ms else None
    return {
        "prompt_tokens": int(inputs["input_ids"].shape[-1]),
        "wall_ms": round(wall_ms, 3),
        # The ceiling on dispatch elimination: a captured prefill still runs
        # these kernels, so wall can at best fall to this.
        "kernel_device_ms": round(kernel_device_ms, 3),
        "kernel_launches": kernel_launches,
        "distinct_kernels": len(kernel_rows),
        "device_busy_fraction": round(busy, 4) if busy else None,
        "idle_ms": round(wall_ms - kernel_device_ms, 3),
        "max_speedup_if_dispatch_free": (
            round(wall_ms / kernel_device_ms, 3) if kernel_device_ms else None
        ),
        # Reported separately to show why summing every row overstates device work.
        "operator_rows": len(operator_rows),
        "operator_calls": operator_calls,
        "operator_attributed_device_ms": round(operator_device_ms, 3),
        "operator_self_cpu_ms": round(operator_cpu_ms, 3),
        "naive_all_rows_device_ms": round(kernel_device_ms + operator_device_ms, 3),
        "top_kernels": [
            {
                "name": str(r.key)[:80],
                "calls": int(r.count),
                "device_ms": round(r.self_device_time_total / 1000.0, 4),
            }
            for r in top_kernels
        ],
    }


def main() -> None:
    args = parse_args()
    model, batches = build_model_and_inputs(args)

    report: dict[str, Any] = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "attn_implementation": model.config.text_config._attn_implementation,
        "purpose": "bound the recoverable TTFT from prefill dispatch elimination",
        "samples": [],
    }
    for index, inputs in enumerate(batches):
        entry = {"index": index}
        entry.update(device_accounting(model, inputs, reps=args.reps))
        report["samples"].append(entry)
        print(json.dumps({k: v for k, v in entry.items() if k != "top_kernels"}, indent=2))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
