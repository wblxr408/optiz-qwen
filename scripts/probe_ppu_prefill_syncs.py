"""Enumerate host-device synchronizations inside one PPU prefill forward.

Why this exists
---------------
``profile_ppu_prefill.py`` measured ``cpu_issue_fraction`` 0.988+ and called
prefill "dispatch-bound".  That metric queues N forwards without synchronizing
and compares CPU issue time to wall time -- but it cannot distinguish two very
different causes:

    (a) the CPU genuinely cannot issue kernels fast enough (decode's disease,
        fixed by capturing a graph), or
    (b) the forward contains host-device syncs -- ``.item()``, ``.tolist()``,
        a bool test on a device tensor -- which *stall* the CPU on the device
        and make it impossible for it to run ahead at all.

Both read as "CPU issue time ~= wall time".  The fix is completely different:
(a) needs graph capture and fixed shapes; (b) needs the offending value kept on
the host, which is cheap and shape-agnostic.

This script tells them apart by turning on ``torch.cuda.set_sync_debug_mode`` and
recording every warning raised during a real prefill, with the Python frame that
caused it.

Run on the target:
    PYTHONPATH=src:. python scripts/probe_ppu_prefill_syncs.py
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
import warnings
from collections import Counter
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
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--output", default="benchmarks/output/ppu_prefill_syncs.json")
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


#: Frames belonging to the probe machinery itself.  Without excluding this file
#: the deepest frame is always the warning hook, which attributes every sync to
#: the probe rather than to the model.
_SELF = Path(__file__).name


def _frame_label(stack: list[traceback.FrameSummary]) -> str:
    """Deepest frame that is not torch/warnings/probe internals -- the culprit."""

    for frame in reversed(stack):
        filename = frame.filename.replace("\\", "/")
        if "/site-packages/torch/" in filename or filename.endswith("warnings.py"):
            continue
        if filename.endswith(_SELF):
            continue
        short = filename.split("/site-packages/")[-1]
        return f"{short}:{frame.lineno} in {frame.name}"
    return "unknown"


def collect_syncs(model: Any, inputs: dict[str, Any]) -> dict[str, Any]:
    """Run one prefill with sync debug on, recording every sync site."""

    sites: Counter[str] = Counter()
    messages: Counter[str] = Counter()

    def _hook(message, category, filename, lineno, file=None, line=None):  # noqa: ANN001
        text = str(message)
        if "call to" in text or "synchroniz" in text.lower():
            sites[_frame_label(traceback.extract_stack())] += 1
            messages[text.split("\n")[0][:120]] += 1

    with torch.inference_mode():
        model(**inputs)  # warm the caches so first-call lazy work is excluded
        _sync()

        previous_hook = warnings.showwarning
        warnings.showwarning = _hook
        torch.cuda.set_sync_debug_mode("warn")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                model(**inputs)
        finally:
            torch.cuda.set_sync_debug_mode("default")
            warnings.showwarning = previous_hook
        _sync()

    return {
        "total_syncs": sum(sites.values()),
        "sites": [{"where": where, "count": count} for where, count in sites.most_common()],
        "messages": [{"text": text, "count": count} for text, count in messages.most_common(6)],
    }


def measure_pipelining(model: Any, inputs: dict[str, Any], *, reps: int = 5) -> dict[str, Any]:
    """How much can the CPU run ahead?  Re-measured here for the same forward."""

    with torch.inference_mode():
        model(**inputs)
        _sync()
        issue_start = time.perf_counter()
        for _ in range(reps):
            model(**inputs)
        issue_end = time.perf_counter()
        _sync()
        wall_end = time.perf_counter()

    issue_ms = (issue_end - issue_start) * 1000.0 / reps
    wall_ms = (wall_end - issue_start) * 1000.0 / reps
    return {
        "cpu_issue_ms_per_forward": round(issue_ms, 3),
        "wall_ms_per_forward": round(wall_ms, 3),
        "cpu_issue_fraction": round(issue_ms / wall_ms, 4) if wall_ms else None,
    }


def main() -> None:
    args = parse_args()
    model, batches = build_model_and_inputs(args)

    report: dict[str, Any] = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "purpose": "distinguish launch-bound prefill from sync-stalled prefill",
        "samples": [],
    }
    for index, inputs in enumerate(batches):
        entry: dict[str, Any] = {
            "index": index,
            "prompt_tokens": int(inputs["input_ids"].shape[-1]),
        }
        entry["pipelining"] = measure_pipelining(model, inputs)
        entry["syncs"] = collect_syncs(model, inputs)
        report["samples"].append(entry)
        print(json.dumps(entry, indent=2)[:4000])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
