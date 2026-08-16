"""Profile where PPU prefill (TTFT) time goes, post fla-core.

Throughput is solved on this hardware (one captured graph per decode token).
TTFT is not: the 50-sample A/B moved it only 57.567 -> 55.382 ms (+3.8%), so it
is the metric with headroom left.  This script answers three questions before
any code is written:

1. How is prefill split across vision tower / language stack / lm_head?
2. Is prefill dispatch-bound like decode was, or is it real device work?
   (kernel launch count + CPU-issue time vs wall time, the same test that
   diagnosed decode)
3. Does the lm_head compute logits for every prompt position, when greedy
   prefill only needs the last one?

Run on the target:
    PYTHONPATH=src:. python scripts/profile_ppu_prefill.py
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import torch


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _median_ms(fn, *, reps: int) -> float:
    samples = []
    for _ in range(reps):
        _sync()
        start = time.perf_counter()
        fn()
        _sync()
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return round(samples[len(samples) // 2], 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="./resources/model_weights/raw/Qwen3.5-2B")
    parser.add_argument(
        "--dataset-path",
        default="./resources/eval_dataset/raw/mmbench_public/mmbench_dev_en.tsv",
    )
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--output", default="benchmarks/output/ppu_prefill_profile.json")
    return parser.parse_args()


def _build_inputs(args: argparse.Namespace) -> tuple[Any, list[dict[str, Any]]]:
    """Load the model and build real MMBench prefill inputs (image + prompt)."""

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

    from pathlib import Path

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


def _stage_breakdown(model: Any, inputs: dict[str, Any], *, reps: int) -> dict[str, Any]:
    """Time the vision tower, the language stack, and lm_head separately.

    The language stack is timed through ``model.model`` so the lm_head cost is
    excluded, then the lm_head is timed on the full hidden-state sequence and
    again on only the last position -- the difference is what greedy prefill is
    paying for logits it throws away.
    """

    prompt_tokens = int(inputs["input_ids"].shape[-1])
    with torch.inference_mode():
        full = _median_ms(lambda: model(**inputs), reps=reps)

        inner = getattr(model, "model", None)
        inner_kwargs = {k: v for k, v in inputs.items() if k != "labels"}
        language_ms = None
        hidden = None
        if inner is not None:
            try:
                out = inner(**inner_kwargs)
                hidden = out.last_hidden_state
                language_ms = _median_ms(lambda: inner(**inner_kwargs), reps=reps)
            except Exception as exc:  # pragma: no cover - shape-dependent
                language_ms = f"unavailable: {type(exc).__name__}: {exc}"

        lm_head = getattr(model, "lm_head", None)
        head_all = head_last = None
        if lm_head is not None and hidden is not None:
            head_all = _median_ms(lambda: lm_head(hidden), reps=reps)
            last = hidden[:, -1:, :]
            head_last = _median_ms(lambda: lm_head(last), reps=reps)

    return {
        "prompt_tokens": prompt_tokens,
        "full_forward_ms": full,
        "language_stack_ms": language_ms,
        "lm_head_all_positions_ms": head_all,
        "lm_head_last_position_ms": head_last,
        "lm_head_waste_ms": (
            round(head_all - head_last, 3)
            if isinstance(head_all, float) and isinstance(head_last, float)
            else None
        ),
    }


def _dispatch_bound_check(model: Any, inputs: dict[str, Any], *, reps: int) -> dict[str, Any]:
    """Is prefill dispatch-bound too?  Same test that diagnosed decode.

    Queue ``reps`` forwards without synchronizing.  If the CPU time to issue
    them is close to the wall time once synchronized, the CPU is the limiter.
    If issue time is much smaller, the device is genuinely busy and prefill is
    real work -- meaning graph capture would not help TTFT.
    """

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
        "queued_forwards": reps,
        "cpu_issue_ms_per_forward": round(issue_ms, 3),
        "wall_ms_per_forward": round(wall_ms, 3),
        "cpu_issue_fraction": round(issue_ms / wall_ms, 4) if wall_ms else None,
        "verdict": (
            "dispatch-bound" if wall_ms and issue_ms / wall_ms > 0.8 else "device-bound"
        ),
    }


def _kernel_count(model: Any, inputs: dict[str, Any]) -> dict[str, Any]:
    """Count kernels in one prefill forward, for comparison with decode's 5778."""

    from torch.profiler import ProfilerActivity, profile

    with torch.inference_mode():
        model(**inputs)
        _sync()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            model(**inputs)
            _sync()

    events = [e for e in prof.events() if getattr(e, "device_type", None) is not None]
    kernels = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    top = sorted(kernels, key=lambda e: -e.self_device_time_total)[:12]
    return {
        "distinct_device_ops": len(kernels),
        "total_device_calls": sum(int(e.count) for e in kernels),
        "recorded_events": len(events),
        "top_device_ops": [
            {
                "name": e.key[:80],
                "calls": int(e.count),
                "device_ms": round(e.self_device_time_total / 1000.0, 4),
            }
            for e in top
        ],
    }


def main() -> None:
    args = parse_args()
    model, batches = _build_inputs(args)

    report: dict[str, Any] = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "attn_implementation": model.config.text_config._attn_implementation,
        "samples": [],
    }

    for index, inputs in enumerate(batches):
        entry = {"index": index}
        entry["stages"] = _stage_breakdown(model, inputs, reps=args.reps)
        entry["dispatch"] = _dispatch_bound_check(model, inputs, reps=args.reps)
        if index == 0:
            entry["kernels"] = _kernel_count(model, inputs)
        report["samples"].append(entry)
        print(json.dumps(entry, indent=2))

    from pathlib import Path

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
