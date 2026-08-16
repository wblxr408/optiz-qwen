"""Does eliding the vision tower's per-block host sync move PPU prefill?

The finding this tests
----------------------
``scripts/probe_ppu_prefill_syncs.py`` counted 93-94 host-device syncs in one
prefill forward and attributed **72 of them to a single line**,
``modeling_qwen3_5.py:968``::

    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    splits = [torch.split(t, lengths.tolist(), dim=2) for t in (q, k, v)]

``.tolist()`` is a device->host copy, so it blocks the CPU until the device
drains.  It runs three times per vision block and there are 24 blocks.  Every
call computes the *same* list, because ``cu_seqlens`` is derived from
``grid_thw`` once per forward and the identical tensor object is handed to every
block.

This matters more than its device cost suggests.  Prefill measured
``cpu_issue_fraction`` 0.989, which was read as "the CPU cannot issue fast
enough".  But a sync also pins issue time to wall time -- while the CPU is
blocked inside ``.tolist()`` it is neither issuing nor running ahead.  So the
0.989 could mean "launch-bound" (needs graph capture) or "sync-stalled" (needs
the value kept on the host).  The device-busy fraction is only 0.47-0.51, i.e.
half the wall clock is idle device time, which is consistent with either.

The experiment: memoize the host-side lengths per ``cu_seqlens`` object and
re-measure.  A win means prefill was partly sync-stalled and the fix is cheap
and shape-agnostic.  A null result means it is genuinely launch-bound and only
a captured/compiled prefill can help.

Numerics are checked, not assumed: the patched vision tower output is compared
byte-for-byte against the unpatched one.

Run on the target:
    PYTHONPATH=src:. python scripts/probe_vision_sync_elision.py
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


def _median_ms(fn, *, reps: int) -> tuple[float, float]:
    samples = []
    for _ in range(reps):
        _sync()
        start = time.perf_counter()
        fn()
        _sync()
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return round(samples[len(samples) // 2], 3), round(sum(samples) / len(samples), 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="./resources/model_weights/raw/Qwen3.5-2B")
    parser.add_argument(
        "--dataset-path",
        default="./resources/eval_dataset/raw/mmbench_public/mmbench_dev_en.tsv",
    )
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--reps", type=int, default=9)
    parser.add_argument("--output", default="benchmarks/output/ppu_vision_sync_elision.json")
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


def _count_syncs(model: Any, inputs: dict[str, Any]) -> int:
    """Syncs in one prefill, via sync debug mode."""

    import warnings

    count = 0

    def _hook(message, category, filename, lineno, file=None, line=None):  # noqa: ANN001
        nonlocal count
        if "synchroniz" in str(message).lower():
            count += 1

    with torch.inference_mode():
        model(**inputs)
        _sync()
        previous = warnings.showwarning
        warnings.showwarning = _hook
        torch.cuda.set_sync_debug_mode("warn")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                model(**inputs)
        finally:
            torch.cuda.set_sync_debug_mode("default")
            warnings.showwarning = previous
        _sync()
    return count


def main() -> None:
    args = parse_args()
    model, batches = build_model_and_inputs(args)

    from optiz_qwen.kernels.vision_prefill_sync import (
        elide_vision_attention_host_sync,
        vision_sync_elision_available,
    )

    report: dict[str, Any] = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "purpose": "does removing the vision tower's per-block .tolist() sync move prefill",
        "patch_applicable": vision_sync_elision_available(model),
        "samples": [],
    }

    for index, inputs in enumerate(batches):
        entry: dict[str, Any] = {
            "index": index,
            "prompt_tokens": int(inputs["input_ids"].shape[-1]),
        }

        with torch.inference_mode():
            before_logits = model(**inputs).logits[:, -1, :].clone()
        entry["baseline_syncs"] = _count_syncs(model, inputs)
        p50, mean = _median_ms(lambda: model(**inputs), reps=args.reps)
        entry["baseline_prefill_p50_ms"] = p50
        entry["baseline_prefill_mean_ms"] = mean

        with elide_vision_attention_host_sync(model):
            with torch.inference_mode():
                after_logits = model(**inputs).logits[:, -1, :].clone()
            entry["patched_syncs"] = _count_syncs(model, inputs)
            p50, mean = _median_ms(lambda: model(**inputs), reps=args.reps)
            entry["patched_prefill_p50_ms"] = p50
            entry["patched_prefill_mean_ms"] = mean

        entry["logits_bitwise_identical"] = bool(torch.equal(before_logits, after_logits))
        entry["max_abs_logit_delta"] = float((before_logits - after_logits).abs().max())
        entry["p50_improvement_pct"] = round(
            (entry["baseline_prefill_p50_ms"] - entry["patched_prefill_p50_ms"])
            / entry["baseline_prefill_p50_ms"]
            * 100.0,
            3,
        )
        entry["syncs_removed"] = entry["baseline_syncs"] - entry["patched_syncs"]
        report["samples"].append(entry)
        print(json.dumps(entry, indent=2))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
