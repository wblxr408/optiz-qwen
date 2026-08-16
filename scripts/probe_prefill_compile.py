"""Can ``torch.compile`` cut PPU prefill's framework overhead?

The gap this targets
--------------------
Measured on the target (``scripts/profile_ppu_prefill_headroom.py``):

    wall                 53.4 ms
    kernel device time   27.3 ms   (device_busy_fraction 0.51)
    idle                 26.0 ms
    kernel launches      2423
    operator calls       13698

2423 launches at the measured ~4.5 us torch launch cost is only ~11 ms, so the
26 ms of idle is not launch cost alone -- it is dispatcher and Python overhead
spread over 13698 operator calls.  ``scripts/probe_vision_sync_elision.py``
already ruled out host syncs (72 of 93 removed, ~1% won).

A CUDA graph would remove all of it, but capture needs fixed shapes and prefill
has neither: 46 distinct prompt lengths and 24 distinct vision grids over the
50-sample set.  ``torch.compile`` with ``dynamic=True`` is the one lever that
attacks framework overhead without fixing shapes -- it fuses pointwise chains,
so both the operator count and the launch count drop.

D1 in ``docs/ppu_optimization_design.md`` rejected ``torch.compile(mode=
"reduce-overhead")`` for *decode*, because Inductor's cudagraph trees raised
``InternalTorchDynamoError: accessing tensor output of CUDAGraphs that has been
overwritten by a subsequent run``.  That failure was specifically about cudagraph
trees.  Prefill wants the opposite configuration -- no cudagraphs, dynamic shapes
-- so the rejection does not transfer and this is worth measuring.

Arms, each independently reported so a partial win is still usable:

    baseline          eager
    compile-language  torch.compile(model.model.language_model, dynamic=True)
    compile-vision    torch.compile(model.model.visual, dynamic=True)

Correctness is checked against eager, not assumed: last-position logits must be
bit-identical, and the greedy first token must match.

Run on the target:
    PYTHONPATH=src:. python scripts/probe_prefill_compile.py
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import torch


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timings_ms(fn, *, reps: int) -> dict[str, float]:
    samples = []
    for _ in range(reps):
        _sync()
        start = time.perf_counter()
        fn()
        _sync()
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return {
        "p50_ms": round(samples[len(samples) // 2], 3),
        "mean_ms": round(sum(samples) / len(samples), 3),
        "min_ms": round(samples[0], 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="./resources/model_weights/raw/Qwen3.5-2B")
    parser.add_argument(
        "--dataset-path",
        default="./resources/eval_dataset/raw/mmbench_public/mmbench_dev_en.tsv",
    )
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument(
        "--target",
        choices=("language", "vision", "both"),
        default="language",
        help="which submodule to compile in this run",
    )
    parser.add_argument("--output", default="benchmarks/output/ppu_prefill_compile.json")
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


def _submodules(model: Any, target: str) -> list[tuple[str, Any, str]]:
    """(label, owner, attribute) triples to compile, in place."""

    inner = model.model
    picks = []
    if target in {"language", "both"}:
        picks.append(("language_model", inner, "language_model"))
    if target in {"vision", "both"}:
        picks.append(("visual", inner, "visual"))
    return picks


def main() -> None:
    args = parse_args()
    model, batches = build_model_and_inputs(args)

    report: dict[str, Any] = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "purpose": "does dynamic-shape torch.compile cut prefill framework overhead",
        "target": args.target,
        "samples": [],
    }

    # Eager reference first, so a compile failure still leaves usable numbers.
    eager: list[dict[str, Any]] = []
    for index, inputs in enumerate(batches):
        with torch.inference_mode():
            logits = model(**inputs).logits[:, -1, :].clone()
        entry = {
            "index": index,
            "prompt_tokens": int(inputs["input_ids"].shape[-1]),
            "eager": _timings_ms(lambda: model(**inputs), reps=args.reps),
            "_logits": logits,
        }
        eager.append(entry)
        print("eager", entry["prompt_tokens"], entry["eager"])

    picks = _submodules(model, args.target)
    originals = [(owner, attribute, getattr(owner, attribute)) for _, owner, attribute in picks]
    compiled_labels = [label for label, _, _ in picks]

    compile_error = None
    compile_seconds = None
    try:
        start = time.perf_counter()
        for _, owner, attribute in picks:
            setattr(
                owner,
                attribute,
                torch.compile(getattr(owner, attribute), dynamic=True),
            )
        # First forward pays the compile; time it separately from steady state.
        with torch.inference_mode():
            model(**batches[0])
        _sync()
        compile_seconds = round(time.perf_counter() - start, 2)
        print(f"compiled {compiled_labels} in {compile_seconds}s")
    except Exception as exc:  # pragma: no cover - target-specific
        compile_error = f"{type(exc).__name__}: {exc}"
        print("COMPILE FAILED:", compile_error)
        print(traceback.format_exc()[-2000:])

    if compile_error is None:
        for entry, inputs in zip(eager, batches):
            try:
                with torch.inference_mode():
                    logits = model(**inputs).logits[:, -1, :].clone()
                entry["compiled"] = _timings_ms(lambda: model(**inputs), reps=args.reps)
                entry["logits_bitwise_identical"] = bool(torch.equal(entry["_logits"], logits))
                entry["max_abs_logit_delta"] = float((entry["_logits"] - logits).abs().max())
                entry["first_token_match"] = bool(
                    torch.equal(
                        torch.argmax(entry["_logits"], dim=-1), torch.argmax(logits, dim=-1)
                    )
                )
                entry["p50_improvement_pct"] = round(
                    (entry["eager"]["p50_ms"] - entry["compiled"]["p50_ms"])
                    / entry["eager"]["p50_ms"]
                    * 100.0,
                    3,
                )
                print("compiled", entry["prompt_tokens"], entry["compiled"],
                      "impr", entry["p50_improvement_pct"], "%")
            except Exception as exc:  # pragma: no cover - target-specific
                entry["compiled_error"] = f"{type(exc).__name__}: {exc}"
                print("compiled forward FAILED:", entry["compiled_error"])

    for owner, attribute, original in originals:
        setattr(owner, attribute, original)

    report["compile_seconds"] = compile_seconds
    report["compile_error"] = compile_error
    report["compiled_modules"] = compiled_labels
    report["samples"] = [{k: v for k, v in e.items() if not k.startswith("_")} for e in eager]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
