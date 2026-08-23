"""Profile the OFFICIAL dndx_participant-v1.2 wrapper on PPU.

Measures, on the untouched official code path:
  - prefill wall time (first token latency)
  - per-decode-step wall time, CUDA kernel count, CPU dispatch fraction
  - HBM theoretical floor vs measured decode step

Usage:
  python3 profile_official_bottleneck.py \
    --official-dir /path/to/dndx_participant-v1.2 \
    --model-path /path/to/Qwen3.5-2B \
    --dataset /path/to/mmbench_dev_en.tsv
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import statistics
import sys
import threading
import time
from pathlib import Path

import torch
from PIL import Image
from torch.profiler import profile, ProfilerActivity


def load_official_wrapper(module_dir: str):
    sys.path.insert(0, module_dir)
    from evaluation_wrapper import GenerationConfig, VLMModel
    return GenerationConfig, VLMModel


def decode_image(image_b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")


def build_prompt(row: dict) -> str:
    options = "\n".join(f"{k}. {row[k]}" for k in ["A", "B", "C", "D"] if row.get(k))
    return (
        "Solve this single-choice question. Your response must make one final choice among A/B/C/D clearly.\n"
        f"Question: {row['question']}\n{options}"
    )


def prof_step_stats(prof) -> dict:
    evs = prof.key_averages()
    cuda = [e for e in evs if e.device_type == ProfilerActivity.CUDA and e.self_device_time_total > 0]
    cpu_us = sum(e.self_cpu_time_total for e in evs if e.device_type == ProfilerActivity.CPU)
    cuda_us = sum(e.self_device_time_total for e in evs if e.device_type == ProfilerActivity.CUDA)
    return {"n_kernels": len(cuda), "cpu_self_us": cpu_us, "cuda_self_us": cuda_us}


def run_measured_sample(model, row, decode_steps: int, out: dict) -> None:
    from transformers import TextIteratorStreamer

    torch_mod = model._torch
    processor = model._processor
    prompt = build_prompt(row)
    image = decode_image(row["image"])
    choices = {k: row[k] for k in ["A", "B", "C", "D"]}

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": prompt},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model._model.device)

    streamer = TextIteratorStreamer(
        model._processor.tokenizer, skip_prompt=True, skip_special_tokens=True,
    )
    gen_kwargs = {
        **inputs,
        "max_new_tokens": decode_steps,
        "temperature": 0.0,
        "top_p": 1.0,
        "do_sample": False,
        "use_cache": True,
        "streamer": streamer,
    }

    stats = {"prefill_wall_ms": None, "decode_walls": [], "n_kernels": [], "issue_fracs": []}
    outer = model._model
    original_forward = outer.forward
    start = time.perf_counter()

    def wrapper(*fargs, **fkwargs):
        ids = fkwargs.get("input_ids") if fkwargs else (fargs[0] if fargs else None)
        is_decode = ids is not None and ids.dim() >= 2 and ids.shape[1] == 1
        with torch.no_grad():
            if is_decode:
                t0 = time.perf_counter()
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    out_ = original_forward(*fargs, **fkwargs)
                torch.cuda.synchronize()
                wall_ms = (time.perf_counter() - t0) * 1000.0
                s = prof_step_stats(prof)
                stats["decode_walls"].append(wall_ms)
                stats["n_kernels"].append(s["n_kernels"])
                stats["issue_fracs"].append(s["cpu_self_us"] / 1000.0 / wall_ms if wall_ms > 0 else 0.0)
            else:
                t0 = time.perf_counter()
                out_ = original_forward(*fargs, **fkwargs)
                torch.cuda.synchronize()
                stats["prefill_wall_ms"] = (time.perf_counter() - t0) * 1000.0
        return out_

    outer.forward = wrapper
    try:
        output_holder: dict = {}

        def _run():
            with torch.no_grad():
                output_holder["output_ids"] = outer.generate(**gen_kwargs)

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        chunks = []
        first_chunk_at = None
        for chunk in streamer:
            now = time.perf_counter()
            if first_chunk_at is None and chunk:
                first_chunk_at = now
            chunks.append(chunk)
        worker.join()
        end = time.perf_counter()
    finally:
        outer.forward = original_forward

    if first_chunk_at is not None:
        stats["ttft_streamer_ms"] = (first_chunk_at - start) * 1000.0
    stats["elapsed_s"] = end - start
    out.update(stats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--official-dir", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--num-samples", type=int, default=3)
    ap.add_argument("--decode-steps", type=int, default=8)
    ap.add_argument("--output", default="/root/official_bottleneck_profile.json")
    args = ap.parse_args()

    GenerationConfig, VLMModel = load_official_wrapper(args.official_dir)
    model = VLMModel(args.model_path, backend="transformers", device="auto")
    torch_mod = model._torch

    rows = list(csv.DictReader(open(args.dataset, encoding="utf-8"), delimiter="\t"))
    rows = rows[: args.num_samples]

    # warmup (kernel compile, excluded from stats)
    run_measured_sample(model, rows[0], 4, {})
    torch.cuda.synchronize()

    sample_stats = []
    for row in rows[1:]:
        out = {}
        run_measured_sample(model, row, args.decode_steps, out)
        sample_stats.append({"index": row["index"], **out})

    decode_walls = [v for s in sample_stats for v in s["decode_walls"]]
    decode_kernels = [v for s in sample_stats for v in s["n_kernels"]]
    decode_issue = [v for s in sample_stats for v in s["issue_fracs"]]
    prefill_walls = [s["prefill_wall_ms"] for s in sample_stats if s.get("prefill_wall_ms")]

    weights_gb = sum(p.stat().st_size for p in Path(args.model_path).rglob("*.safetensors")) / 1e9
    hbm_bw_tbs = 2.0
    hbm_floor_ms = weights_gb * 2 / hbm_bw_tbs

    summary = {
        "backend": "official_v1.2_wrapper_untouched",
        "model_path": args.model_path,
        "torch": torch_mod.__version__,
        "samples_profiled": len(sample_stats),
        "per_sample": sample_stats,
        "prefill_wall_ms": {
            "mean": statistics.mean(prefill_walls) if prefill_walls else None,
            "samples": prefill_walls,
        },
        "decode_step_wall_ms": {
            "mean": statistics.mean(decode_walls) if decode_walls else None,
            "median": statistics.median(decode_walls) if decode_walls else None,
            "count": len(decode_walls),
        },
        "decode_kernels_per_step": {
            "mean": statistics.mean(decode_kernels) if decode_kernels else None,
            "count": len(decode_kernels),
        },
        "decode_cpu_issue_fraction": {
            "mean": statistics.mean(decode_issue) if decode_issue else None,
            "count": len(decode_issue),
        },
        "hbm": {
            "weights_gb": round(weights_gb, 3),
            "assumed_bw_tbs": hbm_bw_tbs,
            "floor_ms_per_token": round(hbm_floor_ms, 3),
            "theoretical_tokens_per_s": round(1000.0 / hbm_floor_ms, 1),
        },
    }
    Path(args.output).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
