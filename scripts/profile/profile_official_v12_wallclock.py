"""Wall-clock profiling of OFFICIAL v1.2 wrapper using register_forward_hook.

No forward replacement, no cuda.synchronize inside the generation thread:
the hook records perf_counter deltas (CPU dispatch dominated on this HW).
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


def run_sample(model, row, decode_steps: int) -> dict:
    from transformers import TextIteratorStreamer

    processor = model._processor
    prompt = build_prompt(row)
    image = decode_image(row["image"])

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": prompt},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model._model.device)

    streamer = TextIteratorStreamer(
        processor.tokenizer, skip_prompt=True, skip_special_tokens=True,
    )
    gen_kwargs = {
        **inputs,
        "max_new_tokens": decode_steps,
        "temperature": 0.0, "top_p": 1.0, "do_sample": False,
        "use_cache": True, "streamer": streamer,
    }

    stats = {"prompt_tokens": inputs.input_ids.shape[1], "prefill_ms": None,
             "decode_ms": [], "n_forward": 0}
    outer = model._model
    prefill_ts: list[float] = []
    decode_ts: list[float] = []
    lock = threading.Lock()

    def hook(module, fargs, fkwargs):
        ids = fkwargs.get("input_ids") if fkwargs else (fargs[0] if fargs else None)
        is_decode = ids is not None and ids.dim() >= 2 and ids.shape[1] == 1
        ts = time.perf_counter()
        with lock:
            (decode_ts if is_decode else prefill_ts).append(ts)

    handle = outer.register_forward_hook(hook)
    start = time.perf_counter()
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
        handle.remove()

    # prefill duration: time of the last prefill forward (hook is pre-hook, use next decode start)
    if prefill_ts and decode_ts:
        stats["prefill_ms"] = (decode_ts[0] - prefill_ts[0]) * 1000.0
    elif prefill_ts:
        stats["prefill_ms"] = (end - prefill_ts[0]) * 1000.0
    if decode_ts:
        ts_all = decode_ts + [end]
        stats["decode_ms"] = [(ts_all[i+1] - ts_all[i]) * 1000.0 for i in range(len(decode_ts))]
    stats["ttft_streamer_ms"] = (first_chunk_at - start) * 1000.0 if first_chunk_at else None
    stats["elapsed_s"] = end - start
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--official-dir", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--num-samples", type=int, default=3)
    ap.add_argument("--decode-steps", type=int, default=16)
    ap.add_argument("--output", default="/root/official_wallclock_profile.json")
    args = ap.parse_args()

    GenerationConfig, VLMModel = load_official_wrapper(args.official_dir)
    model = VLMModel(args.model_path, backend="transformers", device="auto")
    torch_mod = model._torch

    rows = list(csv.DictReader(open(args.dataset, encoding="utf-8"), delimiter="\t"))
    rows = rows[: args.num_samples]

    run_sample(model, rows[0], 4)  # warmup

    per_sample = []
    for row in rows[1:]:
        s = run_sample(model, row, args.decode_steps)
        per_sample.append({"index": row["index"], **s})

    prefill = [s["prefill_ms"] for s in per_sample if s.get("prefill_ms")]
    dec = [v for s in per_sample for v in s["decode_ms"]]
    weights_gb = sum(p.stat().st_size for p in Path(args.model_path).rglob("*.safetensors")) / 1e9
    hbm_bw_tbs = 2.0
    hbm_floor_ms = weights_gb * 2 / hbm_bw_tbs

    summary = {
        "backend": "official_v1.2_wrapper_untouched",
        "torch": torch_mod.__version__,
        "samples": per_sample,
        "prefill_ms": {"mean": statistics.mean(prefill) if prefill else None, "samples": prefill},
        "decode_step_ms": {
            "mean": statistics.mean(dec) if dec else None,
            "median": statistics.median(dec) if dec else None,
            "count": len(dec),
            "samples": dec[:20],
        },
        "decode_tokens_per_s": 1000.0 / statistics.mean(dec) if dec else None,
        "hbm": {
            "weights_gb": round(weights_gb, 3),
            "floor_ms_per_token": round(hbm_floor_ms, 3),
            "theoretical_tokens_per_s": round(1000.0 / hbm_floor_ms, 1),
        },
    }
    Path(args.output).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
