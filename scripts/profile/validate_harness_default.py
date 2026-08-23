"""Harness-faithful validation of teammate default path (CUDA graph decode).

Clears all OPTIZ_QWEN_* env vars, constructs VLMModel directly, and calls
generate_with_metrics -- exactly what the scoring harness does.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import statistics
import time
from pathlib import Path

os.environ["OPTIZ_QWEN_PPU_SDK_BOOTSTRAP"] = "1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/root/optiz_teammate")
    ap.add_argument("--model-path", default="/mnt/nas/optiz-qwen/models/Qwen3.5-2B")
    ap.add_argument("--dataset", default="/mnt/nas/optiz-qwen/datasets/mmbench/mmbench_dev_en.tsv")
    ap.add_argument("--num-samples", type=int, default=50)
    ap.add_argument("--output", default="/root/harness_validate.json")
    args = ap.parse_args()

    # scrub all OPTIZ_QWEN_* env vars to prove default path (like harness)
    for key in list(os.environ):
        if key.startswith("OPTIZ_QWEN_"):
            del os.environ[key]
    os.environ["OPTIZ_QWEN_PPU_SDK_BOOTSTRAP"] = "1"

    import sys
    sys.path.insert(0, args.src)
    sys.path.insert(0, "/root/official_v12")
    from benchmark_public import build_prompt as official_build_prompt
    from optiz_qwen.evaluation.dndx_wrapper import VLMModel, GenerationConfig

    model = VLMModel(args.model_path, backend="transformers", device="auto")
    torch_mod = model._torch

    rows = list(csv.DictReader(open(args.dataset, encoding="utf-8"), delimiter="\t"))
    rows = rows[: args.num_samples]

    records = []
    for i, row in enumerate(rows):
        img = Image.open(io.BytesIO(base64.b64decode(row["image"]))).convert("RGB")
        options = {k: (row.get(k) or "").strip() for k in ["A", "B", "C", "D"]}
        from types import SimpleNamespace
        prompt = official_build_prompt(SimpleNamespace(
            question=row["question"],
            hint=row.get("hint") or "",
            choices=options,
            language="en",
        ))
        res = model.generate_with_metrics(
            image=img,
            prompt=prompt,
            choices={k: row[k] for k in ["A", "B", "C", "D"]},
            generation_config=GenerationConfig(max_new_tokens=256, temperature=0.0, top_p=1.0),
            sample_id=row["index"],
        )
        from optiz_qwen.evaluation.answer_parsing import extract_answer
        parsed = extract_answer(res.text)
        records.append({
            "question_id": row["index"],
            "parsed": parsed,
            "expected": row["answer"],
            "correct": parsed == row["answer"],
            "ttft_ms": round(res.ttft_seconds * 1000, 3),
            "throughput": round(res.token_count / res.elapsed_seconds, 3) if res.elapsed_seconds > 0 else None,
            "token_count": res.token_count,
            "meta": res.meta,
        })
        if i % 10 == 0:
            print(f"[{i}/{len(rows)}] done", flush=True)

    correct = sum(r["correct"] for r in records)
    ttfts = [r["ttft_ms"] for r in records]
    thrus = [r["throughput"] for r in records if r["throughput"]]
    summary = {
        "num_samples": len(records),
        "accuracy": round(correct / len(records), 6),
        "correct": correct,
        "ttft_ms": {"mean": round(statistics.mean(ttfts), 3), "median": round(statistics.median(ttfts), 3)},
        "throughput_tok_s": {"mean": round(statistics.mean(thrus), 3), "median": round(statistics.median(thrus), 3)},
        "sample_meta_example": records[0]["meta"] if records else None,
        "records": records,
    }
    Path(args.output).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    from PIL import Image
    main()
