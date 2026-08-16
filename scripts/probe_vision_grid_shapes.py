"""Are the vision tower's shapes fixed across MMBench samples?

Why this matters
----------------
``scripts/profile_ppu_prefill_headroom.py`` measured prefill's device-busy
fraction at 0.47-0.51: about half the wall clock is the device idle while the CPU
works.  ``scripts/probe_vision_sync_elision.py`` then ruled out host syncs as the
cause -- removing 72 of 93 syncs bought ~1%.  So the CPU cost is framework
overhead spread across ~13.7k operator calls, and the only thing that removes
framework overhead wholesale is capturing the region as a CUDA graph.

Capture needs fixed shapes.  The *language* stack cannot have them: prompt
lengths ranged 137-363 over the 50-sample set with 46 distinct values.  But the
vision tower's shapes come from ``image_grid_thw``, not from the prompt, and the
processor may well resize every image to the same grid.  If it does, the tower is
capturable with no bucketing at all.

This probe needs no model -- processor only -- so it is cheap to run.

Run on the target (or anywhere the processor and dataset are available):
    PYTHONPATH=src:. python scripts/probe_vision_grid_shapes.py --samples 50
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="./resources/model_weights/raw/Qwen3.5-2B")
    parser.add_argument(
        "--dataset-path",
        default="./resources/eval_dataset/raw/mmbench_public/mmbench_dev_en.tsv",
    )
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--output", default="benchmarks/output/ppu_vision_grid_shapes.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from transformers import AutoProcessor

    from optiz_qwen.evaluation.dndx_public_benchmark import (
        build_prompt,
        decode_image,
        load_mmbench_tsv,
    )

    processor = AutoProcessor.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True
    )
    samples = load_mmbench_tsv(Path(args.dataset_path), limit=args.samples)

    grids: Counter[tuple[int, ...]] = Counter()
    pixel_shapes: Counter[tuple[int, ...]] = Counter()
    prompt_lengths: Counter[int] = Counter()
    rows: list[dict[str, Any]] = []

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
        )
        grid = tuple(int(v) for v in inputs["image_grid_thw"].flatten().tolist())
        pixel = tuple(int(v) for v in inputs["pixel_values"].shape)
        plen = int(inputs["input_ids"].shape[-1])
        grids[grid] += 1
        pixel_shapes[pixel] += 1
        prompt_lengths[plen] += 1
        rows.append({"id": sample.sample_id, "grid_thw": grid, "pixel_values": pixel, "plen": plen})

    report = {
        "samples": len(rows),
        "distinct_grids": len(grids),
        "distinct_pixel_shapes": len(pixel_shapes),
        "distinct_prompt_lengths": len(prompt_lengths),
        "grid_counts": [{"grid_thw": list(g), "count": c} for g, c in grids.most_common()],
        "pixel_shape_counts": [
            {"pixel_values": list(s), "count": c} for s, c in pixel_shapes.most_common(8)
        ],
        # The verdict this probe exists for: a single vision shape means the tower
        # is CUDA-graph capturable with no bucketing.
        "vision_shapes_fixed": len(pixel_shapes) == 1 and len(grids) == 1,
        "records": rows,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2)[:3000])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
