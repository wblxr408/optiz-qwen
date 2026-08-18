"""Are the wrong answers position bias, or genuine content errors?

The observation this tests
-------------------------
Across the two saved PPU A/B artifacts (32 + 50 samples, 82 scored answers) the
error direction is perfectly one-sided::

    gold A -> pred A   30      (100.0% correct on gold-A items)
    gold B -> pred B   32
    gold B -> pred A   20      ( 61.5% correct on gold-B items)
    gold A -> pred B    0

Twenty errors, all in the same direction, none in the other.  Two explanations
fit that table and they need different fixes:

1. **Position bias** -- the model prefers the *first* option regardless of
   content.  Then gold-A accuracy is inflated for the wrong reason (it agrees
   with the bias by accident) and the fix is a debiasing or calibration step at
   the software layer, which costs no latency.
2. **Genuine content errors** -- the slice happens to be gold-B-heavy (52 vs
   30), so any concentration of hard items lands on gold B.  Then there is no
   cheap fix and accuracy is a model-capability ceiling.

The decisive experiment
-----------------------
Re-ask each question with the option *contents* swapped between the letters,
keeping everything else identical.  For a 2-option item whose gold is B, the
swapped item's gold is A.

    content-following  ->  the letter flips with the content, answer stays
                           semantically the same, and swapped accuracy is the
                           same as original accuracy
    position-following ->  the letter does not move, so the model answers "A"
                           both times and swapped accuracy jumps on exactly the
                           items that were wrong before

The summary reports ``letter_sticky_rate``: how often the predicted *letter* is
unchanged after the swap.  Content-following predicts ~0; pure position bias
predicts ~1.  Anything in between quantifies how much of the 20-error gap is
recoverable at the software layer.

Only the option block is permuted -- image, question, hint, instruction, decode
settings, and the answer parser are untouched, so a difference cannot come from
anywhere else.

Run on the target:
    PYTHONPATH=src:. python scripts/probe_option_position_bias.py --samples 50
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="./resources/model_weights/raw/Qwen3.5-2B")
    parser.add_argument(
        "--dataset-path",
        default="./resources/eval_dataset/raw/mmbench_public/mmbench_dev_en.tsv",
    )
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--sample-strategy",
        choices=["sequential", "stratified"],
        default="sequential",
        help="sequential reproduces the slice the 0.76 figure came from.",
    )
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--output", default="benchmarks/output/ppu_option_position_bias.json")
    return parser.parse_args()


def swap_choices(sample: Any) -> tuple[Any, dict[str, str]]:
    """Reverse the option contents across the populated letters.

    Returns the rewritten sample plus the ``new_letter -> old_letter`` map, which
    is what turns a prediction on the swapped item back into a prediction about
    the original content.
    """

    letters = [key for key in ("A", "B", "C", "D") if (sample.choices.get(key) or "").strip()]
    rotated = list(reversed(letters))
    choices = dict(sample.choices)
    back_map: dict[str, str] = {}
    for new_letter, old_letter in zip(letters, rotated):
        choices[new_letter] = sample.choices[old_letter]
        back_map[new_letter] = old_letter
    swapped_gold = next(
        (new for new, old in back_map.items() if old == sample.answer), sample.answer
    )
    return replace(sample, choices=choices, answer=swapped_gold), back_map


def main() -> None:
    args = parse_args()

    from optiz_qwen.evaluation.answer_parsing import extract_answer
    from optiz_qwen.evaluation.dndx_public_benchmark import (
        build_prompt,
        decode_image,
        load_mmbench_tsv,
        select_samples,
    )
    from optiz_qwen.evaluation.dndx_wrapper import GenerationConfig, VLMModel

    samples = select_samples(
        load_mmbench_tsv(Path(args.dataset_path)),
        limit=args.samples,
        strategy=args.sample_strategy,
        seed=args.seed,
    )

    model = VLMModel(args.model_path, backend="transformers", device="auto")
    generation_config = GenerationConfig(max_new_tokens=args.max_new_tokens)

    records: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        image = decode_image(sample.image_b64)
        swapped, back_map = swap_choices(sample)

        entry: dict[str, Any] = {
            "id": sample.sample_id,
            "n_options": sum(1 for v in sample.choices.values() if v.strip()),
            "gold": sample.answer,
            "swapped_gold": swapped.answer,
        }
        for label, item in (("original", sample), ("swapped", swapped)):
            result = model.generate_with_metrics(
                image=image,
                prompt=build_prompt(item),
                choices=item.choices,
                generation_config=generation_config,
                sample_id=item.sample_id,
            )
            text = result.text
            parsed = extract_answer(text)
            entry[label] = {
                "parsed": parsed,
                "correct": parsed == item.answer,
                "text": text[:200],
            }
        # Where the swapped prediction points in the *original* lettering; this
        # is the comparison that separates content from position.
        entry["swapped_parsed_as_original_letter"] = back_map.get(
            entry["swapped"]["parsed"] or "", entry["swapped"]["parsed"]
        )
        entry["letter_sticky"] = entry["original"]["parsed"] == entry["swapped"]["parsed"]
        entry["content_stable"] = (
            entry["original"]["parsed"] == entry["swapped_parsed_as_original_letter"]
        )
        records.append(entry)
        print(
            f"[{index + 1}/{len(samples)}] id={sample.sample_id} "
            f"gold={sample.answer}->{swapped.answer} "
            f"pred={entry['original']['parsed']}->{entry['swapped']['parsed']} "
            f"sticky={entry['letter_sticky']}"
        )

    scored = [r for r in records if r["original"]["parsed"] and r["swapped"]["parsed"]]
    n = len(scored) or 1
    summary = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "samples": len(records),
        "scored": len(scored),
        "unparsed_original": sum(1 for r in records if not r["original"]["parsed"]),
        "unparsed_swapped": sum(1 for r in records if not r["swapped"]["parsed"]),
        "accuracy_original": round(
            sum(1 for r in records if r["original"]["correct"]) / len(records), 4
        ),
        "accuracy_swapped": round(
            sum(1 for r in records if r["swapped"]["correct"]) / len(records), 4
        ),
        # ~1.0 => the letter does not move when the content does: position bias.
        # ~0.0 => the letter tracks the content: genuine errors, no cheap fix.
        "letter_sticky_rate": round(sum(1 for r in scored if r["letter_sticky"]) / n, 4),
        "content_stable_rate": round(sum(1 for r in scored if r["content_stable"]) / n, 4),
        "predicted_letters_original": dict(
            Counter(r["original"]["parsed"] for r in records).most_common()
        ),
        "predicted_letters_swapped": dict(
            Counter(r["swapped"]["parsed"] for r in records).most_common()
        ),
        # Both-wrong is the residue no permutation trick can recover.
        "correct_both": sum(1 for r in records if r["original"]["correct"] and r["swapped"]["correct"]),
        "correct_neither": sum(
            1 for r in records if not r["original"]["correct"] and not r["swapped"]["correct"]
        ),
        "max_new_tokens": args.max_new_tokens,
        "sample_strategy": args.sample_strategy,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"summary": summary, "records": records}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
