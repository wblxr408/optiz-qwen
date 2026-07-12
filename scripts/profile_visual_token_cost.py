"""Profile the visual-path cost of Qwen3.5 on paired image/text prefills."""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import statistics
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_DATASET_PATH = "./resources/eval_dataset/raw/mmbench_public/mmbench_dev_en.tsv"
DEFAULT_MODEL_PATH = "./resources/model_weights/raw/Qwen3.5-2B"
DEFAULT_OUTPUT_PATH = "./benchmarks/output/visual_token_cost_profile_en50_mps.json"
DISABLED_OPTIMIZATION_ENV_KEYS = (
    "OPTIZ_QWEN_KIVI_KV_CACHE",
    "OPTIZ_QWEN_KV_CHAIN_ENABLED",
    "OPTIZ_QWEN_TOME_ENABLED",
    "OPTIZ_QWEN_VISUAL_TOKEN_PRUNING",
)


@dataclass(frozen=True)
class ProfileSample:
    question_id: str
    image_width: int
    image_height: int
    image_tokens: int
    multimodal_input_tokens: int
    text_only_input_tokens: int
    multimodal_processor_ms: float
    text_only_processor_ms: float
    multimodal_transfer_ms: float
    text_only_transfer_ms: float
    multimodal_forward_ms: float
    text_only_forward_ms: float
    vision_encoder_ms: float
    multimodal_language_model_ms: float
    text_only_language_model_ms: float
    multimodal_other_ms: float
    text_only_other_ms: float
    visual_path_delta_ms: float
    language_model_visual_token_delta_ms: float
    multimodal_decode_ms: float
    text_only_decode_ms: float
    decode_visual_cache_delta_ms: float
    multimodal_decode_ms_per_token: float
    text_only_decode_ms_per_token: float


class ForwardTimer:
    def __init__(self, module: Any, synchronize) -> None:
        self.module = module
        self.synchronize = synchronize
        self.elapsed_ms = 0.0
        self.calls = 0
        self.call_records: list[tuple[int | None, float]] = []
        self._original_forward = module.forward

    def install(self) -> None:
        timer = self

        def timed_forward(_module, *args, **kwargs):
            sequence = kwargs.get("inputs_embeds")
            if sequence is None:
                sequence = kwargs.get("input_ids")
            if sequence is None and args:
                sequence = args[0]
            sequence_length = int(sequence.shape[-2]) if sequence is not None else None
            timer.synchronize()
            start = time.perf_counter()
            result = timer._original_forward(*args, **kwargs)
            timer.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            timer.elapsed_ms += elapsed_ms
            timer.calls += 1
            timer.call_records.append((sequence_length, elapsed_ms))
            return result

        self.module.forward = types.MethodType(timed_forward, self.module)

    def reset(self) -> None:
        self.elapsed_ms = 0.0
        self.calls = 0
        self.call_records = []

    def restore(self) -> None:
        self.module.forward = self._original_forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--warmup-samples", type=int, default=2)
    parser.add_argument(
        "--decode-steps",
        type=int,
        default=0,
        help="Cached decode steps to profile; keep at 0 for the stable prefill profile.",
    )
    parser.add_argument("--device", default="mps")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def settle_runtime(device: torch.device) -> None:
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    time.sleep(0.01)


def load_rows(path: Path, limit: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"No samples found in {path}")
    return rows[:limit]


def decode_image(image_b64: str) -> Image.Image:
    image = Image.open(io.BytesIO(base64.b64decode(image_b64)))
    return image.convert("RGB")


def build_prompt(row: dict[str, str], language: str) -> str:
    choices = "\n".join(
        f"{key}. {(row.get(key) or '').strip()}"
        for key in ("A", "B", "C", "D")
        if (row.get(key) or "").strip()
    )
    hint = (row.get("hint") or "").strip()
    hint_line = f"Hint: {hint}\n" if hint else ""
    if language == "cn":
        instruction = (
            "请完成这道单选题。请给出你认为正确的选项，并可附带一句简短理由。"
            "答案必须明确，且只能对应 A/B/C/D 中的一个选项。"
        )
    else:
        instruction = (
            "Solve this single-choice question."
            " Your response must make one final choice among A/B/C/D clearly."
            " You may include one short reason."
        )
    return (
        f"{instruction}\n{hint_line}"
        f"Question: {(row.get('question') or '').strip()}\n{choices}\n"
    )


def prepare_inputs(
    processor: Any,
    *,
    prompt: str,
    image: Image.Image | None,
    device: torch.device,
) -> tuple[Any, float, float]:
    content: list[dict[str, Any]] = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    processor_start = time.perf_counter()
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    processor_ms = (time.perf_counter() - processor_start) * 1000.0

    synchronize(device)
    transfer_start = time.perf_counter()
    inputs = inputs.to(device)
    synchronize(device)
    transfer_ms = (time.perf_counter() - transfer_start) * 1000.0
    return inputs, processor_ms, transfer_ms


def run_prefill(
    model: Any,
    inputs: Any,
    *,
    device: torch.device,
    vision_timer: ForwardTimer,
    language_timer: ForwardTimer,
) -> tuple[float, float, float]:
    vision_timer.reset()
    language_timer.reset()
    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(
            **inputs,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
    synchronize(device)
    forward_ms = (time.perf_counter() - start) * 1000.0
    del outputs
    return forward_ms, vision_timer.elapsed_ms, language_timer.elapsed_ms


def run_fixed_length_generation(
    model: Any,
    inputs: Any,
    *,
    decode_steps: int,
    device: torch.device,
    vision_timer: ForwardTimer,
    language_timer: ForwardTimer,
) -> tuple[float, float, float, int]:
    if decode_steps == 0:
        return 0.0, 0.0, 0.0, 0
    if decode_steps < 2:
        raise ValueError("decode_steps must be 0 or at least 2.")
    vision_timer.reset()
    language_timer.reset()
    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            use_cache=True,
            min_new_tokens=decode_steps,
            max_new_tokens=decode_steps,
        )
    synchronize(device)
    generation_ms = (time.perf_counter() - start) * 1000.0
    generated_tokens = int(output_ids.shape[1] - inputs.input_ids.shape[1])
    if generated_tokens != decode_steps:
        raise RuntimeError(
            f"Expected {decode_steps} generated tokens, received {generated_tokens}."
        )
    prefill_records = [elapsed for length, elapsed in language_timer.call_records if length and length > 1]
    decode_records = [elapsed for length, elapsed in language_timer.call_records if length == 1]
    if len(prefill_records) != 1:
        raise RuntimeError(f"Expected one language-model prefill call, received {len(prefill_records)}.")
    if len(decode_records) != decode_steps - 1:
        raise RuntimeError(
            f"Expected {decode_steps - 1} cached decode calls, received {len(decode_records)}."
        )
    del output_ids
    return generation_ms, sum(prefill_records), sum(decode_records), len(decode_records)


def profile_mode(
    model: Any,
    processor: Any,
    *,
    prompt: str,
    image: Image.Image | None,
    device: torch.device,
    vision_timer: ForwardTimer,
    language_timer: ForwardTimer,
    decode_steps: int,
) -> dict[str, float | int]:
    inputs, processor_ms, transfer_ms = prepare_inputs(
        processor,
        prompt=prompt,
        image=image,
        device=device,
    )
    input_tokens = int(inputs.input_ids.shape[1])
    image_tokens = int((inputs.input_ids == processor.image_token_id).sum().item())
    forward_ms, vision_ms, language_ms = run_prefill(
        model,
        inputs,
        device=device,
        vision_timer=vision_timer,
        language_timer=language_timer,
    )
    settle_runtime(device)
    generation_ms, generation_prefill_ms, decode_ms, measured_decode_steps = (
        run_fixed_length_generation(
            model,
            inputs,
            decode_steps=decode_steps,
            device=device,
            vision_timer=vision_timer,
            language_timer=language_timer,
        )
    )
    del inputs
    settle_runtime(device)
    return {
        "input_tokens": input_tokens,
        "image_tokens": image_tokens,
        "processor_ms": processor_ms,
        "transfer_ms": transfer_ms,
        "forward_ms": forward_ms,
        "vision_ms": vision_ms,
        "language_ms": language_ms,
        "other_ms": max(forward_ms - vision_ms - language_ms, 0.0),
        "generation_ms": generation_ms,
        "generation_prefill_ms": generation_prefill_ms,
        "decode_ms": decode_ms,
        "measured_decode_steps": measured_decode_steps,
    }


def profile_sample(
    model: Any,
    processor: Any,
    *,
    row: dict[str, str],
    language: str,
    device: torch.device,
    vision_timer: ForwardTimer,
    language_timer: ForwardTimer,
    image_first: bool,
    decode_steps: int,
) -> ProfileSample:
    image = decode_image(row["image"])
    prompt = build_prompt(row, language)
    modes = ("multimodal", "text_only") if image_first else ("text_only", "multimodal")
    results: dict[str, dict[str, float | int]] = {}
    for mode in modes:
        results[mode] = profile_mode(
            model,
            processor,
            prompt=prompt,
            image=image if mode == "multimodal" else None,
            device=device,
            vision_timer=vision_timer,
            language_timer=language_timer,
            decode_steps=decode_steps,
        )

    multimodal = results["multimodal"]
    text_only = results["text_only"]
    if int(text_only["image_tokens"]) != 0:
        raise RuntimeError("Text-only control unexpectedly contains image tokens.")
    if int(multimodal["image_tokens"]) <= 0:
        raise RuntimeError("Multimodal input does not contain image tokens.")

    return ProfileSample(
        question_id=str(row["index"]),
        image_width=image.width,
        image_height=image.height,
        image_tokens=int(multimodal["image_tokens"]),
        multimodal_input_tokens=int(multimodal["input_tokens"]),
        text_only_input_tokens=int(text_only["input_tokens"]),
        multimodal_processor_ms=float(multimodal["processor_ms"]),
        text_only_processor_ms=float(text_only["processor_ms"]),
        multimodal_transfer_ms=float(multimodal["transfer_ms"]),
        text_only_transfer_ms=float(text_only["transfer_ms"]),
        multimodal_forward_ms=float(multimodal["forward_ms"]),
        text_only_forward_ms=float(text_only["forward_ms"]),
        vision_encoder_ms=float(multimodal["vision_ms"]),
        multimodal_language_model_ms=float(multimodal["language_ms"]),
        text_only_language_model_ms=float(text_only["language_ms"]),
        multimodal_other_ms=float(multimodal["other_ms"]),
        text_only_other_ms=float(text_only["other_ms"]),
        visual_path_delta_ms=float(multimodal["forward_ms"] - text_only["forward_ms"]),
        language_model_visual_token_delta_ms=float(
            multimodal["language_ms"] - text_only["language_ms"]
        ),
        multimodal_decode_ms=float(multimodal["decode_ms"]),
        text_only_decode_ms=float(text_only["decode_ms"]),
        decode_visual_cache_delta_ms=float(multimodal["decode_ms"] - text_only["decode_ms"]),
        multimodal_decode_ms_per_token=float(
            multimodal["decode_ms"] / max(int(multimodal["measured_decode_steps"]), 1)
        ),
        text_only_decode_ms_per_token=float(
            text_only["decode_ms"] / max(int(text_only["measured_decode_steps"]), 1)
        ),
    )


def mean(samples: list[ProfileSample], field: str) -> float:
    return statistics.fmean(float(getattr(sample, field)) for sample in samples)


def percentile(samples: list[ProfileSample], field: str, q: float) -> float:
    values = [float(getattr(sample, field)) for sample in samples]
    return float(np.percentile(values, q))


def correlation(samples: list[ProfileSample], field: str) -> float | None:
    x = np.asarray([sample.image_tokens for sample in samples], dtype=np.float64)
    y = np.asarray([float(getattr(sample, field)) for sample in samples], dtype=np.float64)
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def rounded_correlation(samples: list[ProfileSample], field: str) -> float | None:
    value = correlation(samples, field)
    return round(value, 6) if value is not None else None


def build_summary(samples: list[ProfileSample]) -> dict[str, Any]:
    fields = (
        "image_tokens",
        "multimodal_processor_ms",
        "text_only_processor_ms",
        "multimodal_transfer_ms",
        "text_only_transfer_ms",
        "multimodal_forward_ms",
        "text_only_forward_ms",
        "vision_encoder_ms",
        "multimodal_language_model_ms",
        "text_only_language_model_ms",
        "multimodal_other_ms",
        "text_only_other_ms",
        "visual_path_delta_ms",
        "language_model_visual_token_delta_ms",
        "multimodal_decode_ms",
        "text_only_decode_ms",
        "decode_visual_cache_delta_ms",
        "multimodal_decode_ms_per_token",
        "text_only_decode_ms_per_token",
    )
    averages = {field: round(mean(samples, field), 3) for field in fields}
    p50 = {field: round(percentile(samples, field, 50), 3) for field in fields}
    p95 = {field: round(percentile(samples, field, 95), 3) for field in fields}
    multimodal_forward = averages["multimodal_forward_ms"]
    visual_total = averages["visual_path_delta_ms"]
    return {
        "sample_count": len(samples),
        "average": averages,
        "p50": p50,
        "p95": p95,
        "share_of_multimodal_forward_pct": {
            "vision_encoder": round(averages["vision_encoder_ms"] / multimodal_forward * 100.0, 3),
            "language_model_visual_token_delta": round(
                averages["language_model_visual_token_delta_ms"] / multimodal_forward * 100.0,
                3,
            ),
            "complete_visual_path_delta": round(visual_total / multimodal_forward * 100.0, 3),
        },
        "pearson_r_with_image_tokens": {
            "multimodal_forward_ms": rounded_correlation(samples, "multimodal_forward_ms"),
            "vision_encoder_ms": rounded_correlation(samples, "vision_encoder_ms"),
            "visual_path_delta_ms": rounded_correlation(samples, "visual_path_delta_ms"),
            "language_model_visual_token_delta_ms": rounded_correlation(
                samples,
                "language_model_visual_token_delta_ms",
            ),
            "decode_visual_cache_delta_ms": rounded_correlation(
                samples,
                "decode_visual_cache_delta_ms",
            ),
        },
    }


def write_csv(path: Path, samples: list[ProfileSample]) -> None:
    rows = [asdict(sample) for sample in samples]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    for key in DISABLED_OPTIMIZATION_ENV_KEYS:
        if os.environ.get(key, "").strip().lower() in {"1", "true", "yes", "on"}:
            raise RuntimeError(f"Disable {key} before profiling the baseline.")

    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    dataset_path = Path(args.dataset_path).resolve()
    model_path = Path(args.model_path).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = load_rows(dataset_path, args.num_samples)
    language = "cn" if "_cn" in dataset_path.name.lower() else "en"

    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=dtype,
    ).to(device).eval()

    vision_timer = ForwardTimer(model.model.visual, lambda: synchronize(device))
    language_timer = ForwardTimer(model.model.language_model, lambda: synchronize(device))
    vision_timer.install()
    language_timer.install()
    try:
        for index, row in enumerate(rows[: args.warmup_samples]):
            profile_sample(
                model,
                processor,
                row=row,
                language=language,
                device=device,
                vision_timer=vision_timer,
                language_timer=language_timer,
                image_first=index % 2 == 0,
                decode_steps=args.decode_steps,
            )

        samples = [
            profile_sample(
                model,
                processor,
                row=row,
                language=language,
                device=device,
                vision_timer=vision_timer,
                language_timer=language_timer,
                image_first=index % 2 == 0,
                decode_steps=args.decode_steps,
            )
            for index, row in enumerate(rows)
        ]
    finally:
        vision_timer.restore()
        language_timer.restore()

    payload = {
        "profile_version": "qwen35_visual_token_cost_v1",
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "device": str(device),
        "dtype": str(dtype),
        "decode_steps": args.decode_steps,
        "method": (
            "Paired prefill-only measurements using identical prompts with and without the image. "
            "Device synchronization is applied at every measured module boundary."
        ),
        "summary": build_summary(samples),
        "samples": [asdict(sample) for sample in samples],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = output_path.with_suffix(".csv")
    write_csv(csv_path, samples)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"JSON: {output_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
