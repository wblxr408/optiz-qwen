"""Public DNDX self-test benchmark integrated into the repository layout."""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image

from .answer_parsing import extract_answer, parse_choice_answer
from .dndx_wrapper import GenerationConfig, VLMModel

DEFAULT_DATASET_PATH = "./resources/eval_dataset/raw/mmbench_public/mmbench_dev_en.tsv"
DEFAULT_MODEL_PATH = "./resources/model_weights/raw/Qwen3.5-2B"
DEFAULT_OUTPUT_PATH = "./benchmarks/output/result_public.json"
KIVI_ENV_KEYS = (
    "OPTIZ_QWEN_KIVI_KV_CACHE",
    "OPTIZ_QWEN_KIVI_K_BITS",
    "OPTIZ_QWEN_KIVI_V_BITS",
    "OPTIZ_QWEN_KIVI_GROUP_SIZE",
    "OPTIZ_QWEN_KIVI_RESIDUAL_LENGTH",
)


@dataclass
class Sample:
    sample_id: str
    language: str
    question: str
    hint: str
    choices: dict[str, str]
    answer: str
    image_b64: str
    category: str
    subcategory: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DNDX public self-test benchmark")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="Path to a public MMBench TSV file",
    )
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument(
        "--backend",
        choices=["auto", "dummy", "transformers"],
        default="auto",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--warmup-samples", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--enable-kivi-kv-cache",
        action="store_true",
        help="Enable the local Qwen3.5 KIVI KV-cache adapter for this run.",
    )
    parser.add_argument("--kivi-k-bits", type=int, default=2)
    parser.add_argument("--kivi-v-bits", type=int, default=2)
    parser.add_argument("--kivi-group-size", type=int, default=32)
    parser.add_argument("--kivi-residual-length", type=int, default=32)
    return parser.parse_args()


def load_mmbench_tsv(path: Path, limit: int | None = None) -> list[Sample]:
    language = "cn" if "_cn" in path.name.lower() else "en"
    samples: list[Sample] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            samples.append(
                Sample(
                    sample_id=str(row["index"]),
                    language=language,
                    question=(row.get("question") or "").strip(),
                    hint=(row.get("hint") or "").strip(),
                    choices={key: (row.get(key) or "").strip() for key in ["A", "B", "C", "D"]},
                    answer=(row.get("answer") or "").strip().upper(),
                    image_b64=row["image"],
                    category=(row.get("category") or "").strip(),
                    subcategory=(row.get("l2-category") or "").strip(),
                )
            )
            if limit is not None and len(samples) >= limit:
                break
    return samples


def decode_image(image_b64: str) -> Image.Image:
    raw = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(raw))
    return image.convert("RGB")


def build_prompt(sample: Sample) -> str:
    option_block = "\n".join(
        f"{key}. {value}" for key, value in sample.choices.items() if value.strip()
    )
    hint_block = f"Hint: {sample.hint}\n" if sample.hint else ""
    if sample.language == "cn":
        instruction = (
            "请完成这道单选题。"
            "请给出你认为正确的选项，并可附带一句简短理由。"
            "答案必须明确，且只能对应 A/B/C/D 中的一个选项。"
        )
    else:
        instruction = (
            "Solve this single-choice question."
            " Your response must make one final choice among A/B/C/D clearly."
            " You may include one short reason."
        )
    return (
        f"{instruction}\n"
        f"{hint_block}"
        f"Question: {sample.question}\n"
        f"{option_block}\n"
    )


def fixed_generation_config(max_new_tokens: int = 64) -> GenerationConfig:
    return GenerationConfig(max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0)


def compute_throughput(
    token_count: int,
    ttft_seconds: float,
    elapsed_seconds: float,
) -> float:
    if token_count <= 0 or elapsed_seconds <= 0:
        return 0.0
    decode_window = max(elapsed_seconds - max(ttft_seconds, 0.0), 1e-6)
    effective_tokens = max(token_count - 1, 1)
    return effective_tokens / decode_window


def settle_runtime(model: VLMModel) -> None:
    torch_mod = getattr(model, "_torch", None)
    if torch_mod is None:
        return
    try:
        if torch_mod.cuda.is_available():
            torch_mod.cuda.synchronize()
            torch_mod.cuda.empty_cache()
            torch_mod.cuda.synchronize()
    except Exception:
        pass
    time.sleep(0.01)


def validate_public_result(
    text: str,
    parsed_answer: str | None,
    token_count: int,
    max_new_tokens: int,
) -> list[str]:
    errors: list[str] = []
    normalized = (text or "").strip()
    if not normalized:
        errors.append("empty_output")
    if parsed_answer not in {"A", "B", "C", "D"}:
        errors.append("missing_choice_answer")
    if token_count <= 0:
        errors.append("zero_generated_tokens")
    if token_count > max_new_tokens + 8:
        errors.append("token_count_exceeds_budget")
    if len(normalized) > 1200:
        errors.append("output_too_long")
    return errors


@contextmanager
def kivi_cli_environment(args: argparse.Namespace):
    previous = {key: os.environ.get(key) for key in KIVI_ENV_KEYS}
    if args.enable_kivi_kv_cache:
        os.environ["OPTIZ_QWEN_KIVI_KV_CACHE"] = "1"
        os.environ["OPTIZ_QWEN_KIVI_K_BITS"] = str(args.kivi_k_bits)
        os.environ["OPTIZ_QWEN_KIVI_V_BITS"] = str(args.kivi_v_bits)
        os.environ["OPTIZ_QWEN_KIVI_GROUP_SIZE"] = str(args.kivi_group_size)
        os.environ["OPTIZ_QWEN_KIVI_RESIDUAL_LENGTH"] = str(args.kivi_residual_length)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_benchmark(args: argparse.Namespace) -> dict:
    benchmark_start = time.perf_counter()
    random.seed(args.seed)
    try:
        import numpy as np

        np.random.seed(args.seed)
    except Exception:
        pass

    dataset_path = Path(args.dataset_path).resolve()
    if "/datasets/test/" in str(dataset_path).replace("\\", "/"):
        raise ValueError("benchmark_public.py only supports public dev datasets.")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples = load_mmbench_tsv(dataset_path, limit=args.num_samples)
    if not samples:
        raise ValueError(f"No samples loaded from {dataset_path}")

    kivi_env_enabled = False
    with kivi_cli_environment(args):
        kivi_env_enabled = os.environ.get("OPTIZ_QWEN_KIVI_KV_CACHE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        model = VLMModel(args.model_path, backend=args.backend, device=args.device)

        for sample in samples[: min(args.warmup_samples, len(samples))]:
            settle_runtime(model)
            model.generate_with_metrics(
                image=decode_image(sample.image_b64),
                prompt=build_prompt(sample),
                choices=sample.choices,
                generation_config=fixed_generation_config(args.max_new_tokens),
                sample_id=sample.sample_id,
            )
            settle_runtime(model)

        records = []
        ttfts_ms = []
        throughputs = []
        correct = 0
        validation_errors = 0

        for sample in samples:
            settle_runtime(model)
            config = fixed_generation_config(args.max_new_tokens)
            result = model.generate_with_metrics(
                image=decode_image(sample.image_b64),
                prompt=build_prompt(sample),
                choices=sample.choices,
                generation_config=config,
                sample_id=sample.sample_id,
            )
            parsed_answer, answer_source = parse_choice_answer(result.text, sample.choices)
            errors = validate_public_result(
                result.text,
                parsed_answer,
                result.token_count,
                config.max_new_tokens,
            )
            validation_errors += int(bool(errors))
            is_correct = parsed_answer == sample.answer
            correct += int(is_correct)

            ttft_ms = result.ttft_seconds * 1000.0
            throughput = compute_throughput(
                result.token_count,
                result.ttft_seconds,
                result.elapsed_seconds,
            )
            if math.isfinite(ttft_ms) and ttft_ms > 0:
                ttfts_ms.append(ttft_ms)
            if math.isfinite(throughput) and throughput > 0:
                throughputs.append(throughput)

            records.append(
                {
                    "question_id": sample.sample_id,
                    "text": result.text,
                    "parsed_answer": parsed_answer,
                    "answer_source": answer_source,
                    "correct": is_correct,
                    "ttft_ms": round(ttft_ms, 3),
                    "throughput_tokens_per_sec": round(throughput, 3),
                    "token_count": result.token_count,
                    "validation_errors": errors,
                    "meta": result.meta,
                }
            )
            settle_runtime(model)

    elapsed = time.perf_counter() - benchmark_start
    payload = {
        "benchmark_version": "dndx_public_self_test",
        "timestamp": datetime.now().isoformat(),
        "dataset_path": str(dataset_path),
        "sample_count": len(samples),
        "seed": args.seed,
        "backend": model.backend_name,
        "optimization": {
            "kivi_kv_cache_requested_by_cli": bool(args.enable_kivi_kv_cache),
            "kivi_kv_cache_enabled_by_env": kivi_env_enabled,
        },
        "performance": {
            "avg_ttft_ms": round(sum(ttfts_ms) / len(ttfts_ms), 3) if ttfts_ms else None,
            "avg_throughput_tokens_per_sec": (
                round(sum(throughputs) / len(throughputs), 3) if throughputs else 0.0
            ),
        },
        "timing": {
            "benchmark_elapsed_seconds": round(elapsed, 3),
            "benchmark_elapsed_minutes": round(elapsed / 60.0, 3),
            "avg_seconds_per_sample": round(elapsed / len(samples), 3),
        },
        "accuracy": {
            "score": round(correct / len(samples), 6),
            "correct": correct,
            "total": len(samples),
        },
        "public_validation": {
            "passed": validation_errors == 0,
            "failed_samples": validation_errors,
        },
        "answers": records,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    payload = run_benchmark(parse_args())
    print(
        json.dumps(
            {
                "backend": payload["backend"],
                "sample_count": payload["sample_count"],
                "avg_ttft_ms": payload["performance"]["avg_ttft_ms"],
                "avg_throughput_tokens_per_sec": payload["performance"][
                    "avg_throughput_tokens_per_sec"
                ],
                "accuracy": payload["accuracy"]["score"],
                "public_validation_passed": payload["public_validation"]["passed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
