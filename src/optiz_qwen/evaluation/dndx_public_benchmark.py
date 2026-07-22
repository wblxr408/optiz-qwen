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
OFFICIAL_MAX_NEW_TOKENS = 256
KIVI_ENV_KEYS = (
    "OPTIZ_QWEN_KIVI_KV_CACHE",
    "OPTIZ_QWEN_KIVI_K_BITS",
    "OPTIZ_QWEN_KIVI_V_BITS",
    "OPTIZ_QWEN_KIVI_GROUP_SIZE",
    "OPTIZ_QWEN_KIVI_RESIDUAL_LENGTH",
)
KV_CHAIN_ENV_KEYS = (
    "OPTIZ_QWEN_KV_CHAIN_ENABLED",
    "OPTIZ_QWEN_KV_CHAIN",
    "OPTIZ_QWEN_KV_CHAIN_K_BITS",
    "OPTIZ_QWEN_KV_CHAIN_V_BITS",
    "OPTIZ_QWEN_KV_CHAIN_GROUP_SIZE",
    "OPTIZ_QWEN_KV_CHAIN_RESIDUAL_LENGTH",
)
RUNNER_ENV_KEYS = ("OPTIZ_QWEN_GENERATION_RUNNER",)
VISUAL_ENV_KEYS = ("OPTIZ_QWEN_VISUAL_PIXEL_BUDGET",)
TOME_ENV_KEYS = (
    "OPTIZ_QWEN_TOME_ENABLED",
    "OPTIZ_QWEN_TOME_LAYER",
    "OPTIZ_QWEN_TOME_R",
    "OPTIZ_QWEN_TOME_PROPORTIONAL_ATTENTION",
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
    parser.add_argument(
        "--sample-strategy",
        choices=["sequential", "stratified"],
        default="sequential",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated exact category names to include before sampling.",
    )
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument(
        "--backend",
        choices=["auto", "dummy", "transformers"],
        default="auto",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--warmup-samples", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=OFFICIAL_MAX_NEW_TOKENS)
    parser.add_argument(
        "--generation-runner",
        choices=["generate", "greedy"],
        default="generate",
        help="Use greedy for a runner-identical baseline versus KV-chain comparison.",
    )
    parser.add_argument("--visual-pixel-budget", type=int, default=None)
    parser.add_argument("--enable-tome", action="store_true")
    parser.add_argument("--tome-layer", type=int, default=12)
    parser.add_argument("--tome-r", type=int, default=1)
    parser.add_argument("--tome-proportional-attention", action="store_true")
    parser.add_argument(
        "--enable-kivi-kv-cache",
        action="store_true",
        help="Enable the local Qwen3.5 KIVI KV-cache adapter for this run.",
    )
    parser.add_argument("--kivi-k-bits", type=int, default=2)
    parser.add_argument("--kivi-v-bits", type=int, default=2)
    parser.add_argument("--kivi-group-size", type=int, default=32)
    parser.add_argument("--kivi-residual-length", type=int, default=32)
    parser.add_argument("--enable-kv-chain", action="store_true")
    parser.add_argument("--kv-chain", type=str, default="qserve_kv")
    parser.add_argument("--kv-chain-k-bits", type=int, default=4)
    parser.add_argument("--kv-chain-v-bits", type=int, default=4)
    parser.add_argument("--kv-chain-group-size", type=int, default=32)
    parser.add_argument("--kv-chain-residual-length", type=int, default=32)
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


def select_samples(
    samples: list[Sample],
    *,
    limit: int | None,
    strategy: str,
    seed: int,
    categories: set[str] | None = None,
) -> list[Sample]:
    selected = [sample for sample in samples if categories is None or sample.category in categories]
    if strategy == "sequential":
        return selected if limit is None else selected[:limit]
    if strategy != "stratified":
        raise ValueError(f"unsupported sample strategy: {strategy}")

    buckets: dict[str, list[Sample]] = {}
    for sample in selected:
        buckets.setdefault(sample.category, []).append(sample)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    result: list[Sample] = []
    category_names = sorted(buckets)
    while category_names and (limit is None or len(result) < limit):
        next_names = []
        for category in category_names:
            bucket = buckets[category]
            if bucket:
                result.append(bucket.pop())
                if limit is not None and len(result) >= limit:
                    break
            if bucket:
                next_names.append(category)
        category_names = next_names
    return result


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


def fixed_generation_config(
    max_new_tokens: int = OFFICIAL_MAX_NEW_TOKENS,
) -> GenerationConfig:
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
    else:
        for key in KIVI_ENV_KEYS:
            os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def kv_chain_cli_environment(args: argparse.Namespace):
    previous = {key: os.environ.get(key) for key in KV_CHAIN_ENV_KEYS}
    if getattr(args, "enable_kv_chain", False):
        os.environ["OPTIZ_QWEN_KV_CHAIN_ENABLED"] = "1"
        os.environ["OPTIZ_QWEN_KV_CHAIN"] = str(getattr(args, "kv_chain", "qserve_kv"))
        os.environ["OPTIZ_QWEN_KV_CHAIN_K_BITS"] = str(getattr(args, "kv_chain_k_bits", 2))
        os.environ["OPTIZ_QWEN_KV_CHAIN_V_BITS"] = str(getattr(args, "kv_chain_v_bits", 4))
        os.environ["OPTIZ_QWEN_KV_CHAIN_GROUP_SIZE"] = str(getattr(args, "kv_chain_group_size", 32))
        os.environ["OPTIZ_QWEN_KV_CHAIN_RESIDUAL_LENGTH"] = str(getattr(args, "kv_chain_residual_length", 32))
    else:
        for key in KV_CHAIN_ENV_KEYS:
            os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def runner_cli_environment(args: argparse.Namespace):
    previous = {key: os.environ.get(key) for key in RUNNER_ENV_KEYS}
    os.environ["OPTIZ_QWEN_GENERATION_RUNNER"] = str(getattr(args, "generation_runner", "generate"))
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def visual_cli_environment(args: argparse.Namespace):
    previous = {key: os.environ.get(key) for key in VISUAL_ENV_KEYS}
    budget = getattr(args, "visual_pixel_budget", None)
    if budget is not None:
        os.environ["OPTIZ_QWEN_VISUAL_PIXEL_BUDGET"] = str(budget)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def tome_cli_environment(args: argparse.Namespace):
    previous = {key: os.environ.get(key) for key in TOME_ENV_KEYS}
    if getattr(args, "enable_tome", False):
        os.environ["OPTIZ_QWEN_TOME_ENABLED"] = "1"
        os.environ["OPTIZ_QWEN_TOME_LAYER"] = str(getattr(args, "tome_layer", 12))
        os.environ["OPTIZ_QWEN_TOME_R"] = str(getattr(args, "tome_r", 1))
        os.environ["OPTIZ_QWEN_TOME_PROPORTIONAL_ATTENTION"] = (
            "1" if getattr(args, "tome_proportional_attention", False) else "0"
        )
    else:
        for key in TOME_ENV_KEYS:
            os.environ.pop(key, None)
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
    all_samples = load_mmbench_tsv(dataset_path)
    category_filter = None
    if getattr(args, "categories", None):
        category_filter = {
            value.strip() for value in str(args.categories).split(",") if value.strip()
        }
    samples = select_samples(
        all_samples,
        limit=args.num_samples,
        strategy=getattr(args, "sample_strategy", "sequential"),
        seed=args.seed,
        categories=category_filter,
    )
    if not samples:
        raise ValueError(f"No samples loaded from {dataset_path}")

    kivi_env_enabled = False
    kv_chain_env_enabled = False
    with (
        kivi_cli_environment(args),
        kv_chain_cli_environment(args),
        runner_cli_environment(args),
        visual_cli_environment(args),
        tome_cli_environment(args),
    ):
        kivi_env_enabled = os.environ.get("OPTIZ_QWEN_KIVI_KV_CACHE", "").strip().lower() in {"1", "true", "yes", "on"}
        kv_chain_env_enabled = os.environ.get("OPTIZ_QWEN_KV_CHAIN_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}
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
        "benchmark_version": "dndx_public_self_test_v1.1",
        "timestamp": datetime.now().isoformat(),
        "dataset_path": str(dataset_path),
        "sample_count": len(samples),
        "seed": args.seed,
        "sample_selection": {
            "strategy": getattr(args, "sample_strategy", "sequential"),
            "categories": sorted(category_filter) if category_filter is not None else None,
            "source_sample_count": len(all_samples),
        },
        "backend": model.backend_name,
        "generation": {"max_new_tokens": args.max_new_tokens},
        "optimization": {
            "generation_runner": getattr(args, "generation_runner", "generate"),
            "visual_pixel_budget": getattr(args, "visual_pixel_budget", None),
            "visual_accuracy_risk": (
                "OCR and fine-grained localization require accuracy validation"
                if getattr(args, "visual_pixel_budget", None) is not None
                else None
            ),
            "tome_enabled": bool(getattr(args, "enable_tome", False)),
            "tome_layer": getattr(args, "tome_layer", None) if getattr(args, "enable_tome", False) else None,
            "tome_r": getattr(args, "tome_r", None) if getattr(args, "enable_tome", False) else None,
            "tome_proportional_attention": (
                bool(getattr(args, "tome_proportional_attention", False))
                if getattr(args, "enable_tome", False)
                else False
            ),
            "kivi_kv_cache_requested_by_cli": bool(args.enable_kivi_kv_cache),
            "kivi_kv_cache_enabled_by_env": kivi_env_enabled,
            "kv_chain_requested_by_cli": bool(getattr(args, "enable_kv_chain", False)),
            "kv_chain_enabled_by_env": kv_chain_env_enabled,
            "kv_chain_name": getattr(args, "kv_chain", None) if kv_chain_env_enabled else None,
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
