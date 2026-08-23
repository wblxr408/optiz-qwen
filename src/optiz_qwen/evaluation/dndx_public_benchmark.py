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
from .diagnostics import (
    install_crash_diagnostics,
    log_runtime_environment,
    stage,
)
from .dndx_wrapper import GenerationConfig, VLMModel
from ..scheduling import prefill_last_logit_only_enabled
from ..ppu import ensure_ppu_sdk_env

DEFAULT_DATASET_PATH = "./resources/eval_dataset/raw/mmbench_public/mmbench_dev_en.tsv"
DEFAULT_MODEL_PATH = "./resources/model_weights/raw/Qwen3.5-2B"
DEFAULT_OUTPUT_PATH = "./benchmarks/output/result_public.json"
OFFICIAL_MAX_NEW_TOKENS = 256
KV_CHAIN_ENV_KEYS = (
    "OPTIZ_QWEN_KV_CHAIN_ENABLED",
    "OPTIZ_QWEN_KV_CHAIN",
    "OPTIZ_QWEN_KV_CHAIN_K_BITS",
    "OPTIZ_QWEN_KV_CHAIN_V_BITS",
    "OPTIZ_QWEN_KV_CHAIN_GROUP_SIZE",
    "OPTIZ_QWEN_KV_CHAIN_RESIDUAL_LENGTH",
    "OPTIZ_QWEN_KV_CHAIN_ACTIVATION_THRESHOLD",
    "OPTIZ_QWEN_KV_CHAIN_DECODE_WARMUP_TOKENS",
    "OPTIZ_QWEN_KV_CHAIN_ATTENTION_BACKEND",
)
RUNNER_ENV_KEYS = ("OPTIZ_QWEN_GENERATION_RUNNER",)
VISUAL_ENV_KEYS = ("OPTIZ_QWEN_VISUAL_PIXEL_BUDGET",)
TOME_ENV_KEYS = (
    "OPTIZ_QWEN_TOME_ENABLED",
    "OPTIZ_QWEN_TOME_LAYER",
    "OPTIZ_QWEN_TOME_R",
    "OPTIZ_QWEN_TOME_PROPORTIONAL_ATTENTION",
)
#: The validated PPU hybrid: sdpa prefill + one CUDA graph captured under
#: flash_attention_2 replayed over a StaticCache.  Kept behind a switch so the
#: A/B against the eager baseline stays a one-flag change.
HYBRID_ENV_KEYS = (
    "OPTIZ_QWEN_CUDA_GRAPH_DECODE",
    "OPTIZ_QWEN_CUDA_GRAPH_MAX_CACHE_LEN",
    "OPTIZ_QWEN_CUDA_GRAPH_WARMUP_STEPS",
    "OPTIZ_QWEN_ATTN_PREFILL",
    "OPTIZ_QWEN_ATTN_DECODE",
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
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16"],
        default=None,
        help="Override model floating-point dtype; omitted preserves legacy behavior.",
    )
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
        "--enable-kv-chain",
        action="store_true",
        help="Enable the retained deferred packed-KV experimental chain.",
    )
    parser.add_argument(
        "--kv-chain",
        choices=["qserve_deferred_split_fused_kv"],
        default="qserve_deferred_split_fused_kv",
    )
    parser.add_argument("--kv-chain-k-bits", type=int, default=4)
    parser.add_argument("--kv-chain-v-bits", type=int, default=4)
    parser.add_argument("--kv-chain-group-size", type=int, default=32)
    parser.add_argument("--kv-chain-residual-length", type=int, default=32)
    parser.add_argument(
        "--kv-chain-activation-threshold",
        type=int,
        default=1024,
        help="Keep native dense decode below this real KV-token threshold.",
    )
    parser.add_argument(
        "--kv-chain-decode-warmup-tokens",
        type=int,
        default=4,
        help="Keep initial answer-forming decode tokens on the native cache.",
    )
    parser.add_argument(
        "--kv-chain-attention-backend",
        choices=["triton_int4_split_decode", "triton_int4_decode", "triton_int8_decode"],
        default="triton_int4_split_decode",
    )
    parser.add_argument(
        "--enable-hybrid-cudagraph",
        action="store_true",
        help=(
            "Enable the validated PPU hybrid decode path: sdpa prefill plus one "
            "CUDA graph captured under flash_attention_2 replayed over a StaticCache. "
            "Requires --generation-runner greedy."
        ),
    )
    parser.add_argument(
        "--hybrid-max-cache-len",
        type=int,
        default=2048,
        help="StaticCache length the decode graph is captured against.",
    )
    parser.add_argument(
        "--hybrid-warmup-steps",
        type=int,
        default=3,
        help="Side-stream warmup decode steps run before graph capture.",
    )
    parser.add_argument(
        "--hybrid-prefill-backend",
        choices=["sdpa", "flash_attention_2", "eager"],
        default="sdpa",
        help="Attention backend restored after capture, used by prefill.",
    )
    parser.add_argument(
        "--hybrid-decode-backend",
        choices=["sdpa", "flash_attention_2"],
        default="flash_attention_2",
        help="Attention backend live at capture time, frozen into the graph.",
    )
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


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    return str(value)


def quantization_metadata(quantization_config) -> dict:
    config = _json_safe(quantization_config)
    if not isinstance(config, dict):
        return {
            "enabled": False,
            "quant_method": None,
            "format": None,
            "status": None,
            "weights": None,
            "config": config,
        }

    config_groups = config.get("config_groups") or {}
    first_group = next(iter(config_groups.values()), {})
    weights = first_group.get("weights") if isinstance(first_group, dict) else None
    return {
        "enabled": True,
        "quant_method": config.get("quant_method"),
        "format": config.get("format"),
        "status": config.get("quantization_status"),
        "weights": weights,
        "config": config,
    }


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
def kv_chain_cli_environment(args: argparse.Namespace):
    previous = {key: os.environ.get(key) for key in KV_CHAIN_ENV_KEYS}
    if getattr(args, "enable_kv_chain", False):
        requested_chain = str(
            getattr(args, "kv_chain", "qserve_deferred_split_fused_kv")
        ).strip().lower()
        if requested_chain != "qserve_deferred_split_fused_kv":
            raise ValueError(
                "Only qserve_deferred_split_fused_kv is retained for KV experiments."
            )
        os.environ["OPTIZ_QWEN_KV_CHAIN_ENABLED"] = "1"
        os.environ["OPTIZ_QWEN_KV_CHAIN"] = "qserve_deferred_split_fused_kv"
        os.environ["OPTIZ_QWEN_KV_CHAIN_K_BITS"] = str(getattr(args, "kv_chain_k_bits", 4))
        os.environ["OPTIZ_QWEN_KV_CHAIN_V_BITS"] = str(getattr(args, "kv_chain_v_bits", 4))
        os.environ["OPTIZ_QWEN_KV_CHAIN_GROUP_SIZE"] = str(getattr(args, "kv_chain_group_size", 32))
        os.environ["OPTIZ_QWEN_KV_CHAIN_RESIDUAL_LENGTH"] = str(getattr(args, "kv_chain_residual_length", 32))
        os.environ["OPTIZ_QWEN_KV_CHAIN_ACTIVATION_THRESHOLD"] = str(
            getattr(args, "kv_chain_activation_threshold", 1024)
        )
        os.environ["OPTIZ_QWEN_KV_CHAIN_DECODE_WARMUP_TOKENS"] = str(
            getattr(args, "kv_chain_decode_warmup_tokens", 4)
        )
        os.environ["OPTIZ_QWEN_KV_CHAIN_ATTENTION_BACKEND"] = str(
            getattr(args, "kv_chain_attention_backend", "triton_int4_split_decode")
        )
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


@contextmanager
def hybrid_cli_environment(args: argparse.Namespace):
    previous = {key: os.environ.get(key) for key in HYBRID_ENV_KEYS}
    if getattr(args, "enable_hybrid_cudagraph", False):
        os.environ["OPTIZ_QWEN_CUDA_GRAPH_DECODE"] = "1"
        os.environ["OPTIZ_QWEN_CUDA_GRAPH_MAX_CACHE_LEN"] = str(
            getattr(args, "hybrid_max_cache_len", 2048)
        )
        os.environ["OPTIZ_QWEN_CUDA_GRAPH_WARMUP_STEPS"] = str(
            getattr(args, "hybrid_warmup_steps", 3)
        )
        os.environ["OPTIZ_QWEN_ATTN_PREFILL"] = str(
            getattr(args, "hybrid_prefill_backend", "sdpa")
        )
        os.environ["OPTIZ_QWEN_ATTN_DECODE"] = str(
            getattr(args, "hybrid_decode_backend", "flash_attention_2")
        )
    else:
        for key in HYBRID_ENV_KEYS:
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
    if getattr(args, "enable_kv_chain", False) and getattr(args, "generation_runner", "generate") != "greedy":
        raise ValueError(
            "qserve_deferred_split_fused_kv requires --generation-runner greedy; "
            "the native generate runner cannot install the fused decode path after prefill."
        )
    if getattr(args, "enable_hybrid_cudagraph", False):
        if getattr(args, "generation_runner", "generate") != "greedy":
            raise ValueError(
                "--enable-hybrid-cudagraph requires --generation-runner greedy; "
                "the captured graph owns the decode loop."
            )
        if getattr(args, "enable_kv_chain", False):
            raise ValueError(
                "--enable-hybrid-cudagraph is mutually exclusive with --enable-kv-chain; "
                "the graph is bound to the StaticCache it was captured against."
            )
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
    # Point the PPU RTC toolchain at its SDK before any device work.  On a shell
    # that never sourced envsetup.sh, PPU_SDK/PPU_HOME are unset and the first
    # kernel compilation aborts (rc=134 / SIGABRT) in the vision tower with no
    # JSON and no traceback -- the reported smoke-test symptom.  No-op when the
    # env is already set or no SDK is installed (e.g. non-PPU dev hosts).
    sdk_bootstrap = ensure_ppu_sdk_env()
    # Arm crash diagnostics before any device work: a native abort in the
    # runtime (rc=134 / SIGABRT) unwinds nothing, so without this the process
    # dies with no JSON and no traceback.  faulthandler dumps every thread's
    # Python stack at abort, and the stage markers below survive it.
    fault_log = install_crash_diagnostics(output_path)
    if fault_log is not None:
        stage(f"benchmark start; fault log -> {fault_log}")
        if sdk_bootstrap.applied:
            stage(
                f"ppu sdk env bootstrapped from {sdk_bootstrap.sdk_root} "
                f"({', '.join(sorted(sdk_bootstrap.variables_set))})"
            )
        else:
            stage(f"ppu sdk env: {sdk_bootstrap.reason}")
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

    kv_chain_env_enabled = False
    with (
        kv_chain_cli_environment(args),
        runner_cli_environment(args),
        visual_cli_environment(args),
        tome_cli_environment(args),
        hybrid_cli_environment(args),
    ):
        kv_chain_env_enabled = os.environ.get("OPTIZ_QWEN_KV_CHAIN_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}
        stage(f"loading model backend={args.backend} device={args.device}")
        model = VLMModel(
            args.model_path,
            backend=args.backend,
            device=args.device,
            dtype=getattr(args, "dtype", None),
        )
        log_runtime_environment(model)

        for warmup_index, sample in enumerate(samples[: min(args.warmup_samples, len(samples))]):
            settle_runtime(model)
            stage(f"warmup {warmup_index} sample_id={sample.sample_id} (vision+prefill)")
            model.generate_with_metrics(
                image=decode_image(sample.image_b64),
                prompt=build_prompt(sample),
                choices=sample.choices,
                generation_config=fixed_generation_config(args.max_new_tokens),
                sample_id=sample.sample_id,
            )
            settle_runtime(model)
        stage("warmup complete")

        records = []
        ttfts_ms = []
        throughputs = []
        correct = 0
        validation_errors = 0

        for scored_index, sample in enumerate(samples):
            settle_runtime(model)
            stage(f"scored {scored_index}/{len(samples)} sample_id={sample.sample_id}")
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
        "model_path": str(Path(args.model_path).resolve()),
        "sample_count": len(samples),
        "seed": args.seed,
        "sample_selection": {
            "strategy": getattr(args, "sample_strategy", "sequential"),
            "categories": sorted(category_filter) if category_filter is not None else None,
            "source_sample_count": len(all_samples),
        },
        "backend": model.backend_name,
        "dtype": model.dtype_name,
        "quantization": quantization_metadata(model.quantization_config),
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "do_sample": False,
            "runner": getattr(args, "generation_runner", "generate"),
        },
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
            "kv_chain_requested_by_cli": bool(getattr(args, "enable_kv_chain", False)),
            "kv_chain_enabled_by_env": kv_chain_env_enabled,
            "kv_chain_name": getattr(args, "kv_chain", None) if kv_chain_env_enabled else None,
            "hybrid_cudagraph_enabled": bool(getattr(args, "enable_hybrid_cudagraph", False)),
            "hybrid_max_cache_len": (
                getattr(args, "hybrid_max_cache_len", None)
                if getattr(args, "enable_hybrid_cudagraph", False)
                else None
            ),
            "hybrid_prefill_backend": (
                getattr(args, "hybrid_prefill_backend", None)
                if getattr(args, "enable_hybrid_cudagraph", False)
                else None
            ),
            "hybrid_decode_backend": (
                getattr(args, "hybrid_decode_backend", None)
                if getattr(args, "enable_hybrid_cudagraph", False)
                else None
            ),
            # On by default; recorded here so an arm's TTFT number can never be
            # read without knowing whether the last-logit-only prefill was live.
            "prefill_last_logit_only": prefill_last_logit_only_enabled(),
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
                "dtype": payload["dtype"],
                "quantization": payload["quantization"]["quant_method"],
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
