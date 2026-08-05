"""Paired single-model benchmark for opt-in Qwen3.5 KV experiments."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
from statistics import median
from typing import Iterator

from optiz_qwen.evaluation.dndx_public_benchmark import (
    build_prompt,
    decode_image,
    fixed_generation_config,
    load_mmbench_tsv,
    settle_runtime,
)
from optiz_qwen.evaluation.answer_parsing import parse_choice_answer
from optiz_qwen.evaluation.dndx_wrapper import VLMModel


KV_ENV_KEYS = (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired Qwen3.5 KV-cache benchmark.")
    parser.add_argument("--dataset-path", default="resources/eval_dataset/raw/mmbench_public/mmbench_dev_en.tsv")
    parser.add_argument("--model-path", default="resources/model_weights/raw/Qwen3.5-2B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--kv-chain", default="qserve_deferred_split_fused_kv")
    parser.add_argument("--kv-chain-activation-threshold", type=int, default=1024)
    parser.add_argument("--kv-chain-decode-warmup-tokens", type=int, default=4)
    parser.add_argument("--kv-chain-k-bits", type=int, choices=[4, 8], default=4)
    parser.add_argument("--kv-chain-v-bits", type=int, choices=[4, 8], default=4)
    parser.add_argument(
        "--kv-chain-attention-backend",
        choices=["triton_int4_split_decode", "triton_int4_decode", "triton_int8_decode"],
        default="triton_int4_split_decode",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


@contextmanager
def kv_mode(
    chain_name: str | None,
    activation_threshold: int = 1024,
    decode_warmup_tokens: int = 4,
    k_bits: int = 4,
    v_bits: int = 4,
    attention_backend: str = "triton_int4_split_decode",
) -> Iterator[None]:
    all_keys = (*KV_ENV_KEYS, "OPTIZ_QWEN_GENERATION_RUNNER")
    previous = {key: os.environ.get(key) for key in all_keys}
    try:
        os.environ["OPTIZ_QWEN_GENERATION_RUNNER"] = "greedy"
        if chain_name is None:
            for key in KV_ENV_KEYS:
                os.environ.pop(key, None)
        else:
            os.environ["OPTIZ_QWEN_KV_CHAIN_ENABLED"] = "1"
            os.environ["OPTIZ_QWEN_KV_CHAIN"] = chain_name
            os.environ["OPTIZ_QWEN_KV_CHAIN_K_BITS"] = str(k_bits)
            os.environ["OPTIZ_QWEN_KV_CHAIN_V_BITS"] = str(v_bits)
            os.environ["OPTIZ_QWEN_KV_CHAIN_GROUP_SIZE"] = "32"
            os.environ["OPTIZ_QWEN_KV_CHAIN_RESIDUAL_LENGTH"] = "32"
            os.environ["OPTIZ_QWEN_KV_CHAIN_ACTIVATION_THRESHOLD"] = str(activation_threshold)
            os.environ["OPTIZ_QWEN_KV_CHAIN_DECODE_WARMUP_TOKENS"] = str(decode_warmup_tokens)
            os.environ["OPTIZ_QWEN_KV_CHAIN_ATTENTION_BACKEND"] = attention_backend
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def restore_native_full_attention(model: VLMModel) -> None:
    """Remove an instance-level experimental attention patch before baseline."""

    language_model = model._model.model.language_model
    for layer_idx, decoder_layer in enumerate(language_model.layers):
        if language_model.config.layer_types[layer_idx] != "full_attention":
            continue
        attention = decoder_layer.self_attn
        original = getattr(attention, "_optiz_original_forward", None)
        if original is not None:
            attention.forward = original
            delattr(attention, "_optiz_original_forward")


def run_once(
    model: VLMModel,
    sample,
    chain_name: str | None,
    max_new_tokens: int,
    activation_threshold: int,
    decode_warmup_tokens: int,
    k_bits: int,
    v_bits: int,
    attention_backend: str,
) -> dict:
    restore_native_full_attention(model)
    settle_runtime(model)
    with kv_mode(
        chain_name,
        activation_threshold,
        decode_warmup_tokens,
        k_bits,
        v_bits,
        attention_backend,
    ):
        result = model.generate_with_metrics(
            image=decode_image(sample.image_b64),
            prompt=build_prompt(sample),
            choices=sample.choices,
            generation_config=fixed_generation_config(max_new_tokens),
            sample_id=sample.sample_id,
        )
    settle_runtime(model)
    runtime = result.meta.get("kv_runtime") or {}
    parsed_answer, _ = parse_choice_answer(result.text, sample.choices)
    return {
        "chain": chain_name or "baseline",
        "ttft_ms": round(result.ttft_seconds * 1000.0, 3),
        "throughput_tokens_per_sec": round(
            max(result.token_count - 1, 1) / max(result.elapsed_seconds - result.ttft_seconds, 1e-6), 3
        ),
        "token_count": result.token_count,
        "answer_source": result.meta.get("answer_source"),
        "parsed_answer": parsed_answer,
        "correct": parsed_answer == sample.answer,
        "kernel_calls": int(runtime.get("kernel_calls", 0)),
        "fallback_calls": int(runtime.get("fallback_calls", 0)),
        "active_backend": runtime.get("active_backend"),
        "activation_threshold": (
            result.meta.get("kv_chain") or {}
        ).get("activation_threshold"),
    }


def summarize(records: list[dict]) -> dict:
    return {
        "runs": len(records),
        "all_correct": all(record["correct"] for record in records),
        "median_ttft_ms": round(median(record["ttft_ms"] for record in records), 3),
        "median_throughput_tokens_per_sec": round(
            median(record["throughput_tokens_per_sec"] for record in records), 3
        ),
        "kernel_calls": [record["kernel_calls"] for record in records],
        "fallback_calls": [record["fallback_calls"] for record in records],
    }


def answer_parity(records: list[dict], candidate_name: str) -> bool:
    answers = {(record["repeat"], record["sample_id"], record["chain"]): record["parsed_answer"] for record in records}
    pairs = [
        (answers[(repeat, sample_id, "baseline")], answers[(repeat, sample_id, candidate_name)])
        for repeat, sample_id, chain in answers
        if chain == "baseline"
    ]
    return all(baseline_answer == candidate_answer for baseline_answer, candidate_answer in pairs)


def main() -> None:
    args = parse_args()
    if args.warmups < 1 or args.repeats < 1:
        raise ValueError("--warmups and --repeats must both be positive.")
    all_samples = load_mmbench_tsv(Path(args.dataset_path))
    if not 0 <= args.sample_index < len(all_samples):
        raise ValueError("--sample-index is outside the public dataset.")
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive.")
    samples = all_samples[args.sample_index : args.sample_index + args.num_samples]
    model = VLMModel(args.model_path, backend="transformers", device=args.device)

    for chain_name in (None, args.kv_chain):
        for sample in samples:
            for _ in range(args.warmups):
                run_once(
                    model,
                    sample,
                    chain_name,
                    args.max_new_tokens,
                    args.kv_chain_activation_threshold,
                    args.kv_chain_decode_warmup_tokens,
                    args.kv_chain_k_bits,
                    args.kv_chain_v_bits,
                    args.kv_chain_attention_backend,
                )

    raw_records: list[dict] = []
    for repeat in range(args.repeats):
        order = (args.kv_chain, None) if repeat % 2 else (None, args.kv_chain)
        for sample in samples:
            for chain_name in order:
                record = run_once(
                    model,
                    sample,
                    chain_name,
                    args.max_new_tokens,
                    args.kv_chain_activation_threshold,
                    args.kv_chain_decode_warmup_tokens,
                    args.kv_chain_k_bits,
                    args.kv_chain_v_bits,
                    args.kv_chain_attention_backend,
                )
                record["repeat"] = repeat + 1
                record["sample_id"] = sample.sample_id
                raw_records.append(record)

    baseline = [record for record in raw_records if record["chain"] == "baseline"]
    candidate = [record for record in raw_records if record["chain"] == args.kv_chain]
    baseline_summary = summarize(baseline)
    candidate_summary = summarize(candidate)
    result = {
        "protocol": {
            "same_model_instance": True,
            "runner": "greedy",
            "sample_ids": [sample.sample_id for sample in samples],
            "max_new_tokens": args.max_new_tokens,
            "warmups_per_sample_per_mode": args.warmups,
            "alternating_order": True,
            "kv_chain_activation_threshold": args.kv_chain_activation_threshold,
            "kv_chain_decode_warmup_tokens": args.kv_chain_decode_warmup_tokens,
            "kv_chain_k_bits": args.kv_chain_k_bits,
            "kv_chain_v_bits": args.kv_chain_v_bits,
            "kv_chain_attention_backend": args.kv_chain_attention_backend,
        },
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "candidate_answers_match_baseline": answer_parity(raw_records, args.kv_chain),
        "median_change": {
            "ttft_percent": round(
                (candidate_summary["median_ttft_ms"] / baseline_summary["median_ttft_ms"] - 1.0) * 100.0, 2
            ),
            "throughput_percent": round(
                (candidate_summary["median_throughput_tokens_per_sec"]
                 / baseline_summary["median_throughput_tokens_per_sec"] - 1.0) * 100.0,
                2,
            ),
        },
        "records": raw_records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
