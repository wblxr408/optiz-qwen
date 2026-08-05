"""Aggregate fair paired KV artifacts and enforce the competition speed gate."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate paired Qwen3.5 KV benchmark artifacts.")
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resamples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260806)
    return parser.parse_args()


def _paired_records(payload: dict, source: Path) -> list[dict]:
    candidate_name = "qserve_deferred_split_fused_kv"
    grouped: dict[tuple[int, str], dict[str, dict]] = {}
    for record in payload["records"]:
        key = (int(record["repeat"]), str(record["sample_id"]))
        grouped.setdefault(key, {})[record["chain"]] = record

    pairs = []
    for (repeat, sample_id), pair in sorted(grouped.items()):
        baseline = pair.get("baseline")
        candidate = pair.get(candidate_name)
        if baseline is None or candidate is None:
            raise ValueError(f"{source} has an incomplete pair for repeat={repeat}, sample={sample_id}.")
        if baseline["parsed_answer"] != candidate["parsed_answer"]:
            raise ValueError(f"{source} changes the parsed answer for repeat={repeat}, sample={sample_id}.")
        if baseline["token_count"] != candidate["token_count"]:
            raise ValueError(f"{source} changes generated token count for repeat={repeat}, sample={sample_id}.")
        if int(candidate["kernel_calls"]) <= 0 or int(candidate["fallback_calls"]) != 0:
            raise ValueError(f"{source} did not execute a clean packed-KV kernel path.")
        ttft_gain = 1.0 - float(candidate["ttft_ms"]) / float(baseline["ttft_ms"])
        throughput_gain = (
            float(candidate["throughput_tokens_per_sec"])
            / float(baseline["throughput_tokens_per_sec"])
            - 1.0
        )
        pairs.append(
            {
                "source": source.name,
                "repeat": repeat,
                "sample_id": sample_id,
                "ttft_gain": ttft_gain,
                "throughput_gain": throughput_gain,
                # Accuracy is equal within this paired gate.  The remaining
                # competition speed objectives have equal 30% weights.
                "speed_proxy": 0.5 * (ttft_gain + throughput_gain),
            }
        )
    return pairs


def _summary(values: list[float], *, resamples: int, seed: int) -> dict:
    if not values:
        raise ValueError("cannot summarize an empty paired result set")
    rng = random.Random(seed)
    samples = []
    for _ in range(resamples):
        samples.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    samples.sort()
    return {
        "mean": sum(values) / len(values),
        "median": sorted(values)[len(values) // 2],
        "bootstrap_ci95": [samples[int(resamples * 0.025)], samples[int(resamples * 0.975)]],
    }


def main() -> None:
    args = parse_args()
    if args.resamples < 100:
        raise ValueError("--resamples must be at least 100.")
    all_pairs = []
    protocols = []
    for path in args.input:
        payload = json.loads(path.read_text(encoding="utf-8"))
        protocol = payload.get("protocol", {})
        if protocol.get("kv_chain_k_bits") != 4 or protocol.get("kv_chain_v_bits") != 4:
            raise ValueError(f"{path} is not a 4-bit candidate artifact.")
        if protocol.get("kv_chain_decode_warmup_tokens") != 4:
            raise ValueError(f"{path} does not use the protected four-token prefix.")
        all_pairs.extend(_paired_records(payload, path))
        protocols.append(protocol)

    ttft = _summary([row["ttft_gain"] for row in all_pairs], resamples=args.resamples, seed=args.seed)
    throughput = _summary(
        [row["throughput_gain"] for row in all_pairs], resamples=args.resamples, seed=args.seed + 1
    )
    speed_proxy = _summary(
        [row["speed_proxy"] for row in all_pairs], resamples=args.resamples, seed=args.seed + 2
    )
    payload = {
        "schema_version": "kv_paired_competition_gate_v1",
        "inputs": [str(path) for path in args.input],
        "pair_count": len(all_pairs),
        "protocols": protocols,
        "answer_parity": True,
        "token_count_parity": True,
        "ttft_gain": ttft,
        "throughput_gain": throughput,
        "competition_speed_proxy": speed_proxy,
        "positive_gain_gate": {
            "passed": speed_proxy["bootstrap_ci95"][0] > 0.0,
            "reason": "answer/token parity plus positive bootstrap lower bound of equal-weight TTFT/throughput proxy",
        },
        "pairs": all_pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
