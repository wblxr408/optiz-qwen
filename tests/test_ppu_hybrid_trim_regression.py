"""Performance regression baseline for the PPU hybrid + trimmed-prefill arm.

This is the project's primary reference point, superseding the 32-sample probe
pinned by ``test_ppu_hybrid_regression.py``.  Two things make it better
evidence: it was produced by the real repo entrypoint
(``optiz_qwen.evaluation.dndx_public_benchmark``) rather than an ad-hoc probe
script, and it has three arms so the two independent TTFT levers are separable:

- ``baseline``     greedy runner, no graph, no prefill logits trim
- ``hybrid``       CUDA-graph decode (FA2 capture / sdpa prefill), no trim
- ``hybrid_trim``  the same graph plus ``logits_to_keep=1`` at prefill

Environment of record: PPU-ZW810E, 50 MMBench dev-en samples,
``max_new_tokens=256``, single process, batch size 1, greedy,
``max_cache_len=2048``, ``fla-core==0.5.2`` and ``causal_conv1d==1.6.2.post1``
both installed so ``is_fast_path_available`` is True.  Nothing here runs the
model -- these are assertions about a saved measurement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "output"
    / "ppu_hybrid_trim_ab_50samples.json"
)

SAMPLE_COUNT = 50
ACCURACY = 0.76
BASELINE_TTFT_MS = 58.436
BASELINE_THROUGHPUT = 44.439
HYBRID_TTFT_MS = 55.378
HYBRID_THROUGHPUT = 162.937
TRIM_TTFT_MS = 52.556
TRIM_THROUGHPUT = 162.312
TRIM_TTFT_IMPROVEMENT_PCT = 10.062
TRIM_THROUGHPUT_IMPROVEMENT_PCT = 265.247
# What the prefill trim adds on top of the graph, isolated.
TRIM_ONLY_TTFT_IMPROVEMENT_PCT = 5.096
ANSWER_PARITY = 50
TOKEN_PARITY = 49


@pytest.fixture(scope="module")
def artifact() -> dict:
    if not ARTIFACT.exists():
        pytest.skip(f"validation artifact is missing: {ARTIFACT}")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def summary(artifact: dict) -> dict:
    return artifact["summary"]


def test_measurement_conditions_are_the_official_ones(summary: dict) -> None:
    assert summary["n"] == SAMPLE_COUNT
    assert summary["max_new_tokens"] == 256
    assert summary["max_cache_len"] == 2048
    assert summary["device"] == "PPU-ZW810E"
    # The claim is only worth pinning if it came from the shipped entrypoint.
    assert summary["entrypoint"] == "optiz_qwen.evaluation.dndx_public_benchmark"


def test_accuracy_is_identical_across_all_three_arms(summary: dict) -> None:
    for arm in ("baseline", "hybrid", "hybrid_trim"):
        assert summary[arm]["accuracy"] == pytest.approx(ACCURACY)
        assert summary[arm]["correct"] == 38


def test_the_arms_differ_only_in_the_intended_switches(summary: dict) -> None:
    assert summary["baseline"]["prefill_logits_trimmed"] == [False]
    assert summary["hybrid"]["prefill_logits_trimmed"] == [False]
    assert summary["hybrid_trim"]["prefill_logits_trimmed"] == [True]
    graph = summary["graph"]
    assert graph["captured"] is True
    # The split is the mechanism: FA2 frozen into the graph, sdpa left live for
    # prefill.  If these ever match, one of the two wins has been given up.
    assert graph["capture_backend"] == "flash_attention_2"
    assert graph["prefill_backend"] == "sdpa"


def test_the_gdn_fast_path_was_live(summary: dict) -> None:
    fast_path = summary["fast_path"]
    assert fast_path["is_fast_path_available"] is True
    assert fast_path["fla_core"] == "0.5.2"
    assert fast_path["causal_conv1d"] == "1.6.2.post1"


def test_ttft_does_not_regress(summary: dict) -> None:
    assert summary["baseline"]["avg_ttft_ms"] == pytest.approx(BASELINE_TTFT_MS, abs=0.001)
    # Lower is better, so each optimized arm must stay at or under reference.
    assert summary["hybrid"]["avg_ttft_ms"] <= HYBRID_TTFT_MS + 0.001
    assert summary["hybrid_trim"]["avg_ttft_ms"] <= TRIM_TTFT_MS + 0.001
    assert summary["hybrid_trim"]["ttft_improvement_pct"] >= TRIM_TTFT_IMPROVEMENT_PCT - 0.001


def test_the_prefill_trim_is_a_real_win_on_top_of_the_graph(summary: dict) -> None:
    # This is the number that separates the new lever from the graph's own gain.
    assert summary["trim_only_ttft_improvement_pct"] >= TRIM_ONLY_TTFT_IMPROVEMENT_PCT - 0.001
    assert summary["hybrid_trim"]["avg_ttft_ms"] < summary["hybrid"]["avg_ttft_ms"]


def test_throughput_does_not_regress(summary: dict) -> None:
    assert summary["baseline"]["avg_throughput"] == pytest.approx(BASELINE_THROUGHPUT, abs=0.001)
    assert summary["hybrid"]["avg_throughput"] >= HYBRID_THROUGHPUT - 0.001
    assert summary["hybrid_trim"]["avg_throughput"] >= TRIM_THROUGHPUT - 0.001
    assert (
        summary["hybrid_trim"]["throughput_improvement_pct"]
        >= TRIM_THROUGHPUT_IMPROVEMENT_PCT - 0.001
    )


def test_the_prefill_trim_costs_no_throughput(summary: dict) -> None:
    # The trim touches prefill only, so decode should be unaffected; allow the
    # measurement noise band rather than asserting an exact tie.
    hybrid = summary["hybrid"]["avg_throughput"]
    trimmed = summary["hybrid_trim"]["avg_throughput"]
    assert abs(trimmed - hybrid) / hybrid < 0.01


def test_answer_parity_is_total_and_token_parity_is_near_total(summary: dict) -> None:
    assert summary["answer_parity"] == ANSWER_PARITY == SAMPLE_COUNT
    # The single token-parity miss is a greedy tie-break, not a defect; the
    # 32-sample root-cause ladder in docs/ppu_optimization_design.md showed the
    # graph itself is bit-faithful.  It must not grow.
    assert summary["token_parity"] >= TOKEN_PARITY


def test_summary_matches_its_own_records(artifact: dict) -> None:
    records = artifact["records"]
    assert len(records) == SAMPLE_COUNT

    summary = artifact["summary"]
    for arm in ("baseline", "hybrid", "hybrid_trim"):
        correct = sum(1 for record in records if record[arm]["correct"])
        mean_ttft = sum(record[arm]["ttft_ms"] for record in records) / len(records)
        mean_throughput = sum(record[arm]["throughput"] for record in records) / len(records)

        assert summary[arm]["correct"] == correct
        assert summary[arm]["accuracy"] == pytest.approx(correct / len(records), abs=1e-6)
        assert summary[arm]["avg_ttft_ms"] == pytest.approx(mean_ttft, abs=0.01)
        assert summary[arm]["avg_throughput"] == pytest.approx(mean_throughput, abs=0.01)

    assert summary["token_parity"] == sum(1 for r in records if r["token_parity"])
    assert summary["answer_parity"] == sum(1 for r in records if r["answer_parity"])


def test_improvement_percentages_match_the_arm_means(summary: dict) -> None:
    baseline = summary["baseline"]
    for arm in ("hybrid", "hybrid_trim"):
        ttft_pct = (
            (baseline["avg_ttft_ms"] - summary[arm]["avg_ttft_ms"])
            / baseline["avg_ttft_ms"]
            * 100.0
        )
        throughput_pct = (
            (summary[arm]["avg_throughput"] - baseline["avg_throughput"])
            / baseline["avg_throughput"]
            * 100.0
        )
        assert summary[arm]["ttft_improvement_pct"] == pytest.approx(ttft_pct, abs=0.01)
        assert summary[arm]["throughput_improvement_pct"] == pytest.approx(throughput_pct, abs=0.01)


def test_all_arms_saw_the_same_prompts_and_token_budget(artifact: dict) -> None:
    for record in artifact["records"]:
        lengths = {record[arm]["plen"] for arm in ("baseline", "hybrid", "hybrid_trim")}
        # Same prompt length in every arm, else it is not an A/B.
        assert len(lengths) == 1, record["id"]
        for arm in ("baseline", "hybrid", "hybrid_trim"):
            assert record[arm]["n_tok"] <= 256


def test_the_graph_wins_throughput_on_every_sample(artifact: dict) -> None:
    # Removing per-token dispatch should be uniform, not an average rescued by
    # outliers.
    for record in artifact["records"]:
        assert record["hybrid_trim"]["throughput"] > record["baseline"]["throughput"], record["id"]
