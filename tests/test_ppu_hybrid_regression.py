"""Performance regression baseline for the PPU CUDA-graph hybrid.

Superseded as the primary reference by ``test_ppu_hybrid_trim_regression.py``,
which pins a 50-sample three-arm A/B produced by the real repo entrypoint.
This file is kept because it guards a different artifact -- the original
32-sample probe -- and losing it would lose the earliest reproducible evidence.
Do not update these constants to the newer numbers; they describe a different
run.

This pins the validated A/B result as a reference point.  The
artifact is the evidence; these tests are the guard that keeps it honest:

- they fail if the artifact is edited to claim a different result
- they fail if a future run is written into the same file with worse numbers
- they re-derive the aggregate numbers from the 32 per-sample records, so a
  summary that no longer matches its own records is caught

Environment of record: PPU-ZW810E, 32 MMBench dev-en samples,
``max_new_tokens=256``, single process, batch size 1, greedy, ``max_cache_len``
2048.  Nothing here runs the model -- these are assertions about a saved
measurement, per the repo's reporting discipline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "output"
    / "ppu_cudagraph_hybrid_ab_32samples.json"
)

# The validated reference point.  A regression below any of these fails the
# suite; an improvement requires updating these constants together with a new
# artifact, which is deliberately a visible change.
BASELINE_ACCURACY = 0.75
BASELINE_TTFT_MS = 54.384
BASELINE_THROUGHPUT = 46.743
HYBRID_ACCURACY = 0.75
HYBRID_TTFT_MS = 52.387
HYBRID_THROUGHPUT = 156.864
TTFT_IMPROVEMENT_PCT = 3.672
THROUGHPUT_IMPROVEMENT_PCT = 235.588
SAMPLE_COUNT = 32
TOKEN_PARITY = 31
ANSWER_PARITY = 32


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


def test_accuracy_is_not_traded_away(summary: dict) -> None:
    assert summary["hybrid"]["accuracy"] >= HYBRID_ACCURACY
    assert summary["hybrid"]["accuracy"] >= summary["baseline"]["accuracy"]
    assert summary["baseline"]["accuracy"] == pytest.approx(BASELINE_ACCURACY)


def test_ttft_does_not_regress(summary: dict) -> None:
    assert summary["baseline"]["avg_ttft_ms"] == pytest.approx(BASELINE_TTFT_MS, abs=0.001)
    # Lower is better, so the hybrid must stay at or under the reference.
    assert summary["hybrid"]["avg_ttft_ms"] <= HYBRID_TTFT_MS + 0.001
    assert summary["ttft_improvement_pct"] >= TTFT_IMPROVEMENT_PCT - 0.001


def test_throughput_does_not_regress(summary: dict) -> None:
    assert summary["baseline"]["avg_throughput"] == pytest.approx(BASELINE_THROUGHPUT, abs=0.001)
    assert summary["hybrid"]["avg_throughput"] >= HYBRID_THROUGHPUT - 0.001
    assert summary["throughput_improvement_pct"] >= THROUGHPUT_IMPROVEMENT_PCT - 0.001


def test_answer_parity_is_total_and_token_parity_is_near_total(summary: dict) -> None:
    assert summary["answer_parity"] == ANSWER_PARITY == SAMPLE_COUNT
    # The single token-parity miss is a greedy tie-break (top1-top2 logit gap
    # 0.0000 at the divergence step), not a correctness defect.  It must not grow.
    assert summary["token_parity"] >= TOKEN_PARITY


def test_summary_matches_its_own_records(artifact: dict) -> None:
    records = artifact["records"]
    assert len(records) == SAMPLE_COUNT

    summary = artifact["summary"]
    for arm in ("baseline", "hybrid"):
        correct = sum(1 for record in records if record[arm]["correct"])
        accuracy = correct / len(records)
        mean_ttft = sum(record[arm]["ttft_ms"] for record in records) / len(records)
        mean_throughput = sum(record[arm]["throughput"] for record in records) / len(records)

        assert summary[arm]["correct"] == correct
        assert summary[arm]["accuracy"] == pytest.approx(accuracy, abs=1e-6)
        assert summary[arm]["avg_ttft_ms"] == pytest.approx(mean_ttft, abs=0.01)
        assert summary[arm]["avg_throughput"] == pytest.approx(mean_throughput, abs=0.01)

    assert summary["token_parity"] == sum(1 for r in records if r["token_parity"])
    assert summary["answer_parity"] == sum(1 for r in records if r["answer_parity"])


def test_improvement_percentages_match_the_arm_means(summary: dict) -> None:
    baseline, hybrid = summary["baseline"], summary["hybrid"]
    ttft_pct = (baseline["avg_ttft_ms"] - hybrid["avg_ttft_ms"]) / baseline["avg_ttft_ms"] * 100.0
    throughput_pct = (
        (hybrid["avg_throughput"] - baseline["avg_throughput"]) / baseline["avg_throughput"] * 100.0
    )

    assert summary["ttft_improvement_pct"] == pytest.approx(ttft_pct, abs=0.01)
    assert summary["throughput_improvement_pct"] == pytest.approx(throughput_pct, abs=0.01)


def test_both_arms_saw_the_same_prompts_and_token_budget(artifact: dict) -> None:
    for record in artifact["records"]:
        # Same prompt length in both arms, else the comparison is not an A/B.
        assert record["baseline"]["plen"] == record["hybrid"]["plen"]
        assert record["baseline"]["n_tok"] <= 256
        assert record["hybrid"]["n_tok"] <= 256


def test_hybrid_wins_throughput_on_every_sample(artifact: dict) -> None:
    # The gain comes from removing per-token dispatch, so it should be uniform
    # rather than an average rescued by outliers.
    for record in artifact["records"]:
        assert record["hybrid"]["throughput"] > record["baseline"]["throughput"], record["id"]
