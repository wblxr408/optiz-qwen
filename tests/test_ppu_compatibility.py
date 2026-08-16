from __future__ import annotations

from optiz_qwen.ppu import inspect_ppu_compatibility


def test_ppu_status_reports_target_hardware_validation(tmp_path) -> None:
    status = inspect_ppu_compatibility(tmp_path)

    assert status.claim == "scheduling_path_validated_on_target"
    assert status.checked_on_target_hardware is True
    assert status.can_claim_compatible is True
    # tmp_path has no resources/ppu_docs/raw, so material discovery must still
    # report absent -- hardware validation does not imply official docs exist.
    assert status.official_materials_available is False


def test_ppu_status_scopes_the_claim_to_the_scheduling_path(tmp_path) -> None:
    status = inspect_ppu_compatibility(tmp_path)

    joined = " ".join(status.validated_paths)
    assert "cuda_graph_decode" in joined
    assert "attention_backend" in joined
    # The claim must stay narrow: nothing here may read as PPU-native kernels.
    assert all("native" not in path.lower() for path in status.validated_paths)


def test_ppu_status_records_the_validation_environment(tmp_path) -> None:
    status = inspect_ppu_compatibility(tmp_path)

    environment = dict(status.validation_environment)
    assert environment["target"] == "PPU-ZW810E"
    assert environment["max_new_tokens"] == "256"
    assert "50 samples" in environment["dataset"]
    # The claim is only as good as its source: the numbers must come from the
    # shipped entrypoint, not an ad-hoc probe script.
    assert environment["entrypoint"] == "optiz_qwen.evaluation.dndx_public_benchmark"
    assert environment["artifact"].endswith(".json")


def test_ppu_status_records_that_prefill_is_dispatch_bound(tmp_path) -> None:
    """A measured property of the target, kept so it is not re-litigated."""

    notes = " ".join(inspect_ppu_compatibility(tmp_path).notes)

    assert "dispatch-bound" in notes
    assert "cpu_issue_fraction" in notes
    # causal_conv1d must not read as a speed win anywhere in the status record.
    assert "does not move TTFT" in notes


def test_ppu_status_still_disclaims_native_kernels_and_packed_kv(tmp_path) -> None:
    notes = " ".join(inspect_ppu_compatibility(tmp_path).notes)

    assert "stage 7" in notes
    assert "-2.38%" in notes
    assert "not a performance path" in notes


def test_ppu_status_bounds_the_prefill_dispatch_ceiling(tmp_path) -> None:
    """Prefill's headroom is measured, so it must not read as decode's 8.9x.

    ``scripts/profile_ppu_prefill_headroom.py`` measured device_busy_fraction
    0.47-0.51 over 2423 kernel launches. The older 4700 figure summed CPU
    operator rows and device kernel rows, double-counting the same work.
    """

    notes = " ".join(inspect_ppu_compatibility(tmp_path).notes)

    assert "2423" in notes
    assert "0.47-0.51" in notes
    assert "1.95-2.13x" in notes
    # The corrected count must be present *and* explained, so the stale figure
    # cannot quietly reappear as fact.
    assert "double-counted" in notes


def test_ppu_status_records_why_captured_prefill_is_blocked(tmp_path) -> None:
    notes = " ".join(inspect_ppu_compatibility(tmp_path).notes)

    # Shape variability is the blocker for capture; compile was measured negative.
    assert "46 distinct prompt lengths" in notes
    assert "24 distinct vision grids" in notes
    assert "torch.compile" in notes and "slower" in notes
    # The vision sync elision is a ~1% correctness-neutral win, not a TTFT lever.
    assert "OPTIZ_QWEN_VISION_SYNC_ELISION" in notes
    assert "~1%" in notes
