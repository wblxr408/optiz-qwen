from __future__ import annotations

from optiz_qwen.ppu import inspect_ppu_compatibility


def test_ppu_status_does_not_claim_unverified_compatibility(tmp_path) -> None:
    status = inspect_ppu_compatibility(tmp_path)

    assert status.claim == "unverified"
    assert status.can_claim_compatible is False
    assert status.checked_on_target_hardware is False
    assert status.official_materials_available is False
