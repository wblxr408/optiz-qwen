"""PPU compatibility status reporting.

This module records what can be claimed today.  It does not perform a
hardware check and must not be used as proof of PPU compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PpuCompatibilityStatus:
    target: str
    claim: str
    can_claim_compatible: bool
    checked_on_target_hardware: bool
    official_materials_available: bool
    notes: tuple[str, ...]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def inspect_ppu_compatibility(root: Path | None = None) -> PpuCompatibilityStatus:
    base = root if root is not None else repo_root()
    ppu_docs = base / "resources" / "ppu_docs" / "raw"
    material_files = [
        path
        for path in ppu_docs.glob("**/*")
        if path.is_file() and path.name.lower() != "readme.md"
    ] if ppu_docs.exists() else []
    official_materials_available = bool(material_files)
    notes = [
        "PPU compatibility is not declared until target hardware or official compatibility materials are checked.",
        "Current KIVI KV-cache adapter is validated only through local Python/CUDA smoke paths.",
        "qB MatMul kernel integration is not enabled for Qwen3.5 VLM in the current adapter.",
    ]
    if not official_materials_available:
        notes.append("PPU reference materials are still missing under resources/ppu_docs/raw/.")
    return PpuCompatibilityStatus(
        target="Alibaba Cloud PPU",
        claim="unverified",
        can_claim_compatible=False,
        checked_on_target_hardware=False,
        official_materials_available=official_materials_available,
        notes=tuple(notes),
    )
