"""PPU compatibility status reporting.

What this module now records
---------------------------
The scheduling-layer hybrid path *has* been executed end-to-end on the target
accelerator (PPU-ZW810E), so the previous blanket "unverified" status is no
longer accurate.  Validation environment of record:

    dataset        50 MMBench dev-en samples (public)
    max_new_tokens 256 (official setting)
    process        single process per arm, batch size 1, greedy
    entrypoint     optiz_qwen.evaluation.dndx_public_benchmark
    artifact       benchmarks/output/ppu_hybrid_trim_ab_50samples.json

What was validated is narrow and deliberately stated as such: sdpa prefill plus
one CUDA graph captured under ``flash_attention_2`` and replayed over a
``StaticCache``, the ``fla-core`` Triton gated-delta-net path with
``causal_conv1d`` available, and a last-position-only prefill lm_head.
That is *scheduling, backend selection, and redundant-work removal on a
CUDA-compatible shim*, not PPU native kernel work -- stage 7 of the execution
order in CLAUDE.md is untouched.

Prefill was additionally measured to be dispatch-bound on this device
(``cpu_issue_fraction`` 0.988+), the same diagnosis decode had.  That is a
measured property of the target recorded here so it is not re-litigated:
``causal_conv1d`` builds and is numerically correct, and removes 72 of 2423
kernel launches with no TTFT effect.

The captured-or-compiled-prefill lever has now been measured rather than left
open, and it is bounded (``benchmarks/output/ppu_prefill_headroom.json``,
``..._syncs.json``, ``..._vision_sync_elision.json``, ``..._vision_grid_shapes.json``,
``..._prefill_compile.json``):

    device_busy_fraction     0.47 - 0.51  -> ceiling ~1.95 - 2.13x, not decode's 8.9x
    kernel launches          2423 (the earlier 4700 double-counted operator rows)
    host syncs               93 - 94, of which 72 are one vision-attention line;
                             removing them is bit-exact and worth ~1%
    shape variability        24 distinct vision grids / 18 pixel shapes / 46 prompt
                             lengths over 50 samples -> no shape-free capture
    torch.compile(dynamic)   -12% to -14% TTFT, and not bit-exact

Still not validated, and still must not be claimed:

    - any PPU-native operator implementation
    - a captured or compiled *prefill* as a *win* (measured; see above -- the
      ceiling is ~2x and both routes to it are blocked or negative)
    - the deferred packed-KV INT4 chain as a *performance* path (measured
      -2.38% throughput on PPU; it is a memory-footprint alternative)
    - weight-only quantization as a *performance* path on this hardware
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Environment the hardware validation was performed in.  Kept as a tuple of
#: pairs so ``PpuCompatibilityStatus`` stays frozen and hashable.
VALIDATION_ENVIRONMENT = (
    ("target", "PPU-ZW810E"),
    ("dataset", "MMBench dev-en (public), 50 samples"),
    ("max_new_tokens", "256"),
    ("process", "single process per arm, batch size 1, greedy"),
    ("entrypoint", "optiz_qwen.evaluation.dndx_public_benchmark"),
    ("artifact", "benchmarks/output/ppu_hybrid_trim_ab_50samples.json"),
)

#: Code paths covered by that validation run.
VALIDATED_PATHS = (
    "scheduling.cuda_graph_decode (capture + replay over StaticCache)",
    "scheduling.prefill_decode (greedy split runner, last-position-only prefill lm_head)",
    "kernels.attention_backend (sdpa prefill / flash_attention_2 decode split)",
    "fla-core Triton gated-delta-net prefill path (causal_conv1d fast path available)",
)


@dataclass(frozen=True)
class PpuCompatibilityStatus:
    target: str
    claim: str
    can_claim_compatible: bool
    checked_on_target_hardware: bool
    official_materials_available: bool
    notes: tuple[str, ...]
    validated_paths: tuple[str, ...] = ()
    validation_environment: tuple[tuple[str, str], ...] = ()


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
        "The scheduling hybrid path was executed on PPU-ZW810E; see VALIDATION_ENVIRONMENT.",
        "Validated gains come from kernel-dispatch elimination, attention backend "
        "selection, and removing redundant prefill lm_head work -- not from PPU-native "
        "kernels.",
        "Prefill is dispatch-bound on this device too (cpu_issue_fraction 0.988+), but "
        "its device_busy_fraction is 0.47-0.51, so eliminating prefill dispatch entirely "
        "is bounded at ~1.95-2.13x -- not decode's 8.9x.",
        "One prefill issues 2423 kernel launches across 13698 operator calls; the "
        "previously reported 4700 summed CPU operator rows and device kernel rows and "
        "double-counted the same work.",
        "Captured prefill needs fixed shapes and has none: 46 distinct prompt lengths, "
        "24 distinct vision grids and 18 pixel shapes over 50 samples.",
        "torch.compile(dynamic=True) on the language stack or the vision tower made "
        "prefill 12-14% slower and was not bit-exact; rejected.",
        "72 of the 93-94 prefill host syncs come from one vision-attention .tolist(); "
        "eliding them is bit-exact but worth only ~1% (kernels.vision_prefill_sync, off "
        "by default via OPTIZ_QWEN_VISION_SYNC_ELISION).",
        "causal_conv1d builds on the target and is numerically correct, but removes only "
        "72 of 2423 prefill kernel launches and does not move TTFT.",
        "PPU-native operator development (execution order stage 7) is not started.",
        "The deferred packed-KV INT4 chain measured -2.38% throughput on PPU; it is "
        "retained as a memory-footprint alternative, not a performance path.",
        "No packed-KV PPU operator integration is implemented for Qwen3.5 VLM.",
    ]
    if not official_materials_available:
        notes.append("PPU reference materials are still missing under resources/ppu_docs/raw/.")
    return PpuCompatibilityStatus(
        target="Alibaba Cloud PPU-ZW810E",
        claim="scheduling_path_validated_on_target",
        can_claim_compatible=True,
        checked_on_target_hardware=True,
        official_materials_available=official_materials_available,
        notes=tuple(notes),
        validated_paths=VALIDATED_PATHS,
        validation_environment=VALIDATION_ENVIRONMENT,
    )
