"""Elide the vision tower's per-block host synchronization during prefill.

The problem
-----------
``Qwen3_5VisionAttention.forward`` splits Q/K/V per image chunk::

    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    splits = [torch.split(t, lengths.tolist(), dim=2) for t in (q, k, v)]

``lengths`` lives on the device, so ``.tolist()`` is a device-to-host copy that
blocks the CPU until the queue drains.  It happens once per attention forward and
the tower has 24 blocks, which ``scripts/probe_ppu_prefill_syncs.py`` measured as
**72 of the 93-94 host syncs in one prefill** on PPU-ZW810E.

Why that is worth removing
--------------------------
Prefill measured ``cpu_issue_fraction`` 0.989, which reads as "launch-bound".
But a host sync produces the same signature for a different reason: while the CPU
sits inside ``.tolist()`` it can neither issue nor run ahead, so issue time is
pinned to wall time by stalling rather than by launch cost.  The device-busy
fraction is only 0.47-0.51 (``scripts/profile_ppu_prefill_headroom.py``), so about
half of prefill is idle device time and the two explanations are not
distinguishable from the ratio alone.

The value is recomputable on the host
-------------------------------------
``cu_seqlens`` is derived from ``grid_thw`` once per forward and the *same tensor
object* is handed to every block, so all 24 blocks compute an identical list.
``grid_thw`` also determines it in closed form -- upstream builds it as
``repeat_interleave(grid[:, 1] * grid[:, 2], grid[:, 0]).cumsum()`` -- so the
chunk lengths can be derived once from a single ``grid_thw`` read and served to
every block from the host.

Scope of the patch
------------------
Deliberately narrow.  ``torch.Tensor.tolist`` is overridden only for the duration
of each *attention* forward, not the whole vision forward: inside the vision
forward there are other ``tolist`` calls (``grid_thw``, ``spatial_merge_size``)
and for a single-image batch ``spatial_merge_size`` has the same 1-D shape as
``lengths``, so a wider scope could serve the wrong list.  Inside
``attn.forward`` the only ``tolist`` call is the one being replaced.

Equality is by construction (same list of ints), and
``scripts/probe_vision_sync_elision.py`` additionally checks prefill logits are
bit-identical with the patch on.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

#: Kill switch.  Off by default until the measured payoff justifies changing the
#: default -- see ``benchmarks/output/ppu_vision_sync_elision.json``.
VISION_SYNC_ELISION_ENV = "OPTIZ_QWEN_VISION_SYNC_ELISION"


def vision_sync_elision_enabled() -> bool:
    value = os.environ.get(VISION_SYNC_ELISION_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _resolve_visual(model: Any) -> Any | None:
    """Find the vision tower, whether the caller passed the LM or the inner model."""

    for holder in (model, getattr(model, "model", None)):
        visual = getattr(holder, "visual", None)
        if visual is not None:
            return visual
    return None


def _attention_modules(visual: Any) -> list[Any]:
    blocks = getattr(visual, "blocks", None)
    if blocks is None:
        return []
    modules = []
    for block in blocks:
        attn = getattr(block, "attn", None)
        if attn is not None and callable(getattr(attn, "forward", None)):
            modules.append(attn)
    return modules


def vision_sync_elision_available(model: Any) -> bool:
    """Whether this model exposes the vision blocks the patch needs."""

    visual = _resolve_visual(model)
    return visual is not None and bool(_attention_modules(visual))


def chunk_lengths_from_grid(grid_thw: Any) -> list[int]:
    """Per-chunk attention lengths, derived on the host from ``grid_thw``.

    Mirrors ``transformers.vision_utils.get_vision_cu_seqlens``: each row
    ``(t, h, w)`` contributes ``t`` chunks of ``h * w`` patches.  One host read of
    a ``(num_images, 3)`` tensor replaces one per attention block.
    """

    rows = grid_thw.tolist() if hasattr(grid_thw, "tolist") else list(grid_thw)
    lengths: list[int] = []
    for row in rows:
        temporal, height, width = int(row[0]), int(row[1]), int(row[2])
        lengths.extend([height * width] * temporal)
    return lengths


@contextmanager
def _served_tolist(expected: list[int]) -> Iterator[None]:
    """Serve ``tolist()`` from ``expected`` for the lengths tensor only.

    Guarded on rank, element count, and integer dtype so anything that is not the
    chunk-lengths tensor falls through to the real implementation.
    """

    import torch

    original = torch.Tensor.tolist
    count = len(expected)

    def tolist(self):  # noqa: ANN001, ANN202 - mirrors torch's own signature
        if (
            self.dim() == 1
            and self.numel() == count
            and not self.is_floating_point()
            and not self.is_complex()
        ):
            return list(expected)
        return original(self)

    torch.Tensor.tolist = tolist  # type: ignore[method-assign]
    try:
        yield
    finally:
        torch.Tensor.tolist = original  # type: ignore[method-assign]


@contextmanager
def elide_vision_attention_host_sync(model: Any) -> Iterator[bool]:
    """Wrap the vision tower so per-block chunk lengths never leave the device.

    Yields whether the patch was actually installed, so a caller can record the
    fact instead of assuming it.  Restores every wrapped method on exit, including
    on exception.
    """

    visual = _resolve_visual(model)
    attentions = _attention_modules(visual) if visual is not None else []
    if visual is None or not attentions:
        yield False
        return

    # Populated by the vision forward, read by each attention forward.  A holder
    # rather than a closure variable so the two wrappers stay independent.
    holder: dict[str, list[int] | None] = {"lengths": None}

    original_visual_forward = visual.forward

    def visual_forward(hidden_states, grid_thw, **kwargs):  # noqa: ANN001, ANN202
        previous = holder["lengths"]
        try:
            holder["lengths"] = chunk_lengths_from_grid(grid_thw)
        except Exception:  # pragma: no cover - unexpected grid layout
            holder["lengths"] = None
        try:
            return original_visual_forward(hidden_states, grid_thw, **kwargs)
        finally:
            holder["lengths"] = previous

    originals = [(attn, attn.forward) for attn in attentions]

    def make_attention_forward(original_forward):  # noqa: ANN001, ANN202
        def attention_forward(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            lengths = holder["lengths"]
            if lengths is None:
                return original_forward(*args, **kwargs)
            with _served_tolist(lengths):
                return original_forward(*args, **kwargs)

        return attention_forward

    visual.forward = visual_forward
    for attn, original_forward in originals:
        attn.forward = make_attention_forward(original_forward)
    try:
        yield True
    finally:
        visual.forward = original_visual_forward
        for attn, original_forward in originals:
            attn.forward = original_forward
