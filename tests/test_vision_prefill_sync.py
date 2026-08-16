"""Tests for the vision prefill host-sync elision.

The patch in ``optiz_qwen.kernels.vision_prefill_sync`` replaces a per-block
device-to-host copy with a host-side value.  What can go wrong is not the
arithmetic -- it is the ``tolist`` override serving its memoized list to a
*different* tensor, or failing to restore the override.  Both are pinned here.

Measured payoff, for context: 72 of 93 prefill syncs removed, ~1% wall clock,
logits bit-identical (``benchmarks/output/ppu_vision_sync_elision.json``).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from optiz_qwen.kernels.vision_prefill_sync import (  # noqa: E402
    VISION_SYNC_ELISION_ENV,
    chunk_lengths_from_grid,
    elide_vision_attention_host_sync,
    vision_sync_elision_available,
    vision_sync_elision_enabled,
)


class _Attention:
    def __init__(self) -> None:
        self.seen: list[list[int]] = []

    def forward(self, lengths_tensor):  # noqa: ANN001, ANN202
        # Stands in for the upstream ``lengths.tolist()`` at
        # modeling_qwen3_5.py:968.
        self.seen.append(lengths_tensor.tolist())
        return lengths_tensor


class _Block:
    def __init__(self) -> None:
        self.attn = _Attention()


class _Visual:
    def __init__(self, blocks: int = 3) -> None:
        self.blocks = [_Block() for _ in range(blocks)]

    def forward(self, hidden_states, grid_thw, **kwargs):  # noqa: ANN001, ANN202
        # The real tower hands the same lengths tensor to every block; the
        # values here are deliberately wrong so a served list is detectable.
        lengths = torch.tensor([-1] * _expected_chunks(grid_thw), dtype=torch.int64)
        return [block.attn.forward(lengths) for block in self.blocks]


def _expected_chunks(grid_thw) -> int:  # noqa: ANN001
    try:
        return sum(int(row[0]) for row in grid_thw.tolist())
    except Exception:
        return 1


class _Model:
    """Mirrors the real nesting: the LM owns ``.model``, which owns ``.visual``."""

    def __init__(self) -> None:
        self.model = type("Inner", (), {})()
        self.model.visual = _Visual()


def test_chunk_lengths_single_image() -> None:
    grid = torch.tensor([[1, 16, 24]], dtype=torch.int64)
    assert chunk_lengths_from_grid(grid) == [16 * 24]


def test_chunk_lengths_repeats_per_temporal_frame() -> None:
    grid = torch.tensor([[2, 4, 5], [1, 3, 3]], dtype=torch.int64)
    # Two frames of 20 patches, then one of 9 -- matches upstream's
    # repeat_interleave(h * w, t).
    assert chunk_lengths_from_grid(grid) == [20, 20, 9]


def test_chunk_lengths_accepts_plain_sequences() -> None:
    assert chunk_lengths_from_grid([[1, 2, 3]]) == [6]


def test_availability_requires_vision_blocks() -> None:
    assert vision_sync_elision_available(_Model()) is True
    assert vision_sync_elision_available(object()) is False


def test_availability_accepts_inner_model_directly() -> None:
    inner = _Model().model
    assert vision_sync_elision_available(inner) is True


def test_enabled_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(VISION_SYNC_ELISION_ENV, raising=False)
    assert vision_sync_elision_enabled() is False
    for value in ("1", "true", "YES", "on"):
        monkeypatch.setenv(VISION_SYNC_ELISION_ENV, value)
        assert vision_sync_elision_enabled() is True
    monkeypatch.setenv(VISION_SYNC_ELISION_ENV, "0")
    assert vision_sync_elision_enabled() is False


def test_patch_serves_host_lengths_to_every_block() -> None:
    model = _Model()
    grid = torch.tensor([[1, 16, 24]], dtype=torch.int64)

    with elide_vision_attention_host_sync(model) as installed:
        assert installed is True
        model.model.visual.forward(None, grid)

    for block in model.model.visual.blocks:
        # The sentinel -1 never reaches the caller: each block got the value
        # derived once from grid_thw.
        assert block.attn.seen == [[16 * 24]]


def test_patch_is_a_noop_without_blocks() -> None:
    with elide_vision_attention_host_sync(object()) as installed:
        assert installed is False


def test_forwards_are_restored_on_exit() -> None:
    model = _Model()
    visual = model.model.visual
    before = (visual.forward, [block.attn.forward for block in visual.blocks])

    with elide_vision_attention_host_sync(model):
        assert visual.forward is not before[0]

    assert visual.forward == before[0]
    assert [block.attn.forward for block in visual.blocks] == before[1]


def test_forwards_are_restored_on_exception() -> None:
    model = _Model()
    visual = model.model.visual
    before = visual.forward

    with pytest.raises(RuntimeError):
        with elide_vision_attention_host_sync(model):
            raise RuntimeError("boom")

    assert visual.forward == before


def test_tolist_override_is_removed_after_the_forward() -> None:
    model = _Model()
    grid = torch.tensor([[1, 4, 5]], dtype=torch.int64)

    with elide_vision_attention_host_sync(model):
        model.model.visual.forward(None, grid)
        # Outside the attention forward the real tolist must be back, so an
        # unrelated 1-D int tensor of the same length reads its own values.
        assert torch.tensor([7], dtype=torch.int64).tolist() == [7]

    assert torch.tensor([7], dtype=torch.int64).tolist() == [7]


def test_override_falls_through_for_non_matching_tensors() -> None:
    """The guard is rank + numel + integer dtype; anything else is untouched."""

    model = _Model()
    grid = torch.tensor([[2, 4, 5], [1, 3, 3]], dtype=torch.int64)
    expected = [20, 20, 9]
    observed: dict[str, object] = {}

    def probing_forward(lengths_tensor):  # noqa: ANN001, ANN202
        observed["served"] = lengths_tensor.tolist()
        # Same shape and count but floating point -> must not be served.
        observed["float"] = torch.tensor([1.5, 2.5, 3.5]).tolist()
        # Right dtype, wrong element count -> must not be served.
        observed["short"] = torch.tensor([9, 9], dtype=torch.int64).tolist()
        # Right dtype and count, wrong rank -> must not be served.
        observed["rank2"] = torch.tensor([[1, 2, 3]], dtype=torch.int64).tolist()
        return lengths_tensor

    model.model.visual.blocks = [type("B", (), {"attn": type("A", (), {"forward": staticmethod(probing_forward)})()})()]

    with elide_vision_attention_host_sync(model):
        model.model.visual.forward(None, grid)

    assert observed["served"] == expected
    assert observed["float"] == [1.5, 2.5, 3.5]
    assert observed["short"] == [9, 9]
    assert observed["rank2"] == [[1, 2, 3]]


def test_unpatched_forward_still_syncs_when_grid_is_unusable() -> None:
    """A grid the helper cannot read must degrade to the original path."""

    model = _Model()

    class _BadGrid:
        def tolist(self):  # noqa: ANN202
            raise ValueError("no host copy available")

    with elide_vision_attention_host_sync(model):
        model.model.visual.forward(None, torch.tensor([[1, 2, 3]], dtype=torch.int64))
        # Sanity: with a readable grid the patch is active, so the sentinel is
        # hidden. With the bad grid below it must be visible again.
        served = [block.attn.seen[-1] for block in model.model.visual.blocks]
        assert all(value == [6] for value in served)

        model.model.visual.blocks[0].attn.seen.clear()
        try:
            model.model.visual.forward(None, _BadGrid())
        except Exception:
            pytest.fail("a bad grid must not break the vision forward")
        assert model.model.visual.blocks[0].attn.seen == [[-1]]
