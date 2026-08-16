# PPU Layer

This package maps to the fourth layer of the SVG:

- lookup-table operator coverage
- HBM bandwidth optimization
- weight packing format
- unsupported-op rewrites

No module in this layer should claim completion until checked against real PPU constraints.

Current compatibility status is exposed by:

```python
from optiz_qwen.ppu import inspect_ppu_compatibility
```

The report is now `claim="scheduling_path_validated_on_target"`: the
scheduling-layer hybrid (sdpa prefill + one CUDA graph captured under
`flash_attention_2` replayed over a `StaticCache`, plus the `fla-core` Triton
gated-delta-net prefill) has been executed end-to-end on PPU-ZW810E over 32
MMBench dev-en samples at `max_new_tokens=256` in a single process.

That claim is deliberately narrow. Nothing in this layer is a PPU-native
kernel: execution-order stage 7 has not started, and the measured gains come
from eliminating per-token kernel dispatch and from attention backend
selection. `inspect_ppu_compatibility().validated_paths` enumerates exactly
what the run covered, and `.notes` records what still must not be claimed --
including that the deferred packed-KV INT4 chain measured **-2.38% throughput**
on PPU and is retained as a memory-footprint alternative rather than a
performance path.
