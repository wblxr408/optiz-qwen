# Compression Layer

This package maps to the first layer of the SVG:

- visual token pruning
- text-guided sparsification
- AWQ W4A16
- KV cache quantization
- VLM-specific PTQ
- structured pruning

Recommended implementation priority:

1. AWQ W4A16
2. visual token pruning
3. optional extensions after baseline metrics are stable

## Retained KV experiment

`qserve_kv_cache.py` contains the only retained KV experiment:
`qserve_deferred_split_fused_kv`. It keeps prefill KV dense, transfers native
prefill state after the first token, then uses an INT4 split packed-KV decode
path. The default baseline does not enable it.

The chain is numerically validated but has no meaningful short-context CUDA
gain. It remains only for long-context and PPU operator evaluation.
