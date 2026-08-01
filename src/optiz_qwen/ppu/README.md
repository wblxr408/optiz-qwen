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

The current report is expected to be `claim="unverified"` until target
hardware or official compatibility materials are checked.

## Qwen3.5 GDN decode projection fusion

`install_qwen35_gdn_decode_projection_fusion` packs the QKV, Z, B, and A
weights of every Gated DeltaNet layer into one shared storage allocation. It
uses one projection for the batch-1 single-token path, which is the steady-state
decode path; multi-token prefill keeps the original four-projection execution
order. Install it only after the model has reached its final device and dtype.

The DNDX benchmark exposes this path through the default-off
`--enable-gdn-decode-projection-fusion` switch.

## Qwen3.5 PPU Delta Rule

`install_qwen35_ppu_delta_kernel` compiles the HGGC/CUDA-compatible extension
and replaces chunk prefill on an explicit Gated DeltaNet layer subset. The
validated configuration uses the final nine GDN layers and double-precision
dot-product reduction with FP32 recurrent state. Initial states and packed
sequences are rejected explicitly.

The default benchmark path never compiles or installs this target-specific
kernel. Enable it with `--enable-ppu-delta-kernel`; the default selection is
`--ppu-delta-kernel-layers 9 --ppu-delta-kernel-position last`.
