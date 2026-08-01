# Kernel Layer

This package maps to the third layer of the SVG:

- IO-aware attention kernels
- custom kernel integration
- nonlinear operator fusion
- dequant plus GEMM or GEMV fusion
- FFN-path fusion

This layer should stay behind reproducible benchmarks because kernel changes can improve speed while damaging correctness.

`qwen35_chunk_delta.py` defines the FP32-state numerical reference and
installation contract used to validate target-specific Delta Rule kernels. The
reference is a correctness oracle, not a performance implementation. The PPU
implementation lives under `optiz_qwen.ppu`; packed sequences remain rejected
until that layout has its own kernel and validation evidence.
