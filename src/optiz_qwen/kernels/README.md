# Kernel Layer

This package maps to the third layer of the SVG:

- IO-aware attention kernels
- custom kernel integration
- nonlinear operator fusion
- dequant plus GEMM or GEMV fusion
- FFN-path fusion

This layer should stay behind reproducible benchmarks because kernel changes can improve speed while damaging correctness.
