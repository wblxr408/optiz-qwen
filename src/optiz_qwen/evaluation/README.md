# Evaluation Layer

This package is the proof layer for the competition:

- dataset adapters
- answer matching logic
- TTFT measurement
- throughput measurement
- baseline vs optimized comparison

This should become the first implemented runtime path once the official dataset and model assets are available.

Integrated DNDX public self-test assets:

- `dndx_wrapper.py`: participant `VLMModel` contract implementation
- `dndx_public_benchmark.py`: public MMBench self-test entrypoint
- repository-root `evaluation_wrapper.py` and `benchmark_public.py`: compatibility shims for organizer-style commands
