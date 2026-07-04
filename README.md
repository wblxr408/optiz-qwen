# optiz-qwen

Qwen3.5-2B VLM edge-deployment optimization workspace for the PPU competition track.

## Current status

- The project skeleton is in place.
- The evaluation dataset has not been downloaded yet.
- Model weights and part of the PPU reference materials have not been downloaded yet.
- Optimization modules are intentionally left unimplemented for now.

## Project layout

```text
.
|-- AGENTS.md
|-- CLAUDE.md
|-- configs/
|-- docs/
|-- resources/
|-- scripts/
|-- src/optiz_qwen/
|   |-- common/
|   |-- compression/
|   |-- evaluation/
|   |-- kernels/
|   |-- ppu/
|   `-- scheduling/
|-- benchmarks/
|-- reports/
`-- tests/
```

## Layer mapping from the SVG

- `compression/`: visual token pruning, AWQ, KV cache quantization, VLM PTQ
- `scheduling/`: paged KV, prefill/decode split, continuous batching, mixed precision service
- `kernels/`: attention kernels, fused FFN path, nonlinear operator fusion
- `ppu/`: operator coverage, bandwidth-aware layout, packed weights
- `evaluation/`: dataset adapters, metrics, baseline-vs-optimized comparison

## Important constraints

- Build the baseline and the evaluation loop before claiming any optimization gain.
- Do not report accuracy, TTFT, or throughput numbers without a reproducible command and saved output.
- Do not claim PPU compatibility until it is verified on the target environment.

## Missing external assets

See `resources/MANIFEST.md`.
