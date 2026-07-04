# CLAUDE.md

## Mission

Act as an implementation partner for the `Qwen3.5-2B` VLM optimization competition project. This is not a generic assistant workspace. The repository exists to support:

- baseline deployment
- measurable evaluation
- four-layer optimization from the project SVG
- final competition reports and reproducible source code

## What matters most

Optimize for the competition scoring dimensions:

1. accuracy retention
2. time to first token
3. throughput
4. system-level optimization depth

If a proposed change improves one metric while risking another, state the tradeoff explicitly.

## Four-layer implementation frame

Use the following module boundaries:

- `src/optiz_qwen/compression/`
  - AWQ W4A16
  - visual token pruning
  - KV cache quantization
  - VLM-specific PTQ
- `src/optiz_qwen/scheduling/`
  - paged KV
  - prefill/decode split
  - batching and request-state scheduling
- `src/optiz_qwen/kernels/`
  - dequant plus GEMM or GEMV fusion
  - attention path fusion
  - FFN path fusion
  - lightweight op fusion such as RoPE, RMSNorm, and SiLU
- `src/optiz_qwen/ppu/`
  - operator compatibility mapping
  - unsupported-op rewrite or lookup-table fallback
  - weight packing and layout optimization
- `src/optiz_qwen/evaluation/`
  - dataset adapter
  - accuracy metric
  - TTFT and throughput benchmark entrypoints

## Execution order

Default implementation order:

1. baseline path
2. profiling hooks
3. evaluation pipeline
4. AWQ and visual token pruning
5. scheduling optimizations
6. fused kernels
7. deeper PPU adaptation
8. optional branches such as speculative decoding

Do not reorder this without a task-specific reason.

## Missing assets policy

The repository currently does not include the evaluation set, model weights, or all external references. While these are missing:

- create placeholders, manifests, and interfaces only
- do not mock benchmark results
- do not hardcode fake paths that look final
- mark blockers clearly

## Reporting discipline

When making claims, keep them tied to evidence:

- metric claims require saved outputs
- hardware claims require target-environment verification
- compatibility claims require either a successful run or a clearly cited source inside the repo docs

## Non-goals for the current stage

Until the missing assets are downloaded, avoid pretending to finish:

- final benchmark scripts
- final accuracy reports
- final PPU-specific kernels
- final competition paper content

Instead, focus on clean structure that will let those pieces land without rework.
