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

### Status of stage 7 (PPU native kernels): NOT STARTED

Stage 7 is untouched. No PPU-native operator has been written. Every measured
gain to date comes from three things only:

- **dispatch elimination** — one CUDA graph captured and replayed per decode
  token, replacing ~5778 per-token kernel launches
- **attention backend selection** — `sdpa` for prefill, `flash_attention_2`
  frozen into the captured decode graph
- **removing redundant prefill work** — the lm_head projected all ~340 prompt
  positions at vocab 248320 when greedy reads one; `logits_to_keep=1` is worth
  +5.10% TTFT on top of the graph

Both execute on PPU hardware through the PPU-provided CUDA-compatible runtime/API;
they do not use PPU-native custom kernels. When describing results, say
"validated on PPU hardware" (true) and not "PPU-adapted kernels" (false). The
measured bottleneck on this hardware is CPU kernel dispatch, not
HBM bandwidth — **for prefill as well as decode** (`cpu_issue_fraction` 0.988+
on both). This is why `causal_conv1d`, once built, changed nothing. See
`docs/README.md` and `docs/ppu_optimization_design.md`.

### Prefill TTFT is measured out; do not re-propose it

Prefill dispatch elimination was the standing "next lever". It has been measured
from four directions (design doc section 2 stage 6, decisions D7/D8) and the
lever is bounded and blocked:

- `device_busy_fraction` **0.47–0.51** → ceiling **~1.95–2.13×**, not decode's 8.9×
- **2423** kernel launches / 13698 operator calls per prefill. The **4700** figure
  in older text double-counted CPU operator rows against device kernel rows; do
  not reuse it
- CUDA-graph capture needs fixed shapes and prefill has none: **46** distinct
  prompt lengths, **24** vision grids, **18** pixel shapes over 50 samples
- `torch.compile(dynamic=True)` measures **−12% to −14%** on both the language
  stack and the vision tower, and is not bit-exact
- the cheap route (eliding 72 of 93 host syncs) is bit-exact but worth **~1%**,
  kept default-off behind `OPTIZ_QWEN_VISION_SYNC_ELISION`

Remaining prefill routes both have real costs to weigh first: shape bucketing
with padded capture (padding's accuracy cost is unmeasured), or stage-7 native
kernels — which cut device time, the half of the wall clock that is not the
bottleneck.

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
