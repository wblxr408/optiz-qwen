# Retained Deferred Packed-KV Decision

## Scope

The only retained KV experiment is `qserve_deferred_split_fused_kv`. It is
opt-in and the default baseline continues to use the native cache path.

## Decision record

| Option | Performance evidence | Maintenance and compatibility | Decision |
| --- | --- | --- | --- |
| Legacy KIVI adapter | Slower than baseline and no Qwen3.5 upstream model implementation | External checkout and compatibility surface | Removed |
| Ordinary and early fused QServe chains | No stable short-context win | Multiple public routes for the same unsuccessful hypothesis | Removed |
| Deferred split packed-KV | Real Triton execution, zero fallback, numerically controlled | One isolated opt-in route; prefill remains native | Retained for long-context and PPU study |

The retained chain keeps prefill KV dense, adopts the native cache after the
first token, then performs decode through an INT4 split packed-KV attention
path. This avoids putting prefill quantization on the TTFT path.

## Verified boundary

The split kernel matches the eager SDPA result using the same quantized cache
with about `2.44e-4` maximum absolute error. It has executed on the local RTX
4070 with zero fallbacks.

The paired public-English smoke used one model instance, two warmups per sample
and mode, and three alternating repetitions across ten samples:

| Metric | Baseline median | Retained-chain median | Change |
| --- | ---: | ---: | ---: |
| TTFT | 285.683 ms | 291.971 ms | +2.20% (worse) |
| Decode throughput | 23.089 tok/s | 23.209 tok/s | +0.52% |
| Answers matching baseline | - | 30/30 | maintained |
| Triton fallback | - | 0/30 | none |

Artifact: `benchmarks/output/qserve_deferred_split_decodeonly_paired_en_10x64_3r_20260727.json`.

This is not a short-context performance win and must not be enabled as the
default or presented as a PPU adaptation. Synthetic attention becomes useful
around 1024 or more tokens on the local device, which is why the chain remains
available for a long-context measurement and official PPU-operator comparison.
