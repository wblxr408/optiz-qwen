# KV Competition-Targeted Decision Record (2026-08-06)

## Scope and gate

This record covers only the opt-in `qserve_deferred_split_fused_kv` route for
Qwen3.5-2B VLM.  The competition-facing gate is stricter than an isolated
attention-kernel speedup:

1. paired baseline/candidate runs use the same model instance and greedy runner;
2. TTFT and decode throughput measure generation only, not the optional
   evaluation-side logits answer fallback;
3. parsed answers must match the baseline before a speed result can be used;
4. no local CUDA result is a PPU-adaptation claim.

## Options considered

| Option | Performance / accuracy trade-off | Decision |
| --- | --- | --- |
| Quantize and install split `score+PV` for every decode | Simple, but short public VLM requests pay cache conversion and launch overhead; prior short-context paired runs were negative | Rejected as an always-on policy |
| Native dense cache until an actual token threshold, then packed split decode | Avoids short-request overhead but the two-kernel route was negative on a 567-token target question | Rejected for the target preset |
| One-launch INT4 immediately after prefill | Eliminates the split route's score allocation and second launch, but can alter the answer-forming first tokens | Rejected for the target preset |
| One-launch INT8 after prefill | Restores answer parity on the observed failure, but reduces decode throughput | Rejected for the target preset |
| Native answer prefix, then one-launch INT4 packed decode | Leaves short answers untouched and moves only the long answer tail onto the bandwidth-reduced path | Selected as the target long-answer preset |

The selected behavior is dense native execution until both gates are met: the
actual cache must reach the context threshold and at least four answer-prefix
tokens must already have been produced.  The cache is not merely kept dense: it
is not injected into the model at all until both gates are crossed.  This
prevents a custom-cache overhead from being mislabeled as a KV benefit on
ordinary public questions and preserves short multiple-choice answers exactly.

## Implemented mechanism

- `activation_threshold` and `decode_warmup_tokens` are explicit and switchable.
- Before both gates: native `DynamicCache` stays in control.
- After the protected four-token prefix: native KV is adopted once; the next
  decode packs the full-attention layers; fused attention is installed only
  after packing succeeds.
- Both INT4 and INT8 one-launch backends are switchable.  The evaluated target
  preset is INT4 with `threshold=512`, `decode_warmup_tokens=4`.
- Benchmark timing stops at generation completion.  A logits answer fallback
  remains available for accuracy accounting but is excluded from TTFT and
  throughput, preventing parseability from creating a false speedup.

## Evidence

All results below used original `structuralized_imagetext_understanding` public
image-text questions with `max_new_tokens=32`, one warmup per mode, three
alternating repetitions, CUDA RTX 4070 Laptop GPU, and no Triton fallback.

| Sample | Input tokens | Candidate | TTFT change | Decode throughput change | Answer parity | Status |
| --- | ---: | --- | ---: | ---: | --- | --- |
| 1647 | 567 | immediate one-launch INT4 | -0.92% | +6.69% | yes | Insufficient: one question only |
| 1651 | 673 | immediate one-launch INT4 | +2.69% | -9.66% | yes | Rejected: performance negative |
| 1653 | 669 | immediate one-launch INT4 | -1.07% | +0.80% | no (A -> C) | Rejected: answer changed |
| 1653 | 669 | immediate one-launch INT8 | -2.04% | -3.73% | yes | Rejected: performance negative |
| 1647, 1651, 1653 | 567, 673, 669 | prefix-4 one-launch INT4 | see aggregate below | see aggregate below | yes, 9/9 pairs | Selected target long-answer preset |
| 1670 | 527 | prefix-4 one-launch INT4 | n/a | n/a | yes, 3/3 pairs | Protected: no kernel call for two-token answer |

Artifacts:

- `benchmarks/output/kv_adaptive_threshold512_structured_en_1x32_3r_timingfix_20260806.json`
- `benchmarks/output/kv_adaptive_singlekernel_threshold512_structured_en_1x32_3r_20260806.json`
- `benchmarks/output/kv_adaptive_singlekernel_threshold512_structured_1651_en_1x32_3r_20260806.json`
- `benchmarks/output/kv_adaptive_singlekernel_threshold512_structured_1653_en_1x32_3r_20260806.json`
- `benchmarks/output/kv_adaptive_int8_singlekernel_threshold512_structured_1653_en_1x32_3r_20260806.json`
- `benchmarks/output/kv_adaptive_int4_prefix4_threshold512_structured_3x32_3r_aggregate_20260806.json`

For the selected preset, three original long `structuralized_imagetext_understanding`
questions were each run three times with alternating order and one warmup per
mode.  All nine pairs preserved both the parsed answer and generated-token
count; all candidate runs executed the one-launch kernel with zero fallback.

| Aggregated metric | Mean change | Bootstrap 95% interval |
| --- | ---: | ---: |
| TTFT | +2.76% | -0.28% to +6.18% |
| Decode throughput | +19.55% | +7.30% to +31.59% |
| Equal-weight TTFT/throughput speed proxy | +11.16% | +5.62% to +16.60% |

The speed proxy uses equal weights because the competition assigns the same
30% score weight to TTFT and throughput after accuracy retention.  It passes
the predeclared paired-bootstrap positive-gain gate; TTFT alone is near-neutral
and is not claimed as a stable independent gain.

## Result boundary

The validated claim is intentionally narrow: on the checked long-answer,
567--673-token public VLM subset, prefix-4 INT4 packed-KV has a positive
competition-speed proxy while preserving the paired answer and token count.
This does **not** prove a whole-public-set win, a PPU win, or an independent
TTFT gain.  Short answers remain native by design; further work must validate
more long-answer categories and target PPU hardware before widening the claim.
