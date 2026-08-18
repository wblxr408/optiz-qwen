# PPU-Adapted Optimization Design (Qwen3.5-2B VLM)

Status: design accepted for implementation, grounded in measurements taken on the
competition PPU-ZW810E server (`8.130.213.80`, `PPU_SDK` sourced) against the real
4.5 GB checkpoint at `/mnt/nas/optiz-qwen/models/Qwen3.5-2B` and the real
`mmbench_dev_en.tsv`.

All numbers below are measured. Nothing here is projected.

---

## 1. The finding that reframes the plan

The PPU is **not** memory-bandwidth-bound during decode. It is **dispatch-bound**.

| Quantity | Measured | Source |
|---|---|---|
| Baseline decode step | 19.3 – 21.1 ms (47 – 51 tok/s) | eager + `DynamicCache` |
| HBM floor for 4.426 GB weights @ 2011 GB/s | 2.20 ms (454 tok/s) | roofline |
| Overhead factor | **8.9×** | derived |
| CUDA kernels launched per decode step | **5778** | profiler |
| CPU launch time for 20 queued steps, no sync | 20.656 ms/step | vs 20.678 ms wall |
| Per-kernel launch cost | ~4.5 µs (torch), ~10.5 µs (Triton) | microbench |

The CPU cannot issue work fast enough to keep the device busy. 5778 launches × 4.5 µs
≈ 26 ms of pure issue cost per token — which is the entire step. The device is idle
most of the time.

**Consequence:** every optimization whose mechanism is "move fewer bytes" or "do fewer
FLOPs" pays almost nothing on this hardware until the launch cost is removed first.
That is why the earlier INT4 KV chain, which won +19.55% throughput on a local RTX 4070,
**regressed −2.38% on PPU** — it added Triton launches (10.5 µs each) to a pipeline whose
bottleneck was launches.

This single fact sets the execution order: **kill dispatch overhead first, then revisit
compression as an accuracy/memory play, not a speed play.**

---

## 2. Chain design

Four stages, each independently switchable, ordered by measured payoff.

```
                     ┌──────────────── prefill (TTFT path) ─────────────────┐
  image + prompt ──▶ │ vision tower ──▶ 18× GDN + 6× full-attn ──▶ lm_head │──▶ token 0
                     │        attn_impl = sdpa      fla-core Triton GDN     │
                     └──────────────────────────────────────────────────────┘
                                              │  StaticCache (fixed addresses)
                                              ▼
                     ┌──────────────── decode (throughput path) ────────────┐
                     │  ONE captured CUDA graph, replayed per token         │
                     │  attn_impl = flash_attention_2 (frozen at capture)   │
                     │  explicit position_ids (3,1,1) + cache_position (1,) │
                     └──────────────────────────────────────────────────────┘
```

### Stage 1 — CUDA-Graph decode over `StaticCache` (the primary lever)

Replaces 5778 per-step launches with one graph replay. Two preconditions had to be
discovered on the PPU:

1. **Explicit `position_ids` and `cache_position`.** Letting the model compute them
   internally raises `CUDA error: operation not permitted when stream is capturing`
   (`hggcErrorStreamCaptureUnsupported`) at `modeling_qwen3_5.py:1517`, inside
   `compute_3d_position_ids` — `position_ids.view(1,1,-1).expand(3,B,-1).to(device)`
   is a host-to-device copy, illegal during capture. Passing a pre-allocated
   `(3,1,1)` M-RoPE tensor and a `(1,)` cache position removes the copy.
2. **`StaticCache`, not `DynamicCache`.** `DynamicCache` concatenates into a fresh
   allocation every step, so the addresses baked into the graph go stale; replay
   produced degenerate output ("The The The The…") with `seq_length` frozen at 219.
   `StaticCache` reports `is_compileable: True` and keeps fixed addresses. Replay then
   matches eager greedy exactly.

### Stage 2 — `flash_attention_2` for the 6 full-attention layers

Inside the 8.9 ms graph step, the single largest kernel was
`fmha_cutlassF_bf16_aligned_k256x256x32_sm80` — **2.6453 ms for only 6 calls**. That is
sdpa's `EFFICIENT` backend serving GQA 8Q:2KV at head_dim 256. Microbench at that exact
geometry, kv_len 1024:

| kernel | time | vs HBM floor (1.0 µs) |
|---|---|---|
| sdpa `EFFICIENT` (cutlassF) | 392.9 µs | 393× |
| sdpa `FLASH` / `enable_gqa` | 23.6 µs | 24× |
| `flash_attn_with_kvcache` | 25.3 µs | 25× |

Switching `attn_implementation` to `flash_attention_2` cut the graph step from
**8.91 ms → 6.24 ms** (112.2 → 160.3 tok/s), token-identical.

### Stage 3 — split attention backend across prefill and decode (the TTFT fix)

FA2 is faster at decode but **slower at prefill** on this device (measured on the same
8 MMBench prompts, 5 reps each, p50):

| impl | prefill avg |
|---|---|
| sdpa | 50.68 ms |
| flash_attention_2 | 56.41 ms |

Naively enabling FA2 globally therefore *cost* TTFT: end-to-end TTFT went
53.41 → 63.37 ms, **−18.64%**, wiping out most of the score value of the throughput win.

The fix exploits how transformers dispatches attention. `modeling_qwen3_5.py:702`
resolves the kernel per forward call:

```python
attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
    self.config._attn_implementation, eager_attention_forward)
```

A CUDA graph freezes whatever kernels were live at capture time. So: **capture the decode
graph while the config says `flash_attention_2`, then set the config back to `sdpa`.**
Decode replays frozen FA2 kernels; prefill dispatches sdpa fresh. Both wins, no tradeoff.

Verified: hybrid decode 6.32 ms (158.2 tok/s), hybrid prefill 50.68 ms, token parity 8/8
against pure-sdpa eager.

### Stage 4 — `fla-core` Triton GDN for prefill

TTFT is dominated by the 18 GDN layers, **not** the vision tower — contradicting the
vision-pruning-first assumption in `docs/README.md`:

| component | 448 px (240 tok) | share | 896 px (828 tok) share |
|---|---|---|---|
| GDN (18 layers) | 87.4 ms | 78.4% | — |
| vision tower | 13.7 ms | 12.2% | 23.8% |
| full attention (6) | 5.9 ms | 5.3% | — |
| lm_head | 2.1 ms | 1.9% | — |
| **total prefill** | 111.56 ms | | |

Installing pure-Python `fla-core==0.5.2` (supplying `chunk_gated_delta_rule` and
`fused_recurrent_gated_delta_rule`) **halved prefill: 108.64 → 54.25 ms**, byte-identical
greedy output. This is the whole reason baseline TTFT reads ~53 ms rather than ~110 ms in
the tables below — it is already in the measured baseline and must be declared as such.

`causal_conv1d` remains unavailable (source-only wheel). `is_fast_path_available` stays
False, so a further GDN win is still on the table but unmeasured.

**Update — `causal_conv1d` is now built and installed.** The source wheel compiles
against the PPU SDK (nvcc 13.0): `causal_conv1d-1.6.2.post1-cp312-cp312-linux_x86_64.whl`,
~13 min build. `is_fast_path_available` is now **True** — all four of
`causal_conv1d_fn`, `causal_conv1d_update`, `chunk_gated_delta_rule`,
`fused_recurrent_gated_delta_rule` resolve. The kernel itself is correct on this device
(max abs err 0.0312 vs a `F.conv1d` + SiLU reference at bf16, dim 4096, L 340, width 4 —
in line with bf16 rounding).

It buys almost nothing on prefill, and that is the point worth recording:

| | without `causal_conv1d` | with it |
|---|---|---|
| full prefill forward, 340-token prompt | 54.926 ms | 53.577 ms |
| language stack | 52.866 ms | 50.221 ms |
| device ops (distinct / calls) | 162 / 4772 | 161 / 4700 |
| `aten::copy_` | 572 calls, 2.837 ms | 554 calls, 2.136 ms |

**72 fewer device calls out of 4700, and the verdict does not move.** Full artifacts:
`benchmarks/output/ppu_prefill_profile.json` and `..._cc1d.json`.

> **Correction (stage 6 below).** The "4700 calls" in this table is a *double count*.
> `prof.key_averages()` returns both CPU operator rows (`aten::mm`) and device kernel rows
> (`gemm_ktype0_...`), and operator rows attribute their children's device time and calls
> into themselves. Summing all rows counts the same work twice. The real figure is **2423
> kernel launches** across **13698 operator calls**
> (`scripts/profile_ppu_prefill_headroom.py`). The relative claim is unaffected — 72 of
> 2423 is still negligible, and `causal_conv1d` is still not a speedup — but the absolute
> number must not be quoted.

### Stage 5 — prefill is dispatch-bound too, and the lm_head is the cheap part of it

Two measurements on the target, `scripts/profile_ppu_prefill.py`:

| quantity | measured |
|---|---|
| `cpu_issue_fraction` (CPU issue time ÷ wall time, 5 queued forwards, no sync) | **0.9882 – 0.9893** |
| verdict | **dispatch-bound**, on every sample |
| device ops in one prefill | 161 distinct, ~~4700 calls~~ → **2423 kernel launches** (see stage 6) |
| top-12 device ops, summed | ~~42.5 ms of a ~53 ms wall~~ → double-counted; real device time is **27.3 ms** |

So prefill has the same disease as decode did: the CPU spends ~99% of the wall clock
issuing work. This is why `causal_conv1d` (Stage 4 update) changes nothing — it removes
device work from a path that is not waiting on the device. It also predicts that the
remaining big TTFT lever is a captured/compiled prefill, not a faster kernel.

**Both struck-through figures were wrong in the same way** and stage 6 replaces them. They
came from summing all `key_averages()` rows, which counts CPU operator rows and the device
kernel rows they own as if they were separate work. The 42.5-of-53 reading implied the
device was ~80% busy, which flatly contradicted the dispatch-bound verdict on the line
above it — that contradiction was the signal, and it was resolved by measurement rather
than by picking whichever number suited the story.

The one exception found so far is pure waste rather than dispatch: the lm_head projects
**every** prompt position when greedy prefill reads only `logits[:, -1, :]`.

| | measured (340-token prompt, vocab 248320) |
|---|---|
| `lm_head(hidden)` | 3.017 – 3.040 ms |
| `lm_head(hidden[:, -1:, :])` | 0.472 – 0.482 ms |
| waste | **2.544 – 2.558 ms** |

`modeling_qwen3_5.py:1629` honors `logits_to_keep` (`slice_indices = slice(-logits_to_keep,
None)`), so passing `logits_to_keep=1` narrows the projection. The downstream slice is
unchanged because `logits[:, -1, :]` addresses the same row either way — first token is
identical by construction, not by luck.

Landed in `scheduling/prefill_decode.py`, on by default, with
`OPTIZ_QWEN_PREFILL_LAST_LOGIT_ONLY=0` as the kill switch. Support is probed by
`inspect.signature` rather than try/except so a failure can never land inside the timed
prefill region and corrupt a TTFT number.

### Stage 6 — the captured-prefill lever, measured and closed

Section 7 previously listed "captured or compiled prefill" as *not attempted*. It has now
been attempted from four directions. The summary is that the lever is real but small, and
both routes to it are blocked or negative, so **prefill TTFT work stops here**.

**6a. The ceiling is ~2×, not decode's 8.9×.** `scripts/profile_ppu_prefill_headroom.py`
separates device kernel rows from CPU operator rows, so the recoverable time is measured
rather than inferred:

| prompt | wall | kernel device time | device_busy_fraction | kernel launches | idle | max speedup if dispatch-free |
|---|---|---|---|---|---|---|
| 340 | 53.357 ms | 27.344 ms | 0.5125 | 2423 | 26.013 ms | **1.951×** |
| 360 | 58.748 ms | 27.542 ms | 0.4688 | 2423 | 31.207 ms | **2.133×** |
| 337 | 58.237 ms | 27.336 ms | 0.4694 | 2423 | 30.901 ms | **2.130×** |

`operator_attributed_device_ms` 27.346 ≈ `kernel_device_ms` 27.344 — the same work seen
twice, which is the double count named above; `naive_all_rows_device_ms` is 54.67–55.09.
A perfectly captured prefill lands at ~27 ms, not near zero. Artifact:
`benchmarks/output/ppu_prefill_headroom.json`.

**6b. `cpu_issue_fraction` 0.99 had two possible causes, and it is the expensive one.**
Queueing forwards without syncing cannot distinguish *launch-bound* (CPU cannot issue fast
enough → capture fixes it) from *sync-stalled* (a `.item()`/`.tolist()` blocks the CPU so
it can never run ahead → keeping the value on the host fixes it). Both read ≈0.99.
`scripts/probe_ppu_prefill_syncs.py` resolves it with
`torch.cuda.set_sync_debug_mode("warn")` plus per-line attribution: **93–94 syncs per
prefill over 19–20 distinct sites, 72 of them at one line**,
`modeling_qwen3_5.py:968` in `Qwen3_5VisionAttention.forward`:

```python
lengths = cu_seqlens[1:] - cu_seqlens[:-1]
splits = [torch.split(t, lengths.tolist(), dim=2) for t in (q, k, v)]
```

Three D2H copies per block × 24 blocks. Every one computes the *same* list, because
`cu_seqlens` is derived from `grid_thw` once per forward and the identical tensor object is
handed to every block. Artifact: `benchmarks/output/ppu_prefill_syncs.json`.

**6c. Removing 77% of the syncs is bit-exact and worth ~1%.**
`kernels/vision_prefill_sync.py` derives the chunk lengths on the host from one `grid_thw`
read (upstream builds them as `repeat_interleave(h*w, t).cumsum()`, so this is closed-form,
not a guess) and serves them to every block:

| prompt | syncs | prefill p50 | gain | logits |
|---|---|---|---|---|
| 340 | 94 → 22 (−72) | 62.053 → 61.178 ms | +1.410% | identical, Δ 0.0 |
| 360 | 93 → 22 (−71) | 61.717 → 61.095 ms | +1.008% | identical, Δ 0.0 |
| 337 | 93 → 22 (−71) | 61.605 → 60.960 ms | +1.047% | identical, Δ 0.0 |

`cpu_issue_fraction` stayed ≈0.99 afterwards. **So prefill is launch-bound, not
sync-stalled**: the CPU cost is framework overhead spread thin across 13698 operator calls,
not a handful of stalls, and only wholesale capture or compilation can touch it. Artifact:
`benchmarks/output/ppu_vision_sync_elision.json`. See D7.

**6d. Neither half of prefill has capturable shapes.** Capture needs fixed shapes. The
language stack cannot have them — 46 distinct prompt lengths over 50 samples (137–363;
64-token buckets `{192:12, 256:17, 320:14, 384:7}`, mean pad waste 32.56 tokens). The
vision tower's shapes come from `image_grid_thw` rather than the prompt, so it was worth
checking separately — `scripts/probe_vision_grid_shapes.py`, processor only, no model load:
**24 distinct `image_grid_thw` and 18 distinct `pixel_values` shapes over 50 samples**,
`vision_shapes_fixed: False`. Top grids `[1,16,24]`×9, `[1,14,20]`×7, `[1,16,20]`×4,
`[1,16,16]`×3, then a long tail of 1–2 each. Artifact:
`benchmarks/output/ppu_vision_grid_shapes.json`.

**6e. `torch.compile(dynamic=True)` makes prefill slower, and is not bit-exact.** This is
the one lever that attacks framework overhead without fixing shapes, and D1's rejection of
`torch.compile` was specific to Inductor's *cudagraph trees* under `mode="reduce-overhead"`,
so it did not transfer. Measured per submodule, `scripts/probe_prefill_compile.py`:

| arm | compile time | prompt 340 | prompt 360 | prompt 337 |
|---|---|---|---|---|
| eager (language run) | — | 63.048 ms | 62.967 ms | 63.039 ms |
| `compile(language_model, dynamic=True)` | 53.47 s | 72.073 ms **−14.31%** | 71.140 ms **−12.98%** | 71.051 ms **−12.71%** |
| eager (vision run) | — | 66.058 ms | 65.933 ms | 66.036 ms |
| `compile(visual, dynamic=True)` | 24.30 s | 74.642 ms **−13.00%** | 74.226 ms **−12.58%** | 74.004 ms **−12.07%** |

Both arms are net *negative*, and dynamo says why: the language arm hit
`config.recompile_limit (8)` with `last reason: tensor 'kwargs['attention_mask']' dtype
mismatch. expected Long, actual Bool`, and the vision arm graph-broke inside
`Qwen3_5VisionAttention.forward:989` — the same data-dependent split as 6b. Guard
evaluation and repeated recompilation cost more than the fusion saves. Numerics also moved:
`max_abs_logit_delta` 0.328–0.375 (language) and 0.328–0.500 (vision), with the greedy
first token flipping on 1 of 3 samples in each arm. Artifacts:
`benchmarks/output/ppu_prefill_compile.json`, `..._compile_vision.json`. See D8.

---

## 3. Measured end-to-end result

**Primary evidence** — 50 MMBench dev-en samples through the shipped entrypoint
`optiz_qwen.evaluation.dndx_public_benchmark`, `max_new_tokens=256`,
`max_cache_len=2048`, greedy, batch 1, one process per arm, `fla-core==0.5.2` and
`causal_conv1d==1.6.2.post1` installed. Artifact:
`benchmarks/output/ppu_hybrid_trim_ab_50samples.json`, pinned by
`tests/test_ppu_hybrid_trim_regression.py`.

Three arms, so the two TTFT levers are separable:

| | A baseline | B hybrid | C hybrid + prefill trim |
|---|---|---|---|
| accuracy | 0.76 (38/50) | 0.76 (38/50) | **0.76 (38/50)** |
| TTFT (mean) | 58.436 ms | 55.378 ms | **52.556 ms** |
| TTFT (p50) | 57.615 ms | 54.816 ms | **51.870 ms** |
| throughput | 44.439 tok/s | 162.937 tok/s | **162.312 tok/s** |
| TTFT vs A | — | +5.23% | **+10.06%** |
| throughput vs A | — | +266.65% | **+265.25%** |
| answer parity vs A | — | 50/50 | 50/50 |
| token parity vs A | — | 49/50 | 49/50 |

Reading: the graph is worth **+5.23%** TTFT, the prefill trim another **+5.10%** on top
of it (C vs B), and they compose. C wins TTFT on 47/50 samples against A and 42/50
against B, and throughput on 50/50. The trim costs no throughput (−0.38%, inside noise)
because it touches prefill only.

**Earlier probe-script evidence** (32 samples, ad-hoc script, kept for the divergence
root-cause below). Artifact: `benchmarks/output/ppu_cudagraph_hybrid_ab_32samples.json`.

| | baseline (sdpa eager + DynamicCache) | hybrid | change |
|---|---|---|---|
| accuracy | 0.75 (24/32) | 0.75 (24/32) | **0.00** |
| TTFT | 54.384 ms | 52.387 ms | **+3.67%** |
| throughput | 46.743 tok/s | 156.864 tok/s | **+235.59%** |
| answer parity | — | 32/32 | — |
| token parity | — | 31/32 | — |

The 0.75-vs-0.76 accuracy difference between the two tables is sample size (32 vs 50),
not a regression: answer parity is total within each A/B.

Decode cost is flat across the generation — quartile means over 256 steps:
`[6.33, 6.39, 6.33, 6.32]` ms. No drift as kv_len grows, so the graph holds for the full
official generation length.

### The one token-parity miss, explained

Sample 370 diverged at token 54 of 56. Root-caused by a 5-config ladder on one prompt
(`probe_divergence.py`):

| comparison | result |
|---|---|
| sdpa eager+dynamic **vs** sdpa eager+static | first div idx 30 |
| sdpa eager+static **vs** sdpa graph+static | **IDENTICAL** |
| sdpa graph+static **vs** FA2 graph+static | first div idx 52 |
| FA2 graph+static **vs** FA2 eager+dynamic | first div idx 52 |

Reading: **the CUDA graph itself is bit-faithful** (static-eager ≡ static-graph). The
divergences are pure kernel numerics — cache layout (dynamic vs static) and attention
backend (sdpa vs FA2) — and they only bite where greedy decoding is at a coin-flip:
the top1−top2 logit gap at the divergence step was **0.0000**, and 8 of 255 steps had a
gap < 0.05 against a median of 1.75. Divergence at an exact tie is expected and carries
no accuracy signal; accuracy and answer parity are unchanged (0.75 / 32-of-32).

---

## 4. Decisions, each against at least one rejected alternative

Per `AGENTS.md`, every decision below names what it beat and why.

**D1 — decode dispatch: manual CUDA-Graph capture**, not `torch.compile(mode="reduce-overhead")`.
Inductor's cudagraph trees raised `InternalTorchDynamoError: accessing tensor output of
CUDAGraphs that has been overwritten by a subsequent run` at `modeling_qwen3_5.py:796`.
Manual capture works and is inspectable. Rejected also: keeping eager and shaving kernel
count by hand — 5778 launches cannot be meaningfully reduced by op fusion alone.

**D2 — KV cache: `StaticCache`**, not `DynamicCache`. `DynamicCache` is
`is_compileable: False` and reallocates per step, which corrupts graph replay
(degenerate output, frozen `seq_length`). Cost accepted: a fixed `max_cache_len=2048`
buffer, and prefill over that buffer measures the same as dynamic (50.66 vs 51.29 ms with
sdpa), so the padding is free.

**D3 — attention backend: split sdpa-prefill / FA2-decode**, not one impl everywhere.
Global FA2 costs −18.64% TTFT; global sdpa costs 30% of decode throughput
(8.91 vs 6.24 ms). The split captures both. Rejected: `attn_implementation="eager"`,
which **cannot be CUDA-graph captured at all** on PPU (capture invalidated) — a hard
constraint, not a harness bug.

**D4 — weight-only quantization demoted from the speed plan.** This reverses the
AWQ-W4A16-as-speedup premise in `docs/README.md`, on evidence:

| path | result |
|---|---|
| `torch._int_mm` | 0.17 TOPS (149.7 ms for 1024×2048×6144) |
| torchao `int8wo` on 1×2048×6144 | **0.23×** of bf16 |
| torchao `int4wo` on same | **0.25×** of bf16 |
| bf16 / fp16 dense | ~123 TFLOP/s each |
| decode GEMV achieved bandwidth | 1521 – 2073 GB/s vs 2011 GB/s roofline |

Decode GEMV is already at the roofline and INT paths are 4× *slower*. W4A16 survives only
as a memory-footprint or accuracy-budget play, never billed as throughput. Keep the
existing GPTQ pipeline; stop claiming speed for it on PPU.

**D5 — GDN prefill: `fla-core==0.5.2` Triton path**, not the torch fallback. 108.64 →
54.25 ms, byte-identical output.

**D5b — `causal_conv1d` built from source and kept, but not billed as a speedup.** The
wheel now compiles on the target and `is_fast_path_available` is True, which closes an
open item and removes the transformers fast-path warning. It is worth **72 of 4700 device
calls** and no verdict change (54.926 → 53.577 ms full forward, inside run-to-run spread
on a dispatch-bound path). Rejected framing: reporting it as the GDN prefill win it was
expected to be. Kept because it is a correctness/parity prerequisite for any future GDN
work, and because its null result is itself the evidence that prefill is dispatch-bound.

**D5c — prefill lm_head trimmed to the last position** (`logits_to_keep=1`), not
full-sequence logits. Greedy prefill reads one row; the model projected all ~340 at vocab
248320, costing 2.55 ms. Measured **+5.10% TTFT on top of the graph** across 50 samples,
with 50/50 answer parity. Rejected: slicing hidden states before a manual `lm_head` call
in the wrapper — that would duplicate model-internal logic and break for any model whose
head is not a plain `nn.Linear`, where the supported kwarg does not. On by default with
`OPTIZ_QWEN_PREFILL_LAST_LOGIT_ONLY=0` as the kill switch; support probed by signature so
an unsupported model degrades silently instead of raising inside the timed region.

**D6 — visual token pruning deferred.** The vision tower is 12.2% of prefill at 448 px
and 23.8% at 896 px. Pruning it trades OCR/fine-grained-localization accuracy for a
minority slice of a path that is already 78% GDN. Revisit only after GDN prefill is
exhausted.

**D7 — vision host-sync elision kept behind a default-off switch, not billed as a TTFT
lever.** `kernels/vision_prefill_sync.py` removes 72 of 93 prefill host syncs by deriving
the per-block chunk lengths on the host from `grid_thw` (closed-form, matching upstream's
`repeat_interleave(h*w, t).cumsum()`), and is bit-exact: `max_abs_logit_delta` 0.0 on all
three samples. It is worth **~1.0–1.4%** of prefill wall clock. Kept because it is free and
correct, classified the way `causal_conv1d` was (D5b): a real small win whose *null-ness*
is the evidence — it is what proves prefill is launch-bound rather than sync-stalled.
Default off via `OPTIZ_QWEN_VISION_SYNC_ELISION` because a ~1% gain does not justify
monkeypatching `torch.Tensor.tolist` in the default path. Rejected: patching upstream
`modeling_qwen3_5.py` (unowned file, lost on any transformers upgrade); overriding `tolist`
for the whole vision forward rather than each attention forward — `spatial_merge_size` is
also a 1-D int tensor of the same length for single-image batches, so a wider scope could
serve the wrong list.

**D8 — `torch.compile` rejected for prefill too, now on prefill-specific evidence.** D1
rejected it for decode over cudagraph-tree failures; that reasoning did not cover
`dynamic=True` with cudagraphs off, which is the only configuration that could attack
framework overhead without fixed shapes. Measured: **−12% to −14% TTFT** on both the
language stack and the vision tower, because the language arm exhausts
`recompile_limit (8)` on an `attention_mask` dtype guard (Long vs Bool) and the vision arm
graph-breaks on the data-dependent split at `modeling_qwen3_5.py:989`. It is also not
numerics-neutral (`max_abs_logit_delta` up to 0.5, greedy first token flipped on 1 of 3
samples per arm), so it would need an accuracy re-run even if it were fast. Rejected on
both counts. Rejected alternative: raising `recompile_limit` and accepting the guard churn
— that trades a fixed 24–53 s compile plus per-shape recompiles for at best the ~2× ceiling
of 6a, on a 50-sample set with 46 distinct prompt lengths where almost every sample is a
new shape.

---

## 5. Implementation plan against the repo's four layers

Everything lands behind `OPTIZ_QWEN_*` env toggles so the baseline arm stays byte-identical,
matching the existing contextmanager pattern in
`src/optiz_qwen/evaluation/dndx_public_benchmark.py`.

| module | change | new toggle |
|---|---|---|
| `scheduling/cuda_graph_decode.py` (new) | capture/replay manager: static input tensors, side-stream warmup, `StaticCache` ownership, prompt-length bucketing | `OPTIZ_QWEN_CUDA_GRAPH_DECODE`, `OPTIZ_QWEN_CUDA_GRAPH_MAX_CACHE_LEN` |
| `scheduling/prefill_decode.py` | `run_greedy_prefill_decode` gains a graph-replay decode branch; explicit `position_ids`/`cache_position` instead of `_build_decode_kwargs` reflection | reuses `--generation-runner greedy` |
| `kernels/attention_backend.py` (new) | `set_attn_impl()` + capture-under-FA2 / run-under-sdpa context manager | `OPTIZ_QWEN_ATTN_PREFILL`, `OPTIZ_QWEN_ATTN_DECODE` |
| `ppu/compatibility.py` | flip `checked_on_target_hardware` to a real verified record: device caps, kernel availability, what captured and what did not | — |
| `evaluation/dndx_public_benchmark.py` | wire the two new contextmanagers alongside `kv_chain_cli_environment` / `runner_cli_environment` | CLI flags mirroring the above |
| `scheduling/prefill_decode.py` | last-position-only prefill lm_head (`logits_to_keep=1`), signature-probed | `OPTIZ_QWEN_PREFILL_LAST_LOGIT_ONLY` (on by default) |
| `compression/` | unchanged; GPTQ W4A16 reclassified as memory/accuracy, not speed | existing |

`ppu/compatibility.py` currently hardcodes `can_claim_compatible=False,
checked_on_target_hardware=False`. That is now factually stale — the device has been
exercised directly — and updating it is part of this work, with the verified-kernel list
as its evidence.

---

## 6. Environment of record

torch 2.9.0 · triton 3.5.0 (works; `tl.dot` bit-exact, 103.6 TFLOP/s, 10.5 µs launch) ·
transformers 5.14.1 · flash-attn 2.7.4.post1 (+ flash-attn-3 3.0.0b1) · flashinfer 0.6.4 ·
torchao 0.11.0 · vllm 0.17.1+cu130 (sees the device as `NvmlCudaPlatform`) · nvcc CUDA 13.0 ·
`fla-core` 0.5.2 · `causal_conv1d` 1.6.2.post1 (built from source on the target).
Device: PPU-ZW810E, PPU runtime reporting an sm_80-compatible execution target,
64 SMs, 256 KB shared/SM, 64 MB L2, 97920 MiB HBM, measured 2011 GB/s.

Model geometry: 24 layers = 18 `linear_attention` (GatedDeltaNet) + 6 `full_attention`
(`full_attention_interval: 4`), GQA 8Q:2KV, head_dim 256, hidden 2048, intermediate 6144,
vocab 248320, tied embeddings, interleaved M-RoPE `mrope_section [11,11,10]`,
`partial_rotary_factor 0.25`, MTP head present in the checkpoint (unused).

---

## 7. Still unverified — do not claim these

- **Multi-bucket prompt lengths.** One graph was captured and replayed across prompts of
  137–363 tokens successfully, but bucketing policy for the full public set is untested.
- **Full public-set accuracy.** 0.76 on 50 samples is a parity check, not a leaderboard
  number.
- **`max_cache_len` sizing** beyond 2048, and behaviour when prompt + 256 exceeds it.
- **MTP head / speculative decode.** Present in the checkpoint, never exercised.
- **PPU-native kernels.** Nothing here is a PPU-specific kernel; every win comes from
  dispatch elimination, backend selection, and removing redundant lm_head work. Deeper
  PPU adaptation (stage 7 of the execution order in `CLAUDE.md`) is untouched.

Closed since the first draft: the **`causal_conv1d` build** (see D5b — it builds, it is
numerically correct on the device, and it does not move TTFT), and **captured or compiled
prefill** (see stage 6 and D7/D8 — the ceiling is ~2×, capture is blocked by 46 prompt
lengths × 24 vision grids, `torch.compile(dynamic=True)` measures −12% to −14%, and the
cheap sync-elision shortcut is bit-exact but worth ~1%). Prefill TTFT is therefore
considered exhausted at this level; further gains need either shape bucketing with a
padded-capture policy (untested, and the accuracy cost of padding is unmeasured) or stage-7
native kernels, which reduce device time — the half of the wall clock that is *not* the
bottleneck.
