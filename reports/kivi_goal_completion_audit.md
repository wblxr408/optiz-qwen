# KIVI/Qwen3.5 Goal Completion Audit

Date: 2026-07-08

This audit maps the requested unfinished items to current repository evidence.
It is intentionally conservative: a row is complete only when the current
worktree or command output proves it.

| Requirement | Current status | Evidence |
| --- | --- | --- |
| Upstream KIVI does not directly support Qwen3.5 VLM; keep local adapter and upstream pack/unpack | Implemented as local adapter, not upstream-native support | `src/optiz_qwen/compression/qwen35_kivi_cache.py`; `tests/test_qwen35_kivi_cache.py`; `docs/kivi_kv_cache_module.md` |
| Default DNDX wrapper does not enable KIVI; must be explicit | Implemented with explicit CLI and env paths; baseline remains KIVI-off | `--enable-kivi-kv-cache`; `OPTIZ_QWEN_KIVI_KV_CACHE=1`; `tests/test_dndx_entrypoints.py` |
| Declare PPU compatibility | Not complete | `inspect_ppu_compatibility()` returns `claim="unverified"` and `can_claim_compatible=False`; no official PPU material under `resources/ppu_docs/raw/` |
| Obtain optimized benchmark data | Partially complete | 10-sample baseline/KIVI JSON files and comparison PNG exist under `benchmarks/output/`; this is smoke data, not full optimized benchmark proof |
| Adapter stores KIVI Packed KV and restores dense K/V for Transformers Attention | Implemented | `KiviAttentionLayer` packs via upstream `quant.new_pack` and `materialize()` returns dense tensors |
| Replace adapter with official qB MatMul Kernel | Not complete | `inspect_qb_matmul_kernel()` reports `ModuleNotFoundError("No module named 'kivi_gemv'")`; `nvcc` is not found on PATH; Qwen3.5 Attention path is still native Transformers |
| `triton-windows==3.7.1.post27` allows importing `quant.new_pack` | Verified in `optiz-qwen` Conda env | Command imported `quant.new_pack` and found `triton_quantize_and_pack_along_last_dim` |
| Single-sample CUDA smoke under `OPTIZ_QWEN_KIVI_KV_CACHE=1` runs | Verified | `benchmarks/output/kivi_cuda_cn_1x64_validation_fix.json` |
| Public Validation no longer fails with `missing_choice_answer` | Fixed for the known 10-sample public smoke set | `public_validation.passed=true` and `validation_errors=[]` for both 10-sample baseline and KIVI runs |
| Accuracy Retention is verified | Not complete | 10 public samples have been rerun; full 4029-sample public dev or official validation is still required |

## Current Benchmark Evidence

Small-batch files:

- `benchmarks/output/baseline_cuda_cn_10x64_validation_fix.json`
- `benchmarks/output/kivi_cuda_cn_10x64_validation_fix.json`
- `benchmarks/output/compare_baseline_vs_kivi_cn_10x64_validation_fix.png`

Observed comparison:

| Metric | Baseline | KIVI |
| --- | ---: | ---: |
| sample count | 10 | 10 |
| public validation | passed | passed |
| accuracy | 1.000 | 1.000 |
| avg TTFT ms | 914.752 | 1686.362 |
| throughput tokens/s | 8.753 | 6.108 |

This does not prove performance optimization.  On this smoke set KIVI is
slower than baseline.

## Remaining Hard Blockers

- PPU compatibility requires target hardware or official compatibility
  materials.
- qB MatMul replacement requires a working upstream `kivi_gemv` CUDA extension
  and then a Qwen3.5 VLM Attention integration.
- Accuracy Retention requires a full public dev run or official validation set
  run, not a single-sample smoke test.
