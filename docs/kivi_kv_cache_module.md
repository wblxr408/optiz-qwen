# KIVI KV Cache Module

## Scope

This is a standalone enhancement module for KV Cache quantization using the upstream `jy-yuan/KIVI` implementation from:

`https://github.com/jy-yuan/KIVI`

It belongs to the model-compression layer in `docs/vlm_optimization_full_stack_architecture.svg`. It is not enabled in the default baseline path.

## Decision Log

Candidate options:

- Copy upstream KIVI model and quantization files into `src/optiz_qwen/`.
- Keep upstream KIVI as an external source checkout and expose a local adapter.
- Reimplement the KIVI algorithm locally from the paper.

Evaluation dimensions:

- Correctness: external checkout preserves upstream behavior better than a local rewrite.
- Maintenance: external checkout is easier to refresh to new upstream Qwen or GQA support.
- Compatibility: full vendoring brings CUDA/Triton and model-family assumptions into the competition package.
- License: upstream root license is MIT, but some files include additional permissive notices; external checkout keeps provenance clear.
- Competition fit: a switchable adapter avoids claiming benchmark gains before baseline comparison.

Final choice:

Use an external upstream checkout under `artifacts/third_party/KIVI` plus the local adapter `src/optiz_qwen/compression/kivi_external.py`.

Rejected option notes:

- Full copying is weaker because it vendors a Llama/Mistral-specific research repo into the Qwen3.5 competition code and makes later upstream refresh harder.
- Reimplementation is weaker because the user explicitly asked to use available upstream code directly.

## Current State

Implemented:

- Upstream source inspection and commit/license reporting.
- Direct loader for upstream `LlamaForCausalLM_KIVI` and `MistralForCausalLM_KIVI`.
- Direct loader for upstream `quant.new_pack` pack/unpack functions.
- Qwen3.5 VLM `DynamicCache` adapter that replaces only `full_attention` layers with KIVI-packed cache layers.
- DNDX wrapper switch via `OPTIZ_QWEN_KIVI_KV_CACHE=1`.
- Config preset in `configs/models/kivi_kv_cache.json`.
- Reproducible checkout helper in `scripts/prepare_kivi_upstream.ps1`.
- Unit tests for source inspection, config attachment, and Qwen unsupported reporting.

Not completed:

- Qwen3.5 VLM is not directly supported by upstream KIVI at the checked commit, so this repo adapts Qwen3.5 through a local Cache class while still using upstream KIVI pack/unpack code.
- The default DNDX benchmark wrapper does not enable KIVI; it must be explicitly enabled by environment variable.
- No PPU compatibility claim has been made.
- No optimized benchmark number has been produced.
- The current adapter stores grouped KV in upstream KIVI packed form, then materializes dequantized K/V for the native Transformers attention interface. It does not yet replace attention matmul with upstream qB matmul kernels.
- Upstream `quant.new_pack` imports after installing `triton-windows==3.7.1.post27` in the `optiz-qwen` conda environment.
- Full `benchmark_public.py --backend transformers` smoke with `OPTIZ_QWEN_KIVI_KV_CACHE=1`, `--num-samples 1`, and `--max-new-tokens 1` timed out after 240 seconds on this Windows machine, so end-to-end generation is not yet validated.

Known model boundary:

The local Qwen3.5-2B config uses a mixed attention stack with `linear_attention` and `full_attention` layers. The adapter keeps native linear-attention conv/recurrent cache layers and replaces full-attention layer indices `(3, 7, 11, 15, 19, 23)`.

## Setup

From PowerShell:

```powershell
.\scripts\prepare_kivi_upstream.ps1
```

The upstream project also requires installing its Python package and CUDA quant extension before real inference:

```powershell
pip install -e .\artifacts\third_party\KIVI
pip install .\artifacts\wheels\triton_windows-3.7.1.post27-cp311-cp311-win_amd64.whl
Push-Location .\artifacts\third_party\KIVI\quant
pip install -e .
Pop-Location
```

The CUDA extension under `quant` is only needed for upstream qB matmul kernels. The current Qwen3.5 cache adapter uses `quant.new_pack` and `triton-windows`, so it can be tested before compiling the CUDA extension.
