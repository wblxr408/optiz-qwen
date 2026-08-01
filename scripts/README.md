# Scripts

This directory is reserved for executable workflow entrypoints such as:

- baseline runner
- evaluation launcher
- profiling launcher
- data-preparation helpers
- benchmark collection

## A-direction visual-token experiments

The ToMe experiment entrypoints share the explicit `--matching` switch:

- `tome`: fixed bipartite ToMe matching, which remains the default;
- `pitome`: energy-based PiToMe matching;
- `dtome`: threshold-based dynamic matching.

`calibrate_dtome_threshold.py` builds a scalar threshold and a visual-length
bucketed schedule from a generic image directory. `benchmark_tome_paired.py`
and `profile_tome_visual_blocks.py` accept that calibration JSON through
`--threshold-calibration`. `plot_dtome_allocation.py` visualizes the resulting
per-sample merge allocation and paired TTFT changes.

These strategies are experimental and default off in the DNDX benchmark.
Measured conclusions and figures are under `reports/A/`.

## D-direction AWQ/GDN switches

`run_v11_cuda_matrix.py` has two explicit, default-off switches:

- `--enable-awq`
- `--enable-gdn-fastpath`

With neither switch, exactly one baseline case is selected. The AWQ model and
GDN overlay are external artifacts: the runner never downloads, installs, or
generates them during a benchmark. It validates W4A16 metadata, checks that the
GDN CUDA route is active when requested, and rejects a baseline environment
that already exposes the GDN fast path.

Results are in `reports/d_awq_gdn_results.md`. The complete from-scratch
preparation and four command examples are in
`docs/D方向_AWQ_GDN复现与提交.md`.

At the current stage, model download and one-sample smoke-test helpers are in
place, while the broader optimization workflow scripts are still skeletal.

## Model download

The teacher-provided official model source is ModelScope:

- https://www.modelscope.cn/models/Qwen/Qwen3.5-2B

Create the local development environment with the organizer's minimal runtime
requirements first:

```bash
conda create -n optiz-qwen python=3.11 -y
conda run -n optiz-qwen python -m pip install -r configs/requirements/dndx_public.txt
```

On Windows PowerShell, prefer `conda run --no-capture-output -n optiz-qwen ...`
for benchmark commands so tqdm progress bars are streamed directly instead of
being re-encoded by `conda run`.

For this local Windows GPU smoke-test environment, replace the CPU Torch wheel
with the CUDA wheel overlay:

```bash
conda run -n optiz-qwen python -m pip install --force-reinstall -r configs/requirements/local_gpu_windows_cuda.txt
```

For local development, install extra helper packages separately. These are not
part of the organizer's minimal dependency list:

```bash
conda run -n optiz-qwen python -m pip install -r configs/requirements/local_dev_extra.txt
```

Then place weights in the repository's ignored raw-resource path:

```bash
conda run -n optiz-qwen python scripts/download_qwen35_2b_modelscope.py
```

The target directory is `resources/model_weights/raw/Qwen3.5-2B/`.

## Benchmark comparison

Compare DNDX benchmark JSON files. The script prints aggregate metrics as
tables and generates a per-sample PNG chart.

For daily use, pass only the candidate result you want to compare. The default
baseline paths are hard-coded as:

- `benchmarks/output/result_dev_en_20_mps.json`
- `benchmarks/output/result_dev_cn_20_mps.json`

Compare only the English result:

```bash
python scripts/compare_benchmarks.py \
  --candidate-name after \
  --candidate en=benchmarks/output/after_en.json \
  --plot benchmarks/output/compare_en.png
```

Compare only the Chinese result:

```bash
python scripts/compare_benchmarks.py \
  --candidate-name after \
  --candidate cn=benchmarks/output/after_cn.json \
  --plot benchmarks/output/compare_cn.png
```

You can still compare both datasets or override every path explicitly:

```bash
python scripts/compare_benchmarks.py \
  --baseline-name before \
  --candidate-name after \
  --baseline en=benchmarks/output/before_en.json \
  --baseline cn=benchmarks/output/before_cn.json \
  --candidate en=benchmarks/output/after_en.json \
  --candidate cn=benchmarks/output/after_cn.json \
  --plot benchmarks/output/compare_before_after.png
```

Only the JSON format matters; file names are up to each developer. The PNG
visualizes per-sample answer status changes, validation errors, TTFT deltas,
throughput deltas, and generated-token deltas.

## One-sample public benchmark smoke test

After the model is downloaded, the baseline chain should be:

1. model directory: `resources/model_weights/raw/Qwen3.5-2B/`
2. wrapper: `evaluation_wrapper.py` -> `src/optiz_qwen/evaluation/dndx_wrapper.py`
3. benchmark: `benchmark_public.py`

Run one public sample on the local GPU with:

```bash
conda run -n optiz-qwen python benchmark_public.py --backend transformers --device cuda --num-samples 1
```
