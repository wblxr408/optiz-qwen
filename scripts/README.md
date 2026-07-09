# Scripts

This directory is reserved for executable workflow entrypoints such as:

- baseline runner
- evaluation launcher
- profiling launcher
- data-preparation helpers
- benchmark collection

At the current stage, only the skeleton exists because the external assets are still missing.

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

For local Mac development, install extra helper packages separately. These are
not part of the organizer's minimal dependency list:

```bash
conda run -n optiz-qwen python -m pip install -r configs/requirements/local_dev_extra.txt
```

Then place weights in the repository's ignored raw-resource path:

```bash
conda run -n optiz-qwen bash scripts/download_qwen35_2b_modelscope.sh
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
