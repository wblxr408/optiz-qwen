# AWQ W4A16 Runbook

## Phase 1 Scope

This phase adds a dry-run CLI only. It validates paths and prints the planned
AWQ W4A16 metadata preview.

It does not:

- load model weights
- import or call `torch`, `transformers`, or AutoAWQ for quantization
- download models or datasets
- install dependencies
- generate files under `artifacts/quantized/`
- claim accuracy, TTFT, throughput, memory, or bandwidth gains

## Dry-Run Command

```bash
D:\Miniconda\envs\optiz-qwen\python.exe scripts/quantize_awq.py ^
  --model-path resources/model_weights/raw/Qwen3.5-2B ^
  --calibration-tsv resources/eval_dataset/raw/mmbench_public/mmbench_dev_en.tsv ^
  --output-dir artifacts/quantized/qwen35_2b_awq_w4a16 ^
  --num-calibration-samples 128 ^
  --dry-run
```

The output is a JSON plan with `performance_claim: not_benchmarked` and
`writes_artifacts: false`.

## Phase 2 Calibration Data Adapter

Phase 2 adds a lightweight MMBench TSV adapter in
`src/optiz_qwen/compression/awq_calibration.py`.

It reads rows from a repository-relative TSV path or a `Path` object and returns
prompt-plan records with:

- `sample_id`
- `question`
- non-empty `A/B/C/D` options
- optional `hint`
- optional `answer`
- `image_present`
- `prompt_text`

The adapter supports `max_samples` for small calibration subsets.

It does not:

- decode image base64
- write image files
- load model weights
- import or call `torch`, `transformers`, or AutoAWQ
- write files under `artifacts/`
- claim accuracy, TTFT, throughput, memory, or bandwidth gains

Example validation command:

```bash
D:\Miniconda\envs\optiz-qwen\python.exe -m pytest tests/test_awq_calibration.py -q
```

## Phase 2 Decision Note

Candidate options:

- Reuse the public benchmark MMBench loader.
- Add a compression-layer calibration adapter.

Evaluation dimensions:

- Keeps AWQ preparation inside the model-compression layer.
- Avoids decoding images or loading models during calibration planning.
- Leaves public benchmark entrypoints unchanged for fair baseline comparison.
- Keeps the path switchable before real AWQ execution.

Final choice:

Use a separate compression-layer adapter. Reusing the benchmark loader is weaker
for this phase because that path is tied to evaluation execution, image decoding,
and generation metrics, while AWQ calibration planning needs only lightweight
records and prompt text.

## Output Directory Rule

The output directory must be repository-relative and must be either:

- `artifacts/quantized/qwen35_2b_awq_w4a16`
- a subdirectory of `artifacts/quantized/qwen35_2b_awq_w4a16`

The dry-run command does not create this directory.

## Future Real Quantization

A later phase may add real AWQ execution on a suitable Linux NVIDIA GPU server.
That phase must keep BF16 baseline loading available and must save any generated
large artifacts under the ignored `artifacts/quantized/` tree.

No result should be described as optimized until a real BF16 baseline vs AWQ
benchmark summary exists.
