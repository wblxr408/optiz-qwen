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

## Phase 3 Local Smoke Validation Spec

Phase 3 defines a local-only smoke validation contract:

- config: `configs/experiments/local_awq_smoke.yaml`
- sample landing path: `resources/local_validation/samples.jsonl`
- sample format note: `resources/local_validation/README.md`

This is not an official benchmark. It is only for quick smoke, regression, and
sanity checks before any formal baseline-vs-AWQ comparison.

The config keeps:

- `validation_scope: local_smoke`
- `quantization: awq_w4a16`
- `performance_claim: not_benchmarked`
- `max_samples: 10`
- `max_new_tokens: 64`

The phase does not create `samples.jsonl`, image files, benchmark outputs, or
quantized artifacts. It also does not load models, run inference, quantize
weights, or use the official competition evaluation dataset.

## Phase 3 Decision Note

Candidate options:

- Use official or public MMBench data for the smoke set.
- Define a separate team-authored local validation area.

Evaluation dimensions:

- Avoids contaminating official evaluation data.
- Keeps local smoke checks separate from `benchmark_public.py`.
- Allows tiny hand-authored OCR, localization, and instruction-following cases.
- Preserves `performance_claim: not_benchmarked` until real benchmark artifacts exist.

Final choice:

Use a separate `resources/local_validation/` spec. Reusing official or public
MMBench data is weaker for this phase because local smoke checks are not score
artifacts and must not be reported as official benchmark results.

## Phase 4 Local Smoke Runner Dry-Run

Phase 4 adds `scripts/validate_local_awq.py`, a dry-run CLI for the local smoke
validation plan.

Example:

```bash
D:\Miniconda\envs\optiz-qwen\python.exe scripts/validate_local_awq.py ^
  --config configs/experiments/local_awq_smoke.yaml ^
  --dry-run
```

The runner:

- reads `configs/experiments/local_awq_smoke.yaml`
- requires `validation_scope: local_smoke`
- requires `performance_claim: not_benchmarked`
- validates repository-relative config and sample paths
- reports `samples_status: missing` when `samples.jsonl` has not been created
- validates only JSONL metadata fields when a local sample file exists
- prints a JSON plan to stdout

It does not:

- load model weights
- read or decode images
- run local inference
- quantize weights
- write benchmark outputs or artifacts
- use the official competition evaluation dataset
- touch `benchmark_public.py`
- claim accuracy, TTFT, throughput, memory, or bandwidth gains

## Phase 4 Decision Note

Candidate options:

- Add a dry-run local smoke runner now.
- Wait until real AWQ inference can run and add validation then.

Evaluation dimensions:

- Keeps local sample contracts testable before model execution exists.
- Preserves the separation between smoke checks and official benchmark code.
- Avoids accidental model loading or artifact writes on small local GPUs.
- Maintains `performance_claim: not_benchmarked`.

Final choice:

Add the dry-run runner now. Waiting for real inference is weaker for this phase
because the sample and config contract can be validated safely before hardware,
artifacts, and final local samples are available.

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
