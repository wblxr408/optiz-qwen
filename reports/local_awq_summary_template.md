# Local AWQ Smoke Summary Template

This template is not an official benchmark and is not a result report.

Use it only after a real local BF16 vs AWQ smoke run has been executed with
team-authored local samples. Do not fill in accuracy, TTFT, throughput, latency,
or memory values from dry-run, preflight, or status-summary output.

## Scope

- Model family: Qwen3.5-2B VLM
- Quantization path: AWQ W4A16
- Validation scope: local smoke / regression / sanity check
- Performance claim: not_benchmarked
- Official benchmark status: not run here

## Required Inputs Before Filling Results

- BF16 baseline model is present locally.
- AWQ artifact is present locally under the configured artifact path.
- Local `samples.jsonl` exists and contains only team-authored smoke samples.
- No official competition evaluation dataset was used for this local smoke report.

## Smoke Comparison Results

Fill this section only after a real BF16 vs AWQ smoke run.

| Field | BF16 Baseline | AWQ W4A16 | Notes |
| --- | --- | --- | --- |
| Sample count | TBD | TBD | Local samples only |
| Pass / sanity status | TBD | TBD | Not an official score |
| OCR risk observations | TBD | TBD | Check text-heavy images |
| Fine localization risk observations | TBD | TBD | Check small objects and positions |
| TTFT | TBD | TBD | Do not report unless measured |
| Throughput | TBD | TBD | Do not report unless measured |

## Boundaries

- This template is not an official benchmark.
- The local smoke result must not claim performance gains.
- The local smoke result must not be reported as an official score.
- Artifacts must not be committed to Git.
- Official evaluation remains separate and must be handled through the official
  benchmark path in a later phase.

## Notes

- Record hardware, dependency versions, and exact commands only after the real
  smoke run happens.
- Keep `performance_claim: not_benchmarked` until a real BF16 baseline vs AWQ
  benchmark artifact exists.
