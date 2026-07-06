# Resource Manifest

This repository intentionally does not vendor large external assets. The following resources are still missing and must be downloaded before full implementation and benchmarking can proceed.

## Required assets

1. Evaluation dataset
- Purpose: accuracy evaluation
- Expected landing path: `resources/eval_dataset/raw/`
- Expected format: image, question, reference-answer triples
- Current state: official final dataset not downloaded; organizer public self-test TSV files are placed under `resources/eval_dataset/raw/mmbench_public/`

2. Qwen3.5-2B model weights
- Purpose: baseline deployment and all optimization work
- Expected landing path: `resources/model_weights/raw/Qwen3.5-2B/`
- Expected source: official model release path chosen by the team
- Current state: not downloaded

3. PPU reference materials
- Purpose: operator support mapping, runtime constraints, and packing/layout decisions
- Expected landing path: `resources/ppu_docs/raw/`
- Current state: not fully downloaded

4. Standardized evaluation environment details
- Purpose: reproducible benchmarking and compatibility checks
- Expected landing path: `resources/ppu_docs/raw/` or a later environment-specific directory
- Current state: pending

## Rules

- Keep raw external assets out of git unless explicitly required.
- Add a short note here whenever a resource is downloaded, moved, or renamed.
- Do not replace "not downloaded" with guessed paths or guessed filenames.

## Notes

- 2026-07-06: the imported `dndx_participant` public self-test TSV files were normalized from `dndx_participant/datasets/mmbench/` to `resources/eval_dataset/raw/mmbench_public/`.
- 2026-07-06: the local model placeholder path was normalized from `./Qwen3.5-2B` to `resources/model_weights/raw/Qwen3.5-2B/`.
