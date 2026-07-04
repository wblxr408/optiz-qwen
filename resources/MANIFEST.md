# Resource Manifest

This repository intentionally does not vendor large external assets. The following resources are still missing and must be downloaded before full implementation and benchmarking can proceed.

## Required assets

1. Evaluation dataset
- Purpose: accuracy evaluation
- Expected landing path: `resources/eval_dataset/raw/`
- Expected format: image, question, reference-answer triples
- Current state: not downloaded

2. Qwen3.5-2B model weights
- Purpose: baseline deployment and all optimization work
- Expected landing path: `resources/model_weights/raw/`
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
