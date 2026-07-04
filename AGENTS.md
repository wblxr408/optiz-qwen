# AGENTS.md

## Scope

This repository is for one specific competition task only:

- model: `Qwen3.5-2B` VLM
- target: edge deployment optimization
- hardware focus: Alibaba Cloud `PPU`
- scoring focus: accuracy retention, `TTFT`, throughput, and system-level optimization depth

General-purpose agent behavior is not enough for this repo. Every change must align with the competition flow and the four-layer architecture in `docs/vlm_optimization_full_stack_architecture.svg`.

## Canonical truth sources

Read these first before making structural or technical changes:

1. `docs/赛题.md`
2. `docs/README.md`
3. `docs/vlm_optimization_full_stack_architecture.svg`

If these sources disagree, stop and ask for clarification before implementation.

## Delivery priorities

Always work in this order unless the user explicitly changes it:

1. baseline deployment path
2. evaluation pipeline
3. low-risk mainline optimizations
4. kernel and hardware adaptation
5. optional enhancement branches

Do not start from speculative decoding or other high-variance ideas before the baseline and evaluation pipeline are in place.

## Repository mapping

- `src/optiz_qwen/compression/`: algorithm-side compression
- `src/optiz_qwen/scheduling/`: inference-engine scheduling
- `src/optiz_qwen/kernels/`: fused kernels and operator-path optimization
- `src/optiz_qwen/ppu/`: PPU-specific adaptation
- `src/optiz_qwen/evaluation/`: metrics, dataset adapters, benchmark entrypoints
- `resources/`: external assets not stored in git
- `benchmarks/`: saved benchmark outputs and comparison tables
- `reports/`: competition-facing reports and paper materials

## Competition-specific working rules

- Never fabricate benchmark numbers.
- Never claim "optimized" unless there is a baseline comparison artifact.
- Never claim "PPU-adapted" unless the path was checked on target hardware or an official compatibility source.
- Every optimization must be switchable so that baseline and optimized paths can be compared fairly.
- Accuracy-sensitive changes must explicitly note OCR and fine-grained localization risk.

## Resource reality

At this stage, the following assets are still missing:

- evaluation dataset
- model weights
- part of the PPU documents or environment details

When these are missing:

- add structure and manifests only
- do not invent file names that pretend the assets already exist
- record expected landing paths in `resources/MANIFEST.md`

## Definition of done

Work may be called complete only when all relevant items below are true:

- the requested file or module exists in the correct layer
- the change matches the competition scope
- commands, tests, or manual checks were actually run when applicable
- any missing resource, bug, or unimplemented part is stated explicitly

## Decision logging

For technical decisions, compare at least two options and record:

- candidate options
- evaluation dimensions
- final choice
- why the rejected option is weaker for this competition task

Keep the note concise, but do not skip it.
