# Option-position bias probe — 2026-08-18

## Evidence

On `PPU-ZW810E`, the target worktree ran the paired original/reversed-option
probe on the sequential 50-sample English MMBench development slice.  Both
arms used the same image, question, generation configuration (greedy,
`max_new_tokens=64`), model, and parser.  All 50 pairs parsed successfully.
The raw, ignored benchmark artifact is
`benchmarks/output/ppu_option_position_bias_20260818.json`.

| Measure | Original order | Reversed order |
| --- | ---: | ---: |
| Accuracy | 74% (37/50) | 76% (38/50) |
| Predicted A | 32 | 43 |
| Predicted B | 18 | 7 |

Of the 50 pairs, 25 retained the same answer letter after reversal (all `A`)
and 25 retained the same semantic option after mapping the reversed letter
back.  The original errors were all repaired by reversal (13), but reversal
also created 12 errors from originally-correct answers.  No pair was wrong in
both arms.

## Decision

Candidate options were (1) add a software-level option-position debiasing rule
and claim an accuracy gain, or (2) retain the current path and treat the
one-sided error table as a mixed-behaviour signal.  The evaluation dimensions
were paired semantic stability, net accuracy, inference cost, and OCR/fine
localization safety.

Choose **option 2**.  The 50% letter-sticky rate confirms material first-option
behaviour, but the equally large content-stable group means a global letter
flip, fixed calibration, or always-swapped prompt would damage valid answers.
The observed net change is only +1/50 and running both permutations would
double inference work, so it is not an accuracy-retaining or latency-safe
mainline optimization.  A future debiasing proposal needs an independently
validated confidence/tie-breaker and explicit OCR/fine-grained localization
evaluation before it can be enabled.
