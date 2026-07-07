# Compression Layer

This package maps to the first layer of the SVG:

- visual token pruning
- text-guided sparsification
- AWQ W4A16
- KV cache quantization
- VLM-specific PTQ
- structured pruning

Recommended implementation priority:

1. AWQ W4A16
2. visual token pruning
3. optional extensions after baseline metrics are stable

## KIVI KV Cache module

`kivi_external.py` is a standalone adapter for the upstream
`jy-yuan/KIVI` codebase. It does not reimplement KIVI locally. It
validates an external checkout under `artifacts/third_party/KIVI` and can
load the upstream Llama/Mistral KIVI model classes.

Current boundary: upstream KIVI does not directly ship a Qwen3.5 VLM model
class, so this module must remain disabled by default until a Qwen
attention/cache adapter is added and benchmarked.
