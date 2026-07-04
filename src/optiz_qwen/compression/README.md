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
