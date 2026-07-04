# Scheduling Layer

This package maps to the second layer of the SVG:

- paged KV management
- prefill/decode split
- mixed precision service
- continuous batching
- speculative decoding
- attention windowing when applicable

Recommended implementation priority:

1. paged KV interfaces
2. prefill/decode separation
3. optional advanced scheduling branches
