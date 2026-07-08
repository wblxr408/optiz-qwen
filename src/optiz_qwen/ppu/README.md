# PPU Layer

This package maps to the fourth layer of the SVG:

- lookup-table operator coverage
- HBM bandwidth optimization
- weight packing format
- unsupported-op rewrites

No module in this layer should claim completion until checked against real PPU constraints.

Current compatibility status is exposed by:

```python
from optiz_qwen.ppu import inspect_ppu_compatibility
```

The current report is expected to be `claim="unverified"` until target
hardware or official compatibility materials are checked.
