"""Shared hashing helpers for AWQ experiment identity and execution authorization."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def semantic_config_sha256(config: dict[str, Any]) -> str:
    """Hash scientific settings while excluding the execution authorization bit."""
    scientific = dict(config)
    scientific.pop("execution_enabled", None)
    serialized = json.dumps(
        scientific,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
