from __future__ import annotations

import sys

import pytest

from optiz_qwen.compression.awq_backend import AWQBackendReadiness, probe_awq_backend

HEAVY_MODULES = {"torch", "transformers", "awq", "autoawq"}


def test_autoawq_probe_has_stable_readiness_shape() -> None:
    readiness = probe_awq_backend("autoawq")
    payload = readiness.to_dict()

    assert isinstance(readiness, AWQBackendReadiness)
    assert payload["backend_name"] == "autoawq"
    assert isinstance(payload["package_available"], bool)
    assert payload["can_quantize"] == payload["package_available"]
    assert payload["reason"]
    assert "recommended_environment" in payload
    assert "NVIDIA GPU" in payload["recommended_environment"]


def test_backend_probe_does_not_import_heavy_dependencies() -> None:
    before = {name for name in HEAVY_MODULES if name in sys.modules}

    probe_awq_backend("autoawq")

    after = {name for name in HEAVY_MODULES if name in sys.modules}
    assert after == before


def test_backend_name_is_case_insensitive() -> None:
    readiness = probe_awq_backend("AutoAWQ")

    assert readiness.backend_name == "autoawq"


def test_unknown_backend_is_rejected_with_clear_error() -> None:
    with pytest.raises(ValueError, match="unsupported AWQ backend"):
        probe_awq_backend("gptq")
