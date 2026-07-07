from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from optiz_qwen.compression.kivi_external import (
    KiviConfig,
    KiviUnsupportedModelError,
    apply_kivi_config_to_transformers_config,
    infer_upstream_model_family,
    inspect_kivi_source,
    load_upstream_quant_module,
)


def write_minimal_kivi_layout(tmp_path: Path) -> None:
    for relative in (
        "README.md",
        "LICENSE",
        "models/llama_kivi.py",
        "models/mistral_kivi.py",
        "quant/new_pack.py",
        "quant/matmul.py",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("MIT License\n", encoding="utf-8")
    (tmp_path / "quant" / "__init__.py").write_text("", encoding="utf-8")


def test_kivi_source_inspection_accepts_expected_layout(tmp_path: Path) -> None:
    write_minimal_kivi_layout(tmp_path)

    status = inspect_kivi_source(tmp_path)

    assert status.exists is True
    assert status.usable_source_tree is True
    assert status.license_name == "MIT"
    assert status.supported_model_families == ("llama", "mistral")


def test_quant_module_loader_uses_upstream_source_path(tmp_path: Path) -> None:
    write_minimal_kivi_layout(tmp_path)
    (tmp_path / "quant" / "new_pack.py").write_text(
        "SOURCE = 'upstream-kivi-test'\n",
        encoding="utf-8",
    )

    module = load_upstream_quant_module(tmp_path)

    assert module.SOURCE == "upstream-kivi-test"


def test_kivi_config_is_attached_to_transformers_config() -> None:
    hf_config = SimpleNamespace()
    kivi_config = KiviConfig(k_bits=2, v_bits=4, group_size=32, residual_length=64)

    returned = apply_kivi_config_to_transformers_config(hf_config, kivi_config)

    assert returned is hf_config
    assert hf_config.k_bits == 2
    assert hf_config.v_bits == 4
    assert hf_config.group_size == 32
    assert hf_config.residual_length == 64
    assert hf_config.use_flash is True


def test_kivi_config_rejects_non_divisible_residual_length() -> None:
    with pytest.raises(ValueError, match="residual_length"):
        KiviConfig(group_size=64, residual_length=32).validate()


def test_model_family_detection_for_supported_upstream_families() -> None:
    assert infer_upstream_model_family({"model_type": "llama"}) == "llama"
    assert infer_upstream_model_family({"architectures": ["MistralForCausalLM"]}) == "mistral"


def test_qwen35_is_reported_as_not_directly_supported() -> None:
    qwen_config = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {"model_type": "qwen3_5_text"},
    }

    with pytest.raises(KiviUnsupportedModelError, match="no Qwen3.5"):
        infer_upstream_model_family(qwen_config)
