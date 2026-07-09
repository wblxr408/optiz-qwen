"""Real AWQ execution helpers with delayed heavy imports.

Importing this module does not import torch, transformers, awq, or autoawq.
Those imports are intentionally delayed until ``execute_autoawq_quantization``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .awq_backend import AWQBackendReadiness, probe_awq_backend
from .awq_calibration import load_mmbench_calibration_records

PERFORMANCE_CLAIM = "not_benchmarked"


def _load_autoawq_interfaces() -> tuple[Any, Any]:
    try:
        import awq as awq_module
    except ImportError as exc:
        try:
            import autoawq as awq_module
        except ImportError as fallback_exc:
            raise RuntimeError(
                "AutoAWQ backend is not importable. Install and verify AutoAWQ "
                "only on the dedicated quantization host before using --execute."
            ) from fallback_exc

    try:
        import transformers as transformers_module
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for real AWQ execution but is not importable."
        ) from exc

    model_cls = getattr(awq_module, "AutoAWQForCausalLM", None)
    tokenizer_cls = getattr(transformers_module, "AutoTokenizer", None)
    if model_cls is None:
        raise RuntimeError(
            "AutoAWQ interface AutoAWQForCausalLM is unavailable. The installed "
            "backend may not support this real execution path or VLM architecture."
        )
    if tokenizer_cls is None:
        raise RuntimeError("transformers.AutoTokenizer is unavailable.")
    return model_cls, tokenizer_cls


def execute_autoawq_quantization(
    *,
    model_path: Path,
    calibration_tsv: Path,
    output_dir: Path,
    num_calibration_samples: int,
    weight_bits: int,
    group_size: int,
    activation_dtype: str,
    zero_point: bool,
    confirm_write_artifacts: bool,
    backend_readiness: AWQBackendReadiness | None = None,
) -> dict[str, Any]:
    """Execute AutoAWQ quantization after explicit artifact-write confirmation."""

    if not confirm_write_artifacts:
        raise RuntimeError(
            "real AWQ execution requires --confirm-write-artifacts before any "
            "model load or artifact write is attempted"
        )

    readiness = backend_readiness or probe_awq_backend("autoawq")
    if readiness.backend_name != "autoawq":
        raise RuntimeError(f"unsupported execution backend: {readiness.backend_name}")
    if not readiness.package_available:
        raise RuntimeError(
            "AutoAWQ backend is not available according to preflight readiness. "
            "Real execution cannot continue."
        )

    calibration_records = load_mmbench_calibration_records(
        calibration_tsv,
        max_samples=num_calibration_samples,
    )
    calibration_prompts = [record.prompt_text for record in calibration_records]
    if not calibration_prompts:
        raise RuntimeError("calibration prompt list is empty")

    model_cls, tokenizer_cls = _load_autoawq_interfaces()
    tokenizer = tokenizer_cls.from_pretrained(str(model_path), trust_remote_code=True)
    model = model_cls.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        safetensors=True,
    )

    quant_config = {
        "zero_point": zero_point,
        "q_group_size": group_size,
        "w_bit": weight_bits,
        "version": "GEMM",
    }
    try:
        model.quantize(
            tokenizer,
            quant_config=quant_config,
            calib_data=calibration_prompts,
        )
    except TypeError as exc:
        raise RuntimeError(
            "AutoAWQ quantize interface did not accept the planned VLM calibration "
            "arguments. Verify backend support for Qwen3.5-2B VLM before rerun."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    if not hasattr(model, "save_quantized"):
        raise RuntimeError("AutoAWQ model object does not expose save_quantized")
    model.save_quantized(str(output_dir))
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(str(output_dir))

    return {
        "mode": "execute",
        "backend": readiness.to_dict(),
        "model_path": str(model_path),
        "calibration_tsv": str(calibration_tsv),
        "output_dir": str(output_dir),
        "calibration_sample_count": len(calibration_prompts),
        "quantization": {
            "method": "awq",
            "scheme": "W4A16",
            "weight_bits": weight_bits,
            "activation_dtype": activation_dtype.lower(),
            "group_size": group_size,
            "zero_point": zero_point,
        },
        "writes_artifacts": True,
        "performance_claim": PERFORMANCE_CLAIM,
    }
