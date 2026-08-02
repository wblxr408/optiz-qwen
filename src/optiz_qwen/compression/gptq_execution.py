"""Formal llmcompressor GPTQ W4A16 execution for Qwen3.5-2B VLM.

Heavy dependencies are imported only inside the guarded execute path. Importing
this module for CLI dry-run or preflight does not import torch, transformers,
Pillow, llmcompressor, or compressed-tensors and does not load a model.
"""

from __future__ import annotations

import base64
import csv
import io
import json
from pathlib import Path
from typing import Any

from .gptq_backend import GPTQBackendReadiness, probe_gptq_backend

OPTION_KEYS = ("A", "B", "C", "D")
PERFORMANCE_CLAIM = "not_benchmarked"


def _load_gptq_interfaces() -> tuple[Any, ...]:
    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        from llmcompressor import oneshot
        from llmcompressor.modifiers.gptq import GPTQModifier
    except ImportError as exc:
        raise RuntimeError(
            "The llmcompressor GPTQ execution dependencies are not importable. "
            "Use the verified .venv-awq-linux environment and run --preflight "
            "before --execute."
        ) from exc

    return (
        torch,
        DataLoader,
        AutoProcessor,
        Qwen3_5ForConditionalGeneration,
        oneshot,
        GPTQModifier,
    )


def _load_calibration_rows(
    calibration_tsv: Path,
    *,
    max_samples: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with calibration_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required_columns = {"question", "image"}
        missing_columns = required_columns.difference(reader.fieldnames or ())
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"calibration TSV is missing columns: {missing}")

        for row in reader:
            if not (row.get("question") or "").strip():
                continue
            if not (row.get("image") or "").strip():
                continue
            rows.append({key: value or "" for key, value in row.items()})
            if len(rows) == max_samples:
                break

    if len(rows) != max_samples:
        raise ValueError(
            f"expected {max_samples} usable multimodal calibration rows, "
            f"found {len(rows)}"
        )
    return rows


def _build_prompt(row: dict[str, str]) -> str:
    parts: list[str] = []
    hint = row.get("hint", "").strip()
    if hint:
        parts.append(f"Hint: {hint}")
    parts.append(f"Question: {row['question'].strip()}")
    for key in OPTION_KEYS:
        option = row.get(key, "").strip()
        if option:
            parts.append(f"{key}. {option}")
    parts.append("Answer with one option from A/B/C/D.")
    return "\n".join(parts)


def _decode_image(image_b64: str) -> Any:
    try:
        from PIL import Image

        image_bytes = base64.b64decode(image_b64, validate=True)
        with Image.open(io.BytesIO(image_bytes)) as image:
            return image.convert("RGB")
    except Exception as exc:
        raise ValueError("failed to decode a calibration image") from exc


def _encode_calibration_samples(
    rows: list[dict[str, str]],
    *,
    processor: Any,
    torch_module: Any,
) -> list[dict[str, Any]]:
    """Encode rows eagerly so calibration state contains only plain values."""

    samples: list[dict[str, Any]] = []
    for row in rows:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": _decode_image(row["image"])},
                    {"type": "text", "text": _build_prompt(row)},
                ],
            }
        ]
        encoded = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        sample: dict[str, Any] = {}
        for key, value in dict(encoded).items():
            if torch_module.is_tensor(value):
                sample[str(key)] = value.detach().cpu().clone()
            elif isinstance(value, (str, int, float, bool)):
                sample[str(key)] = value
            elif isinstance(value, list) and all(
                isinstance(item, str) for item in value
            ):
                sample[str(key)] = list(value)
            else:
                raise TypeError(
                    "calibration processor output must contain only tensors, "
                    "scalars, or list[str]; "
                    f"key {key!r} has unsupported type {type(value).__name__}"
                )
        samples.append(sample)

    return samples


class _TensorCalibrationDataset:
    """Pickle-safe calibration samples with no processor or PIL references."""

    def __init__(self, samples: list[dict[str, Any]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


def _single_sample_collator(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) != 1:
        raise ValueError(f"GPTQ calibration expects batch size 1, got {len(batch)}")
    return batch[0]


def _verify_saved_artifact(output_dir: Path) -> dict[str, list[str]]:
    safetensors_files = sorted(output_dir.glob("*.safetensors"))
    recipe_files = sorted(output_dir.glob("*recipe*.yaml"))
    required_files = (
        output_dir / "config.json",
        output_dir / "processor_config.json",
        output_dir / "tokenizer_config.json",
    )

    if not safetensors_files:
        raise RuntimeError("GPTQ execution did not save compressed safetensors")
    if not recipe_files:
        raise RuntimeError("GPTQ execution did not save a recipe YAML")
    missing = [path.name for path in required_files if not path.is_file()]
    if missing:
        raise RuntimeError("GPTQ artifact is missing required files: " + ", ".join(missing))

    return {
        "safetensors": [path.name for path in safetensors_files],
        "recipes": [path.name for path in recipe_files],
        "configuration": [path.name for path in required_files],
    }


def execute_llmcompressor_gptq_quantization(
    *,
    model_path: Path,
    calibration_tsv: Path,
    output_dir: Path,
    num_calibration_samples: int,
    weight_bits: int,
    group_size: int,
    activation_dtype: str,
    confirm_write_artifacts: bool,
    backend_readiness: GPTQBackendReadiness | None = None,
) -> dict[str, Any]:
    """Quantize language-model Linear layers and save compressed-tensors output."""

    if not confirm_write_artifacts:
        raise RuntimeError(
            "real GPTQ execution requires --confirm-write-artifacts before any "
            "model load or artifact write is attempted"
        )
    if weight_bits != 4:
        raise ValueError("GPTQ W4A16 execution requires weight_bits=4")
    if group_size != 128:
        raise ValueError("GPTQ W4A16 execution requires group_size=128")
    if activation_dtype.lower() != "bf16":
        raise ValueError("GPTQ W4A16 execution requires activation_dtype=bf16")
    if num_calibration_samples <= 0:
        raise ValueError("num_calibration_samples must be positive")
    if not model_path.is_dir():
        raise ValueError(f"model_path must be an existing directory: {model_path}")
    if not calibration_tsv.is_file():
        raise ValueError(f"calibration_tsv must be an existing file: {calibration_tsv}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"output directory is not empty: {output_dir}. Choose an empty output "
            "subdirectory to avoid overwriting an existing quantized artifact."
        )

    readiness = backend_readiness or probe_gptq_backend("llmcompressor")
    if readiness.backend_name != "llmcompressor":
        raise RuntimeError(f"unsupported execution backend: {readiness.backend_name}")
    if not readiness.can_quantize:
        raise RuntimeError(
            "llmcompressor GPTQ dependencies are unavailable according to preflight"
        )

    rows = _load_calibration_rows(
        calibration_tsv,
        max_samples=num_calibration_samples,
    )
    (
        torch,
        data_loader_class,
        processor_class,
        model_class,
        oneshot,
        modifier_class,
    ) = _load_gptq_interfaces()

    processor = processor_class.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    calibration_samples = _encode_calibration_samples(
        rows,
        processor=processor,
        torch_module=torch,
    )
    calibration_loader = data_loader_class(
        _TensorCalibrationDataset(calibration_samples),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=_single_sample_collator,
    )
    model = model_class.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    recipe = modifier_class(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head", "re:.*visual.*"],
        block_size=128,
    )
    oneshot(
        model=model,
        dataset=calibration_loader,
        recipe=recipe,
        clear_sparse_session=True,
        num_calibration_samples=num_calibration_samples,
        pipeline="sequential",
        sequential_targets=["Qwen3_5DecoderLayer"],
        sequential_offload_device="cpu",
        save_compressed=True,
        output_dir=str(output_dir),
    )
    processor.save_pretrained(output_dir)

    artifact_files = _verify_saved_artifact(output_dir)
    metadata = {
        "backend": "llmcompressor",
        "algorithm": "gptq",
        "scheme": "W4A16",
        "weight_bits": weight_bits,
        "activation_dtype": activation_dtype.lower(),
        "group_size": group_size,
        "targets": "language model Linear layers",
        "ignored_modules": ["lm_head", "re:.*visual.*"],
        "vision_dtype": "bf16",
        "serialization": "compressed-tensors",
        "model_class": "Qwen3_5ForConditionalGeneration",
        "calibration_sample_count": len(rows),
        "calibration_tsv": str(calibration_tsv),
        "base_model_path": str(model_path),
        "performance_claim": PERFORMANCE_CLAIM,
        "artifact_files": artifact_files,
    }
    metadata_path = output_dir / "quantization_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {
        "mode": "execute",
        "backend": readiness.to_dict(),
        "model_path": str(model_path),
        "calibration_tsv": str(calibration_tsv),
        "output_dir": str(output_dir),
        "quantization": metadata,
        "metadata_path": str(metadata_path),
        "writes_artifacts": True,
        "performance_claim": PERFORMANCE_CLAIM,
    }
