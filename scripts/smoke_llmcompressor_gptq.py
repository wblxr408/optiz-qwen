"""One-off llmcompressor GPTQ smoke test for the local Qwen3.5-2B VLM.

This script intentionally lives outside the production quantization path. It
loads ten local MMBench rows, applies GPTQ W4A16/G128 to language-model linear
layers, saves a compressed-tensors checkpoint, and reloads that checkpoint.
"""

from __future__ import annotations

import base64
import csv
import gc
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "resources/model_weights/raw/Qwen3.5-2B"
CALIBRATION_TSV = (
    REPO_ROOT / "resources/eval_dataset/raw/mmbench_public/mmbench_dev_en.tsv"
)
OUTPUT_DIR = REPO_ROOT / "artifacts/smoke_gptq"
NUM_CALIBRATION_SAMPLES = 10
OPTION_KEYS = ("A", "B", "C", "D")


def require_inputs() -> None:
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {MODEL_DIR}")
    if not CALIBRATION_TSV.is_file():
        raise FileNotFoundError(f"calibration TSV does not exist: {CALIBRATION_TSV}")
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise RuntimeError(
            f"output directory is not empty: {OUTPUT_DIR}. "
            "Move or remove the previous smoke artifact before rerunning."
        )


def load_calibration_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with CALIBRATION_TSV.open("r", encoding="utf-8", newline="") as handle:
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
            if len(rows) == NUM_CALIBRATION_SAMPLES:
                break

    if len(rows) != NUM_CALIBRATION_SAMPLES:
        raise ValueError(
            f"expected {NUM_CALIBRATION_SAMPLES} usable calibration rows, "
            f"found {len(rows)}"
        )
    return rows


def build_prompt(row: dict[str, str]) -> str:
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


def decode_image(image_b64: str) -> Image.Image:
    try:
        image_bytes = base64.b64decode(image_b64, validate=True)
        with Image.open(io.BytesIO(image_bytes)) as image:
            return image.convert("RGB")
    except Exception as exc:
        raise ValueError("failed to decode a calibration image") from exc


class MMBenchCalibrationDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, rows: list[dict[str, str]], processor: Any) -> None:
        self.rows = rows
        self.processor = processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": decode_image(row["image"])},
                    {"type": "text", "text": build_prompt(row)},
                ],
            }
        ]
        encoded = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return dict(encoded)


def single_sample_collator(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if len(batch) != 1:
        raise ValueError(f"smoke calibration expects batch size 1, got {len(batch)}")
    return batch[0]


def as_serializable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): as_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_serializable(item) for item in value]
    if hasattr(value, "to_dict"):
        return as_serializable(value.to_dict())
    return str(value)


def print_reload_summary(model: Qwen3_5ForConditionalGeneration) -> None:
    model_class = f"{type(model).__module__}.{type(model).__name__}"
    quantization_config = getattr(model.config, "quantization_config", None)

    representative_name = None
    representative_dtype = None
    weight_dtypes: Counter[str] = Counter()
    for name, parameter in model.named_parameters():
        if "weight" not in name:
            continue
        weight_dtypes[str(parameter.dtype)] += parameter.numel()
        if (
            representative_name is None
            and "language_model.layers" in name
            and parameter.dtype not in {torch.bfloat16, torch.float16, torch.float32}
        ):
            representative_name = name
            representative_dtype = parameter.dtype

    print(f"model class: {model_class}")
    print(
        "quantization config: "
        + json.dumps(as_serializable(quantization_config), indent=2, ensure_ascii=False)
    )
    print(
        f"weight dtype: {representative_dtype} "
        f"(representative parameter: {representative_name})"
    )
    print(f"weight dtype element counts: {dict(weight_dtypes)}")


def verify_saved_artifact() -> None:
    safetensors_files = sorted(OUTPUT_DIR.glob("*.safetensors"))
    recipe_files = sorted(OUTPUT_DIR.glob("*recipe*.yaml"))
    processor_files = [
        OUTPUT_DIR / "processor_config.json",
        OUTPUT_DIR / "tokenizer_config.json",
    ]
    if not safetensors_files:
        raise RuntimeError("no compressed safetensors file was saved")
    if not (OUTPUT_DIR / "config.json").is_file():
        raise RuntimeError("saved artifact is missing config.json")
    if not recipe_files:
        raise RuntimeError("saved artifact is missing a recipe YAML")
    if not all(path.is_file() for path in processor_files):
        raise RuntimeError("saved artifact is missing tokenizer/processor configuration")

    print("saved safetensors:", ", ".join(path.name for path in safetensors_files))
    print("saved recipe:", ", ".join(path.name for path in recipe_files))


def main() -> None:
    require_inputs()
    rows = load_calibration_rows()

    processor = AutoProcessor.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        trust_remote_code=True,
    )
    calibration_loader = DataLoader(
        MMBenchCalibrationDataset(rows, processor),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=single_sample_collator,
    )

    # The checkpoint is a full VLM and calibration includes images, so use the
    # conditional-generation class rather than the text-only CausalLM mapping.
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    # llmcompressor's W4A16 preset uses static per-group quantization with a
    # group size of 128. Keep the visual tower unquantized for this smoke test.
    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head", "re:.*visual.*"],
        block_size=128,
    )

    quantized_model = oneshot(
        model=model,
        processor=processor,
        dataset=calibration_loader,
        recipe=recipe,
        clear_sparse_session=True,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        pipeline="sequential",
        sequential_targets=["Qwen3_5DecoderLayer"],
        sequential_offload_device="cpu",
        save_compressed=True,
        output_dir=str(OUTPUT_DIR),
    )
    processor.save_pretrained(OUTPUT_DIR)
    verify_saved_artifact()

    del quantized_model
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    reloaded_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        OUTPUT_DIR,
        local_files_only=True,
        trust_remote_code=True,
        dtype="auto",
        device_map="auto",
    ).eval()
    print_reload_summary(reloaded_model)


if __name__ == "__main__":
    main()
