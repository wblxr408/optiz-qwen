"""MMBench TSV adapter for future AWQ calibration.

This module only builds lightweight records and prompt text. It does not decode
image base64, load models, import torch/transformers/AutoAWQ, or write artifacts.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

REPO_ROOT = Path(__file__).resolve().parents[3]
OPTION_KEYS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class AWQCalibrationRecord:
    """Lightweight MMBench sample prepared for a later AWQ calibration pass."""

    sample_id: str
    question: str
    options: dict[str, str]
    hint: str | None
    answer: str | None
    image_present: bool
    prompt_text: str


def _is_absolute_path(value: str) -> bool:
    return PureWindowsPath(value).is_absolute() or value.startswith("/")


def _resolve_tsv_path(path: str | Path) -> Path:
    if isinstance(path, str):
        if not path or path.strip() == "":
            raise ValueError("tsv_path must not be empty")
        if _is_absolute_path(path) or PureWindowsPath(path).drive:
            raise ValueError(f"tsv_path string must be repository-relative: {path}")
        normalized = path.replace("\\", "/").strip("/")
        relative = Path(normalized)
        if ".." in relative.parts:
            raise ValueError(f"tsv_path must not contain path traversal '..': {path}")
        resolved = REPO_ROOT / relative
    else:
        windows_path = PureWindowsPath(str(path))
        if path.is_absolute() or windows_path.drive:
            raise ValueError(f"tsv_path Path must be repository-relative: {path}")
        if ".." in path.parts:
            raise ValueError(f"tsv_path must not contain path traversal '..': {path}")
        resolved = REPO_ROOT / path

    if not resolved.exists():
        raise FileNotFoundError(f"MMBench TSV does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"MMBench TSV path must be a file: {resolved}")
    return resolved


def _clean(value: str | None) -> str:
    return (value or "").strip()


def _sample_id_from_row(row: dict[str, str], row_number: int) -> str:
    for key in ("index", "sample_id", "id"):
        value = _clean(row.get(key))
        if value:
            return value
    return str(row_number)


def build_prompt_text(
    *,
    question: str,
    options: dict[str, str],
    hint: str | None = None,
    image_present: bool = False,
) -> str:
    """Build text that a future VLM processor can pair with image input."""

    parts = [
        "Calibrate the Qwen3.5-2B VLM on this single-choice image-text task.",
    ]
    if image_present:
        parts.append("Image: <image>")
    if hint:
        parts.append(f"Hint: {hint}")
    parts.append(f"Question: {question}")
    for key in OPTION_KEYS:
        option = options.get(key)
        if option:
            parts.append(f"{key}. {option}")
    parts.append("Answer with one option from A/B/C/D.")
    return "\n".join(parts)


def _record_from_row(row: dict[str, str], row_number: int) -> AWQCalibrationRecord:
    question = _clean(row.get("question"))
    if not question:
        raise ValueError(f"MMBench TSV row {row_number} is missing required question text")

    options = {key: _clean(row.get(key)) for key in OPTION_KEYS if _clean(row.get(key))}
    hint = _clean(row.get("hint")) or None
    answer = _clean(row.get("answer")).upper() or None
    image_present = bool(_clean(row.get("image")))
    prompt_text = build_prompt_text(
        question=question,
        options=options,
        hint=hint,
        image_present=image_present,
    )

    return AWQCalibrationRecord(
        sample_id=_sample_id_from_row(row, row_number),
        question=question,
        options=options,
        hint=hint,
        answer=answer,
        image_present=image_present,
        prompt_text=prompt_text,
    )


def load_mmbench_calibration_records(
    tsv_path: str | Path,
    *,
    max_samples: int | None = None,
) -> list[AWQCalibrationRecord]:
    """Read MMBench TSV rows into AWQ calibration prompt records.

    Args:
        tsv_path: Repository-relative string path or a ``Path`` object.
        max_samples: Optional positive limit for the number of rows to load.
    """

    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided")

    resolved = _resolve_tsv_path(tsv_path)
    if resolved.stat().st_size == 0:
        raise ValueError(f"MMBench TSV is empty: {resolved}")

    records: list[AWQCalibrationRecord] = []
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"MMBench TSV is missing a header row: {resolved}")
        if "question" not in reader.fieldnames:
            raise ValueError("MMBench TSV must include a 'question' column")

        for row_number, row in enumerate(reader, start=1):
            records.append(_record_from_row(row, row_number))
            if max_samples is not None and len(records) >= max_samples:
                break

    if not records:
        raise ValueError(f"MMBench TSV contains no samples: {resolved}")
    return records
