from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from collections.abc import Iterator

import pytest

from optiz_qwen.compression.awq_calibration import (
    AWQCalibrationRecord,
    build_prompt_text,
    load_mmbench_calibration_records,
)


@pytest.fixture()
def workspace_tmp_dir() -> Iterator[Path]:
    root = Path(__file__).resolve().parents[1] / ".pytest-workspace-tmp"
    path = root / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        try:
            root.rmdir()
        except OSError:
            pass


def write_tsv(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_load_mmbench_calibration_records_builds_prompt_plan(
    workspace_tmp_dir: Path,
) -> None:
    tsv = write_tsv(
        workspace_tmp_dir / "mmbench_sample.tsv",
        "\t".join(["index", "image", "question", "hint", "A", "B", "C", "D", "answer"])
        + "\n"
        + "\t".join(
            [
                "42",
                "base64-image-is-not-decoded",
                "Which label is shown?",
                "Look at the sign.",
                "Cat",
                "Dog",
                "Bus",
                "Train",
                "C",
            ]
        )
        + "\n",
    )

    records = load_mmbench_calibration_records(tsv)

    assert len(records) == 1
    record = records[0]
    assert isinstance(record, AWQCalibrationRecord)
    assert record.sample_id == "42"
    assert record.question == "Which label is shown?"
    assert record.options == {"A": "Cat", "B": "Dog", "C": "Bus", "D": "Train"}
    assert record.hint == "Look at the sign."
    assert record.answer == "C"
    assert record.image_present is True
    assert "base64-image-is-not-decoded" not in record.prompt_text
    assert "Image: <image>" in record.prompt_text
    assert "Question: Which label is shown?" in record.prompt_text
    assert "C. Bus" in record.prompt_text


def test_load_mmbench_calibration_records_honors_max_samples(
    workspace_tmp_dir: Path,
) -> None:
    tsv = write_tsv(
        workspace_tmp_dir / "mmbench_two_rows.tsv",
        "index\timage\tquestion\tA\tB\tanswer\n"
        "1\timg-1\tQuestion one?\tYes\tNo\tA\n"
        "2\timg-2\tQuestion two?\tRed\tBlue\tB\n",
    )

    records = load_mmbench_calibration_records(tsv, max_samples=1)

    assert [record.sample_id for record in records] == ["1"]


def test_load_mmbench_calibration_records_accepts_repo_relative_string(
    workspace_tmp_dir: Path,
) -> None:
    tsv = write_tsv(
        workspace_tmp_dir / "mmbench_relative.tsv",
        "index\timage\tquestion\tA\tB\tanswer\n"
        "3\timg-3\tQuestion three?\tLeft\tRight\tA\n",
    )
    repo_root = Path(__file__).resolve().parents[1]
    repo_relative_tsv = tsv.relative_to(repo_root).as_posix()

    records = load_mmbench_calibration_records(repo_relative_tsv)

    assert len(records) == 1
    assert records[0].sample_id == "3"
    assert records[0].answer == "A"


def test_load_mmbench_calibration_records_supports_missing_optional_fields(
    workspace_tmp_dir: Path,
) -> None:
    tsv = write_tsv(
        workspace_tmp_dir / "mmbench_minimal.tsv",
        "index\timage\tquestion\tA\tB\n"
        "7\t\tWhat is visible?\tText\tNothing\n",
    )

    record = load_mmbench_calibration_records(tsv)[0]

    assert record.sample_id == "7"
    assert record.options == {"A": "Text", "B": "Nothing"}
    assert record.hint is None
    assert record.answer is None
    assert record.image_present is False
    assert "Image: <image>" not in record.prompt_text


def test_build_prompt_text_keeps_options_in_abcd_order() -> None:
    prompt = build_prompt_text(
        question="Pick the matching option.",
        options={"D": "four", "A": "one", "C": "three"},
        hint="Use the image.",
        image_present=True,
    )

    assert prompt.index("A. one") < prompt.index("C. three") < prompt.index("D. four")
    assert "Hint: Use the image." in prompt


def test_missing_tsv_has_clear_error(workspace_tmp_dir: Path) -> None:
    missing = workspace_tmp_dir / "missing.tsv"

    with pytest.raises(FileNotFoundError, match="MMBench TSV does not exist"):
        load_mmbench_calibration_records(missing)


def test_empty_tsv_has_clear_error(workspace_tmp_dir: Path) -> None:
    empty = write_tsv(workspace_tmp_dir / "empty.tsv", "")

    with pytest.raises(ValueError, match="MMBench TSV is empty"):
        load_mmbench_calibration_records(empty)


def test_header_only_tsv_has_clear_error(workspace_tmp_dir: Path) -> None:
    tsv = write_tsv(workspace_tmp_dir / "header_only.tsv", "index\timage\tquestion\n")

    with pytest.raises(ValueError, match="contains no samples"):
        load_mmbench_calibration_records(tsv)


def test_max_samples_must_be_positive(workspace_tmp_dir: Path) -> None:
    tsv = write_tsv(
        workspace_tmp_dir / "mmbench_sample.tsv",
        "index\timage\tquestion\n1\timg\tQ?\n",
    )

    with pytest.raises(ValueError, match="max_samples must be positive"):
        load_mmbench_calibration_records(tsv, max_samples=0)


def test_missing_question_column_has_clear_error(workspace_tmp_dir: Path) -> None:
    tsv = write_tsv(workspace_tmp_dir / "bad_header.tsv", "index\timage\tA\n1\timg\tYes\n")

    with pytest.raises(ValueError, match="question"):
        load_mmbench_calibration_records(tsv)
