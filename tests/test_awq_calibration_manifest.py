from __future__ import annotations

import base64
import csv
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/prepare_awq_calibration.py"
SPEC = importlib.util.spec_from_file_location("prepare_awq_calibration", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
calibration = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = calibration
SPEC.loader.exec_module(calibration)


def png_base64(color: tuple[int, int, int]) -> str:
    image = Image.new("RGB", (2, 2), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class AWQCalibrationManifestTests(unittest.TestCase):
    def make_fixture(self, root: Path, *, sample_count: int = 8) -> tuple[Path, Path]:
        dataset = root / "mmbench_dev_en.tsv"
        columns = [
            "index",
            "question",
            "hint",
            "A",
            "B",
            "C",
            "D",
            "answer",
            "category",
            "image",
            "l2-category",
        ]
        with dataset.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
            writer.writeheader()
            for index in range(sample_count):
                writer.writerow(
                    {
                        "index": str(index),
                        "question": f"question-{index}",
                        "hint": "",
                        "A": "a",
                        "B": "b",
                        "C": "c",
                        "D": "d",
                        "answer": "A",
                        "category": "test",
                        "image": png_base64((index, 0, 0)),
                        "l2-category": "unit",
                    }
                )
        config = root / "config.json"
        config.write_text(
            json.dumps(
                {
                    "experiment_id": "test",
                    "execution_enabled": False,
                    "calibration": {
                        "dataset": str(dataset),
                        "expected_dataset_sha256": calibration.sha256_file(dataset),
                        "num_samples": 3,
                        "seed": 7,
                        "exclude_fixed_eval_prefix_count": 2,
                    },
                }
            ),
            encoding="utf-8",
        )
        return dataset, config

    def test_selection_is_deterministic_disjoint_and_image_validated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, config = self.make_fixture(root)
            first = calibration.build_manifest(repo_root=root, config_path=config)
            second = calibration.build_manifest(repo_root=root, config_path=config)

        self.assertEqual(first["selection"], second["selection"])
        self.assertEqual(first["selection"]["selected_calibration_samples"], 3)
        self.assertEqual(first["selection"]["overlap_count"], 0)
        self.assertTrue(first["selection"]["all_images_decoded"])
        self.assertTrue(
            all(row["image_mode"] == "RGB" for row in first["selection"]["samples"])
        )
        self.assertFalse(first["execution_enabled"])

    def test_dataset_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, config = self.make_fixture(root)
            dataset.write_text("changed", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "SHA-256"):
                calibration.build_manifest(repo_root=root, config_path=config)

    def test_existing_output_requires_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "manifest.json"
            output.write_text("existing", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                calibration.write_manifest(output, {"ok": True}, overwrite=False)
            calibration.write_manifest(output, {"ok": True}, overwrite=True)

            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"ok": True})


if __name__ == "__main__":
    unittest.main()
