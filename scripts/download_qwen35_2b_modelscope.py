"""Download the teacher-provided Qwen3.5-2B snapshot from ModelScope."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_MODEL_ID = "Qwen/Qwen3.5-2B"
DEFAULT_TARGET_DIR = Path("resources/model_weights/raw/Qwen3.5-2B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Qwen3.5-2B from ModelScope into the local raw-resource path."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--target-dir",
        default=str(DEFAULT_TARGET_DIR),
        help="Local directory that will receive the ModelScope snapshot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_dir = Path(args.target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "modelscope is not installed. Run "
            "`python -m pip install -r configs/requirements/local_dev_extra.txt` first."
        ) from exc

    snapshot_download(
        args.model_id,
        local_dir=str(target_dir),
    )
    print(f"Downloaded {args.model_id} to {target_dir}")


if __name__ == "__main__":
    main()
