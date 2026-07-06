#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-2B}"
TARGET_DIR="${TARGET_DIR:-resources/model_weights/raw/Qwen3.5-2B}"

mkdir -p "$TARGET_DIR"

if command -v modelscope >/dev/null 2>&1; then
  modelscope download --model "$MODEL_ID" --local_dir "$TARGET_DIR"
else
  python - <<'PY'
import os
from modelscope import snapshot_download

model_id = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-2B")
target_dir = os.environ.get("TARGET_DIR", "resources/model_weights/raw/Qwen3.5-2B")
snapshot_download(model_id, local_dir=target_dir)
PY
fi

echo "Downloaded $MODEL_ID to $TARGET_DIR"
