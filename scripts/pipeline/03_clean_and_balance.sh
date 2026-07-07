#!/usr/bin/env bash
# [3/7] Чистка, perspective crops, балансировка VLM6.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-./venv/bin/python}"

META_RAW="${META_RAW:-data/ml_perspective/metadata_perspective.json}"
META_CLEAN="${META_CLEAN:-data/ml_perspective/metadata_perspective_clean.json}"

if [[ -f "$META_RAW" ]]; then
  echo "[03] Clean dataset (heading disambiguation)"
  "$PY" scripts/data_quality/clean_dataset.py --meta "$META_RAW" --out-meta "$META_CLEAN"
fi

if [[ -f data/ml_dataset/metadata.json ]] || [[ -d data/dataset ]]; then
  echo "[03] Perspective crops"
  if [[ ! -f data/ml_dataset/metadata.json ]]; then
    "$PY" scripts/training/prepare_ml_dataset.py \
      --dataset-dir data/dataset --output-dir data/ml_dataset --val-ratio 0.2
  fi
  "$PY" scripts/training/crop_perspective_dataset.py \
    --metadata data/ml_dataset/metadata.json \
    --output-dir data/ml_perspective \
    --multi-heading-when-unknown "0,90,180,270"
fi

echo "[03] Balance VLM6 OSS metadata"
if [[ -f data/ml_perspective/metadata_perspective_vlm6_oss.json ]]; then
  echo "    metadata_perspective_vlm6_oss.json exists"
else
  "$PY" scripts/data_quality/balance_vlm_dataset.py \
    --out data/ml_perspective/metadata_perspective_vlm6_oss.json 2>/dev/null || \
    echo "    skip balance (needs VLM labels)"
fi
