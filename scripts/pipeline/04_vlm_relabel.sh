#!/usr/bin/env bash
# [4/7] VLM-переразметка (опционально, RUN_VLM_RELABEL=1).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-./venv/bin/python}"

if [[ "${RUN_VLM_RELABEL:-0}" != "1" ]]; then
  echo "[04] SKIP VLM relabel (set RUN_VLM_RELABEL=1 to enable)"
  exit 0
fi

echo "[04] VLM relabel + build vlm6 dataset"
"$PY" scripts/data_quality/vlm_relabel.py --taxonomy 6 --out data/vlm_relabel/labels6.jsonl
"$PY" scripts/data_quality/build_dataset_from_vlm.py --labels data/vlm_relabel/labels6.jsonl
"$PY" scripts/data_quality/balance_vlm_dataset.py \
  --out data/ml_perspective/metadata_perspective_vlm6_oss.json
