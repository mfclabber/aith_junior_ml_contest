#!/usr/bin/env bash
# [1/7] Сбор панорам OSM + Яндекс (опционально, если data/dataset пуст).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-./venv/bin/python}"

SKIP_COLLECT="${SKIP_COLLECT:-0}"
if [[ "$SKIP_COLLECT" == "1" ]]; then
  echo "[01] SKIP collect (SKIP_COLLECT=1)"
  exit 0
fi

if [[ -d data/dataset ]] && [[ -n "$(ls -A data/dataset 2>/dev/null)" ]]; then
  echo "[01] data/dataset exists — skip collection"
  exit 0
fi

echo "[01] Collect balanced panoramas (target 500 — уменьшено для демо)"
"$PY" scripts/data_collection/balanced_collector.py \
  --target-panoramas "${TARGET_PANORAMAS:-500}" \
  --output-dir data/dataset
