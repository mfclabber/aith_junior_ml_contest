#!/usr/bin/env bash
# [2/7] EDA и аудит качества данных.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-./venv/bin/python}"
mkdir -p results

echo "[02] Dataset audit"
if [[ -d data/dataset ]]; then
  "$PY" scripts/data_quality/audit_dataset.py \
    --dataset-dir data/dataset \
    --out results/dataset_audit.json
  echo "    → results/dataset_audit.json"
else
  echo "[02] data/dataset missing — audit demo metadata only"
  "$PY" -c "
import json
from pathlib import Path
m = json.loads(Path('data/demo/metadata_demo.json').read_text())
print('demo samples:', len(m['train_samples'])+len(m['val_samples']))
"
fi
