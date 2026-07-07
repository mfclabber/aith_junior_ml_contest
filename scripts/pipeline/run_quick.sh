#!/usr/bin/env bash
# Быстрый воспроизводимый пайплайн для CI и жюри без полного датасета.
# EDA на demo-metadata → linear probe (CPU) → smoke API.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${PY:-./venv/bin/python}"
[[ -x "$PY" ]] || PY=python3

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Quick pipeline (demo data, CPU-friendly)                ║"
echo "╚══════════════════════════════════════════════════════════╝"

echo "→ [1/4] Unit tests"
"$PY" -m pytest tests/ -q --tb=line

echo "→ [2/4] EDA / audit demo metadata"
"$PY" scripts/data_quality/audit_dataset.py \
  --metadata data/demo/metadata_demo.json 2>/dev/null || \
  "$PY" -c "
import json
from pathlib import Path
m = json.loads(Path('data/demo/metadata_demo.json').read_text())
print('demo train', len(m['train_samples']), 'val', len(m['val_samples']))
"

echo "→ [3/4] Linear probe on demo (CPU, few epochs)"
CUDA_VISIBLE_DEVICES="" "$PY" scripts/training/linear_probe_clip.py \
  --metadata data/demo/metadata_demo.json \
  --data-dir data/demo \
  --device cpu \
  --epochs 50 \
  --cache-tag demo \
  --save checkpoints/clip_probe_demo.pt 2>&1 | tail -6

echo "→ [4/4] Smoke import web_app"
"$PY" scripts/agent/smoke_test_api.py --heuristic-only 2>/dev/null || \
  "$PY" -c "from web_app import app; from web_app.taxonomy import UTT_LABELS; print('OK', len(UTT_LABELS), 'classes')"

echo "✓ Quick pipeline done. Checkpoint: checkpoints/clip_probe_demo.pt"
