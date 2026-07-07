#!/usr/bin/env bash
# Полный пересбор пайплайна: аудит → ML metadata → crops → train → eval
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/venv/bin/python"
[ -x "$PY" ] || PY=python3

echo "=== 1. Аудит датасета ==="
"$PY" scripts/data_quality/audit_dataset.py

echo "=== 2. ML metadata (object split) ==="
"$PY" scripts/training/prepare_ml_dataset.py \
  --dataset-dir data/dataset \
  --output-dir data/ml_dataset \
  --val-ratio 0.2

echo "=== 3. Perspective crops ==="
"$PY" scripts/training/crop_perspective_dataset.py \
  --metadata data/ml_dataset/metadata.json \
  --output-dir data/ml_perspective \
  --default-bearing-when-unknown-deg 0

echo "=== 4. Train ResNet18 baseline ==="
"$PY" scripts/training/train_baseline.py --epochs 30 --device cuda

echo "=== 5. Fine-tune OpenCLIP ==="
"$PY" scripts/training/train_clip_classifier.py \
  --finetune-last-n 12 --epochs 60 --batch-size 12 --device cuda --val-tta

echo "=== 6. Eval ==="
"$PY" scripts/training/eval_clip_zeroshot.py --device cuda | tee results/eval_clip_zeroshot.txt
"$PY" scripts/training/eval_ensemble_clip_resnet.py --device cuda --sweep-weights | tee results/eval_ensemble.txt

echo "=== Готово. Запуск UI с ML: ==="
echo "  CLASSIFIER_MODE=ml CLASSIFIER_PROBE=clip_probe_vlm6_oss.pt python -m web_app.app"
