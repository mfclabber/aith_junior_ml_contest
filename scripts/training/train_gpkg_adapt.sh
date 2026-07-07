#!/usr/bin/env bash
# Дообучение probe под метки аналитиков (GPKG) с warm-start от vlm6_oss.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=./venv/bin/python
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

echo "== merge metadata =="
$PY scripts/data_quality/merge_finetune_metadata.py \
  --analyst-weight 4 \
  --greenery data/ml_perspective/metadata_greenery.json \
  --greenery-weight 6 \
  --no-base-val-only \
  --out data/ml_perspective/metadata_gpkg_adapt.json

echo "== train probe =="
$PY scripts/training/linear_probe_clip.py \
  --metadata data/ml_perspective/metadata_gpkg_adapt.json \
  --init-head checkpoints/clip_probe_vlm6_oss.pt \
  --select-by macro_f1 \
  --epochs 400 \
  --finetune-lr 5e-3 \
  --save checkpoints/clip_probe_vlm6_gpkg.pt

echo "== eval GPKG sample =="
ML_MAX_VIEW_POINTS=3 CLASSIFIER_PROBE=clip_probe_vlm6_gpkg.pt \
  $PY scripts/eval/compare_analyst_gpkg.py --n 30 --out results/analyst_compare_gpkg_adapt.json

ML_MAX_VIEW_POINTS=3 CLASSIFIER_PROBE=clip_probe_vlm6_gpkg.pt \
  $PY scripts/eval/eval_greenery_gpkg.py --n 40 --out results/greenery_eval_gpkg_adapt.json
