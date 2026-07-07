#!/usr/bin/env bash
# [5/7] Обучение CLIP linear probe (продакшен).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-./venv/bin/python}"

META="${METADATA_VLM6:-data/ml_perspective/metadata_perspective_vlm6_oss.json}"
MODEL="${CLIP_MODEL:-ViT-L-14}"
PRETRAINED="${CLIP_PRETRAINED:-laion2b_s32b_b82k}"
DEVICE="${TRAIN_DEVICE:-cuda}"
OUT="${CHECKPOINT_DIR:-checkpoints}/clip_probe_vlm6_oss.pt"
TRACKING="${TRACKING_BACKENDS:-tensorboard}"
RUNS_DIR="${RUNS_DIR:-runs}"

if [[ ! -f "$META" ]]; then
  echo "[05] $META not found — train on demo"
  META=data/demo/metadata_demo.json
  DEVICE=cpu
  OUT=checkpoints/clip_probe_demo.pt
fi

mkdir -p "$(dirname "$OUT")"
echo "[05] Train linear probe: $META → $OUT (tracking=$TRACKING)"
"$PY" scripts/training/linear_probe_clip.py \
  --metadata "$META" \
  --model "$MODEL" \
  --pretrained "$PRETRAINED" \
  --select-by macro_f1 \
  --device "$DEVICE" \
  --tracking "$TRACKING" \
  --runs-dir "$RUNS_DIR" \
  --save "$OUT"
