#!/usr/bin/env bash
# Стабильный воспроизводимый пайплайн классификации UTT.
#
# Идея стабильности: НЕ полный fine-tune (переобучается на малых данных), а
# линейный пробинг на замороженных CLIP-фичах (детерминированно, без переобучения,
# фичи кэшируются). Плюс честная чистка данных и честный val (known-bearing).
#
# Использование:
#   CUDA_VISIBLE_DEVICES=3 bash scripts/run_stable_pipeline.sh
set -euo pipefail

PY="./venv/bin/python"
META_RAW="data/ml_perspective/metadata_perspective.json"
META_CLEAN="data/ml_perspective/metadata_perspective_clean.json"
MODEL="ViT-L-14"
PRETRAINED="laion2b_s32b_b82k"

echo "== 1. Аудит датасета =="
$PY scripts/data_quality/audit_dataset.py || true

echo "== 2. Чистка: heading-дизамбигуация + дедуп + честный val =="
$PY scripts/data_quality/clean_dataset.py --meta "$META_RAW" --out-meta "$META_CLEAN"

echo "== 3. Линейный пробинг (стабильно), перебор гранулярности таксономии и обрезки неба =="
for GROUP in 6 5 3; do
  for BC in 1.0 0.55; do
    echo "--- taxonomy=$GROUP  bottom_crop=$BC ---"
    $PY scripts/training/linear_probe_clip.py \
        --metadata "$META_CLEAN" --model "$MODEL" --pretrained "$PRETRAINED" \
        --group "$GROUP" --bottom-crop "$BC" 2>&1 | grep -E "features|BEST"
  done
done

echo "== 4. РАБОЧИЙ путь: переразметка по содержимому VLM-судьёй + правильное обучение =="
echo "   (тяжёлый шаг: качает Qwen2.5-VL ~16GB; JSONL резюмируемый)"
if [ "${RUN_VLM:-0}" = "1" ]; then
  $PY scripts/data_quality/vlm_relabel.py --taxonomy 3 --out data/vlm_relabel/labels3.jsonl
  $PY scripts/data_quality/build_dataset_from_vlm.py --labels data/vlm_relabel/labels3.jsonl
  $PY scripts/training/linear_probe_clip.py \
      --metadata data/ml_perspective/metadata_perspective_vlm3.json \
      --model "$MODEL" --pretrained "$PRETRAINED" \
      --save checkpoints/clip_probe_vlm3.pt 2>&1 | tail -8
else
  echo "   Пропущено. Запустите с RUN_VLM=1, чтобы выполнить переразметку и обучение."
fi

echo "== Готово. Смотрите лучшие BEST-строки по конфигурациям. =="
