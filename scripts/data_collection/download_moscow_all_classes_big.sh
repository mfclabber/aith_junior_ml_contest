#!/bin/bash
# Надежная загрузка большого датасета по Москве: все классы, батчами.
#
# Пример:
#   bash scripts/data_collection/download_moscow_all_classes_big.sh \
#     /home/novichkovde/projects/classification_street_buildings/yandex-pano-downloader/data/big_dataset \
#     2000 \
#     1
#
# Аргументы:
#   1) OUTPUT_DIR  - куда складывать датасет
#   2) MAX_RESULTS - максимум OSM-объектов на класс (попробовать 500..5000)
#   3) ZOOM        - зум панорамы (0..4), обычно 1
#
# Важно: скрипт можно перезапускать — благодаря стабильным item_id (OSM type+id)
# докачка происходит без дублей в тех же папках.

set -euo pipefail

OUTPUT_DIR="${1:-data/big_dataset}"
MAX_RESULTS="${2:-2000}"
ZOOM="${3:-1}"

cd "$(dirname "$0")/../.."

echo "============================================================"
echo "BIG DATASET (Moscow): all classes"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "MAX_RESULTS per class: ${MAX_RESULTS}"
echo "ZOOM: ${ZOOM}"
echo "============================================================"

CLASSES=(
  natural_areas
  low_density_degraded
  underused_infrastructure
  frozen_construction
  active_construction
  active_urban
)

for cls in "${CLASSES[@]}"; do
  echo ""
  echo "------------------------------------------------------------"
  echo "CLASS: ${cls}"
  echo "------------------------------------------------------------"
  python3 scripts/data_collection/dataset_collector.py \
    --class "${cls}" \
    --max-results "${MAX_RESULTS}" \
    --collect-seasons \
    --download \
    --output-dir "${OUTPUT_DIR}" \
    --zoom "${ZOOM}"

  # небольшая пауза между классами, чтобы снизить риск rate limit
  sleep 5
done

echo ""
echo "============================================================"
echo "DONE"
echo "============================================================"

