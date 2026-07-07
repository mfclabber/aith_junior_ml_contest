#!/bin/bash
# Быстрый запуск сбалансированного сбора на 5000+ панорам
#
# Использование:
#   bash scripts/data_collection/collect_balanced_5k.sh
#   bash scripts/data_collection/collect_balanced_5k.sh 10000  # для 10к панорам

set -euo pipefail

TARGET="${1:-5000}"
OUTPUT_DIR="${2:-data/balanced_dataset_5k}"

cd "$(dirname "$0")/../.."

echo "============================================================"
echo "СБАЛАНСИРОВАННЫЙ СБОР ДАТАСЕТА"
echo "============================================================"
echo "Целевое количество панорам: ${TARGET}"
echo "Выходная директория: ${OUTPUT_DIR}"
echo "============================================================"
echo ""

python3 scripts/data_collection/balanced_collector.py \
  --target-panoramas "${TARGET}" \
  --output-dir "${OUTPUT_DIR}" \
  --initial-max-results 1500 \
  --max-iterations 6 \
  --zoom 1

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✓ Сбор завершен успешно!"
    echo "============================================================"
    echo "Результаты: ${OUTPUT_DIR}"
    echo "Сводка: ${OUTPUT_DIR}/collection_summary.json"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "⚠ Сбор завершен с ошибками (код: $EXIT_CODE)"
    echo "Можно перезапустить - досбор продолжится автоматически"
    echo "============================================================"
fi

exit $EXIT_CODE
