#!/bin/bash
# Скрипт для автоматической загрузки всех классов со всеми сезонами
# До 100 объектов на класс, радиус поиска 20 метров

cd "$(dirname "$0")/.."

echo "============================================================"
echo "Загрузка всех классов со всеми сезонами"
echo "До 100 объектов на класс"
echo "Радиус поиска панорам: 20 метров"
echo "============================================================"
echo ""

python3 scripts/data_collection/dataset_collector.py \
    --class all \
    --max-results 100 \
    --collect-seasons \
    --download \
    --output-dir data/dataset \
    --zoom 1

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "Загрузка завершена успешно!"
    echo "Next step: python3 scripts/training/prepare_ml_dataset.py --dataset-dir data/dataset"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "Загрузка завершена с ошибками (код: $EXIT_CODE)"
    echo "============================================================"
fi

exit $EXIT_CODE
