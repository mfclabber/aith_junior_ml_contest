# EDA — разведочный анализ данных

Критерий **Data Science** JMLC: понимание данных, предобработка, валидация.

## Источники

| Источник | Объём | Роль |
|----------|-------|------|
| OSM + Яндекс | ~3200 объектов, ~13k кропов | обучение |
| VLM content labels | переразметка | чистые метки |
| GPKG аналитиков | 7535 участков | продуктовая валидация |

## Ключевые находки

### 1. Мусорный азимут (29% кропов)

Панорама в точке объекта → `bearing=0` → модель смотрит мимо участка.

**Решение:** multi-heading `0,90,180,270` в `crop_perspective_dataset.py`.

### 2. OSM vs VLM

Согласие меток **~20%** → OSM ненадёжен как единственный источник.

**Решение:** VLM-судья (Qwen) + weak supervision + analyst crops.

### 3. «Зелень во дворах»

1712 участков с **одинаковым текстом**, классы: 1294 natural / 380 urban.

**Вывод:** текстовая эвристика не может быть единственным продакшен-методом для этого кластера; нужна панорама.

### 4. Баланс классов

`active_urban` доминирует → oversample / cap в `balance_vlm_dataset.py`.

## Скрипты EDA

```bash
# Полный аудит
python scripts/data_quality/audit_dataset.py --dataset-dir data/dataset

# Результат
cat results/dataset_audit.json
```

## Валидация

- Split по **object_id** (не по кропам) — без утечки сезонов
- Отдельные метрики: research val vs GPKG analysts
- Per-class F1, confusion matrix, balanced macro-F1

См. `docs/METRICS.md`.
