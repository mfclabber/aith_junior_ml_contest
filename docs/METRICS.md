# Метрики и валидация

Документ для жюри JMLC: **что измерено, на каких данных, с какими ограничениями**.

## Продакшен-модель: CLIP ViT-L-14 linear probe

Чекпойнт: `checkpoints/clip_probe_vlm6_oss.pt`  
Датасет: `metadata_perspective_vlm6_oss.json` (VLM content-метки, не сырые OSM)  
Val: **n = 239**, split по object_id (без утечки сезонов)

| Метрика | Значение |
|---------|----------|
| Accuracy | **76.2%** |
| macro-F1 | **61.0%** |
| balanced macro-F1 (cap 35/class) | **73.8%** |

### Per-class F1 (research val)

| Класс | F1 |
|-------|-----|
| natural_areas | 0.69 |
| low_density_degraded | 0.44 |
| underused_infrastructure | 0.38 |
| frozen_construction | **0.80** |
| active_construction | 0.50 |
| active_urban | **0.85** |

Все **6 классов UTT** представлены в val и в confusion matrix.

## Качество данных

| Проверка | Результат |
|----------|-----------|
| OSM-метки vs VLM content | согласие **~20%** |
| Кропы с `bearing=0` (мусорный ракурс) | было **29%** → исправлено multi-heading |
| Аудит аналитиков (GPKG, 7535 уч.) | 165 без категории, 1712× «зелень во дворах» с одним текстом |

## Сравнение с правками аналитиков (GPKG)

| Метод | Class accuracy | Выборка |
|-------|----------------|---------|
| Эвристика по тексту GPKG | **87.6%** | 7370 |
| ML mode + vlm6_oss | 44–80% | 25–50 с панорамами |
| Greenery hard-case (natural vs urban) | ~23% | 30 участков |

### Почему ML ниже эвристики на GPKG?

1. Текст GPKG уже содержит сильные подсказки для эвристики.
2. 1712 участков с одинаковым текстом «зелень во дворах» — класс различается только панорамой.
3. ML обучен на VLM/OSM val, не на полном московском GPKG (domain shift).

## Воспроизведение

```bash
CUDA_VISIBLE_DEVICES=0 ./venv/bin/python scripts/training/linear_probe_clip.py \
  --metadata data/ml_perspective/metadata_perspective_vlm6_oss.json \
  --model ViT-L-14 --pretrained laion2b_s32b_b82k --select-by macro_f1
```

Архитектура и пайплайн: `docs/ARCHITECTURE.md`, `docs/PIPELINE.md`.
