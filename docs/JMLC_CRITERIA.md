# Соответствие критериям JMLC 2026

Источник: [ai.itmo.ru/junior_ml_contest](https://ai.itmo.ru/junior_ml_contest)

## 1. Разработка и инженерия

| Требование | Реализация |
|------------|------------|
| Git | Структурированный репозиторий, `.gitignore`, ветки |
| Docker | `Dockerfile`, `docker-compose.yml` |
| CI | `.github/workflows/ci.yml` — pytest + quick pipeline + docker build |
| MLOps | `Makefile`, `scripts/pipeline/01–07`, `config/pipeline.env.example` |
| Конфигурация | **Hydra** (`conf/`), композиция data/model/train/tracking, multirun |
| Трекинг экспериментов | **TensorBoard / W&B / MLflow** (`scripts/training/tracking.py`) |
| Версионирование данных | **DVC** (`dvc.yaml`, `dvc repro`) |
| Качество кода | `tests/`, `make lint`, **pre-commit + ruff + black**, type hints |
| ML-пайплайны | `docs/PIPELINE.md`, `docs/MLOPS.md`, воспроизводимые bash-шаги |

**Доказательство:** `make pipeline-quick` проходит на CI без GPU; `make train-cfg`
запускает обучение через Hydra с логами в TensorBoard.

## 2. Data Science

| Требование | Реализация |
|------------|------------|
| EDA | `docs/EDA.md`, `scripts/data_quality/audit_dataset.py` |
| Предобработка | `clean_dataset.py`, multi-heading crops |
| Выбор моделей | CLIP probe vs ensemble vs VLM — сравнение в `docs/METRICS.md` |
| Метрики | accuracy, macro-F1, balanced-F1, per-class, confusion |
| Валидация | object-level split, отдельно research val / GPKG |

## 3. Применение ИИ

| Требование | Реализация |
|------------|------------|
| AI-инструменты | Cursor agents, codegen в разработке |
| AI-агенты | `scripts/agent/langgraph_agent.py` — автономный цикл улучшения |
| VLM | weak supervision, `vlm_relabel.py`, PaliGemma LoRA |

**Запуск агента:** `make agent` или `RUN_AGENT=1 make pipeline`

## 4. Продуктовое мышление

| Требование | Реализация |
|------------|------------|
| Проблема / ЦА | `docs/PRODUCT.md` |
| MVP | `web_app/` — GPKG in/out, карта, классификация |
| Импакт | 7535 участков с feedback аналитиков |
| Конкуренты | текст vs спутник vs ручной просмотр |

## 5. Мотивация

`submission/pdf/MOTIVATION_LETTER_Новичков_Дмитрий.pdf`

## Презентация (5 мин + Q&A)

- `presentation/JMLC_pitch.pptx` — 12 слайдов
- Готовность к вопросам: `docs/METRICS.md` § GPKG domain shift
