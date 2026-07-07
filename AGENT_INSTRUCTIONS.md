# Инструкция для Metric Agent — 6 классов UTT

Ты автономный агент. **Не спрашивай пользователя.** Выполняй шаги, замеряй, принимай решения.

## Цель (обязательные метрики)

Таксономия: **все 6 классов UTT** (`CLASSIFIER_TAXONOMY=6`).

| Метрика | Цель | Текущий лучший |
|---------|------|----------------|
| Accuracy (val) | **≥ 0.75** | 0.762 |
| macro-F1 (val) | **≥ 0.60** | 0.610 |
| balanced macro-F1 (cap=35/class) | **≥ 0.70** | 0.738 |
| per-class F1 | **≥ 0.35** каждый | все 6 выполнены |

Чекпойнт продакшена: `checkpoints/clip_probe_vlm6_oss.pt`  
Датасет: `data/ml_perspective/metadata_perspective_vlm6_oss.json`

**Веб/API:** `CLASSIFIER_MODE=ml` — только ML по панораме, эвристика отключена (`CLASSIFIER_FALLBACK_HEURISTIC=0`).

## Запуск

```bash
cd aith_junior_ml_contest
pip install -r requirements-dev.txt

# главная команда
CLASSIFIER_TAXONOMY=6 ./venv/bin/python scripts/agent/langgraph_agent.py

# или shell-обёртка
bash scripts/agent/run_metric_agent.sh
```

## Пайплайн (строго по порядку)

### Шаг 1. Данные

```bash
./venv/bin/python scripts/data_quality/audit_dataset.py
./venv/bin/python scripts/data_quality/clean_dataset.py
# -> metadata_perspective_clean.json
```

### Шаг 2. VLM-переразметка (content-based, 6 классов)

```bash
CUDA_VISIBLE_DEVICES=3 ./venv/bin/python scripts/data_quality/vlm_relabel.py \
  --taxonomy 6 --out data/vlm_relabel/labels.jsonl
# резюмируемо, нужно 1183 строк
```

### Шаг 3. Сборка датасета

```bash
./venv/bin/python scripts/data_quality/build_dataset_from_vlm.py \
  --taxonomy 6 --labels data/vlm_relabel/labels.jsonl
# -> metadata_perspective_vlm6.json
```

### Шаг 4. Балансировка train (ключ к высоким метрикам)

```bash
./venv/bin/python scripts/data_quality/balance_vlm_dataset.py \
  --min-per-class 80 --max-majority 120 --oversample-factor 5
# -> metadata_perspective_vlm6_oss.json
```

**Не используй** `build_vlm6_strong.py` (stratified split) — ухудшает macro-F1.

### Шаг 5. Обучение CLIP probe

```bash
CUDA_VISIBLE_DEVICES=3 ./venv/bin/python scripts/training/linear_probe_clip.py \
  --metadata data/ml_perspective/metadata_perspective_vlm6_oss.json \
  --model ViT-L-14 --pretrained laion2b_s32b_b82k \
  --group 6 --select-by macro_f1 \
  --save checkpoints/clip_probe_vlm6_oss.pt
```

**Не используй:** sample weights в loss, early stopping, lr < 1e-2 — проверено, ухудшает.

### Шаг 6. Проверка и деплой

```bash
CLASSIFIER_TAXONOMY=6 CUDA_VISIBLE_DEVICES=3 ./venv/bin/python scripts/agent/smoke_test_api.py
```

## Дерево решений

```
START
  ├─ нет clean metadata?        → Шаг 1
  ├─ labels.jsonl < 1183?       → Шаг 2
  ├─ нет vlm6 metadata?         → Шаг 3
  ├─ нет vlm6_oss metadata?     → Шаг 4
  ├─ macro-F1 < 0.60?           → Шаг 5 (переобучить)
  │     ├─ если всё ещё < 0.60 после 2 попыток → НЕ крути гиперпараметры
  │     │   → задокументируй потолок в AGENT_RUN.md
  │     └─ иначе → Шаг 6, FINISH
  └─ все цели достигнуты?       → FINISH
```

## Что НЕ делать

1. **Не завышать метрики** — только честный val
2. **Не обучать на OSM-метках** для высоких цифр (потолок ~0.27)
3. **Не переключаться на 3 класса** без явного `CLASSIFIER_TAXONOMY=3`
4. **Не fine-tune CLIP** на малых данных (переобучение) — только linear probe
5. **Не коммить** без запроса
6. **Не трогать** `application/ai_talent_hub/`

## Как поднять метрики дальше (если цели не достигнуты)

1. Увеличить `--min-per-class` до 100 (не выше 120 — дубликаты)
2. Уменьшить `--max-majority` до 100
3. Дособрать кадры, где редкие классы видны в кадре (construction, degraded)
4. Переразметить val с stratified cap urban@40 (только для отчёта)

## Логи и отчёты

- `results/langgraph_agent.log` — ход агента
- `results/langgraph_state.json` — состояние
- `results/AGENT_RUN.md` — итоговый отчёт
- `results/train_probe_vlm6_oss_final.log` — метрики обучения

## Критерий успеха (STOP)

Все условия одновременно на `vlm6_oss`:

- [ ] accuracy ≥ 0.75
- [ ] macro-F1 ≥ 0.60
- [ ] balanced-F1 ≥ 0.70
- [ ] smoke_test OK
- [ ] docs synced
