# Metric Agent (LangGraph)

Автономный цикл `evaluate → plan → act → evaluate` до достижения целевых метрик.
Не спрашивает пользователя — сам выбирает действие.

## Запуск

```bash
./venv/bin/python -m pip install -r requirements-dev.txt
./venv/bin/python scripts/agent/langgraph_agent.py --target-acc 0.75 --target-f1 0.60
# или линейный оркестратор фаз 0→3
./venv/bin/python scripts/agent/run_autonomous.py
```

Состояние: `results/langgraph_state.json` · лог: `results/langgraph_agent.log` ·
отчёт: `results/AGENT_RUN.md`.

## Дерево решений

```
evaluate → plan → [action] → evaluate → … → finish (metrics ≥ target)
```

| Действие | Зачем |
|----------|-------|
| `clean_data` | heading-дизамбигуация, чистка меток |
| `vlm_relabel_6` | content-based метки через VLM |
| `build_vlm6` / `balance_vlm6_oss` | сборка и балансировка датасета |
| `train_probe_vlm6_oss` | обучение CLIP probe (6 классов UTT) |
| `smoke_api` | проверка инференса web-API |

## Файлы

- `langgraph_agent.py` — граф решений (главный)
- `graph_state.py`, `graph_tools.py` — состояние и инструменты
- `run_autonomous.py` — линейный оркестратор
- `smoke_test_api.py` — smoke-тест API
