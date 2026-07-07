# Архитектура

## Конвейер

```
GPKG → точки панорам (URL / грани полигона) → кроп 896×672
     → CLIP ViT-L-14 → linear probe (6 классов) → класс + подкласс
```

## Ключевые модули

| Модуль | Роль |
|--------|------|
| `web_app/classify_service.py` | ML-инференс, голосование по ракурсам |
| `web_app/probe_classifier.py` | CLIP probe |
| `web_app/gpkg_io.py` | GPKG, view_points, export |
| `scripts/training/linear_probe_clip.py` | Обучение probe |
| `scripts/agent/langgraph_agent.py` | Автономный ML-агент |

Полный пайплайн — в [`PIPELINE.md`](PIPELINE.md), про агентов — в
[`../AGENT_INSTRUCTIONS.md`](../AGENT_INSTRUCTIONS.md).
