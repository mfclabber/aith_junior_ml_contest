# Demo-данные

Минимальный набор для `make pipeline-quick` и CI **без** полного `data/dataset/`.

| Файл | Описание |
|------|----------|
| `metadata_demo.json` | 6 train + 2 val сэмпла, 2 класса |
| `crops/` | Демо-изображения (PNG как JPEG-заглушки) |

Полный датасет (~13k кропов) собирается пайплайном:

```bash
bash scripts/pipeline/01_collect_data.sh
```

См. `docs/PIPELINE.md`.
