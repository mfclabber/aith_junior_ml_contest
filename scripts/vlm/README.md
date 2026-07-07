# VLM: PaliGemma 3B (LoRA) — класс UTT + evidence-bbox

Дообучение открытой VLM выдавать по перспективному кропу одновременно класс
городской территории (UTT) и evidence-bbox (область-обоснование).

## Установка

```bash
pip install -r requirements-vlm.txt
```

Нужен GPU ≥ 16 GB и принятая лицензия PaliGemma на Hugging Face
(`huggingface-cli login`).

## Пайплайн

1. Собрать VLM-датасет из perspective-кропов. Класс берётся из
   `metadata_perspective.json`, weak evidence-bbox добывается Grad-CAM'ом
   обученного ResNet18 (ручной разметки боксов нет):

   ```bash
   CUDA_VISIBLE_DEVICES=3 python3 scripts/vlm/build_vlm_dataset.py \
       --data-dir data/ml_perspective --out-dir data/vlm
   ```

   Выход: `data/vlm/{train,val}.jsonl` (1937 / 494 записи) + `meta.json`.
   Формат suffix (класс-первый): `"<class_name> <loc..><loc..><loc..><loc..>"`.

2. Дообучить PaliGemma 3B LoRA:

   ```bash
   CUDA_VISIBLE_DEVICES=3 python3 scripts/vlm/train_paligemma_lora.py \
       --epochs 6 --batch-size 8 --lr 2e-4 --out-dir checkpoints/paligemma_lora
   ```

   Адаптер (~90 MB) и метрики сохраняются в `checkpoints/paligemma_lora/`.

3. (Опционально) zero-shot бенчмарк открытой VLM для сравнения — тяжёлая
   загрузка весов:

   ```bash
   CUDA_VISIBLE_DEVICES=3 python3 scripts/vlm/eval_qwen_zeroshot.py --limit 200
   ```

4. Инференс из веб-сервиса: `POST /api/vlm-infer` (файл `file` либо lat/lon),
   реализация в `web_app/vlm_classifier.py`.

## Фактические метрики (val = 494 кропа)

| Прогон | Эпохи | class_acc | evidence IoU |
|--------|-------|-----------|--------------|
| v1 (bbox-первый) | 3 | 0.326 | 0.300 |
| v2 (класс-первый) | 6 | **0.385** | **0.317** |

Замечания:
- Специализированный классификатор OpenCLIP+ResNet на тех же кропах даёт
  существенно выше по классу (~0.83), т.е. дообученная VLM пока не бьёт его
  по чистой классификации, но добавляет **локализацию признака** (evidence-bbox),
  чего у классификатора нет.
- Evidence-bbox обучен на weak-метках (Grad-CAM), поэтому IoU ~0.32 — это по сути
  воспроизведение шумной псевдоразметки; ручная разметка боксов заметно подняла бы
  потолок.
- Основные ошибки класса: `active_construction` ↔ `frozen_construction` (визуально
  близкие стадии стройки).
