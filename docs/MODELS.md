# Модели и чекпойнты

Чекпойнты не в Git — см. `docs/MODELS.md` в README § Быстрый старт.

**Продакшен:** `checkpoints/clip_probe_vlm6_oss.pt` (ViT-L-14 linear head, ~21 KB)

Обучение:

```bash
CUDA_VISIBLE_DEVICES=0 ./venv/bin/python scripts/training/linear_probe_clip.py \
  --metadata data/ml_perspective/metadata_perspective_vlm6_oss.json \
  --model ViT-L-14 --pretrained laion2b_s32b_b82k \
  --select-by macro_f1 --save checkpoints/clip_probe_vlm6_oss.pt
```

Demo без GPU: `CLASSIFIER_MODE=heuristic CLASSIFY_REQUIRE_PANORAMA=0`
