"""VLM-инференс: PaliGemma 3B + LoRA на perspective crop.

Возвращает класс UTT и evidence-bbox (где на кропе признак, обосновывающий класс).
Модель тяжёлая, поэтому грузится лениво (lru_cache) и только при первом запросе.
"""

from __future__ import annotations

import io
import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADAPTER = ROOT / "checkpoints" / "paligemma_lora_clean6"
DEFAULT_VLM_META = ROOT / "data" / "vlm_clean6" / "meta.json"
BASE_MODEL = "google/paligemma-3b-pt-224"
DEFAULT_PREFIX = "detect urban parcel state"
LOC_BINS = 1024

LOC_RE = re.compile(r"<loc(\d{4})>")
log = logging.getLogger(__name__)


class _PaliGemmaVLM:
    def __init__(
        self,
        adapter_dir: Path,
        meta_path: Path,
        *,
        base_model: str = BASE_MODEL,
        prefix: str = DEFAULT_PREFIX,
        device: str | None = None,
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import (
            PaliGemmaForConditionalGeneration,
            PaliGemmaProcessor,
        )

        self.torch = torch
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.class_names = list(meta["class_mapping"].keys())
        self.prefix = meta.get("prefix", prefix)

        self.processor = PaliGemmaProcessor.from_pretrained(adapter_dir)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        base = PaliGemmaForConditionalGeneration.from_pretrained(
            base_model, torch_dtype=dtype
        )
        self.model = PeftModel.from_pretrained(base, adapter_dir)
        self.model.to(self.device).eval()

    def _parse(self, text: str) -> tuple[str | None, list[float] | None]:
        nums = [int(n) for n in LOC_RE.findall(text)]
        bbox = None
        if len(nums) >= 4:
            y0, x0, y1, x1 = nums[:4]
            bbox = [
                round(x0 / (LOC_BINS - 1), 4),
                round(y0 / (LOC_BINS - 1), 4),
                round(x1 / (LOC_BINS - 1), 4),
                round(y1 / (LOC_BINS - 1), 4),
            ]
        tail = LOC_RE.sub("", text).strip().lower()
        cls = None
        for c in self.class_names:
            if c.lower() in tail:
                cls = c
                break
        return cls, bbox

    def predict_image(self, img: Image.Image, max_new_tokens: int = 32) -> dict[str, Any]:
        enc = self.processor(text=self.prefix, images=img, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        gen = self.processor.batch_decode(
            out[:, enc["input_ids"].shape[1]:], skip_special_tokens=False
        )[0]
        cls, bbox = self._parse(gen)
        return {
            "class_key": cls,
            "evidence_bbox_xyxy_norm": bbox,
            "raw": gen.replace("<eos>", "").strip(),
        }

    def predict_jpeg(self, jpeg_bytes: bytes) -> dict[str, Any]:
        img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        return self.predict_image(img)


def checkpoints_available() -> bool:
    return (DEFAULT_ADAPTER / "adapter_config.json").is_file() and DEFAULT_VLM_META.is_file()


@lru_cache(maxsize=1)
def get_vlm() -> _PaliGemmaVLM | None:
    if not checkpoints_available():
        log.warning("VLM (PaliGemma LoRA) checkpoint not found under checkpoints/paligemma_lora_clean6/")
        return None
    try:
        return _PaliGemmaVLM(DEFAULT_ADAPTER, DEFAULT_VLM_META)
    except Exception:
        log.exception("Failed to load PaliGemma VLM")
        return None
