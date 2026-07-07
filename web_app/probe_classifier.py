"""CLIP linear probe — 6 классов UTT по умолчанию."""

from __future__ import annotations

import io
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]

# 6-классовые чекпойнты (приоритет: macro-F1, потом acc)
_PROBE_CANDIDATES_6 = [
    ROOT / "checkpoints" / "clip_probe_vlm6_oss.pt",
    ROOT / "checkpoints" / "clip_probe_vlm6_strong.pt",
    ROOT / "checkpoints" / "clip_probe_vlm6_bal.pt",
    ROOT / "checkpoints" / "clip_probe_vlm6.pt",
    ROOT / "checkpoints" / "clip_probe_osm6.pt",
]

_PROBE_CANDIDATES_3 = [
    ROOT / "checkpoints" / "clip_probe_vlm3.pt",
]


def _taxonomy() -> str:
    return os.environ.get("CLASSIFIER_TAXONOMY", "6").strip()


def _best_probe_path() -> Path:
    forced = os.environ.get("CLASSIFIER_PROBE", "").strip()
    if forced:
        p = ROOT / "checkpoints" / forced
        if p.is_file():
            return p
    want_n = 6 if _taxonomy() != "3" else 3
    candidates = _PROBE_CANDIDATES_6 if want_n == 6 else _PROBE_CANDIDATES_3
    best_p, best_score = None, -1.0
    for p in candidates:
        if not p.is_file():
            continue
        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            ncls = len(ckpt.get("class_mapping", {}))
            if ncls != want_n:
                continue
            # для 6 классов оптимизируем macro-F1; для 3 — accuracy
            if want_n == 6:
                score = float(ckpt.get("val_macro_f1", ckpt.get("val_balanced_f1", 0.0)))
            else:
                score = float(ckpt.get("val_acc", 0.0))
            if score > best_score:
                best_score, best_p = score, p
        except Exception:
            continue
    fallback = ROOT / "checkpoints" / "clip_probe_vlm6.pt"
    return best_p or fallback


def _maybe_bottom_crop(img: Image.Image, bottom: float) -> Image.Image:
    if bottom >= 0.999:
        return img
    w, h = img.size
    top = int(h * (1.0 - bottom))
    return img.crop((0, top, w, h))


class _ClipProbe:
    def __init__(self, ckpt_path: Path, *, device: str | None = None) -> None:
        import open_clip

        self.ckpt_path = ckpt_path
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.class_mapping: dict[str, int] = ckpt["class_mapping"]
        self.id_to_class = {v: k for k, v in self.class_mapping.items()}
        self.num_classes = len(self.class_mapping)
        self.bottom_crop = float(ckpt.get("bottom_crop", 1.0))
        self.normalize = bool(ckpt.get("normalize_features", True))
        self.val_acc = float(ckpt.get("val_acc", 0.0))
        self.val_macro_f1 = float(ckpt.get("val_macro_f1", 0.0))

        model_name = ckpt.get("model", "ViT-L-14")
        pretrained = ckpt.get("pretrained", "laion2b_s32b_b82k")
        self.clip, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.clip.eval()
        dim = self.clip.visual.output_dim
        self.head = nn.Linear(dim, self.num_classes).to(self.device)
        self.head.load_state_dict(ckpt["head_state_dict"])
        self.head.eval()

    @torch.no_grad()
    def predict_image(self, img: Image.Image) -> dict[str, Any]:
        img = _maybe_bottom_crop(img, self.bottom_crop)
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        feat = self.clip.encode_image(x)
        if self.normalize:
            feat = feat / feat.norm(dim=-1, keepdim=True)
        logits = self.head(feat)[0]
        probs = torch.softmax(logits, dim=0)
        conf, pred_id = torch.max(probs, dim=0)
        class_key = self.id_to_class[int(pred_id.item())]
        return {
            "class_key": class_key,
            "confidence": float(conf.item()),
            "probs": {self.id_to_class[i]: float(probs[i].item()) for i in range(self.num_classes)},
            "backend": self.ckpt_path.stem,
            "val_acc": self.val_acc,
            "val_macro_f1": self.val_macro_f1,
            "num_classes": self.num_classes,
        }

    def predict_jpeg(self, jpeg_bytes: bytes) -> dict[str, Any]:
        img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        return self.predict_image(img)


def checkpoints_available() -> bool:
    return _best_probe_path().is_file()


@lru_cache(maxsize=1)
def get_probe_classifier() -> _ClipProbe | None:
    path = _best_probe_path()
    if not path.is_file():
        log.warning("CLIP probe checkpoint not found (taxonomy=%s)", _taxonomy())
        return None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        log.info(
            "Loading probe %s taxonomy=%s n=%d acc=%.3f f1=%.3f",
            path.name, _taxonomy(), len(ckpt.get("class_mapping", {})),
            float(ckpt.get("val_acc", 0)), float(ckpt.get("val_macro_f1", 0)),
        )
        return _ClipProbe(path)
    except Exception:
        log.exception("Failed to load CLIP probe classifier")
        return None
