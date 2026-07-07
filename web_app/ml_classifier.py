"""Inference: OpenCLIP + ResNet18 ensemble на perspective crop."""

from __future__ import annotations

import io
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_META = ROOT / "data" / "ml_perspective" / "metadata_perspective.json"
DEFAULT_CLIP = ROOT / "data" / "ml_perspective" / "clip_finetuned_best.pt"
DEFAULT_RESNET = ROOT / "data" / "ml_perspective" / "resnet18_baseline.pt"

log = logging.getLogger(__name__)


class _Ensemble:
    def __init__(
        self,
        clip_ckpt: Path,
        resnet_ckpt: Path,
        meta_path: Path,
        *,
        w_clip: float = 0.70,
        w_resnet: float = 0.30,
        device: str | None = None,
    ) -> None:
        import open_clip

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.class_mapping: dict[str, int] = meta["class_mapping"]
        self.id_to_class = {v: k for k, v in self.class_mapping.items()}
        self.num_classes = len(self.class_mapping)

        cpt = torch.load(clip_ckpt, map_location=self.device, weights_only=False)
        arg = cpt.get("args") or {}
        model_name = arg.get("model", "ViT-B-32")
        pretrained = arg.get("pretrained", "laion2b_s34b_b79k")

        self.clip, _, self.preprocess_clip = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.clip.visual.load_state_dict(cpt["model_visual_state_dict"])
        dim = self.clip.visual.output_dim
        self.head = nn.Linear(dim, self.num_classes).to(self.device)
        self.head.load_state_dict(cpt["head_state_dict"])
        self.clip.eval()
        self.head.eval()

        rnet = models.resnet18(weights=None)
        rnet.fc = nn.Linear(rnet.fc.in_features, self.num_classes)
        rstate = torch.load(resnet_ckpt, map_location=self.device, weights_only=False)
        rnet.load_state_dict(rstate["model_state_dict"])
        rnet.to(self.device).eval()
        self.resnet = rnet
        self.tf_resnet = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        s = w_clip + w_resnet
        self.w_clip = w_clip / s
        self.w_resnet = w_resnet / s

    @torch.no_grad()
    def predict_image(self, img: Image.Image) -> dict[str, Any]:
        x_clip = self.preprocess_clip(img).unsqueeze(0).to(self.device)
        x_res = self.tf_resnet(img).unsqueeze(0).to(self.device)
        feat = self.clip.encode_image(x_clip, normalize=False)
        logits = self.w_clip * self.head(feat) + self.w_resnet * self.resnet(x_res)
        probs = torch.softmax(logits, dim=1)[0]
        conf, pred_id = torch.max(probs, dim=0)
        class_key = self.id_to_class[int(pred_id.item())]
        return {
            "class_key": class_key,
            "confidence": float(conf.item()),
            "probs": {self.id_to_class[i]: float(probs[i].item()) for i in range(self.num_classes)},
        }

    def predict_jpeg(self, jpeg_bytes: bytes) -> dict[str, Any]:
        img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        return self.predict_image(img)


def checkpoints_available() -> bool:
    return DEFAULT_CLIP.is_file() and DEFAULT_RESNET.is_file() and DEFAULT_META.is_file()


@lru_cache(maxsize=1)
def get_classifier() -> _Ensemble | None:
    if not checkpoints_available():
        log.warning("ML checkpoints not found under data/ml_perspective/")
        return None
    try:
        return _Ensemble(DEFAULT_CLIP, DEFAULT_RESNET, DEFAULT_META)
    except Exception:
        log.exception("Failed to load ML classifier")
        return None
