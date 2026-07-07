#!/usr/bin/env python3
"""Собрать VLM-датасет (JSONL) для дообучения PaliGemma из perspective-кропов.

Для каждого кропа:
  - class_name / class_id берутся из metadata_perspective.json;
  - weak evidence-bbox добывается Grad-CAM'ом обученного ResNet18 (weak supervision,
    настоящей ручной разметки боксов нет);
  - формируется пара (prefix, suffix) в detection-формате PaliGemma:
        prefix: "detect urban parcel state"
        suffix: "<locYYYY><locXXXX><locYYYY><locXXXX> <class_name>"
    где координаты y_min,x_min,y_max,x_max нормированы в бины 0..1023.

Выход: data/vlm/train.jsonl, data/vlm/val.jsonl (+ meta.json).

Пример:
  python3 scripts/vlm/build_vlm_dataset.py --data-dir data/ml_perspective \
      --out-dir data/vlm --device cuda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
LOC_BINS = 1024


def loc_str(x0: float, y0: float, x1: float, y1: float) -> str:
    """xyxy в [0,1] -> строка PaliGemma <loc> (порядок y_min,x_min,y_max,x_max)."""
    def q(v: float) -> int:
        return int(min(LOC_BINS - 1, max(0, round(v * (LOC_BINS - 1)))))
    return f"<loc{q(y0):04d}><loc{q(x0):04d}><loc{q(y1):04d}><loc{q(x1):04d}>"


class GradCAM:
    """Grad-CAM на последнем conv-блоке ResNet18 (layer4)."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        target = model.layer4
        target.register_forward_hook(self._fwd)
        target.register_full_backward_hook(self._bwd)

    def _fwd(self, module, inp, out):
        self.activations = out.detach()

    def _bwd(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def cam(self, x: torch.Tensor, class_id: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        score = logits[0, class_id]
        score.backward()
        grads = self.gradients  # [1,C,h,w]
        acts = self.activations  # [1,C,h,w]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        denom = cam.max()
        if denom > 0:
            cam = cam / denom
        return cam.cpu().numpy()


def bbox_from_cam(cam: np.ndarray, thr: float = 0.5) -> tuple[float, float, float, float]:
    """Возвращает xyxy в [0,1] по маске cam >= thr. Фолбэк - центральный бокс."""
    h, w = cam.shape
    mask = cam >= thr
    if mask.sum() < max(16, 0.002 * h * w):
        return 0.15, 0.15, 0.85, 0.85
    ys, xs = np.where(mask)
    x0, x1 = xs.min() / (w - 1), xs.max() / (w - 1)
    y0, y1 = ys.min() / (h - 1), ys.max() / (h - 1)
    if x1 - x0 < 0.05:
        x0, x1 = max(0.0, x0 - 0.05), min(1.0, x1 + 0.05)
    if y1 - y0 < 0.05:
        y0, y1 = max(0.0, y0 - 0.05), min(1.0, y1 + 0.05)
    return float(x0), float(y0), float(x1), float(y1)


def load_resnet(ckpt_path: Path, num_classes: int, device: torch.device) -> torch.nn.Module:
    m = models.resnet18(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    state = torch.load(ckpt_path, map_location="cpu")
    m.load_state_dict(state["model_state_dict"])
    m.to(device).eval()
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description="Build PaliGemma VLM dataset from crops.")
    ap.add_argument("--data-dir", type=Path, default=Path("data/ml_perspective"))
    ap.add_argument("--meta", type=Path, default=None,
                    help="metadata json (default: <data-dir>/metadata_perspective.json)")
    ap.add_argument("--out-dir", type=Path, default=Path("data/vlm"))
    ap.add_argument("--cam-thr", type=float, default=0.5)
    ap.add_argument("--prefix", type=str, default="detect urban parcel state")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="0 = все сэмплы")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    data_dir = args.data_dir if args.data_dir.is_absolute() else root / args.data_dir
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = args.meta or (data_dir / "metadata_perspective.json")
    meta_path = meta_path if Path(meta_path).is_absolute() else root / meta_path
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    class_mapping: dict[str, int] = meta["class_mapping"]
    num_classes = len(class_mapping)
    device = torch.device(args.device)

    resnet = load_resnet(data_dir / "resnet18_baseline.pt", num_classes, device)
    cam_engine = GradCAM(resnet)
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(NORM_MEAN, NORM_STD)]
    )

    counts = {"train": 0, "val": 0}
    for split, key in (("train", "train_samples"), ("val", "val_samples")):
        rows = meta[key]
        if args.limit:
            rows = rows[: args.limit]
        out_path = out_dir / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for i, s in enumerate(rows):
                img_path = data_dir / s["image_path"]
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception as e:  # noqa: BLE001
                    print(f"skip {img_path}: {e}")
                    continue
                x = tf(img).unsqueeze(0).to(device)
                x.requires_grad_(True)
                cid = int(s["class_id"])
                cam = cam_engine.cam(x, cid)
                x0, y0, x1, y1 = bbox_from_cam(cam, args.cam_thr)
                rec: dict[str, Any] = {
                    "image": str(img_path),
                    "image_rel": s["image_path"],
                    "class_name": s["class_name"],
                    "class_id": cid,
                    "object_id": s.get("object_id"),
                    "prefix": args.prefix,
                    "suffix": f"{s['class_name']} {loc_str(x0, y0, x1, y1)}",
                    "bbox_xyxy_norm": [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)],
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts[split] += 1
                if (i + 1) % 200 == 0:
                    print(f"{split}: {i + 1}/{len(rows)}")
        print(f"wrote {out_path} ({counts[split]} records)")

    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "class_mapping": class_mapping,
                "prefix": args.prefix,
                "cam_thr": args.cam_thr,
                "counts": counts,
                "loc_bins": LOC_BINS,
                "bbox_source": "weak_supervision_gradcam_resnet18",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("done:", counts)


if __name__ == "__main__":
    main()
