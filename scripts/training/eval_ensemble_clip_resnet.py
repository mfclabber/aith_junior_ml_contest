#!/usr/bin/env python3
"""
Average logits from fine-tuned CLIP (visual + head) and ResNet18 baseline on val.

  python3 scripts/training/eval_ensemble_clip_resnet.py \\
      --clip-ckpt data/ml_perspective/clip_finetuned_best.pt \\
      --resnet-ckpt data/ml_perspective/resnet18_baseline.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import open_clip
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class RowsDataset(Dataset):
    """CLIP needs OpenCLIP preprocess (224); ResNet baseline used raw crop + tensor normalize."""

    def __init__(self, root: Path, rows: list, preprocess_clip, tf_resnet):
        self.root = root
        self.rows = rows
        self.preprocess_clip = preprocess_clip
        self.tf_resnet = tf_resnet

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        s = self.rows[i]
        img = Image.open(self.root / s["image_path"]).convert("RGB")
        return self.preprocess_clip(img), self.tf_resnet(img), int(s["class_id"])


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/ml_perspective"))
    parser.add_argument("--clip-ckpt", type=Path, default=Path("data/ml_perspective/clip_finetuned_best.pt"))
    parser.add_argument("--resnet-ckpt", type=Path, default=Path("data/ml_perspective/resnet18_baseline.pt"))
    parser.add_argument("--w-clip", type=float, default=0.5)
    parser.add_argument("--w-resnet", type=float, default=0.5)
    parser.add_argument(
        "--sweep-weights",
        action="store_true",
        help="Try clip weights 0,0.1,...,1 and report best val accuracy",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--metadata", type=Path, default=None,
                        help="Defaults to <data-dir>/metadata_perspective.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    data_dir = args.data_dir if args.data_dir.is_absolute() else project_root / args.data_dir
    meta_path = args.metadata or (data_dir / "metadata_perspective.json")
    meta_path = meta_path if Path(meta_path).is_absolute() else project_root / meta_path
    meta = json.loads(Path(meta_path).read_text())
    val_rows = meta["val_samples"]
    class_mapping = meta["class_mapping"]
    num_classes = len(class_mapping)
    wsum = args.w_clip + args.w_resnet
    wc, wr = args.w_clip / wsum, args.w_resnet / wsum

    device = torch.device(args.device)
    clip_path = args.clip_ckpt if args.clip_ckpt.is_absolute() else project_root / args.clip_ckpt
    rnet_path = args.resnet_ckpt if args.resnet_ckpt.is_absolute() else project_root / args.resnet_ckpt
    cpt = torch.load(clip_path, map_location=device, weights_only=False)
    arg = cpt.get("args") or {}
    model_name = arg.get("model", "ViT-B-32")
    pretrained = arg.get("pretrained", "laion2b_s34b_b79k")

    m_clip, _preprocess_train, preprocess_clip = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    tf_resnet = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    loader = DataLoader(
        RowsDataset(data_dir, val_rows, preprocess_clip, tf_resnet),
        batch_size=args.batch_size,
        shuffle=False,
    )
    m_clip.visual.load_state_dict(cpt["model_visual_state_dict"])
    dim = m_clip.visual.output_dim
    head = nn.Linear(dim, num_classes).to(device)
    head.load_state_dict(cpt["head_state_dict"])
    m_clip.eval()
    head.eval()

    m_r = models.resnet18(weights=None)
    m_r.fc = nn.Linear(m_r.fc.in_features, num_classes)
    rsd = torch.load(rnet_path, map_location=device, weights_only=False)
    m_r.load_state_dict(rsd["model_state_dict"])
    m_r.eval()
    m_r.to(device)

    # Precompute logits (expensive models once per batch)
    all_lc: List[torch.Tensor] = []
    all_lr: List[torch.Tensor] = []
    all_y: List[torch.Tensor] = []
    with torch.no_grad():
        for xc, xr, y in loader:
            xc, xr = xc.to(device), xr.to(device)
            all_lc.append(head(m_clip.encode_image(xc, normalize=False)).cpu())
            all_lr.append(m_r(xr).cpu())
            all_y.append(y)

    lc_cat = torch.cat(all_lc, dim=0)
    lr_cat = torch.cat(all_lr, dim=0)
    y_cat = torch.cat(all_y, dim=0).long()

    def _acc(w_clip: float) -> Tuple[float, np.ndarray]:
        w_c = w_clip
        w_r = 1.0 - w_clip
        logits = w_c * lc_cat + w_r * lr_cat
        pred = logits.argmax(dim=1)
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        correct = int((pred == y_cat).sum().item())
        for t, p in zip(y_cat.numpy(), pred.numpy()):
            cm[int(t), int(p)] += 1
        return correct / len(y_cat), cm

    if args.sweep_weights:
        best_a, best_w = -1.0, 0.0
        for k in range(21):
            w = k / 20.0
            a, _ = _acc(w)
            print(f"w_clip={w:.2f}  val_acc={a:.4f}")
            if a > best_a:
                best_a, best_w = a, w
        print(f"\nBest w_clip={best_w:.2f}  val_acc={best_a:.4f}")
        wc, wr = best_w, 1.0 - best_w
        correct = int(best_a * len(y_cat))
        total = len(y_cat)
        _, cm = _acc(best_w)
    else:
        logits = wc * lc_cat + wr * lr_cat
        pred = logits.argmax(dim=1)
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        correct = int((pred == y_cat).sum().item())
        total = len(y_cat)
        for t, p in zip(y_cat.numpy(), pred.numpy()):
            cm[int(t), int(p)] += 1

    print(
        f"Ensemble val acc: {correct/total:.4f}  (n={total})  weights clip={wc:.2f} resnet={wr:.2f}\n"
        f"  clip: {clip_path.name}  resnet: {rnet_path.name}"
    )
    short = [k[:12] for k, _ in sorted(class_mapping.items(), key=lambda x: x[1])]
    print(" " * 14 + "".join(f"{s:>14}" for s in short))
    for i, name in enumerate(short):
        print(f"{name:>14}" + "".join(f"{cm[i, j]:>14}" for j in range(num_classes)))


if __name__ == "__main__":
    main()
