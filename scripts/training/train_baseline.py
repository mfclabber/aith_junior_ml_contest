#!/usr/bin/env python3
"""
Baseline image classifier on perspective crops.

Uses metadata_perspective.json (same labels as prepare_ml_dataset class_id).
Expects layout under --data-dir:
  train/<class_name>/*.jpg, val/<class_name>/*.jpg, metadata_perspective.json

Install deps:
  pip install -r requirements.txt

Example:
  python3 scripts/training/train_baseline.py --data-dir data/ml_perspective --epochs 10
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


class PerspectiveCropDataset(Dataset):
    """Indexed by metadata samples; labels use class_id (taxonomy order, not alphabetical)."""

    def __init__(self, root: Path, sample_rows: List[Dict[str, Any]], transform):
        self.root = root
        self.sample_rows = sample_rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.sample_rows)

    def __getitem__(self, idx: int):
        s = self.sample_rows[idx]
        path = self.root / s["image_path"]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        y = int(s["class_id"])
        return img, y


def _seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResNet18 baseline on perspective crops.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/ml_perspective"))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Defaults to <data-dir>/metadata_perspective.json",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    data_dir = args.data_dir if args.data_dir.is_absolute() else project_root / args.data_dir
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    meta_path = args.metadata or (data_dir / "metadata_perspective.json")
    if not meta_path.is_file():
        raise SystemExit(
            f"Missing {meta_path}. Run scripts/training/crop_perspective_dataset.py first."
        )
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise SystemExit(f"Need {train_dir} and {val_dir}.")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    class_mapping: Dict[str, int] = meta["class_mapping"]

    _seed_all(args.seed)
    device = torch.device(args.device)

    tf_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tf_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = PerspectiveCropDataset(data_dir, meta["train_samples"], tf_train)
    val_ds = PerspectiveCropDataset(data_dir, meta["val_samples"], tf_val)

    num_classes = len(class_mapping)
    print("class_mapping:", class_mapping)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    weights = models.ResNet18_Weights.IMAGENET1K_V1
    m = models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    m = m.to(device)

    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        m.train()
        running = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = m(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        train_loss = running / max(n, 1)

        m.eval()
        correct = 0
        total = 0
        vloss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = m(x)
                vloss += crit(logits, y).item() * x.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += x.size(0)

        val_acc = correct / max(total, 1)
        val_loss = vloss / max(total, 1)
        print(
            f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

    ckpt_path = data_dir / "resnet18_baseline.pt"
    torch.save(
        {
            "model_state_dict": m.state_dict(),
            "class_mapping": class_mapping,
            "metadata_path": str(meta_path),
        },
        ckpt_path,
    )
    print(f"Saved {ckpt_path}")


if __name__ == "__main__":
    main()
