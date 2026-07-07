#!/usr/bin/env python3
"""
Fine-tune OpenCLIP image tower + linear head on perspective crops.

Modes:
  --finetune-last-n 0   : linear probe only (frozen ViT)
  --finetune-last-n 4-8 : also train last N transformer blocks (+ ln_final)

Example:
  python3 scripts/training/train_clip_classifier.py --finetune-last-n 6 --epochs 60 \\
      --lr-head 3e-3 --lr-backbone 5e-6 --batch-size 24 --label-smoothing 0.08
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import open_clip
import torch
import torch.nn as nn
from PIL import Image
try:
    from torch.amp import GradScaler, autocast
    _AMP_DEVICE = "cuda"
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

    _AMP_DEVICE = None  # type: ignore[misc,assignment]
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _class_weights(class_mapping: Dict[str, int], train_rows: List[dict], device: torch.device) -> torch.Tensor:
    counts = np.zeros(len(class_mapping), dtype=np.float64)
    for s in train_rows:
        counts[int(s["class_id"])] += 1
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (len(class_mapping) * counts)
    return torch.tensor(w, dtype=torch.float32, device=device)


class CropDataset(Dataset):
    def __init__(
        self,
        root: Path,
        rows: List[dict],
        preprocess_branches,
        train: bool,
        aug_strong: bool,
    ):
        self.root = root
        self.rows = rows
        self.pb = preprocess_branches
        self.train = train
        self.aug_strong = aug_strong

        if train:
            ops = [transforms.RandomHorizontalFlip(p=0.5)]
            if aug_strong:
                ops.extend(
                    [
                        transforms.ColorJitter(0.35, 0.35, 0.35, 0.12),
                        transforms.RandomAffine(degrees=8, translate=(0.06, 0.06), scale=(0.92, 1.08)),
                    ]
                )
            else:
                ops.append(transforms.ColorJitter(0.2, 0.2, 0.2, 0.08))
            self.before_tensor = transforms.Compose(ops)
        else:
            self.before_tensor = None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        s = self.rows[idx]
        path = self.root / s["image_path"]
        img = Image.open(path).convert("RGB")
        if self.train and self.before_tensor is not None:
            img = self.before_tensor(img)
        t = self.pb(img)
        return t, int(s["class_id"])


def _set_trainable_backbone(model: nn.Module, finetune_last_n: int) -> None:
    vis = model.visual
    for p in vis.parameters():
        p.requires_grad = False
    if finetune_last_n <= 0:
        return
    blocks = vis.transformer.resblocks
    for b in blocks[-finetune_last_n:]:
        for p in b.parameters():
            p.requires_grad = True
    for name in ("ln_final", "ln_post"):
        if hasattr(vis, name):
            mod = getattr(vis, name)
            if isinstance(mod, nn.Identity):
                continue
            for p in mod.parameters():
                p.requires_grad = True
    if getattr(vis, "proj", None) is not None and hasattr(vis.proj, "parameters"):
        for p in vis.proj.parameters():
            p.requires_grad = True


@torch.no_grad()
def _accuracy(
    model: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    tta_flip: bool,
    use_amp: bool,
) -> Tuple[float, np.ndarray]:
    model.eval()
    head.eval()
    nc = head.weight.shape[0]
    cm = np.zeros((nc, nc), dtype=np.int64)
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        def _forward_batch(xx: torch.Tensor) -> torch.Tensor:
            return head(model.encode_image(xx, normalize=False))

        if device.type == "cuda" and use_amp:
            with autocast("cuda"):
                logits = _forward_batch(x)
                if tta_flip:
                    xf = torch.flip(x, dims=(3,))
                    logits = logits + _forward_batch(xf)
                    logits = logits * 0.5
        else:
            logits = _forward_batch(x)
            if tta_flip:
                xf = torch.flip(x, dims=(3,))
                logits = (logits + _forward_batch(xf)) * 0.5
        pred = logits.argmax(dim=1)
        for t, p in zip(y.view(-1), pred.view(-1)):
            cm[t.item(), p.item()] += 1
            if t == p:
                correct += 1
            total += 1
    return correct / max(total, 1), cm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/ml_perspective"))
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--finetune-last-n", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr-head", type=float, default=3e-3)
    parser.add_argument("--lr-backbone", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.08)
    parser.add_argument("--aug-strong", action="store_true", help="Stronger color/affine augmentations")
    parser.add_argument(
        "--val-tta",
        action="store_true",
        help="Validation: average logits with horizontal flip (can bump val acc a few points)",
    )
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    data_dir = args.data_dir if args.data_dir.is_absolute() else project_root / args.data_dir
    meta_path = args.metadata or (data_dir / "metadata_perspective.json")
    if not meta_path.is_file():
        meta_path = project_root / meta_path
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    class_mapping: Dict[str, int] = meta["class_mapping"]
    num_classes = len(class_mapping)
    train_rows = meta["train_samples"]
    val_rows = meta["val_samples"]

    _seed_all(args.seed)
    device = torch.device(args.device)

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device
    )
    dim = model.visual.output_dim
    head = nn.Linear(dim, num_classes, bias=True).to(device)
    nn.init.normal_(head.weight, std=0.02)
    nn.init.zeros_(head.bias)

    _set_trainable_backbone(model, args.finetune_last_n)

    params: List[Dict[str, Any]] = [
        {"params": head.parameters(), "lr": args.lr_head, "name": "head"},
    ]
    if args.finetune_last_n > 0:
        bt = [p for p in model.visual.parameters() if p.requires_grad]
        if bt:
            params.append({"params": bt, "lr": args.lr_backbone, "name": "backbone"})

    opt = torch.optim.AdamW(params, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=0.0
    )

    w = None if args.no_class_weights else _class_weights(class_mapping, train_rows, device)

    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=args.label_smoothing)
    use_amp = device.type == "cuda"
    scaler_device = "cuda" if device.type == "cuda" else "cpu"
    scaler = GradScaler(scaler_device, enabled=use_amp and device.type == "cuda")

    train_ds = CropDataset(data_dir, train_rows, preprocess, train=True, aug_strong=args.aug_strong)
    val_ds = CropDataset(data_dir, val_rows, preprocess, train=False, aug_strong=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    best_acc = 0.0
    best_ep = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        head.train()
        running = 0.0
        n_seen = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if device.type == "cuda" and use_amp:
                with autocast("cuda"):
                    feat = model.encode_image(x, normalize=False)
                    logits = head(feat)
                    loss = loss_fn(logits, y)
            else:
                feat = model.encode_image(x, normalize=False)
                logits = head(feat)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                list(head.parameters()) + [p for p in model.visual.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            scaler.step(opt)
            scaler.update()
            running += loss.item() * x.size(0)
            n_seen += x.size(0)
        scheduler.step()

        val_acc, cm = _accuracy(
            model, head, val_loader, device, tta_flip=args.val_tta, use_amp=use_amp
        )
        train_loss = running / max(n_seen, 1)
        print(
            f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}  "
            f"lr_head={opt.param_groups[0]['lr']:.2e}"
            + (
                f"  lr_bb={opt.param_groups[1]['lr']:.2e}"
                if len(opt.param_groups) > 1
                else ""
            )
        )
        if val_acc > best_acc:
            best_acc = val_acc
            best_ep = epoch
            best_state = {
                "model_visual_state_dict": model.visual.state_dict(),
                "head_state_dict": head.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "class_mapping": class_mapping,
                "args": vars(args),
                "confusion_matrix_val": cm.tolist(),
            }

    out_path = data_dir / "clip_finetuned_best.pt"
    if best_state:
        best_state["best_val_acc"] = best_acc
        best_state["best_epoch"] = best_ep
        torch.save(best_state, out_path)
        print(f"\nBest val_acc={best_acc:.4f} at epoch {best_ep}"
              f"\nSaved {out_path}")
    else:
        print("No checkpoint saved.")


if __name__ == "__main__":
    main()
