#!/usr/bin/env python3
"""
Zero-shot classification with OpenCLIP on perspective crops (val set).

Uses text prompts per UTT class (English descriptions tuned for street panoramas).
Compare logits_argmax to metadata class_id.

  pip install open-clip-torch
  python3 scripts/training/eval_clip_zeroshot.py \\
      --data-dir data/ml_perspective --batch-size 32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# English prompts: street panorama domain + taxonomy semantics (CLIP is EN-biased).
DEFAULT_PROMPTS_EN: Dict[str, str] = {
    "natural_areas": (
        "a street panorama of parks, forests, meadows, green recreational nature"
    ),
    "low_density_degraded": (
        "a street panorama of abandoned buildings, vandalism, degraded "
        "low-density dilapidated urban fabric"
    ),
    "underused_infrastructure": (
        "a street panorama of underused parking, empty storefronts, "
        "obsolete urban infrastructure zones"
    ),
    "frozen_construction": (
        "a street panorama of stalled construction sites, frozen unfinished buildings, "
        "inactive cranes"
    ),
    "active_construction": (
        "a street panorama of active construction sites, cranes, new buildings under construction"
    ),
    "active_urban": (
        "a street panorama of busy urban streets, dense mixed-use city life, "
        "commerce and intensive urban activity"
    ),
}


class PathsDataset(Dataset):
    def __init__(
        self,
        root: Path,
        rows: List[dict],
        preprocess,
    ):
        self.root = root
        self.rows = rows
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        s = self.rows[idx]
        path = self.root / s["image_path"]
        img = Image.open(path).convert("RGB")
        return self.preprocess(img), int(s["class_id"]), idx


def _ordered_class_names(class_mapping: Dict[str, int]) -> List[str]:
    return [k for k, _ in sorted(class_mapping.items(), key=lambda x: x[1])]


def _encode_texts(model, tokenizer, device: torch.device, prompts: Sequence[str]) -> torch.Tensor:
    with torch.no_grad():
        tokens = tokenizer(prompts).to(device)
        tf = model.encode_text(tokens)
        tf = tf / tf.norm(dim=-1, keepdim=True)
    return tf


@torch.no_grad()
def _evaluate(
    model,
    preprocess,
    tokenizer,
    data_root: Path,
    val_rows: List[dict],
    class_names: List[str],
    prompts: List[str],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[float, np.ndarray]:
    text_feat = _encode_texts(model, tokenizer, device, prompts)
    text_feat = text_feat.T  # (dim, num_classes) for logits = image @ text_feat

    ds = PathsDataset(data_root, val_rows, preprocess)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    correct = 0
    total = 0
    scale = model.logit_scale.exp()

    for batch_x, batch_y, _ in dl:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.numpy()
        img_feat = model.encode_image(batch_x)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        logits = scale * (img_feat @ text_feat)  # (B, C)
        pred = logits.argmax(dim=-1).cpu().numpy()
        for t, p in zip(batch_y, pred):
            cm[t, p] += 1
            if t == p:
                correct += 1
            total += 1

    acc = correct / max(total, 1)
    return acc, cm


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenCLIP zero-shot on val crops.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/ml_perspective"))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Defaults to <data-dir>/metadata_perspective.json",
    )
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--prompt-source",
        choices=("default_en", "taxonomy_ru"),
        default="default_en",
        help="default_en: built-in English prompts; taxonomy_ru: Russian labels from taxonomy.json",
    )
    parser.add_argument("--taxonomy", type=Path, default=Path("data/dataset/taxonomy.json"))
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    data_dir = args.data_dir if args.data_dir.is_absolute() else project_root / args.data_dir
    meta_path = args.metadata or (data_dir / "metadata_perspective.json")
    if not meta_path.is_file():
        meta_path = project_root / meta_path
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    class_mapping: Dict[str, int] = meta["class_mapping"]
    class_names = _ordered_class_names(class_mapping)
    val_rows = meta["val_samples"]

    if args.prompt_source == "taxonomy_ru":
        tax_path = args.taxonomy if args.taxonomy.is_absolute() else project_root / args.taxonomy
        tax = json.loads(tax_path.read_text(encoding="utf-8"))
        prompts = [tax[c]["label"] for c in class_names]
    else:
        prompts = [DEFAULT_PROMPTS_EN[c] for c in class_names]

    import open_clip

    device = torch.device(args.device)
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model,
        pretrained=args.pretrained,
        device=device,
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model)

    acc, cm = _evaluate(
        model,
        preprocess,
        tokenizer,
        data_dir,
        val_rows,
        class_names,
        prompts,
        device,
        args.batch_size,
        args.num_workers,
    )

    print(f"OpenCLIP zero-shot  model={args.model}  pretrained={args.pretrained}")
    print(f"prompt_source={args.prompt_source}")
    print(f"Val accuracy: {acc:.4f}  (n={cm.sum()})")
    print("\nConfusion matrix (rows=true, cols=pred):")
    short = [n[:14] for n in class_names]
    header = "".join(f"{s:>15}" for s in short)
    print("true \\ pred" + header)
    for i, name in enumerate(class_names):
        row = "".join(f"{cm[i, j]:>15}" for j in range(len(class_names)))
        print(f"{name[:14]:>14}" + row)


if __name__ == "__main__":
    main()
