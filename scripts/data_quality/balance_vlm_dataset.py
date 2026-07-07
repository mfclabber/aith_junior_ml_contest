#!/usr/bin/env python3
"""Балансировка VLM6 датасета: cap majority + oversample minority для 6 классов."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


CLASSES_6 = [
    "natural_areas",
    "low_density_degraded",
    "underused_infrastructure",
    "frozen_construction",
    "active_construction",
    "active_urban",
]


def balance(
    rows: list[dict],
    *,
    max_majority: int,
    min_per_class: int,
    oversample_factor: int,
) -> list[dict]:
    by_cls: dict[str, list[dict]] = {c: [] for c in CLASSES_6}
    for r in rows:
        by_cls.setdefault(r["class_name"], []).append(r)

    out: list[dict] = []
    for cls in CLASSES_6:
        items = list(by_cls.get(cls, []))
        if not items:
            continue
        random.shuffle(items)
        if cls == "active_urban" and len(items) > max_majority:
            items = items[:max_majority]
        # oversample minority до min_per_class
        while len(items) < min_per_class and items:
            need = min_per_class - len(items)
            extra = (items * oversample_factor)[:need]
            items.extend(extra)
        out.extend(items)
    random.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-meta", type=Path, default=Path("data/ml_perspective/metadata_perspective_vlm6.json"))
    ap.add_argument("--out", type=Path, default=Path("data/ml_perspective/metadata_perspective_vlm6_oss.json"))
    ap.add_argument("--max-majority", type=int, default=120)
    ap.add_argument("--min-per-class", type=int, default=80)
    ap.add_argument("--oversample-factor", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    inp = args.in_meta if args.in_meta.is_absolute() else root / args.in_meta
    out = args.out if args.out.is_absolute() else root / args.out
    random.seed(args.seed)

    meta = json.loads(inp.read_text(encoding="utf-8"))
    train = balance(
        meta["train_samples"],
        max_majority=args.max_majority,
        min_per_class=args.min_per_class,
        oversample_factor=args.oversample_factor,
    )
    meta["train_samples"] = train
    dup = Counter(s["image_path"] for s in train)
    for s in train:
        s["sample_weight"] = 1.0 / dup[s["image_path"]]
    meta["dataset_info"]["total_train"] = len(train)
    meta["dataset_info"]["balanced_oversample"] = {
        "max_majority": args.max_majority,
        "min_per_class": args.min_per_class,
        "oversample_factor": args.oversample_factor,
    }
    meta["train_statistics"] = dict(Counter(r["class_name"] for r in train))
    out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"train={len(train)} stats={meta['train_statistics']}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
