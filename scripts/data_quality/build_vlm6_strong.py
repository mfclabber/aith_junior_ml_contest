#!/usr/bin/env python3
"""Сильный 6-классовый датасет: stratified object-split + баланс train.

Исправляет:
- val перекошен в active_urban (195/239) → stratified split по классу объекта
- переобучение на дубликатах → sample_weight в metadata

Выход: data/ml_perspective/metadata_perspective_vlm6_strong.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

CLASSES_6 = [
    "natural_areas",
    "low_density_degraded",
    "underused_infrastructure",
    "frozen_construction",
    "active_construction",
    "active_urban",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=Path("data/vlm_relabel/labels.jsonl"))
    ap.add_argument("--clean-meta", type=Path,
                    default=Path("data/ml_perspective/metadata_perspective_clean.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/ml_perspective/metadata_perspective_vlm6_strong.json"))
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--min-val-per-class", type=int, default=12)
    ap.add_argument("--train-min-per-class", type=int, default=100)
    ap.add_argument("--train-max-urban", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    labels_p = root / args.labels if not args.labels.is_absolute() else args.labels
    clean_p = root / args.clean_meta if not args.clean_meta.is_absolute() else args.clean_meta
    out_p = root / args.out if not args.out.is_absolute() else args.out
    random.seed(args.seed)

    clean = json.loads(clean_p.read_text(encoding="utf-8"))
    rec_by_path = {s["image_path"]: s for k in ("train_samples", "val_samples") for s in clean[k]}
    class_mapping = {c: i for i, c in enumerate(CLASSES_6)}

    labeled: list[dict] = []
    for line in labels_p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("vlm_confident", True):
            continue
        if r.get("vlm_label") not in class_mapping:
            continue
        rec = rec_by_path.get(r["image_path"])
        if rec is None:
            continue
        s = dict(rec)
        s["class_name"] = r["vlm_label"]
        s["class_id"] = class_mapping[r["vlm_label"]]
        s["osm_label"] = r.get("osm_label")
        labeled.append(s)

    # Класс объекта = метка первого кропа (стабильно при object-split)
    obj_label: dict[str, str] = {}
    for s in labeled:
        oid = s["object_id"]
        if oid not in obj_label:
            obj_label[oid] = s["class_name"]

    by_cls_objs: dict[str, list[str]] = defaultdict(list)
    for oid, cls in obj_label.items():
        by_cls_objs[cls].append(oid)

    val_objs: set[str] = set()
    train_objs: set[str] = set()
    for cls in CLASSES_6:
        objs = sorted(by_cls_objs.get(cls, []))
        random.shuffle(objs)
        if not objs:
            continue
        n_val = max(args.min_val_per_class, int(len(objs) * args.val_ratio))
        n_val = min(n_val, len(objs) - 1) if len(objs) > 1 else 1
        val_objs.update(objs[:n_val])
        train_objs.update(objs[n_val:])

    train_raw, val_samples = [], []
    for s in labeled:
        (val_samples if s["object_id"] in val_objs else train_raw).append(s)

    # Баланс train: oversample minority, cap urban; вес = 1/число копий
    by_cls: dict[str, list[dict]] = defaultdict(list)
    for s in train_raw:
        by_cls[s["class_name"]].append(s)

    train_samples: list[dict] = []
    dup_count: Counter[str] = Counter()
    for cls in CLASSES_6:
        items = list(by_cls.get(cls, []))
        if not items:
            continue
        random.shuffle(items)
        if cls == "active_urban" and len(items) > args.train_max_urban:
            items = items[: args.train_max_urban]
        target = args.train_min_per_class
        pool = list(items)
        while len(pool) < target and items:
            pool.extend(items)
        pool = pool[:target] if len(pool) > target else pool
        for s in pool:
            dup_count[s["image_path"]] += 1
            ns = dict(s)
            ns["sample_weight"] = 1.0  # обновим ниже
            train_samples.append(ns)

    # нормализовать веса (меньше вес у дубликатов)
    for s in train_samples:
        s["sample_weight"] = 1.0 / dup_count[s["image_path"]]

    random.shuffle(train_samples)

    def stats(rows: list[dict]) -> dict[str, int]:
        c = Counter(r["class_name"] for r in rows)
        return {k: c.get(k, 0) for k in CLASSES_6}

    meta = {
        "class_mapping": class_mapping,
        "dataset_info": {
            "source": "vlm6_strong_stratified_balanced",
            "taxonomy": "6class_utt_vlm_content",
            "split_policy": "stratified_by_object_class",
            "train_min_per_class": args.train_min_per_class,
            "train_max_urban": args.train_max_urban,
            "min_val_per_class": args.min_val_per_class,
            "total_train": len(train_samples),
            "total_val": len(val_samples),
        },
        "train_statistics": stats(train_samples),
        "val_statistics": stats(val_samples),
        "train_samples": train_samples,
        "val_samples": val_samples,
    }
    out_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"train={len(train_samples)} val={len(val_samples)}")
    print(f"train_stats={stats(train_samples)}")
    print(f"val_stats={stats(val_samples)}")
    print(f"Saved {out_p}")


if __name__ == "__main__":
    main()
