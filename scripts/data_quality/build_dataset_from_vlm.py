#!/usr/bin/env python3
"""Собрать чистый датасет из VLM-меток (content-based).

Берёт метки от VLM-судьи (scripts/data_quality/vlm_relabel.py --taxonomy 3),
оставляет только уверенные, джойнит с кроп-записями из чистого metadata и
пересобирает train/val с разбивкой ПО ОБЪЕКТАМ (без утечки).

Выход: data/ml_perspective/metadata_perspective_vlm3.json

Пример:
  ./venv/bin/python scripts/data_quality/build_dataset_from_vlm.py \
      --labels data/vlm_relabel/labels3.jsonl \
      --clean-meta data/ml_perspective/metadata_perspective_clean.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

CLASSES_3 = ["natural_areas", "construction", "built_up"]

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
    ap.add_argument("--labels", type=Path, default=Path("data/vlm_relabel/labels3.jsonl"))
    ap.add_argument("--clean-meta", type=Path,
                    default=Path("data/ml_perspective/metadata_perspective_clean.json"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--taxonomy", choices=("3", "6"), default="3")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--only-confident", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    labels_p = args.labels if args.labels.is_absolute() else root / args.labels
    clean_p = args.clean_meta if args.clean_meta.is_absolute() else root / args.clean_meta

    class_names = CLASSES_6 if args.taxonomy == "6" else CLASSES_3
    default_out = (
        "data/ml_perspective/metadata_perspective_vlm6.json"
        if args.taxonomy == "6"
        else "data/ml_perspective/metadata_perspective_vlm3.json"
    )
    out_p = args.out if args.out else root / default_out
    if not out_p.is_absolute():
        out_p = root / out_p

    clean = json.loads(clean_p.read_text(encoding="utf-8"))
    rec_by_path = {}
    for key in ("train_samples", "val_samples"):
        for s in clean[key]:
            rec_by_path[s["image_path"]] = s

    class_mapping = {c: i for i, c in enumerate(class_names)}
    labeled = []
    skipped = Counter()
    for line in labels_p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("vlm_label") not in class_mapping:
            skipped["bad_label"] += 1
            continue
        if args.only_confident and not r.get("vlm_confident", True):
            skipped["unsure"] += 1
            continue
        rec = rec_by_path.get(r["image_path"])
        if rec is None:
            skipped["no_crop_record"] += 1
            continue
        new = dict(rec)
        new["class_name"] = r["vlm_label"]
        new["class_id"] = class_mapping[r["vlm_label"]]
        new["osm_label"] = r.get("osm_label")
        labeled.append(new)

    # Разбивка по объектам (без утечки)
    by_obj = defaultdict(list)
    for s in labeled:
        by_obj[s["object_id"]].append(s)
    objs = sorted(by_obj)
    random.Random(args.seed).shuffle(objs)
    n_val = int(len(objs) * args.val_ratio)
    val_objs = set(objs[:n_val])

    train_samples, val_samples = [], []
    for obj, items in by_obj.items():
        (val_samples if obj in val_objs else train_samples).extend(items)

    def stats(rws):
        c = Counter(r["class_name"] for r in rws)
        return {k: c.get(k, 0) for k in class_names}

    taxonomy_tag = (
        "6class_utt_vlm_content"
        if args.taxonomy == "6"
        else "3class_natural_construction_builtup"
    )
    meta = {
        "class_mapping": class_mapping,
        "dataset_info": {
            "source": "vlm_relabel_qwen2.5-vl_content_based",
            "taxonomy": taxonomy_tag,
            "total_train": len(train_samples),
            "total_val": len(val_samples),
            "total": len(labeled),
            "val_ratio": args.val_ratio,
            "split_policy": "object_stratified",
            "train_val_object_overlap": 0,
        },
        "train_statistics": stats(train_samples),
        "val_statistics": stats(val_samples),
        "train_samples": train_samples,
        "val_samples": val_samples,
    }
    out_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    agree = sum(1 for s in labeled if s.get("osm_label") == s["class_name"])
    print(f"labeled={len(labeled)} skipped={dict(skipped)} agree_osm={agree} ({agree/max(len(labeled),1):.1%})")
    print(f"train={len(train_samples)} val={len(val_samples)}")
    print(f"train_stats={stats(train_samples)}")
    print(f"val_stats={stats(val_samples)}")
    print(f"Saved {out_p}")


if __name__ == "__main__":
    main()
