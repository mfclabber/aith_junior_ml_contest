#!/usr/bin/env python3
"""Аудит датасета: баланс классов, bearing, дубликаты, битые файлы."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def audit_ml_metadata(meta_path: Path) -> dict:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    issues: dict[str, list] = defaultdict(list)
    stats = {
        "null_bearing": 0,
        "close_target_m": 0,
        "missing_source": 0,
        "missing_crop": 0,
    }
    per_class = Counter()

    for split in ("train_samples", "val_samples"):
        for s in meta.get(split, []):
            per_class[s["class_name"]] += 1
            cr = s.get("classification_region") or {}
            if cr.get("bearing_geographic_deg") is None:
                stats["null_bearing"] += 1
                issues["null_bearing"].append(s.get("object_id"))
            if (cr.get("distance_m_approx") or 0) < 5:
                stats["close_target_m"] += 1
            src = Path(s.get("source_path", ""))
            if not src.is_file():
                stats["missing_source"] += 1
                issues["missing_source"].append(str(src))

    root = meta_path.parent
    for split in ("train_samples", "val_samples"):
        for s in meta.get(split, []):
            crop = root / s["image_path"]
            if not crop.is_file():
                stats["missing_crop"] += 1
                issues["missing_crop"].append(s["image_path"])

    return {
        "dataset_info": meta.get("dataset_info"),
        "per_class_total": dict(per_class),
        "stats": stats,
        "issue_samples": {k: v[:20] for k, v in issues.items()},
        "issue_counts": {k: len(v) for k, v in issues.items()},
    }


def audit_raw_dataset(dataset_dir: Path) -> dict:
    per_class_objects: dict[str, int] = {}
    seasons = Counter()
    no_panorama = []

    for cls_dir in sorted(dataset_dir.iterdir()):
        if not cls_dir.is_dir() or cls_dir.name.startswith("."):
            continue
        objs = [p for p in cls_dir.iterdir() if p.is_dir()]
        per_class_objects[cls_dir.name] = len(objs)
        for obj in objs:
            pans = list(obj.glob("panorama*.jpg"))
            if not pans:
                no_panorama.append(str(obj))
            for p in pans:
                name = p.stem.replace("panorama_", "")
                seasons[name or "default"] += 1

    return {
        "per_class_objects": per_class_objects,
        "total_objects": sum(per_class_objects.values()),
        "seasons": dict(seasons),
        "objects_without_panorama": no_panorama[:20],
        "objects_without_panorama_count": len(no_panorama),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit UTT dataset quality")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/dataset"))
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path("data/ml_perspective/metadata_perspective.json"),
    )
    parser.add_argument("--out", type=Path, default=Path("results/dataset_audit.json"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    dataset_dir = args.dataset_dir if args.dataset_dir.is_absolute() else root / args.dataset_dir
    meta_path = args.meta if args.meta.is_absolute() else root / args.meta
    out_path = args.out if args.out.is_absolute() else root / args.out

    report = {
        "raw": audit_raw_dataset(dataset_dir) if dataset_dir.is_dir() else {},
        "ml": audit_ml_metadata(meta_path) if meta_path.is_file() else {},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
