#!/usr/bin/env python3
"""Объединить vlm6_oss + analyst metadata для дообучения probe."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path("data/ml_perspective/metadata_perspective_vlm6_oss.json"))
    ap.add_argument("--analyst", type=Path, default=Path("data/ml_perspective/metadata_analyst.json"))
    ap.add_argument("--out", type=Path, default=Path("data/ml_perspective/metadata_vlm6_analyst.json"))
    ap.add_argument("--analyst-weight", type=int, default=3, help="дубликаты analyst train samples")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    base = json.loads((root / args.base).read_text(encoding="utf-8"))
    analyst = json.loads((root / args.analyst).read_text(encoding="utf-8"))

    train = list(base["train_samples"])
    val = list(base["val_samples"])
    for _ in range(max(1, args.analyst_weight)):
        train.extend(analyst["train_samples"])
    val.extend(analyst["val_samples"])

    out = {
        "class_mapping": base["class_mapping"],
        "dataset_info": {
            "source": "vlm6_oss + analyst_gpkg",
            "total_train": len(train),
            "total_val": len(val),
            "analyst_weight": args.analyst_weight,
        },
        "train_statistics": dict(Counter(s["class_name"] for s in train)),
        "val_statistics": dict(Counter(s["class_name"] for s in val)),
        "train_samples": train,
        "val_samples": val,
    }
    out_p = root / args.out
    out_p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"merged train={len(train)} val={len(val)} -> {out_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
