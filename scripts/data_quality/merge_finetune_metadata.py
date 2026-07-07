#!/usr/bin/env python3
"""Объединить vlm6_oss + analyst + greenery для fine-tune probe."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _extend_weighted(dst: list, samples: list, repeats: int, default_weight: float = 1.0) -> None:
    for _ in range(max(1, repeats)):
        for s in samples:
            row = dict(s)
            row.setdefault("sample_weight", default_weight)
            dst.append(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path("data/ml_perspective/metadata_perspective_vlm6_oss.json"))
    ap.add_argument("--analyst", type=Path, default=Path("data/ml_perspective/metadata_analyst.json"))
    ap.add_argument("--greenery", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("data/ml_perspective/metadata_vlm6_finetune.json"))
    ap.add_argument("--analyst-weight", type=int, default=2,
                    help="0 = не добавлять analyst train")
    ap.add_argument("--greenery-weight", type=int, default=3)
    ap.add_argument("--base-val-only", action="store_true", default=True,
                    help="val только из base (research val), без analyst/greenery")
    ap.add_argument("--no-base-val-only", dest="base_val_only", action="store_false")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    base = json.loads((root / args.base).read_text(encoding="utf-8"))
    analyst = json.loads((root / args.analyst).read_text(encoding="utf-8"))

    train: list[dict] = []
    for s in base["train_samples"]:
        row = dict(s)
        row.setdefault("sample_weight", 1.0)
        train.append(row)
    if args.analyst_weight > 0:
        _extend_weighted(train, analyst["train_samples"], args.analyst_weight, default_weight=2.0)

    greenery_n = 0
    if args.greenery is not None:
        gpath = args.greenery if args.greenery.is_absolute() else root / args.greenery
        if gpath.is_file():
            greenery = json.loads(gpath.read_text(encoding="utf-8"))
            _extend_weighted(
                train, greenery["train_samples"], args.greenery_weight, default_weight=4.0
            )
            greenery_n = len(greenery["train_samples"]) * max(1, args.greenery_weight)

    val = list(base["val_samples"])
    if not args.base_val_only:
        val.extend(analyst["val_samples"])
        if args.greenery and (root / args.greenery).is_file():
            greenery = json.loads((root / args.greenery).read_text(encoding="utf-8"))
            val.extend(greenery["val_samples"])

    out = {
        "class_mapping": base["class_mapping"],
        "dataset_info": {
            "source": "vlm6_oss + analyst + greenery finetune",
            "total_train": len(train),
            "total_val": len(val),
            "analyst_weight": args.analyst_weight,
            "greenery_weight": args.greenery_weight,
            "greenery_train_effective": greenery_n,
            "base_val_only": args.base_val_only,
        },
        "train_statistics": dict(Counter(s["class_name"] for s in train)),
        "val_statistics": dict(Counter(s["class_name"] for s in val)),
        "train_samples": train,
        "val_samples": val,
    }
    out_p = root / args.out
    out_p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"merged train={len(train)} val={len(val)} greenery_eff={greenery_n} -> {out_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
