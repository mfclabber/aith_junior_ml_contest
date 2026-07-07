#!/usr/bin/env python3
"""Метадата-варианты под правки аналитиков: val = только analyst val."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DM = ROOT / "data" / "ml_perspective"


def main() -> None:
    base = json.loads((DM / "metadata_perspective_vlm6_oss.json").read_text(encoding="utf-8"))
    analyst = json.loads((DM / "metadata_analyst.json").read_text(encoding="utf-8"))
    val = analyst["val_samples"]

    for w in (3, 8):
        train = list(base["train_samples"])
        for _ in range(w):
            train.extend(analyst["train_samples"])
        out = {
            "class_mapping": base["class_mapping"],
            "dataset_info": {"source": f"vlm6_oss + analyst x{w}, val=analyst_only"},
            "train_statistics": dict(Counter(s["class_name"] for s in train)),
            "val_statistics": dict(Counter(s["class_name"] for s in val)),
            "train_samples": train,
            "val_samples": val,
        }
        p = DM / f"metadata_analyst_target_w{w}.json"
        p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        print(f"w={w}: train={len(train)} val={len(val)}")

    # чистый analyst train с oversample меньшинств (cap x5)
    train_a = list(analyst["train_samples"])
    cnt = Counter(s["class_name"] for s in train_a)
    mx = max(cnt.values())
    extra = []
    for s in train_a:
        k = max(1, min(5, mx // max(1, cnt[s["class_name"]]))) - 1
        extra.extend([s] * k)
    train_bal = train_a + extra
    out = {
        "class_mapping": base["class_mapping"],
        "dataset_info": {"source": "analyst_only_oversampled"},
        "train_statistics": dict(Counter(s["class_name"] for s in train_bal)),
        "val_statistics": dict(Counter(s["class_name"] for s in val)),
        "train_samples": train_bal,
        "val_samples": val,
    }
    (DM / "metadata_analyst_bal.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"analyst_bal: train={len(train_bal)} dist={dict(Counter(s['class_name'] for s in train_bal))}")


if __name__ == "__main__":
    main()
