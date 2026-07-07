#!/usr/bin/env python3
"""Сравнение предсказаний с правками аналитиков (out_for_katya_new.gpkg)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import geopandas as gpd

from web_app.classify_service import apply_classification
from web_app.gpkg_io import classify_from_description, load_parcel_gdf, parcel_panorama_available


def _acc(pred: list[str], corr: list[str]) -> float:
    if not pred:
        return 0.0
    return sum(p == c for p, c in zip(pred, corr)) / len(pred)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpkg", type=Path, default=ROOT / "results" / "out_for_katya_new.gpkg")
    ap.add_argument("--n", type=int, default=50, help="ML sample size (with panorama)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "analyst_compare.json")
    args = ap.parse_args()

    gdf = load_parcel_gdf(args.gpkg)
    corr_cls = gdf["class_corrected_ui"].astype(str).str.strip()
    corr_sub = gdf["subclass_corrected_ui"].astype(str).str.strip()
    valid = corr_cls.ne("") & corr_cls.ne("Категории нет") & corr_cls.ne("nan")

    # Heuristic full set
    h_cls, h_sub = [], []
    for desc in gdf.loc[valid, "generated_land_use"].astype(str):
        c, s = classify_from_description(desc)
        h_cls.append(c)
        h_sub.append(s)

    report: dict = {
        "gpkg": str(args.gpkg),
        "total": int(len(gdf)),
        "evaluated": int(valid.sum()),
        "no_category": int((corr_cls == "Категории нет").sum()),
        "heuristic": {
            "class_acc": round(_acc(h_cls, corr_cls[valid].tolist()), 4),
            "subclass_acc": round(_acc(h_sub, corr_sub[valid].tolist()), 4),
        },
    }

    # ML/smart sample — только строки с URL панорам (быстрый отбор)
    has_urls = gdf["photos_ui"].fillna("").astype(str).str.contains("ll=", case=False)
    candidates = [i for i in gdf.index[valid & has_urls]]
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    sample_idx = candidates[: args.n]

    if sample_idx:
        sub = gdf.loc[sample_idx].copy()
        apply_classification(sub, mode="ml", overwrite=True, max_rows=len(sub))
        ml_cls = sub["class_ui"].astype(str).tolist()
        ml_sub = sub["subclass_ui"].astype(str).tolist()
        c_cls = corr_cls.loc[sample_idx].tolist()
        c_sub = corr_sub.loc[sample_idx].tolist()
        report["ml_sample"] = {
            "n": len(sample_idx),
            "mode": "ml",
            "class_acc": round(_acc(ml_cls, c_cls), 4),
            "subclass_acc": round(_acc(ml_sub, c_sub), 4),
        }
        errors = []
        for i, pos in enumerate(sample_idx):
            if ml_cls[i] != c_cls[i]:
                errors.append({
                    "oid": int(gdf.iloc[pos]["oid"]),
                    "desc": str(gdf.iloc[pos]["generated_land_use"])[:80],
                    "pred": ml_cls[i],
                    "corrected": c_cls[i],
                })
        report["ml_class_errors"] = errors[:20]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
