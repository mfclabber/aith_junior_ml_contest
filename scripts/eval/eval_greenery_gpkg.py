#!/usr/bin/env python3
"""Оценка ML на hard-case «зелень во дворах» (natural vs urban)."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from web_app.classify_service import apply_classification
from web_app.gpkg_io import load_parcel_gdf, parse_photo_coords


def _acc(pred: list[str], corr: list[str]) -> float:
    if not pred:
        return 0.0
    return sum(p == c for p, c in zip(pred, corr)) / len(pred)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpkg", type=Path, default=ROOT / "results" / "out_for_katya_new.gpkg")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=ROOT / "results/greenery_eval.json")
    args = ap.parse_args()

    gdf = load_parcel_gdf(args.gpkg)
    desc = gdf["generated_land_use"].astype(str)
    corr = gdf["class_corrected_ui"].astype(str).str.strip()
    mask = (
        desc.str.contains("зелень", case=False)
        & desc.str.contains("двор", case=False)
        & corr.isin(["Природные территории", "Активные городские территории"])
    )
    has_pano = gdf["photos_ui"].fillna("").astype(str).apply(
        lambda s: bool(parse_photo_coords(s))
    )
    candidates = [i for i in gdf.index[mask & has_pano]]
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    sample_idx = candidates[: args.n]

    sub = gdf.loc[sample_idx].copy()
    apply_classification(sub, mode="ml", overwrite=True, max_rows=len(sub))
    pred = sub["class_ui"].astype(str).tolist()
    truth = corr.loc[sample_idx].tolist()

    report = {
        "subset": "greenery_courtyard_natural_vs_urban",
        "n": len(sample_idx),
        "class_acc": round(_acc(pred, truth), 4),
        "natural_correct": sum(
            1 for p, t in zip(pred, truth) if t == "Природные территории" and p == t
        ),
        "urban_correct": sum(
            1 for p, t in zip(pred, truth) if t == "Активные городские территории" and p == t
        ),
        "natural_total": sum(1 for t in truth if t == "Природные территории"),
        "urban_total": sum(1 for t in truth if t == "Активные городские территории"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
