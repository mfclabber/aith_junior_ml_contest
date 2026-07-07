#!/usr/bin/env python3
"""Собрать кропы панорам + метки из GPKG с правками аналитиков.

  ./venv/bin/python scripts/data_quality/build_analyst_dataset.py \
      --gpkg results/out_for_katya_new.gpkg --max-samples 250 --prioritize-errors
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import geopandas as gpd

from scripts.data_quality.analyst_class_map import analyst_class_to_key
from web_app.classify_service import classify_from_description
from web_app.gpkg_io import (
    boundary_view_points,
    bearing_geographic_deg,
    load_parcel_gdf,
    parse_photo_coords,
)
from web_app.panorama_crop import render_context_crop_jpeg

CLASSES_6 = [
    "natural_areas",
    "low_density_degraded",
    "underused_infrastructure",
    "frozen_construction",
    "active_construction",
    "active_urban",
]


def _pick_view_point(geometry, photos_ui: str):
    coords = parse_photo_coords(photos_ui)
    if coords:
        lon, lat = coords[0]
        tgt = geometry.representative_point()
        bearing = bearing_geographic_deg(lat, lon, float(tgt.y), float(tgt.x))
        return lat, lon, bearing
    points = boundary_view_points(geometry)
    if not points:
        return None
    vp = points[0]
    return float(vp["lat"]), float(vp["lon"]), float(vp.get("bearing_deg") or 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpkg", type=Path, default=ROOT / "results" / "out_for_katya_new.gpkg")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "ml_perspective" / "analyst_crops")
    ap.add_argument("--out-meta", type=Path,
                    default=ROOT / "data" / "ml_perspective" / "metadata_analyst.json")
    ap.add_argument("--max-samples", type=int, default=250)
    ap.add_argument("--min-per-class", type=int, default=40,
                    help="минимум кандидатов на класс при стратификации")
    ap.add_argument("--skip-existing", action="store_true", default=True,
                    help="не перекачивать уже сохранённые кропы")
    ap.add_argument("--prioritize-errors", action="store_true", default=True)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    gdf = load_parcel_gdf(args.gpkg if args.gpkg.is_absolute() else ROOT / args.gpkg)
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_meta = args.out_meta if args.out_meta.is_absolute() else ROOT / args.out_meta
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for pos in range(len(gdf)):
        corr_cls = str(gdf.iloc[pos]["class_corrected_ui"] or "").strip()
        key = analyst_class_to_key(corr_cls)
        if not key:
            continue
        photos = str(gdf.iloc[pos]["photos_ui"] or "")
        if not parse_photo_coords(photos) and not boundary_view_points(gdf.geometry.iloc[pos]):
            continue
        desc = str(gdf.iloc[pos]["generated_land_use"] or "")
        pred_cls = classify_from_description(desc)[0]
        is_err = pred_cls != corr_cls
        rows.append({
            "pos": pos,
            "oid": int(gdf.iloc[pos]["oid"]),
            "class_key": key,
            "is_error": is_err,
        })

    if args.prioritize_errors:
        by_class: dict[str, list] = {c: [] for c in CLASSES_6}
        for r in rows:
            by_class.setdefault(r["class_key"], []).append(r)
        rng = random.Random(args.seed)
        picked: list[dict] = []
        per_class = max(args.min_per_class, args.max_samples // len(CLASSES_6))
        for cls in CLASSES_6:
            pool = by_class.get(cls, [])
            pool.sort(key=lambda x: (not x["is_error"], rng.random()))
            picked.extend(pool[:per_class])
        if len(picked) < args.max_samples:
            rest = [r for r in rows if r not in picked]
            rng.shuffle(rest)
            picked.extend(rest[: args.max_samples - len(picked)])
        picked = picked[: args.max_samples]
    else:
        random.Random(args.seed).shuffle(rows)
        picked = rows[: args.max_samples]

    rng = random.Random(args.seed)
    rng.shuffle(picked)
    n_val = max(1, int(len(picked) * args.val_ratio))
    val_oids = {r["oid"] for r in picked[:n_val]}

    class_mapping = {c: i for i, c in enumerate(CLASSES_6)}
    train_samples, val_samples = [], []
    saved, failed = 0, 0

    out_root = out_dir.parent  # data/ml_perspective
    existing_train, existing_val = [], []

    for item in picked:
        pos = item["pos"]
        oid = item["oid"]
        key = item["class_key"]
        split = "val" if oid in val_oids else "train"
        rel = f"analyst_crops/{split}/{key}/analyst_{oid}.jpg"
        dest = out_root / rel

        if args.skip_existing and dest.is_file():
            sample = {
                "image_path": rel,
                "class_name": key,
                "class_id": class_mapping[key],
                "object_id": f"analyst_{oid}",
                "source": "analyst_gpkg_corrected",
                "oid": oid,
            }
            (existing_val if split == "val" else existing_train).append(sample)
            saved += 1
            continue

        geom = gdf.geometry.iloc[pos]
        photos = str(gdf.iloc[pos]["photos_ui"] or "")
        pt = _pick_view_point(geom, photos)
        if not pt:
            failed += 1
            continue
        lat, lon, bearing = pt
        jpeg = render_context_crop_jpeg(lat, lon, heading_deg=bearing, out_width=896, out_height=672)
        if not jpeg:
            failed += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(jpeg)
        saved += 1
        sample = {
            "image_path": rel,
            "class_name": key,
            "class_id": class_mapping[key],
            "object_id": f"analyst_{oid}",
            "source": "analyst_gpkg_corrected",
            "oid": oid,
        }
        (val_samples if split == "val" else train_samples).append(sample)

        if saved % 25 == 0:
            print(f"  ... {saved} кропов", flush=True)

    train_samples = existing_train + train_samples
    val_samples = existing_val + val_samples

    meta = {
        "class_mapping": class_mapping,
        "dataset_info": {
            "source": "out_for_katya_new.gpkg corrected labels",
            "total_train": len(train_samples),
            "total_val": len(val_samples),
            "render_failed": failed,
            "picked": len(picked),
        },
        "train_samples": train_samples,
        "val_samples": val_samples,
    }
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"saved={saved} failed={failed} train={len(train_samples)} val={len(val_samples)}")
    print(f"train dist: {Counter(s['class_name'] for s in train_samples)}")
    print(f"val dist:   {Counter(s['class_name'] for s in val_samples)}")
    print(f"metadata -> {out_meta}")
    return 0 if saved > 20 else 1


if __name__ == "__main__":
    raise SystemExit(main())
