#!/usr/bin/env python3
"""Кропы для hard-case «зелень во дворах»: natural vs urban (одинаковый текст)."""
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

from scripts.data_quality.analyst_class_map import analyst_class_to_key
from scripts.data_quality.build_analyst_dataset import _pick_view_point, CLASSES_6
from web_app.gpkg_io import load_parcel_gdf, parse_photo_coords
from web_app.panorama_crop import render_context_crop_jpeg

TARGET_KEYS = {"natural_areas", "active_urban"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpkg", type=Path, default=ROOT / "results" / "out_for_katya_new.gpkg")
    ap.add_argument("--out-meta", type=Path, default=ROOT / "data/ml_perspective/metadata_greenery.json")
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    gdf = load_parcel_gdf(args.gpkg)
    desc = gdf["generated_land_use"].astype(str)
    corr = gdf["class_corrected_ui"].astype(str).str.strip()
    mask = desc.str.contains("зелень", case=False) & desc.str.contains("двор", case=False)
    buckets: dict[str, list[int]] = {k: [] for k in TARGET_KEYS}
    for pos in range(len(gdf)):
        if not bool(mask.iloc[pos]):
            continue
        key = analyst_class_to_key(corr.iloc[pos])
        if key not in TARGET_KEYS:
            continue
        photos = str(gdf.iloc[pos]["photos_ui"] or "")
        if not parse_photo_coords(photos):
            continue
        buckets[key].append(pos)

    rng = random.Random(args.seed)
    picked: list[tuple[int, str]] = []
    for key in TARGET_KEYS:
        pool = buckets[key][:]
        rng.shuffle(pool)
        for pos in pool[: args.per_class]:
            picked.append((pos, key))

    rng.shuffle(picked)
    n_val = max(2, int(len(picked) * 0.15))
    val_pos = {p for p, _ in picked[:n_val]}
    class_mapping = {c: i for i, c in enumerate(CLASSES_6)}
    train_samples, val_samples = [], []
    saved = failed = 0
    out_root = ROOT / "data" / "ml_perspective"

    for pos, key in picked:
        oid = int(gdf.iloc[pos]["oid"])
        split = "val" if pos in val_pos else "train"
        rel = f"analyst_crops/{split}/{key}/greenery_{oid}.jpg"
        dest = out_root / rel
        if dest.is_file():
            sample = {
                "image_path": rel, "class_name": key, "class_id": class_mapping[key],
                "object_id": f"greenery_{oid}", "source": "greenery_hard",
                "sample_weight": 4.0, "oid": oid,
            }
            (val_samples if split == "val" else train_samples).append(sample)
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
            "image_path": rel, "class_name": key, "class_id": class_mapping[key],
            "object_id": f"greenery_{oid}", "source": "greenery_hard",
            "sample_weight": 4.0, "oid": oid,
        }
        (val_samples if split == "val" else train_samples).append(sample)
        if saved % 20 == 0:
            print(f"  ... {saved}", flush=True)

    meta = {
        "class_mapping": class_mapping,
        "dataset_info": {"source": "greenery_courtyard_hard", "saved": saved, "failed": failed},
        "train_samples": train_samples,
        "val_samples": val_samples,
    }
    args.out_meta.parent.mkdir(parents=True, exist_ok=True)
    args.out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved={saved} failed={failed} train={len(train_samples)} val={len(val_samples)}")
    print("train:", Counter(s["class_name"] for s in train_samples))
    print("val:", Counter(s["class_name"] for s in val_samples))
    return 0 if saved >= 40 else 1


if __name__ == "__main__":
    raise SystemExit(main())
