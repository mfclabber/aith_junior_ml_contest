#!/usr/bin/env python3
"""Бенчмарк probe-чекпойнтов на analyst-val и GPKG (чистый ML)."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from web_app.classify_service import apply_classification
from web_app.gpkg_io import load_parcel_gdf, parse_photo_coords
from web_app.probe_classifier import _ClipProbe


def eval_analyst_val(ckpt: Path) -> dict:
    meta = json.loads((ROOT / "data/ml_perspective/metadata_analyst.json").read_text())
    probe = _ClipProbe(ckpt)
    ok = 0
    for s in meta["val_samples"]:
        img = Image.open(ROOT / "data/ml_perspective" / s["image_path"]).convert("RGB")
        if probe.predict_image(img)["class_key"] == s["class_name"]:
            ok += 1
    n = len(meta["val_samples"])
    return {"analyst_val_acc": round(ok / n, 4), "analyst_val_n": n}


def eval_gpkg_ml(ckpt: Path, n: int = 40, seed: int = 42) -> dict:
    import web_app.probe_classifier as pc

    os.environ["CLASSIFIER_PROBE"] = ckpt.name
    os.environ["CLASSIFIER_MODE"] = "ml"
    pc.get_probe_classifier.cache_clear()

    gdf = load_parcel_gdf(ROOT / "results/out_for_katya_new.gpkg")
    corr = gdf["class_corrected_ui"].astype(str).str.strip()
    rows = [
        i for i in range(len(gdf))
        if parse_photo_coords(str(gdf.iloc[i]["photos_ui"] or ""))
        and corr.iloc[i] not in ("", "Категории нет", "nan")
    ]
    random.seed(seed)
    random.shuffle(rows)
    sample = rows[:n]
    sub = gdf.iloc[sample].copy().reset_index(drop=True)
    stats = apply_classification(sub, mode="ml", overwrite=True, max_rows=n)
    ok = sum(1 for j, p in enumerate(sample) if sub.iloc[j]["class_ui"] == corr.iloc[p])
    return {
        "gpkg_ml_acc": round(ok / n, 4),
        "gpkg_n": n,
        "stats": stats,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpkg-n", type=int, default=40)
    args = ap.parse_args()

    ckpts = sorted((ROOT / "checkpoints").glob("clip_probe*.pt"))
    results = []
    for p in ckpts:
        try:
            r = {"checkpoint": p.name, **eval_analyst_val(p)}
            if args.gpkg_n > 0:
                r.update(eval_gpkg_ml(p, n=args.gpkg_n))
            results.append(r)
            print(
                f"{p.name}: analyst_val={r['analyst_val_acc']:.1%}"
                + (f" gpkg_ml={r.get('gpkg_ml_acc', 0):.1%}" if args.gpkg_n else "")
            )
        except Exception as exc:
            print(f"{p.name}: SKIP {exc}")

    out = ROOT / "results" / "probe_benchmark.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    best = max(results, key=lambda x: (x.get("gpkg_ml_acc", 0), x["analyst_val_acc"]))
    print(f"\nBEST for GPKG ML: {best['checkpoint']} gpkg={best.get('gpkg_ml_acc')} analyst_val={best['analyst_val_acc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
