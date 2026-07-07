#!/usr/bin/env python3
"""
Build train/val folders + metadata.json from data/dataset.

- Split is by OSM object (metadata["id"]), not by individual panorama files:
  all seasons of one object stay in the same split — no train/val leakage.
- Each sample record includes classification_region: geographic bearing from the
  panorama capture point toward the labeled OSM coordinates, plus FOV hint for
  perspective crops (see perspective.extract_perspective).

Usage:
  python3 scripts/training/prepare_ml_dataset.py \\
    --dataset-dir data/dataset --output-dir data/ml_dataset --val-ratio 0.2
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EARTH_RADIUS_M = 6_371_000.0

# Default horizontal FOV for the "classification window" (document only; tune in perspective.py)
DEFAULT_CLASSIFICATION_FOV_DEG = 90.0


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def _bearing_geographic_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from point 1 to point 2, degrees clockwise from true north [0, 360)."""
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δλ = math.radians(lon2 - lon1)
    y = math.sin(Δλ) * math.cos(φ2)
    x = math.cos(φ1) * math.sin(φ2) - math.sin(φ1) * math.cos(φ2) * math.cos(Δλ)
    θ = math.degrees(math.atan2(y, x))
    return (θ + 360.0) % 360.0


def _season_from_filename(name: str) -> str:
    stem = Path(name).stem
    if stem == "panorama":
        return "default"
    if stem.startswith("panorama_"):
        return stem.replace("panorama_", "", 1)
    return stem


def _pano_lat_lon(meta: Dict[str, Any], season: str) -> Tuple[float, float]:
    """Panorama capture position; falls back to OSM coordinates if unknown."""
    pbs = meta.get("panoramas_by_season") or {}
    if season in pbs and pbs[season].get("lat") is not None and pbs[season].get("lon") is not None:
        return float(pbs[season]["lat"]), float(pbs[season]["lon"])
    if season == "default" and pbs:
        first = next(iter(pbs.values()))
        if first.get("lat") is not None and first.get("lon") is not None:
            return float(first["lat"]), float(first["lon"])
    c = meta.get("coordinates") or {}
    return float(c["lat"]), float(c["lon"])


def _target_lat_lon(meta: Dict[str, Any]) -> Tuple[float, float]:
    c = meta.get("coordinates") or {}
    return float(c["lat"]), float(c["lon"])


def _classification_region(
    meta: Dict[str, Any],
    season: str,
    fov_deg: float,
) -> Dict[str, Any]:
    pano_lat, pano_lon = _pano_lat_lon(meta, season)
    tgt_lat, tgt_lon = _target_lat_lon(meta)
    dist = _haversine_m(pano_lat, pano_lon, tgt_lat, tgt_lon)
    if dist < 2.0:
        bearing: Optional[float] = None
        note = "pano_and_target_coincident_or_close; bearing_undefined"
    else:
        bearing = round(_bearing_geographic_deg(pano_lat, pano_lon, tgt_lat, tgt_lon), 4)
        note = "bearing_from_panorama_to_osm_label_coordinates_clockwise_from_north"

    return {
        "reference": "bearing_from_panorama_capture_to_osm_object_coordinates",
        "bearing_geographic_deg": bearing,
        "fov_deg": fov_deg,
        "panorama": {"lat": pano_lat, "lon": pano_lon},
        "label_target_osm": {"lat": tgt_lat, "lon": tgt_lon},
        "distance_m_approx": round(dist, 2),
        "notes": note,
    }


def _collect_samples(
    dataset_dir: Path,
    class_names: Sequence[str],
    fov_deg: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Returns flat list of sample dicts (pre-split) and object_id -> class_name.
    """
    object_class: Dict[str, str] = {}
    samples: List[Dict[str, Any]] = []

    for cls in class_names:
        class_dir = dataset_dir / cls
        if not class_dir.is_dir():
            continue
        for obj_dir in sorted(class_dir.iterdir()):
            if not obj_dir.is_dir():
                continue
            meta_path = obj_dir / "metadata.json"
            if not meta_path.is_file():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            oid = meta.get("id")
            if not oid:
                continue
            mcls = meta.get("class")
            if mcls and mcls != cls:
                raise ValueError(f"class mismatch {meta_path}: folder={cls} metadata.class={mcls}")
            object_class[oid] = cls

            jpgs = sorted(obj_dir.glob("panorama*.jpg"))
            if not jpgs:
                continue
            for jpg in jpgs:
                season = _season_from_filename(jpg.name)
                src_rel = Path("data") / "dataset" / jpg.relative_to(dataset_dir)

                sample = {
                    "object_id": oid,
                    "class_name": cls,
                    "season": season if season != "default" else None,
                    "source_path": str(src_rel).replace("\\", "/"),
                    "abs_source": jpg,
                    "classification_region": _classification_region(meta, season, fov_deg),
                }
                samples.append(sample)

    return samples, object_class


def _stratified_object_split(
    object_class: Dict[str, str],
    val_ratio: float,
    seed: int,
) -> Dict[str, str]:
    """Assign each object_id to 'train' or 'val', stratified by class."""
    by_class: Dict[str, List[str]] = defaultdict(list)
    for oid, cls in object_class.items():
        by_class[cls].append(oid)

    rng = random.Random(seed)
    assignment: Dict[str, str] = {}

    for cls, oids in by_class.items():
        ids = list(oids)
        rng.shuffle(ids)
        n = len(ids)
        n_val = int(round(n * val_ratio))
        n_val = max(0, min(n, n_val))
        # ensure at least one train when n > 1
        if n > 1 and n_val == n:
            n_val = n - 1
        val_set = set(ids[:n_val])
        for oid in ids:
            assignment[oid] = "val" if oid in val_set else "train"

    return assignment


def _class_mapping(class_names: Sequence[str]) -> Dict[str, int]:
    return {name: i for i, name in enumerate(class_names)}


def prepare(
    dataset_dir: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
    fov_deg: float,
    copy_files: bool,
) -> None:
    taxonomy_path = dataset_dir / "taxonomy.json"
    if taxonomy_path.is_file():
        taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
        class_names = sorted(taxonomy.keys(), key=lambda k: taxonomy[k].get("id", 0))
    else:
        class_names = sorted(
            p.name
            for p in dataset_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".") and p.name not in ("utt_examples",)
        )

    samples, object_class = _collect_samples(dataset_dir, class_names, fov_deg)
    if not samples:
        raise SystemExit(f"No panorama*.jpg found under {dataset_dir}")

    split = _stratified_object_split(object_class, val_ratio, seed)
    class_mapping = _class_mapping(class_names)

    train_root = output_dir / "train"
    val_root = output_dir / "val"
    if output_dir.exists():
        shutil.rmtree(train_root, ignore_errors=True)
        shutil.rmtree(val_root, ignore_errors=True)
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)
    for cls in class_names:
        (train_root / cls).mkdir(parents=True, exist_ok=True)
        (val_root / cls).mkdir(parents=True, exist_ok=True)

    train_samples: List[Dict[str, Any]] = []
    val_samples: List[Dict[str, Any]] = []

    # Stable global index for filenames (sorted order)
    samples.sort(key=lambda s: (s["class_name"], s["object_id"], s["season"] or "", str(s["abs_source"])))

    for idx, s in enumerate(samples):
        oid = s["object_id"]
        cls = s["class_name"]
        group = split[oid]
        root = train_root if group == "train" else val_root
        season_tag = s["season"] or "default"
        base = f"{Path(s['abs_source']).parent.name}_{season_tag}_{idx:04d}.jpg"
        dest = root / cls / base
        if copy_files:
            shutil.copy2(s["abs_source"], dest)
        else:
            if dest.exists():
                dest.unlink()
            dest.symlink_to(s["abs_source"].resolve())

        rel_image = dest.relative_to(output_dir)
        record = {
            "image_path": str(rel_image).replace("\\", "/"),
            "class_name": cls,
            "class_id": class_mapping[cls],
            "object_id": oid,
            "season": s["season"],
            "source_path": s["source_path"],
            "classification_region": s["classification_region"],
        }
        if group == "train":
            train_samples.append(record)
        else:
            val_samples.append(record)

    train_ids = {s["object_id"] for s in train_samples}
    val_ids = {s["object_id"] for s in val_samples}
    leak = train_ids & val_ids

    meta_out = {
        "dataset_info": {
            "total_train": len(train_samples),
            "total_val": len(val_samples),
            "total": len(train_samples) + len(val_samples),
            "num_classes": len(class_names),
            "split_policy": "object_stratified",
            "val_ratio": val_ratio,
            "random_seed": seed,
            "train_unique_objects": len(train_ids),
            "val_unique_objects": len(val_ids),
            "train_val_object_overlap": len(leak),
            "default_classification_fov_deg": fov_deg,
        },
        "class_mapping": class_mapping,
        "train_statistics": {
            cls: sum(1 for s in train_samples if s["class_name"] == cls) for cls in class_names
        },
        "val_statistics": {
            cls: sum(1 for s in val_samples if s["class_name"] == cls) for cls in class_names
        },
        "train_samples": train_samples,
        "val_samples": val_samples,
    }

    if leak:
        raise RuntimeError(f"split bug: overlapping object_ids: {leak}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "metadata.json"
    out_path.write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(train_samples)} train, {len(val_samples)} val)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ML dataset with object-level split.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/ml_dataset"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fov-deg", type=float, default=DEFAULT_CLASSIFICATION_FOV_DEG)
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Symlink images instead of copying (saves space).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = args.dataset_dir if args.dataset_dir.is_absolute() else project_root / args.dataset_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir

    prepare(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        fov_deg=args.fov_deg,
        copy_files=not args.symlink,
    )


if __name__ == "__main__":
    main()
