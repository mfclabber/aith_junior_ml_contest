#!/usr/bin/env python3
"""
Build rectilinear crops from equirect sources using classification_region / bearing.

Reads data/ml_dataset/metadata.json (or --metadata), opens each source_path
panorama, runs perspective.extract_perspective toward bearing_geographic_deg
(plus --heading-offset-deg for equirect↔north calibration), writes images under
--output-dir preserving train/<class>/ and val/<class>/.

Writes metadata_perspective.json: same samples with updated image_path and
extra fields for the perspective used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "scripts" / "data_collection") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts" / "data_collection"))
from perspective import extract_perspective  # noqa: E402


def _heading_for_sample(
    cr: Dict[str, Any],
    heading_offset_deg: float,
    default_bearing_when_unknown_deg: float,
) -> float:
    b = cr.get("bearing_geographic_deg")
    if b is None:
        return default_bearing_when_unknown_deg % 360.0
    return (float(b) + heading_offset_deg) % 360.0


def _class_counts(records: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in records:
        counts[r["class_name"]] = counts.get(r["class_name"], 0) + 1
    return counts


def _headings_for_sample(
    cr: Dict[str, Any],
    heading_offset_deg: float,
    default_bearing_when_unknown_deg: float,
    multi_heading_when_unknown: Optional[List[float]],
) -> List[Tuple[float, str]]:
    """Возвращает список (heading, tag). Для null-bearing — несколько ракурсов."""
    b = cr.get("bearing_geographic_deg")
    if b is not None:
        return [((float(b) + heading_offset_deg) % 360.0, "")]
    if multi_heading_when_unknown:
        return [((h + heading_offset_deg) % 360.0, f"h{int(h):03d}") for h in multi_heading_when_unknown]
    return [(default_bearing_when_unknown_deg % 360.0, "")]


def _process_split(
    project_root: Path,
    samples: List[Dict[str, Any]],
    split_name: str,
    out_root: Path,
    heading_offset_deg: float,
    default_bearing_when_unknown_deg: float,
    fov_override: Optional[float],
    out_width: int,
    out_height: int,
    pitch_deg: float,
    multi_heading_when_unknown: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in samples:
        cr = s.get("classification_region") or {}
        fov = float(fov_override if fov_override is not None else cr.get("fov_deg", 90.0))
        src = project_root / s["source_path"]
        if not src.is_file():
            raise FileNotFoundError(f"Missing equirect: {src}")

        headings = _headings_for_sample(
            cr, heading_offset_deg, default_bearing_when_unknown_deg, multi_heading_when_unknown
        )
        rel_class = s["class_name"]
        stem = Path(s["image_path"]).stem
        suffix = Path(s["image_path"]).suffix or ".jpg"

        img = Image.open(src)
        try:
            for heading, tag in headings:
                name = f"{stem}_{tag}{suffix}" if tag else f"{stem}{suffix}"
                dest = out_root / split_name / rel_class / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                crop = extract_perspective(
                    img,
                    heading_deg=heading,
                    pitch_deg=pitch_deg,
                    fov_deg=fov,
                    out_width=out_width,
                    out_height=out_height,
                )
                crop.save(dest, quality=92)

                rel_image = dest.relative_to(out_root)
                rec = dict(s)
                rec["image_path"] = str(rel_image).replace("\\", "/")
                rec["perspective"] = {
                    "heading_deg": round(heading, 4),
                    "heading_offset_deg": heading_offset_deg,
                    "pitch_deg": pitch_deg,
                    "fov_deg": fov,
                    "out_width": out_width,
                    "out_height": out_height,
                    "source_equirect": s["source_path"],
                    "multi_heading": bool(tag),
                }
                out.append(rec)
        finally:
            img.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Perspective crops from equirect + bearing metadata.")
    parser.add_argument("--project-root", type=Path, default=None, help="Repo root (default: auto)")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/ml_dataset/metadata.json"),
        help="Input metadata.json from prepare_ml_dataset",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/ml_perspective"))
    parser.add_argument(
        "--heading-offset-deg",
        type=float,
        default=0.0,
        help="Add to geographic bearing so equirect center aligns with your provider's orientation.",
    )
    parser.add_argument(
        "--default-bearing-when-unknown-deg",
        type=float,
        default=0.0,
        help="When bearing is null (pano≈OSM), center view uses this perspective heading.",
    )
    parser.add_argument(
        "--multi-heading-when-unknown",
        type=str,
        default="0,90,180,270",
        help="Comma-separated headings for null-bearing samples (pano at object). "
        "Empty string disables (uses --default-bearing-when-unknown-deg).",
    )
    parser.add_argument("--fov-deg", type=float, default=None, help="Override FOV from metadata (optional)")
    parser.add_argument("--out-width", type=int, default=896)
    parser.add_argument("--out-height", type=int, default=672)
    parser.add_argument("--pitch-deg", type=float, default=0.0)
    args = parser.parse_args()

    project_root = args.project_root or _ROOT
    meta_path = args.metadata if args.metadata.is_absolute() else project_root / args.metadata
    out_root = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir

    data = json.loads(meta_path.read_text(encoding="utf-8"))
    train = data["train_samples"]
    val = data["val_samples"]

    multi_heading = [
        float(x) for x in str(args.multi_heading_when_unknown).split(",") if x.strip() != ""
    ] or None

    train_out = _process_split(
        project_root,
        train,
        "train",
        out_root,
        args.heading_offset_deg,
        args.default_bearing_when_unknown_deg,
        args.fov_deg,
        args.out_width,
        args.out_height,
        args.pitch_deg,
        multi_heading_when_unknown=multi_heading,
    )
    val_out = _process_split(
        project_root,
        val,
        "val",
        out_root,
        args.heading_offset_deg,
        args.default_bearing_when_unknown_deg,
        args.fov_deg,
        args.out_width,
        args.out_height,
        args.pitch_deg,
        multi_heading_when_unknown=multi_heading,
    )

    out_meta = {
        "dataset_info": {
            **data.get("dataset_info", {}),
            "perspective": {
                "heading_offset_deg": args.heading_offset_deg,
                "default_bearing_when_unknown_deg": args.default_bearing_when_unknown_deg,
                "out_size": [args.out_width, args.out_height],
                "pitch_deg": args.pitch_deg,
                "fov_override": args.fov_deg,
            },
            "source_metadata": str(meta_path.relative_to(project_root)) if meta_path.is_relative_to(project_root) else str(meta_path),
        },
        "class_mapping": data["class_mapping"],
        "train_statistics": _class_counts(train_out),
        "val_statistics": _class_counts(val_out),
        "train_samples": train_out,
        "val_samples": val_out,
    }

    out_root.mkdir(parents=True, exist_ok=True)
    out_json = out_root / "metadata_perspective.json"
    out_json.write_text(json.dumps(out_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(train_out)} train + {len(val_out)} val crops under {out_root}")
    print(f"Metadata: {out_json}")


if __name__ == "__main__":
    main()
