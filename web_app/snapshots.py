"""Кэш JPEG-снимков по участку (кроп с каждой грани)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from web_app.gpkg_io import boundary_view_points, bearing_geographic_deg
from web_app.panorama_crop import render_context_crop_jpeg


def snapshot_dir(upload_dir: Path, dataset_id: str, parcel_key: str) -> Path:
    d = upload_dir / dataset_id / "snapshots" / parcel_key
    d.mkdir(parents=True, exist_ok=True)
    return d


def yandex_maps_panorama_url(lon: float, lat: float, bearing_deg: float | None = None) -> str:
    """Ссылка на Яндекс.Карты: панорама в точке съёмки + метка."""
    base = (
        f"https://yandex.ru/maps/?ll={lon}%2C{lat}&z=19&l=stv"
        f"&pt={lon}%2C{lat},pm2rdm"
    )
    if bearing_deg is not None:
        base += f"&panorama%5Bdirection%5D={int(round(bearing_deg))}"
    return base


def ensure_parcel_snapshots(
    upload_dir: Path,
    dataset_id: str,
    parcel_key: str,
    geometry,
    photos_ui: str = "",
) -> dict[str, Any]:
    """Сгенерировать кропы по граням участка (если ещё нет)."""
    out_dir = snapshot_dir(upload_dir, dataset_id, parcel_key)
    meta_path = out_dir / "index.json"
    if meta_path.is_file():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    points = boundary_view_points(geometry)
    if not points:
        return {"files": [], "view_points": []}

    try:
        tgt = geometry.representative_point()
        tgt_lat, tgt_lon = float(tgt.y), float(tgt.x)
    except Exception:
        tgt_lat = tgt_lon = None

    files: list[dict[str, str]] = []
    for vp in points:
        idx = vp["index"]
        lon, lat = float(vp["lon"]), float(vp["lat"])
        bearing = vp.get("bearing_deg")
        if bearing is None and tgt_lat is not None:
            bearing = bearing_geographic_deg(lat, lon, tgt_lat, tgt_lon)
        if bearing is None:
            bearing = 0.0
        fname = f"edge_{idx:02d}_crop.jpg"
        fpath = out_dir / fname
        if not fpath.is_file():
            jpg = render_context_crop_jpeg(lat, lon, heading_deg=float(bearing))
            if jpg:
                fpath.write_bytes(jpg)
        if fpath.is_file():
            files.append(
                {
                    "name": fname,
                    "label": vp.get("label") or f"Грань {idx}",
                    "url": f"/api/parcel-snapshots/{dataset_id}/{parcel_key}/{fname}",
                    "yandex_url": yandex_maps_panorama_url(lon, lat, bearing),
                }
            )

    meta = {"files": files, "view_points": points}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta
