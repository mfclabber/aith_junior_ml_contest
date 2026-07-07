"""Прямые URL картинок панорам/улицы по координатам (опциональные ключи API)."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

_CACHE: dict[tuple[float, float], tuple[float, list[dict[str, str]]]] = {}
_CACHE_TTL_SEC = 3600
_CACHE_MAX = 3000


def _cache_get(lon_r: float, lat_r: float) -> list[dict[str, str]] | None:
    key = (lon_r, lat_r)
    ent = _CACHE.get(key)
    if not ent:
        return None
    ts, images = ent
    if time.monotonic() - ts > _CACHE_TTL_SEC:
        del _CACHE[key]
        return None
    return images


def _cache_set(lon_r: float, lat_r: float, images: list[dict[str, str]]) -> None:
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.clear()
    _CACHE[(lon_r, lat_r)] = (time.monotonic(), images)


def _http_json(url: str, headers: dict[str, str] | None = None, timeout: float = 8.0) -> Any:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def providers_configured() -> dict[str, bool]:
    g = bool(
        (os.environ.get("GOOGLE_MAPS_API_KEY") or os.environ.get("GOOGLE_STREET_VIEW_KEY") or "").strip()
    )
    m = bool(os.environ.get("MAPILLARY_ACCESS_TOKEN", "").strip())
    return {"google_street_view": g, "mapillary": m}


def _google_streetview_images(lat: float, lon: float) -> list[dict[str, str]]:
    key = (
        os.environ.get("GOOGLE_MAPS_API_KEY") or os.environ.get("GOOGLE_STREET_VIEW_KEY") or ""
    ).strip()
    if not key:
        return []
    q_meta = urllib.parse.urlencode({"location": f"{lat},{lon}", "key": key})
    meta_url = f"https://maps.googleapis.com/maps/api/streetview/metadata?{q_meta}"
    try:
        data = _http_json(meta_url, timeout=6.0)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, dict) or data.get("status") != "OK":
        return []
    loc = f"{lat},{lon}"
    out: list[dict[str, str]] = []
    # Два азимута — меньше расход квоты, чем полный круг
    for heading in (0, 180):
        q_img = urllib.parse.urlencode(
            {
                "size": "640x400",
                "location": loc,
                "fov": "90",
                "pitch": "10",
                "heading": str(heading),
                "key": key,
            }
        )
        url = f"https://maps.googleapis.com/maps/api/streetview?{q_img}"
        out.append({"url": url, "label": f"Google Street View ({heading}°)"})
    return out


def _mapillary_images(lat: float, lon: float, limit: int = 4) -> list[dict[str, str]]:
    token = os.environ.get("MAPILLARY_ACCESS_TOKEN", "").strip()
    if not token:
        return []
    # ~35 м по экватору — достаточно для ближайшего кадра
    d = 0.00035
    bbox = f"{lon - d},{lat - d},{lon + d},{lat + d}"
    url = (
        f"https://graph.mapillary.com/images?bbox={bbox}"
        f"&fields=thumb_1024_url,thumb_256_url&limit={limit}"
    )
    try:
        payload = _http_json(url, headers={"Authorization": f"OAuth {token}"}, timeout=8.0)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return []
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    out: list[dict[str, str]] = []
    for i, item in enumerate(rows):
        if not isinstance(item, dict):
            continue
        u = item.get("thumb_1024_url") or item.get("thumb_256_url")
        if not u:
            continue
        out.append({"url": str(u), "label": f"Mapillary #{i + 1}"})
    return out


def _crop_url(
    lat_r: float,
    lon_r: float,
    *,
    tgt_lat: float | None,
    tgt_lon: float | None,
    bearing_deg: float | None,
) -> str:
    q: dict[str, str] = {"lat": str(lat_r), "lon": str(lon_r)}
    if bearing_deg is not None:
        q["bearing"] = str(round(bearing_deg, 1))
    elif tgt_lat is not None and tgt_lon is not None:
        q["tgt_lat"] = str(tgt_lat)
        q["tgt_lon"] = str(tgt_lon)
    return "/api/yandex-panorama-crop.jpg?" + urllib.parse.urlencode(q)


def preview_images_for_point(
    lat: float,
    lon: float,
    *,
    tgt_lat: float | None = None,
    tgt_lon: float | None = None,
    bearing_deg: float | None = None,
) -> list[dict[str, str]]:
    """
    Список {url, label} — прямые ссылки на JPEG/WebP превью.
    Сначала Mapillary (часто лучше покрытие РФ), затем Google (несколько азимутов).
    """
    lon_r = round(float(lon), 4)
    lat_r = round(float(lat), 4)
    ck = (lon_r, lat_r, round(tgt_lat or 0, 4), round(tgt_lon or 0, 4), round(bearing_deg or -1, 1))
    cached = _CACHE.get(ck)
    if cached and (time.monotonic() - cached[0] < _CACHE_TTL_SEC):
        return cached[1]

    images: list[dict[str, str]] = []
    seen: set[str] = set()

    def add_batch(batch: list[dict[str, str]]) -> None:
        for it in batch:
            u = it.get("url")
            if not u or u in seen:
                continue
            seen.add(u)
            images.append(
                {
                    "url": u,
                    "label": it.get("label") or "Панорама",
                    "kind": it.get("kind") or "photo",
                }
            )

    images.append(
        {
            "url": _crop_url(lat_r, lon_r, tgt_lat=tgt_lat, tgt_lon=tgt_lon, bearing_deg=bearing_deg),
            "label": "Вид на участок (кроп)",
            "kind": "crop",
        }
    )
    images.append(
        {
            "url": f"/api/yandex-panorama.jpg?lat={lat_r}&lon={lon_r}&zoom=0&w=1024",
            "label": "Панорама 360° (Яндекс)",
            "kind": "equirect",
        }
    )
    add_batch(_mapillary_images(lat_r, lon_r))
    add_batch(_google_streetview_images(lat_r, lon_r))

    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.clear()
    _CACHE[ck] = (time.monotonic(), images)
    return images
