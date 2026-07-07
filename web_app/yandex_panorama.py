"""Яндекс-панорамы: найти по координате и собрать JPEG из тайлов.

Основано на существующем коде `scripts/data_collection/pano.py`, адаптировано под веб:
- минимальный zoom по умолчанию (быстро)
- кэш в памяти по округлённым координатам
"""

from __future__ import annotations

import json
import io
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image


@dataclass(frozen=True)
class YandexPanoMeta:
    image_id: str
    tile_w: int
    tile_h: int
    zoom: int
    width: int
    height: int


_META_CACHE: dict[tuple[float, float, int], tuple[float, YandexPanoMeta | None]] = {}
_IMG_CACHE: dict[tuple[float, float, int, int], tuple[float, bytes]] = {}
_TTL_SEC = 3600


def _http_json(url: str, timeout: float = 8.0) -> dict | None:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
            "Referer": "https://yandex.ru/maps/",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            return data if isinstance(data, dict) else None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def _http_bytes(url: str, timeout: float = 10.0) -> bytes | None:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
            "Referer": "https://yandex.ru/maps/",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            b = resp.read()
            return b if b else None
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def locate_meta(lat: float, lon: float, zoom: int = 0) -> YandexPanoMeta | None:
    """Найти ближайшую панораму Яндекса к координате и метаданные тайлов/размера."""
    lat_r = round(float(lat), 5)
    lon_r = round(float(lon), 5)
    ck = (lat_r, lon_r, int(zoom))
    ent = _META_CACHE.get(ck)
    now = time.monotonic()
    if ent and (now - ent[0] < _TTL_SEC):
        return ent[1]

    api_url = (
        "https://api-maps.yandex.ru/services/panoramas/1.x/"
        f"?l=stv&lang=ru_RU&ll={lon_r},{lat_r}&origin=userAction&provider=streetview"
    )
    data = _http_json(api_url, timeout=7.0)
    if not data or data.get("status") == "error":
        _META_CACHE[ck] = (now, None)
        return None

    try:
        pano_data = data["data"]["Data"]
        images = pano_data["Images"]
        image_id = str(images["imageId"])
        tiles = images["Tiles"]
        tile_w = int(tiles["width"])
        tile_h = int(tiles["height"])
        zooms = images.get("Zooms", []) or []
        z = int(zoom)
        zoom_item = None
        for it in zooms:
            if isinstance(it, dict) and int(it.get("level", -999)) == z:
                zoom_item = it
                break
        if zoom_item is None:
            # если запрошенный zoom не доступен — взять самый маленький (самый быстрый)
            zoom_item = min(
                (it for it in zooms if isinstance(it, dict) and "level" in it),
                key=lambda x: int(x.get("level", 0)),
            )
            z = int(zoom_item["level"])
        width = int(zoom_item["width"])
        height = int(zoom_item["height"])
        meta = YandexPanoMeta(image_id=image_id, tile_w=tile_w, tile_h=tile_h, zoom=z, width=width, height=height)
    except Exception:
        _META_CACHE[ck] = (now, None)
        return None

    _META_CACHE[ck] = (now, meta)
    return meta


def render_equirect_jpeg(lat: float, lon: float, zoom: int = 0, max_w: int = 900, quality: int = 80) -> bytes | None:
    """Собрать equirect panorama как JPEG (уменьшенный), вернуть байты."""
    lat_r = round(float(lat), 5)
    lon_r = round(float(lon), 5)
    meta = locate_meta(lat_r, lon_r, zoom=zoom)
    if not meta:
        return None

    ik = (lat_r, lon_r, int(meta.zoom), int(max_w))
    ent = _IMG_CACHE.get(ik)
    now = time.monotonic()
    if ent and (now - ent[0] < _TTL_SEC):
        return ent[1]

    x_range = (meta.width + meta.tile_w - 1) // meta.tile_w
    y_range = (meta.height + meta.tile_h - 1) // meta.tile_h

    pano = Image.new("RGB", (meta.width, meta.height))

    # Параллельная загрузка тайлов: заметно ускоряет сборку превью.
    def load_one(xy: tuple[int, int]) -> tuple[int, int, bytes] | None:
        x, y = xy
        tile_url = f"https://pano.maps.yandex.net/{meta.image_id}/{meta.zoom}.{x}.{y}"
        b = _http_bytes(tile_url, timeout=10.0)
        if not b:
            return None
        return (x, y, b)

    coords = [(x, y) for x in range(x_range) for y in range(y_range)]
    max_workers = min(12, max(4, len(coords)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(load_one, xy) for xy in coords]
        for fut in as_completed(futs):
            res = fut.result()
            if not res:
                continue
            x, y, b = res
            try:
                img = Image.open(io.BytesIO(b))
                img.load()
            except Exception:
                continue
            pano.paste(img.convert("RGB"), (x * meta.tile_w, y * meta.tile_h))

    if max_w and pano.width > max_w:
        new_h = max(1, int(pano.height * (max_w / pano.width)))
        pano = pano.resize((max_w, new_h), Image.Resampling.LANCZOS)

    out = io.BytesIO()
    pano.save(out, format="JPEG", quality=int(quality), optimize=True, progressive=True)
    jpg = out.getvalue()
    _IMG_CACHE[ik] = (now, jpg)
    return jpg

