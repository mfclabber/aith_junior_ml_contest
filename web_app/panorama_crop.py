"""Ракурс на участок: перспективный кроп из equirect панорамы Яндекса."""

from __future__ import annotations

import io
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data_collection.perspective import extract_perspective  # noqa: E402

from web_app.yandex_panorama import render_equirect_jpeg  # noqa: E402


def render_context_crop_jpeg(
    lat: float,
    lon: float,
    *,
    heading_deg: float,
    pitch_deg: float = 8.0,
    fov_deg: float = 75.0,
    out_width: int = 640,
    out_height: int = 400,
    equirect_max_w: int = 1200,
) -> bytes | None:
    """Кадр «на участок», не полная 360° панорама."""
    raw = render_equirect_jpeg(lat=lat, lon=lon, zoom=0, max_w=equirect_max_w)
    if not raw:
        return None
    equirect = Image.open(io.BytesIO(raw))
    crop = extract_perspective(
        equirect,
        heading_deg=float(heading_deg),
        pitch_deg=float(pitch_deg),
        fov_deg=float(fov_deg),
        out_width=int(out_width),
        out_height=int(out_height),
    )
    out = io.BytesIO()
    crop.save(out, format="JPEG", quality=82, optimize=True)
    return out.getvalue()
