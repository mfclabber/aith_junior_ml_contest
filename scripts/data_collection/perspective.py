from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ViewSpec:
    heading_deg: float  # yaw, degrees, 0 = east? (we treat as longitude, see below)
    pitch_deg: float    # pitch, degrees, +up
    fov_deg: float      # horizontal FOV in degrees
    out_size: Tuple[int, int]  # (width, height)


def _wrap_lon(lon: np.ndarray) -> np.ndarray:
    """Wrap longitude to [-pi, pi)."""
    return (lon + np.pi) % (2 * np.pi) - np.pi


def extract_perspective(
    equirect: Image.Image,
    *,
    heading_deg: float,
    pitch_deg: float,
    fov_deg: float,
    out_width: int,
    out_height: int,
) -> Image.Image:
    """
    Convert an equirectangular panorama to a rectilinear (perspective) view.

    Conventions:
    - Equirectangular image maps longitude (theta) to x and latitude (phi) to y:
      theta in [-pi, pi), phi in [-pi/2, pi/2]
    - heading_deg rotates around vertical axis (yaw), pitch_deg rotates around x-axis (look up/down).
    """
    if equirect.mode != "RGB":
        equirect = equirect.convert("RGB")

    src = np.asarray(equirect, dtype=np.uint8)
    src_h, src_w = src.shape[:2]

    # Output pixel grid in normalized camera plane coordinates
    xs = (np.linspace(0.5, out_width - 0.5, out_width) / out_width - 0.5) * 2.0
    ys = (np.linspace(0.5, out_height - 0.5, out_height) / out_height - 0.5) * 2.0
    xv, yv = np.meshgrid(xs, ys)  # shape (H, W)

    # Horizontal FOV -> focal length in normalized coordinates
    fov = np.deg2rad(float(fov_deg))
    f = 1.0 / np.tan(fov / 2.0)

    # Camera rays in camera coordinates (z forward)
    # x right, y down in image; we use y up for math, so flip sign
    x_cam = xv
    y_cam = -yv
    z_cam = np.full_like(x_cam, f)
    rays = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (H, W, 3)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True) + 1e-9

    # Rotation: yaw (heading) then pitch
    yaw = np.deg2rad(float(heading_deg))
    pitch = np.deg2rad(float(pitch_deg))

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)

    # Yaw about +Y axis
    R_yaw = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    # Pitch about +X axis
    R_pitch = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    R = R_yaw @ R_pitch

    rays_world = rays @ R.T  # (H, W, 3)
    xw = rays_world[..., 0]
    yw = rays_world[..., 1]
    zw = rays_world[..., 2]

    # Spherical coordinates
    lon = np.arctan2(xw, zw)  # [-pi, pi)
    lat = np.arcsin(np.clip(yw, -1.0, 1.0))  # [-pi/2, pi/2]
    lon = _wrap_lon(lon)

    # Map lon/lat to source pixel coordinates
    src_x = (lon + np.pi) / (2 * np.pi) * src_w
    src_y = (np.pi / 2 - lat) / np.pi * src_h

    # Bilinear sampling
    x0 = np.floor(src_x).astype(np.int32) % src_w
    x1 = (x0 + 1) % src_w
    y0 = np.clip(np.floor(src_y).astype(np.int32), 0, src_h - 1)
    y1 = np.clip(y0 + 1, 0, src_h - 1)

    wx = (src_x - np.floor(src_x)).astype(np.float32)
    wy = (src_y - np.floor(src_y)).astype(np.float32)

    Ia = src[y0, x0]
    Ib = src[y0, x1]
    Ic = src[y1, x0]
    Id = src[y1, x1]

    wa = (1 - wx) * (1 - wy)
    wb = wx * (1 - wy)
    wc = (1 - wx) * wy
    wd = wx * wy

    out = (
        Ia * wa[..., None]
        + Ib * wb[..., None]
        + Ic * wc[..., None]
        + Id * wd[..., None]
    )
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def generate_default_views(
    pano_path: str,
    output_dir: str,
    *,
    headings: Tuple[float, ...] = (0.0, 90.0, 180.0, 270.0),
    pitches: Tuple[float, ...] = (0.0,),
    fov_deg: float = 90.0,
    out_width: int = 1024,
    out_height: int = 768,
    prefix: str = "view",
) -> int:
    """Generate a grid of perspective views from a panorama file. Returns #generated."""
    img = Image.open(pano_path)
    count = 0
    for pitch in pitches:
        for heading in headings:
            view = extract_perspective(
                img,
                heading_deg=heading,
                pitch_deg=pitch,
                fov_deg=fov_deg,
                out_width=out_width,
                out_height=out_height,
            )
            out_name = f"{prefix}_h{int(round(heading))}_p{int(round(pitch))}_f{int(round(fov_deg))}_{out_width}x{out_height}.jpg"
            out_path = f"{output_dir.rstrip('/')}/{out_name}"
            view.save(out_path, quality=92)
            count += 1
    return count

