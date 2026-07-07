"""Классификация участков: эвристика по тексту и/или ML по панораме."""

from __future__ import annotations

import logging
import os
from typing import Literal

import geopandas as gpd

from web_app.gpkg_io import (
    apply_classify_all,
    boundary_view_points,
    bearing_geographic_deg,
    classify_from_description,
    heuristic_class,
    mark_no_panorama,
    parcel_panorama_available,
    view_points_for_parcel,
)
from web_app.ml_classifier import checkpoints_available as ensemble_ok, get_classifier
from web_app.panorama_crop import render_context_crop_jpeg
from web_app.probe_classifier import checkpoints_available as probe_ok, get_probe_classifier
from web_app.taxonomy import labels_for_class_key, refine_subclass

log = logging.getLogger(__name__)

Mode = Literal["heuristic", "ml", "ml+heuristic", "smart"]

_PANO_OK_STATUS = "Панорама доступна"

# Описания, где текст не различает класс — только панорама (ОС: 1712× «зелень во дворах»)
_AMBIGUOUS_TEXT = (
    "просто зелень",
    "прочие нежилые",
)


def _require_panorama() -> bool:
    return os.environ.get("CLASSIFY_REQUIRE_PANORAMA", "1").lower() not in ("0", "false", "no")


def _ml_min_confidence() -> float:
    return float(os.environ.get("ML_MIN_CONFIDENCE", "0.28"))


def _heuristic_fallback() -> bool:
    """Разрешить текстовую эвристику только если явно включено."""
    return os.environ.get("CLASSIFIER_FALLBACK_HEURISTIC", "0").lower() in ("1", "true", "yes")


def _description_ambiguous(desc: str) -> bool:
    d = desc.lower().strip()
    return any(p in d for p in _AMBIGUOUS_TEXT)


def _heuristic_prediction(desc: str) -> tuple[str, str] | None:
    cls, sub = heuristic_class(desc)
    if cls and sub:
        return cls, sub
    return None


def _active_classifier():
    """probe (VLM-метки) > ensemble > None. Решение без участия пользователя."""
    backend = os.environ.get("CLASSIFIER_BACKEND", "auto").lower()
    if backend in ("probe", "auto") and probe_ok():
        m = get_probe_classifier()
        if m is not None:
            return m, m.ckpt_path.stem
    if backend in ("ensemble", "auto") and ensemble_ok():
        m = get_classifier()
        if m is not None:
            return m, "ensemble"
    return None, None


def ml_checkpoints_available() -> bool:
    return probe_ok() or ensemble_ok()


def _target_centroid(geometry):
    try:
        pt = geometry.representative_point()
        if pt is None or pt.is_empty:
            return None
        return float(pt.y), float(pt.x)
    except Exception:
        return None


def _classify_parcel_ml(
    geometry,
    photos_ui: str = "",
    *,
    min_confidence: float | None = None,
) -> tuple[str, str, float, str] | None:
    if min_confidence is None:
        min_confidence = _ml_min_confidence()
    clf, backend = _active_classifier()
    if clf is None:
        return None

    max_views = int(os.environ.get("ML_MAX_VIEW_POINTS", "5"))
    points = view_points_for_parcel(photos_ui, geometry)
    if not points:
        points = boundary_view_points(geometry)
    if not points:
        return None

    target = _target_centroid(geometry)
    votes: list[tuple[str, float]] = []

    for vp in points[:max(1, max_views)]:
        lat, lon = float(vp["lat"]), float(vp["lon"])
        bearing = vp.get("bearing_deg")
        if bearing is None and target is not None:
            bearing = bearing_geographic_deg(lat, lon, target[0], target[1])
        if bearing is None:
            bearing = 0.0

        jpeg = render_context_crop_jpeg(
            lat, lon, heading_deg=float(bearing), out_width=896, out_height=672
        )
        if not jpeg:
            continue
        pred = clf.predict_jpeg(jpeg)
        votes.append((pred["class_key"], pred["confidence"]))

    if not votes:
        return None

    acc_score: dict[str, float] = {}
    acc_conf: dict[str, list[float]] = {}
    for ck, conf in votes:
        acc_score[ck] = acc_score.get(ck, 0.0) + conf * conf
        acc_conf.setdefault(ck, []).append(conf)
    best_key = max(acc_score, key=acc_score.get)
    best_conf = sum(acc_conf[best_key]) / len(acc_conf[best_key])
    if best_conf < min_confidence:
        return None

    cls_ru, sub_ru = labels_for_class_key(best_key)
    return cls_ru, sub_ru, best_conf, best_key


def apply_classification(
    gdf: gpd.GeoDataFrame,
    *,
    mode: Mode = "ml",
    overwrite: bool = False,
    max_rows: int = 100,
) -> dict:
    """
    Классифицировать участки.
    mode:
      ml — только панорама + CLIP probe (**продакшен по умолчанию**)
      ml+heuristic — ML, fallback на текст если CLASSIFIER_FALLBACK_HEURISTIC=1
      heuristic — только текст GPKG
      smart — устаревший гибрид (не рекомендуется)
    """
    stats = {
        "heuristic": 0,
        "ml": 0,
        "skipped": 0,
        "ml_low_confidence": 0,
        "no_panorama": 0,
        "ml_available": ml_checkpoints_available(),
        "require_panorama": _require_panorama(),
        "mode": mode,
    }

    if mode == "heuristic" and not _require_panorama():
        stats["heuristic"] = apply_classify_all(gdf, overwrite=overwrite)
        return stats

    ci = gdf.columns.get_loc("class_ui")
    si = gdf.columns.get_loc("subclass_ui")
    psi = gdf.columns.get_loc("panorama_status_ui")
    processed = 0

    for pos in range(len(gdf)):
        if processed >= max_rows:
            break
        cur_cls = str(gdf.iloc[pos, ci] or "").strip()
        cur_sub = str(gdf.iloc[pos, si] or "").strip()
        if not overwrite and cur_cls and cur_sub and cur_cls != "Категории нет":
            continue

        geom = gdf.iloc[pos].geometry
        photos = str(gdf.iloc[pos].get("photos_ui") or "")
        has_pano = parcel_panorama_available(geom, photos)

        if _require_panorama() and not has_pano:
            mark_no_panorama(gdf, pos)
            stats["no_panorama"] += 1
            processed += 1
            continue

        gdf.iloc[pos, psi] = _PANO_OK_STATUS
        desc = str(gdf.iloc[pos].get("generated_land_use", "") or "").strip()
        if desc.lower() == "nan":
            desc = ""

        use_ml = mode in ("ml", "ml+heuristic") or (
            mode == "smart" and _description_ambiguous(desc)
        )
        use_heuristic_first = mode == "smart" and not _description_ambiguous(desc)

        if use_heuristic_first:
            h = _heuristic_prediction(desc) or classify_from_description(desc)
            gdf.iloc[pos, ci], gdf.iloc[pos, si] = h[0], h[1]
            stats["heuristic"] += 1
            processed += 1
            continue

        ml_result = _classify_parcel_ml(geom, photos) if use_ml else None

        if ml_result:
            cls_ru, sub_ru, conf, class_key = ml_result
            sub_ru = refine_subclass(class_key, desc, sub_ru)
            gdf.iloc[pos, ci] = cls_ru
            gdf.iloc[pos, si] = sub_ru
            stats["ml"] += 1
        elif mode == "ml":
            gdf.iloc[pos, psi] = "ML: низкая уверенность"
            stats["ml_low_confidence"] += 1
            stats["skipped"] += 1
        elif mode in ("heuristic", "ml+heuristic", "smart") and _heuristic_fallback():
            cls, sub = classify_from_description(desc)
            gdf.iloc[pos, ci] = cls
            gdf.iloc[pos, si] = sub
            stats["heuristic"] += 1
        elif mode in ("ml+heuristic", "smart") and not _heuristic_fallback():
            gdf.iloc[pos, psi] = "ML: низкая уверенность"
            stats["ml_low_confidence"] += 1
            stats["skipped"] += 1
        processed += 1

    return stats
