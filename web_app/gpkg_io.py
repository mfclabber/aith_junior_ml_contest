"""Загрузка GPKG и подготовка данных для веб-клиента."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple
from uuid import uuid4

import geopandas as gpd

from web_app.taxonomy import NO_PANORAMA_CLASS, NO_PANORAMA_STATUS

_LL_RE = re.compile(r"[?&]ll=([\d.+-]+)(?:%2C|,)([\d.+-]+)", re.I)
_SNAPSHOTS_RE = re.compile(r"^/snapshots/[^/\s]+/[^/\s]+/?", re.I)
_EARTH_R_M = 6_371_000.0

# Поле с текстовым описанием / ВРИ
_DESC_COLUMN_ALIASES = (
    "generated_land_use",
    "land_use",
    "landuse",
    "land_use_text",
    "description",
    "описание",
    "generated_landuse",
    "class_Описание (исходное)",
    "class_описание (исходное)",
)
_CLASS_COLUMN_ALIASES = (
    "class_ui", "class", "utt_class", "predicted_class", "класс", "Class",
    "class_Класс", "class_класс",
)
_SUBCLASS_COLUMN_ALIASES = (
    "subclass_ui", "subclass", "sub_class", "predicted_subclass", "подкласс",
    "class_Подкласс", "class_подкласс",
)
_CORRECTED_CLASS_ALIASES = (
    "class_corrected_ui", "откорректированный класс",
    "class_Откорректированный класс", "class_откорректированный класс",
)
_CORRECTED_SUBCLASS_ALIASES = (
    "subclass_corrected_ui", "откорректированный подкласс",
    "class_Откорректированный подкласс", "class_откорректированный подкласс",
)
_PHOTOS_COLUMN_ALIASES = ("photos_ui", "class_Снимки", "снимки", "photos")
_OID_COLUMN_ALIASES = ("oid", "class_ID участка", "id участка", "parcel_id")

DEFAULT_CLASS = "Активные городские территории"
DEFAULT_SUBCLASS = "Жилая застройка с активностью"


def list_layers(path: Path) -> List[str]:
    con = sqlite3.connect(str(path))
    try:
        rows = con.execute(
            "SELECT table_name FROM gpkg_contents WHERE data_type='features'"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        con.close()


def yandex_widget_url(lon: float, lat: float) -> str:
    return f"https://yandex.ru/map-widget/v1/?ll={lon}%2C{lat}&z=18&l=stv&lang=ru_RU"


def snapshots_gallery_url(dataset_id: str, parcel_key: str) -> str:
    return f"/snapshots/{dataset_id}/{parcel_key}/"


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dλ / 2) ** 2
    return 2 * _EARTH_R_M * math.asin(min(1.0, math.sqrt(a)))


def _destination_point(lat: float, lon: float, bearing_deg: float, distance_m: float) -> tuple[float, float]:
    δ = distance_m / _EARTH_R_M
    θ = math.radians(bearing_deg)
    φ1, λ1 = math.radians(lat), math.radians(lon)
    φ2 = math.asin(
        math.sin(φ1) * math.cos(δ) + math.cos(φ1) * math.sin(δ) * math.cos(θ)
    )
    λ2 = λ1 + math.atan2(
        math.sin(θ) * math.sin(δ) * math.cos(φ1),
        math.cos(δ) - math.sin(φ1) * math.sin(φ2),
    )
    return math.degrees(φ2), (math.degrees(λ2) + 540) % 360 - 180


def _exterior_ring(geometry) -> List[Tuple[float, float]]:
    if geometry is None or geometry.is_empty:
        return []
    gt = geometry.geom_type
    if gt == "Polygon":
        return [(float(x), float(y)) for x, y in geometry.exterior.coords]
    if gt == "MultiPolygon":
        polys = sorted(geometry.geoms, key=lambda g: g.area, reverse=True)
        if not polys:
            return []
        return [(float(x), float(y)) for x, y in polys[0].exterior.coords]
    return []


def boundary_view_points(
    geometry,
    *,
    offset_m: float = 22.0,
    min_edge_m: float = 4.0,
    max_points: int = 14,
) -> List[Dict[str, Any]]:
    """
    Точки панорам у каждой значимой грани участка (съёмка с улицы, взгляд на участок).
    """
    ring = _exterior_ring(geometry)
    if len(ring) < 4:
        return []
    try:
        cent = geometry.representative_point()
        c_lat, c_lon = float(cent.y), float(cent.x)
    except Exception:
        return []

    raw_edges: List[Dict[str, Any]] = []
    n = len(ring) - 1
    for i in range(n):
        lon1, lat1 = ring[i]
        lon2, lat2 = ring[i + 1]
        edge_len = _haversine_m(lat1, lon1, lat2, lon2)
        if edge_len < min_edge_m:
            continue
        m_lon = (lon1 + lon2) / 2.0
        m_lat = (lat1 + lat2) / 2.0
        edge_bearing = bearing_geographic_deg(lat1, lon1, lat2, lon2)
        for outward in (edge_bearing + 90.0, edge_bearing - 90.0):
            o_lat, o_lon = _destination_point(m_lat, m_lon, outward % 360.0, offset_m)
            # наружу — дальше от центроида
            if _haversine_m(o_lat, o_lon, c_lat, c_lon) < _haversine_m(m_lat, m_lon, c_lat, c_lon):
                continue
            look_bearing = round(bearing_geographic_deg(o_lat, o_lon, c_lat, c_lon), 1)
            raw_edges.append(
                {
                    "lon": round(o_lon, 6),
                    "lat": round(o_lat, 6),
                    "bearing_deg": look_bearing,
                    "edge_len_m": round(edge_len, 1),
                    "edge_index": i + 1,
                    "edge": [
                        [round(lon1, 6), round(lat1, 6)],
                        [round(lon2, 6), round(lat2, 6)],
                    ],
                }
            )
            break

    if not raw_edges:
        p = geometry.representative_point()
        return [
            {
                "index": 1,
                "lon": round(float(p.x), 6),
                "lat": round(float(p.y), 6),
                "bearing_deg": None,
                "label": "Центр участка",
            }
        ]

    # объединить близкие точки
    merged: List[Dict[str, Any]] = []
    for item in raw_edges:
        dup = False
        for ex in merged:
            if _haversine_m(item["lat"], item["lon"], ex["lat"], ex["lon"]) < 12.0:
                dup = True
                break
        if not dup:
            merged.append(item)
    merged.sort(key=lambda x: -float(x.get("edge_len_m") or 0))
    merged = merged[:max_points]
    out: List[Dict[str, Any]] = []
    for j, item in enumerate(merged, start=1):
        out.append(
            {
                "index": j,
                "lon": item["lon"],
                "lat": item["lat"],
                "bearing_deg": item["bearing_deg"],
                "edge_index": int(item["edge_index"]),
                "edge": item.get("edge"),
                "label": f"Грань {item['edge_index']}",
            }
        )
    return out


def build_photos_ui(dataset_id: str, parcel_key: str, geometry) -> str:
    """Ссылка на папку снимков + URL панорам по граням."""
    parts = [snapshots_gallery_url(dataset_id, parcel_key)]
    for vp in boundary_view_points(geometry):
        parts.append(yandex_widget_url(vp["lon"], vp["lat"]))
    return " ".join(parts)


def assign_panorama_links(
    gdf: gpd.GeoDataFrame,
    dataset_id: str,
    *,
    parcel_keys: set[str] | None = None,
    max_rows: int | None = None,
) -> int:
    """Заполнить «Снимки» для пустых строк: папка + панорамы с граней участка."""
    pi = gdf.columns.get_loc("photos_ui")
    filled = 0
    for pos in range(len(gdf)):
        if max_rows is not None and filled >= max_rows:
            break
        key = str(gdf.iloc[pos]["parcel_key"])
        if parcel_keys is not None and key not in parcel_keys:
            continue
        if str(gdf.iloc[pos, pi] or "").strip():
            continue
        geom = gdf.geometry.iloc[pos]
        try:
            gdf.iloc[pos, pi] = build_photos_ui(dataset_id, key, geom)
            filled += 1
        except Exception:
            continue
    return filled


def load_parcel_gdf(path: Path, dataset_id: str | None = None, simplify_tolerance_deg: float = 6e-5) -> gpd.GeoDataFrame:
    layers = list_layers(path)
    layer = layers[0] if layers else None
    gdf = gpd.read_file(path, layer=layer, engine="pyogrio")
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    else:
        gdf = gdf.to_crs(4326)
    gdf["geometry"] = gdf.geometry.simplify(simplify_tolerance_deg, preserve_topology=True)
    for col in (
        "class_ui", "subclass_ui", "class_corrected_ui", "subclass_corrected_ui",
        "photos_ui", "panorama_status_ui",
    ):
        if col not in gdf.columns:
            gdf[col] = ""
    oid_src = _find_column(gdf, _OID_COLUMN_ALIASES, prefer_data=True)
    if oid_src and oid_src != "oid":
        gdf["oid"] = gdf[oid_src]
    if "oid" not in gdf.columns:
        gdf["oid"] = (gdf.index + 1).astype(int)
    gdf["parcel_key"] = gdf["oid"].astype(str)
    normalize_columns_from_gpkg(gdf)
    if dataset_id:
        assign_panorama_links(gdf, dataset_id)
    return gdf


def _copy_col_if_empty(gdf: gpd.GeoDataFrame, target: str, source: str | None) -> None:
    if not source or source == target or target not in gdf.columns or source not in gdf.columns:
        return
    empty = gdf[target].fillna("").astype(str).str.strip().isin(("", "nan"))
    src = gdf[source].fillna("").astype(str).str.strip()
    src = src.mask(src.str.lower().eq("nan"), "")
    gdf.loc[empty & src.ne(""), target] = src[empty & src.ne("")]


def normalize_columns_from_gpkg(gdf: gpd.GeoDataFrame) -> None:
    """Сопоставить типичные имена колонок GPKG с полями UI."""
    desc_src = _find_column(gdf, _DESC_COLUMN_ALIASES, prefer_data=True)
    if desc_src and desc_src != "generated_land_use":
        gdf["generated_land_use"] = gdf[desc_src].fillna("").astype(str)
    elif desc_src is None and "generated_land_use" not in gdf.columns:
        gdf["generated_land_use"] = ""

    class_src = _find_column(gdf, _CLASS_COLUMN_ALIASES, prefer_data=True)
    _copy_col_if_empty(gdf, "class_ui", class_src)
    sub_src = _find_column(gdf, _SUBCLASS_COLUMN_ALIASES, prefer_data=True)
    _copy_col_if_empty(gdf, "subclass_ui", sub_src)

    corr_cls_src = _find_column(gdf, _CORRECTED_CLASS_ALIASES, prefer_data=True)
    _copy_col_if_empty(gdf, "class_corrected_ui", corr_cls_src)
    corr_sub_src = _find_column(gdf, _CORRECTED_SUBCLASS_ALIASES, prefer_data=True)
    _copy_col_if_empty(gdf, "subclass_corrected_ui", corr_sub_src)

    photos_src = _find_column(gdf, _PHOTOS_COLUMN_ALIASES, prefer_data=True)
    _copy_col_if_empty(gdf, "photos_ui", photos_src)


def _find_column(
    gdf: gpd.GeoDataFrame,
    names: tuple[str, ...],
    *,
    prefer_data: bool = False,
) -> str | None:
    lower_map = {c.lower(): c for c in gdf.columns}
    candidates: list[str] = []
    for name in names:
        if name in gdf.columns:
            candidates.append(name)
        elif name.lower() in lower_map:
            candidates.append(lower_map[name.lower()])
    if not candidates:
        return None
    if prefer_data:
        def _filled(col: str) -> int:
            s = gdf[col].fillna("").astype(str).str.strip()
            return int((s.ne("") & s.ne("nan")).sum())

        return max(candidates, key=_filled)
    return candidates[0]


def parse_photo_coords(photos_ui: str) -> List[Tuple[float, float]]:
    """Уникальные (lon, lat) из столбца «Снимки» (ссылки Яндекс и др.)."""
    seen: set[Tuple[float, float]] = set()
    out: List[Tuple[float, float]] = []
    for m in _LL_RE.finditer(str(photos_ui or "")):
        lon, lat = float(m.group(1)), float(m.group(2))
        key = (round(lon, 6), round(lat, 6))
        if key in seen:
            continue
        seen.add(key)
        out.append((lon, lat))
    return out


def bearing_geographic_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Азимут от точки 1 к точке 2, градусы по часовой от севера [0, 360)."""
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δλ = math.radians(lon2 - lon1)
    y = math.sin(Δλ) * math.cos(φ2)
    x = math.cos(φ1) * math.sin(φ2) - math.sin(φ1) * math.cos(φ2) * math.cos(Δλ)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def view_points_for_parcel(photos_ui: str, geometry) -> List[Dict[str, Any]]:
    """Точки обзора панорам + направление взгляда на участок."""
    try:
        target = geometry.representative_point() if geometry is not None else None
        if target is None or target.is_empty:
            target = None
        tgt_lat = float(target.y) if target is not None else None
        tgt_lon = float(target.x) if target is not None else None
    except Exception:
        tgt_lat = tgt_lon = None

    if _SNAPSHOTS_RE.search(str(photos_ui or "")) and geometry is not None:
        bnd = boundary_view_points(geometry)
        if bnd:
            return bnd

    coords = parse_photo_coords(photos_ui)
    if not coords and geometry is not None:
        return boundary_view_points(geometry)

    points: List[Dict[str, Any]] = []
    for j, (lon, lat) in enumerate(coords, start=1):
        bearing: float | None = None
        if tgt_lat is not None and tgt_lon is not None:
            bearing = round(bearing_geographic_deg(lat, lon, tgt_lat, tgt_lon), 1)
        points.append({"index": j, "lon": lon, "lat": lat, "bearing_deg": bearing, "label": f"Точка {j}"})
    return points


def parcel_panorama_available(geometry, photos_ui: str = "") -> bool:
    """Есть ли панорама у участка."""
    # Ссылки из GPKG аналитиков — доверяем без повторной проверки API
    if parse_photo_coords(photos_ui):
        return True

    from web_app.yandex_panorama import locate_meta

    points = boundary_view_points(geometry) if geometry is not None else []
    for vp in points[:3]:
        if locate_meta(float(vp["lat"]), float(vp["lon"])):
            return True
    return False


def mark_no_panorama(gdf: gpd.GeoDataFrame, pos: int) -> None:
    """Явная метка: панорамы нет (ОС аналитиков)."""
    ci = gdf.columns.get_loc("class_ui")
    si = gdf.columns.get_loc("subclass_ui")
    psi = gdf.columns.get_loc("panorama_status_ui")
    gdf.iloc[pos, ci] = NO_PANORAMA_CLASS
    gdf.iloc[pos, si] = ""
    gdf.iloc[pos, psi] = NO_PANORAMA_STATUS


def gdf_to_geojson_dict(gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
    cols = [
        c
        for c in (
            "parcel_key",
            "oid",
            "cadastral_number",
            "generated_land_use",
            "class_ui",
            "subclass_ui",
            "class_corrected_ui",
            "subclass_corrected_ui",
            "photos_ui",
            "panorama_status_ui",
        )
        if c in gdf.columns
    ]
    sub = gdf[cols + ["geometry"]]
    fc = json.loads(sub.to_json())
    for feat, (_, row) in zip(fc.get("features") or [], gdf.iterrows()):
        props = feat.setdefault("properties", {})
        props["view_points"] = view_points_for_parcel(
            str(row.get("photos_ui") or ""), row.geometry
        )
    return fc


def gdf_to_rows(gdf: gpd.GeoDataFrame) -> List[Dict[str, Any]]:
    out = []
    for pos in range(len(gdf)):
        r = gdf.iloc[pos]
        out.append(
            {
                "parcel_key": str(r.get("parcel_key", "")),
                "oid": int(r["oid"]) if r.get("oid") == r.get("oid") else None,
                "cadastral_number": r.get("cadastral_number"),
                "generated_land_use": str(r.get("generated_land_use") or ""),
                "class_ui": str(r.get("class_ui") or ""),
                "subclass_ui": str(r.get("subclass_ui") or ""),
                "class_corrected_ui": str(r.get("class_corrected_ui") or ""),
                "subclass_corrected_ui": str(r.get("subclass_corrected_ui") or ""),
                "photos_ui": str(r.get("photos_ui") or ""),
                "panorama_status_ui": str(r.get("panorama_status_ui") or ""),
            }
        )
    return out


def gdf_for_export(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Формат колонок как в GPKG аналитиков (out_for_katya)."""
    e = gdf.copy()
    e["class_ID участка"] = e["oid"]
    e["class_Класс"] = e["class_ui"]
    e["class_Подкласс"] = e["subclass_ui"]
    e["class_Откорректированный класс"] = e["class_corrected_ui"]
    e["class_Откорректированный подкласс"] = e["subclass_corrected_ui"]
    e["class_Снимки"] = e["photos_ui"]
    e["class_Описание (исходное)"] = e["generated_land_use"]
    if "cadastral_number" in e.columns:
        e["class_Кадастровый номер"] = e["cadastral_number"]
    e["panorama_status"] = e.get("panorama_status_ui", "")
    keep = [
        "oid", "cadastral_number", "readable_address", "generated_land_use",
        "class_ID участка", "class_Класс", "class_Подкласс",
        "class_Откорректированный класс", "class_Откорректированный подкласс",
        "class_Снимки", "class_Описание (исходное)", "class_Кадастровый номер",
        "panorama_status", "geometry",
    ]
    cols = [c for c in keep if c in e.columns]
    return e[cols]


def persist_gdf_by_sid(upload_dir: Path, sid: str, gdf: gpd.GeoDataFrame, *, layer: str = "parcels") -> Path:
    path = upload_dir / f"{sid}.gpkg"
    gdf_for_export(gdf).to_file(path, driver="GPKG", layer=layer)
    return path


def heuristic_class(desc: str) -> tuple[str, str]:
    """
    Черновой класс/подкласс из поля «Описание (исходное)» (generated_land_use).
    Полный ML-классификатор подключается отдельно; здесь — словарь ключевых фраз.
    """
    if not desc:
        return "", ""
    d = desc.lower().strip()

    if any(x in d for x in ("зелень", "дерев", "парк", "лес", "газон", "сквер", "насажден")):
        if any(x in d for x in ("лес", "лесополос")):
            return "Природные территории", "Лесные массивы"
        if any(x in d for x in ("дорог", "проезж")) and any(x in d for x in ("зелень", "дерев")):
            return "Природные территории", "Растительность вдоль дорог"
        if any(x in d for x in ("благоустро", "сквер")):
            return "Природные территории", "Благоустроенные парки и скверы"
        return "Природные территории", "Парки и скверы"
    if any(x in d for x in ("сельхоз", "распаш", "пашн", "поля ", "поле")):
        return "Природные территории", "Поле"
    if any(x in d for x in ("вод", "река", "пруд", "канал")):
        return "Природные территории", "Водные объекты"

    if any(x in d for x in ("стройк", "кран", "стройплощад")):
        return "Активное строительство", "Стройплощадки с активностью"
    if any(x in d for x in ("незаверш", "заморож", "скелет")):
        return "Незавершенное/приостановленное строительство", "Замороженные объекты"

    if any(x in d for x in ("школ", "детск", "образоват", "вуз", "университет", "научн")):
        return "Активные городские территории", "Образовательные/медицинские/офисные комплексы"

    if any(x in d for x in ("дорог", "транспорт", "развязк", "магистрал", "жд ")):
        return "Активные городские территории", "Интенсивные транспортные коридоры"
    if any(x in d for x in ("асфальт", "грунт", "открыт")):
        return "Активные городские территории", "Интенсивные транспортные коридоры"

    if "мкд" in d or any(x in d for x in ("многоэтаж", "много квартир", "жилой фонд")):
        return "Активные городские территории", "Жилая застройка с активностью"
    if any(x in d for x in ("ижс", "индивидуальн", "коттедж", "частный сектор", "малоэтаж")):
        return "Активные городские территории", "Жилая застройка с активностью"

    if any(x in d for x in ("администр", "муниципал", "государствен", "ведомств")):
        return "Активные городские территории", "Образовательные/медицинские/офисные комплексы"
    if any(x in d for x in ("коммерц", "бц", "бизнес-ц", "торгов", "тц", "трц", "офисн", "ритейл")):
        return "Активные городские территории", "Коммерческие улицы"

    if any(
        x in d
        for x in (
            "гараж",
            "нежил",
            "прочие нежил",
            "склад",
            "ангар",
            "производств",
            "промышлен",
        )
    ):
        if any(x in d for x in ("промышлен", "завод", "фабрик")):
            return "Недоиспользуемые инфраструктурные/городские зоны", "Промзона"
        if "прочие нежил" in d:
            # ~44% правок → urban, но подкласс часто «парковки»; класс оставляем underused
            return "Недоиспользуемые инфраструктурные/городские зоны", "Парковки с низкой загрузкой"
        if "гараж" in d:
            return "Недоиспользуемые инфраструктурные/городские зоны", "Парковки с низкой загрузкой"
        return "Недоиспользуемые инфраструктурные/городские зоны", "Парковки с низкой загрузкой"

    if any(x in d for x in ("заброшен", "разруш", "вандал", "деград")):
        return "Низкоплотная застройка / Деградировавшие объекты", "Заброшенные здания"

    if any(
        x in d
        for x in (
            "жил",
            "застрой",
            "дом",
            "квартал",
            "микрорайон",
            "здани",
            "сооруж",
        )
    ):
        return "Активные городские территории", "Жилая застройка с активностью"

    if any(
        x in d
        for x in (
            "объект",
            "прочие",
            "иной",
            "разное",
            "территор",
            "участ",
            "земел",
            "надел",
        )
    ):
        return "Активные городские территории", "Жилая застройка с активностью"

    return "", ""


def classify_from_description(desc: str) -> tuple[str, str]:
    """Всегда возвращает непустой класс и подкласс."""
    cls, sub = heuristic_class(desc)
    if cls and sub:
        return cls, sub
    if cls and not sub:
        return cls, DEFAULT_SUBCLASS
    return DEFAULT_CLASS, DEFAULT_SUBCLASS


def apply_classify_all(gdf: gpd.GeoDataFrame, *, overwrite: bool = False) -> int:
    """
    Классифицировать все участки по описанию.
    overwrite=False: заполнить только пустые class_ui / subclass_ui.
    overwrite=True: пересчитать все строки.
    """
    ci = gdf.columns.get_loc("class_ui")
    si = gdf.columns.get_loc("subclass_ui")
    filled = 0
    for pos in range(len(gdf)):
        cur_cls = str(gdf.iloc[pos, ci] or "").strip()
        cur_sub = str(gdf.iloc[pos, si] or "").strip()
        if not overwrite and cur_cls and cur_sub:
            continue
        desc = str(gdf.iloc[pos].get("generated_land_use", "") or "").strip()
        if desc.lower() == "nan":
            desc = ""
        cls, sub = classify_from_description(desc)
        if overwrite or not cur_cls:
            gdf.iloc[pos, ci] = cls
        if overwrite or not cur_sub:
            gdf.iloc[pos, si] = sub
        filled += 1
    return filled


def apply_heuristic_full(gdf: gpd.GeoDataFrame) -> None:
    """Перезаписать класс/подкласс из описания для всех строк."""
    apply_classify_all(gdf, overwrite=True)


def apply_heuristic_only_empty(gdf: gpd.GeoDataFrame) -> None:
    """Заполнить пустые class_ui / subclass_ui (гарантированно без дыр)."""
    apply_classify_all(gdf, overwrite=False)


def store_upload(file_storage, upload_dir: Path) -> tuple[str, Path]:
    upload_dir.mkdir(parents=True, exist_ok=True)
    sid = str(uuid4())
    dest = upload_dir / f"{sid}.gpkg"
    file_storage.save(str(dest))
    return sid, dest


def load_gdf_by_sid(upload_dir: Path, sid: str) -> gpd.GeoDataFrame:
    path = upload_dir / f"{sid}.gpkg"
    if not path.is_file():
        raise FileNotFoundError("dataset not found")
    return load_parcel_gdf(path, dataset_id=sid)


def update_rows_from_payload(gdf: gpd.GeoDataFrame, rows: List[Dict[str, Any]]) -> None:
    """Обновить поля таблицы клиента (классы, правки, снимки)."""
    by_key = {str(r.get("parcel_key")): r for r in rows}
    ci = gdf.columns.get_loc("class_ui")
    si = gdf.columns.get_loc("subclass_ui")
    cci = gdf.columns.get_loc("class_corrected_ui")
    sci = gdf.columns.get_loc("subclass_corrected_ui")
    psi = gdf.columns.get_loc("panorama_status_ui")
    pi = gdf.columns.get_loc("photos_ui")
    for pos in range(len(gdf)):
        key = str(gdf.iloc[pos]["parcel_key"])
        pr = by_key.get(key)
        if not pr:
            continue
        gdf.iloc[pos, ci] = str(pr.get("class_ui") or "")
        gdf.iloc[pos, si] = str(pr.get("subclass_ui") or "")
        gdf.iloc[pos, cci] = str(pr.get("class_corrected_ui") or "")
        gdf.iloc[pos, sci] = str(pr.get("subclass_corrected_ui") or "")
        gdf.iloc[pos, pi] = str(pr.get("photos_ui") or "")
        if "panorama_status_ui" in pr:
            gdf.iloc[pos, psi] = str(pr.get("panorama_status_ui") or "")
