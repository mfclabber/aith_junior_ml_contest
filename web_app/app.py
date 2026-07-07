#!/usr/bin/env python3
"""
Веб-интерфейс «Классификатор улиц»: загрузка GPKG, карта, таблица, галерея.

  python -m pip install -r requirements.txt
  python -m web_app.app
  # локально: http://127.0.0.1:5050
  # из сети:  http://<IP_этого_ПК>:5050   (слушает 0.0.0.0 по умолчанию)
  #
  # Порт/хост:  FLASK_HOST=0.0.0.0 FLASK_PORT=5050 WEB_APP_DEBUG=0 python -m web_app.app
  #
  # Превью снимков улицы в галерее (опционально):
  #   MAPILLARY_ACCESS_TOKEN — OAuth-токен Mapillary (graph.mapillary.com)
  #   GOOGLE_MAPS_API_KEY или GOOGLE_STREET_VIEW_KEY — Street View Static + metadata
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_from_directory

from web_app.classify_service import apply_classification
from web_app.gpkg_io import (
    assign_panorama_links,
    bearing_geographic_deg,
    gdf_to_geojson_dict,
    gdf_to_rows,
    load_gdf_by_sid,
    load_parcel_gdf,
    persist_gdf_by_sid,
    store_upload,
    update_rows_from_payload,
)
from web_app.panorama_crop import render_context_crop_jpeg
from web_app.panorama_preview import preview_images_for_point, providers_configured
from web_app.snapshots import ensure_parcel_snapshots, snapshot_dir, yandex_maps_panorama_url
from web_app.yandex_panorama import render_equirect_jpeg

ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = ROOT / "web_app" / "uploads"
MAX_UPLOAD_MB = 120

# Продакшен: только ML по панораме (6 классов, vlm6_oss probe)
os.environ.setdefault("CLASSIFIER_TAXONOMY", "6")
os.environ.setdefault("CLASSIFIER_MODE", "ml")
os.environ.setdefault("CLASSIFIER_PROBE", "clip_probe_vlm6_oss.pt")
os.environ.setdefault("CLASSIFIER_FALLBACK_HEURISTIC", "0")
os.environ.setdefault("ML_MIN_CONFIDENCE", "0.28")
os.environ.setdefault("CLASSIFY_MAX_ROWS", "500")

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


@app.route("/")
def index():
    return render_template("index.html")


@app.post("/api/upload")
def api_upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "Файл не передан"}), 400
    if not f.filename.lower().endswith(".gpkg"):
        return jsonify({"error": "Нужен файл .gpkg (GeoPackage)"}), 400
    try:
        sid, path = store_upload(f, UPLOAD_DIR)
        gdf = load_parcel_gdf(path, dataset_id=sid)
        fc = gdf_to_geojson_dict(gdf)
        rows = gdf_to_rows(gdf)
        return jsonify({"dataset_id": sid, "geojson": fc, "rows": rows, "count": len(rows)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/classify")
def api_classify():
    try:
        payload = request.get_json(force=True, silent=False) or {}
        sid = payload.get("dataset_id")
        rows_in = payload.get("rows")
        if not sid:
            return jsonify({"error": "dataset_id обязателен"}), 400
        gdf = load_gdf_by_sid(UPLOAD_DIR, sid)
        if isinstance(rows_in, list):
            update_rows_from_payload(gdf, rows_in)
        overwrite = bool(payload.get("overwrite_all", False))
        mode = str(payload.get("mode") or os.environ.get("CLASSIFIER_MODE", "ml"))
        max_rows = int(payload.get("max_rows") or os.environ.get("CLASSIFY_MAX_ROWS", "500"))
        clf_stats = apply_classification(
            gdf, mode=mode, overwrite=overwrite, max_rows=max_rows  # type: ignore[arg-type]
        )
        # Только строки без «Снимков» — не пересчитывать тысячи участков при каждом ▶
        assign_panorama_links(gdf, sid, max_rows=200)
        persist_gdf_by_sid(UPLOAD_DIR, sid, gdf)
        fc = gdf_to_geojson_dict(gdf)
        rows = gdf_to_rows(gdf)
        return jsonify({"geojson": fc, "rows": rows, "classify_stats": clf_stats})
    except FileNotFoundError:
        return jsonify({"error": "Набор данных не найден; загрузите снова"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/export-gpkg")
def api_export_gpkg():
    """Экспорт GPKG с колонками аналитиков (включая откорректированные)."""
    try:
        payload = request.get_json(force=True, silent=False) or {}
        sid = payload.get("dataset_id")
        rows_in = payload.get("rows")
        if not sid:
            return jsonify({"error": "dataset_id обязателен"}), 400
        gdf = load_gdf_by_sid(UPLOAD_DIR, sid)
        if isinstance(rows_in, list):
            update_rows_from_payload(gdf, rows_in)
        out_path = persist_gdf_by_sid(UPLOAD_DIR, sid, gdf)
        return send_from_directory(
            out_path.parent,
            out_path.name,
            as_attachment=True,
            download_name=f"parcels_{sid[:8]}.gpkg",
        )
    except FileNotFoundError:
        return jsonify({"error": "Набор данных не найден"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/panorama-previews")
def api_panorama_previews():
    """Превью улицы по точке: Яндекс (локальная генерация) + Mapillary/Google (опционально)."""
    try:
        lon = float(request.args.get("lon", ""))
        lat = float(request.args.get("lat", ""))
    except (TypeError, ValueError):
        return jsonify({"error": "Укажите числовые lon и lat"}), 400
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        return jsonify({"error": "lon/lat вне допустимого диапазона"}), 400
    tgt_lat = request.args.get("tgt_lat")
    tgt_lon = request.args.get("tgt_lon")
    bearing = request.args.get("bearing")
    try:
        tgt_lat_f = float(tgt_lat) if tgt_lat not in (None, "") else None
        tgt_lon_f = float(tgt_lon) if tgt_lon not in (None, "") else None
        bearing_f = float(bearing) if bearing not in (None, "") else None
    except (TypeError, ValueError):
        tgt_lat_f = tgt_lon_f = bearing_f = None
    images = preview_images_for_point(
        lat, lon, tgt_lat=tgt_lat_f, tgt_lon=tgt_lon_f, bearing_deg=bearing_f
    )
    return jsonify({"images": images, "providers": providers_configured()})


@app.get("/api/yandex-panorama.jpg")
def api_yandex_panorama_jpg():
    """JPEG панорамы Яндекса, собранный из тайлов по координате."""
    try:
        lon = float(request.args.get("lon", ""))
        lat = float(request.args.get("lat", ""))
    except (TypeError, ValueError):
        return jsonify({"error": "Укажите числовые lon и lat"}), 400
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        return jsonify({"error": "lon/lat вне допустимого диапазона"}), 400
    try:
        zoom = int(request.args.get("zoom", "0"))
    except (TypeError, ValueError):
        zoom = 0
    try:
        w = int(request.args.get("w", "900"))
    except (TypeError, ValueError):
        w = 900
    w = max(640, min(1600, w))
    jpg = render_equirect_jpeg(lat=lat, lon=lon, zoom=zoom, max_w=w, quality=88)
    if not jpg:
        return jsonify({"error": "Панорама Яндекса не найдена для этой точки"}), 404
    return Response(jpg, mimetype="image/jpeg", headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/yandex-panorama-crop.jpg")
def api_yandex_panorama_crop_jpg():
    """Перспективный кадр на участок (не 360°)."""
    try:
        lon = float(request.args.get("lon", ""))
        lat = float(request.args.get("lat", ""))
    except (TypeError, ValueError):
        return jsonify({"error": "Укажите числовые lon и lat"}), 400
    bearing = request.args.get("bearing")
    tgt_lat = request.args.get("tgt_lat")
    tgt_lon = request.args.get("tgt_lon")
    try:
        heading = float(bearing) if bearing not in (None, "") else None
        if heading is None and tgt_lat not in (None, "") and tgt_lon not in (None, ""):
            heading = bearing_geographic_deg(lat, lon, float(tgt_lat), float(tgt_lon))
    except (TypeError, ValueError):
        heading = 0.0
    if heading is None:
        heading = 0.0
    jpg = render_context_crop_jpeg(lat=lat, lon=lon, heading_deg=heading)
    if not jpg:
        return jsonify({"error": "Не удалось собрать кадр панорамы"}), 404
    return Response(jpg, mimetype="image/jpeg", headers={"Cache-Control": "public, max-age=3600"})


@app.post("/api/vlm-infer")
def api_vlm_infer():
    """VLM (PaliGemma+LoRA): класс UTT + evidence-bbox.

    Принимает либо загруженный файл `file`, либо JSON/форму с lat/lon (+ bearing или
    tgt_lat/tgt_lon), тогда перспективный кроп собирается из панорамы Яндекса.
    """
    from web_app.vlm_classifier import checkpoints_available, get_vlm

    if not checkpoints_available():
        return jsonify({"error": "VLM-чекпойнт не найден (checkpoints/paligemma_lora)"}), 503
    vlm = get_vlm()
    if vlm is None:
        return jsonify({"error": "Не удалось загрузить VLM"}), 500

    f = request.files.get("file")
    if f and f.filename:
        try:
            return jsonify(vlm.predict_jpeg(f.read()))
        except Exception as e:  # noqa: BLE001
            return jsonify({"error": str(e)}), 500

    payload = request.get_json(silent=True) or request.form
    try:
        lon = float(payload.get("lon"))
        lat = float(payload.get("lat"))
    except (TypeError, ValueError):
        return jsonify({"error": "Передайте файл `file` или числовые lat/lon"}), 400
    bearing = payload.get("bearing")
    tgt_lat = payload.get("tgt_lat")
    tgt_lon = payload.get("tgt_lon")
    try:
        heading = float(bearing) if bearing not in (None, "") else None
        if heading is None and tgt_lat not in (None, "") and tgt_lon not in (None, ""):
            heading = bearing_geographic_deg(lat, lon, float(tgt_lat), float(tgt_lon))
    except (TypeError, ValueError):
        heading = 0.0
    if heading is None:
        heading = 0.0
    jpg = render_context_crop_jpeg(lat=lat, lon=lon, heading_deg=heading)
    if not jpg:
        return jsonify({"error": "Не удалось собрать кадр панорамы"}), 404
    try:
        result = vlm.predict_jpeg(jpg)
        result["heading_deg"] = heading
        return jsonify(result)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500


@app.get("/snapshots/<sid>/<parcel_key>/")
def snapshots_gallery_page(sid: str, parcel_key: str):
    try:
        gdf = load_gdf_by_sid(UPLOAD_DIR, sid)
    except FileNotFoundError:
        return "Набор данных не найден", 404
    row = gdf[gdf["parcel_key"].astype(str) == str(parcel_key)]
    if row.empty:
        return "Участок не найден", 404
    geom = row.geometry.iloc[0]
    photos = str(row.iloc[0].get("photos_ui") or "")
    meta = ensure_parcel_snapshots(UPLOAD_DIR, sid, parcel_key, geom, photos)
    return render_template(
        "snapshots.html",
        dataset_id=sid,
        parcel_key=parcel_key,
        files=meta.get("files") or [],
        view_points=meta.get("view_points") or [],
    )


@app.get("/api/parcel-snapshots/<sid>/<parcel_key>/<filename>")
def api_parcel_snapshot_file(sid: str, parcel_key: str, filename: str):
    if ".." in filename or "/" in filename:
        return jsonify({"error": "invalid name"}), 400
    d = snapshot_dir(UPLOAD_DIR, sid, parcel_key)
    path = d / filename
    if not path.is_file():
        return jsonify({"error": "not found"}), 404
    return send_from_directory(d, filename, mimetype="image/jpeg", max_age=3600)


@app.get("/api/yandex-map-link")
def api_yandex_map_link():
    try:
        lon = float(request.args.get("lon", ""))
        lat = float(request.args.get("lat", ""))
        bearing = float(request.args.get("bearing", "0"))
    except (TypeError, ValueError):
        return jsonify({"error": "lon/lat обязательны"}), 400
    return jsonify({"url": yandex_maps_panorama_url(lon, lat, bearing)})


@app.delete("/api/dataset/<sid>")
def api_delete_dataset(sid: str):
    p = UPLOAD_DIR / f"{sid}.gpkg"
    if p.is_file():
        p.unlink()
    return jsonify({"ok": True})


def _guess_lan_ip() -> str | None:
    """Приблизительный «основной» IPv4 (один адрес)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)
        s.connect(("10.254.254.254", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip if ip and not ip.startswith("127.") else None
    except Exception:
        return None


def _global_ipv4_candidates() -> list[str]:
    """Все глобальные IPv4 интерфейсов — клиенту нужен адрес из своей подсети (не Docker/WSL по возможности)."""
    ips: list[str] = []
    try:
        p = subprocess.run(
            ["ip", "-4", "-br", "addr", "show", "scope", "global"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        for line in (p.stdout or "").strip().splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            iface, _state, cidr = parts[0], parts[1], parts[2]
            if iface == "lo":
                continue
            addr = cidr.split("/")[0].strip()
            if addr and addr not in ips:
                ips.append(addr)
    except Exception:
        pass
    if not ips:
        one = _guess_lan_ip()
        return [one] if one else []
    # Чуть приоритетнее «обычная» LAN 10/192.168 перед одиночными docker-хостами
    def sort_key(a: str) -> tuple[int, str]:
        if a.startswith("192.168."):
            return (0, a)
        if a.startswith("10."):
            return (1, a)
        if a.startswith("172."):
            return (2, a)
        return (3, a)

    ips.sort(key=sort_key)
    return ips


def main() -> None:
    parser = argparse.ArgumentParser(description="Street classifier web UI")
    parser.add_argument(
        "--host",
        default=os.environ.get("FLASK_HOST", "0.0.0.0"),
        help="0.0.0.0 — доступ по IP сервера в LAN (по умолчанию)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("FLASK_PORT", "5050")),
        help="Порт HTTP",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Режим отладки Flask (перезагрузка при изменении кода)",
    )
    args = parser.parse_args()

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    debug_mode = args.debug or os.environ.get("WEB_APP_DEBUG", "").lower() in (
        "1",
        "true",
        "yes",
    )

    prov = providers_configured()
    parts = []
    if prov["mapillary"]:
        parts.append("Mapillary")
    if prov["google_street_view"]:
        parts.append("Google Street View")
    prev_hint = (
        "превью панорам: " + ", ".join(parts)
        if parts
        else "превью панорам выключены (задайте MAPILLARY_ACCESS_TOKEN и/или GOOGLE_MAPS_API_KEY)"
    )

    candidates = _global_ipv4_candidates()
    print(f"Сервер слушает http://{args.host}:{args.port} (все интерфейсы)")
    print(f"  {prev_hint}")
    print(f"  локально:           http://127.0.0.1:{args.port}")
    if candidates:
        print("  попробуйте с другого ПК в той же сети (один из адресов):")
        for ip in candidates[:8]:
            print(f"                       http://{ip}:{args.port}")
        if len(candidates) > 8:
            print(f"                       … и ещё {len(candidates) - 8} адрес(ов)")
    print()
    print(
        "Если по IP не открывается: (1) проверьте облачную «security group» / inbound TCP для этого порта; "
        "(2) роутер/Wi‑Fi isolation; (3) запустите на другом порту, например:\n"
        f"    FLASK_PORT=8080 python -m web_app.app --port 8080\n"
        f"    или: sudo ufw allow {args.port}/tcp && sudo ufw reload"
    )

    # threaded=True — параллельная отдача /api/yandex-panorama*.jpg (иначе 11 граней встают в очередь)
    app.run(host=args.host, port=args.port, debug=debug_mode, threaded=True)


if __name__ == "__main__":
    main()
