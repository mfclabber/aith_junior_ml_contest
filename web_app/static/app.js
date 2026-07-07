/* global L */

(function () {
  "use strict";

  let map = null;
  let parcelLayer = null;
  let viewMarkersLayer = null;
  let datasetId = null;
  let selectedKey = null;
  let galleryReqId = 0;
  let currentPanoImg = null;
  let currentPanoViewport = null;
  let uploadBusy = false;
  let currentViewPoints = [];
  let activeViewPointIndex = 0;
  let syncViewPointTabs = null;

  const EDGE_PALETTE = [
    "#e91e8c",
    "#1565c0",
    "#2e7d32",
    "#ef6c00",
    "#6a1b9a",
    "#00838f",
    "#c62828",
    "#5d4037",
    "#4527a0",
    "#558b2f",
    "#ad1457",
    "#0277bd",
    "#f9a825",
    "#4e342e",
  ];

  const el = (id) => document.getElementById(id);

  function escapeHtml(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function splitUrls(raw) {
    if (!raw || !String(raw).trim()) return [];
    return String(raw)
      .replace(/\\/g, "/")
      .split(/[\s,;|]+/)
      .map((x) => x.trim())
      .filter((x) => /^https?:\/\//i.test(x));
  }

  function isDirectImageUrl(url) {
    return /\.(jpe?g|png|gif|webp|bmp|svg)(\?|#|$)/i.test(url);
  }

  /** Возвращает картинку (Street View Static и т.п.). */
  function isRasterImageUrl(url) {
    if (isDirectImageUrl(url)) return true;
    return /maps\.googleapis\.com\/maps\/api\/streetview\?/i.test(url);
  }

  /** ll=lon,lat или ll=lon%2Clat (Яндекс и др.) */
  function parseLlFromUrl(url) {
    const m = url.match(/[?&]ll=([\d.+-]+)(?:%2C|,)([\d.+-]+)/i);
    if (!m) return null;
    const lon = parseFloat(m[1]);
    const lat = parseFloat(m[2]);
    if (Number.isNaN(lat) || Number.isNaN(lon)) return null;
    return { lon, lat };
  }

  /** Все уникальные точки панорам из текста столбца «Снимки». */
  function isSnapshotsGalleryUrl(url) {
    return /^\/snapshots\/[^/\s]+\/[^/\s]+\/?/i.test(String(url || "").trim());
  }

  function splitPhotosField(raw) {
    const text = String(raw || "").trim();
    if (!text) return { gallery: null, rest: "" };
    const parts = text.split(/\s+/);
    const gallery = parts.find((p) => isSnapshotsGalleryUrl(p)) || null;
    const rest = parts.filter((p) => !isSnapshotsGalleryUrl(p)).join(" ");
    return { gallery, rest };
  }

  function offsetLatLon(lat, lon, bearingDeg, distM) {
    const R = 6371000;
    const δ = distM / R;
    const θ = (bearingDeg * Math.PI) / 180;
    const φ1 = (lat * Math.PI) / 180;
    const λ1 = (lon * Math.PI) / 180;
    const φ2 = Math.asin(
      Math.sin(φ1) * Math.cos(δ) + Math.cos(φ1) * Math.sin(δ) * Math.cos(θ)
    );
    const λ2 =
      λ1 +
      Math.atan2(
        Math.sin(θ) * Math.sin(δ) * Math.cos(φ1),
        Math.cos(δ) - Math.sin(φ1) * Math.sin(φ2)
      );
    return { lat: (φ2 * 180) / Math.PI, lon: (((λ2 * 180) / Math.PI + 540) % 360) - 180 };
  }

  function parseAllLlFromText(raw) {
    const seen = new Set();
    const out = [];
    const re = /[?&]ll=([\d.+-]+)(?:%2C|,)([\d.+-]+)/gi;
    let m;
    while ((m = re.exec(String(raw || ""))) !== null) {
      const lon = parseFloat(m[1]);
      const lat = parseFloat(m[2]);
      if (Number.isNaN(lat) || Number.isNaN(lon)) continue;
      const key = lon.toFixed(5) + "," + lat.toFixed(5);
      if (seen.has(key)) continue;
      seen.add(key);
      out.push({ lon, lat });
    }
    return out;
  }

  function bearingGeographicDeg(lat1, lon1, lat2, lon2) {
    const φ1 = (lat1 * Math.PI) / 180;
    const φ2 = (lat2 * Math.PI) / 180;
    const Δλ = ((lon2 - lon1) * Math.PI) / 180;
    const y = Math.sin(Δλ) * Math.cos(φ2);
    const x =
      Math.cos(φ1) * Math.sin(φ2) - Math.sin(φ1) * Math.cos(φ2) * Math.cos(Δλ);
    return (Math.atan2(y, x) * (180 / Math.PI) + 360) % 360;
  }

  function parcelTargetLatLon(feat) {
    if (!feat || !feat.geometry) return null;
    const g = feat.geometry;
    let ring = null;
    if (g.type === "Point") return { lat: g.coordinates[1], lon: g.coordinates[0] };
    if (g.type === "Polygon") ring = g.coordinates[0];
    else if (g.type === "MultiPolygon") ring = g.coordinates[0] && g.coordinates[0][0];
    if (!ring || !ring.length) return null;
    let slat = 0;
    let slon = 0;
    ring.forEach((c) => {
      slon += c[0];
      slat += c[1];
    });
    return { lat: slat / ring.length, lon: slon / ring.length };
  }

  function featureForKey(key) {
    if (!parcelLayer || !key) return null;
    let found = null;
    parcelLayer.eachLayer((layer) => {
      if (found) return;
      const p = layer.feature && layer.feature.properties;
      if (!p) return;
      const k = String(p.parcel_key != null ? p.parcel_key : p.oid);
      if (k === String(key)) found = layer.feature;
    });
    return found;
  }

  function viewPointsForParcel(key, photosText) {
    const feat = featureForKey(key);
    const meta = (feat && feat.properties && feat.properties.view_points) || [];
    const tgt = parcelTargetLatLon(feat);
    if (Array.isArray(meta) && meta.length) {
      return meta
        .map((vp, i) => {
          const lon = Number(vp.lon);
          const lat = Number(vp.lat);
          if (Number.isNaN(lat) || Number.isNaN(lon)) return null;
          let bearing = vp.bearing_deg != null ? vp.bearing_deg : null;
          if (bearing == null && tgt) {
            bearing = Math.round(bearingGeographicDeg(lat, lon, tgt.lat, tgt.lon) * 10) / 10;
          }
          return {
            index: vp.index != null ? vp.index : i + 1,
            lon,
            lat,
            bearing_deg: bearing,
            edge_index: vp.edge_index != null ? vp.edge_index : null,
            edge: vp.edge || null,
            label: vp.label || null,
          };
        })
        .filter(Boolean);
    }
    const parsed = parseAllLlFromText(photosText);
    if (!parsed.length) return [];
    return parsed.map((p, i) => {
      let bearing = null;
      if (tgt) {
        bearing = Math.round(bearingGeographicDeg(p.lat, p.lon, tgt.lat, tgt.lon) * 10) / 10;
      }
      return {
        index: i + 1,
        lon: p.lon,
        lat: p.lat,
        bearing_deg: bearing,
        edge_index: null,
        edge: null,
        label: "Точка " + (i + 1),
      };
    });
  }

  function exteriorRingLatLng(feat) {
    if (!feat || !feat.geometry) return [];
    const g = feat.geometry;
    let ring = null;
    if (g.type === "Polygon") ring = g.coordinates[0];
    else if (g.type === "MultiPolygon" && g.coordinates[0]) ring = g.coordinates[0][0];
    if (!ring || ring.length < 2) return [];
    return ring.map((c) => [c[1], c[0]]);
  }

  function haversineM(lat1, lon1, lat2, lon2) {
    const R = 6371000;
    const φ1 = (lat1 * Math.PI) / 180;
    const φ2 = (lat2 * Math.PI) / 180;
    const Δφ = ((lat2 - lat1) * Math.PI) / 180;
    const Δλ = ((lon2 - lon1) * Math.PI) / 180;
    const a =
      Math.sin(Δφ / 2) * Math.sin(Δφ / 2) +
      Math.cos(φ1) * Math.cos(φ2) * Math.sin(Δλ / 2) * Math.sin(Δλ / 2);
    return 2 * R * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  }

  /** Привязать точку панорамы к ближайшему отрезку контура (если edge не пришёл с сервера). */
  function enrichViewPointsWithEdges(viewPoints, feat) {
    const ring = exteriorRingLatLng(feat);
    if (!ring.length) return viewPoints;
    return viewPoints.map((vp) => {
      if (vp.edge && vp.edge.length === 2) return vp;
      const lat = Number(vp.lat);
      const lon = Number(vp.lon);
      if (Number.isNaN(lat) || Number.isNaN(lon)) return vp;
      let best = null;
      let bestScore = Infinity;
      for (let i = 0; i < ring.length - 1; i++) {
        const a = ring[i];
        const b = ring[i + 1];
        const mlat = (a[0] + b[0]) / 2;
        const mlon = (a[1] + b[1]) / 2;
        const d = haversineM(lat, lon, mlat, mlon);
        if (d < bestScore) {
          bestScore = d;
          best = {
            edge_index: i + 1,
            edge: [
              [a[1], a[0]],
              [b[1], b[0]],
            ],
          };
        }
      }
      if (!best) return vp;
      return {
        ...vp,
        edge_index: best.edge_index,
        edge: best.edge,
        label: vp.label || "Грань " + best.edge_index,
      };
    });
  }

  function edgeColor(idx) {
    return EDGE_PALETTE[idx % EDGE_PALETTE.length];
  }

  function viewPointCaption(vp, tabIdx) {
    const n = tabIdx + 1;
    if (vp.edge_index != null) {
      return "№" + n + " — грань участка " + vp.edge_index + " (на карте " + n + ")";
    }
    return vp.label || "Точка " + (vp.index != null ? vp.index : n);
  }

  function updateGalleryHintForViewPoint(vp, tabIdx, total) {
    const hint = el("galleryHint");
    if (!hint) return;
    const cap = viewPointCaption(vp, tabIdx);
    if (total > 1) {
      hint.textContent =
        "Сейчас: " +
        cap +
        ". На карте цветной отрезок — эта грань; маячок с номером " +
        (tabIdx + 1) +
        " — точка съёмки. Переключайте вкладки или кликайте маячки.";
    } else {
      hint.textContent =
        "Сейчас: " + cap + ". Маячок на карте — точка съёмки, стрелка — взгляд на участок.";
    }
  }

  function setActiveViewPointIndex(idx) {
    if (!currentViewPoints.length) return;
    const i = Math.max(0, Math.min(idx, currentViewPoints.length - 1));
    activeViewPointIndex = i;
    renderViewBeacons(currentViewPoints, i);
    if (typeof syncViewPointTabs === "function") syncViewPointTabs(i);
    updateGalleryHintForViewPoint(currentViewPoints[i], i, currentViewPoints.length);
  }

  function clearViewBeacons() {
    if (viewMarkersLayer) {
      map.removeLayer(viewMarkersLayer);
      viewMarkersLayer = null;
    }
  }

  function beaconIcon(bearingDeg, num, color, active) {
    const hasBearing = bearingDeg != null && !Number.isNaN(Number(bearingDeg));
    const rot = hasBearing ? "transform:rotate(" + bearingDeg + "deg)" : "";
    const arrow = hasBearing
      ? '<span class="pano-beacon-arrow" style="' + rot + ";color:" + color + '">▲</span>'
      : "";
    const cls = "pano-beacon" + (active ? " pano-beacon--active" : "");
    return L.divIcon({
      className: "pano-beacon-wrap",
      html:
        '<div class="' +
        cls +
        '" style="--beacon-color:' +
        color +
        '">' +
        arrow +
        '<span class="pano-beacon-num">' +
        num +
        "</span>" +
        '<span class="pano-beacon-dot"></span></div>',
      iconSize: [34, 34],
      iconAnchor: [17, 17],
    });
  }

  function edgeToLatLngs(edge) {
    if (!edge || edge.length !== 2) return null;
    return [
      [edge[0][1], edge[0][0]],
      [edge[1][1], edge[1][0]],
    ];
  }

  function renderViewBeacons(viewPoints, activeIdx) {
    clearViewBeacons();
    if (!map || !viewPoints || !viewPoints.length) return;
    const active = activeIdx == null ? 0 : activeIdx;
    viewMarkersLayer = L.layerGroup();

    viewPoints.forEach((vp, i) => {
      const latlngs = edgeToLatLngs(vp.edge);
      if (latlngs) {
        const color = edgeColor(i);
        const isActive = i === active;
        L.polyline(latlngs, {
          color,
          weight: isActive ? 7 : 3,
          opacity: isActive ? 1 : 0.45,
          lineCap: "round",
          className: isActive ? "pano-edge-line pano-edge-line--active" : "pano-edge-line",
        }).addTo(viewMarkersLayer);
      }
    });

    viewPoints.forEach((vp, i) => {
      const lat = Number(vp.lat);
      const lon = Number(vp.lon);
      if (Number.isNaN(lat) || Number.isNaN(lon)) return;
      const bearing = vp.bearing_deg;
      const color = edgeColor(i);
      const num = i + 1;
      const m = L.marker([lat, lon], {
        icon: beaconIcon(bearing, num, color, i === active),
        zIndexOffset: i === active ? 800 : 400 + i,
      });
      const cap = viewPointCaption(vp, i);
      m.bindTooltip(cap + " — клик, чтобы открыть панораму", {
        direction: "top",
        offset: [0, -12],
      });
      m.on("click", (e) => {
        L.DomEvent.stopPropagation(e);
        setActiveViewPointIndex(i);
      });
      viewMarkersLayer.addLayer(m);
      if (bearing != null && !Number.isNaN(Number(bearing))) {
        const end = offsetLatLon(lat, lon, Number(bearing), 38);
        L.polyline(
          [
            [lat, lon],
            [end.lat, end.lon],
          ],
          { color, weight: i === active ? 4 : 2, opacity: i === active ? 0.95 : 0.55, dashArray: "6 4" }
        ).addTo(viewMarkersLayer);
      }
    });

    viewMarkersLayer.addTo(map);

    const vp = viewPoints[active];
    const ll = edgeToLatLngs(vp && vp.edge);
    if (ll) {
      try {
        map.fitBounds(L.latLngBounds(ll), { padding: [56, 56], maxZoom: 18, animate: true });
      } catch (_) {}
    }
  }

  function yandexMapsAtPointUrl(lon, lat, bearingDeg) {
    let u =
      "https://yandex.ru/maps/?ll=" +
      encodeURIComponent(lon + "," + lat) +
      "&z=19&l=stv&pt=" +
      encodeURIComponent(lon + "," + lat) +
      ",pm2rdm";
    if (bearingDeg != null && !Number.isNaN(Number(bearingDeg))) {
      u += "&panorama%5Bdirection%5D=" + Math.round(Number(bearingDeg));
    }
    return u;
  }

  /** Режим панорам Яндекса (встраиваемый виджет или полная карта). */
  function isYandexStvUrl(url) {
    return /yandex\.ru\/(maps|map-widget)/i.test(url) && /(?:^|[?&])l=stv\b/i.test(url);
  }

  /** URL для iframe: виджет с панорамой; для старых ссылок /maps/ — эквивалентный widget. */
  function yandexPanoramaIframeSrc(url) {
    if (!isYandexStvUrl(url)) return null;
    const ll = parseLlFromUrl(url);
    if (!ll) return null;
    const llq = encodeURIComponent(ll.lon + "," + ll.lat);
    return `https://yandex.ru/map-widget/v1/?ll=${llq}&z=18&l=stv&lang=ru_RU`;
  }

  /** Полная страница карт (новая вкладка), если нужна не виджет-ссылка. */
  function yandexMapsFullPanoramaUrl(url) {
    const ll = parseLlFromUrl(url);
    if (!ll || !isYandexStvUrl(url)) return url;
    return `https://yandex.ru/maps/?ll=${ll.lon}%2C${ll.lat}&z=18&l=stv`;
  }

  function osmStaticThumb(lat, lon) {
    return (
      "https://staticmap.openstreetmap.de/staticmap.php?center=" +
      encodeURIComponent(lat + "," + lon) +
      "&zoom=17&size=320x200&maptype=mapnik"
    );
  }

  function safeHostname(url) {
    try {
      return new URL(url).hostname;
    } catch (_) {
      return "ссылка";
    }
  }

  function baseStyle() {
    return {
      color: "#b8860b",
      weight: 1.5,
      opacity: 0.95,
      fillColor: "#ffdc33",
      fillOpacity: 0.38,
    };
  }

  function selectedStyle() {
    return {
      color: "#e91e8c",
      weight: 4,
      opacity: 1,
      fillColor: "#ffeb66",
      fillOpacity: 0.55,
    };
  }

  function fixLeafletIconPaths() {
    const base = "/static/vendor/leaflet/images/";
    delete L.Icon.Default.prototype._getIconUrl;
    L.Icon.Default.mergeOptions({
      iconRetinaUrl: base + "marker-icon-2x.png",
      iconUrl: base + "marker-icon.png",
      shadowUrl: base + "marker-shadow.png",
    });
  }

  function scheduleMapResize() {
    if (!map) return;
    const run = () => {
      try {
        map.invalidateSize(true);
      } catch (_) {}
    };
    requestAnimationFrame(run);
    setTimeout(run, 120);
    setTimeout(run, 500);
  }

  function showMapBootError(msg) {
    const mapEl = el("map");
    if (!mapEl) return;
    mapEl.innerHTML =
      '<div class="map-boot-error"><strong>Карта не запустилась</strong><p>' +
      escapeHtml(msg) +
      "</p><p>Обновите страницу (Ctrl+F5). Адрес: <code>" +
      escapeHtml(window.location.href) +
      "</code></p></div>";
  }

  function initMap() {
    const mapEl = el("map");
    if (!mapEl) return;
    if (typeof L === "undefined") {
      showMapBootError("Не загружена библиотека Leaflet (проверьте /static/vendor/leaflet/).");
      return;
    }
    try {
      fixLeafletIconPaths();
      map = L.map("map", { zoomControl: true, attributionControl: false }).setView([55.75, 37.62], 12);

      const osm = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 19,
        attribution: "",
      });
      const imagery = L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        { maxZoom: 19, attribution: "" }
      );
      const labels = L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        { maxZoom: 19, opacity: 0.65, attribution: "" }
      );

      osm.addTo(map);
      imagery.addTo(map);
      labels.addTo(map);

      L.control
        .layers(
          { "Спутник (Esri)": imagery, "Схема (OSM)": osm },
          { Подписи: labels },
          { collapsed: true }
        )
        .addTo(map);

      scheduleMapResize();
      window.addEventListener("resize", scheduleMapResize);
    } catch (err) {
      showMapBootError(err && err.message ? err.message : String(err));
    }
  }

  function loadParcels(fc) {
    if (parcelLayer) {
      map.removeLayer(parcelLayer);
      parcelLayer = null;
    }
    parcelLayer = L.geoJSON(fc, {
      style: baseStyle,
      onEachFeature(feature, layer) {
        layer.on({
          click(e) {
            L.DomEvent.stopPropagation(e);
            const p = feature.properties || {};
            const key = String(p.parcel_key != null ? p.parcel_key : p.oid);
            selectByKey(key);
          },
        });
      },
    }).addTo(map);
    try {
      map.fitBounds(parcelLayer.getBounds(), { padding: [24, 24], maxZoom: 15 });
      scheduleMapResize();
    } catch (_) {}
  }

  function resetStyles() {
    if (!parcelLayer) return;
    parcelLayer.eachLayer((layer) => layer.setStyle(baseStyle()));
  }

  function highlightOnMap(key) {
    selectedKey = key;
    resetStyles();
    if (!parcelLayer || !key) return;
    parcelLayer.eachLayer((layer) => {
      const p = layer.feature && layer.feature.properties;
      if (!p) return;
      const k = String(p.parcel_key != null ? p.parcel_key : p.oid);
      if (k === String(key)) {
        layer.setStyle(selectedStyle());
        map.fitBounds(layer.getBounds(), { padding: [48, 48], maxZoom: 17 });
      }
    });
  }

  function renderTable(rows) {
    const tb = document.querySelector("#report tbody");
    tb.innerHTML = "";
    rows.forEach((r) => {
      const tr = document.createElement("tr");
      const pk = String(r.parcel_key);
      tr.dataset.parcelKey = pk;
      const cells = [
        pk,
        r.class_ui || "",
        r.subclass_ui || "",
        r.class_corrected_ui || "",
        r.subclass_corrected_ui || "",
        r.panorama_status_ui || "",
        r.photos_ui || "",
        r.generated_land_use || "",
        r.cadastral_number != null ? String(r.cadastral_number) : "",
      ];
      tr.dataset.photosUi = r.photos_ui || "";
      cells.forEach((text, i) => {
        const td = document.createElement("td");
        if (i === 6) {
          const split = splitPhotosField(text);
          if (split.gallery) {
            const a = document.createElement("a");
            a.className = "photos-gallery-link";
            a.href = split.gallery;
            a.target = "_blank";
            a.rel = "noopener noreferrer";
            a.textContent = "Папка снимков участка";
            td.appendChild(a);
            const span = document.createElement("span");
            span.className = "photos-urls editable";
            span.contentEditable = "true";
            span.spellcheck = false;
            span.textContent = split.rest;
            td.appendChild(span);
          } else {
            td.contentEditable = "true";
            td.className = "editable";
            td.spellcheck = false;
            td.textContent = text;
          }
        } else if (i === 5) {
          td.textContent = text;
          if (text === "Нет панорамы") td.className = "no-pano";
        } else if (i >= 1 && i <= 4) {
          td.contentEditable = "true";
          td.className = "editable";
          td.spellcheck = false;
          td.textContent = text;
        } else {
          td.textContent = text;
        }
        tr.appendChild(td);
      });
      tr.addEventListener("click", () => selectByKey(pk));
      tb.appendChild(tr);
    });
  }

  function selectByKey(key) {
    selectedKey = String(key);
    document.querySelectorAll("#report tbody tr").forEach((tr) => {
      tr.classList.toggle("selected", tr.dataset.parcelKey === selectedKey);
    });
    let row = null;
    document.querySelectorAll("#report tbody tr").forEach((tr) => {
      if (tr.dataset.parcelKey === selectedKey) row = tr;
    });
    if (row) {
      row.scrollIntoView({ block: "nearest", behavior: "smooth" });
      void updateGalleryFromRow(row);
    }
    highlightOnMap(selectedKey);
  }

  function parseLatLonFromApiUrl(url) {
    try {
      const u = new URL(url, window.location.origin);
      const lat = parseFloat(u.searchParams.get("lat"));
      const lon = parseFloat(u.searchParams.get("lon"));
      if (Number.isNaN(lat) || Number.isNaN(lon)) return null;
      return { lat, lon };
    } catch (_) {
      return null;
    }
  }

  function appendYandexIframeFallback(viewport, lat, lon) {
    if (viewport.querySelector(".gallery-yandex-iframe")) return;
    const iframe = document.createElement("iframe");
    iframe.className = "gallery-yandex-iframe";
    iframe.title = "Панорама Яндекс";
    iframe.loading = "lazy";
    iframe.referrerPolicy = "no-referrer";
    iframe.src = yandexWidgetUrl(lon, lat);
    viewport.appendChild(iframe);
  }

  function appendRasterThumb(gal, url, caption, opts) {
    const fit = opts && opts.fit;
    const eager = opts && opts.eager;
    const wrap = document.createElement("div");
    wrap.className = "gallery-preview-wrap";
    const viewport = document.createElement("div");
    viewport.className = fit
      ? "gallery-preview-viewport gallery-preview-viewport--fit"
      : "gallery-preview-viewport gallery-preview-viewport--scroll";
    const loadingEl = document.createElement("div");
    loadingEl.className = "gallery-preview-loading";
    loadingEl.textContent = "Загрузка снимка…";
    const img = document.createElement("img");
    img.loading = eager ? "eager" : "lazy";
    img.alt = "";
    img.decoding = "async";
    img.className = fit ? "gallery-preview-img gallery-preview-img--fit" : "gallery-preview-img";
    img.src = url;
    img.referrerPolicy = "no-referrer";
    viewport.appendChild(loadingEl);
    img.addEventListener("load", () => {
      loadingEl.remove();
      if (fit) {
        img.style.transform = "none";
        img.dataset.scale = "1";
      }
    });
    img.addEventListener("error", () => {
      loadingEl.remove();
      img.style.display = "none";
      const ll = parseLatLonFromApiUrl(url);
      if (ll) {
        appendYandexIframeFallback(viewport, ll.lat, ll.lon);
      }
      const err = document.createElement("div");
      err.className = "thumb-row gallery-preview-error";
      err.textContent = ll
        ? "JPEG недоступен — показан виджет Яндекс.Карт ниже."
        : "Не удалось загрузить снимок.";
      wrap.appendChild(err);
    });
    const cap = document.createElement("div");
    cap.className = "thumb-row";
    cap.textContent = caption || safeHostname(url);
    viewport.appendChild(img);
    wrap.appendChild(viewport);
    wrap.appendChild(cap);
    gal.appendChild(wrap);
    if (!fit) {
      currentPanoImg = img;
      currentPanoViewport = viewport;
      applyZoomFromControl();
    }
  }

  function applyZoomFromControl(anchor) {
    const rng = el("panoZoom");
    const valEl = el("panoZoomVal");
    const pct = rng ? Number(rng.value || 100) : 100;
    if (valEl) valEl.textContent = String(pct) + "%";
    if (!currentPanoImg) return;
    const s = pct / 100;
    if (anchor && currentPanoViewport) {
      // удержать точку под курсором при масштабировании
      const vp = currentPanoViewport;
      const rect = vp.getBoundingClientRect();
      const ax = anchor.clientX - rect.left + vp.scrollLeft;
      const ay = anchor.clientY - rect.top + vp.scrollTop;
      const prev = Number(currentPanoImg.dataset.scale || "1");
      const k = s / prev;
      currentPanoImg.style.transform = "scale(" + s + ")";
      currentPanoImg.dataset.scale = String(s);
      vp.scrollLeft = ax * k - (anchor.clientX - rect.left);
      vp.scrollTop = ay * k - (anchor.clientY - rect.top);
      return;
    }
    currentPanoImg.style.transform = "scale(" + s + ")";
    currentPanoImg.dataset.scale = String(s);
  }

  function togglePanoFullscreen() {
    const panel = document.querySelector(".gallery-panel");
    const btn = el("btnPanoFullscreen");
    if (!panel) return;
    panel.classList.toggle("pano-fullscreen");
    if (btn) {
      btn.textContent = panel.classList.contains("pano-fullscreen") ? "Вернуться" : "На весь экран";
    }
  }

  function togglePanoCollapse() {
    const panel = document.querySelector(".gallery-panel");
    const btn = el("btnPanoCollapse");
    if (!panel) return;
    panel.classList.toggle("gallery-collapsed");
    if (btn) {
      btn.textContent = panel.classList.contains("gallery-collapsed")
        ? "Развернуть панорамы"
        : "Свернуть панорамы";
    }
  }

  function photosTextFromRow(tr) {
    const split = splitPhotosField(tr.dataset.photosUi || "");
    const cell = tr.cells[5];
    let rest = "";
    if (cell) {
      const span = cell.querySelector(".photos-urls");
      rest = (span ? span.textContent : cell.textContent).trim();
    }
    const parts = [];
    if (split.gallery) parts.push(split.gallery);
    if (rest) parts.push(rest);
    return parts.join(" ").trim();
  }

  function yandexWidgetUrl(lon, lat) {
    return (
      "https://yandex.ru/map-widget/v1/?ll=" +
      encodeURIComponent(lon + "," + lat) +
      "&z=18&l=stv&lang=ru_RU"
    );
  }

  async function loadPanoramaBlock(container, point, urls, reqId, parcelKey) {
    const ll = { lon: point.lon, lat: point.lat };
    const block = document.createElement("div");
    block.className = "gallery-point-block";
    const title = document.createElement("h3");
    title.className = "gallery-point-title";
    const tabIdx = currentViewPoints.findIndex(
      (p) =>
        Math.abs(Number(p.lon) - Number(point.lon)) < 1e-5 &&
        Math.abs(Number(p.lat) - Number(point.lat)) < 1e-5
    );
    const cap =
      tabIdx >= 0
        ? viewPointCaption(currentViewPoints[tabIdx], tabIdx)
        : point.label || "Точка " + point.index;
    title.textContent =
      cap +
      (point.bearing_deg != null
        ? " · взгляд " + point.bearing_deg + "° на участок"
        : "");
    block.appendChild(title);
    const inner = document.createElement("div");
    block.appendChild(inner);
    container.appendChild(block);

    const mapLink = document.createElement("p");
    mapLink.className = "gallery-point-note";
    const ma = document.createElement("a");
    ma.href = yandexMapsAtPointUrl(ll.lon, ll.lat, point.bearing_deg);
    ma.target = "_blank";
    ma.rel = "noopener noreferrer";
    ma.textContent = "Яндекс.Карты: панорама в точке съёмки";
    mapLink.appendChild(ma);
    inner.appendChild(mapLink);

    if (datasetId && parcelKey) {
      fetch("/snapshots/" + encodeURIComponent(datasetId) + "/" + encodeURIComponent(parcelKey) + "/")
        .catch(() => {});
    }

    const tgt = parcelTargetLatLon(featureForKey(selectedKey));
    let q =
      "/api/panorama-previews?lon=" +
      encodeURIComponent(ll.lon) +
      "&lat=" +
      encodeURIComponent(ll.lat);
    if (point.bearing_deg != null) {
      q += "&bearing=" + encodeURIComponent(point.bearing_deg);
    } else if (tgt) {
      q += "&tgt_lon=" + encodeURIComponent(tgt.lon) + "&tgt_lat=" + encodeURIComponent(tgt.lat);
    }

    let previews = [];
    try {
      const res = await fetch(q);
      const data = await res.json().catch(() => ({}));
      if (reqId !== galleryReqId) return false;
      if (res.ok && Array.isArray(data.images)) previews = data.images;
    } catch (_) {
      if (reqId !== galleryReqId) return false;
    }

    const equirectItems = previews.filter((it) => it && it.kind === "equirect");
    const cropItems = previews.filter((it) => it && it.kind === "crop");
    const otherItems = previews.filter((it) => it && it.kind !== "crop" && it.kind !== "equirect");

    let thumbIdx = 0;
    cropItems.forEach((item) => {
      appendRasterThumb(inner, item.url, item.label || "Вид на участок", {
        fit: true,
        eager: thumbIdx++ === 0,
      });
    });
    equirectItems.forEach((item) => {
      if (!item.url) return;
      let u = item.url;
      if (u.indexOf("w=") >= 0) {
        u = u.replace(/w=\d+/, "w=1024");
      } else {
        u += (u.indexOf("?") >= 0 ? "&" : "?") + "w=1024";
      }
      appendRasterThumb(inner, u, item.label || "Панорама 360° (Яндекс)", {
        fit: true,
        eager: thumbIdx++ === 0,
      });
    });
    otherItems.slice(0, 4).forEach((item) => {
      if (!item.url) return;
      appendRasterThumb(inner, item.url, item.label || "Панорама", { fit: true });
    });

    urls.forEach((url) => {
      if (isSnapshotsGalleryUrl(url)) return;
      const ull = parseLlFromUrl(url);
      if (ull && Math.abs(ull.lon - ll.lon) < 1e-4 && Math.abs(ull.lat - ll.lat) < 1e-4) return;
      if (isRasterImageUrl(url)) appendRasterThumb(inner, url, safeHostname(url));
    });
    return true;
  }

  async function updateGalleryFromRow(tr) {
    const gal = el("gallery");
    const hint = el("galleryHint");
    const myId = ++galleryReqId;
    gal.innerHTML = "";
    currentPanoImg = null;
    currentPanoViewport = null;

    const photosText = photosTextFromRow(tr);
    const urls = splitUrls(photosText);
    const parcelKey = tr.dataset.parcelKey;
    const viewPoints = viewPointsForParcel(parcelKey, photosText);

    if (!viewPoints.length && !urls.length) {
      currentViewPoints = [];
      clearViewBeacons();
      hint.textContent =
        "Нет точек панорамы. Запустите классификацию или укажите ссылки в «Снимки».";
      return;
    }

    const feat = featureForKey(parcelKey);
    const points = enrichViewPointsWithEdges(
      viewPoints.map((vp, i) => ({
        index: vp.index != null ? vp.index : i + 1,
        lon: vp.lon,
        lat: vp.lat,
        bearing_deg: vp.bearing_deg,
        edge_index: vp.edge_index != null ? vp.edge_index : null,
        edge: vp.edge || null,
        label: vp.label || null,
      })),
      feat
    );

    const validPoints = points.filter((pt) => pt.lon != null && pt.lat != null);
    currentViewPoints = validPoints;
    activeViewPointIndex = 0;
    syncViewPointTabs = null;
    if (validPoints.length > 1) {
      const tabs = document.createElement("div");
      tabs.className = "gallery-point-tabs";
      tabs.setAttribute("role", "tablist");
      const panesHost = document.createElement("div");
      panesHost.className = "gallery-point-panes";
      const panes = [];

      const tabButtons = [];

      validPoints.forEach((pt, idx) => {
        const tab = document.createElement("button");
        tab.type = "button";
        tab.className = "gallery-point-tab" + (idx === 0 ? " is-active" : "");
        tab.setAttribute("role", "tab");
        tab.style.borderColor = edgeColor(idx);
        tab.style.setProperty("--tab-edge-color", edgeColor(idx));
        tab.textContent = viewPointCaption(pt, idx);
        tabButtons.push(tab);
        const pane = document.createElement("div");
        pane.className = "gallery-point-pane" + (idx === 0 ? " is-active" : "");
        pane.dataset.loaded = idx === 0 ? "0" : "";
        panes.push(pane);
        panesHost.appendChild(pane);
        tabs.appendChild(tab);

        tab.addEventListener("click", () => {
          if (myId !== galleryReqId) return;
          setActiveViewPointIndex(idx);
        });
      });

      syncViewPointTabs = (idx) => {
        tabButtons.forEach((b, j) => {
          b.classList.toggle("is-active", j === idx);
          b.setAttribute("aria-selected", j === idx ? "true" : "false");
        });
        panes.forEach((p, j) => p.classList.toggle("is-active", j === idx));
        void loadPaneIfNeeded(idx);
      };

      gal.appendChild(tabs);
      gal.appendChild(panesHost);

      async function loadPaneIfNeeded(idx) {
        if (myId !== galleryReqId) return;
        const pane = panes[idx];
        if (!pane || pane.dataset.loaded === "1") return;
        pane.dataset.loaded = "pending";
        const ok = await loadPanoramaBlock(pane, validPoints[idx], urls, myId, parcelKey);
        if (myId !== galleryReqId) return;
        if (ok) pane.dataset.loaded = "1";
        else pane.dataset.loaded = "";
      }

      setActiveViewPointIndex(0);
      await loadPaneIfNeeded(0);
      return;
    }

    setActiveViewPointIndex(0);
    for (const pt of validPoints) {
      await loadPanoramaBlock(gal, pt, urls, myId, parcelKey);
      if (myId !== galleryReqId) return;
    }
  }

  function collectRows() {
    return [...document.querySelectorAll("#report tbody tr")].map((tr) => ({
      parcel_key: tr.dataset.parcelKey,
      class_ui: tr.cells[1] ? tr.cells[1].textContent.trim() : "",
      subclass_ui: tr.cells[2] ? tr.cells[2].textContent.trim() : "",
      class_corrected_ui: tr.cells[3] ? tr.cells[3].textContent.trim() : "",
      subclass_corrected_ui: tr.cells[4] ? tr.cells[4].textContent.trim() : "",
      panorama_status_ui: tr.cells[5] ? tr.cells[5].textContent.trim() : "",
      photos_ui: photosTextFromRow(tr),
    }));
  }

  function setStatus(text) {
    el("statusText").textContent = text;
  }

  function setProgress(pct) {
    el("progressBar").value = Math.min(100, Math.max(0, pct));
  }

  async function uploadFile(file) {
    const fd = new FormData();
    fd.append("file", file);
    setStatus("Статус: загрузка…");
    setProgress(5);
    const xhr = new XMLHttpRequest();
    return await new Promise((resolve, reject) => {
      xhr.upload.onprogress = (ev) => {
        if (ev.lengthComputable) setProgress(5 + (ev.loaded / ev.total) * 85);
      };
      xhr.onload = () => {
        setProgress(100);
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText));
          } catch (e) {
            reject(e);
          }
        } else {
          let msg = xhr.responseText;
          try {
            msg = JSON.parse(xhr.responseText).error || msg;
          } catch (_) {}
          reject(new Error(msg));
        }
      };
      xhr.onerror = () => reject(new Error("Сеть"));
      xhr.open("POST", "/api/upload");
      xhr.send(fd);
    });
  }

  function updateRunButton() {
    const btn = el("btnRun");
    if (!btn) return;
    const busy = uploadBusy;
    const ready = Boolean(datasetId);
    btn.disabled = busy;
    btn.title = busy
      ? "Идёт загрузка файла…"
      : ready
        ? "Запуск классификации"
        : "Сначала выберите .gpkg (Обзор…) и дождитесь загрузки";
    btn.style.opacity = busy ? "0.45" : ready ? "1" : "0.65";
  }

  function applyUploadResult(data, fileName) {
    if (!data || !data.dataset_id) {
      throw new Error("Сервер не вернул идентификатор набора данных. Проверьте формат .gpkg.");
    }
    datasetId = data.dataset_id;
    if (data.geojson) loadParcels(data.geojson);
    if (data.rows) renderTable(data.rows);
    el("reportTitle").textContent = fileName ? "— " + fileName : "";
    setStatus("Статус: загружено участков — " + (data.count != null ? data.count : "?"));
    updateRunButton();
  }

  function uploadSelectedFile(file) {
    if (!file) return Promise.resolve();
    if (uploadBusy) {
      return Promise.reject(new Error("Подождите, идёт загрузка предыдущего файла…"));
    }
    uploadBusy = true;
    updateRunButton();
    el("fileName").textContent = file.name + " — загрузка…";
    return uploadFile(file)
      .then((data) => {
        if (data && data.error) throw new Error(data.error);
        applyUploadResult(data, file.name);
        el("fileName").textContent = file.name;
      })
      .catch((e) => {
        datasetId = null;
        updateRunButton();
        throw e;
      })
      .finally(() => {
        uploadBusy = false;
        updateRunButton();
      });
  }

  async function runClassify() {
    if (uploadBusy) {
      alert("Подождите, файл ещё загружается на сервер…");
      return;
    }
    if (!datasetId) {
      const f = el("fileInput").files && el("fileInput").files[0];
      if (f) {
        try {
          await uploadSelectedFile(f);
        } catch (e) {
          alert(e.message || String(e));
          setStatus("Статус: ошибка загрузки");
          setProgress(0);
          return;
        }
      }
    }
    if (!datasetId) {
      alert(
        "GeoPackage ещё не загружен.\n\n1. Нажмите «Обзор…» и выберите .gpkg\n2. Дождитесь статуса «загружено участков»\n3. Затем нажмите ▶"
      );
      return;
    }
    const overwrite = el("overwriteAll").checked;
    setStatus("Статус: классификация…");
    setProgress(10);
    const res = await fetch("/api/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_id: datasetId,
        rows: collectRows(),
        overwrite_all: overwrite,
      }),
    });
    const data = await res.json();
    setProgress(100);
    if (!res.ok) throw new Error(data.error || res.statusText);
    loadParcels(data.geojson);
    renderTable(data.rows);
    const st = data.classify_stats || {};
    const parts = [];
    if (st.ml) parts.push(`ML: ${st.ml}`);
    if (st.heuristic) parts.push(`эвристика: ${st.heuristic}`);
    if (st.no_panorama) parts.push(`нет панорамы: ${st.no_panorama}`);
    if (st.ml_low_confidence) parts.push(`ML низкая уверенность: ${st.ml_low_confidence}`);
    if (st.skipped) parts.push(`пропущено: ${st.skipped}`);
    setStatus(parts.length ? `Статус: готово (${parts.join(", ")})` : "Статус: готово");
  }

  async function exportGpkg() {
    if (!datasetId) {
      alert("Сначала загрузите GeoPackage.");
      return;
    }
    const res = await fetch("/api/export-gpkg", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_id: datasetId, rows: collectRows() }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      alert(err.error || res.statusText);
      return;
    }
    const blob = await res.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `parcels_${datasetId.slice(0, 8)}.gpkg`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  function exportCsv() {
    const rows = [...document.querySelectorAll("#report tbody tr")];
    if (!rows.length) {
      alert("Нет данных для экспорта.");
      return;
    }
    const headers = [...document.querySelectorAll("#report thead th")].map((th) =>
      th.textContent.trim()
    );
    const lines = [headers.join(";")];
    rows.forEach((tr) => {
      const cells = [...tr.cells].map((td) => '"' + td.textContent.replace(/"/g, '""') + '"');
      lines.push(cells.join(";"));
    });
    const blob = new Blob(["\ufeff" + lines.join("\n")], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "report_parcels.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  }

  function resetUi() {
    datasetId = null;
    selectedKey = null;
    el("fileName").textContent = "Участки.gpkg — выберите файл";
    el("reportTitle").textContent = "";
    document.querySelector("#report tbody").innerHTML = "";
    el("gallery").innerHTML = "";
    el("galleryHint").textContent = "Выберите строку в отчёте или полигон на карте.";
    setStatus("Статус: ожидание файла");
    setProgress(0);
    clearViewBeacons();
    if (parcelLayer) {
      map.removeLayer(parcelLayer);
      parcelLayer = null;
    }
    updateRunButton();
  }

  document.addEventListener("DOMContentLoaded", () => {
    initMap();

    const zoom = el("panoZoom");
    if (zoom) zoom.addEventListener("input", () => applyZoomFromControl());
    const fs = el("btnPanoFullscreen");
    if (fs) fs.addEventListener("click", togglePanoFullscreen);
    const col = el("btnPanoCollapse");
    if (col) col.addEventListener("click", togglePanoCollapse);

    // Зум колёсиком мыши по панораме
    document.addEventListener(
      "wheel",
      (ev) => {
        if (!currentPanoViewport || !currentPanoImg) return;
        if (!currentPanoViewport || !currentPanoViewport.classList.contains("gallery-preview-viewport--scroll")) return;
        if (!currentPanoViewport.contains(ev.target)) return;
        ev.preventDefault();
        const rng = el("panoZoom");
        if (!rng) return;
        const step = ev.deltaY < 0 ? 10 : -10;
        const next = Math.max(Number(rng.min), Math.min(Number(rng.max), Number(rng.value) + step));
        rng.value = String(next);
        applyZoomFromControl(ev);
      },
      { passive: false }
    );

    const fileInput = el("fileInput");

    let lastPickToken = "";

    function onFilePicked() {
      const f = fileInput.files && fileInput.files[0];
      if (!f) return;
      const token = [f.name, f.size, f.lastModified].join("|");
      if (token === lastPickToken) return;
      lastPickToken = token;
      const name = (f.name || "").toLowerCase();
      if (!name.endsWith(".gpkg")) {
        alert("Нужен файл с расширением .gpkg");
        fileInput.value = "";
        lastPickToken = "";
        return;
      }
      uploadSelectedFile(f)
        .catch((e) => {
          alert(e.message || String(e));
          setStatus("Статус: ошибка загрузки");
          setProgress(0);
          el("fileName").textContent = "Ошибка — выберите файл снова";
          lastPickToken = "";
        })
        .finally(() => {
          fileInput.value = "";
        });
    }

    fileInput.addEventListener("change", onFilePicked);

    updateRunButton();

    el("btnRun").addEventListener("click", async () => {
      try {
        await runClassify();
      } catch (e) {
        alert(e.message || String(e));
        setStatus("Статус: ошибка");
      }
    });

    el("btnReset").addEventListener("click", () => {
      if (datasetId) {
        fetch("/api/dataset/" + encodeURIComponent(datasetId), { method: "DELETE" }).catch(() => {});
      }
      resetUi();
      el("fileInput").value = "";
    });

    el("btnExport").addEventListener("click", exportCsv);
    el("btnExportGpkg").addEventListener("click", exportGpkg);
  });
})();
