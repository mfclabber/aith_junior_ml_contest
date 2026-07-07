#!/usr/bin/env python3
"""
Сбор датасета панорам по таксономии UTT классов через OSM фильтры.

Использование:
    python3 dataset_collector.py --class all --output-dir data/dataset --check-only
    python3 dataset_collector.py --class natural_areas --max-results 100 --download
"""

import requests
import json
import subprocess
import os
import time
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import argparse
from collections import defaultdict
from PIL import Image
import math

try:
    # When executed as a module: python -m scripts.data_collection.dataset_collector
    from .perspective import generate_default_views
except ImportError:
    # When executed as a script: python scripts/data_collection/dataset_collector.py
    try:
        from perspective import generate_default_views
    except ImportError:
        # Fallback for runpy / unusual sys.path: import from the same directory as this file
        _this_dir = Path(__file__).resolve().parent
        if str(_this_dir) not in sys.path:
            sys.path.insert(0, str(_this_dir))
        from perspective import generate_default_views

# Границы Москвы
MOSCOW_BOUNDS = {
    "south": 55.5,
    "north": 55.9,
    "west": 37.3,
    "east": 37.9
}


def set_bounds(south: float, west: float, north: float, east: float) -> None:
    """Update global bounds used for Overpass queries."""
    MOSCOW_BOUNDS["south"] = float(south)
    MOSCOW_BOUNDS["west"] = float(west)
    MOSCOW_BOUNDS["north"] = float(north)
    MOSCOW_BOUNDS["east"] = float(east)

def query_overpass(query: str, timeout: int = 180) -> Optional[dict]:
    """Выполняет запрос к Overpass API"""
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    try:
        print(f"  Запрос к Overpass API (timeout={timeout}s)...")
        response = requests.post(
            overpass_url,
            data=query,
            headers={'Content-Type': 'text/plain'},
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"  Error: Overpass API: {e}")
        return None


class OSMQueryBuilder:
    """Построитель Overpass запросов для разных классов UTT"""
    
    @staticmethod
    def get_natural_areas() -> str:
        """1. Природные (парковые) территории"""
        bounds = MOSCOW_BOUNDS
        return f"""
        [out:json][timeout:180];
        (
          way["landuse"="forest"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
          way["landuse"="meadow"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
          way["landuse"="grass"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
          way["leisure"="park"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
          way["leisure"="nature_reserve"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
          way["natural"="wood"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
          way["natural"="scrub"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
          relation["landuse"="forest"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
          relation["leisure"="park"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
        );
        out center;
        """
    
    @staticmethod
    def get_low_density_degraded() -> str:
        """2. Низкоплотная застройка / Деградировавшие антропогенные объекты"""
        bounds = MOSCOW_BOUNDS
        return f"""
        [out:json][timeout:180];
        (
          way["building"="abandoned"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
          way["abandoned:building"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
          way["disused:building"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
          way["abandoned"="yes"]["building"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
          way["disused"="yes"]["building"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
          way["landuse"="residential"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
          way["amenity"="parking"]["parking"="garage_boxes"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
          relation["building"="abandoned"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
          relation["abandoned"="yes"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
        );
        out center;
        """
    
    @staticmethod
    def get_underused_infrastructure() -> str:
        """3. Недоиспользуемые инфраструктурные/городские зоны"""
        bounds = MOSCOW_BOUNDS
        return f"""
        [out:json][timeout:180];
        (
          way["shop"="vacant"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
          way["amenity"="vacant"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
          way["office"="vacant"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
          way["amenity"="parking"]["parking"~"surface|multi-storey"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
          way["building"="commercial"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
          way["building"="retail"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
          way["highway"="bus_stop"]["disused:highway"="bus_stop"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
          relation["shop"="vacant"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
        );
        out center;
        """
    
    @staticmethod
    def get_frozen_construction() -> str:
        """4. Незавершенное/приостановленное строительство"""
        bounds = MOSCOW_BOUNDS
        return f"""
        [out:json][timeout:180];
        (
          way["landuse"="construction"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
          way["construction"="yes"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
          way["building"="construction"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
          relation["landuse"="construction"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
        );
        out center;
        """
    
    @staticmethod
    def get_active_construction() -> str:
        """5. Активное строительство (стройплощадки)"""
        # Используем те же фильтры, что и для замороженного строительства
        # Различие будет по дате в метаданных или визуальному анализу
        bounds = MOSCOW_BOUNDS
        return f"""
        [out:json][timeout:180];
        (
          way["landuse"="construction"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
          way["construction"="yes"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
          relation["landuse"="construction"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
        );
        out center;
        """
    
    @staticmethod
    def get_active_urban() -> str:
        """6. Активные городские территории (фон/контрольный класс)"""
        bounds = MOSCOW_BOUNDS
        return f"""
        [out:json][timeout:180];
        (
          way["building"="apartments"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
          way["building"="residential"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
          way["shop"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
          way["amenity"~"cafe|restaurant"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
          way["highway"~"primary|secondary"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
          relation["building"="apartments"]({bounds["south"]},{bounds["west"]},{bounds["north"]},{bounds["east"]});
        );
        out center;
        """


def parse_osm_results(data: dict, class_name: str) -> List[Tuple[float, float, str, dict, Optional[str], Optional[int]]]:
    """Парсит результаты Overpass API"""
    if not data or "elements" not in data:
        return []
    
    results = []
    for element in data["elements"]:
        # Получаем координаты
        if "center" in element:
            lat = element["center"]["lat"]
            lon = element["center"]["lon"]
        elif "lat" in element and "lon" in element:
            lat = element["lat"]
            lon = element["lon"]
        else:
            continue
        
        # Получаем теги
        tags = element.get("tags", {})
        
        # Генерируем имя
        name = tags.get("name", tags.get("landuse", tags.get("leisure", class_name)))
        
        element_type = element.get("type")
        element_id = element.get("id")
        results.append((lat, lon, name, tags, element_type, element_id))
    
    return results


def get_season(month: int) -> str:
    """Определяет сезон по месяцу"""
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "autumn"


def check_panorama_exists(lat: float, lon: float) -> Tuple[bool, Optional[dict]]:
    """Проверяет наличие панорамы и возвращает метаданные"""
    api_url = (
        f"https://api-maps.yandex.ru/services/panoramas/1.x/?l=stv&lang=ru_RU&ll="
        f"{lon},{lat}&origin=userAction&provider=streetview"
    )
    
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(api_url, headers=headers, timeout=10)
        data = response.json()
        
        if data.get("status") == "error":
            return False, None
        
        # Извлекаем метаданные
        metadata = {}
        if "data" in data and "Data" in data["data"]:
            pano_data = data["data"]["Data"]
            timestamp = pano_data.get("timestamp")
            if timestamp:
                capture_date = datetime.fromtimestamp(timestamp)
                metadata = {
                    "capture_date": capture_date.strftime('%Y-%m-%d'),
                    "capture_year": capture_date.year,
                    "capture_month": capture_date.month,
                    "capture_day": capture_date.day,
                    "capture_season": get_season(capture_date.month),
                    "timestamp": timestamp
                }
        
        return True, metadata
    except Exception as e:
        return False, None


def get_nearby_panoramas(lat: float, lon: float, radius_meters: int = 20, 
                        max_points: int = 20) -> List[Tuple[float, float, dict]]:
    """
    Ищет панорамы вблизи заданных координат.
    Возвращает список (lat, lon, metadata) с расстоянием до объекта в метаданных (_distance_m),
    отсортированный по возрастанию расстояния.
    """
    panoramas: List[Tuple[float, float, dict]] = []
    
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Возвращает расстояние между точками в метрах"""
        R = 6371000  # Радиус Земли в метрах
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c
    
    # Сначала проверяем центральную точку
    exists, metadata = check_panorama_exists(lat, lon)
    if exists and metadata:
        md = metadata.copy()
        md["_distance_m"] = 0.0
        panoramas.append((lat, lon, md))
    
    # Генерируем точки вокруг (в радиусе radius_meters метров)
    # 1 градус ≈ 111 км
    lat_step = radius_meters / 111000
    lon_step = radius_meters / (111000 * math.cos(math.radians(lat)))
    
    num_points = min(max_points, 20)  # Ограничиваем количество проверок
    
    checked = 0
    for i in range(-num_points, num_points + 1):
        for j in range(-num_points, num_points + 1):
            if i == 0 and j == 0:
                continue
            
            new_lat = lat + i * lat_step
            new_lon = lon + j * lon_step
            
            distance_m = haversine_distance(lat, lon, new_lat, new_lon)
            if distance_m > radius_meters or checked >= max_points:
                continue
            
            exists, metadata = check_panorama_exists(new_lat, new_lon)
            if exists and metadata:
                md = metadata.copy()
                md["_distance_m"] = float(distance_m)
                panoramas.append((new_lat, new_lon, md))
            
            checked += 1
            time.sleep(0.2)  # Задержка для rate limiting
    
    # Сортируем по расстоянию до объекта
    panoramas.sort(key=lambda x: x[2].get("_distance_m", 0.0))
    return panoramas


def find_panoramas_by_seasons(lat: float, lon: float, 
                              target_seasons: List[str] = None) -> Dict[str, List[dict]]:
    """
    Находит панорамы разных времен года для координат.
    Возвращает словарь {сезон: [список метаданных с координатами]}
    """
    if target_seasons is None:
        target_seasons = ["spring", "summer", "autumn", "winter"]
    
    # Группируем по сезонам
    by_season = {season: [] for season in target_seasons}
    
    # Ищем панорамы вблизи (отсортированы по расстоянию до объекта)
    nearby_panos = get_nearby_panoramas(lat, lon, radius_meters=10, max_points=30)
    
    for pano_lat, pano_lon, metadata in nearby_panos:
        season = metadata.get("capture_season")
        if season and season in by_season:
            # Добавляем координаты и расстояние в метаданные
            pano_info = metadata.copy()
            pano_info["lat"] = pano_lat
            pano_info["lon"] = pano_lon
            by_season[season].append(pano_info)
    
    return by_season


def get_image_info(image_path: Path) -> Optional[Dict]:
    """Получает информацию о разрешении изображения"""
    try:
        with Image.open(image_path) as img:
            return {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode,
                "size_bytes": image_path.stat().st_size
            }
    except Exception:
        return None


def download_panorama(lat: float, lon: float, output_path: str, zoom: int = 1) -> bool:
    """Download panorama using pano.py"""
    script_dir = Path(__file__).parent
    pano_script = script_dir / "pano.py"
    
    coords_str = f"{lat},{lon}"
    
    try:
        result = subprocess.run(
            ['python3', str(pano_script), '-c', coords_str, '-z', str(zoom), '-o', str(output_path)],
            capture_output=True,
            text=True,
            timeout=300  # Увеличиваем таймаут до 5 минут
        )
        
        if result.returncode != 0:
            # Логируем ошибку только если она не про "панорама не найдена" (это нормально)
            error_msg = result.stderr.strip() if result.stderr else result.stdout.strip()
            if error_msg and "Panorama not found" not in error_msg and "API error: 404" not in error_msg:
                # Показываем только первые 200 символов ошибки
                print(f"        Error details: {error_msg[:200]}")
            return False
        
        # Проверяем, что файл действительно создан и не пустой
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size > 0:
                return True
            else:
                print(f"        Warning: File created but empty ({file_size} bytes)")
                return False
        else:
            return False
    except subprocess.TimeoutExpired:
        print(f"        Error: Timeout (>5 min)")
        return False
    except Exception as e:
        print(f"        Error: {str(e)[:100]}")
        return False


def maybe_generate_views(
    image_path: Path,
    *,
    enabled: bool,
    views_dir_name: str = "views",
    headings: List[float] = None,
    pitches: List[float] = None,
    fov_deg: float = 90.0,
    size: Tuple[int, int] = (1024, 768),
) -> int:
    """Optionally generate multiple perspective views from an equirect panorama."""
    if not enabled:
        return 0
    if not image_path.exists():
        return 0

    headings = headings or [0.0, 90.0, 180.0, 270.0]
    pitches = pitches or [0.0]

    views_dir = image_path.parent / views_dir_name
    views_dir.mkdir(parents=True, exist_ok=True)

    out_w, out_h = size
    return generate_default_views(
        str(image_path),
        str(views_dir),
        headings=tuple(headings),
        pitches=tuple(pitches),
        fov_deg=fov_deg,
        out_width=out_w,
        out_height=out_h,
        prefix=image_path.stem,
    )


def create_item_metadata(item: Dict, image_path: Path, class_name: str) -> Dict:
    """Создает полные метаданные для отдельного изображения"""
    # Если image_path не существует (для папок с несколькими панорамами), используем первую доступную
    if not image_path.exists():
        image_path = image_path.parent / "panorama.jpg"
        if not image_path.exists():
            # Ищем любую панораму в папке
            panorama_files = list(image_path.parent.glob("panorama*.jpg"))
            if panorama_files:
                image_path = panorama_files[0]
    
    image_info = get_image_info(image_path) if image_path.exists() else None
    
    metadata = {
        # Базовая информация
        "id": item.get("id", ""),
        "name": item.get("name", ""),
        "class": class_name,
        "class_label": TAXONOMY[class_name]["label"],
        "class_id": TAXONOMY[class_name]["id"],
        
        # Таксономия
        "taxonomy": {
            "main_class": {
                "id": TAXONOMY[class_name]["id"],
                "name": class_name,
                "label": TAXONOMY[class_name]["label"]
            },
            "subclasses": TAXONOMY[class_name]["subclasses"],
            "subclass_id": None,  # Будет заполнено при разметке
            "subclass_label": None  # Будет заполнено при разметке
        },
        
        # Географические данные
        "coordinates": {
            "lat": item.get("lat"),
            "lon": item.get("lon")
        },
        
        # OSM данные
        "osm": {
            "tags": item.get("tags", {}),
            "name": item.get("name", "")
        },
        "osm_element": {
            "type": item.get("osm_element_type"),
            "id": item.get("osm_element_id")
        },
        
        # Информация о панораме
        "panorama": {
            "has_panorama": item.get("has_panorama", False),
            "capture_date": item.get("capture_date"),
            "capture_year": item.get("capture_year"),
            "capture_month": item.get("capture_month"),
            "capture_day": item.get("capture_day"),
            "capture_season": item.get("capture_season") or item.get("season"),
            "timestamp": item.get("timestamp"),
            "zoom_level": item.get("zoom_level", 1)
        },
        
        # Информация об изображении
        "image": image_info or {},
        
        # Файловая информация
        "file": {
            "filename": image_path.name,
            "path": str(image_path.relative_to(image_path.parent.parent.parent)),
            "directory": image_path.parent.name
        },
        
        # Метаданные датасета
        "dataset": {
            "created_at": datetime.now().isoformat(),
            "version": "1.0"
        }
    }
    
    return metadata


# Полная таксономия классов и подклассов
TAXONOMY = {
    "natural_areas": {
        "label": "Природные (парковые) территории",
        "id": 1,
        "subclasses": {
            "1.1": "Парки и скверы (благоустроенные)",
            "1.2": "Лесные массивы",
            "1.3": "Парковые дорожки"
        }
    },
    "low_density_degraded": {
        "label": "Низкоплотная застройка / Деградировавшие антропогенные объекты",
        "id": 2,
        "subclasses": {
            "2.1": "Заброшенные здания",
            "2.2": "Объекты с признаками вандализма (разбитые окна, граффити, обвалившиеся ограждения)"
        }
    },
    "underused_infrastructure": {
        "label": "Недоиспользуемые инфраструктурные/городские зоны",
        "id": 3,
        "subclasses": {
            "3.1": "Парковки с низкой загрузкой",
            "3.2": "Неиспользуемые коммерческие фасады (пустые витрины >6 месяцев)",
            "3.5": "«Некрасивые» фасады (старые ТЦ или БЦ)"
        }
    },
    "frozen_construction": {
        "label": "Незавершенное/приостановленное строительство",
        "id": 4,
        "subclasses": {
            "4.1": "Стройплощадки без активности >1 года",
            "4.2": "Замороженные объекты"
        }
    },
    "active_construction": {
        "label": "Активное строительство (стройплощадки)",
        "id": 5,
        "subclasses": {
            "5.1": "Стройплощадки с активностью",
            "5.2": "Новая застройка"
        }
    },
    "active_urban": {
        "label": "Активные городские территории (фон/контрольный класс)",
        "id": 6,
        "subclasses": {
            "6.1": "Жилая застройка с активностью",
            "6.2": "Коммерческие улицы",
            "6.3": "Благоустроенные парки и скверы",
            "6.4": "Интенсивные транспортные коридоры",
            "6.5": "Образовательные/медицинские/офисные комплексы с посетителями"
        }
    }
}

CLASS_QUERIES = {
    "natural_areas": (OSMQueryBuilder.get_natural_areas, TAXONOMY["natural_areas"]["label"]),
    "low_density_degraded": (OSMQueryBuilder.get_low_density_degraded, TAXONOMY["low_density_degraded"]["label"]),
    "underused_infrastructure": (OSMQueryBuilder.get_underused_infrastructure, TAXONOMY["underused_infrastructure"]["label"]),
    "frozen_construction": (OSMQueryBuilder.get_frozen_construction, TAXONOMY["frozen_construction"]["label"]),
    "active_construction": (OSMQueryBuilder.get_active_construction, TAXONOMY["active_construction"]["label"]),
    "active_urban": (OSMQueryBuilder.get_active_urban, TAXONOMY["active_urban"]["label"]),
}


def collect_class_data(
    class_name: str,
    max_results: Optional[int] = None,
    check_panoramas: bool = True,
    download: bool = False,
    output_dir: Path = None,
    zoom: int = 1,
    collect_seasons: bool = False,
    target_seasons: Optional[List[str]] = None,
    generate_views: bool = False,
    view_headings: Optional[List[float]] = None,
    view_pitches: Optional[List[float]] = None,
    view_fov: float = 90.0,
    view_size: Tuple[int, int] = (1024, 768),
) -> List[Dict]:
    """Собирает данные для указанного класса"""
    
    if class_name not in CLASS_QUERIES:
        print(f"Error: Unknown class: {class_name}")
        return []
    
    query_func, class_label = CLASS_QUERIES[class_name]
    print(f"\n{'='*70}")
    print(f"Класс: {class_label} ({class_name})")
    print(f"{'='*70}")
    
    # Выполняем запрос
    query = query_func()
    data = query_overpass(query)
    
    if not data:
        print(f"  Error: Query execution failed")
        return []
    
    # Парсим результаты
    results = parse_osm_results(data, class_name)
    print(f"  Found objects in OSM: {len(results)}")
    
    if max_results:
        results = results[:max_results]
        print(f"  → Ограничено до {max_results} объектов")
    
    # Проверяем наличие панорам
    coords_with_pano = []
    if check_panoramas:
        if collect_seasons:
            print(f"\n  Поиск панорам разных времен года...")
        else:
            print(f"\n  Проверка наличия панорам...")
        
        for i, (lat, lon, name, tags, element_type, element_id) in enumerate(results, 1):
            if collect_seasons:
                # Ищем панорамы разных сезонов для этих координат
                seasons_panoramas = find_panoramas_by_seasons(lat, lon, target_seasons)
                
                # Выбираем по одной панораме на каждый сезон (максимум 4)
                selected_panos = []
                for season, panos in seasons_panoramas.items():
                    if panos:
                        # Берем панораму этого сезона, где объект ближе всего к точке (минимальное _distance_m),
                        # а при равном расстоянии - самую свежую по дате
                        panos_sorted = sorted(
                            panos, 
                            key=lambda x: (
                                x.get("_distance_m", 0.0), 
                                -x.get("timestamp", 0)
                            )
                        )
                        best = panos_sorted[0]
                        selected_panos.append({
                            "lat": best["lat"],
                            "lon": best["lon"],
                            "name": name,
                            "tags": tags,
                            "class": class_name,
                            "class_label": class_label,
                            "has_panorama": True,
                            "season": season,
                            **{k: v for k, v in best.items() if k not in ["lat", "lon"]}
                        })
                
                if selected_panos:
                    print(f"    [{i}/{len(results)}] OK {name[:50]} - found {len(selected_panos)} seasons")
                    # Сохраняем как один элемент с несколькими панорамами
                    coords_with_pano.append({
                        "base_lat": lat,
                        "base_lon": lon,
                        "lat": lat,  # Для обратной совместимости
                        "lon": lon,
                        "name": name,
                        "tags": tags,
                        "osm_element_type": element_type,
                        "osm_element_id": element_id,
                        "class": class_name,
                        "class_label": class_label,
                        "has_panorama": True,
                        "seasons": selected_panos  # Список панорам по сезонам
                    })
                else:
                    print(f"    [{i}/{len(results)}] SKIP {name[:50]} (no panoramas)")
            else:
                # Обычная проверка одной панорамы
                exists, metadata = check_panorama_exists(lat, lon)
                if exists:
                    coords_with_pano.append({
                        "lat": lat,
                        "lon": lon,
                        "name": name,
                        "tags": tags,
                        "osm_element_type": element_type,
                        "osm_element_id": element_id,
                        "class": class_name,
                        "class_label": class_label,
                        "has_panorama": True,
                        **(metadata or {})
                    })
                    print(f"    [{i}/{len(results)}] OK {name[:50]}")
                else:
                    print(f"    [{i}/{len(results)}] SKIP {name[:50]} (no panorama)")
            
            # Небольшая задержка чтобы не получить rate limit
            time.sleep(0.3)
    else:
        # Добавляем все результаты без проверки панорам
        coords_with_pano = [
            {
                "lat": lat,
                "lon": lon,
                "name": name,
                "tags": tags,
                "osm_element_type": element_type,
                "osm_element_id": element_id,
                "class": class_name,
                "class_label": class_label,
                "has_panorama": None
            }
            for lat, lon, name, tags, element_type, element_id in results
        ]
    
    print(f"\n  Found with panoramas: {len(coords_with_pano)}")
    
    # Загружаем панорамы если нужно
    if download and output_dir and coords_with_pano:
        print(f"\n  Загрузка панорам в {output_dir}...")
        class_output_dir = output_dir / class_name
        class_output_dir.mkdir(parents=True, exist_ok=True)
        
        successful = 0
        successful_items = 0  # Количество папок с загруженными панорамами
        total_panoramas = 0   # Общее количество загруженных панорам
        items_metadata = []
        
        for i, item in enumerate(coords_with_pano, 1):
            lat, lon = item["lat"], item["lon"]
            safe_name = "".join(c for c in item["name"] if c.isalnum() or c in (' ', '-', '_')).strip()[:40]
            if not safe_name:
                safe_name = "unnamed"
            
            # Создаем уникальный ID для папки (без сезона - одна папка для всех сезонов)
            osm_type = item.get("osm_element_type")
            osm_id = item.get("osm_element_id")
            osm_suffix = f"{osm_type}{osm_id}" if osm_type and osm_id else f"{i:04d}"

            base_item_id = f"{class_name}_{osm_suffix}_{lat:.6f}_{lon:.6f}_{safe_name}"
            # Очищаем ID от недопустимых символов для имени папки
            item_id = "".join(c for c in base_item_id if c.isalnum() or c in (' ', '-', '_', '.')).strip()
            item_id = item_id.replace(' ', '_')
            
            # Создаем папку для изображения
            item_dir = class_output_dir / item_id
            item_dir.mkdir(exist_ok=True)
            metadata_path = item_dir / "metadata.json"
            
            # Проверяем, есть ли уже панорамы в папке
            existing_files = list(item_dir.glob("panorama*.jpg"))
            
            # Если собираем сезоны, загружаем панорамы для каждого сезона
            if collect_seasons and "seasons" in item and item["seasons"]:
                seasons_data = item["seasons"][:4]  # Максимум 4 панорамы
                print(f"    [{i}/{len(coords_with_pano)}] ↓ {item_id} - загрузка {len(seasons_data)} панорам по сезонам")
                
                downloaded_panos = []
                for season_item in seasons_data:
                    season = season_item.get("season", "")
                    pano_lat = season_item.get("lat", lat)
                    pano_lon = season_item.get("lon", lon)
                    
                    image_filename = f"panorama_{season}.jpg"
                    image_path = item_dir / image_filename
                    
                    # Пропускаем если уже есть
                    if image_path.exists():
                        print(f"      ⏭ {image_filename} (уже существует)")
                        continue
                    
                    print(f"      ↓ {image_filename} ({season})")
                    
                    # Загружаем панораму с retry
                    max_retries = 2
                    downloaded = False
                    for retry in range(max_retries):
                        if download_panorama(pano_lat, pano_lon, str(image_path), zoom):
                            image_info = get_image_info(image_path)
                            if image_info:
                                print(f"        OK {image_info['width']}x{image_info['height']} ({image_info['size_bytes']/1024:.1f} KB)")
                            else:
                                print(f"        OK Downloaded")
                            # Генерируем ракурсы если включено
                            views_generated = maybe_generate_views(
                                image_path,
                                enabled=generate_views,
                                headings=view_headings,
                                pitches=view_pitches,
                                fov_deg=view_fov,
                                size=view_size,
                            )
                            if views_generated:
                                print(f"        Views generated: {views_generated}")
                            
                            downloaded_panos.append({
                                "season": season,
                                "filename": image_filename,
                                "lat": pano_lat,
                                "lon": pano_lon,
                                "image": image_info or {},
                                **{k: v for k, v in season_item.items() if k not in ["lat", "lon", "season"]}
                            })
                            successful += 1
                            downloaded = True
                            break
                        else:
                            if retry < max_retries - 1:
                                wait_time = (retry + 1) * 5  # 5, 10 секунд
                                print(f"        Retry {retry + 1}/{max_retries} after {wait_time}s...")
                                time.sleep(wait_time)
                    
                    if not downloaded:
                        print(f"        ERROR Download failed after {max_retries} attempts")
                    
                    time.sleep(2)  # Увеличиваем задержку между загрузками до 2 секунд
                
                # Загружаем информацию о существующих панорамах
                existing_panos_metadata = []
                for existing_file in existing_files:
                    season_from_filename = existing_file.stem.replace("panorama_", "").replace("panorama", "")
                    if season_from_filename and season_from_filename in ["spring", "summer", "autumn", "winter"]:
                        season_name = season_from_filename
                    else:
                        season_name = "unknown"
                    
                    existing_info = get_image_info(existing_file)
                    existing_panos_metadata.append({
                        "season": season_name,
                        "filename": existing_file.name,
                        "image": existing_info or {}
                    })
                
                # Объединяем новые и существующие
                all_panos = downloaded_panos + existing_panos_metadata
                
                # Создаем метаданные со всеми панорамами
                if all_panos or downloaded_panos:
                    item["id"] = item_id
                    item["zoom_level"] = zoom
                    
                    # Определяем основную панораму для базовых метаданных (самая свежая)
                    main_pano = None
                    if downloaded_panos:
                        main_pano = downloaded_panos[0]
                    elif all_panos:
                        main_pano = all_panos[0]
                    
                    main_image_path = item_dir / (main_pano["filename"] if main_pano else "panorama.jpg")
                    item_metadata = create_item_metadata(item, main_image_path, class_name)
                    
                    # Добавляем информацию о всех панорамах
                    item_metadata["panoramas_by_season"] = {
                        pano["season"]: {
                            "filename": pano["filename"],
                            "image": pano.get("image", {}),
                            "lat": pano.get("lat"),
                            "lon": pano.get("lon"),
                            "capture_date": pano.get("capture_date"),
                            "capture_year": pano.get("capture_year"),
                            "capture_month": pano.get("capture_month"),
                            "capture_season": pano.get("season")
                        }
                        for pano in all_panos
                    }
                    item_metadata["panoramas"] = {
                        "total": len(all_panos),
                        "seasons_available": [pano["season"] for pano in all_panos]
                    }
                    
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(item_metadata, f, ensure_ascii=False, indent=2)
                    
                    items_metadata.append(item_metadata)
                    
                    if downloaded_panos or all_panos:
                        successful_items += 1
                        total_panoramas += len(downloaded_panos)  # Добавляем только новые загруженные
                        if downloaded_panos:
                            print(f"      Total panoramas in folder: {len(all_panos)}")
            
            else:
                # Обычная загрузка одной панорамы
                image_filename = "panorama.jpg"
                image_path = item_dir / image_filename
                
                # Проверяем, существует ли уже изображение
                if image_path.exists() and metadata_path.exists():
                    print(f"    [{i}/{len(coords_with_pano)}] ⏭ {item_id} (уже существует)")
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            existing_metadata = json.load(f)
                            items_metadata.append(existing_metadata)
                            successful += 1
                    except:
                        pass
                    continue
                
                print(f"    [{i}/{len(coords_with_pano)}] ↓ {item_id}")
                
                # Добавляем zoom_level в item
                item["zoom_level"] = zoom
                item["id"] = item_id
                
                # Загружаем панораму с retry
                max_retries = 2
                downloaded = False
                for retry in range(max_retries):
                    if download_panorama(lat, lon, str(image_path), zoom):
                        # Получаем информацию об изображении
                        image_info = get_image_info(image_path)
                        if image_info:
                            print(f"      OK Downloaded: {image_info['width']}x{image_info['height']} ({image_info['size_bytes']/1024:.1f} KB)")
                        else:
                            print(f"      ✓ Загружено")

                        # Генерируем ракурсы если включено
                        views_generated = maybe_generate_views(
                            image_path,
                            enabled=generate_views,
                            headings=view_headings,
                            pitches=view_pitches,
                            fov_deg=view_fov,
                            size=view_size,
                        )
                        if views_generated:
                            print(f"      Views generated: {views_generated}")
                        
                        # Создаем полные метаданные
                        item_metadata = create_item_metadata(item, image_path, class_name)
                        
                        # Сохраняем метаданные
                        with open(metadata_path, 'w', encoding='utf-8') as f:
                            json.dump(item_metadata, f, ensure_ascii=False, indent=2)
                        
                        items_metadata.append(item_metadata)
                        successful += 1
                        successful_items += 1
                        downloaded = True
                        break
                    else:
                        if retry < max_retries - 1:
                            wait_time = (retry + 1) * 5  # 5, 10 секунд
                            print(f"      Retry {retry + 1}/{max_retries} after {wait_time}s...")
                            time.sleep(wait_time)
                
                if not downloaded:
                    print(f"      ERROR Download failed after {max_retries} attempts")
                    # Удаляем пустую папку если загрузка не удалась
                    if item_dir.exists() and not any(item_dir.iterdir()):
                        item_dir.rmdir()
                
                time.sleep(2)  # Увеличиваем задержку между загрузками до 2 секунд
        
        # Создаем файл метаданных класса
        class_metadata = {
            "class": class_name,
            "class_label": TAXONOMY[class_name]["label"],
            "class_id": TAXONOMY[class_name]["id"],
            "taxonomy": {
                "main_class": {
                    "id": TAXONOMY[class_name]["id"],
                    "name": class_name,
                    "label": TAXONOMY[class_name]["label"]
                },
                "subclasses": TAXONOMY[class_name]["subclasses"]
            },
            "statistics": {
                "total_items": len(items_metadata),
                "with_panoramas": sum(1 for item in items_metadata if item.get("panorama", {}).get("has_panorama")),
                "with_images": successful,
                "created_at": datetime.now().isoformat()
            },
            "image_statistics": {
                "total_images": successful,
                "total_size_bytes": sum(item.get("image", {}).get("size_bytes", 0) for item in items_metadata),
                "average_size_bytes": sum(item.get("image", {}).get("size_bytes", 0) for item in items_metadata) / successful if successful > 0 else 0,
                "resolutions": {}
            }
        }
        
        # Собираем статистику по разрешениям
        resolutions = defaultdict(int)
        for item in items_metadata:
            img_info = item.get("image", {})
            if img_info.get("width") and img_info.get("height"):
                res = f"{img_info['width']}x{img_info['height']}"
                resolutions[res] += 1
        class_metadata["image_statistics"]["resolutions"] = dict(resolutions)
        
        # Сохраняем метаданные класса
        class_metadata_path = class_output_dir / "class_metadata.json"
        with open(class_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(class_metadata, f, ensure_ascii=False, indent=2)
        
        if collect_seasons:
            print(f"\n  Loaded objects: {successful_items}/{len(coords_with_pano)}")
            print(f"  Total panoramas: {total_panoramas + successful}")
        else:
            print(f"\n  Loaded panoramas: {successful}/{len(coords_with_pano)}")
        print(f"  Class metadata: {class_metadata_path}")
        
        # Обновляем items с путями для обратной совместимости
        for item in coords_with_pano:
            if "id" in item:
                item["item_id"] = item["id"]
                item["directory"] = item["id"]
                item["filename"] = "panorama.jpg"
    
    return coords_with_pano


def main():
    parser = argparse.ArgumentParser(
        description="Сбор датасета панорам по таксономии UTT классов через OSM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Load all classes with all seasons (up to 100 objects per class)
  python3 dataset_collector.py --all-classes-seasons --download
  
  # Проверить все классы (без загрузки)
  python3 dataset_collector.py --class all --check-only
  
  # Собрать данные по одному классу
  python3 dataset_collector.py --class natural_areas --max-results 50
  
  # Собрать и загрузить панорамы
  python3 dataset_collector.py --class low_density_degraded --download --output-dir data/dataset
  
  # Собрать панорамы разных времен года (до 4 панорам на объект)
  python3 dataset_collector.py --class natural_areas --collect-seasons --download
  
  # Собрать только летние и зимние панорамы
  python3 dataset_collector.py --class natural_areas --collect-seasons --target-seasons summer winter --download
        """
    )
    
    parser.add_argument(
        "--class",
        dest="class_name",
        choices=list(CLASS_QUERIES.keys()) + ["all"],
        default="all",
        help="Класс для сбора данных (по умолчанию: all)"
    )
    
    parser.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="Максимальное количество результатов на класс"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/dataset",
        help="Директория для сохранения данных (по умолчанию: data/dataset)"
    )

    parser.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        metavar=("SOUTH", "WEST", "NORTH", "EAST"),
        default=None,
        help="Границы bbox для Overpass: SOUTH WEST NORTH EAST (по умолчанию: Москва)"
    )
    
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Только проверить наличие панорам, не загружать"
    )
    
    parser.add_argument(
        "--download",
        action="store_true",
        help="Загружать панорамы после проверки"
    )
    
    parser.add_argument(
        "--zoom",
        type=int,
        default=1,
        help="Уровень зума для панорам (по умолчанию: 1)"
    )
    
    parser.add_argument(
        "--collect-seasons",
        action="store_true",
        help="Собирать панорамы разных времен года (ищет панорамы вблизи координат, снятые в разные сезоны)"
    )
    
    parser.add_argument(
        "--target-seasons",
        nargs="+",
        choices=["spring", "summer", "autumn", "winter"],
        default=None,
        help="Целевые сезоны для сбора (по умолчанию: все сезоны)"
    )
    
    parser.add_argument(
        "--all-classes-seasons",
        action="store_true",
        help="Загрузить все классы со всеми сезонами (до 100 объектов на класс). Эквивалентно --class all --collect-seasons --max-results 100"
    )

    parser.add_argument(
        "--generate-views",
        action="store_true",
        help="После загрузки панорамы сгенерировать несколько перспективных ракурсов (views/) из equirectangular panorama*.jpg"
    )
    parser.add_argument(
        "--view-headings",
        nargs="+",
        type=float,
        default=[0.0, 90.0, 180.0, 270.0],
        help="Список heading (yaw) в градусах для генерации ракурсов (по умолчанию: 0 90 180 270)"
    )
    parser.add_argument(
        "--view-pitches",
        nargs="+",
        type=float,
        default=[0.0],
        help="Список pitch в градусах для генерации ракурсов (по умолчанию: 0)"
    )
    parser.add_argument(
        "--view-fov",
        type=float,
        default=90.0,
        help="Horizontal FOV в градусах для перспективных ракурсов (по умолчанию: 90)"
    )
    parser.add_argument(
        "--view-size",
        nargs=2,
        type=int,
        default=[1024, 768],
        metavar=("W", "H"),
        help="Размер перспективных ракурсов (по умолчанию: 1024 768)"
    )
    
    args = parser.parse_args()

    # Переопределяем границы, если переданы
    if args.bounds is not None:
        set_bounds(args.bounds[0], args.bounds[1], args.bounds[2], args.bounds[3])
    
    # Автоматическая настройка параметров для --all-classes-seasons
    if args.all_classes_seasons:
        args.class_name = "all"
        args.collect_seasons = True
        if args.max_results is None:
            args.max_results = 100
        if args.download is False:
            args.download = True
        print("\n" + "="*70)
        print("РЕЖИМ: Загрузка всех классов со всеми сезонами")
        print("="*70)
        print(f"  Классов: все ({len(CLASS_QUERIES)} классов)")
        print(f"  Максимум объектов на класс: {args.max_results}")
        print(f"  Сезоны: все (spring, summer, autumn, winter)")
        print(f"  Выходная директория: {args.output_dir}")
        print("="*70 + "\n")
    
    # Определяем классы для обработки
    if args.class_name == "all":
        classes_to_process = list(CLASS_QUERIES.keys())
    else:
        classes_to_process = [args.class_name]
    
    # Создаем выходную директорию
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Собираем данные по каждому классу
    all_metadata = []
    stats = defaultdict(int)
    
    print(f"\n{'='*70}")
    print(f"СБОР ДАТАСЕТА UTT ТЕРРИТОРИЙ")
    print(f"{'='*70}")
    print(f"Классы для обработки: {', '.join(classes_to_process)}")
    print(f"Максимум результатов на класс: {args.max_results or 'без ограничений'}")
    print(f"Выходная директория: {output_dir}")
    print(f"Режим: {'только проверка' if args.check_only else 'загрузка' if args.download else 'поиск в OSM'}")
    
    for class_name in classes_to_process:
        results = collect_class_data(
            class_name=class_name,
            max_results=args.max_results,
            check_panoramas=not args.check_only or True,  # Всегда проверяем панорамы
            download=args.download and not args.check_only,
            output_dir=output_dir,
            zoom=args.zoom,
            collect_seasons=args.collect_seasons,
            target_seasons=args.target_seasons,
            generate_views=args.generate_views,
            view_headings=args.view_headings,
            view_pitches=args.view_pitches,
            view_fov=args.view_fov,
            view_size=(args.view_size[0], args.view_size[1]),
        )
        
        all_metadata.extend(results)
        stats[class_name] = len(results)
        
        # Сохраняем промежуточные результаты
        class_metadata_file = output_dir / f"{class_name}_metadata.json"
        with open(class_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  Metadata saved: {class_metadata_file}")
    
    # Сохраняем общие метаданные
    all_metadata_file = output_dir / "all_metadata.json"
    with open(all_metadata_file, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)
    
    # Сохраняем таксономию в корне датасета
    taxonomy_file = output_dir / "taxonomy.json"
    with open(taxonomy_file, 'w', encoding='utf-8') as f:
        json.dump(TAXONOMY, f, ensure_ascii=False, indent=2)
    
    # Создаем общую статистику датасета
    dataset_stats = {
        "created_at": datetime.now().isoformat(),
        "version": "1.0",
        "statistics": {
            "total_items": len(all_metadata),
            "by_class": {class_name: count for class_name, count in stats.items()}
        },
        "taxonomy": TAXONOMY
    }
    
    dataset_stats_file = output_dir / "dataset_statistics.json"
    with open(dataset_stats_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_stats, f, ensure_ascii=False, indent=2)
    
    # Выводим статистику
    print(f"\n{'='*70}")
    print(f"РЕЗУЛЬТАТЫ")
    print(f"{'='*70}")
    print(f"Всего объектов с панорамами: {len(all_metadata)}")
    print(f"\nПо классам:")
    for class_name, count in stats.items():
        label = CLASS_QUERIES[class_name][1]
        print(f"  {label}: {count}")
    print(f"\nMetadata: {all_metadata_file}")
    print(f"Taxonomy: {taxonomy_file}")
    print(f"Dataset statistics: {dataset_stats_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

