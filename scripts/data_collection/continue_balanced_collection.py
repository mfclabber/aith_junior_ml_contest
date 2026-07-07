#!/usr/bin/env python3
"""
Продолжение сбалансированного сбора в существующую директорию.

Использование:
    python3 continue_balanced_collection.py --existing-dir data/big_dataset --target-panoramas 5000
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List
import argparse

# Все классы для сбора
ALL_CLASSES = [
    "natural_areas",
    "low_density_degraded",
    "underused_infrastructure",
    "frozen_construction",
    "active_construction",
    "active_urban",
]

def count_panoramas_in_class(output_dir: Path, class_name: str) -> int:
    """Подсчитывает количество панорам в классе (с учетом сезонов)"""
    class_dir = output_dir / class_name
    if not class_dir.exists():
        return 0
    
    count = 0
    for item_dir in class_dir.iterdir():
        if not item_dir.is_dir():
            continue
        
        # Проверяем metadata.json для подсчета панорам
        metadata_path = item_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Если есть panoramas_by_season, считаем их
                if "panoramas_by_season" in metadata:
                    count += len(metadata["panoramas_by_season"])
                # Если есть panoramas.total (для сезонов)
                elif "panoramas" in metadata and isinstance(metadata["panoramas"], dict):
                    count += metadata["panoramas"].get("total", 0)
                # Или одна панорама
                elif "panorama" in metadata and metadata["panorama"].get("has_panorama"):
                    count += 1
            except Exception as e:
                # Если не удалось прочитать metadata, считаем файлы
                pass
        
        # Если нет metadata или не удалось прочитать, считаем файлы panorama*.jpg
        panorama_files = list(item_dir.glob("panorama*.jpg"))
        if panorama_files:
            count += len(panorama_files)
    
    return count


def get_class_statistics(output_dir: Path) -> Dict[str, int]:
    """Получает статистику по всем классам"""
    stats = {}
    for class_name in ALL_CLASSES:
        stats[class_name] = count_panoramas_in_class(output_dir, class_name)
    return stats


def collect_class(
    class_name: str,
    max_results: int,
    output_dir: Path,
    zoom: int = 1,
    collect_seasons: bool = True,
) -> int:
    """Собирает данные для одного класса и возвращает количество загруженных панорам"""
    print(f"\n{'='*70}")
    print(f"Сбор класса: {class_name}")
    print(f"  MAX_RESULTS: {max_results}")
    print(f"{'='*70}")
    
    cmd = [
        sys.executable,
        "scripts/data_collection/dataset_collector.py",
        "--class", class_name,
        "--max-results", str(max_results),
        "--collect-seasons" if collect_seasons else "",
        "--download",
        "--output-dir", str(output_dir),
        "--zoom", str(zoom),
    ]
    cmd = [c for c in cmd if c]  # Убираем пустые строки
    
    try:
        # Запускаем без capture_output, чтобы видеть прогресс в реальном времени
        result = subprocess.run(cmd, timeout=3600)
        if result.returncode != 0:
            print(f"  WARNING: Exit code {result.returncode}")
        
        # Подсчитываем результат
        new_count = count_panoramas_in_class(output_dir, class_name)
        return new_count
    except subprocess.TimeoutExpired:
        print(f"  ERROR: Timeout (>1 hour)")
        return count_panoramas_in_class(output_dir, class_name)
    except Exception as e:
        print(f"  ERROR: {e}")
        return count_panoramas_in_class(output_dir, class_name)


def main():
    parser = argparse.ArgumentParser(
        description="Продолжение сбалансированного сбора в существующую директорию",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Продолжить сбор в data/big_dataset до 5000 панорам
  python3 continue_balanced_collection.py --existing-dir data/big_dataset --target-panoramas 5000
  
  # Продолжить сбор до 10000 панорам
  python3 continue_balanced_collection.py --existing-dir data/big_dataset --target-panoramas 10000
        """
    )
    
    parser.add_argument(
        "--existing-dir",
        type=str,
        required=True,
        help="Существующая директория с уже загруженными панорамами"
    )
    
    parser.add_argument(
        "--target-panoramas",
        type=int,
        default=5000,
        help="Целевое количество панорам (по умолчанию: 5000)"
    )
    
    parser.add_argument(
        "--initial-max-results",
        type=int,
        default=1000,
        help="Начальный лимит OSM объектов на класс (по умолчанию: 1000)"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Максимум итераций досбора для каждого класса (по умолчанию: 5)"
    )
    
    parser.add_argument(
        "--zoom",
        type=int,
        default=1,
        help="Уровень зума панорам (по умолчанию: 1)"
    )
    
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=None,
        help="Минимум панорам на класс (по умолчанию: target_panoramas / количество_классов)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.existing_dir)
    if not output_dir.exists():
        print(f"ERROR: Директория {output_dir} не существует!")
        sys.exit(1)
    
    target_per_class = args.min_per_class or (args.target_panoramas // len(ALL_CLASSES))
    
    # Получаем текущую статистику
    current_stats = get_class_statistics(output_dir)
    current_total = sum(current_stats.values())
    
    print(f"\n{'='*70}")
    print(f"ПРОДОЛЖЕНИЕ СБОРА В СУЩЕСТВУЮЩУЮ ДИРЕКТОРИЮ")
    print(f"{'='*70}")
    print(f"Директория: {output_dir}")
    print(f"Текущее количество панорам: {current_total}")
    print(f"Целевое количество панорам: {args.target_panoramas}")
    print(f"Минимум на класс: {target_per_class}")
    print(f"Начальный лимит OSM объектов: {args.initial_max_results}")
    print(f"{'='*70}\n")
    
    print("ТЕКУЩАЯ СТАТИСТИКА:")
    print("-" * 70)
    for class_name, count in current_stats.items():
        status = "✓" if count >= target_per_class else "✗"
        print(f"  {status} {class_name:30s}: {count:4d} панорам")
    print(f"{'='*70}\n")
    
    # Определяем классы, которые нужно собрать или дособрать
    classes_to_collect = []
    for class_name in ALL_CLASSES:
        current = current_stats[class_name]
        if current < target_per_class:
            classes_to_collect.append((class_name, target_per_class - current))
    
    if not classes_to_collect:
        print("✓ Все классы уже достигли целевого количества!")
        print(f"  Всего панорам: {current_total}")
        return
    
    print(f"Классы для сбора/досбора: {len(classes_to_collect)}")
    for class_name, needed in classes_to_collect:
        print(f"  - {class_name}: текущее {current_stats[class_name]}, нужно еще ~{needed}")
    print()
    
    # Первый проход: собираем недостающие классы
    print("ПЕРВЫЙ ПРОХОД: сбор недостающих классов")
    print("-" * 70)
    
    for class_name, needed in classes_to_collect:
        # Если класс уже есть, но недостаточно - увеличиваем лимит
        current = current_stats[class_name]
        if current > 0:
            additional_objects = max(500, needed * 3)
            max_results = args.initial_max_results + additional_objects
        else:
            max_results = args.initial_max_results
        
        collect_class(
            class_name=class_name,
            max_results=max_results,
            output_dir=output_dir,
            zoom=args.zoom,
            collect_seasons=True,
        )
        time.sleep(5)  # Пауза между классами
        
        # Обновляем статистику
        current_stats[class_name] = count_panoramas_in_class(output_dir, class_name)
    
    # Проверяем статистику
    stats = get_class_statistics(output_dir)
    total = sum(stats.values())
    
    print(f"\n{'='*70}")
    print(f"СТАТИСТИКА ПОСЛЕ ПЕРВОГО ПРОХОДА")
    print(f"{'='*70}")
    for class_name, count in stats.items():
        status = "✓" if count >= target_per_class else "✗"
        print(f"  {status} {class_name:30s}: {count:4d} панорам")
    print(f"{'='*70}")
    print(f"Всего: {total} панорам (цель: {args.target_panoramas})")
    print(f"{'='*70}\n")
    
    # Итеративная досборка
    iteration = 0
    while iteration < args.max_iterations:
        iteration += 1
        print(f"\n{'='*70}")
        print(f"ИТЕРАЦИЯ ДОСБОРКИ #{iteration}")
        print(f"{'='*70}")
        
        needs_more = []
        for class_name in ALL_CLASSES:
            current = stats[class_name]
            if current < target_per_class:
                needs_more.append((class_name, target_per_class - current))
        
        if not needs_more:
            print("  Все классы достигли целевого количества!")
            break
        
        print(f"  Классы, требующие досборки: {len(needs_more)}")
        for class_name, needed in needs_more:
            print(f"    - {class_name}: нужно еще ~{needed} панорам")
        
        # Дособираем для каждого класса
        for class_name, needed in needs_more:
            additional_objects = max(500, needed * 3)
            new_max = args.initial_max_results + additional_objects
            
            print(f"\n  Досборка: {class_name}")
            print(f"    Текущее: {stats[class_name]} панорам")
            print(f"    Нужно: ~{needed} панорам")
            print(f"    Новый лимит OSM объектов: {new_max}")
            
            before = stats[class_name]
            collect_class(
                class_name=class_name,
                max_results=new_max,
                output_dir=output_dir,
                zoom=args.zoom,
                collect_seasons=True,
            )
            after = count_panoramas_in_class(output_dir, class_name)
            stats[class_name] = after
            added = after - before
            print(f"    Добавлено: {added} панорам")
            
            time.sleep(5)
        
        # Обновляем общую статистику
        total = sum(stats.values())
        print(f"\n  Обновленная статистика:")
        for class_name, count in stats.items():
            status = "✓" if count >= target_per_class else "✗"
            print(f"    {status} {class_name:30s}: {count:4d} панорам")
        print(f"  Всего: {total} панорам")
        
        # Проверяем, достигли ли цели
        if total >= args.target_panoramas:
            all_classes_ok = all(count >= target_per_class for count in stats.values())
            if all_classes_ok:
                print(f"\n  ✓ Цель достигнута! Все классы сбалансированы.")
                break
    
    # Финальная статистика
    stats = get_class_statistics(output_dir)
    total = sum(stats.values())
    
    print(f"\n{'='*70}")
    print(f"ФИНАЛЬНАЯ СТАТИСТИКА")
    print(f"{'='*70}")
    for class_name, count in stats.items():
        pct = (count / total * 100) if total > 0 else 0
        status = "✓" if count >= target_per_class else "⚠"
        print(f"  {status} {class_name:30s}: {count:5d} панорам ({pct:5.1f}%)")
    print(f"{'='*70}")
    print(f"  ВСЕГО: {total} панорам")
    print(f"  Цель: {args.target_panoramas} панорам")
    print(f"  Достигнуто: {'✓' if total >= args.target_panoramas else '✗'}")
    print(f"{'='*70}\n")
    
    # Сохраняем сводку
    summary = {
        "target_panoramas": args.target_panoramas,
        "target_per_class": target_per_class,
        "total_panoramas": total,
        "by_class": stats,
        "iterations": iteration,
        "existing_dir": str(output_dir),
    }
    
    summary_path = output_dir / "collection_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"Сводка сохранена: {summary_path}")


if __name__ == "__main__":
    main()
