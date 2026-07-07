"""Маппинг правок аналитиков → UTT class_key (6 классов)."""

from __future__ import annotations

ANALYST_CLASS_TO_KEY: dict[str, str] = {
    "природные территории": "natural_areas",
    "природные (парковые) территории": "natural_areas",
    "низкоплотная застройка / деградировавшие антропогенные объекты": "low_density_degraded",
    "низкоплотная застройка / деградировавшие объекты": "low_density_degraded",
    "недоиспользуемые инфраструктурные/городские зоны": "underused_infrastructure",
    "незавершенное/приостановленное строительство": "frozen_construction",
    "активное строительство": "active_construction",
    "активное строительство (стройплощадки)": "active_construction",
    "активные городские территории": "active_urban",
    "активные городские территории (фон/контрольный класс)": "active_urban",
}


def analyst_class_to_key(class_ru: str) -> str | None:
    if not class_ru or class_ru.strip().lower() in ("", "nan", "категории нет"):
        return None
    key = ANALYST_CLASS_TO_KEY.get(class_ru.strip().lower())
    if key:
        return key
    low = class_ru.strip().lower()
    for pat, k in ANALYST_CLASS_TO_KEY.items():
        if pat in low or low in pat:
            return k
    return None
