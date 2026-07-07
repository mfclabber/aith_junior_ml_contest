"""Каноническая таксономия UTT: ключ модели → подписи для UI (формат аналитиков)."""

from __future__ import annotations

# class_key → (class_ru, default_subclass_ru)
UTT_LABELS: dict[str, tuple[str, str]] = {
    "natural_areas": (
        "Природные территории",
        "Парки и скверы",
    ),
    "low_density_degraded": (
        "Низкоплотная застройка / Деградировавшие антропогенные объекты",
        "Заброшенные здания",
    ),
    "underused_infrastructure": (
        "Недоиспользуемые инфраструктурные/городские зоны",
        "Парковки с низкой загрузкой",
    ),
    "frozen_construction": (
        "Незавершенное/приостановленное строительство",
        "Замороженные объекты",
    ),
    "active_construction": (
        "Активное строительство",
        "Стройплощадки с активностью",
    ),
    "active_urban": (
        "Активные городские территории",
        "Жилая застройка с активностью",
    ),
}

NO_PANORAMA_CLASS = "Категории нет"
NO_PANORAMA_SUBCLASS = ""
NO_PANORAMA_STATUS = "Нет панорамы"


def labels_for_class_key(class_key: str) -> tuple[str, str]:
    return UTT_LABELS.get(
        class_key,
        (
            "Активные городские территории",
            "Жилая застройка с активностью",
        ),
    )


def refine_subclass(class_key: str, desc: str, default_sub: str) -> str:
    """Уточнение подкласса по описанию — подклассы из ОС аналитиков."""
    if not desc:
        return default_sub
    d = desc.lower()

    if class_key == "natural_areas":
        if any(x in d for x in ("вод", "река", "пруд", "канал", "водоём", "водоем")):
            return "Водные объекты"
        if any(x in d for x in ("лес", "лесополос", "лесной")):
            return "Лесные массивы"
        if any(x in d for x in ("поле", "распаш", "пашн", "сельхоз")):
            return "Поле"
        if any(x in d for x in ("дорог", "проезж", "тротуар")) and any(
            x in d for x in ("зелень", "дерев", "раститель", "газон")
        ):
            return "Растительность вдоль дорог"
        if any(x in d for x in ("благоустро", "сквер", "парк")):
            return "Благоустроенные парки и скверы"
        return default_sub

    if class_key == "underused_infrastructure":
        if any(x in d for x in ("промышлен", "промзон", "завод", "фабрик", "склад", "ангар")):
            return "Промзона"
        if any(x in d for x in ("ларьк", "киоск", "будк", "трансформатор", "остановк", "вентиляц")):
            return "Отдельные сооружения"
        return default_sub

    if class_key == "active_urban":
        if any(x in d for x in ("школ", "детск", "сад", "образоват", "вуз", "университет", "научн")):
            return "Образовательные/медицинские/офисные комплексы"
        if any(x in d for x in ("администр", "больниц", "поликлин", "медицин")):
            return "Образовательные/медицинские/офисные комплексы"
        if any(x in d for x in ("коммерц", "магазин", "торгов", "тц", "трц", "ритейл")):
            return "Коммерческие улицы"
        if any(x in d for x in ("дорог", "транспорт", "развязк", "магистрал", "асфальт", "грунт")):
            return "Интенсивные транспортные коридоры"
        if any(x in d for x in ("бц", "бизнес", "офис")):
            return "Образовательные/медицинские/офисные комплексы"
        return default_sub

    if class_key == "active_construction":
        if any(x in d for x in ("новая застройк", "новострой")):
            return "Новая застройка"
        return default_sub

    return default_sub
