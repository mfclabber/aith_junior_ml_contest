#!/usr/bin/env python3
"""Переразметка кропов по СОДЕРЖИМОМУ независимым VLM-судьёй (Qwen2.5-VL).

Зачем: исходные метки берутся из тегов OSM и НЕ соответствуют тому, что реально
видно на кропе (проверено визуально: natural_areas -> шоссе, active_construction ->
историческое здание). На таких метках хорошие метрики недостижимы.

Скрипт прогоняет сильную открытую VLM как разметчика: по каждому кропу выдаёт
класс UTT + «уверен/не уверен». Результат пишется инкрементально в JSONL
(резюмируемо), из него потом собирается self-consistent датасет.

Разметчик (Qwen2.5-VL) НЕЗАВИСИМ от классификатора (CLIP linear probe), поэтому
последующая метрика learnable и не циркулярна.

Пример:
  CUDA_VISIBLE_DEVICES=3 ./venv/bin/python scripts/data_quality/vlm_relabel.py \
      --metadata data/ml_perspective/metadata_perspective_clean.json \
      --out data/vlm_relabel/labels.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

CLASS_DESC = {
    "natural_areas": "природная территория: парк, лес, поле, вода, зелень, без застройки",
    "low_density_degraded": "деградировавшая малоэтажная застройка: пустыри, гаражи, ветхие/заброшенные строения",
    "underused_infrastructure": "недоиспользуемая инфраструктура: большие парковки, склады, ЛЭП, техзоны",
    "frozen_construction": "замороженная стройка: брошенный недострой, ржавый каркас, без активности",
    "active_construction": "активная стройка: краны, техника, котлован, монолит, идут работы",
    "active_urban": "живой город: жилые/офисные дома, магазины, улицы, благоустройство, люди",
}

# 3 макро-класса, надёжно определимые по одному уличному кадру
CLASSES_3 = ["natural_areas", "construction", "built_up"]
CLASS_DESC_3 = {
    "natural_areas": "преобладает природа: зелень, парк, лес, поле, вода, деревья, газоны",
    "construction": "видна стройка: башенные краны, строительная техника, котлован, "
                    "монолитный каркас, недострой, строительный забор",
    "built_up": "обычная городская застройка: жилые/офисные/торговые здания, дороги, "
                "дворы, благоустройство (без активной стройки)",
}


def build_prompt(class_names):
    lines = [f"{i+1}. {c} - {CLASS_DESC[c]}" for i, c in enumerate(class_names)]
    return (
        "На фото - перспективный вид городской территории. Определи ОДИН тип территории "
        "по тому, что РЕАЛЬНО видно на снимке (игнорируй небо).\n\n"
        + "\n".join(lines)
        + "\n\nОтветь строго в формате: `<ключ_класса> | <confident|unsure>`. "
        "Например: `active_urban | confident`. Ключ - англ. название из списка."
    )


def build_prompt3():
    lines = [f"- {c}: {CLASS_DESC_3[c]}" for c in CLASSES_3]
    return (
        "На фото - вид городской территории с улицы. Что ПРЕОБЛАДАЕТ в кадре "
        "(смотри на землю и объекты, игнорируй небо)? Выбери ровно одно:\n\n"
        + "\n".join(lines)
        + "\n\nОтветь строго: `<natural_areas|construction|built_up> | <confident|unsure>`."
    )


def parse(text, class_names):
    t = text.strip().lower()
    cls = None
    # порядок важен: сначала более специфичные ключи
    for c in sorted(class_names, key=len, reverse=True):
        if c.lower() in t:
            cls = c
            break
    conf = "unsure" not in t
    return cls, conf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=Path,
                    default=Path("data/ml_perspective/metadata_perspective_clean.json"))
    ap.add_argument("--data-dir", type=Path, default=Path("data/ml_perspective"))
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--out", type=Path, default=Path("data/vlm_relabel/labels.jsonl"))
    ap.add_argument("--taxonomy", choices=("6", "3"), default="6",
                    help="6=исходные классы; 3=natural/construction/built_up (определимо с земли)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    data_dir = args.data_dir if args.data_dir.is_absolute() else root / args.data_dir
    meta_path = args.metadata if args.metadata.is_absolute() else root / args.metadata
    out = args.out if args.out.is_absolute() else root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if args.taxonomy == "3":
        class_names = CLASSES_3
        prompt = build_prompt3()
    else:
        class_names = [k for k, _ in sorted(meta["class_mapping"].items(), key=lambda x: x[1])]
        prompt = build_prompt(class_names)

    rows = []
    for split, key in (("train", "train_samples"), ("val", "val_samples")):
        for s in meta[key]:
            rows.append({"split": split, **s})
    if args.limit:
        rows = rows[: args.limit]

    done = set()
    if out.is_file():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["image_path"])
        print(f"resume: уже размечено {len(done)}")

    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} ...")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device
    ).eval()

    n = 0
    with out.open("a", encoding="utf-8") as fh:
        for i, s in enumerate(rows):
            if s["image_path"] in done:
                continue
            img = Image.open(data_dir / s["image_path"]).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[img], return_tensors="pt").to(device)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=24, do_sample=False)
            ans = processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:],
                                         skip_special_tokens=True)[0]
            pred, conf = parse(ans, class_names)
            fh.write(json.dumps({
                "image_path": s["image_path"], "split": s["split"],
                "osm_label": s["class_name"], "vlm_label": pred,
                "vlm_confident": conf, "object_id": s.get("object_id"),
                "raw": ans.strip()[:80],
            }, ensure_ascii=False) + "\n")
            fh.flush()
            n += 1
            if n % 25 == 0:
                print(f"{n} размечено (последний: {s['class_name']} -> {pred})")
    print(f"Готово: +{n} меток -> {out}")


if __name__ == "__main__":
    main()
