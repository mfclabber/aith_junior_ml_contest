#!/usr/bin/env python3
"""Zero-shot бенчмарк Qwen3-VL на perspective-кропах (классификация UTT).

Сравнительная точка для дообученной PaliGemma: насколько сильная открытая VLM
справляется с нашей таксономией «из коробки», без дообучения.

Модель тяжёлая (скачивание весов). По умолчанию берётся Qwen/Qwen2.5-VL-7B-Instruct;
можно указать другую через --model.

Пример:
  CUDA_VISIBLE_DEVICES=3 python3 scripts/vlm/eval_qwen_zeroshot.py \
      --limit 200 --model Qwen/Qwen2.5-VL-7B-Instruct
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

CLASS_HINTS = {
    "natural_areas": "природная территория, зелень, лес, поле, вода, без застройки",
    "low_density_degraded": "низкая плотность, деградация, пустыри, гаражи, ветхое",
    "underused_infrastructure": "недоиспользуемая инфраструктура, склады, парковки, ЛЭП",
    "frozen_construction": "замороженная стройка, брошенный недострой, ржавый каркас",
    "active_construction": "активная стройка, краны, техника, котлован, монолит",
    "active_urban": "живой город, жилые дома, магазины, благоустройство, люди",
}


def build_prompt(class_names: list[str]) -> str:
    lines = [f"- {c}: {CLASS_HINTS.get(c, '')}" for c in class_names]
    return (
        "Ты классифицируешь тип городской территории (urban tissue) по уличному фото.\n"
        "Выбери РОВНО один класс из списка и ответь только его англоязычным ключом.\n\n"
        + "\n".join(lines)
        + "\n\nОтвет (только ключ класса):"
    )


def parse_class(text: str, class_names: list[str]) -> str | None:
    t = text.strip().lower()
    for c in class_names:
        if c.lower() in t:
            return c
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen VLM zero-shot eval on crops.")
    ap.add_argument("--data-dir", type=Path, default=Path("data/ml_perspective"))
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("results/eval_qwen_zeroshot.json"))
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    data_dir = args.data_dir if args.data_dir.is_absolute() else root / args.data_dir
    out = args.out if args.out.is_absolute() else root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    meta = json.loads((data_dir / "metadata_perspective.json").read_text(encoding="utf-8"))
    class_names = list(meta["class_mapping"].keys())
    val = meta["val_samples"]
    if args.limit:
        val = val[: args.limit]
    prompt = build_prompt(class_names)

    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} ...")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device
    ).eval()

    per_class_tot: dict[str, int] = defaultdict(int)
    per_class_ok: dict[str, int] = defaultdict(int)
    correct = 0
    n = 0
    for i, s in enumerate(val):
        img = Image.open(data_dir / s["image_path"]).convert("RGB")
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ]}
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], images=[img], return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        ans = processor.batch_decode(
            gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0]
        pred = parse_class(ans, class_names)
        gt = s["class_name"]
        per_class_tot[gt] += 1
        if pred == gt:
            correct += 1
            per_class_ok[gt] += 1
        n += 1
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(val)} acc={correct / n:.3f}")

    acc = correct / max(n, 1)
    per_class_f1 = {c: (per_class_ok[c] / per_class_tot[c] if per_class_tot[c] else 0.0)
                    for c in class_names}
    macro = sum(per_class_f1.values()) / len(class_names)
    result = {
        "model": args.model,
        "n": n,
        "accuracy": round(acc, 4),
        "macro_recall": round(macro, 4),
        "per_class_recall": {k: round(v, 4) for k, v in per_class_f1.items()},
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
