#!/usr/bin/env python3
"""Data-centric чистка perspective-датасета.

Главная проблема качества: для объектов с неизвестным азимутом (null-bearing)
генерировалось 4 кропа (N/E/S/W), но объект реально виден лишь в 1 из них -
остальные 3 это шум метки. Плюс битые/пустые/размытые кадры и дубликаты.

Что делает скрипт:
  1. Метрики качества каждого кропа: std яркости (пустой/однотонный кадр),
     резкость (edge energy), aHash (near-duplicates).
  2. Zero-shot CLIP-скоринг вероятности размеченного класса (НЕ дообученная
     модель, чтобы не было циркулярности).
  3. Heading-дизамбигуация: для мультихединг-объектов оставляем один кроп -
     с максимальной вероятностью размеченного класса, остальные отбрасываем.
  4. Quality-фильтр (пустые/размытые) и дедуп внутри объекта.
  5. confident-learning флаги: сильное несогласие CLIP с меткой (для аудита).

Честное разделение:
  - VAL = только single-heading (known-bearing) кропы: азимут «на объект»
    известен геометрически, метка достоверна, выбор не зависит от модели.
  - TRAIN = single-heading + лучший heading на мультихединг-объект.

Выход:
  - data/ml_perspective/metadata_perspective_clean.json
  - results/cleaning_report.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

HEADING_RE = re.compile(r"_h(\d{3})\.jpg$")

PROMPTS_EN: dict[str, str] = {
    "natural_areas": "a street panorama of parks, forests, meadows, green recreational nature",
    "low_density_degraded": (
        "a street panorama of abandoned buildings, vandalism, degraded low-density "
        "dilapidated urban fabric"
    ),
    "underused_infrastructure": (
        "a street panorama of underused parking, empty storefronts, obsolete urban "
        "infrastructure zones"
    ),
    "frozen_construction": (
        "a street panorama of stalled construction sites, frozen unfinished buildings, "
        "inactive cranes"
    ),
    "active_construction": (
        "a street panorama of active construction sites, cranes, new buildings under construction"
    ),
    "active_urban": (
        "a street panorama of busy urban streets, dense mixed-use city life, commerce "
        "and intensive urban activity"
    ),
}


def is_multiheading(image_path: str) -> bool:
    return HEADING_RE.search(image_path) is not None


def quality_metrics(img: Image.Image) -> tuple[float, float, int]:
    """Возвращает (std_яркости, резкость, aHash64). Работает на уменьшенной копии."""
    g = img.convert("L").resize((128, 128))
    a = np.asarray(g, dtype=np.float32)
    std = float(a.std())
    gx = np.diff(a, axis=1)
    gy = np.diff(a, axis=0)
    sharp = float((gx * gx).mean() + (gy * gy).mean())
    small = np.asarray(img.convert("L").resize((8, 8)), dtype=np.float32)
    bits = (small > small.mean()).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return std, sharp, h


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


@torch.no_grad()
def clip_scores(rows, data_dir: Path, class_names, device, batch_size=64):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    prompts = [PROMPTS_EN[c] for c in class_names]
    tokens = tokenizer(prompts).to(device)
    tf = model.encode_text(tokens)
    tf = tf / tf.norm(dim=-1, keepdim=True)
    scale = model.logit_scale.exp()

    probs = np.zeros((len(rows), len(class_names)), dtype=np.float32)
    buf, idxs = [], []

    def flush():
        if not buf:
            return
        x = torch.stack(buf).to(device)
        f = model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        logits = scale * (f @ tf.T)
        p = torch.softmax(logits, dim=-1).cpu().numpy()
        for j, gi in enumerate(idxs):
            probs[gi] = p[j]
        buf.clear()
        idxs.clear()

    for i, s in enumerate(rows):
        img = Image.open(data_dir / s["image_path"]).convert("RGB")
        buf.append(preprocess(img))
        idxs.append(i)
        if len(buf) >= batch_size:
            flush()
    flush()
    return probs


def main() -> None:
    ap = argparse.ArgumentParser(description="Data-centric cleaning of perspective dataset.")
    ap.add_argument("--data-dir", type=Path, default=Path("data/ml_perspective"))
    ap.add_argument("--meta", type=Path, default=None)
    ap.add_argument("--out-meta", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=Path("results/cleaning_report.json"))
    ap.add_argument("--std-min", type=float, default=10.0, help="ниже - пустой/однотонный кадр")
    ap.add_argument("--sharp-pct", type=float, default=3.0, help="дропнуть нижние N%% по резкости")
    ap.add_argument("--dup-hamming", type=int, default=4, help="aHash <= N -> дубликат")
    ap.add_argument("--suspect-prob", type=float, default=0.10,
                    help="если CLIP argmax != метка и prob метки < N -> suspect (аудит)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    data_dir = args.data_dir if args.data_dir.is_absolute() else root / args.data_dir
    meta_path = args.meta or (data_dir / "metadata_perspective.json")
    meta_path = meta_path if Path(meta_path).is_absolute() else root / meta_path
    out_meta = args.out_meta or (data_dir / "metadata_perspective_clean.json")
    out_meta = out_meta if Path(out_meta).is_absolute() else root / out_meta
    report_path = args.report if args.report.is_absolute() else root / args.report

    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    class_mapping: dict[str, int] = meta["class_mapping"]
    class_names = [k for k, _ in sorted(class_mapping.items(), key=lambda x: x[1])]
    device = torch.device(args.device)

    # Собрать все сэмплы с исходным сплитом
    rows: list[dict[str, Any]] = []
    for split, key in (("train", "train_samples"), ("val", "val_samples")):
        for s in meta[key]:
            r = dict(s)
            r["_split"] = split
            r["_multi"] = is_multiheading(s["image_path"])
            rows.append(r)

    print(f"Загружено {len(rows)} кропов. Считаю метрики качества...")
    stds, sharps, hashes = [], [], []
    for s in rows:
        img = Image.open(data_dir / s["image_path"]).convert("RGB")
        std, sharp, h = quality_metrics(img)
        stds.append(std)
        sharps.append(sharp)
        hashes.append(h)
    stds = np.array(stds)
    sharps = np.array(sharps)
    sharp_thr = float(np.percentile(sharps, args.sharp_pct))

    print("Zero-shot CLIP-скоринг классов...")
    probs = clip_scores(rows, data_dir, class_names, device)
    clip_pred = probs.argmax(axis=1)
    label_prob = np.array([probs[i, int(s["class_id"])] for i, s in enumerate(rows)])

    # Пометки качества
    drops = Counter()
    for i, s in enumerate(rows):
        bad = []
        if stds[i] < args.std_min:
            bad.append("empty")
        if sharps[i] < sharp_thr:
            bad.append("blurry")
        s["_quality_bad"] = bad
        s["_label_prob"] = float(label_prob[i])
        s["_clip_pred"] = class_names[int(clip_pred[i])]
        s["_suspect"] = bool(clip_pred[i] != int(s["class_id"]) and label_prob[i] < args.suspect_prob)

    # Дедуп внутри объекта (по aHash), оставляем самый резкий
    by_obj: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(rows):
        by_obj[s["object_id"]].append(i)
    dup_dropped = set()
    for obj, idxs in by_obj.items():
        for a_pos in range(len(idxs)):
            ia = idxs[a_pos]
            if ia in dup_dropped:
                continue
            for b_pos in range(a_pos + 1, len(idxs)):
                ib = idxs[b_pos]
                if ib in dup_dropped:
                    continue
                if hamming(hashes[ia], hashes[ib]) <= args.dup_hamming:
                    worse = ia if sharps[ia] < sharps[ib] else ib
                    dup_dropped.add(worse)
    for i in dup_dropped:
        drops["duplicate"] += 1

    # Heading-дизамбигуация: лучший heading на мультихединг-объект
    keep_multi = set()
    multi_objs: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(rows):
        if s["_multi"]:
            multi_objs[s["object_id"]].append(i)
    for obj, idxs in multi_objs.items():
        cand = [i for i in idxs if i not in dup_dropped and not rows[i]["_quality_bad"]]
        if not cand:
            cand = [i for i in idxs if i not in dup_dropped] or idxs
        best = max(cand, key=lambda i: label_prob[i])
        keep_multi.add(best)
        drops["heading_pruned"] += len(idxs) - 1

    # Финальная сборка
    train_clean, val_clean = [], []
    dropped_records = []
    for i, s in enumerate(rows):
        reason = None
        if i in dup_dropped:
            reason = "duplicate"
        elif s["_quality_bad"]:
            reason = "+".join(s["_quality_bad"])
        elif s["_multi"] and i not in keep_multi:
            reason = "heading_pruned"
        elif s["_split"] == "val" and s["_multi"]:
            reason = "val_multiheading_excluded"  # честный val = только known-bearing

        clean = {k: v for k, v in s.items() if not k.startswith("_")}
        if reason:
            dropped_records.append({"image_path": s["image_path"], "reason": reason,
                                    "label_prob": round(float(label_prob[i]), 3)})
            continue
        if s["_split"] == "val":
            val_clean.append(clean)
        else:
            train_clean.append(clean)

    def stats(rws):
        c = Counter(r["class_name"] for r in rws)
        return {k: c.get(k, 0) for k in class_names}

    train_objs = {r["object_id"] for r in train_clean}
    val_objs = {r["object_id"] for r in val_clean}
    clean_meta = dict(meta)
    clean_meta["train_samples"] = train_clean
    clean_meta["val_samples"] = val_clean
    clean_meta["train_statistics"] = stats(train_clean)
    clean_meta["val_statistics"] = stats(val_clean)
    clean_meta["dataset_info"] = {
        **meta.get("dataset_info", {}),
        "total_train": len(train_clean),
        "total_val": len(val_clean),
        "total": len(train_clean) + len(val_clean),
        "cleaned": True,
        "cleaning": {
            "val_policy": "single_heading_only (known bearing)",
            "train_policy": "single_heading + best_heading_per_object",
            "train_val_object_overlap": len(train_objs & val_objs),
        },
    }
    Path(out_meta).write_text(json.dumps(clean_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    suspects = [
        {"image_path": s["image_path"], "label": s["class_name"],
         "clip_pred": s["_clip_pred"], "label_prob": round(float(label_prob[i]), 3)}
        for i, s in enumerate(rows) if s["_suspect"]
    ]
    dropped_by_reason = Counter(d["reason"] for d in dropped_records)
    report = {
        "input": {"total": len(rows),
                  "train": len(meta["train_samples"]), "val": len(meta["val_samples"])},
        "output": {"train": len(train_clean), "val": len(val_clean),
                   "train_stats": clean_meta["train_statistics"],
                   "val_stats": clean_meta["val_statistics"]},
        "dropped_counts": dict(dropped_by_reason),
        "thresholds": {"std_min": args.std_min, "sharp_thr": round(sharp_thr, 2),
                       "sharp_pct": args.sharp_pct, "dup_hamming": args.dup_hamming},
        "suspect_count": len(suspects),
        "suspects_sample": suspects[:40],
        "dropped_sample": dropped_records[:40],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Чистка завершена ===")
    print(f"train: {len(meta['train_samples'])} -> {len(train_clean)}")
    print(f"val:   {len(meta['val_samples'])} -> {len(val_clean)}  (только known-bearing)")
    print(f"dropped: {dict(drops)}")
    print(f"suspects (аудит): {len(suspects)}")
    print(f"train_stats: {clean_meta['train_statistics']}")
    print(f"val_stats:   {clean_meta['val_statistics']}")
    print(f"Saved: {out_meta}\n       {report_path}")


if __name__ == "__main__":
    main()
