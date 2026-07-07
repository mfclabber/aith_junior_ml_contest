#!/usr/bin/env python3
"""Дообучение PaliGemma 3B (LoRA) на VLM-датасете городских участков.

Учит модель по кропу выдавать detection-ответ:
    "detect urban parcel state" -> "<loc..><loc..><loc..><loc..> <class_name>"
то есть одновременно класс UTT и evidence-bbox.

Датасет: data/vlm/{train,val}.jsonl (см. build_vlm_dataset.py).
Веса LoRA сохраняются в checkpoints/paligemma_lora.

Пример:
  CUDA_VISIBLE_DEVICES=3 python3 scripts/vlm/train_paligemma_lora.py \
      --epochs 3 --batch-size 4 --model google/paligemma-3b-pt-224
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model

LOC_RE = re.compile(r"<loc(\d{4})>")


class JsonlVLM(Dataset):
    def __init__(self, path: Path):
        self.rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def make_collate(processor: PaliGemmaProcessor):
    def collate(batch: list[dict[str, Any]]):
        images = [Image.open(b["image"]).convert("RGB") for b in batch]
        prefixes = [b["prefix"] for b in batch]
        suffixes = [b["suffix"] for b in batch]
        enc = processor(
            text=prefixes,
            suffix=suffixes,
            images=images,
            return_tensors="pt",
            padding="longest",
        )
        return enc

    return collate


def parse_boxes(text: str) -> list[tuple[int, int, int, int]]:
    nums = [int(n) for n in LOC_RE.findall(text)]
    boxes = []
    for i in range(0, len(nums) - 3, 4):
        y0, x0, y1, x1 = nums[i : i + 4]
        boxes.append((x0, y0, x1, y1))
    return boxes


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    ua = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    ub = max(0, bx1 - bx0) * max(0, by1 - by0)
    denom = ua + ub - inter
    return inter / denom if denom > 0 else 0.0


def class_from_text(text: str, class_names: list[str]) -> str | None:
    tail = LOC_RE.sub("", text).strip().lower()
    for c in class_names:
        if c.lower() in tail:
            return c
    return None


@torch.no_grad()
def evaluate(model, processor, val_rows, class_names, device, max_items: int, gen_len: int,
             per_class: bool = False):
    model.eval()
    rows = val_rows[:max_items] if max_items else val_rows
    correct, ious, n = 0, [], 0
    tot = {c: 0 for c in class_names}
    tp = {c: 0 for c in class_names}
    pred_pos = {c: 0 for c in class_names}
    for b in rows:
        img = Image.open(b["image"]).convert("RGB")
        enc = processor(text=b["prefix"], images=img, return_tensors="pt").to(device)
        out = model.generate(**enc, max_new_tokens=gen_len, do_sample=False)
        gen = processor.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=False)[0]
        pred_c = class_from_text(gen, class_names)
        gt = b["class_name"]
        tot[gt] += 1
        if pred_c in pred_pos:
            pred_pos[pred_c] += 1
        if pred_c == gt:
            correct += 1
            tp[gt] += 1
        pb = parse_boxes(gen)
        gb = parse_boxes(b["suffix"])
        if pb and gb:
            ious.append(iou(pb[0], gb[0]))
        n += 1
    acc = correct / max(n, 1)
    miou = sum(ious) / max(len(ious), 1)
    model.train()
    res = {"n": n, "class_acc": acc, "evidence_iou": miou}
    if per_class:
        f1 = {}
        for c in class_names:
            prec = tp[c] / pred_pos[c] if pred_pos[c] else 0.0
            rec = tp[c] / tot[c] if tot[c] else 0.0
            f1[c] = round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0
        res["per_class_f1"] = f1
        res["macro_f1"] = round(sum(f1.values()) / len(class_names), 4)
        res["support"] = tot
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA fine-tune PaliGemma on urban VLM dataset.")
    ap.add_argument("--data-dir", type=Path, default=Path("data/vlm"))
    ap.add_argument("--out-dir", type=Path, default=Path("checkpoints/paligemma_lora"))
    ap.add_argument("--model", type=str, default="google/paligemma-3b-pt-224")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--eval-items", type=int, default=120)
    ap.add_argument("--gen-len", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train", type=int, default=0, help="0 = все")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = Path(__file__).resolve().parents[2]
    data_dir = args.data_dir if args.data_dir.is_absolute() else root / args.data_dir
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    class_names = list(meta["class_mapping"].keys())

    print(f"Loading {args.model} ...")
    processor = PaliGemmaProcessor.from_pretrained(args.model)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    )

    # Заморозить vision tower и мультимодальный проектор, тюнить только язык через LoRA
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)

    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.to(device)
    model.print_trainable_parameters()

    train_ds = JsonlVLM(data_dir / "train.jsonl")
    if args.max_train:
        train_ds.rows = train_ds.rows[: args.max_train]
    val_rows = JsonlVLM(data_dir / "val.jsonl").rows
    collate = make_collate(processor)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, num_workers=4, pin_memory=True)

    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    sched = get_cosine_schedule_with_warmup(
        opt, int(total_steps * args.warmup_ratio), total_steps
    )

    print(f"train={len(train_ds)} val={len(val_rows)} steps/epoch={steps_per_epoch} total={total_steps}")
    model.train()
    gstep = 0
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        opt.zero_grad(set_to_none=True)
        for it, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            running += out.loss.item()
            if (it + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0
                )
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                gstep += 1
                if gstep % 20 == 0:
                    print(f"epoch {epoch} step {gstep}/{total_steps} "
                          f"loss={running / (it + 1):.4f} lr={sched.get_last_lr()[0]:.2e}")
        metrics = evaluate(model, processor, val_rows, class_names, device,
                           args.eval_items, args.gen_len)
        print(f"[epoch {epoch}] train_loss={running / len(loader):.4f} "
              f"val_class_acc={metrics['class_acc']:.4f} "
              f"val_evidence_iou={metrics['evidence_iou']:.4f} (n={metrics['n']})")

    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    final = evaluate(model, processor, val_rows, class_names, device, 0, args.gen_len,
                     per_class=True)
    (out_dir / "metrics.json").write_text(
        json.dumps({"model": args.model, "final_val": final}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Saved LoRA to", out_dir, "| final val:", final)


if __name__ == "__main__":
    main()
