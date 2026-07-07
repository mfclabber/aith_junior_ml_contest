#!/usr/bin/env python3
"""Линейный пробинг на замороженных CLIP-фичах (устойчив к переобучению на малых данных).

Извлекает эмбеддинги CLIP (кэширует в .npy), обучает регуляризованный линейный
классификатор с перебором weight-decay и выбором лучшего по val. Считает accuracy,
macro-F1 и confusion matrix. Опция слияния construction-классов (6 -> 5 классов).

Пример:
  CUDA_VISIBLE_DEVICES=3 ./venv/bin/python scripts/training/linear_probe_clip.py \
      --metadata data/ml_perspective/metadata_perspective_clean.json \
      --model ViT-B-32 --pretrained laion2b_s34b_b79k
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _maybe_bottom_crop(img: Image.Image, bottom: float) -> Image.Image:
    """Оставить нижнюю долю кадра (убрать небо сверху)."""
    if bottom >= 0.999:
        return img
    w, h = img.size
    top = int(h * (1.0 - bottom))
    return img.crop((0, top, w, h))


def load_features(rows, data_dir, model, preprocess, device, cache: Path | None,
                  bs=64, bottom_crop=1.0):
    if cache and cache.is_file():
        d = np.load(cache)
        w = d["w"] if "w" in d.files else None
        return d["feat"], d["y"], w
    feats, ys, ws = [], [], []
    buf, buf_w = [], []
    with torch.no_grad():
        for s in rows:
            img = Image.open(data_dir / s["image_path"]).convert("RGB")
            img = _maybe_bottom_crop(img, bottom_crop)
            buf.append(preprocess(img))
            buf_w.append(float(s.get("sample_weight", 1.0)))
            ys.append(int(s["class_id"]))
            if len(buf) >= bs:
                x = torch.stack(buf).to(device)
                f = model.encode_image(x)
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.cpu().numpy())
                ws.extend(buf_w)
                buf, buf_w = [], []
        if buf:
            x = torch.stack(buf).to(device)
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
            ws.extend(buf_w)
    feat = np.concatenate(feats, axis=0).astype(np.float32)
    y = np.array(ys, dtype=np.int64)
    w = np.array(ws, dtype=np.float32) if ws else None
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        if w is not None:
            np.savez(cache, feat=feat, y=y, w=w)
        else:
            np.savez(cache, feat=feat, y=y)
    return feat, y, w


def train_probe(Xtr, ytr, Xva, yva, num_classes, wd, device, epochs=300,
                select_by="acc", wtr=None, val_cap: int = 35,
                init_state: dict | None = None, lr: float = 1e-2,
                tracker=None, log_tag: str = ""):
    Xtr = torch.tensor(Xtr, device=device)
    ytr = torch.tensor(ytr, device=device)
    Xva = torch.tensor(Xva, device=device)
    dim = Xtr.shape[1]
    head = torch.nn.Linear(dim, num_classes).to(device)
    if init_state is not None:
        head.load_state_dict(init_state, strict=True)
    cnt = np.bincount(ytr.cpu().numpy(), minlength=num_classes).astype(np.float64)
    if wtr is not None:
        sw = np.asarray(wtr, dtype=np.float64)
        for i in range(num_classes):
            m = ytr.cpu().numpy() == i
            if m.any():
                cnt[i] = float(sw[m].sum())
    cw = torch.tensor((cnt.sum() / (num_classes * np.maximum(cnt, 1))),
                      dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    lossf = torch.nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05, reduction="none")
    wtr_t = None
    if wtr is not None:
        wtr_t = torch.tensor(wtr, dtype=torch.float32, device=device)
    best_score, best_state = -1.0, None
    for ep in range(epochs):
        head.train()
        opt.zero_grad()
        logits = head(Xtr)
        per = lossf(logits, ytr)
        if wtr_t is not None:
            loss = (per * wtr_t).sum() / wtr_t.sum().clamp_min(1e-6)
        else:
            loss = per.mean()
        loss.backward()
        opt.step()
        head.eval()
        with torch.no_grad():
            pred = head(Xva).argmax(1).cpu().numpy()
        acc = float((pred == yva).mean())
        f1, _ = macro_f1(yva, pred, num_classes)
        bf1 = balanced_macro_f1(yva, pred, num_classes, val_cap)
        if select_by == "balanced_f1":
            score = bf1
        elif select_by == "macro_f1":
            score = f1
        else:
            score = acc
        if tracker is not None:
            p = f"{log_tag}/" if log_tag else ""
            tracker.log_metrics(
                {f"{p}train_loss": float(loss.item()), f"{p}val_acc": acc,
                 f"{p}val_macro_f1": f1, f"{p}val_balanced_f1": bf1},
                step=ep,
            )
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    if best_state is not None:
        head.load_state_dict(best_state)
    with torch.no_grad():
        pred = head(Xva).argmax(1).cpu().numpy()
    acc = float((pred == yva).mean())
    return head, acc, pred


def macro_f1(y_true, y_pred, num_classes):
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)), f1s


def balanced_macro_f1(y_true, y_pred, num_classes, cap_per_class: int = 35):
    """macro-F1 на сбалансированном подмножестве val (cap majority)."""
    idx = []
    for c in range(num_classes):
        cls_idx = np.where(y_true == c)[0]
        if len(cls_idx) == 0:
            continue
        if len(cls_idx) > cap_per_class:
            cls_idx = np.random.default_rng(42).choice(cls_idx, cap_per_class, replace=False)
        idx.extend(cls_idx.tolist())
    if not idx:
        return macro_f1(y_true, y_pred, num_classes)[0]
    idx = np.array(idx)
    return macro_f1(y_true[idx], y_pred[idx], num_classes)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/ml_perspective"))
    ap.add_argument("--metadata", type=Path,
                    default=Path("data/ml_perspective/metadata_perspective_clean.json"))
    ap.add_argument("--model", type=str, default="ViT-B-32")
    ap.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    ap.add_argument("--merge-construction", action="store_true",
                    help="frozen+active construction -> один класс construction (6->5)")
    ap.add_argument("--group", choices=("6", "5", "3"), default=None,
                    help="6=исходные; 5=слить construction; 3=natural/construction/built_up")
    ap.add_argument("--bottom-crop", type=float, default=1.0,
                    help="оставить нижнюю долю кадра (0.55 = убрать верхние 45%% неба)")
    ap.add_argument("--save", type=Path, default=None,
                    help="сохранить обученную голову + конфиг в .pt (готовый классификатор)")
    ap.add_argument("--select-by", choices=("acc", "macro_f1", "balanced_f1"), default="acc",
                    help="критерий выбора лучшей головы на val (balanced_f1 = macro на cap val)")
    ap.add_argument("--val-cap", type=int, default=35, help="cap per class для balanced_f1")
    ap.add_argument("--cache-tag", type=str, default="")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--init-head", type=Path, default=None,
                    help="чекпойнт .pt — warm-start головы (fine-tune)")
    ap.add_argument("--finetune-lr", type=float, default=3e-3,
                    help="lr при --init-head (иначе 1e-2)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--tracking", type=str, default="",
                    help="бэкенды трекинга через запятую: tensorboard,wandb,mlflow")
    ap.add_argument("--experiment", type=str, default="",
                    help="имя запуска для трекера (по умолчанию из cache-tag)")
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"),
                    help="каталог логов TensorBoard/MLflow")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    data_dir = args.data_dir if args.data_dir.is_absolute() else root / args.data_dir
    meta_path = args.metadata if args.metadata.is_absolute() else root / args.metadata
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    class_mapping = dict(meta["class_mapping"])
    class_names = [k for k, _ in sorted(class_mapping.items(), key=lambda x: x[1])]

    group = args.group or ("5" if args.merge_construction else "6")
    remap = None
    if group != "6":
        if group == "5":
            def to_group(c):
                return "construction" if c in ("frozen_construction", "active_construction") else c
            merged_names = ["natural_areas", "low_density_degraded",
                            "underused_infrastructure", "active_urban", "construction"]
        else:  # group == "3": визуально разделимые макро-классы
            builtup = {"low_density_degraded", "underused_infrastructure", "active_urban"}
            constr = {"frozen_construction", "active_construction"}

            def to_group(c):
                if c in constr:
                    return "construction"
                if c in builtup:
                    return "built_up"
                return "natural_areas"
            merged_names = ["natural_areas", "built_up", "construction"]
        new_id = {c: i for i, c in enumerate(merged_names)}
        if set(class_names) != set(merged_names):
            remap = {class_mapping[c]: new_id[to_group(c)] for c in class_names}
            class_names = merged_names
    num_classes = len(class_names)

    init_state = None
    if args.init_head is not None:
        init_p = args.init_head if args.init_head.is_absolute() else root / args.init_head
        ckpt = torch.load(init_p, map_location="cpu", weights_only=False)
        init_state = ckpt["head_state_dict"]
        if ckpt.get("model"):
            args.model = ckpt["model"]
        if ckpt.get("pretrained"):
            args.pretrained = ckpt["pretrained"]
        if ckpt.get("bottom_crop") is not None:
            args.bottom_crop = float(ckpt["bottom_crop"])
        print(f"warm-start head from {init_p.name} ({args.model})")

    import open_clip
    device = torch.device(args.device)
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device)
    model.eval()

    tag = args.cache_tag or f"{meta_path.stem}_{args.model}_{args.pretrained}".replace("/", "_")
    if args.bottom_crop < 0.999:
        tag += f"_bc{int(args.bottom_crop*100)}"
    cdir = root / "data" / "ml_perspective" / "feat_cache"
    Xtr, ytr, wtr = load_features(meta["train_samples"], data_dir, model, preprocess, device,
                                  cdir / f"train_{tag}.npz", bottom_crop=args.bottom_crop)
    Xva, yva, _ = load_features(meta["val_samples"], data_dir, model, preprocess, device,
                                cdir / f"val_{tag}.npz", bottom_crop=args.bottom_crop)
    if remap is not None:
        ytr = np.array([remap[int(v)] for v in ytr], dtype=np.int64)
        yva = np.array([remap[int(v)] for v in yva], dtype=np.int64)

    tracker = None
    if args.tracking:
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from scripts.training.tracking import ExperimentTracker
        run_name = args.experiment or tag
        tracker = ExperimentTracker(
            backends=[b.strip() for b in args.tracking.split(",") if b.strip()],
            run_name=run_name,
            log_dir=str(args.runs_dir if args.runs_dir.is_absolute() else root / args.runs_dir),
            config={
                "model": args.model, "pretrained": args.pretrained,
                "group": group, "select_by": args.select_by,
                "bottom_crop": args.bottom_crop, "epochs": args.epochs,
                "num_classes": num_classes, "n_train": int(Xtr.shape[0]),
                "n_val": int(Xva.shape[0]), "warm_start": init_state is not None,
            },
        )

    print(f"features: train {Xtr.shape} val {Xva.shape}  classes={num_classes}")
    best = (-1.0, None, None, None, None)
    best_head = None
    lr = args.finetune_lr if init_state is not None else 1e-2
    wd_grid = (0.01, 0.05, 0.1, 0.3, 1.0) if init_state else (0.001, 0.01, 0.05, 0.1, 0.3, 1.0, 3.0)
    for gi, wd in enumerate(wd_grid):
        head, acc, pred = train_probe(
            Xtr, ytr, Xva, yva, num_classes, wd, device,
            select_by=args.select_by, val_cap=args.val_cap, wtr=wtr,
            init_state=init_state, lr=lr, epochs=args.epochs,
            tracker=tracker, log_tag=f"wd_{wd}")
        f1, _ = macro_f1(yva, pred, num_classes)
        bf1 = balanced_macro_f1(yva, pred, num_classes, args.val_cap)
        score = bf1 if args.select_by == "balanced_f1" else (f1 if args.select_by == "macro_f1" else acc)
        print(f"  wd={wd:<5} val_acc={acc:.4f} macro_f1={f1:.4f} balanced_f1={bf1:.4f}")
        if tracker is not None:
            tracker.log_metrics(
                {"grid/val_acc": acc, "grid/macro_f1": f1, "grid/balanced_f1": bf1}, step=gi)
        if score > best[0]:
            best = (score, wd, pred, acc, f1)
            best_head = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    score, wd, pred, acc, f1 = best
    if best_head is None:
        print("ERROR: probe did not converge", file=sys.stderr)
        raise SystemExit(1)
    print(f"\nBEST linear probe: acc={acc:.4f} macro_f1={f1:.4f} (wd={wd})  n_val={len(yva)}")
    _, per_f1 = macro_f1(yva, pred, num_classes)
    bf1 = balanced_macro_f1(yva, pred, num_classes, args.val_cap)
    print("per-class F1: " + ", ".join(f"{n}={v:.3f}" for n, v in zip(class_names, per_f1)))
    print(f"balanced macro-F1 (cap={args.val_cap}): {bf1:.4f}")

    if args.save and best_head is not None:
        save_p = args.save if args.save.is_absolute() else root / args.save
        save_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "head_state_dict": best_head,
            "class_mapping": {c: i for i, c in enumerate(class_names)},
            "model": args.model, "pretrained": args.pretrained,
            "bottom_crop": args.bottom_crop, "weight_decay": wd,
            "val_acc": acc, "val_macro_f1": f1, "val_balanced_f1": bf1,
            "normalize_features": True,
        }, save_p)
        print(f"Saved classifier -> {save_p}")
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(yva, pred):
        cm[t, p] += 1
    print("confusion (rows=true):")
    print("            " + "".join(f"{n[:11]:>12}" for n in class_names))
    for i, n in enumerate(class_names):
        print(f"{n[:11]:>11} " + "".join(f"{cm[i,j]:>12}" for j in range(num_classes)))

    if tracker is not None:
        tracker.log_summary({
            "best_val_acc": acc, "best_macro_f1": f1, "best_balanced_f1": bf1,
            "best_weight_decay": wd,
        })
        tracker.log_params({"best_weight_decay": wd})
        if args.save:
            tracker.log_artifact(save_p)
        tracker.close()


if __name__ == "__main__":
    main()
