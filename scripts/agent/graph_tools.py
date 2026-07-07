"""Инструменты для LangGraph-агента: eval, обучение, данные."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / "venv" / "bin" / "python"
GPU = os.environ.get("CUDA_VISIBLE_DEVICES", "3")

BEST_RE = re.compile(r"BEST linear probe: acc=([\d.]+) macro_f1=([\d.]+)")
BALANCED_RE = re.compile(r"balanced macro-F1 \(cap=\d+\): ([\d.]+)")


def _run(cmd: list[str], timeout: int | None = None) -> tuple[int, str]:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU}
    r = subprocess.run(
        cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout
    )
    out = (r.stdout or "") + (r.stderr or "")
    return r.returncode, out


def eval_probe(
    metadata: str,
    *,
    group: str = "6",
    model: str = "ViT-L-14",
    pretrained: str = "laion2b_s32b_b82k",
    save: str | None = None,
    select_by: str = "macro_f1",
) -> dict:
    cmd = [
        str(PY), "scripts/training/linear_probe_clip.py",
        "--metadata", metadata,
        "--model", model, "--pretrained", pretrained,
        "--group", group, "--select-by", select_by,
    ]
    if save:
        cmd += ["--save", save]
    rc, out = _run(cmd, timeout=600)
    m = BEST_RE.search(out)
    bm = BALANCED_RE.search(out)
    if not m:
        return {"ok": False, "rc": rc, "accuracy": 0.0, "macro_f1": 0.0, "balanced_f1": 0.0}
    return {
        "ok": rc == 0,
        "rc": rc,
        "accuracy": float(m.group(1)),
        "macro_f1": float(m.group(2)),
        "balanced_f1": float(bm.group(1)) if bm else 0.0,
        "metadata": metadata,
        "group": group,
    }


def clean_data() -> tuple[bool, str]:
    rc1, _ = _run([str(PY), "scripts/data_quality/audit_dataset.py"])
    rc2, out = _run([str(PY), "scripts/data_quality/clean_dataset.py"], timeout=600)
    return rc2 == 0, out[-300:]


def vlm_relabel(taxonomy: str) -> tuple[bool, str]:
    out_path = "data/vlm_relabel/labels3.jsonl" if taxonomy == "3" else "data/vlm_relabel/labels.jsonl"
    rc, out = _run([
        str(PY), "scripts/data_quality/vlm_relabel.py",
        "--taxonomy", taxonomy, "--out", out_path,
    ], timeout=7200)
    return rc == 0, out[-300:]


def build_dataset(taxonomy: str) -> tuple[bool, str]:
    labels = "data/vlm_relabel/labels3.jsonl" if taxonomy == "3" else "data/vlm_relabel/labels.jsonl"
    rc, out = _run([
        str(PY), "scripts/data_quality/build_dataset_from_vlm.py",
        "--taxonomy", taxonomy, "--labels", labels,
    ])
    return rc == 0, out[-300:]


def balance_vlm6_oss() -> tuple[bool, str]:
    """Лучшая балансировка: min 80, cap urban 120."""
    rc, out = _run([
        str(PY), "scripts/data_quality/balance_vlm_dataset.py",
        "--in-meta", "data/ml_perspective/metadata_perspective_vlm6.json",
        "--out", "data/ml_perspective/metadata_perspective_vlm6_oss.json",
        "--min-per-class", "80", "--max-majority", "120", "--oversample-factor", "5",
    ])
    return rc == 0, out[-300:]


def train_probe_vlm6_oss() -> tuple[bool, str]:
    meta = "data/ml_perspective/metadata_perspective_vlm6_oss.json"
    if not file_exists(meta):
        return False, "vlm6_oss metadata missing"
    # сброс кэша фичей для свежего обучения
    cache = ROOT / "data/ml_perspective/feat_cache"
    if cache.is_dir():
        for p in cache.glob("*vlm6_oss*"):
            p.unlink(missing_ok=True)
    r = eval_probe(
        meta, group="6", save="checkpoints/clip_probe_vlm6_oss.pt", select_by="macro_f1",
    )
    return r["ok"], f"acc={r['accuracy']:.4f} f1={r['macro_f1']:.4f} bal={r['balanced_f1']:.4f}"


def balance_vlm6(max_majority: int = 200) -> tuple[bool, str]:
    """Балансировка: cap active_urban, сохранить все minority."""
    meta_path = ROOT / "data/ml_perspective/metadata_perspective_vlm6.json"
    out_path = ROOT / "data/ml_perspective/metadata_perspective_vlm6_balanced.json"
    if not meta_path.is_file():
        return False, "vlm6 metadata missing"
    import random
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    random.seed(42)

    def balance_rows(rows: list[dict]) -> list[dict]:
        by_cls: dict[str, list[dict]] = {}
        for r in rows:
            by_cls.setdefault(r["class_name"], []).append(r)
        out: list[dict] = []
        for cls, items in by_cls.items():
            if cls == "active_urban" and len(items) > max_majority:
                random.shuffle(items)
                items = items[:max_majority]
            out.extend(items)
        random.shuffle(out)
        return out

    train = balance_rows(meta["train_samples"])
    val = meta["val_samples"]  # val не трогаем
    from collections import Counter
    meta["train_samples"] = train
    meta["dataset_info"]["total_train"] = len(train)
    meta["dataset_info"]["balanced"] = True
    meta["dataset_info"]["max_majority"] = max_majority
    meta["train_statistics"] = dict(Counter(r["class_name"] for r in train))
    out_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return True, f"balanced train={len(train)} stats={meta['train_statistics']}"


def train_paligemma() -> tuple[bool, str]:
    pali = ROOT / "checkpoints/paligemma_lora_clean6/adapter_config.json"
    if pali.is_file():
        return True, "paligemma checkpoint exists, skip"
    vlm_data = ROOT / "data/vlm_clean6/train.jsonl"
    if not vlm_data.is_file():
        _run([
            str(PY), "scripts/vlm/build_vlm_dataset.py",
            "--meta", "data/ml_perspective/metadata_perspective_clean.json",
            "--out-dir", "data/vlm_clean6",
        ], timeout=3600)
    rc, out = _run([
        str(PY), "scripts/vlm/train_paligemma_lora.py",
        "--data-dir", "data/vlm_clean6",
        "--out-dir", "checkpoints/paligemma_lora_clean6",
        "--epochs", "8", "--batch-size", "8",
    ], timeout=14400)
    return rc == 0, out[-400:]


def smoke_api() -> tuple[bool, str]:
    rc, out = _run([str(PY), "scripts/agent/smoke_test_api.py"], timeout=120)
    return rc == 0, out[-200:]


def count_jsonl(rel: str) -> int:
    p = ROOT / rel
    if not p.is_file():
        return 0
    return sum(1 for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip())


def file_exists(rel: str) -> bool:
    return (ROOT / rel).is_file()
