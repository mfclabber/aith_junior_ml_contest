#!/usr/bin/env python3
"""Автономный Improvement Agent — выполняет фазы без вопросов пользователю.

Принимает решения по метрикам и состоянию файлов, пишет лог в results/AGENT_RUN.md.

  ./venv/bin/python scripts/agent/run_autonomous.py
  ./venv/bin/python scripts/agent/run_autonomous.py --from-phase 3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / "venv" / "bin" / "python"
STATE_PATH = ROOT / "results" / "agent_state.json"
LOG_PATH = ROOT / "results" / "agent_autonomous.log"
RUN_MD = ROOT / "results" / "AGENT_RUN.md"

GPU = os.environ.get("CUDA_VISIBLE_DEVICES", "3")


def log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(cmd: list[str], *, env: dict | None = None, timeout: int | None = None) -> int:
    log("$ " + " ".join(cmd))
    e = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU, **(env or {})}
    r = subprocess.run(cmd, cwd=ROOT, env=e, timeout=timeout)
    return r.returncode


def load_state() -> dict:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"completed_phases": [], "decisions": []}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def append_run_md(section: str) -> None:
    with RUN_MD.open("a", encoding="utf-8") as fh:
        fh.write(section)


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())


def phase_done(state: dict, n: int) -> bool:
    return n in state.get("completed_phases", [])


def mark_done(state: dict, n: int, note: str) -> None:
    if n not in state["completed_phases"]:
        state["completed_phases"].append(n)
    state["decisions"].append({"phase": n, "note": note, "at": datetime.now(timezone.utc).isoformat()})
    save_state(state)


def decide_phase0(state: dict) -> None:
    if phase_done(state, 0):
        log("phase 0 skip (done)")
        return
    run([str(PY), "scripts/data_quality/audit_dataset.py"])
    run([str(PY), "scripts/data_quality/clean_dataset.py"])
    mark_done(state, 0, "audit + clean")
    append_run_md("\n### auto phase 0\n- audit + clean_dataset\n")


def decide_phase1(state: dict) -> None:
    if phase_done(state, 1):
        log("phase 1 skip (done)")
        return
    labels = ROOT / "data" / "vlm_relabel" / "labels.jsonl"
    need = 1183
    have = count_jsonl(labels)
    if have < need:
        log(f"phase 1: relabel {have}/{need}")
        run(
            [str(PY), "scripts/data_quality/vlm_relabel.py", "--taxonomy", "6", "--out", str(labels)],
            timeout=7200,
        )
    run([
        str(PY), "scripts/data_quality/build_dataset_from_vlm.py",
        "--taxonomy", "6", "--labels", str(labels),
    ])
    mark_done(state, 1, f"vlm relabel + vlm6 metadata ({count_jsonl(labels)} labels)")
    append_run_md(f"\n### auto phase 1\n- labels.jsonl: {count_jsonl(labels)}\n")


def decide_phase2(state: dict) -> None:
    if phase_done(state, 2):
        log("phase 2 skip (done)")
        return
    meta = ROOT / "data" / "ml_perspective" / "metadata_perspective_vlm6.json"
    probe = ROOT / "checkpoints" / "clip_probe_vlm6.pt"
    if not meta.is_file():
        decide_phase1(state)

    if not probe.is_file():
        run([
            str(PY), "scripts/training/linear_probe_clip.py",
            "--metadata", str(meta),
            "--model", "ViT-L-14", "--pretrained", "laion2b_s32b_b82k",
            "--group", "6", "--save", str(probe),
        ])

    # PaliGemma: обучаем если чекпойнта нет
    pali = ROOT / "checkpoints" / "paligemma_lora_clean6" / "adapter_config.json"
    vlm_data = ROOT / "data" / "vlm_clean6" / "train.jsonl"
    if not vlm_data.is_file():
        run([
            str(PY), "scripts/vlm/build_vlm_dataset.py",
            "--meta", "data/ml_perspective/metadata_perspective_clean.json",
            "--out-dir", "data/vlm_clean6",
        ])
    if not pali.is_file():
        log("phase 2: train PaliGemma (long)")
        run([
            str(PY), "scripts/vlm/train_paligemma_lora.py",
            "--data-dir", "data/vlm_clean6",
            "--out-dir", "checkpoints/paligemma_lora_clean6",
            "--epochs", "8", "--batch-size", "8",
        ], timeout=14400)

    mark_done(state, 2, "probe + paligemma checkpoints")
    append_run_md("\n### auto phase 2\n- clip_probe_vlm6 + paligemma_lora_clean6\n")


def decide_phase3(state: dict) -> None:
    if phase_done(state, 3):
        log("phase 3 skip (done)")
        return
    rc = run([str(PY), "scripts/agent/smoke_test_api.py"])
    if rc != 0:
        log(f"phase 3 smoke failed rc={rc}, continuing anyway")
    mark_done(state, 3, f"smoke_test rc={rc}")
    append_run_md(f"\n### auto phase 3\n- smoke_test_api rc={rc}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-phase", type=int, default=0, choices=range(0, 4))
    ap.add_argument("--reset", action="store_true", help="сбросить agent_state.json")
    args = ap.parse_args()

    if args.reset and STATE_PATH.is_file():
        STATE_PATH.unlink()

    state = load_state()
    phases = [
        (0, decide_phase0),
        (1, decide_phase1),
        (2, decide_phase2),
        (3, decide_phase3),
    ]
    for n, fn in phases:
        if n >= args.from_phase:
            log(f"=== phase {n} ===")
            try:
                fn(state)
            except Exception as exc:
                log(f"phase {n} error: {exc}")
                append_run_md(f"\n### auto phase {n} ERROR\n- {exc}\n")
                return 1

    log("autonomous run complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
