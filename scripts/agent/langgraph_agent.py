#!/usr/bin/env python3
"""LangGraph Metric Agent — достижение целевых метрик на 6 классах UTT.

Инструкция: AGENT_INSTRUCTIONS.md

  bash scripts/agent/run_metric_agent.sh
  CLASSIFIER_TAXONOMY=6 ./venv/bin/python scripts/agent/langgraph_agent.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langgraph.graph import END, StateGraph

from scripts.agent.graph_state import AgentState, MetricRecord
from scripts.agent import graph_tools as tools

LOG_PATH = ROOT / "results" / "langgraph_agent.log"
STATE_PATH = ROOT / "results" / "langgraph_state.json"
INSTRUCTIONS = ROOT / "AGENT_INSTRUCTIONS.md"


def log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _targets_met(state: AgentState, acc: float, f1: float, bf1: float) -> bool:
    return (
        acc >= state["target_accuracy"]
        and f1 >= state["target_macro_f1"]
        and bf1 >= state["target_balanced_f1"]
    )


def evaluate(state: AgentState) -> dict:
    """Замер 6-классовых конфигураций."""
    candidates = [
        ("vlm6_oss", "data/ml_perspective/metadata_perspective_vlm6_oss.json",
         "checkpoints/clip_probe_vlm6_oss.pt"),
        ("vlm6", "data/ml_perspective/metadata_perspective_vlm6.json", None),
        ("osm6", "data/ml_perspective/metadata_perspective_clean.json", None),
    ]
    records: list[MetricRecord] = []
    best_acc = state.get("best_accuracy", 0.0)
    best_f1 = state.get("best_macro_f1", 0.0)
    best_bf1 = state.get("best_balanced_f1", 0.0)
    best_task = state.get("best_task", "")

    for task, meta, save in candidates:
        if not tools.file_exists(meta):
            continue
        log(f"eval {task} ...")
        r = tools.eval_probe(meta, group="6", save=save, select_by="macro_f1")
        if not r.get("ok"):
            continue
        rec: MetricRecord = {
            "task": task,
            "accuracy": r["accuracy"],
            "macro_f1": r["macro_f1"],
            "balanced_f1": r.get("balanced_f1", 0.0),
            "metadata": meta,
        }
        records.append(rec)
        log(f"  {task}: acc={r['accuracy']:.4f} f1={r['macro_f1']:.4f} "
            f"bal={r.get('balanced_f1', 0):.4f}")
        if r["macro_f1"] > best_f1 or (
            r["macro_f1"] == best_f1 and r["accuracy"] > best_acc
        ):
            best_acc = r["accuracy"]
            best_f1 = r["macro_f1"]
            best_bf1 = r.get("balanced_f1", 0.0)
            best_task = task

    done = _targets_met(state, best_acc, best_f1, best_bf1)
    return {
        "metrics_history": records,
        "best_accuracy": best_acc,
        "best_macro_f1": best_f1,
        "best_balanced_f1": best_bf1,
        "best_task": best_task,
        "done": done,
        "message": (f"TARGETS MET: {best_task}" if done else
                    f"best={best_task} acc={best_acc:.4f} f1={best_f1:.4f} bal={best_bf1:.4f}"),
    }


def planner(state: AgentState) -> dict:
    if state.get("done"):
        return {"next_action": "finish", "iteration": state["iteration"]}
    if state["iteration"] >= state["max_iterations"]:
        return {"next_action": "finish", "iteration": state["iteration"],
                "message": "max iterations"}

    actions = set(state.get("actions_taken", []))

    if not tools.file_exists("data/ml_perspective/metadata_perspective_clean.json"):
        if "clean_data" not in actions:
            return {"next_action": "clean_data", "iteration": state["iteration"] + 1}

    if tools.count_jsonl("data/vlm_relabel/labels.jsonl") < 1183:
        if "vlm_relabel_6" not in actions:
            return {"next_action": "vlm_relabel_6", "iteration": state["iteration"] + 1}

    if not tools.file_exists("data/ml_perspective/metadata_perspective_vlm6.json"):
        if "build_vlm6" not in actions:
            return {"next_action": "build_vlm6", "iteration": state["iteration"] + 1}

    if not tools.file_exists("data/ml_perspective/metadata_perspective_vlm6_oss.json"):
        if "balance_vlm6_oss" not in actions:
            return {"next_action": "balance_vlm6_oss", "iteration": state["iteration"] + 1}

    best_f1 = state.get("best_macro_f1", 0.0)
    if best_f1 < state["target_macro_f1"] and "train_probe_vlm6_oss" not in actions:
        return {"next_action": "train_probe_vlm6_oss", "iteration": state["iteration"] + 1}

    if "smoke_api" not in actions:
        return {"next_action": "smoke_api", "iteration": state["iteration"] + 1}

    return {"next_action": "finish", "iteration": state["iteration"]}


def _action_node(name: str, fn):
    def node(state: AgentState) -> dict:
        log(f"ACTION {name}")
        try:
            ok, detail = fn()
            err = "" if ok else detail
        except Exception as exc:
            ok, err, detail = False, str(exc), ""
        log(f"  {name}: ok={ok} {(detail or err)[:120]}")
        return {"actions_taken": [name], "last_error": err, "done": False}
    return node


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("evaluate", evaluate)
    g.add_node("plan", planner)
    g.add_node("clean_data", _action_node("clean_data", tools.clean_data))
    g.add_node("vlm_relabel_6", _action_node("vlm_relabel_6", lambda: tools.vlm_relabel("6")))
    g.add_node("build_vlm6", _action_node("build_vlm6", lambda: tools.build_dataset("6")))
    g.add_node("balance_vlm6_oss", _action_node("balance_vlm6_oss", tools.balance_vlm6_oss))
    g.add_node("train_probe_vlm6_oss", _action_node("train_probe_vlm6_oss", tools.train_probe_vlm6_oss))
    g.add_node("smoke_api", _action_node("smoke_api", tools.smoke_api))
    g.add_node("finish", lambda s: {"done": True, "message": s.get("message", "done")})

    g.set_entry_point("evaluate")
    g.add_conditional_edges("evaluate", lambda s: "finish" if s.get("done") else "plan",
                            {"plan": "plan", "finish": "finish"})
    g.add_conditional_edges("plan", lambda s: s.get("next_action", "finish"), {
        "clean_data": "clean_data",
        "vlm_relabel_6": "vlm_relabel_6",
        "build_vlm6": "build_vlm6",
        "balance_vlm6_oss": "balance_vlm6_oss",
        "train_probe_vlm6_oss": "train_probe_vlm6_oss",
        "smoke_api": "smoke_api",
        "finish": "finish",
    })
    for a in ("clean_data", "vlm_relabel_6", "build_vlm6", "balance_vlm6_oss",
              "train_probe_vlm6_oss", "smoke_api"):
        g.add_edge(a, "evaluate")
    g.add_edge("finish", END)
    return g.compile()


def main() -> int:
    ap = argparse.ArgumentParser(description="LangGraph Metric Agent (6 классов UTT)")
    ap.add_argument("--taxonomy", type=str, default="6", choices=("6", "3"))
    ap.add_argument("--target-acc", type=float, default=0.75)
    ap.add_argument("--target-f1", type=float, default=0.60)
    ap.add_argument("--target-balanced-f1", type=float, default=0.70)
    ap.add_argument("--max-iter", type=int, default=12)
    args = ap.parse_args()

    os.environ["CLASSIFIER_TAXONOMY"] = args.taxonomy

    initial: AgentState = {
        "taxonomy": args.taxonomy,
        "iteration": 0,
        "max_iterations": args.max_iter,
        "target_accuracy": args.target_acc,
        "target_macro_f1": args.target_f1,
        "target_balanced_f1": args.target_balanced_f1,
        "best_accuracy": 0.0,
        "best_macro_f1": 0.0,
        "best_balanced_f1": 0.0,
        "best_task": "",
        "metrics_history": [],
        "actions_taken": [],
        "next_action": "clean_data",
        "last_error": "",
        "done": False,
        "message": "",
    }

    log("=== Metric Agent start (6-class UTT) ===")
    log(f"targets: acc≥{args.target_acc} f1≥{args.target_f1} bal≥{args.target_balanced_f1}")
    log(f"instructions: {INSTRUCTIONS}")

    final = build_graph().invoke(initial, {"recursion_limit": 40})
    STATE_PATH.write_text(json.dumps(final, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    done = final.get("done", False)
    log(f"{'SUCCESS' if done else 'DONE'}: {final.get('message')}")
    log(f"actions: {final.get('actions_taken')}")

    with (ROOT / "results" / "AGENT_RUN.md").open("a", encoding="utf-8") as fh:
        fh.write(f"\n## Metric Agent {datetime.now(timezone.utc).isoformat()}\n")
        fh.write(f"- targets: acc≥{args.target_acc} f1≥{args.target_f1} bal≥{args.target_balanced_f1}\n")
        fh.write(f"- result: **{final.get('best_task')}** acc={final.get('best_accuracy'):.4f} "
                 f"f1={final.get('best_macro_f1'):.4f} bal={final.get('best_balanced_f1', 0):.4f}\n")
        fh.write(f"- targets_met: {done}\n")
        fh.write(f"- actions: {', '.join(final.get('actions_taken', []))}\n")

    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
