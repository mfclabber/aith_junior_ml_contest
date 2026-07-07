"""Состояние LangGraph Metric Agent."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict


Action = Literal[
    "clean_data",
    "vlm_relabel_6",
    "build_vlm6",
    "balance_vlm6_oss",
    "train_probe_vlm6_oss",
    "smoke_api",
    "finish",
]


class MetricRecord(TypedDict, total=False):
    task: str
    accuracy: float
    macro_f1: float
    balanced_f1: float
    metadata: str


class AgentState(TypedDict):
    taxonomy: str
    iteration: int
    max_iterations: int
    target_accuracy: float
    target_macro_f1: float
    target_balanced_f1: float
    best_accuracy: float
    best_macro_f1: float
    best_balanced_f1: float
    best_task: str
    metrics_history: Annotated[list[MetricRecord], operator.add]
    actions_taken: Annotated[list[str], operator.add]
    next_action: Action
    last_error: str
    done: bool
    message: str
