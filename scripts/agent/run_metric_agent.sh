#!/usr/bin/env bash
# Metric Agent — достижение целевых метрик на 6 классах UTT.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export CLASSIFIER_TAXONOMY=6
export CLASSIFIER_MODE=ml
export CLASSIFIER_PROBE=clip_probe_vlm6_oss.pt
export CLASSIFIER_FALLBACK_HEURISTIC=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

PY="./venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "venv not found. Create: python3 -m venv venv && ./venv/bin/pip install -r requirements-dev.txt"
  exit 1
fi

"$PY" -m pip install -q -r requirements-dev.txt 2>/dev/null || true

echo "=== Metric Agent (6 классов UTT) ==="
echo "Цели: acc≥0.75  macro-F1≥0.60  balanced-F1≥0.70"
echo "Инструкция: AGENT_INSTRUCTIONS.md"
echo ""

exec "$PY" scripts/agent/langgraph_agent.py \
  --taxonomy 6 \
  --target-acc 0.75 \
  --target-f1 0.60 \
  --target-balanced-f1 0.70 \
  --max-iter 12 \
  "$@"
