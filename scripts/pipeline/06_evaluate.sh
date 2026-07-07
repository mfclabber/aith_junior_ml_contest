#!/usr/bin/env bash
# [6/7] Оценка моделей и бенчмарк.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-./venv/bin/python}"
mkdir -p results

echo "[06] Benchmark probes"
if [[ -d checkpoints ]] && ls checkpoints/*.pt 1>/dev/null 2>&1; then
  "$PY" scripts/eval/benchmark_probes.py --out results/probe_benchmark.json 2>/dev/null || true
fi

if [[ -f results/out_for_katya_new.gpkg ]]; then
  echo "[06] Compare with analyst GPKG"
  CLASSIFIER_PROBE="${CLASSIFIER_PROBE:-clip_probe_vlm6_oss.pt}" \
    "$PY" scripts/eval/compare_analyst_gpkg.py \
      --gpkg results/out_for_katya_new.gpkg --n 30 \
      --out results/analyst_compare.json
else
  echo "[06] GPKG eval skipped (no results/out_for_katya_new.gpkg)"
fi

echo "[06] Metric agent thresholds"
if [[ "${RUN_AGENT:-0}" == "1" ]]; then
  bash scripts/agent/run_metric_agent.sh
fi
