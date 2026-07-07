#!/usr/bin/env bash
# Полный продакшен-пайплайн: сбор → EDA → чистка → VLM-метки → probe → eval → отчёт.
#
# Требования: data/dataset/ (OSM+панорамы), GPU для train/VLM.
# Конфиг: config/pipeline.env.example → .env
#
#   cp config/pipeline.env.example .env
#   source .env
#   bash scripts/pipeline/run_production.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-./venv/bin/python}"
[[ -x "$PY" ]] || PY=python3

if [[ -f .env ]]; then set -a; source .env; set +a; fi

STEPS=(
  "01_collect_data.sh"
  "02_eda_audit.sh"
  "03_clean_and_balance.sh"
  "04_vlm_relabel.sh"
  "05_train_probe.sh"
  "06_evaluate.sh"
  "07_report.sh"
)

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Production ML pipeline — UTT panorama classifier        ║"
echo "╚══════════════════════════════════════════════════════════╝"

for step in "${STEPS[@]}"; do
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  bash "scripts/pipeline/$step"
done

echo ""
echo "✓ Pipeline complete. Start UI: make serve"
