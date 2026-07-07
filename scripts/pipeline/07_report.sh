#!/usr/bin/env bash
# [7/7] Сводный отчёт по пайплайну.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p results

REPORT=results/PIPELINE_REPORT.md
{
  echo "# Pipeline report"
  echo ""
  echo "Generated: $(date -Iseconds)"
  echo ""
  echo "## Checkpoints"
  ls -la checkpoints/*.pt 2>/dev/null || echo "(none)"
  echo ""
  echo "## Results"
  ls -la results/*.json results/*.txt 2>/dev/null || echo "(none)"
} > "$REPORT"

echo "[07] → $REPORT"
