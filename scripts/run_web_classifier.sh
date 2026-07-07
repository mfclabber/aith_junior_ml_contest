#!/usr/bin/env bash
# Запуск веб-классификатора на порту, который часто не блокируют в LAN (можно переопределить PORT).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PORT="${PORT:-8765}"
exec ./venv/bin/python -m web_app.app --host 0.0.0.0 --port "$PORT" "$@"
