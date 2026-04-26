#!/usr/bin/env bash
set -euo pipefail

# Generate training stats report from checkpoint log.txt
#
# Usage:
#   bash make_train_report.sh
#
# Optional env vars:
#   PYTHON_BIN=/usr/bin/python3
#   TRAIN_LOG=checkpoints/transnext_micro/log.txt
#   OUT_DIR=runs/train_report_latest
#   INSTALL_PLOT_DEPS=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
TRAIN_LOG="${TRAIN_LOG:-checkpoints/transnext_micro/log.txt}"
OUT_DIR="${OUT_DIR:-runs/train_report_latest}"
INSTALL_PLOT_DEPS="${INSTALL_PLOT_DEPS:-1}"

echo "[train-report] python: $PYTHON_BIN"
echo "[train-report] train log: $TRAIN_LOG"
echo "[train-report] out dir: $OUT_DIR"

if [[ ! -f "$TRAIN_LOG" ]]; then
  echo "[train-report] ERROR: missing train log: $TRAIN_LOG"
  exit 1
fi

if [[ "$INSTALL_PLOT_DEPS" == "1" ]]; then
  if ! "$PYTHON_BIN" -c "import matplotlib" >/dev/null 2>&1; then
    echo "[train-report] matplotlib missing; installing..."
    "$PYTHON_BIN" -m pip install --break-system-packages matplotlib
  fi
fi

"$PYTHON_BIN" tools/generate_train_report.py "$TRAIN_LOG" --out-dir "$OUT_DIR"

echo "[train-report] done"
