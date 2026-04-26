#!/usr/bin/env bash
set -euo pipefail

# Build CSV + plots + markdown report from evaluation logs.
#
# Usage:
#   bash make_eval_report.sh
#
# Optional env vars:
#   PYTHON_BIN=/usr/bin/python3
#   INPUTS="runs"
#   PATTERN="*.log"
#   OUT_DIR="runs/report_$(date +%Y%m%d_%H%M%S)"
#   INSTALL_PLOT_DEPS=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
INPUTS="${INPUTS:-runs}"
PATTERN="${PATTERN:-*.log}"
OUT_DIR="${OUT_DIR:-runs/report_$(date +%Y%m%d_%H%M%S)}"
INSTALL_PLOT_DEPS="${INSTALL_PLOT_DEPS:-1}"

echo "[report] python: $PYTHON_BIN"
echo "[report] inputs: $INPUTS"
echo "[report] pattern: $PATTERN"
echo "[report] out dir: $OUT_DIR"

if [[ "$INSTALL_PLOT_DEPS" == "1" ]]; then
	if ! "$PYTHON_BIN" -c "import matplotlib" >/dev/null 2>&1; then
		echo "[report] matplotlib missing; installing for plot generation..."
		"$PYTHON_BIN" -m pip install --break-system-packages matplotlib
	fi
fi

# shellcheck disable=SC2086
"$PYTHON_BIN" tools/generate_eval_report.py $INPUTS --pattern "$PATTERN" --out-dir "$OUT_DIR"

echo "[report] done"
