#!/usr/bin/env bash
set -euo pipefail

# Simple smoke-test runner for TransNeXt classification eval on low-resource machines.
# Usage:
#   bash smoke_eval.sh
# Optional env vars:
#   PYTHON_BIN=/usr/bin/python3
#   CONFIG=classification/configs/transnext_tiny.py
#   DATA_PATH=data/imagenet-100
#   BATCH_SIZE=16
#   EVAL_BATCH_SIZE=16
#   TIMEOUT_SEC=120
#   INSTALL_DEPS=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
CONFIG="${CONFIG:-classification/configs/transnext_tiny.py}"
DATA_PATH="${DATA_PATH:-data/imagenet-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
TIMEOUT_SEC="${TIMEOUT_SEC:-120}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"

VAL_DIR="$DATA_PATH/val"
VAL_X_DIR="$DATA_PATH/val.X"

if [[ ! -d "$VAL_DIR" ]]; then
  if [[ -d "$VAL_X_DIR" ]]; then
    echo "[smoke] creating symlink: $VAL_DIR -> val.X"
    ln -sfn val.X "$VAL_DIR"
  else
    echo "[smoke] ERROR: missing validation folder. Expected one of:"
    echo "        - $VAL_DIR"
    echo "        - $VAL_X_DIR"
    exit 1
  fi
fi

if [[ "$INSTALL_DEPS" == "1" ]]; then
  echo "[smoke] installing minimal deps (timm 0.5.4)..."
  "$PYTHON_BIN" -m pip install --break-system-packages "setuptools<81" "timm==0.5.4"
fi

TIMM_VERSION="$($PYTHON_BIN -c "import importlib; m=importlib.import_module('timm'); print(m.__version__)" 2>/dev/null || true)"
if [[ -z "$TIMM_VERSION" ]]; then
  echo "[smoke] timm not found, installing timm==0.5.4"
  "$PYTHON_BIN" -m pip install --break-system-packages "setuptools<81" "timm==0.5.4"
elif [[ "$TIMM_VERSION" != "0.5.4" ]]; then
  echo "[smoke] timm version $TIMM_VERSION is incompatible, pinning to 0.5.4"
  "$PYTHON_BIN" -m pip install --break-system-packages --upgrade --force-reinstall "setuptools<81" "timm==0.5.4"
fi

echo "[smoke] using python: $PYTHON_BIN"
"$PYTHON_BIN" -c "import sys; print('[smoke] python version:', sys.version.split()[0])"

echo "[smoke] starting eval smoke test..."
timeout "$TIMEOUT_SEC" "$PYTHON_BIN" classification/main.py \
  --config "$CONFIG" \
  --eval \
  --data-path "$DATA_PATH" \
  --device auto \
  --disable-distributed \
  --num_workers 0 \
  --batch-size "$BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --no-compile-model \
  --no-pin-mem

echo "[smoke] done"
