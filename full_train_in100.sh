#!/usr/bin/env bash
set -euo pipefail

# Full training from scratch on ImageNet-100 subset (single split or selected shard).
#
# Usage:
#   bash full_train_in100.sh
#
# Optional env vars:
#   PYTHON_BIN=/usr/bin/python3
#   CONFIG=classification/configs/transnext_micro.py
#   DATA_PATH=data/imagenet-100
#   TRAIN_SPLIT=train.X1
#   BATCH_SIZE=32
#   EVAL_BATCH_SIZE=32
#   EPOCHS=30
#   OUTPUT_DIR=checkpoints/in100_full_micro
#   INSTALL_DEPS=0
#   DEVICE=cpu

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
CONFIG="${CONFIG:-classification/configs/transnext_micro.py}"
DATA_PATH="${DATA_PATH:-data/imagenet-100}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train.X1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-30}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/in100_full_micro}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"
DEVICE="${DEVICE:-cpu}"

if [[ "$INSTALL_DEPS" == "1" ]]; then
  echo "[full-train-in100] installing minimal deps (timm==0.5.4)..."
  "$PYTHON_BIN" -m pip install --break-system-packages "setuptools<81" "timm==0.5.4"
fi

TIMM_VERSION="$($PYTHON_BIN -c "import importlib; m=importlib.import_module('timm'); print(m.__version__)" 2>/dev/null || true)"
if [[ -z "$TIMM_VERSION" ]]; then
  echo "[full-train-in100] timm not found, installing timm==0.5.4"
  "$PYTHON_BIN" -m pip install --break-system-packages "setuptools<81" "timm==0.5.4"
elif [[ "$TIMM_VERSION" != "0.5.4" ]]; then
  echo "[full-train-in100] timm version $TIMM_VERSION is incompatible, pinning to 0.5.4"
  "$PYTHON_BIN" -m pip install --break-system-packages --upgrade --force-reinstall "setuptools<81" "timm==0.5.4"
fi

echo "[full-train-in100] preparing dataset links"
DATA_PATH="$DATA_PATH" TRAIN_SPLIT="$TRAIN_SPLIT" bash prepare_imagenet100_train.sh

mkdir -p "$OUTPUT_DIR"

echo "[full-train-in100] python: $PYTHON_BIN"
"$PYTHON_BIN" -c "import sys; print('[full-train-in100] python version:', sys.version.split()[0])"
echo "[full-train-in100] device: $DEVICE"

echo "[full-train-in100] starting training..."
PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u classification/main.py \
  --config "$CONFIG" \
  --data-path "$DATA_PATH" \
  --device "$DEVICE" \
  --disable-distributed \
  --num_workers 0 \
  --batch-size "$BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --output_dir "$OUTPUT_DIR" \
  --mixup 0 \
  --cutmix 0 \
  --smoothing 0 \
  --no-compile-model \
  --no-pin-mem

echo "[full-train-in100] done"
