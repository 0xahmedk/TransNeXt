#!/usr/bin/env bash
set -euo pipefail

# Smoke train+eval from scratch on ImageNet-100 subset.
#
# Usage:
#   bash smoke_train_eval_in100.sh
#
# Optional env vars:
#   PYTHON_BIN=/usr/bin/python3
#   CONFIG=classification/configs/transnext_micro.py
#   DATA_PATH=data/imagenet-100
#   TRAIN_SPLIT=train.X1
#   BATCH_SIZE=16
#   EVAL_BATCH_SIZE=16
#   EPOCHS=1
#   OUTPUT_DIR=checkpoints/in100_smoke_micro
#   INSTALL_DEPS=0
#   DEVICE=cpu

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
CONFIG="${CONFIG:-classification/configs/transnext_micro.py}"
DATA_PATH="${DATA_PATH:-data/imagenet-100}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train.X1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/in100_smoke_micro}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"
DEVICE="${DEVICE:-cpu}"

if [[ "$INSTALL_DEPS" == "1" ]]; then
  echo "[smoke-train-in100] installing minimal deps (timm==0.5.4)..."
  "$PYTHON_BIN" -m pip install --break-system-packages "setuptools<81" "timm==0.5.4"
fi

TIMM_VERSION="$($PYTHON_BIN -c "import importlib; m=importlib.import_module('timm'); print(m.__version__)" 2>/dev/null || true)"
if [[ -z "$TIMM_VERSION" ]]; then
  echo "[smoke-train-in100] timm not found, installing timm==0.5.4"
  "$PYTHON_BIN" -m pip install --break-system-packages "setuptools<81" "timm==0.5.4"
elif [[ "$TIMM_VERSION" != "0.5.4" ]]; then
  echo "[smoke-train-in100] timm version $TIMM_VERSION is incompatible, pinning to 0.5.4"
  "$PYTHON_BIN" -m pip install --break-system-packages --upgrade --force-reinstall "setuptools<81" "timm==0.5.4"
fi

echo "[smoke-train-in100] preparing dataset links"
DATA_PATH="$DATA_PATH" TRAIN_SPLIT="$TRAIN_SPLIT" bash prepare_imagenet100_train.sh

mkdir -p "$OUTPUT_DIR"

echo "[smoke-train-in100] python: $PYTHON_BIN"
"$PYTHON_BIN" -c "import sys; print('[smoke-train-in100] python version:', sys.version.split()[0])"
echo "[smoke-train-in100] device: $DEVICE"

echo "[smoke-train-in100] starting smoke training..."
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

CKPT="$OUTPUT_DIR/last.pth"
if [[ ! -f "$CKPT" ]]; then
  echo "[smoke-train-in100] ERROR: expected checkpoint not found: $CKPT"
  exit 1
fi

echo "[smoke-train-in100] training done, starting eval..."
PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u classification/main.py \
  --config "$CONFIG" \
  --eval \
  --data-path "$DATA_PATH" \
  --resume "$CKPT" \
  --device "$DEVICE" \
  --disable-distributed \
  --num_workers 0 \
  --batch-size "$BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --no-compile-model \
  --no-pin-mem

echo "[smoke-train-in100] done"
