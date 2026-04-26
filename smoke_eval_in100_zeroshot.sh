#!/usr/bin/env bash
set -euo pipefail

# Smoke-test: evaluate ImageNet-1K pretrained TransNeXt-Micro on ImageNet-100 val subset.
# This script keeps model outputs at 1000 classes and remaps subset WNID labels to 1K ids.
#
# Usage:
#   bash smoke_eval_in100_zeroshot.sh
#
# Optional env vars:
#   PYTHON_BIN=/usr/bin/python3
#   CONFIG=classification/configs/transnext_micro.py
#   DATA_PATH=data/imagenet-100
#   CHECKPOINT=models/transnext_micro_224_1k.pth
#   CLASS_INDEX_JSON=data/imagenet_class_index.json
#   BATCH_SIZE=16
#   EVAL_BATCH_SIZE=16
#   TIMEOUT_SEC=180
#   INSTALL_DEPS=0

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
CONFIG="${CONFIG:-classification/configs/transnext_micro.py}"
DATA_PATH="${DATA_PATH:-data/imagenet-100}"
CHECKPOINT="${CHECKPOINT:-transnext_micro_224_1k.pth}"
CLASS_INDEX_JSON="${CLASS_INDEX_JSON:-data/imagenet_class_index.json}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
TIMEOUT_SEC="${TIMEOUT_SEC:-180}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"

VAL_DIR="$DATA_PATH/val"
VAL_X_DIR="$DATA_PATH/val.X"

if [[ ! -d "$VAL_DIR" ]]; then
  if [[ -d "$VAL_X_DIR" ]]; then
    echo "[zero-shot-smoke] creating symlink: $VAL_DIR -> val.X"
    ln -sfn val.X "$VAL_DIR"
  else
    echo "[zero-shot-smoke] ERROR: missing validation folder. Expected one of:"
    echo "  - $VAL_DIR"
    echo "  - $VAL_X_DIR"
    exit 1
  fi
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  if [[ "$CHECKPOINT" == "transnext_micro_224_1k.pth" && -f "models/transnext_micro_224_1k.pth" ]]; then
    CHECKPOINT="models/transnext_micro_224_1k.pth"
    echo "[zero-shot-smoke] using checkpoint at $CHECKPOINT"
  else
    echo "[zero-shot-smoke] ERROR: checkpoint not found at $CHECKPOINT"
    echo "[zero-shot-smoke] Set CHECKPOINT=/path/to/transnext_micro_224_1k.pth"
    exit 1
  fi
fi

if [[ ! -f "$CLASS_INDEX_JSON" ]]; then
  echo "[zero-shot-smoke] class index JSON missing, downloading to $CLASS_INDEX_JSON"
  mkdir -p "$(dirname "$CLASS_INDEX_JSON")"

  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json" -o "$CLASS_INDEX_JSON" \
      || curl -fsSL "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json" -o "$CLASS_INDEX_JSON"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$CLASS_INDEX_JSON" "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json" \
      || wget -qO "$CLASS_INDEX_JSON" "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"
  else
    "$PYTHON_BIN" - <<'PY'
import json
import os
import urllib.request

target = os.environ['CLASS_INDEX_JSON']
urls = [
    'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json',
    'https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json',
]
ok = False
for url in urls:
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            data = r.read().decode('utf-8')
        obj = json.loads(data)
        if isinstance(obj, dict) and ('0' in obj or 'n01440764' in obj):
            with open(target, 'w') as f:
                f.write(data)
            ok = True
            break
    except Exception:
        pass
if not ok:
    raise SystemExit(1)
PY
  fi

  if [[ ! -f "$CLASS_INDEX_JSON" ]]; then
    echo "[zero-shot-smoke] ERROR: failed to download class index JSON"
    echo "[zero-shot-smoke] Set CLASS_INDEX_JSON=/path/to/imagenet_class_index.json"
    exit 1
  fi
fi

CLASS_INDEX_JSON="$CLASS_INDEX_JSON" "$PYTHON_BIN" - <<'PY'
import json
import os

path = os.environ['CLASS_INDEX_JSON']
with open(path, 'r') as f:
    data = json.load(f)
if not isinstance(data, dict):
    raise SystemExit('invalid class index JSON: not an object')
if not data:
    raise SystemExit('invalid class index JSON: empty')
first_key = next(iter(data.keys()))
if not (first_key.startswith('n') or first_key.isdigit()):
    raise SystemExit('invalid class index JSON: unrecognized key format')
print('[zero-shot-smoke] class index JSON OK:', path)
PY

if [[ "$INSTALL_DEPS" == "1" ]]; then
  echo "[zero-shot-smoke] installing minimal deps (timm==0.5.4)..."
  "$PYTHON_BIN" -m pip install --break-system-packages "setuptools<81" "timm==0.5.4"
fi

TIMM_VERSION="$($PYTHON_BIN -c "import importlib; m=importlib.import_module('timm'); print(m.__version__)" 2>/dev/null || true)"
if [[ -z "$TIMM_VERSION" ]]; then
  echo "[zero-shot-smoke] timm not found, installing timm==0.5.4"
  "$PYTHON_BIN" -m pip install --break-system-packages "setuptools<81" "timm==0.5.4"
elif [[ "$TIMM_VERSION" != "0.5.4" ]]; then
  echo "[zero-shot-smoke] timm version $TIMM_VERSION is incompatible, pinning to 0.5.4"
  "$PYTHON_BIN" -m pip install --break-system-packages --upgrade --force-reinstall "setuptools<81" "timm==0.5.4"
fi

echo "[zero-shot-smoke] using python: $PYTHON_BIN"
"$PYTHON_BIN" -c "import sys; print('[zero-shot-smoke] python version:', sys.version.split()[0])"

echo "[zero-shot-smoke] starting eval smoke test..."
timeout "$TIMEOUT_SEC" "$PYTHON_BIN" classification/main.py \
  --config "$CONFIG" \
  --eval \
  --data-path "$DATA_PATH" \
  --resume "$CHECKPOINT" \
  --imagenet-class-index "$CLASS_INDEX_JSON" \
  --device auto \
  --disable-distributed \
  --num_workers 0 \
  --batch-size "$BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --no-compile-model \
  --no-pin-mem

echo "[zero-shot-smoke] done"
