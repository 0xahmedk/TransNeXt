#!/usr/bin/env bash
set -euo pipefail

# Prepare ImageNet-100 Kaggle layout for torchvision ImageFolder training.
#
# Expected under DATA_PATH:
#   Labels.json
#   train.X1/   (or another shard selected by TRAIN_SPLIT)
#   val.X/
#
# This script creates/refreshes symlinks:
#   train -> TRAIN_SPLIT
#   val   -> val.X
#
# Usage:
#   bash prepare_imagenet100_train.sh
# Optional env vars:
#   DATA_PATH=data/imagenet-100
#   TRAIN_SPLIT=train.X1
#   ALIGN_VAL_TO_TRAIN=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DATA_PATH="${DATA_PATH:-data/imagenet-100}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train.X1}"
ALIGN_VAL_TO_TRAIN="${ALIGN_VAL_TO_TRAIN:-1}"

if [[ ! -d "$DATA_PATH" ]]; then
  echo "[prepare-in100] ERROR: missing dataset directory: $DATA_PATH"
  exit 1
fi

if [[ ! -f "$DATA_PATH/Labels.json" ]]; then
  echo "[prepare-in100] ERROR: missing Labels.json in $DATA_PATH"
  exit 1
fi

if [[ ! -d "$DATA_PATH/$TRAIN_SPLIT" ]]; then
  echo "[prepare-in100] ERROR: missing selected train split: $DATA_PATH/$TRAIN_SPLIT"
  echo "[prepare-in100] You can set TRAIN_SPLIT=train.X2|train.X3|train.X4 if available"
  exit 1
fi

if [[ ! -d "$DATA_PATH/val.X" ]]; then
  echo "[prepare-in100] ERROR: missing val.X in $DATA_PATH"
  exit 1
fi

echo "[prepare-in100] linking train -> $TRAIN_SPLIT"
ln -sfn "$TRAIN_SPLIT" "$DATA_PATH/train"

train_classes=$(find -L "$DATA_PATH/train" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
valx_classes=$(find "$DATA_PATH/val.X" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')

if [[ "$ALIGN_VAL_TO_TRAIN" == "1" && "$train_classes" -lt "$valx_classes" ]]; then
  subset_name="val.from_${TRAIN_SPLIT//./_}"
  subset_dir="$DATA_PATH/$subset_name"
  rm -rf "$subset_dir"
  mkdir -p "$subset_dir"

  while IFS= read -r class_dir; do
    class_name="$(basename "$class_dir")"
    if [[ -d "$DATA_PATH/val.X/$class_name" ]]; then
      ln -sfn "../val.X/$class_name" "$subset_dir/$class_name"
    fi
  done < <(find -L "$DATA_PATH/train" -mindepth 1 -maxdepth 1 -type d | sort)

  echo "[prepare-in100] linking val -> $subset_name (class-aligned with train)"
  ln -sfn "$subset_name" "$DATA_PATH/val"
else
  echo "[prepare-in100] linking val -> val.X"
  ln -sfn "val.X" "$DATA_PATH/val"
fi

val_classes=$(find -L "$DATA_PATH/val" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')

echo "[prepare-in100] train classes: $train_classes"
echo "[prepare-in100] val classes:   $val_classes"

if [[ "$train_classes" -ne "$val_classes" ]]; then
  echo "[prepare-in100] WARNING: train/val class counts differ ($train_classes vs $val_classes)."
fi

if [[ "$train_classes" -lt 100 ]]; then
  echo "[prepare-in100] NOTE: training on subset shard '$TRAIN_SPLIT' with $train_classes classes (not full IN100)."
fi

echo "[prepare-in100] done"
