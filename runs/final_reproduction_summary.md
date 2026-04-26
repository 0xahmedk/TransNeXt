# TransNeXt Reproduction Summary (Resource-Constrained)

Generated: 2026-04-22

## Executive Summary

This project aimed to reproduce TransNeXt behavior under constrained compute and time, using CPU-only execution and an ImageNet-100 subset setup.

Two phases were completed:

1. Zero-shot evaluation using an official ImageNet-1K pretrained TransNeXt-Micro checkpoint.
2. Training-from-scratch on the available ImageNet-100 training shard(s), followed by validation.

The experiments successfully validated both inference and training pipelines end-to-end in the local environment.

## Environment and Constraints

- Hardware mode: CPU-only (no NVIDIA GPU available).
- Dataset used: ImageNet-100 variant from Kaggle with shard-based training directories.
- Available training shard during this run: `train.X1`.
- Validation source: `val.X` (class-aligned subset generated for shard-consistent training/evaluation).
- Time/resource limits: full ImageNet-1K-scale reproduction was not feasible.

## Phase 1 — Zero-Shot Evaluation (Pretrained)

### Objective

Evaluate pretrained `TransNeXt-Micro` on the local ImageNet-100 validation setup without additional training.

### Method

- Checkpoint: `models/transnext_micro_224_1k.pth`
- Label-space alignment: WNID-to-ImageNet1K mapping enabled.
- Device: CPU
- Batch size: 16

### Achieved Results

From the generated evaluation report:

- Final Acc@1: **87.50%**
- Final Acc@5: **97.86%**
- Final loss: **0.565**
- Validation images evaluated: **5000**

Reference artifact:

- `runs/report_20260421_172454/report.md`

## Phase 2 — Training From Scratch

### Objective

Train TransNeXt-Micro from initialization on available ImageNet-100 training data under limited resources, then evaluate.

### Method

- Model: `transnext_micro`
- Device: CPU
- Training log source: `checkpoints/transnext_micro/log.txt`
- Available epochs in this tracked run: 4 (epochs 0–3)

### Achieved Results (latest recorded epoch)

From the generated training report:

- Epoch: **3**
- Train loss: **2.5856**
- Test loss: **1.8977**
- Test Acc@1: **43.52%**
- Test Acc@5: **80.00%**
- Best Test Acc@1 in this run: **43.52%**

Reference artifact:

- `runs/train_report_latest/report.md`

## Interpretation

- The zero-shot stage demonstrated strong transfer performance with pretrained TransNeXt-Micro on the local ImageNet-100 validation protocol.
- The from-scratch stage demonstrated successful optimization and early convergence trends under CPU-only constraints.
- Absolute metrics from the from-scratch run are expected to be lower than paper-reported ImageNet-1K results due to:
  - reduced dataset coverage (shard-limited training),
  - CPU-only budget constraints,
  - shorter effective training horizon.

## What Was Successfully Reproduced

- End-to-end TransNeXt inference pipeline.
- End-to-end TransNeXt training pipeline (from initialization).
- Checkpoint save/load and evaluation reporting workflow.
- Automated reporting with CSV + plots + markdown summaries.

## Scope Limitations (Explicit)

This work is a **resource-constrained reproduction**, not a full benchmark parity attempt. Therefore, the outcomes should be interpreted as:

- pipeline validation,
- practical reproducibility under local constraints,
- performance trend confirmation,
  not strict replication of full-scale ImageNet-1K paper baselines.

## Recommended Next Steps (if more resources become available)

1. Add additional training shards (`train.X2`, `train.X3`, `train.X4`) to increase class/data coverage.
2. Extend training epochs and compare convergence curves.
3. Repeat runs (3+ seeds) and report mean ± std for statistical stability.
4. If GPU becomes available, rerun with larger batch size for faster/stronger convergence.
