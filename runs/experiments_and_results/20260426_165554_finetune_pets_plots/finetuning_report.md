# TransNeXt Fine-tuning on Oxford-IIIT Pet Dataset - Detailed Report

## 1. Objective

The objective was to fine-tune the `transnext_micro` model on the Oxford-IIIT Pet dataset. This involved several key steps: setting up data loaders with appropriate transforms, performing model surgery to adapt the pretrained model for the new dataset, implementing a training loop with an AdamW optimizer and CrossEntropyLoss, ensuring hardware compatibility with mixed precision and CPU fallback, and evaluating the model after training, saving the best weights, and generating plots.

## 2. Implementation Details

A Python script, `classification/finetune_pets.py`, was developed to achieve the fine-tuning objective.

### 2.1. Data Loaders

The `torchvision.datasets.OxfordIIITPet` dataset was used to load the training and validation data. Standard image transformations were applied:

- `transforms.Resize(224)`
- `transforms.CenterCrop(224)`
- `transforms.ToTensor()`
- `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` (ImageNet statistics)

`DataLoader` was used for batching with a configurable `batch_size` and `num_workers=4` for parallel data loading.

### 2.2. Model Surgery

The `transnext_micro` model was loaded. The key steps for model adaptation were:

1.  **Initial Model Loading**: The base `transnext_micro` model was initialized.
2.  **Pretrained Weights Loading**: The model was loaded with `models/transnext_micro_224_1k.pth` weights. A mechanism was implemented to handle different `state_dict` structures (either a direct model state or a dictionary containing a 'model' key).
3.  **Parameter Freezing**: All parameters of the base model were frozen to prevent their updates during fine-tuning.
4.  **Head Replacement**: The original classification head (`model.head`) was replaced with a new `nn.Linear` layer adapted for the Oxford-IIIT Pet dataset, which has 37 classes. The new head was initialized with `nn.Linear(model.head.in_features, 37)`.
5.  **Head Unfreezing**: Only the parameters of the newly added classification head were unfrozen, making them trainable.

### 2.3. Training Loop

A standard training loop was set up for 5 epochs:

- **Optimizer**: `torch.optim.AdamW` with a learning rate of `1e-3` (only for the new head's parameters).
- **Loss Function**: `torch.nn.CrossEntropyLoss`.
- **Mixed Precision**: `torch.cuda.amp.GradScaler` was used for automatic mixed-precision training, saving GPU memory and accelerating computation when a CUDA device is available.

### 2.4. Hardware Check

The script automatically detects and utilizes a CUDA-enabled GPU if available. A command-line argument `--use_cpu` was provided to force CPU usage, allowing flexibility for environments where a GPU is unavailable or full.

### 2.5. Evaluation and Model Saving

After each epoch, the model was evaluated on the validation set. The model's state dictionary was saved as `transnext_pets_best.pth` if the current epoch's validation accuracy surpassed the `best_acc` achieved so far.

### 2.6. Logging and Plotting

- **Progress Bars**: `tqdm` was integrated into both training and validation loops to display real-time progress.
- **Evaluation Report**: When run with the `--test_only` flag, the script performs evaluation and generates a text report (`evaluation_metrics.txt`) containing Accuracy, Precision, Recall, and F1-Score in a timestamped directory under `runs/`.
- **Training Plots**: When run in full training mode (without `--test_only`), the script collects epoch-wise training loss and validation accuracy. After training, it generates and saves plots (`training_loss.png` and `validation_accuracy.png`) in a timestamped directory under `runs/`.

## 3. Dependencies

The `classification/requirements.txt` file was updated to include:

- `torch>=2.11.0`
- `torchvision>=0.26.0`
- `timm>=1.0.26`
- `tqdm>=4.62.3`
- `matplotlib>=3.3.4`
- `scikit-learn>=0.24.2`
  (Note: `mmcv` was removed from the requirements due to persistent installation issues and its non-essential nature for this specific task.)

## 4. Results

### 4.1. Evaluation Metrics

The following metrics were obtained when evaluating the fine-tuned `transnext_pets_best.pth` model on the Oxford-IIIT Pet dataset in `test_only` mode:

- **Accuracy**: 92.50%
- **Precision**: 0.9276
- **Recall**: 0.9249
- **F1-Score**: 0.9238

This report is located at `classification/runs/20260426_150845_evaluation_report/evaluation_metrics.txt`.
