import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from datetime import datetime

# Import the model definition
from transnext import transnext_micro

def main():
    parser = argparse.ArgumentParser(description='Finetune TransNeXt on Oxford-IIIT Pet Dataset')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory for the dataset')
    parser.add_argument('--weights', type=str, default='transnext_pets_best.pth', help='Path to the fine-tuned model weights (for saving or loading in test_only mode)')
    parser.add_argument('--initial_weights', type=str, default='models/transnext_micro_224_1k.pth', help='Path to the initial pretrained weights for full training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the new head')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--use_cpu', action='store_true', help='Use CPU even if GPU is available')
    parser.add_argument('--test_only', action='store_true', help='Only run evaluation on the test set')
    args = parser.parse_args()

    # Hardware check
    device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")
    print(f"Using device: {device}")

    # Data Loaders
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pet_test_dataset = datasets.OxfordIIITPet(root=args.data_dir, split='test', download=True, transform=transform)
    val_loader = DataLoader(pet_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model Surgery and Weight Loading
    if args.test_only:
        # In test_only mode, load the fine-tuned model directly with 37 classes
        model = transnext_micro(pretrained=False, num_classes=37)
        print(f"Loading fine-tuned weights from {args.weights}")
        if not os.path.exists(args.weights):
            print(f"Error: Fine-tuned weights file not found at {args.weights}")
            return
        checkpoint = torch.load(args.weights, map_location='cpu')
        model.load_state_dict(checkpoint, strict=True)
        model = model.to(device)
        print("Running in test-only mode.")
        accuracy, precision, recall, f1 = evaluate_model(model, val_loader, device)
        generate_evaluation_report(accuracy, precision, recall, f1)
        return
    else:
        # In training mode, load the base model with 1000 classes, then load initial weights, then replace head
        model = transnext_micro(pretrained=False, num_classes=1000)
        print(f"Loading initial pretrained weights from {args.initial_weights}")
        if not os.path.exists(args.initial_weights):
            print(f"Error: Initial pretrained weights file not found at {args.initial_weights}")
            return
        checkpoint = torch.load(args.initial_weights, map_location='cpu')
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Replace the head for the new number of classes (37)
        num_features = model.head.in_features
        model.head = nn.Linear(num_features, 37)
        model = model.to(device) # Move to device after modifying
        
        # Unfreeze the parameters of the new head
        for param in model.head.parameters():
            param.requires_grad = True
    
    # Data Loaders for training
    pet_train_dataset = datasets.OxfordIIITPet(root=args.data_dir, split='trainval', download=True, transform=transform)
    train_loader = DataLoader(pet_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Training Loop (only if not test_only)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.head.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

    best_acc = 0.0
    train_losses = []
    val_accuracies = []

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            train_pbar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{args.epochs}, Average Training Loss: {epoch_loss:.4f}")

        # Evaluation
        accuracy, _, _, _ = evaluate_model(model, val_loader, device) # Evaluate returns 4 values, but we only need accuracy for comparison
        val_accuracies.append(accuracy)

        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), args.weights)
            print(f"Saved best model weights to {args.weights}")
    
    # Generate plots after training
    generate_plots(train_losses, val_accuracies, args.epochs)

def evaluate_model(model, data_loader, device):
    model.eval()
    all_labels = []
    all_predictions = []
    correct = 0
    total = 0
    with torch.no_grad():
        val_pbar = tqdm(data_loader, desc="Evaluation")
        for inputs, labels in val_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
            val_pbar.set_postfix(accuracy=f"{(100 * correct / total):.2f}%")

    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')

    print(f"Validation Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    return accuracy, precision, recall, f1

def generate_evaluation_report(accuracy, precision, recall, f1):
    output_dir = os.path.join('runs', datetime.now().strftime('%Y%m%d_%H%M%S_evaluation_report'))
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(report_path, 'w') as f:
        f.write(f"Evaluation Report (Test-Only Mode)\n")
        f.write(f"-----------------------------------\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
    print(f"Evaluation report saved to {report_path}")

def generate_plots(train_losses, val_accuracies, epochs):
    output_dir = os.path.join('runs', datetime.now().strftime('%Y%m%d_%H%M%S_finetune_pets_plots'))
    os.makedirs(output_dir, exist_ok=True)

    epochs_range = range(1, epochs + 1)

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    plt.close()

    # Plot validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'validation_accuracy.png'))
    plt.close()

    print(f"Plots saved to {output_dir}")

if __name__ == '__main__':
    main()