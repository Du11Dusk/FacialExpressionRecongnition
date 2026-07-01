"""Full training with output logging"""
import sys
import os

# Redirect stdout to a log file
log_file = open("training_log.txt", "w", encoding="utf-8")
sys.stdout = log_file
sys.stderr = log_file

print("=" * 40)
print("人脸表情识别系统 - 完整训练脚本")
print("=" * 40)

import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import FER2013
from model import SimpleCNN

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([transforms.ToTensor()])

    print("Loading dataset...")
    t0 = time.time()
    train_dataset = FER2013("data/fer2013.csv", transform=train_transform, phase="Training")
    val_dataset = FER2013("data/fer2013.csv", transform=test_transform, phase="PublicTest")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)} ({time.time()-t0:.1f}s)")

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SimpleCNN(dropout_rate=0.5).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6)

    os.makedirs("checkpoints", exist_ok=True)

    epochs = 50
    best_acc = 0.0
    patience_counter = 0
    early_stop_patience = 10
    training_start = time.time()

    print("Starting training...")
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
                log_file.flush()

        avg_loss = running_loss / len(train_loader)
        val_acc = evaluate(model, val_loader, device)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_acc)
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={val_acc*100:.2f}%, LR={current_lr:.6f}, Time={epoch_time:.1f}s")
        log_file.flush()

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print(f"  >> Saved best model! Acc={best_acc*100:.2f}%")
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    total_time = time.time() - training_start
    print(f"\nTraining complete! Best val acc: {best_acc*100:.2f}%")
    print(f"Total time: {total_time:.1f}s")
    print(f"Model saved to checkpoints/best_model.pth")

    # Save history
    history = {"best_accuracy": best_acc, "total_epochs": epoch+1, "training_time": total_time}
    with open("checkpoints/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    log_file.close()

if __name__ == "__main__":
    main()
