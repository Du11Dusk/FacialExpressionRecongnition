"""Fast training with fewer epochs for testing"""
import os
import time
import json
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from dataset import FER2013
from model import SimpleCNN

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Use smaller subset for faster training
    transform = transforms.Compose([transforms.ToTensor()])
    
    print("Loading dataset...")
    t0 = time.time()
    full_dataset = FER2013("data/fer2013.csv", transform=transform, phase="Training")
    print(f"Full dataset: {len(full_dataset)} samples ({time.time()-t0:.1f}s)")
    
    # Use subset for speed
    subset_size = min(5000, len(full_dataset))
    indices = list(range(subset_size))
    train_dataset = Subset(full_dataset, indices)
    
    val_dataset = FER2013("data/fer2013.csv", transform=transform, phase="PublicTest")
    
    print(f"Train subset: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    model = SimpleCNN(dropout_rate=0.5).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    os.makedirs("checkpoints", exist_ok=True)
    
    epochs = 5
    best_acc = 0.0
    
    print("Starting training...")
    for epoch in range(epochs):
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
            
            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = correct / total
        print(f"Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f}, Val Acc={val_acc*100:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print(f"  >> Saved best model! Acc={best_acc*100:.2f}%")
    
    print(f"\nTraining complete! Best val acc: {best_acc*100:.2f}%")
    print("Model saved to checkpoints/best_model.pth")

if __name__ == "__main__":
    main()
