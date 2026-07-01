import os
import time
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import FER2013
from model import SimpleCNN


def evaluate(model, dataloader, device):
    """
    在验证集上计算准确率
    """
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

    accuracy = correct / total
    return accuracy


def main():
    # ==========================
    # 1. 选择运行设备
    # ==========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 40)
    print("人脸表情识别系统 - 训练脚本")
    print("=" * 40)
    print(f"当前设备：{device}")
    if torch.cuda.is_available():
        print(f"GPU 型号：{torch.cuda.get_device_name(0)}")
    print("=" * 40)

    # ==========================
    # 2. 图像预处理（含数据增强）
    # ==========================
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),          # 随机水平翻转
        transforms.RandomRotation(degrees=10),            # 随机旋转 ±10度
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),                         # 随机平移
            scale=(0.9, 1.1),                             # 随机缩放
        ),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # ==========================
    # 3. 加载数据集
    # ==========================
    print("正在加载数据集...")
    train_dataset = FER2013(
        csv_file="data/fer2013.csv",
        transform=train_transform,
        phase="Training"
    )

    val_dataset = FER2013(
        csv_file="data/fer2013.csv",
        transform=test_transform,
        phase="PublicTest"
    )

    print(f"训练集样本数：{len(train_dataset)}")
    print(f"验证集样本数：{len(val_dataset)}")

    # ==========================
    # 4. DataLoader
    # ==========================
    batch_size = 64
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )

    # ==========================
    # 5. 创建模型
    # ==========================
    model = SimpleCNN(dropout_rate=0.5).to(device)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量：{total_params:,}")
    print(f"可训练参数量：{trainable_params:,}")

    # ==========================
    # 6. 损失函数
    # ==========================
    criterion = nn.CrossEntropyLoss()

    # ==========================
    # 7. 优化器
    # ==========================
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4,  # L2 正则化
    )

    # ==========================
    # 8. 学习率调度器
    # ==========================
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",        # 监控验证集准确率
        factor=0.5,        # 学习率衰减因子
        patience=3,        # 连续 3 个 epoch 不提升则衰减
        verbose=True,
        min_lr=1e-6,
    )

    # ==========================
    # 9. 创建模型保存目录
    # ==========================
    os.makedirs("checkpoints", exist_ok=True)

    # ==========================
    # 10. 训练参数
    # ==========================
    epochs = 50
    best_acc = 0.0
    patience_counter = 0
    early_stop_patience = 10  # 连续 10 个 epoch 不提升则提前停止

    # 记录训练历史
    history = {
        "train_loss": [],
        "val_accuracy": [],
        "learning_rates": [],
    }

    # 记录训练开始时间
    training_start_time = time.time()

    print("=" * 40)
    print("开始训练...")
    print("=" * 40)

    for epoch in range(epochs):
        epoch_start_time = time.time()

        # ======================
        # 训练阶段
        # ======================
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)

            # 计算损失
            loss = criterion(outputs, labels)

            # 梯度清零
            optimizer.zero_grad()

            # 反向传播
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 更新参数
            optimizer.step()

            running_loss += loss.item()

            # 计算训练准确率
            _, predicted = torch.max(outputs, dim=1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        # ======================
        # 验证阶段
        # ======================
        val_accuracy = evaluate(model, val_loader, device)

        # ======================
        # 学习率调度
        # ======================
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_accuracy)

        # ======================
        # 记录历史
        # ======================
        history["train_loss"].append(avg_train_loss)
        history["val_accuracy"].append(val_accuracy)
        history["learning_rates"].append(current_lr)

        epoch_time = time.time() - epoch_start_time

        # ======================
        # 打印进度
        # ======================
        print(
            f"Epoch [{epoch + 1:2d}/{epochs}] "
            f"Loss: {avg_train_loss:.4f} | "
            f"Train Acc: {train_accuracy * 100:.2f}% | "
            f"Val Acc: {val_accuracy * 100:.2f}% | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # ======================
        # 保存最佳模型
        # ======================
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            patience_counter = 0

            torch.save(
                model.state_dict(),
                "checkpoints/best_model.pth"
            )

            print(f">> 🏆 保存最佳模型！验证集准确率：{best_acc * 100:.2f}%")
        else:
            patience_counter += 1

        # ======================
        # 提前停止
        # ======================
        if patience_counter >= early_stop_patience:
            print(f">> ⏹️ 提前停止训练（连续 {early_stop_patience} 个 epoch 未提升）")
            break

    # ==========================
    # 保存训练历史
    # ==========================
    total_training_time = time.time() - training_start_time
    history["best_accuracy"] = best_acc
    history["total_epochs"] = epoch + 1
    history["training_time"] = total_training_time

    with open("checkpoints/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("=" * 40)
    print(f"🎉 训练结束！")
    print(f"最佳验证集准确率：{best_acc * 100:.2f}%")
    print(f"模型已保存至：checkpoints/best_model.pth")
    print(f"训练历史已保存至：checkpoints/training_history.json")
    print(f"总训练时间：{total_training_time:.1f}s")
    print("=" * 40)


if __name__ == "__main__":
    main()
