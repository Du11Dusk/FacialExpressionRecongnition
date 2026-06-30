import os

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
    print("当前设备：", device)
    print("=" * 40)

    # ==========================
    # 2. 图像预处理
    # ==========================
    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # ==========================
    # 3. 加载训练集
    # ==========================
    train_dataset = FER2013(
        csv_file="data/fer2013.csv",
        transform=train_transform,
        phase="Training"
    )

    # ==========================
    # 4. 加载验证集
    # ==========================
    val_dataset = FER2013(
        csv_file="data/fer2013.csv",
        transform=test_transform,
        phase="PublicTest"
    )

    # ==========================
    # 5. DataLoader
    # ==========================
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False
    )

    # ==========================
    # 6. 创建模型
    # ==========================
    model = SimpleCNN().to(device)

    # ==========================
    # 7. 损失函数
    # ==========================
    criterion = nn.CrossEntropyLoss()

    # ==========================
    # 8. 优化器
    # ==========================
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # ==========================
    # 9. 创建模型保存目录
    # ==========================
    os.makedirs("checkpoints", exist_ok=True)

    # ==========================
    # 10. 开始训练
    # ==========================
    epochs = 10

    best_acc = 0.0

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

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

            # 更新参数
            optimizer.step()

            running_loss += loss.item()

        # ======================
        # 验证
        # ======================
        acc = evaluate(model, val_loader, device)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {running_loss:.4f} "
            f"Accuracy: {acc * 100:.2f}%"
        )

        # ======================
        # 保存最佳模型
        # ======================
        if acc > best_acc:
            best_acc = acc

            torch.save(
                model.state_dict(),
                "checkpoints/best_model.pth"
            )

            print(">> 已保存最佳模型！")

    print("=" * 40)
    print(f"训练结束！最佳准确率：{best_acc * 100:.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    main()