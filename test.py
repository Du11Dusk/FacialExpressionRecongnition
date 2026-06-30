import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import FER2013
from model import SimpleCNN


def main():

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("当前设备：", device)

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 加载测试集（PublicTest）
    test_dataset = FER2013(
        csv_file="data/fer2013.csv",
        transform=transform,
        phase="PublicTest"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )

    # 创建模型
    model = SimpleCNN().to(device)

    # 加载训练好的参数
    model.load_state_dict(
        torch.load(
            "checkpoints/best_model.pth",
            map_location=device
        )
    )

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, dim=1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    print("=" * 40)
    print("测试集准确率：{:.2f}%".format(accuracy * 100))
    print("=" * 40)


if __name__ == "__main__":
    main()