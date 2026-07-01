"""
单张图片预测脚本
用法：python predict.py [图片路径]
默认：images/happy.png
"""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from model import SimpleCNN


EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
EMOTION_EMOJIS = ["😠", "🤢", "😨", "😊", "😢", "😲", "😐"]


def predict_image(image_path):
    """对单张图片进行表情识别"""
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"❌ 文件不存在：{image_path}")
        return

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前设备：{device}")

    # 创建模型
    model = SimpleCNN().to(device)

    # 加载模型
    model_path = "checkpoints/best_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在：{model_path}")
        print("💡 请先运行 python train.py 训练模型")
        return

    try:
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        return

    model.eval()

    # 图片预处理
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
    ])

    # 打开图片
    try:
        img = Image.open(image_path)
        print(f"📷 图片：{image_path}")
        print(f"📐 尺寸：{img.size}")
    except Exception as e:
        print(f"❌ 图片打开失败：{e}")
        return

    # 预处理
    input_tensor = transform(img)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        all_probs = probabilities[0].cpu().numpy()

    # 输出结果
    print()
    print("=" * 40)
    print("📊 识别结果")
    print("=" * 40)
    print(f"预测表情：{EMOTION_EMOJIS[predicted_class]} {EMOTIONS[predicted_class]}")
    print(f"置信度：{confidence * 100:.2f}%")
    print()

    # 显示所有类别的概率
    print("各类别概率：")
    print("-" * 30)
    sorted_indices = np.argsort(all_probs)[::-1]
    for i in sorted_indices:
        bar = "█" * int(all_probs[i] * 20) + "░" * (20 - int(all_probs[i] * 20))
        print(f"{EMOTIONS[i]:10s}: {all_probs[i] * 100:5.2f}% {bar}")
    print("=" * 40)


if __name__ == "__main__":
    # 支持命令行参数
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "images/happy.png"
        print("💡 提示：可以使用 python predict.py <图片路径> 指定图片")
        print()

    predict_image(image_path)
