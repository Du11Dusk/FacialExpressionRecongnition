import torch
from torchvision import transforms
from PIL import Image

from model import SimpleCNN


classes = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建模型
model = SimpleCNN().to(device)

# 加载模型
model.load_state_dict(
    torch.load(
        "checkpoints/best_model.pth",
        map_location=device
    )
)

model.eval()

# 图片预处理
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# 打开图片
img = Image.open("images/happy.png")

img = transform(img)

# 增加Batch维度
img = img.unsqueeze(0).to(device)

with torch.no_grad():

    output = model(img)

    prediction = torch.argmax(output, dim=1)

    print("预测结果：", classes[prediction.item()])