import torch
import torch.nn as nn


class SimpleCNN(nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 7)

    def forward(self, x):

        # 第一层
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # 第二层
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接
        x = self.fc1(x)
        x = self.relu(x)

        # 输出
        x = self.fc2(x)

        return x