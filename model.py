import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    简单 CNN 模型 - 用于 FER2013 人脸表情识别
    输入: 1x48x48 灰度图
    输出: 7 类表情
    """

    def __init__(self, dropout_rate=0.5):
        super(SimpleCNN, self).__init__()

        # 第一层卷积: 1 -> 32
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)  # 48 -> 24

        # 第二层卷积: 32 -> 64
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)
        # 第二次池化: 24 -> 12

        # 第三层卷积: 64 -> 128
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.bn3 = nn.BatchNorm2d(128)
        # 第三次池化: 12 -> 6

        # Dropout 层
        self.dropout = nn.Dropout(p=dropout_rate)

        # 全连接层
        # 经过三次池化: 48/2/2/2 = 6, 所以 128*6*6 = 4608
        self.fc1 = nn.Linear(
            in_features=128 * 6 * 6,
            out_features=256
        )
        self.fc2 = nn.Linear(
            in_features=256,
            out_features=7
        )

    def forward(self, x):
        # conv1 + bn + relu + pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        # conv2 + bn + relu + pool
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        # conv3 + bn + relu + pool
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        # 展平
        x = x.view(x.size(0), -1)

        # Dropout + 全连接层
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
