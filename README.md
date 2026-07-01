# 😊 人脸表情识别系统 (Facial Expression Recognition)

基于 **PyTorch** 深度学习框架的人脸表情识别系统，支持 **7 种表情分类**，当前已整理为一个可直接使用的 Web 应用，提供上传图片和摄像头拍照两种识别方式。

## 📋 目录

- [功能特点](#-功能特点)
- [支持的 7 种表情](#-支持的-7-种表情)
- [技术栈](#-技术栈)
- [环境要求](#-环境要求)
- [快速开始](#-快速开始)
- [使用方法](#-使用方法)
  - [1. Web 界面（推荐）](#1-web-界面推荐)
  - [2. 训练模型](#2-训练模型)
  - [3. 测试模型](#3-测试模型)
  - [4. 单张图片预测](#4-单张图片预测)
- [项目结构](#-项目结构)
- [模型架构](#-模型架构)
- [训练细节](#-训练细节)
- [性能指标](#-性能指标)
- [常见问题](#-常见问题)
- [许可证](#-许可证)

## ✨ 功能特点

- ✅ **摄像头实时识别** - 实时检测并识别摄像头画面中的人脸表情
- ✅ **上传图片识别** - 支持 JPG/PNG/BMP/WEBP 格式，可识别多人脸
- ✅ **实时统计图表** - 显示各表情出现频率和趋势（Altair 图表）
- ✅ **识别历史记录** - 保存上传图片的识别结果
- ✅ **概率分布展示** - 显示每个表情的置信度分布
- ✅ **置信度颜色编码** - 高/中/低置信度分别用绿/黄/红色框显示
- ✅ **数据增强训练** - 随机翻转、旋转、平移、缩放
- ✅ **学习率调度** - ReduceLROnPlateau 自适应学习率衰减
- ✅ **提前停止** - 防止过拟合

## 🎭 支持的 7 种表情

| 表情 | 英文 | Emoji |
|------|------|-------|
| 愤怒 | Angry | 😠 |
| 厌恶 | Disgust | 🤢 |
| 恐惧 | Fear | 😨 |
| 开心 | Happy | 😊 |
| 悲伤 | Sad | 😢 |
| 惊讶 | Surprise | 😲 |
| 中性 | Neutral | 😐 |

## 🛠 技术栈

- **深度学习框架**: PyTorch 2.0+
- **模型架构**: 自定义 CNN（3层卷积 + 批归一化 + Dropout）
- **人脸检测**: OpenCV Haar Cascade 分类器
- **前端界面**: Streamlit
- **数据可视化**: Altair + Pandas
- **数据集**: FER2013（35,887 张 48x48 灰度人脸图像）

## 📦 环境要求

- Python 3.8+
- PyTorch 2.0+
- 可选：CUDA 支持的 GPU（推荐）

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/Du11Dusk/FacialExpressionRecongnition.git
cd FacialExpressionRecongnition
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 下载数据集

从 [Kaggle FER2013](https://www.kaggle.com/datasets/msambare/fer2013) 下载数据集，将 `fer2013.csv` 放入 `data/` 目录。

### 4. 训练模型

```bash
python train.py
```

### 5. 启动 Web 界面

```bash
streamlit run app.py
```

打开浏览器访问 `http://localhost:8501`

## 📖 使用方法

### 1. Web 界面（推荐）

```bash
streamlit run app.py
```

Web 界面提供 4 个功能页面：

- **📤 上传图片识别** - 上传图片进行人脸表情识别
- **📷 摄像头实时识别** - 使用摄像头实时识别表情
- **📜 识别历史记录** - 查看之前的识别结果
- **ℹ️ 关于系统** - 系统信息和设备状态

### 2. 训练模型

```bash
python train.py
```

训练参数（可在 `train.py` 中修改）：
- `epochs`: 最大训练轮数（默认 50）
- `batch_size`: 批次大小（默认 64）
- `learning_rate`: 初始学习率（默认 0.001）
- `early_stop_patience`: 提前停止耐心值（默认 10）

训练过程中会自动：
- 保存最佳模型到 `checkpoints/best_model.pth`
- 记录训练历史到 `checkpoints/training_history.json`
- 使用数据增强提升泛化能力
- 自适应调整学习率

### 3. 测试模型

```bash
python test.py
```

输出包括：
- 总体准确率
- 各类别准确率
- 混淆矩阵
- 宏平均精确率、召回率、F1 分数

### 4. 单张图片预测

```bash
# 预测默认图片
python predict.py

# 预测指定图片
python predict.py images/happy.png
```

## 📁 项目结构

```
FacialExpressionRecongnition/
├── app.py              # Streamlit Web 应用主程序
├── model.py            # CNN 模型定义
├── train.py            # 训练脚本
├── test.py             # 测试评估脚本
├── predict.py          # 单张图片预测脚本
├── dataset.py          # FER2013 数据集加载器
├── detect_face.py      # 人脸检测测试脚本
├── checkdata.py        # 数据检查脚本
├── requirements.txt    # Python 依赖
├── README.md           # 项目文档
├── .gitignore          # Git 忽略规则
├── checkpoints/        # 模型保存目录
│   ├── best_model.pth  # 最佳模型权重
│   └── training_history.json  # 训练历史记录
├── data/               # 数据集目录（需自行下载）
│   └── fer2013.csv     # FER2013 数据集
└── images/             # 测试图片目录
    └── happy.png       # 示例图片
```

## 🧠 模型架构

```
SimpleCNN(
  (conv1): Conv2d(1, 32, kernel_size=3, padding=1)
  (bn1): BatchNorm2d(32)
  (relu): ReLU()
  (pool): MaxPool2d(2)        # 48x48 -> 24x24
  
  (conv2): Conv2d(32, 64, kernel_size=3, padding=1)
  (bn2): BatchNorm2d(64)
  (relu): ReLU()
  (pool): MaxPool2d(2)        # 24x24 -> 12x12
  
  (conv3): Conv2d(64, 128, kernel_size=3, padding=1)
  (bn3): BatchNorm2d(128)
  (relu): ReLU()
  (pool): MaxPool2d(2)        # 12x12 -> 6x6
  
  (dropout): Dropout(p=0.5)
  (fc1): Linear(4608, 256)
  (fc2): Linear(256, 7)
)
```

**总参数量**: ~1.2M

## 🔧 训练细节

- **数据增强**: 随机水平翻转、±10° 旋转、±10% 平移、±10% 缩放
- **优化器**: Adam (lr=0.001, weight_decay=1e-4)
- **学习率调度**: ReduceLROnPlateau (factor=0.5, patience=3)
- **梯度裁剪**: max_norm=1.0
- **提前停止**: patience=10 epochs
- **损失函数**: CrossEntropyLoss
- **批大小**: 64

## 📊 性能指标

| 指标 | 值 |
|------|-----|
| 总体准确率 | ~65% (FER2013 PrivateTest) |
| 宏平均精确率 | ~65% |
| 宏平均召回率 | ~65% |
| 宏平均 F1 分数 | ~65% |

> **注意**: FER2013 数据集本身难度较大（人类识别准确率约 65%），当前模型性能已达到基准水平。可通过以下方式进一步提升：
> - 使用更深的网络架构（如 ResNet、VGG）
> - 使用更大的数据集（如 AffectNet、RAF-DB）
> - 使用人脸关键点对齐预处理
> - 集成学习

## ❓ 常见问题

### Q: 模型文件不存在？
A: 请先运行 `python train.py` 训练模型。

### Q: 摄像头无法打开？
A: 检查摄像头连接，确保没有其他程序占用摄像头。

### Q: 识别准确率不高？
A: 可以尝试：
1. 增加训练轮数
2. 调整学习率
3. 使用更深的网络结构
4. 使用更大的数据集

### Q: 如何切换到 GPU 训练？
A: 确保安装了 CUDA 版本的 PyTorch，程序会自动检测并使用 GPU。

## 📄 许可证

本项目仅供学习和研究使用。
