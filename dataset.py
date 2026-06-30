import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image

class FER2013(Dataset):
    def __init__(self, csv_file, transform=None, phase="Training"):
        """
        csv_file: fer2013.csv路径
        transform: 图像预处理
        phase: Training / PublicTest / PrivateTest
        """

        self.data = pd.read_csv(csv_file)
        self.transform = transform

        self.data = self.data[self.data["Usage"] == phase].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        pixels = self.data.iloc[idx]["pixels"]
        label = int(self.data.iloc[idx]["emotion"])
        
        img = np.fromstring(pixels, sep=" ").reshape(48, 48).astype(np.uint8)
        
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label