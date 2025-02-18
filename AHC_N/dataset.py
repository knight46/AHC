import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class AIGCDataset(Dataset):
    """AI生成内容检测数据集"""

    def __init__(self, csv_path, data_root, img_size=256):
        """
        Args:
            csv_path (str): CSV文件路径
            data_root (str): 数据根目录
            img_size (int): 图像缩放尺寸
        """
        self.data_root = data_root
        self.df = pd.read_csv(csv_path)

        # 数据增强
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_root, self.df.iloc[idx]['file_name'])
        label = self.df.iloc[idx]['label']

        # 加载图像
        img = Image.open(img_path).convert('RGB')

        # 应用转换
        if self.transform:
            img = self.transform(img)

        return img, label