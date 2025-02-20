import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class AIGCDataset(Dataset):
    """
    每个样本返回一对图片：
      - real_img: 真实图片（label 0）
      - aigc_img: AI生成图片（label 1）
    假设 CSV 文件中每两行构成一对，且文件格式如下：
      index,file_name,label
      0,train_data/xxx.jpg,1
      1,train_data/yyy.jpg,0
      2,train_data/zzz.jpg,1
      3,train_data/www.jpg,0
      ...
    """

    def __init__(self, csv_path, data_root, img_size=256):
        self.data_root = data_root
        self.df = pd.read_csv(csv_path)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        # 检查行数是否为偶数
        if len(self.df) % 2 != 0:
            raise ValueError("CSV 文件中的行数必须为偶数，每对样本包含一张真实图片和一张 AI 生成图片。")

    def __len__(self):
        return len(self.df) // 2

    def __getitem__(self, idx):
        # 取连续两行数据作为一对
        row1 = self.df.iloc[2 * idx]
        row2 = self.df.iloc[2 * idx + 1]

        # 根据 label 判断哪行是真实图片（label=0）哪行是 AI 图片（label=1）
        if row1['label'] == 0 and row2['label'] == 1:
            real_row, aigc_row = row1, row2
        elif row1['label'] == 1 and row2['label'] == 0:
            real_row, aigc_row = row2, row1
        else:
            raise ValueError(f"索引 {2 * idx} 与 {2 * idx + 1} 这两行数据没有形成一对（应包含一个真实、一张 AI 生成）。")

        real_img_path = os.path.join(self.data_root, real_row['file_name'])
        aigc_img_path = os.path.join(self.data_root, aigc_row['file_name'])

        # 加载图片
        real_img = Image.open(real_img_path).convert('RGB')
        aigc_img = Image.open(aigc_img_path).convert('RGB')

        if self.transform:
            real_img = self.transform(real_img)
            aigc_img = self.transform(aigc_img)

        # 返回两张图片及各自标签（真实图片标签为 0，AI 图片标签为 1）
        return real_img, aigc_img, real_row['label'], aigc_row['label']
