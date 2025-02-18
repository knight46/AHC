import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from net import AIGCClassificationNet, info_nce_loss


class PairedDataset(Dataset):
    """处理配对数据的自定义数据集"""

    def __init__(self, pairs, root_dir, transform=None):
        self.pairs = pairs
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        real_img = self._load_image(pair['real'])
        aigc_img = self._load_image(pair['aigc'])
        return real_img, aigc_img

    def _load_image(self, filename):
        # filename = str(filename).split('/')[-1]
        image = plt.imread(os.path.join(self.root_dir, filename))
        if image.shape[-1] == 4:  # 处理RGBA图像
            image = image[..., :3]
        image = transforms.ToPILImage()(image)
        if self.transform:
            image = self.transform(image)
        return image


def create_pairs(csv_path):
    """从CSV创建配对数据"""
    df = pd.read_csv(csv_path)
    pairs = []

    # 创建配对字典
    pair_dict = {}
    for _, row in df.iterrows():
        filename = os.path.splitext(row['file_name'])[0]
        if filename not in pair_dict:
            pair_dict[filename] = {'real': None, 'aigc': None}

        if row['label'] == 0:
            pair_dict[filename]['real'] = row['file_name']
        else:
            pair_dict[filename]['aigc'] = row['file_name']

    # 生成有效配对
    for pair in pair_dict.values():
        if pair['real'] or pair['aigc']:
            pairs.append(pair)
    return pairs


def get_transforms(img_size=256):
    """数据增强配置"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def plot_loss_curve(losses, save_path):
    """绘制损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def main(args):
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建配对数据
    pairs = create_pairs(args.csv_path)

    # 数据转换
    transform = get_transforms(args.img_size)

    # 数据集和数据加载器
    dataset = PairedDataset(pairs, args.data_root, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4, pin_memory=True)

    # 模型初始化
    model = AIGCClassificationNet(embed_dim=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()

    # 训练记录
    best_loss = float('inf')
    all_losses = []


    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for real_imgs, aigc_imgs in loader:
            real_imgs = real_imgs.to(device)
            aigc_imgs = aigc_imgs.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                real_emb, aigc_emb = model(real_imgs, aigc_imgs)
                loss = info_nce_loss(real_emb, aigc_emb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * real_imgs.size(0)

        # 计算平均损失
        avg_loss = epoch_loss / len(dataset)
        all_losses.append(avg_loss)

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_model.pth')

        print(f'Epoch {epoch + 1}/{args.epochs} | Loss: {avg_loss:.4f}')

    # 保存最终模型
    torch.save(model.state_dict(), 'final_model.pth')

    # 绘制损失曲线
    plot_loss_curve(all_losses, 'train_loss_curve.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AIGC Detection Training')
    parser.add_argument('--data_root', type=str, default='../datasets/train',
                        help='Root directory of dataset')
    parser.add_argument('--csv_path', type=str, default='../datasets/train.csv',
                        help='Path to training CSV file')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Input batch size for training')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Input image size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')

    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs('results', exist_ok=True)

    main(args)
    #/home/azathothlxl/Experiments/AHC/datasets