import os
import time
import argparse
import csv
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
import timm
import matplotlib.pyplot as plt

class CustomDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.data = []
        self.root_dir = root_dir
        self.transform = transform
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                try:
                    idx = int(row[0])
                    img_path = row[1]
                    label = int(row[2])
                    self.data.append((img_path, label))
                except Exception as e:
                    continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        full_path = os.path.join(self.root_dir, img_path)
        image = Image.open(full_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def main():
    # 解析命令令行参数
    parser = argparse.ArgumentParser(description='VIT Classification Training')
    parser.add_argument('--data-root', default='../datasets', type=str)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    args = parser.parse_args()

    runs_dir = 'runs'
    os.makedirs(runs_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d%H", time.localtime())
    output_dir = os.path.join(runs_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = CustomDataset(csv_file=os.path.join(args.data_path, 'train.csv'),
                            root_dir=args.data_path, transform=transform)

    val_size = int(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, 2)

    for name, param in model.named_parameters():
        if "head" not in name:
            param.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    train_losses = []

    # 训练过程
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / train_size
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{args.epochs}: Loss {epoch_loss:.4f}")

    model_path = os.path.join(output_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"模型权重已保存至: {model_path}")

    plt.figure()
    plt.plot(range(1, args.epochs + 1), train_losses, marker='o', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'epoch_loss.png'))
    plt.close()
    print(f"Loss曲线图已保存至: {os.path.join(output_dir, 'epoch_loss.png')}")

if __name__ == '__main__':
    main()
