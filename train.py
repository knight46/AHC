import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import argparse
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from net import MyModel


# 训练参数解析
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='runs/exp')
    parser.add_argument('--val_ratio', type=float, default=0.2)
    return parser.parse_args()


# 数据加载
def load_data(batch_size, val_ratio):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    train_dataset = ImageFolder('datasets/train', transform=train_transform)
    test_dataset = ImageFolder('datasets/test', transform=test_transform)

    # 划分验证集
    val_size = int(len(train_dataset) * val_ratio)
    train_size = len(train_dataset) - val_size
    train_set, val_set = random_split(train_dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, train_dataset.class_to_idx


# 训练函数
def train(model, device, train_loader, criterion, optimizer, epoch, writer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Train Epoch {epoch}')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})

    avg_loss = total_loss / len(train_loader)
    avg_acc = 100. * correct / total
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Acc/train', avg_acc, epoch)
    return avg_loss, avg_acc


# 验证函数
def validate(model, device, val_loader, criterion, epoch, writer):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    avg_acc = 100. * correct / total
    writer.add_scalar('Loss/val', avg_loss, epoch)
    writer.add_scalar('Acc/val', avg_acc, epoch)
    return avg_loss, avg_acc


# 测试评估
def evaluate(model, device, test_loader, class_names, save_dir):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

    # 生成分类报告
    report = classification_report(all_labels, all_preds, target_names=class_names)
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    return cm, report


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(args.save_dir)

    # 加载数据
    train_loader, val_loader, test_loader, class_dict = load_data(
        args.batch_size, args.val_ratio)
    class_names = list(class_dict.keys())

    # 初始化模型
    model = MyModel(num_classes=len(class_names), dropout=args.dropout).to(device)

    # 加载预训练权重
    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained))

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 训练循环
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(model, device, train_loader, criterion, optimizer, epoch, writer)
        val_loss, val_acc = validate(model, device, val_loader, criterion, epoch, writer)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))

        print(f'Epoch {epoch}/{args.epochs}')
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%')
        print('-' * 50)

    # 最终评估
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth')))
    evaluate(model, device, test_loader, class_names, args.save_dir)

    # 保存训练曲线
    writer.close()


if __name__ == '__main__':
    main()