import os
import time
import torch
import argparse
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import AIGCDataset
from net import AIGCClassificationNet, info_nce_loss
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.cuda.amp import GradScaler, autocast

# 训练参数配置
parser = argparse.ArgumentParser(description='AIGC Classification Training')
parser.add_argument('--data-root', default='../datasets', type=str)
parser.add_argument('--csv-path', default='../datasets/train.csv', type=str)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--batch-size', default=4, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--save-dir', default='./runs', type=str)
args = parser.parse_args()

# 创建输出目录
timestamp = time.strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(args.save_dir, timestamp)
os.makedirs(output_dir, exist_ok=True)

# 数据集和数据加载器
train_dataset = AIGCDataset(args.csv_path, args.data_root)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

# 初始化模型和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AIGCClassificationNet().to(device)
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()  # 分类损失
scaler = GradScaler()  # 混合精度

# 训练记录
best_loss = float('inf')
train_losses = []


def save_confusion_matrix(y_true, y_pred, epoch):
    """保存混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Epoch {epoch})')
    plt.savefig(os.path.join(output_dir, f'cm_epoch{epoch}.png'))
    plt.close()


def train(epoch):
    global best_loss
    model.train()
    total_loss = 0.0

    # 用于混淆矩阵
    all_preds = []
    all_labels = []

    for batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        # 混合精度训练
        with autocast():
            _, _, real_logits, aigc_logits = model(imgs, imgs)  # 使用相同图片作为输入
            loss = criterion(real_logits, labels)  # 分类损失

        # 反向传播
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 记录损失
        total_loss += loss.item()

        # 记录预测结果
        all_preds.extend(torch.argmax(real_logits, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(imgs)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    # 记录平均损失
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # 保存最佳模型
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))

    # 绘制损失曲线
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()

    # 保存混淆矩阵
    save_confusion_matrix(all_labels, all_preds, epoch)


if __name__ == '__main__':
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train(epoch)
        print(f'Epoch {epoch} completed in {time.time() - start_time:.2f}s')

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))
    print(f'Training complete. Results saved to: {output_dir}')