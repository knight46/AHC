import os
import time
import torch
import argparse
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import AIGCDataset
from net import AIGCClassificationNet, supervised_contrastive_loss
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.cuda.amp import GradScaler, autocast

parser = argparse.ArgumentParser(description='AIGC Classification Training')
parser.add_argument('--data-root', default='../datasets', type=str)
parser.add_argument('--csv-path', default='../datasets/train.csv', type=str)
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--batch-size', default=6, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--save-dir', default='./runs', type=str)
args = parser.parse_args()

timestamp = time.strftime("%Y%m%d_%H")
output_dir = os.path.join(args.save_dir, timestamp)
os.makedirs(output_dir, exist_ok=True)

train_dataset = AIGCDataset(args.csv_path, args.data_root)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AIGCClassificationNet().to(device)
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()
lambda_coef = 0.5

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

    all_preds = []
    all_labels = []

    for batch_idx, (real_img, aigc_img, real_labels, aigc_labels) in enumerate(train_loader):
        real_img = real_img.to(device)
        aigc_img = aigc_img.to(device)
        real_labels = real_labels.to(device)
        aigc_labels = aigc_labels.to(device)

        with autocast():
            real_emb, aigc_emb, real_logits, aigc_logits = model(real_img, aigc_img)
            contrastive_loss = supervised_contrastive_loss(real_emb, aigc_emb, temperature=0.1)
            cls_loss = (criterion(real_logits, real_labels) + criterion(aigc_logits, aigc_labels)) / 2
            loss = (1-lambda_coef)*cls_loss + lambda_coef * contrastive_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        all_preds.extend(torch.argmax(real_logits, dim=1).cpu().numpy())
        all_labels.extend(real_labels.cpu().numpy())

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(real_img*2)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}    cls_loss: {cls_loss.item():.6f}    contrastive_loss: {contrastive_loss.item():.6f}')

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))

    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()

    save_confusion_matrix(all_labels, all_preds, epoch)


if __name__ == '__main__':
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train(epoch)
        print(f'Epoch {epoch} completed in {time.time() - start_time:.2f}s')

    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))
    print(f'Training complete. Results saved to: {output_dir}')