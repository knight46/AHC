import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import argparse
import os
from net import MyModel
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def load_data(batch_size, val_ratio):

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


    train_dataset = ImageFolder('datasets/train', transform=train_transform)


    val_size = int(len(train_dataset) * val_ratio)
    train_size = len(train_dataset) - val_size
    train_set, val_set = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, train_dataset.class_to_idx


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


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(args.save_dir)

    train_loader, val_loader, class_dict = load_data(args.batch_size, args.val_ratio)
    class_names = list(class_dict.keys())

    model = MyModel(num_classes=len(class_names), dropout=args.dropout).to(device)

    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(model, device, train_loader, criterion, optimizer, epoch, writer)
        val_loss, val_acc = validate(model, device, val_loader, criterion, epoch, writer)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))

        print(f'Epoch {epoch}/{args.epochs}')
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%')
        print('-' * 50)

    torch.save(model.state_dict(), os.path.join(args.save_dir, 'final_model.pth'))


    writer.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='runs/exp')
    parser.add_argument('--val_ratio', type=float, default=0.2)
    return parser.parse_args()


if __name__ == '__main__':
    main()