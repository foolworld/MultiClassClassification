import torch
import torch.nn as nn
import torch.optim as optim
from pandas.core.config_init import val_mca
from torch import ScriptDictIterator
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from PIL import Image

train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

val_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class LeafCNN(nn.Module):
    def __init__(self,num_classes=8):
        super(LeafCNN, self).__init__()
        #conv1:3-》12
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn1=nn.BatchNorm2d(12)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        #conv2:12->24
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.bn2=nn.BatchNorm2d(24)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=1)
        self.bn3=nn.BatchNorm2d(48)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(48 * 8 * 8, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x=self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x=self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x=self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.relu4(self.bn4(self.fc1(x)))  # fc1在前
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_data = datasets.ImageFolder('../data/dataset1/train', transform=train_transform)
    val_data = datasets.ImageFolder('../data/dataset1/val', transform=val_transform)

    print(f"训练集：{len(train_data)}张，验证集：{len(val_data)}张")
    print(f'类别：{train_data.classes}')

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=True)

    model = LeafCNN(num_classes=len(train_data.classes)).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-4)

    scheduler =  optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = 150)

    writer = SummaryWriter('./logs/leaf_pytorch')

    epochs = 300
    best_acc = 0

    for epoch in range(epochs):

        model.train()
        train_loss =0
        train_correct =0

        for data,target in train_loader:
            data,target = data.to(device),target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (output.argmax(1)==target).sum().item()

        model.eval()
        val_loss = 0
        val_correct = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                val_correct += (output.argmax(1) == target).sum().item()

        train_acc = 100. * train_correct / len(train_data)
        val_acc = 100. * val_correct / len(val_data)
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}: '
                  f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Acc/val', val_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'classes': train_data.classes
            }, 'leaf_best_model.pth')
            print(f'   ✅ 保存最佳模型，准确率: {best_acc:.2f}%')

    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': train_data.classes
    }, 'leaf_final_model.pth')

    print(f'\n训练完成！最佳验证准确率: {best_acc:.2f}%')
    writer.close()

if __name__ == '__main__':
    main()