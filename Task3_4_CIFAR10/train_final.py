import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from model_seresnet import SEResNet

# 1. 解决 OMP 报错 (Windows 特有)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# ==========================================
# 核心功能函数 (Mixup & Model)
# ==========================================

def mixup_data(x, y, alpha=1.0, device='cpu'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 2 * 2, 1024)
        self.dropout1 = nn.Dropout(0.6)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.6)
        self.fc3 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 128 * 2 * 2)
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# ==========================================
# 主程序入口 (必须放在这里面)
# ==========================================
if __name__ == '__main__':
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
    ])
 
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 数据集加载 (使用多进程 num_workers=2)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # 初始化模型
    model = SEResNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # Warmup 调度器
    epochs = 40
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, epochs=epochs, 
                                                    steps_per_epoch=len(trainloader))

    # 记录器初始化 (来自 Code A 的优点)
    best_acc = 0.0
    train_losses = []
    test_accs = []
    
    print("Start Final Training (Mixup + Warmup + BestSave)...")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Mixup 数据增强
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=1.0, device=device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Mixup Loss 计算
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            
            loss.backward()
            optimizer.step()
            scheduler.step() # 每个 batch 更新学习率

            running_loss += loss.item()
        
        # 记录平均 Loss
        epoch_loss = running_loss / len(trainloader)
        train_losses.append(epoch_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'[Epoch {epoch + 1}/{epochs}] Loss: {epoch_loss:.4f} | LR: {current_lr:.5f}')

        # 验证 (来自 Code A 的优点)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        test_accs.append(acc)
        print(f'Test Accuracy: {acc:.2f} %')

        # 保存最佳模型 (来自 Code A 的优点)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'cifar10_best_final.pth')
            print(f'🌟 Best Model Saved! Accuracy: {best_acc:.2f} %')

    total_time = time.time() - start_time
    print(f'Finished Training. Total Time: {total_time:.2f}s')
    print(f'Best Accuracy: {best_acc:.2f} %')

    # 绘制曲线 (来自 Code A 的优点)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Training Loss (Mixup)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(test_accs, label='Test Accuracy', color='orange')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.savefig('training_curves_final.png')
    print("Training curves saved as 'training_curves_final.png'")
    plt.show()