import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import time

# 1. 解决 OMP 报错 (你的老朋友)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 2. 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 3. 数据预处理 (保持不变)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# 4. 定义模型 (保持 SimpleCNN 不变)
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
# 🔥 核心技巧 1: Mixup 数据增强函数
# ==========================================
def mixup_data(x, y, alpha=1.0):
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

# ==========================================
# 主训练逻辑
# ==========================================
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# 🔥 核心技巧 2: Warmup (使用 OneCycleLR)
# max_lr=0.1: 最大学习率
# epochs=40: 总训练轮数
# steps_per_epoch: 每一轮有多少个 batch
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, epochs=40, 
                                                steps_per_epoch=len(trainloader))

print("Start Advanced Training (Mixup + Warmup)...")

for epoch in range(40):  # 训练 40 轮
    model.train()
    running_loss = 0.0
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # 1. 使用 Mixup 生成混合数据
        inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=1.0)
        
        # 2. 正常的前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # 3. 计算 Mixup Loss (混合后的 Loss)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        
        loss.backward()
        optimizer.step()
        
        # 4. 更新学习率 (Warmup 调度器在每个 batch 后更新)
        scheduler.step()

        running_loss += loss.item()
        
    # 打印当前轮次的 Loss 和 学习率
    current_lr = optimizer.param_groups[0]['lr']
    print(f'[Epoch {epoch + 1}] Loss: {running_loss / len(trainloader):.3f} | LR: {current_lr:.5f}')

    # 测试集验证 (验证时不需要 Mixup)
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
    print(f'Accuracy of the network on test images: {acc:.2f} %')

print('Finished Advanced Training')
torch.save(model.state_dict(), 'cifar10_advanced.pth')