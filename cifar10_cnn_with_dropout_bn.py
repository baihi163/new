import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR  # 导入学习率调度器

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义改进的 CNN 模型，包含 Dropout 和 BatchNorm
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
        self.fc1 = nn.Linear(128 * 2 * 2, 1024)  # 增加维度
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

# 数据加载
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# 定义数据加载器
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

# 模型、损失函数和优化器
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 定义学习率调度器
scheduler = StepLR(optimizer, step_size=3, gamma=0.7)  # 每3个epoch学习率减少到70%

# 训练循环，增加轮数并加入早停机制
best_acc = 0.0
patience = 5  # 容忍的epoch数，连续5个epoch测试准确率无提升则停止
counter = 0
num_epochs = 35  # 增加到35个epoch
train_losses, train_accs, test_accs = [], [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    epoch_loss_sum = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        epoch_loss_sum += loss.item() # <--- 新增：累加所有Loss
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if i % 200 == 199:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch {i+1}, Loss: {running_loss / 200:.3f}')
            running_loss = 0.0
    train_loss = epoch_loss_sum / len(trainloader)
    train_acc = 100 * correct / total if total > 0 else 0
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%')
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = 100 * correct / total if total > 0 else 0
    test_accs.append(test_acc)
    print(f'Epoch [{epoch+1}/{num_epochs}], Test Acc: {test_acc:.2f}%')
    
    # 早停机制：如果测试准确率提升，则保存模型并重置计数器；否则增加计数器
    if test_acc > best_acc:
        best_acc = test_acc
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth')  # 保存最佳模型
        print(f"Best model saved with Test Acc: {best_acc:.2f}%")
    else:
        counter += 1
        print(f"No improvement for {counter}/{patience} epochs")
        if counter >= patience:
            print("Early stopping triggered")
            break
    
    # 更新学习率并打印当前学习率
    scheduler.step()
    print(f'Current Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

# 保存最终模型
torch.save(model.state_dict(), 'cifar10_cnn_with_dropout_bn.pth')
print('Final model saved to cifar10_cnn_with_dropout_bn.pth')

# 绘制曲线并保存
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(test_accs, label='Test Acc')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.savefig('training_curves_with_dropout_bn.png')  # 保存图像
plt.show()
