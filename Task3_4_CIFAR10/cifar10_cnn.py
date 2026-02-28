import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 设备设置：如果有 GPU 就用 GPU，否则用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理：对 CIFAR-10 图像进行标准化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载 CIFAR-10 数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# 类别名称，用于后续显示
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 定义 SimpleCNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # 卷积层1：输入3通道(RGB)，输出16个特征图
        self.pool = nn.MaxPool2d(2, 2)               # 池化层：尺寸减半
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1) # 卷积层2：输入16，输出32个特征图
        self.fc1 = nn.Linear(32 * 8 * 8, 512)        # 全连接层1：展平后综合特征
        self.fc2 = nn.Linear(512, 256)               # 全连接层2
        self.fc3 = nn.Linear(256, 10)                # 输出层：10个类别
        self.relu = nn.ReLU()                        # 激活函数：引入非线性
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 卷积+激活+池化，32x32->16x16
        x = self.pool(self.relu(self.conv2(x)))  # 再次卷积+激活+池化，16x16->8x8
        x = x.view(-1, 32 * 8 * 8)              # 展平特征图
        x = self.relu(self.fc1(x))               # 全连接层处理
        x = self.relu(self.fc2(x))
        x = self.fc3(x)                          # 输出分类结果
        return x

# 实例化模型、损失函数和优化器
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，适合分类任务
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器

# 记录训练过程中的损失和准确率
train_losses = []
train_accuracies = []
test_accuracies = []

if __name__ == '__main__':
    # 训练模型
    num_epochs = 10  # 训练 10 个 epoch
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()  # 清空梯度
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 200 == 199:  # 每 200 个批次打印一次
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 200:.3f}')
                running_loss = 0.0
        
        train_loss = running_loss / len(trainloader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        print(f'Epoch {epoch + 1} Train Accuracy: {train_acc:.2f}%')
        
        # 测试阶段
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
        
        test_acc = 100 * correct / total
        test_accuracies.append(test_acc)
        print(f'Epoch {epoch + 1} Test Accuracy: {test_acc:.2f}%')

    print('训练完成！')

    # 绘制损失和准确率曲线
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 测试集上可视化一些预测结果
    def imshow(img):
        img = img / 2 + 0.5  # 反标准化
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')

    model.eval()
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # 显示前 5 张图像的预测结果
    plt.figure(figsize=(10, 2))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        imshow(images[i].cpu())
        plt.title(f'Pred: {classes[predicted[i]]}\nTrue: {classes[labels[i]]}')
    plt.show()
