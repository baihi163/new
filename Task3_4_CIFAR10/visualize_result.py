import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# 解决 OMP 冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. 必须重新定义一遍模型结构，才能加载权重
# ==========================================
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
        x = self.dropout1(x) # 预测时 dropout 不起作用，但结构要保持一致
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# ==========================================
# 2. 加载数据和模型
# ==========================================
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True, num_workers=0)

# 加载模型
model = SimpleCNN().to(device)
model.load_state_dict(torch.load('cifar10_cnn_with_dropout_bn.pth'))
model.eval() # 切换到评估模式（非常重要！）

# ==========================================
# 3. 寻找正确和错误的样本
# ==========================================
correct_samples = []
wrong_samples = []

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        for i in range(len(labels)):
            img = images[i].cpu()
            lbl = labels[i].item()
            pred = predicted[i].item()
            
            if len(correct_samples) < 5 and pred == lbl:
                correct_samples.append((img, lbl, pred))
            elif len(wrong_samples) < 5 and pred != lbl:
                wrong_samples.append((img, lbl, pred))
            
            if len(correct_samples) >= 5 and len(wrong_samples) >= 5:
                break
        if len(correct_samples) >= 5 and len(wrong_samples) >= 5:
            break

# ==========================================
# 4. 画图
# ==========================================
def imshow(img):
    img = img / 2 + 0.5     # 反归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

plt.figure(figsize=(10, 5))

# 画正确的
for i in range(5):
    plt.subplot(2, 5, i+1)
    img, lbl, pred = correct_samples[i]
    imshow(img)
    plt.title(f"True: {classes[lbl]}\nPred: {classes[pred]}", color='green', fontsize=8)
    plt.axis('off')

# 画错误的
for i in range(5):
    plt.subplot(2, 5, i+6)
    img, lbl, pred = wrong_samples[i]
    imshow(img)
    plt.title(f"True: {classes[lbl]}\nPred: {classes[pred]}", color='red', fontsize=8)
    plt.axis('off')

plt.tight_layout()
plt.savefig('predictions_visualization.png')
print("图片已保存为 predictions_visualization.png")
plt.show()
