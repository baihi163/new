import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# 确保这里导入的是 model_seresnet 文件里的 SEResNet18 函数
from model_seresnet import SEResNet18 
import time
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 参数设置
# ==========================================
BATCH_SIZE = 128
LR = 0.1
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. Mixup 工具函数
# ==========================================
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ==========================================
# 3. 主程序 (必须有这一段才会运行！)
# ==========================================
if __name__ == '__main__':
    print(f"Using device: {DEVICE}")  # <--- 如果代码正确，你第一眼应该看到这句话

    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print("Building SE-ResNet18 model...")
    # 初始化模型
    model = SEResNet18(num_classes=10).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    train_losses = []
    test_accs = []
    best_acc = 0.0
    start_time = time.time()

    print("Start SE-ResNet Training (with Mixup)...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Mixup Forward
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=1.0)
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        # Test Phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        acc = 100 * correct / total
        avg_loss = running_loss / len(trainloader)
        
        train_losses.append(avg_loss)
        test_accs.append(acc)

        print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} | LR: {current_lr:.5f}")
        print(f"Test Accuracy: {acc:.2f} %")
        
        if acc > best_acc:
            best_acc = acc
            print(f"🌟 Best SE-ResNet Saved! Accuracy: {best_acc:.2f} %")
            torch.save(model.state_dict(), 'seresnet_best.pth')

    total_time = time.time() - start_time
    print(f"Finished Training. Total Time: {total_time:.2f}s")
    print(f"Best Accuracy: {best_acc:.2f} %")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(test_accs, label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.legend()

    plt.savefig('curves_seresnet.png')
    print("Training curves saved as curves_seresnet.png")