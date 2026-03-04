import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 配置参数 (你可以修改这里来实验)
# ==========================================
CSV_FILE_PATH = 'data.csv'  # 请确保你的csv文件叫这个名字，且和代码在同一目录
LEARNING_RATE = 0.05        # 学习率
EPOCHS = 1000               # 总训练轮数
HIDDEN_SIZE = 20            # 隐藏层神经元数量

# ==========================================
# 2. 数据加载
# ==========================================
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        # 假设csv的第一列是x，第二列是y。如果报错，请检查csv表头
        x = df.iloc[:, 0].values.astype(np.float32).reshape(-1, 1)
        y = df.iloc[:, 1].values.astype(np.float32).reshape(-1, 1)
        
        # 转为 PyTorch 的 Tensor
        x_tensor = torch.from_numpy(x)
        y_tensor = torch.from_numpy(y)
        return x_tensor, y_tensor, x, y
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {file_path}")
        print("💡 请将群里的 csv 文件下载并重命名为 data.csv 放在同级目录下")
        exit()

# ==========================================
# 3. 模型构建 (全连接神经网络 MLP)
# ==========================================
class SimpleNet(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),  # 输入层 -> 隐藏层
            nn.ReLU(),                  # 激活函数 (关键！没有它就只是线性回归)
            nn.Linear(hidden_size, hidden_size), # 隐藏层 -> 隐藏层 (加深网络)
            nn.ReLU(),
            nn.Linear(hidden_size, 1)   # 隐藏层 -> 输出层
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 4. 主程序
# ==========================================
if __name__ == '__main__':
    # 1. 准备数据
    x_tensor, y_tensor, x_raw, y_raw = load_data(CSV_FILE_PATH)
    
    # 2. 初始化模型、损失函数、优化器
    model = SimpleNet(HIDDEN_SIZE)
    criterion = nn.MSELoss()  # 均方误差损失 (回归任务常用)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 用于记录特定 Epoch 的预测结果用于画图
    snapshots = {}
    target_epochs = [10, 100, 1000]

    # 3. 训练循环
    print(" 开始训练...")
    loss_history = []

    for epoch in range(1, EPOCHS + 1):
        # 前向传播
        prediction = model(x_tensor)
        loss = criterion(prediction, y_tensor)

        # 反向传播
        optimizer.zero_grad() # 清空梯度
        loss.backward()       # 计算梯度
        optimizer.step()      # 更新参数

        loss_history.append(loss.item())

        # 记录特定 Epoch 的结果
        if epoch in target_epochs:
            snapshots[epoch] = prediction.detach().numpy()
            print(f"Epoch {epoch}/{EPOCHS}, Loss: {loss.item():.6f}")
        
        if epoch % 100 == 0 and epoch not in target_epochs:
            print(f"Epoch {epoch}/{EPOCHS}, Loss: {loss.item():.6f}")

    # ==========================================
    # 5. 可视化对比 (任务要求)
    # ==========================================
    plt.figure(figsize=(12, 5))

    # 图1: 拟合曲线对比
    plt.subplot(1, 2, 1)
    plt.scatter(x_raw, y_raw, s=10, c='gray', label='Ground Truth', alpha=0.5)
    
    colors = {10: 'green', 100: 'orange', 1000: 'red'}
    for epoch, pred in snapshots.items():
        # 为了画图好看，按 x 排序
        sorted_indices = x_raw.flatten().argsort()
        plt.plot(x_raw.flatten()[sorted_indices], pred.flatten()[sorted_indices], 
                 color=colors.get(epoch, 'blue'), linewidth=2, label=f'Epoch={epoch}')
    
    plt.title(f"Function Fitting (LR={LEARNING_RATE})")
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')

    # 图2: Loss 变化
    plt.subplot(1, 2, 2)
    plt.plot(loss_history)
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale('log') # 使用对数坐标看 Loss 下降更清晰

    plt.tight_layout()
    plt.show()
    
    print("\n✅ 训练完成！请查看弹出的图像。")