# 任务四：模型调优与深度思考报告 (Task 4 Report)

## 1. 实验概述 (Overview)
本任务旨在基于任务三的基础模型，通过**结构优化**（引入注意力机制）和**策略优化**（高级数据增强），在保持模型轻量化的同时，大幅提升 CIFAR-10 数据集的分类准确率。

- **基础模型**: ResNet18
- **最终模型**: **SE-ResNet18** (Squeeze-and-Excitation ResNet)
- **最终精度**: **94.55%** (Top-1 Accuracy)
- **核心策略**: Mixup Data Augmentation + Cosine Annealing LR

---

## 2. 创新点与原理 (Innovations & Principles)

### 2.1 结构优化：引入 SE-Block (Attention Mechanism)
**改动**：在 ResNet 的 BasicBlock 中嵌入了 **SE (Squeeze-and-Excitation)** 模块。

**原理分析**：
传统的卷积网络对所有通道（Channel）的特征图一视同仁。然而，并非所有特征都对分类任务同等重要。
- **Squeeze (压缩)**：通过全局平均池化（Global Average Pooling），将每个通道的空间特征压缩为一个实数，获得全局感受野。
- **Excitation (激励)**：通过两个全连接层（FC）学习每个通道的权重（Importance），生成一个权重向量（0~1之间）。
- **Scale (加权)**：将权重向量乘回原始特征图，从而“强调”重要特征，“抑制”无用特征。

**代码实现关键点**：
```python
# 核心代码片段
self.fc = nn.Sequential(
    nn.Linear(channel, channel // reduction, bias=False),
    nn.ReLU(inplace=True),
    nn.Linear(channel // reduction, channel, bias=False),
    nn.Sigmoid() # 输出 0-1 的权重
)
```

### 2.2 策略优化：Mixup 数据增强
**改动**：在训练过程中使用 Mixup 策略 (`alpha=1.0`)。

**原理分析**：
Mixup 通过线性插值的方式混合两张图片及其标签：
$$ \tilde{x} = \lambda x_i + (1 - \lambda) x_j $$
$$ \tilde{y} = \lambda y_i + (1 - \lambda) y_j $$
其中 $\lambda \sim Beta(\alpha, \alpha)$。

**优势**：
1.  **正则化效果**：迫使模型在样本之间学习线性的过渡，而不是死记硬背训练样本，极大降低了过拟合风险。
2.  **鲁棒性**：增强了模型对对抗样本和噪声的鲁棒性。

---

## 3. 性能评估 (Performance Evaluation)

根据考核要求，使用 `thop` 库对模型进行了参数量和计算量的评估，结果如下：

| 指标 (Metric) | 数值 (Value) | 评价 (Comments) |
| :--- | :--- | :--- |
| **Test Accuracy** | **94.55%** | 远超 70% 的进阶线，达到 SOTA 水平 |
| **Parameters** | **11.26 M** | 参数量极低，仅比原版 ResNet18 增加约 0.09M |
| **FLOPs** | **558.22 M** | 计算量适中，适合在边缘设备部署 |

> **结论**：SE 模块以极小的计算代价（<1% 的参数增加），换来了显著的性能提升。

---

## 4. 实验过程与复盘 (Experiments & Analysis)

### 4.1 训练曲线
![Training Curves](curves_resnet18_9487.png)


**观察**：
- **Loss 曲线**：由于使用了 Mixup，训练 Loss 并没有像传统训练那样降到接近 0，而是维持在 0.8 左右。这是正常现象，因为标签被软化了，模型永远无法以 100% 的置信度预测混合图像。
- **Accuracy 曲线**：测试集准确率稳步上升，且没有出现明显的震荡，说明学习率调度（Cosine Annealing）起到了很好的稳定作用。

### 4.2 失败复盘与挑战 (Challenges)
在实验过程中，我尝试了以下调整并获得了经验：

1.  **学习率调整**：
    - *尝试*：最初使用固定的学习率 0.1。
    - *问题*：模型在训练后期（Epoch 40+）准确率卡在 92% 左右无法提升。
    - *解决*：引入 `torch.optim.lr_scheduler.CosineAnnealingLR`，让学习率在后期衰减到 0.0001，帮助模型收敛到更优的极小值点，最终突破了 94%。

2.  **Mixup 的 Alpha 参数**：
    - *尝试*：设置 `alpha=0.2`。
    - *现象*：提升效果不明显，接近标准训练。
    - *解决*：将 `alpha` 调整为 `1.0`（均匀分布），增强了混合的强度，虽然训练初期的 Loss 变高了，但最终泛化能力更强。

---

## 5. 参考文献 (References)
1.  Hu, J., Shen, L., & Sun, G. (2018). **Squeeze-and-Excitation Networks**. CVPR.
2.  Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2017). **mixup: Beyond Empirical Risk Minimization**. ICLR.