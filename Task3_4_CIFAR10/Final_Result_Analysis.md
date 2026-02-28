## 🏆 最终训练结果 (Final Training Results)

通过引入 **Label Smoothing (标签平滑)** 和 **Random Erasing (随机擦除)** 策略，模型性能取得了显著突破。

### 1. 性能指标
- **最佳测试集准确率 (Best Test Accuracy)**: `81.61%` 🚀
- **总训练时长**: 约 14 分钟 (860s)
- **Epochs**: 40

### 2. 训练日志分析
从训练日志中可以观察到以下现象：
1.  **Loss 值维持较高 (1.5~1.6)**：这是由于使用了 Label Smoothing 和 Mixup。软标签导致 CrossEntropy Loss 的理论下界不再是 0，这是正则化生效的标志。
2.  **后期显著提升**：在 Epoch 35 之后，随着学习率衰减至 0.01 以下，模型准确率从 78% 跃升至 81.6%，证明了 Learning Rate Scheduler 在收敛阶段的重要性。
3.  **无过拟合迹象**：Test Accuracy 持续上升直至训练结束，说明 Random Erasing 有效增强了模型的泛化能力。

### 3. 结果可视化
*(此处插入你的 training_curves_final.png 图片)*
![Training Curves](training_curves_final.png)
