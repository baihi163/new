cat <<EOF > README.md
# 📘 Deep Learning Practice (Tasks 2-5)

这是我的深度学习入门实践项目仓库。主要包含从基础神经网络到 ResNet 以及行人识别的完整训练代码。

本项目主要通过 **AI 辅助教学** 完成，旨在理解深度学习的核心概念与代码实现。

## 📂 项目结构 (Project Structure)

### Task 2: 神经网络基础 (Neural Network Basics)
- 初步了解神经网络的反向传播与梯度下降。
- 实现基础的分类模型。

### Task 3 & 4: CIFAR-10 图像分类 (CIFAR-10 Classification)
- **核心内容**：使用卷积神经网络 (CNN) 和 ResNet 对 CIFAR-10 数据集进行分类。
- **最终代码 (Final Code)**：
  - 请运行 \`new/Task3_4_CIFAR10/train_resnet_all_in_one.py\` (或 \`main_task4.py\`)。
  - 该脚本集成了完整的训练、验证和测试流程。
- **实验记录**：包含了不同学习率、Batch Size 下的 Loss 曲线对比。

### Task 5: 行人识别 (Pedestrian Detection/Classification)
- **核心内容**：针对特定场景（行人）的目标识别任务。
- 位于 \`Task5_Pedestrian\` 目录下。

---

## 💡 学习心得与技术难点 (Learning Journey)

这是我**第一次**完整地训练神经网络。虽然过程中遇到了不少挑战，但通过 AI 的教学和一步步调试，最终看着 Loss 曲线下降，非常有成就感！

### 🛠️ Git 版本控制实战
在项目提交过程中，我遇到了棘手的 **Git 远程分支冲突 (Merge Conflicts)**。
- **问题**：本地代码与远程仓库版本不一致，导致 push 失败。
- **解决**：通过查阅资料，我成功掌握了标准的协作流程：
  1. \`git pull\` (拉取远程代码)
  2. 手动解决文件冲突
  3. \`git add\` & \`git commit\` (合并提交)
  4. \`git push\` (推送到远程)

这次经历让我不仅入门了深度学习，也掌握了程序员必备的版本控制技能。

---

*Last Updated: $(date "+%Y-%m-%d")*
EOF

git add README.md
git commit -m "Docs: 更新README，总结Task2-5及Git解决冲突的心得"
git push
