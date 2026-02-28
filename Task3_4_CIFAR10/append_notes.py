import os

# 这里的笔记内容对应 Task 5 的“笔记必答”要求
notes_content = """

## 7. 深度思考（笔记必答）

### 7.1 分割任务的数据处理 (Mask) 与分类任务 (Label) 有什么不同？
在本次实验中，我深刻体会到实例分割与传统分类任务在数据结构上的巨大差异：
*   **维度不同**：
    *   **分类任务 (Label)**：标签通常只是一个标量整数（如 `1` 代表行人），数据量极小。
    *   **分割任务 (Mask)**：标签是一个与图像空间尺寸对应的二维矩阵 $(H, W)$。对于 Mask R-CNN，如果有 $N$ 个目标，Mask 的形状甚至是 $(N, H, W)$。
*   **信息密度**：Mask 包含了目标的**形状**和**拓扑结构**信息，是像素级的密集预测（Dense Prediction），而 Label 仅是图像级或框级的稀疏预测。
*   **处理逻辑**：Label 通常只需要做 One-hot 编码；而 Mask 需要进行几何变换（如 Resize, Flip），且必须保证变换参数与原图严格一致（Geometric Alignment），否则会导致 Mask 错位。

### 7.2 数据处理时的难点与解决
在处理 PennFudanPed 数据集时，我遇到了以下挑战：
1.  **Mask 的多实例解析**：
    *   **难点**：原始 Mask 图像是一张 PNG 图片，不同的像素值（如 1, 2, 3...）代表不同的行人实例，而不是简单的 0/1 二值图。
    *   **解决**：利用广播机制 `obj_ids = np.unique(mask)` 获取所有实例 ID，然后通过 `mask == obj_ids[:, None, None]` 一次性将单张 Mask 图拆分为 $(N, H, W)$ 的二进制掩膜张量。
2.  **数据增强的同步性**：
    *   **难点**：如果在训练中对图片进行了“随机水平翻转”，Mask 也必须进行完全一样的翻转，否则分割结果会和人反着来。
    *   **解决**：使用了 `torchvision.transforms` 中的函数式接口，确保 Image 和 Target 共享相同的随机种子或变换参数。
"""

file_path = "MaskRCNN_Report.md"

# 检查文件是否存在
if os.path.exists(file_path):
    # 读取文件内容，防止重复添加
    with open(file_path, "r", encoding="utf-8") as f:
        current_content = f.read()
    
    if "7. 深度思考" in current_content:
        print("⚠️ 报告中似乎已经包含了“深度思考”部分，跳过添加。")
    else:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(notes_content)
        print(f"✅ 成功将笔记追加到 {file_path} 末尾！")
else:
    print(f"❌ 找不到文件 {file_path}，请确认你在正确的目录下。")