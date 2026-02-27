import os
# --- 修复 OMP: Error #15 ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import task5_dataset

# --- 1. 定义模型构建函数 ---
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_model_instance_segmentation(num_classes):
    model = maskrcnn_resnet50_fpn(weights=None) 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

# --- 2. 定义转换函数 ---
def get_transform():
    return T.Compose([T.ToTensor()])

# --- 3. 加载模型 ---
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 2

print("正在加载模型...")
model = get_model_instance_segmentation(num_classes)
model.load_state_dict(torch.load("mask_rcnn_model.pth", map_location=device))
model.to(device)
model.eval()
print("✅ 模型加载成功！")

# --- 4. 预测与可视化 ---
# 加载数据集的一张图 (比如第 0 张)
dataset = task5_dataset.PennFudanDataset('PennFudanPed', transforms=get_transform())
img, _ = dataset[0] 

print("正在进行预测...")
with torch.no_grad():
    prediction = model([img.to(device)])

# 获取预测结果
pred_masks = prediction[0]['masks']
pred_boxes = prediction[0]['boxes']
pred_scores = prediction[0]['scores']

# 筛选置信度 > 0.5 的
keep = pred_scores > 0.5
pred_masks = pred_masks[keep]
pred_boxes = pred_boxes[keep]

print(f"✅ 检测到 {len(pred_boxes)} 个目标")

# 绘图
img_np = img.mul(255).permute(1, 2, 0).byte().numpy()
plt.figure(figsize=(10, 10))

# 1. 先画原图
plt.imshow(img_np)
ax = plt.gca()

# 2. 画 Mask (使用 RGBA 图层方法，避免形状报错)
if len(pred_masks) > 0:
    H, W = img_np.shape[:2]
    # 合并所有检测到的 Mask
    combined_mask = np.zeros((H, W))
    for i in range(len(pred_masks)):
        mask = pred_masks[i, 0].cpu().numpy()
        combined_mask = np.maximum(combined_mask, mask)
    
    # 创建一个红色的 RGBA 图层
    # 形状: (H, W, 4) -> R, G, B, Alpha
    overlay = np.zeros((H, W, 4))
    overlay[:, :, 0] = 1.0  # 红色通道设为 1 (最大)
    overlay[:, :, 1] = 0.0  # 绿色通道设为 0
    overlay[:, :, 2] = 0.0  # 蓝色通道设为 0
    
    # 设置透明度：Mask > 0.5 的地方透明度为 0.5，其他地方完全透明(0)
    overlay[:, :, 3] = np.where(combined_mask > 0.5, 0.5, 0.0)
    
    # 画上去
    plt.imshow(overlay)

# 3. 画 Box (黄色框)
for box in pred_boxes:
    box = box.cpu().numpy()
    rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                             linewidth=2, edgecolor='yellow', facecolor='none')
    ax.add_patch(rect)

plt.axis('off')
plt.title(f"Result: {len(pred_boxes)} Pedestrians Detected")
plt.show()

print("🎉 可视化完成！窗口已弹出。")