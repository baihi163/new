import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import DataLoader
import task5_dataset  # 导入我们刚才写的 Dataset 类

# --- 1. 定义模型构建函数 ---
def get_model_instance_segmentation(num_classes):
    # 加载在 COCO 上预训练过的模型
    # weights="DEFAULT" 等同于 weights="Mask_RCNN_ResNet50_FPN_Weights.COCO_V1"
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")

    # 获取分类器的输入特征数
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # 替换预训练的头部 (Box Predictor)
    # 这里的 num_classes 需要包含背景
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 获取 Mask 分支的输入特征数
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    
    # 替换 Mask 头部 (Mask Predictor)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

# --- 2. 这里的 collate_fn 非常重要！ ---
# 因为目标检测中，每张图片的 Box 数量不一样，不能简单堆叠
def collate_fn(batch):
    return tuple(zip(*batch))

# --- 3. 训练主循环 ---
def main():
    # 检查是否有 GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # 数据集只有两类：背景(0) + 行人(1)
    num_classes = 2
    
    # 准备数据
    dataset = task5_dataset.PennFudanDataset('PennFudanPed', transforms=task5_dataset.get_transform())
    
    # 划分训练集和测试集 (这里简单取前 150 张训练，后 20 张测试)
    indices = torch.randperm(len(dataset)).tolist()
    dataset_train = torch.utils.data.Subset(dataset, indices[:-20])
    dataset_test = torch.utils.data.Subset(dataset, indices[-20:])

    # 定义 DataLoader
    data_loader = DataLoader(
        dataset_train, batch_size=2, shuffle=True, 
        num_workers=0, collate_fn=collate_fn
    )

    # 获取模型
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    # 定义优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # 训练 1 个 Epoch (为了演示，只跑 1 轮，实际可以多跑几轮)
    num_epochs = 1
    
    print("Start training...")
    model.train() # 设置为训练模式
    
    for epoch in range(num_epochs):
        i = 0
        for images, targets in data_loader:
            # 把数据搬到 GPU/CPU
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 前向传播
            # torchvision 的模型在训练模式下，会自动计算所有 Loss 并返回字典
            loss_dict = model(images, targets)
            
            #把所有 Loss 加起来 (分类Loss + Box回归Loss + Mask Loss + ...)
            losses = sum(loss for loss in loss_dict.values())

            # 反向传播
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch: {epoch}, Iter: {i}, Loss: {losses.item():.4f}")
            i += 1

    print("Training finished!")
    
    # 保存模型
    torch.save(model.state_dict(), "mask_rcnn_model.pth")
    print("Model saved to mask_rcnn_model.pth")

if __name__ == "__main__":
    # 这里的 get_transform 需要我们在 task5_dataset.py 里定义过
    # 为了方便，我们在 task5_dataset.py 里加一个简单的 transform 函数
    # 或者直接在这里临时定义一个
    import torchvision.transforms as T
    task5_dataset.get_transform = lambda: T.Compose([T.ToTensor()])
    
    main()