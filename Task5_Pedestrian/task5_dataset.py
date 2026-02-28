import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class PennFudanDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # 加载所有图片文件，并排序确保一一对应
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # 1. 加载图片和 Mask
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        
        img = Image.open(img_path).convert("RGB")
        # Mask 必须是 0, 1, 2... 这样的索引图，不要转成 RGB
        mask = Image.open(mask_path)
        
        # 把 PIL 图片转成 numpy 数组，方便计算
        mask = np.array(mask)
        
        # 2. 获取所有的物体 ID
        # np.unique 会返回 mask 里出现过的所有数字
        # 比如 mask 里有 0(背景), 1(人A), 2(人B)，obj_ids 就是 [0, 1, 2]
        obj_ids = np.unique(mask)
        
        # 去掉背景 (ID=0)，剩下的就是人的 ID
        obj_ids = obj_ids[1:]

        # 3. 生成二进制 Mask (核心步骤！)
        # 这里的逻辑是：对于每个 ID，生成一个 True/False 的矩阵
        # 最终 masks 的形状是 [N, H, W]，N 是人的数量
        masks = mask == obj_ids[:, None, None]

        # 4. 自动计算 Bounding Box (检测框)
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i]) # 找到第 i 个人所有像素的位置
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # 转成 Tensor 格式
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64) # 只有一类：人 (Label=1)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) # 面积
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64) # 假设没有拥挤情况

        # 把所有目标信息打包成字典
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)
            # 注意：这里我们暂时只对 img 做 transform，
            # 实际项目中，如果做翻转(Flip)，Mask 和 Box 也得跟着翻！
            # 但为了简单，我们先不加几何变换。

        return img, target

    def __len__(self):
        return len(self.imgs)

# --- 测试代码 ---
if __name__ == "__main__":
    # 简单的 Transform：转 Tensor
    def get_transform():
        return T.Compose([T.ToTensor()])

    # 实例化 Dataset
    # 假设你的数据在当前目录下的 PennFudanPed 文件夹
    dataset = PennFudanDataset('PennFudanPed', transforms=get_transform())
    
    # 取第 0 个数据看看
    img, target = dataset[0]
    
    print("-" * 30)
    print("✅ 数据加载成功！")
    print(f"图片形状: {img.shape}")
    print(f"检测到人数: {len(target['labels'])}")
    print(f"Mask 形状: {target['masks'].shape} (应该是 [人数, H, W])")
    print(f"Box 形状: {target['boxes'].shape}")
    print("-" * 30)
    print("Dataset 类编写正确，可以进行下一步训练了！")