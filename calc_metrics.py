import torch
from thop import profile
from model_seresnet import SEResNet # 导入刚才写的模型

# 1. 实例化模型
model = SEResNet()

# 2. 创建一个虚拟输入 (CIFAR-10 图片大小)
input_tensor = torch.randn(1, 3, 32, 32)

# 3. 计算
flops, params = profile(model, inputs=(input_tensor, ))

print("="*30)
print(f"Model: SE-ResNet18 (CIFAR version)")
print(f"FLOPs (计算量): {flops / 1e9:.3f} G (十亿次)")
print(f"Params (参数量): {params / 1e6:.3f} M (百万个)")
print("="*30)
