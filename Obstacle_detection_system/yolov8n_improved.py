import torch
import torch.nn as nn
from ultralytics import YOLO

# 定义 SE 模块
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# 修改 YOLOv8n 模型
class ImprovedYOLOv8n(nn.Module):
    def __init__(self, original_model):
        super(ImprovedYOLOv8n, self).__init__()
        self.backbone = original_model.model  # 获取原始模型的 Backbone
        self.se_block = SEBlock(channel=256)  # 在某个层后添加 SE 模块

    def forward(self, x):
        x = self.backbone(x)  # 原始 Backbone 的前向传播
        x = self.se_block(x)  # 添加 SE 模块
        return x

# 加载预训练的 YOLOv8n 模型
original_model = YOLO('yolov8n.pt')

# 创建改进后的模型
improved_model = ImprovedYOLOv8n(original_model)

# 打印改进后的模型结构
# print(improved_model)
# 保存改进后的模型
torch.save(improved_model.state_dict(), 'improved_yolov8n.pt')