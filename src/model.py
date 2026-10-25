import torch.nn as nn
from torchvision import models
import os
from transformers import AutoModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # src 的上一级 = 项目根目录
DINO_PATH = os.path.join(ROOT, "weights", "dinov3-vits16")

class DinoLinear(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(DINO_PATH)
        for p in self.backbone.parameters():
            p.requires_grad = False
        hidden = self.backbone.config.hidden_size #从 config 读
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):                    # x: [B, 3, 224, 224]
        out = self.backbone(pixel_values=x)  # last_hidden_state: [B, 201, 384]
        feat = out.pooler_output            # [B, 384] 属性
        return self.head(feat)
def build_model(num_classes=37, mode="finetune"):
    """
    model 决定迁移学习方式
    “scratch”  : 不加载预训练权重，从头训练
    "freeze"   : 加载预训练权重，冻结backbone,只训练fc
    "finetune" : 加载预训练权重，全部参数都训练
    """
    if mode == "dino_linear":
        return DinoLinear(num_classes)
    ## 1.根据 mode 决定 weights 传 None 还是 DEFAULT
    if mode == "scratch":
        m = models.resnet18(weights = None)
    else:
        m = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)

    ## 2.如果是 linear 把所有参数的 requires_grad 设为 False
    if mode == "linear":
        for p in m.parameters():
            p.requires_grad = False

    ## 3: 替换 model.fc —— 输入维度用 model.fc.in_features，不要手写 512
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

if __name__ == "__main__":
    #  4: 对 3 种 mode 分别 build 一次，
    #         打印 输出 shape 和 可训练参数数量
    #         提示: sum(p.numel() for p in model.parameters() if p.requires_grad)
        import torch

        x = torch.randn(2, 3, 224, 224)
        for mode in ["dino_linear","scratch", "linear", "finetune"]:
            m = build_model(mode=mode)
            out = m(x)
            n = sum(p.numel() for p in m.parameters() if p.requires_grad)
            print(f"{mode:8s} 输出 {tuple(out.shape)}  可训练参数 {n:,}")