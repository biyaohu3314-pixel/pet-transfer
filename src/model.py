import torch.nn as nn
from torchvision import models

def build_model(num_classes=37, model="finetune"):
    """
    model 决定迁移学习方式
    “scratch”  : 不加载预训练权重，从头训练
    "freeze"   : 加载预训练权重，冻结backbone,只训练fc
    "finetune" : 加载预训练权重，全部参数都训练
    """
    ## TODO 1.根据 model 决定 weights 传 None 还是 DEFAULT
    if model == "scratch":
        m = models.resnet18(weights = None)
    else:
        m = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)

    ## TODO 2.如果是 freeze 把所有参数的 requires_grad 设为 False
    if model == "freeze":
        model.parameters(requires_grad = False)

    ## TODO 3: 替换 model.fc —— 输入维度用 model.fc.in_features，不要手写 512
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

if __name__ == "__main__":
    # TODO 4: 对 3 种 mode 分别 build 一次，
    #         打印 输出 shape 和 可训练参数数量
    #         提示: sum(p.numel() for p in model.parameters() if p.requires_grad)
    pass