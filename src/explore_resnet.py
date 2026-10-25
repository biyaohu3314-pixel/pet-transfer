import torch
from torchvision import models
from torchvision.models import ResNet

if __name__ == "__main__":
    # 1.加载预训练的 resnet18
    model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)

    # 2.打印整个模型
    print(model)

    # 3.选一批假照片：2 张、3 通道、224 x 224
    x = torch.randn(2, 3, 224, 224)
    print("输入:",x.shape)

    # 4.把 x 送进模型，打印输出的shape
    out = model(x)
    print("输出:",out.shape)

    # 一层一层手动往前走，每步打印shape
    h = x
    for name in ["conv1", "bn1","relu","maxpool","layer1","layer2","layer3","layer4","avgpool"]:
        layer = getattr(model,name) # 按名字取出这一层，相当于m.conv1,m.layer1...
        h = layer(h)
        print(f"{name:8s} -> {tuple(h.shape)}")

    h = torch.flatten(h,1)
    print("flatten ->",tuple(h.shape))

    h = model.fc(h)
    print("fc       ->", tuple(h.shape))