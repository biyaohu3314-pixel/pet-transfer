import os, sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
# TODO: 像 train.py 那样设置 ROOT、切换工作目录、把 src 加进 sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from dataset import get_loaders
from model import DinoLinear          # 复用你写好的类，只用里面的 backbone

@torch.no_grad()
def extract(fn, loader, device):
    """fn: 一个函数，输入图片 x，返回整图特征"""
    feats, labels = [], []
    for x, y in loader:
        f = fn(x.to(device))           # [B, ?]
        feats.append(f.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)   # 拼接后的 shape？

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_loader = get_loaders(1.0)     # 只要 test 集，ratio 填多少都一样

    # --- ResNet18：预训练 + 去掉 fc ---
    r = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    r.fc = nn.Identity()      # TODO: 换成什么，才能让输出变成 512 维的特征而不是 1000 类？
    # nn.Identity()是一个"什么都不做"的层：输入什么就原样输出什么。

    r = r.to(device).eval()

    # --- DINOv3：只要主干 ---
    d = DinoLinear(37).backbone.to(device).eval()

    # TODO: 分别调用 extract，得到 resnet 特征、dino 特征、labels
    resnet_feats, labels = extract(r, test_loader, device)

    def dino_fn(x):
        return d(pixel_values=x).pooler_output  # 填：取哪个属性？

    dino_feats, _ = extract(dino_fn, test_loader, device)

    # TODO: 保存到 outputs/features/ 下（先 os.makedirs）
    out_dir = os.path.join("outputs", "features")
    os.makedirs(out_dir, exist_ok=True)  # 文件夹不存在就新建；已存在也不报错
    np.save(os.path.join(out_dir, "resnet18.npy"), resnet_feats)
    # 照着这一行，自己写 dino.npy 和 labels.npy
    np.save(os.path.join(out_dir, "labels.npy"), labels)
    np.save(os.path.join(out_dir, "dino.npy"), dino_feats)

    # TODO: 打印三个数组的 shape，和手册核对
    print("resnet18:", resnet_feats.shape)
    # 另外两个照着写
    print("labels:", labels.shape)
    print("dino:", dino_feats.shape)

if __name__ == "__main__":
    main()