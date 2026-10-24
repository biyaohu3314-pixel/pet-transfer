import os
from torchvision.datasets import OxfordIIITPet

os.makedirs("data", exist_ok=True)
print("开始下载 trainval 数据集...")
OxfordIIITPet(root="data", split="trainval", download=True)
print("开始下载 test 数据集...")
OxfordIIITPet(root="data", split="test", download=True)
print("下载完成！")