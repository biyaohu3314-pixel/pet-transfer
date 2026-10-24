import json
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import Subset, DataLoader

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
test_tf = ...   # TODO: Resize(256) → CenterCrop(224) → ToTensor → Normalize

def load_idx(name):
    # TODO: 打开 splits/{name}.json，返回列表（参考你的 check_splits.py）
    ...

def get_loaders(ratio, batch_size=32, num_workers=2):
    # TODO 1: 创建两个 trainval 数据集，一个用 train_tf，一个用 test_tf
    #         OxfordIIITPet(root="data", split="trainval", transform=...)
    # TODO 2: 创建 test 数据集（split="test"，用 test_tf）
    # TODO 3: train_set = Subset(带 train_tf 的那个, load_idx(f"train_{ratio}"))
    #         val_set   = Subset(带 test_tf 的那个,  load_idx("val"))
    # TODO 4: 用 DataLoader 包装三个数据集；只有 train 需要 shuffle=True
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_loaders(0.1)
    print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
    x, y = next(iter(train_loader))
    print(x.shape, y.shape, y[:8])