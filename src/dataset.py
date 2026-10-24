import json
import os
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader, Subset

# 新增：自动找到项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data")
splits_path = os.path.join(BASE_DIR, "splits")

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def get_loaders(ratio, batch_size=32):
    ratio_str = str(ratio) if str(ratio) in ["0.1", "0.5", "1.0"] else f"{float(ratio):.1f}"

    # 这里 root 换成 data_path
    train_base = OxfordIIITPet(root=data_path, split="trainval", transform=train_tf)
    eval_base = OxfordIIITPet(root=data_path, split="trainval", transform=eval_tf)
    test_base = OxfordIIITPet(root=data_path, split="test", transform=eval_tf)

    # 这里路径换成 splits_path
    with open(os.path.join(splits_path, f"train_{ratio_str}.json"), "r") as f:
        train_idx = json.load(f)
    with open(os.path.join(splits_path, "val.json"), "r") as f:
        val_idx = json.load(f)

    train_ds = Subset(train_base, train_idx)
    val_ds = Subset(eval_base, val_idx)
    test_ds = test_base

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0),
    )

if __name__ == "__main__":
    train_loader, _, _ = get_loaders(0.1)
    x, y = next(iter(train_loader))
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")