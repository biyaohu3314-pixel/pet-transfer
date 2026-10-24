import os
import json
from torchvision.datasets import OxfordIIITPet
from sklearn.model_selection import train_test_split

os.makedirs("splits", exist_ok=True)
print("正在读取数据标签...")
ds = OxfordIIITPet(root="data", split="trainval", download=False)
labels = [ds[i][1] for i in range(len(ds))]
all_indices = list(range(len(labels)))

# 1. 划分 train 和 val (8:2, 分层抽样)
train_idx, val_idx = train_test_split(
    all_indices, test_size=0.2, stratify=labels, random_state=42
)
train_labels = [labels[i] for i in train_idx]

# 2. 抽出 10%、50%
train_10_idx, _ = train_test_split(
    train_idx, train_size=0.1, stratify=train_labels, random_state=42
)
train_50_idx, _ = train_test_split(
    train_idx, train_size=0.5, stratify=train_labels, random_state=42
)
train_100_idx = train_idx

# 3. 存成 json
splits = {
    "train_0.1.json": train_10_idx,
    "train_0.5.json": train_50_idx,
    "train_1.0.json": train_100_idx,
    "val.json": val_idx,
}

for filename, idx_list in splits.items():
    with open(os.path.join("splits", filename), "w") as f:
        json.dump(idx_list, f)
    print(f"已保存 {filename}，包含 {len(idx_list)} 张图片的索引")