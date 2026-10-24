import os
import matplotlib.pyplot as plt
from torchvision.datasets import OxfordIIITPet
from collections import Counter

# 魔法代码：自动找到项目根目录（无论你在哪运行都不会错）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data")
figures_path = os.path.join(BASE_DIR, "figures")

# 确保 figures 文件夹存在
os.makedirs(figures_path, exist_ok=True)

print("正在统计类别数量...")
# 注意这里，把 root 换成了自动找到的绝对路径
ds = OxfordIIITPet(root=data_path, split="trainval", download=False)

# 获取所有标签
labels = [ds[i][1] for i in range(len(ds))]
class_counts = Counter(labels)

# 提取类别名称（比如 "Abyssinian" 等）
class_names = ds.classes

# 开始画图
plt.figure(figsize=(15, 6))
plt.bar(class_names, [class_counts[i] for i in range(37)], color='skyblue')
plt.xticks(rotation=90)  # 名字太长，转90度
plt.xlabel("Breed")
plt.ylabel("Number of images")
plt.title("Oxford-IIIT Pet Dataset (Trainval) Class Distribution")
plt.tight_layout()

# 保存到 figures 文件夹
save_path = os.path.join(figures_path, "class_dist.png")
plt.savefig(save_path, dpi=300)
print(f"类别分布图已保存到 {save_path}")