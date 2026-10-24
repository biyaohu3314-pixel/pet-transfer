# import json
#
# if __name__ == "__main__":
#     splits = {}
#     for name in ["train_0.1", "train_0.5", "train_1.0", "val"]:
#         with open(f"splits/{name}.json") as f:
#             splits[name] = ...        # TODO: 用 json.load 读进来
#         print(name, ...)              # TODO: 打印里面有多少个索引
#
#     # TODO: 检查 train_1.0 和 val 有没有重复的索引
#     #   提示：转成 set，用 & 求交集，打印交集的长度

import json

if __name__ == "__main__":
    splits = {}   # 用字典存：名字 -> 索引列表

    for name in ["train_0.1", "train_0.5", "train_1.0", "val"]:
        with open(f"{name}.json") as f:
            splits[name] = json.load(f)            # TODO 1: 读进来
        print(name, len(splits[name]))                  # TODO 2: 打印这个列表的长度

    # TODO 3: 检查 train_1.0 和 val 有没有重复的编号
    train_set = set(splits["train_1.0"])  # 列表转成集合
    val_set = set(splits["val"])                         # 同样把 val 转成集合
    print("train 和 val 重复的个数:", len(train_set & val_set))  # 提示：两个集合用 & 求交集，再取 len