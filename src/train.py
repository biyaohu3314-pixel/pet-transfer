import argparse
import random
import numpy as np
import torch

from model import build_model
# from dataset import get_loaders   # 等 B 推送后再打开

def set_seed(seed=42):
    # TODO: 分别给 random、numpy、torch、torch.cuda 设种子（4 行）
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="finetune",
                   choices=["scratch", "linear", "finetune"])
    # TODO: 再加 --ratio(float, 默认 0.1)、--epochs(int, 默认 15)、
    #       --lr(float, 默认 None)、--batch_size(int, 默认 32)
    p.add_argument("--ratio", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()

def main():
    args = parse_args()
    if args.lr is None:
        # TODO: 按文档规定，finetune 用 1e-4，其他两种用 1e-3
            args.lr = 1e-4 if args.mode == "finetune" else 1e-3
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # TODO: 有 GPU 就用 "cuda"，没有就用 "cpu"

    model = build_model(mode=args.mode).to(device)
    # TODO: optimizer = Adam，只传入 requires_grad=True 的参数，参考文档第五章第 4 点
    # TODO: criterion = 交叉熵损失
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    exp_name = f"{args.mode}_{args.ratio}"
    print(exp_name, "lr =", args.lr, "device =", device)
    # 打印 optimizer 实际管理的参数数量，验证 filter 是否生效：
    n = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
    print("optimizer 管理的参数:", f"{n:,}")

if __name__ == "__main__":      # Windows 下 num_workers>0 时必须有这一行
    main()