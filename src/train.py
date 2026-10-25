"""
pet-transfer 训练脚本 —— 三个人用同一份，只改命令行参数。

用法（在项目根目录运行；在 src 里运行也可以，脚本会自动切到项目根目录）：
    python src/train.py --mode linear   --ratio 0.1
    python src/train.py --mode scratch  --ratio 0.5
    python src/train.py --mode finetune --ratio 1.0

冒烟测试（用随机假数据，不需要 dataset.py，2 个 epoch 几秒钟跑完）：
    python src/train.py --mode linear --ratio 0.1 --smoke

每组实验的输出在 outputs/{mode}_{ratio}/：
    log.csv     每个 epoch 一行：epoch, train_loss, train_acc, val_acc, epoch_time_s
    best.pth    val_acc 最高时的模型权重（不上传 Git）
    result.txt  最终结果：test_acc、best_val_acc、可训练参数量、训练耗时、设备等
冒烟测试的输出在 outputs/_smoke/，看完删掉，不要提交。

约定（README）：seed=42, 224x224, batch_size=32, epochs=15,
             lr: scratch/linear = 1e-3, finetune = 1e-4, 优化器 Adam
"""
import argparse
import csv
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

# 不管从哪个目录运行，都切到项目根目录，保证 data/、splits/、outputs/ 这些相对路径都能找到
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from model import build_model  # noqa: E402

# 固定随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 读命令行参数 按mode自动选择学习率
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="finetune",
                   choices=["dino_linear","scratch", "linear", "finetune"])
    p.add_argument("--ratio", type=float, default=0.1, choices=[0.1, 0.5, 1.0])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=None)          # 不传就按 mode 自动选
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true",
                   help="用随机假数据快速检查流程能否跑通")
    return p.parse_args()

# 拿到train /val /test 三个DataLoader
def get_data(args):
    """返回 train_loader, val_loader, test_loader"""
    if args.smoke:
        from torch.utils.data import TensorDataset, DataLoader
        # 64 张假图: x [64, 3, 224, 224]，标签 y [64]，取值 0~36
        fake = TensorDataset(torch.randn(64, 3, 224, 224), torch.randint(0, 37, (64,)))
        loader = DataLoader(fake, batch_size=args.batch_size)
        return loader, loader, loader
    from dataset import get_loaders  # B 写的 src/dataset.py
    return get_loaders(args.ratio, batch_size=args.batch_size)


def train_one_epoch(model, loader, criterion, optimizer, device, show_shape=False):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)       # x: [B, 3, 224, 224]   y: [B]
        logits = model(x)                        # logits: [B, 37]
        loss = criterion(logits, y)              # 标量，是这个 batch 的平均 loss

        if show_shape:                           # 只在第一个 batch 打印一次，用来核对维度
            print(f"  shape 检查: x {tuple(x.shape)}  y {tuple(y.shape)}  logits {tuple(logits.shape)}")
            show_shape = False

        optimizer.zero_grad()    # ① 清空上一轮的梯度
        loss.backward()          # ② 反向传播
        optimizer.step()         # ③ 更新参数

        total_loss += loss.item() * x.size(0)    # 平均 loss × B = 这个 batch 的总 loss
        correct += (logits.argmax(dim=1) == y).sum().item()   # argmax(dim=1): [B,37] -> [B]
        total += x.size(0)                       # 最后一个 batch 可能不满 32
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)                        # [B, 37]
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return correct / total


def main():
    args = parse_args()
    if args.lr is None:
        args.lr = 1e-4 if args.mode == "finetune" else 1e-3
    if args.smoke:
        args.epochs = min(args.epochs, 2)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"

    exp_name = f"{args.mode}_{args.ratio}"                       # 例如 finetune_0.1
    out_dir = os.path.join("outputs", "_smoke" if args.smoke else "", exp_name)
    os.makedirs(out_dir, exist_ok=True)

    # ---- 模型 / 优化器 / 损失 ----
    model = build_model(mode=args.mode).to(device)
    params = [p for p in model.parameters() if p.requires_grad]   # linear 模式下只有 fc 的参数
    n_trainable = sum(p.numel() for p in params)
    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_loader, val_loader, test_loader = get_data(args)

    print(f"=== {exp_name} | lr={args.lr} | epochs={args.epochs} | device={device_name}"
          f"{' | SMOKE' if args.smoke else ''}")
    print(f"可训练参数: {n_trainable:,}")
    print(f"数据量: train {len(train_loader.dataset)} | val {len(val_loader.dataset)} | "
          f"test {len(test_loader.dataset)} | 每个 epoch {len(train_loader)} 个 batch")

    # ---- 训练 ----
    log_path = os.path.join(out_dir, "log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_acc", "epoch_time_s"])

    best_path = os.path.join(out_dir, "best.pth")
    best_val, best_epoch = -1.0, 0
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer,
                                                device, show_shape=(epoch == 1))
        val_acc = evaluate(model, val_loader, device)
        dt = time.time() - t0

        with open(log_path, "a", newline="") as f:              # 每个 epoch 追加一行
            csv.writer(f).writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}",
                                    f"{val_acc:.4f}", f"{dt:.1f}"])

        flag = ""
        if val_acc > best_val:                                  # 用 val 挑最好的模型
            best_val, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), best_path)
            flag = "  *best"
        print(f"[{epoch:02d}/{args.epochs}] loss {train_loss:.4f}  train_acc {train_acc:.3f}  "
              f"val_acc {val_acc:.3f}  ({dt:.0f}s){flag}")
    train_min = (time.time() - t_start) / 60

    # ---- 测试：加载 val 最好的那一版，只在 test 上评估一次 ----
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_acc = evaluate(model, test_loader, device)

    with open(os.path.join(out_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"mode={args.mode}\n"
                f"ratio={args.ratio}\n"
                f"test_acc={test_acc:.4f}\n"
                f"best_val_acc={best_val:.4f}\n"
                f"best_epoch={best_epoch}\n"
                f"epochs={args.epochs}\n"
                f"lr={args.lr}\n"
                f"batch_size={args.batch_size}\n"
                f"seed={args.seed}\n"
                f"trainable_params={n_trainable}\n"
                f"train_time_min={train_min:.1f}\n"
                f"device={device_name}\n")
    print(f"=== 完成 {exp_name}: test_acc {test_acc:.4f} | best_val {best_val:.4f} "
          f"(epoch {best_epoch}) | 用时 {train_min:.1f} 分钟")
    print(f"结果保存在 {out_dir}")


if __name__ == "__main__":      # Windows 下 DataLoader 用多进程时必须有这一行
    main()
