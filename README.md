# pet-transfer：预训练到底帮了多少？

## 1. 实验问题
在小数据集 Oxford-IIIT Pet 上，用 ResNet18 分别以从零训练、冻结主干、全量微调三种方式，在 10% / 50% / 100% 训练数据下比较测试集准确率，回答：数据越少时，ImageNet 预训练的帮助是否越大？

### 三种训练方式

| 方式 | 起点 | 训练哪些参数 | 打个比方 |
| ---- | ---- | ---- | ---- |
| scratch从零训练 | 随机初始化，什么都不会 | 全部参数 | 一个从没见过动物的人，从头学认猫狗品种 |
|linear 冻结主干| ImageNet 预训练权重 |只训练 fc 层|一个已经会认动物的人，只学“这 37 个品种叫什么名字”|
|finetune 全量微调|加载 ImageNet 预训练权重|全部参数，但学习率小|一个会认动物的人，再针对猫狗品种整体精细调整

## 2. 环境
- Python 3.10.22
- torch 2.11.0+cu128

## 3. 统一约定
| 项目     | 值         |
|--------|-----------|
| 随机种子   | 42        |
| 图片尺寸   | 224 x 224 |
| batch size | 32        |
| epoch  | 15        |
| 实验命名   |      `{mode}_{ratio}`，例如 `finetune_0.1`     |
| 学习率    |       scratch / linear：1e-3；finetune：1e-4     |

## 4. 分工
| 成员 | 负责的 3 组实验 | 学习率 |
|---|-|---|
| 姜 |download.py、dataset.py、splits/||
| 胡 |画图、PPT ||
| 胡 |model.py、train.py，跑全部 9 组实验||

## 5. 运行方式
python src/train.py --mode ? --ratio ?

## 6. 协作规则
- 开始干活前先 git pull
- git add 之前先 git status 检查
- 走 Git 的：代码、splits/ 里的 json、日志和图；不走 Git 的：data/ 数据集、.pth 权重

## 7. 实验记录
| 实验名 | test acc | 备注 |
|---|---|---|
|     |       |    |