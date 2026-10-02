"""
04_pytorch_cnn.py —— 用卷积神经网络（CNN）做手写数字识别

为什么图像要用 CNN 而不是全连接层？
  全连接层要先把图片拉平成一维，会丢掉"上下左右相邻"的空间信息；
  卷积层用一个小的滑动窗口（卷积核）扫描图片，能提取局部特征（笔画、边缘），
  参数还比全连接层少得多，所以图像任务几乎都用 CNN。

数据：scikit-learn 自带的 8x8 手写数字（1797 张，10 个类别），完全离线、极小。
     真做项目时换成 MNIST / CIFAR-10，代码结构不用变。

用法：
    python 04_pytorch_cnn.py                          # 默认 10 个 epoch
    python 04_pytorch_cnn.py --epochs 20 --lr 0.005   # 换超参数再跑一次
    python 04_pytorch_cnn.py --outdir outputs/exp1    # 结果存到别的目录

每次运行都会把超参数和训练曲线写进 <outdir>/04_pytorch_cnn_metrics.json，
方便对比不同设置的效果。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:  # 某些 IDE / 重定向环境下的 stdout 不支持重配置，忽略即可
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 默认超参数：命令行不传参时就跑这一组，和最初版本保持一致
DEFAULT_CONFIG = {
    "epochs": 10,
    "batch_size": 64,
    "lr": 0.01,
    "seed": 0,
    "outdir": "outputs",
}


def parse_args(argv=None):
    """解析命令行参数，让同一份代码靠换参数就能重复做实验。"""
    parser = argparse.ArgumentParser(
        description="用小型 CNN 做 8x8 手写数字识别",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"], help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"], help="每批样本数")
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"], help="Adam 学习率")
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_CONFIG["seed"], help="随机种子（数据划分 / 权重初始化 / 打乱）"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=DEFAULT_CONFIG["outdir"],
        help="结果输出目录，相对路径按脚本所在目录解释",
    )
    return parser.parse_args(argv)


def resolve_outdir(outdir):
    """把 --outdir 解析成绝对路径并创建好，返回 Path。"""
    path = Path(outdir)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    path.mkdir(parents=True, exist_ok=True)
    return path

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("=" * 62)
    print("未检测到 PyTorch，本示例已跳过。")
    print("=" * 62)
    print("安装命令：pip install torch")
    print("装好后重新运行：python 04_pytorch_cnn.py")
    sys.exit(0)


def load_digits_data(seed=0):
    """加载 scikit-learn 自带的 8x8 手写数字数据集，归一化并转成 torch 张量。"""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    digits = load_digits()
    images = digits.images.astype(np.float32) / 16.0   # 像素值 0~16 -> 0~1
    labels = digits.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=seed, stratify=labels
    )

    # 卷积层要求输入形状为 (批大小, 通道数, 高, 宽)，灰度图通道数为 1
    X_train = torch.tensor(X_train).unsqueeze(1)
    X_test = torch.tensor(X_test).unsqueeze(1)
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)
    return X_train, y_train, X_test, y_test


class SmallCNN(nn.Module):
    """两层卷积 + 两层全连接的小型 CNN，8x8 图片上 CPU 训练也很快。"""

    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # 输入 1x8x8 -> 卷积出 8 个通道 -> ReLU -> 池化
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                     # 8x8 -> 4x4
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                     # 4x4 -> 2x2
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                        # 16x2x2 = 64 维
            nn.Linear(16 * 2 * 2, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),            # 输出 10 个类别的分数(logits)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_model(model, train_loader, X_test, y_test, epochs, lr):
    """标准训练循环：前向 -> 算损失 -> 反向 -> 更新，每个 epoch 记录损失和测试准确率。"""
    criterion = nn.CrossEntropyLoss()          # 多分类常用损失（内含 softmax）
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"loss": [], "acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            pred = model(X_test).argmax(dim=1)
            acc = (pred == y_test).float().mean().item()

        history["loss"].append(epoch_loss)
        history["acc"].append(acc)
        print(f"  epoch {epoch:>2}/{epochs} | 损失 {epoch_loss:.4f} | 测试准确率 {acc:.3f}")

    return history


def save_metrics(out_dir, config, history, n_train, n_test, n_errors):
    """把这次实验的超参数和结果写成 JSON，方便多次运行之间对比。"""
    metrics = {
        "config": config,
        "torch_version": torch.__version__,
        "n_train": n_train,
        "n_test": n_test,
        "n_errors": n_errors,
        "final_test_accuracy": history["acc"][-1],
        "best_test_accuracy": max(history["acc"]),
        "history": history,
    }
    path = out_dir / "04_pytorch_cnn_metrics.json"
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main(argv=None):
    args = parse_args(argv)
    out_dir = resolve_outdir(args.outdir)

    print("=" * 62)
    print("04 PyTorch CNN —— 8x8 手写数字识别（10 分类）")
    print(f"PyTorch 版本 {torch.__version__} | 计算设备 CPU")
    print(f"超参数 epochs={args.epochs} batch_size={args.batch_size} lr={args.lr} seed={args.seed}")
    print("=" * 62)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    X_train, y_train, X_test, y_test = load_digits_data(seed=args.seed)
    print(f"训练集 {len(X_train)} 张，测试集 {len(X_test)} 张，图片尺寸 8x8，类别 10")

    model = SmallCNN()
    print(f"模型结构：\n{model}\n")

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True
    )
    history = train_model(model, train_loader, X_test, y_test, epochs=args.epochs, lr=args.lr)

    # 看一下预测错的样本长什么样
    model.eval()
    with torch.no_grad():
        pred = model(X_test).argmax(dim=1)
    wrong = (pred != y_test).nonzero(as_tuple=True)[0]
    print(f"\n测试集错误 {len(wrong)} / {len(X_test)} 张，最终准确率 {history['acc'][-1]:.3f}")

    # 可视化：训练曲线 + 部分预测结果（标题里 T=真实值，P=预测值）
    fig = plt.figure(figsize=(11, 5.2))
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(history["loss"], color="tab:blue")
    ax1.set_title("训练损失")
    ax1.set_xlabel("epoch")
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(history["acc"], color="tab:green")
    ax2.set_title("测试准确率")
    ax2.set_xlabel("epoch")
    ax2.grid(alpha=0.3)

    # 取几张错得最典型的样本（2x3 网格中第 3~6 格留给图片）
    show = wrong[:4] if len(wrong) >= 4 else wrong
    for i, idx in enumerate(show):
        ax = fig.add_subplot(2, 3, 3 + i)
        ax.imshow(X_test[idx, 0], cmap="gray")
        ax.set_title(f"T={y_test[idx].item()} / P={pred[idx].item()}", fontsize=9)
        ax.axis("off")
    if len(show) == 0:
        ax = fig.add_subplot(2, 3, 3)
        ax.text(0.5, 0.5, "测试集全部预测正确", ha="center", va="center")
        ax.axis("off")

    fig.tight_layout()
    save_path = out_dir / "04_pytorch_cnn.png"
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    print(f"图片已保存：{save_path}")

    metrics_path = save_metrics(
        out_dir,
        config=vars(args),
        history=history,
        n_train=len(X_train),
        n_test=len(X_test),
        n_errors=len(wrong),
    )
    print(f"训练记录已保存：{metrics_path}")

    print("\n小结：卷积负责提取图像局部特征，全连接层负责最后的分类决策。")
    print("      换个数据集（MNIST/CIFAR/自己的图片）时，改的是数据加载部分。")


if __name__ == "__main__":
    main()
