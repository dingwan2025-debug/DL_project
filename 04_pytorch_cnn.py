"""
04_pytorch_cnn.py —— 用卷积神经网络（CNN）做手写数字识别

为什么图像要用 CNN 而不是全连接层？
  全连接层要先把图片拉平成一维，会丢掉"上下左右相邻"的空间信息；
  卷积层用一个小的滑动窗口（卷积核）扫描图片，能提取局部特征（笔画、边缘），
  参数还比全连接层少得多，所以图像任务几乎都用 CNN。

数据：scikit-learn 自带的 8x8 手写数字（1797 张，10 个类别），完全离线、极小。
     真做项目时换成 MNIST / CIFAR-10，代码结构不用变。

用法：
    python 04_pytorch_cnn.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 0

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


def load_digits_data():
    """加载 scikit-learn 自带的 8x8 手写数字数据集，归一化并转成 torch 张量。"""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    digits = load_digits()
    images = digits.images.astype(np.float32) / 16.0   # 像素值 0~16 -> 0~1
    labels = digits.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=SEED, stratify=labels
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


def main():
    print("=" * 62)
    print("04 PyTorch CNN —— 8x8 手写数字识别（10 分类）")
    print(f"PyTorch 版本 {torch.__version__} | 计算设备 CPU")
    print("=" * 62)

    torch.manual_seed(SEED)
    X_train, y_train, X_test, y_test = load_digits_data()
    print(f"训练集 {len(X_train)} 张，测试集 {len(X_test)} 张，图片尺寸 8x8，类别 10")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

    model = SmallCNN()
    criterion = nn.CrossEntropyLoss()          # 多分类常用损失（内含 softmax）
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    print(f"模型结构：\n{model}\n")

    epochs = 10
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
        epoch_loss /= len(X_train)

        model.eval()
        with torch.no_grad():
            pred = model(X_test).argmax(dim=1)
            acc = (pred == y_test).float().mean().item()

        history["loss"].append(epoch_loss)
        history["acc"].append(acc)
        print(f"  epoch {epoch:>2}/{epochs} | 损失 {epoch_loss:.4f} | 测试准确率 {acc:.3f}")

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
    save_path = OUTPUT_DIR / "04_pytorch_cnn.png"
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    print(f"图片已保存：{save_path}")

    print("\n小结：卷积负责提取图像局部特征，全连接层负责最后的分类决策。")
    print("      换个数据集（MNIST/CIFAR/自己的图片）时，改的是数据加载部分。")


if __name__ == "__main__":
    main()
