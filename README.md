# DL_project

深度学习入门练习代码。当前包含一个用 PyTorch 卷积神经网络（CNN）做手写数字识别的示例。

## 内容

| 文件 | 说明 |
| --- | --- |
| `04_pytorch_cnn.py` | 两层卷积 + 两层全连接的小型 CNN，在 scikit-learn 自带的 8x8 手写数字数据集（1797 张，10 类）上训练并评估，最后输出训练曲线和错分样本的可视化图。 |

## 运行

```bash
pip install -r requirements.txt
python 04_pytorch_cnn.py
```

常用参数（都有默认值，不传就用默认那组）：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--epochs` | `10` | 训练轮数 |
| `--batch-size` | `64` | 每批样本数 |
| `--lr` | `0.01` | Adam 学习率 |
| `--seed` | `0` | 随机种子，固定数据划分与权重初始化 |
| `--outdir` | `outputs` | 结果目录，相对路径按脚本所在目录解释 |

例如换一组超参数再跑一次做对比：

```bash
python 04_pytorch_cnn.py --epochs 20 --lr 0.005 --outdir outputs/lr0005
```

说明：

- 数据来自 `sklearn.datasets.load_digits`，无需联网下载，CPU 上十几秒即可跑完 10 个 epoch。
- 未安装 PyTorch 时脚本会打印提示并直接跳过，不会报错。
- 运行结果图保存在 `outputs/04_pytorch_cnn.png`。`outputs/` 目录由脚本自动创建，目录本身纳入版本库，里面生成的图片不提交。
- 每次运行会把超参数和逐轮损失/准确率写成 `outputs/04_pytorch_cnn_metrics.json`，多次实验之间可以直接对比。
