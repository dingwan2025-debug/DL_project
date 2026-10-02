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

说明：

- 数据来自 `sklearn.datasets.load_digits`，无需联网下载，CPU 上十几秒即可跑完 10 个 epoch。
- 未安装 PyTorch 时脚本会打印提示并直接跳过，不会报错。
- 运行结果图保存在 `outputs/04_pytorch_cnn.png`。`outputs/` 目录由脚本自动创建，目录本身纳入版本库，里面生成的图片不提交。
