##  Debug 复盘记录

### 问题描述
在完成 CIFAR-10 训练代码并尝试绘制 Loss 曲线时，程序在训练结束后突然崩溃，并在终端输出了以下错误信息，导致 `plt.show()` 无法弹出窗口，图片未能保存。

```text
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program.
...
```

### 原因分析
通过阅读报错信息中的 `Hint` 部分以及搜索引擎查询，我了解到这是一个经典的**库冲突问题**。
*   **根本原因**：我的环境中安装了多个并行计算库（OpenMP）。PyTorch 和 NumPy（或其他依赖库如 matplotlib）都试图加载自己的 OpenMP 动态链接库（`libiomp5md.dll`）。
*   **冲突点**：当代码运行到绘图部分（涉及 NumPy 运算）时，第二个 OpenMP 实例试图初始化，被系统检测到并强制中止了程序，这是一种保护机制，防止并行计算出错。

### 解决方案
根据报错信息的提示和网上的资料，我找到了两种解决方案：

1.  **方案**：设置环境变量允许库重复加载。
    在代码的最开头（`import torch` 之前）添加以下代码：
    ```python
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    ```
    这行代码告诉 OpenMP 运行时忽略重复加载的错误，允许程序继续执行。


