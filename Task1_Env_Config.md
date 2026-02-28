# Task 1: 环境配置与 Git 学习笔记

## 1. PyTorch 版本选择
**问题：你是如何判断自己应该安装哪个版本的 PyTorch 的？**

*   **我的判断逻辑**：
    1.  首先，我检查了自己的电脑显卡配置。
        *   *(如果是 N 卡)*：打开任务管理器或终端输入 `nvidia-smi`，发现我有 NVIDIA 显卡（型号：RTX 3060 / 4060 / ...），所以我选择了 **CUDA 版 (GPU)**，因为深度学习训练需要 GPU 加速。
        *   *(如果没显卡)*：我的电脑是集成显卡 / Mac，所以我选择了 **CPU 版**。
    2.  然后，我去 PyTorch 官网 (pytorch.org)，根据我的系统 (Windows/Linux) 和 CUDA 版本（比如 CUDA 11.8 或 12.1），复制了对应的安装命令。
    3.  最终安装命令：`pip3 install torch torchvision ... --index-url ...`

## 2. 遇到的报错与解决方案
**问题：在此过程中遇到了什么报错？你是如何解决的？**

*   **报错 1：下载速度过慢 / Connection Timeout**
    *   **原因**：默认的 Anaconda 或 pip 源在国外，连接不稳定。
    *   **解决**：我切换了国内镜像源（清华源/阿里源）。
        ```bash
        pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
        ```

*   **报错 2：Conda 命令未找到 (Command not found)**
    *   **原因**：安装 Anaconda 时没有自动配置环境变量。
    *   **解决**：手动将 Anaconda 的 `Scripts` 目录添加到了系统的 Path 环境变量中，或者直接使用 Anaconda Prompt 进行操作。

*   **报错 3：虚拟环境创建失败**
    *   **解决**：使用了 `conda create -n ud_lab python=3.8` 明确指定了 Python 版本，确保环境纯净。

## 3. Git 的使用体验
这是我第一次系统地使用 Git。
*   学会了 `git init` 初始化仓库。
*   学会了 `git add .` 和 `git commit -m "..."` 来保存代码版本。
*   学会了 `git push` 将本地代码同步到 GitHub，感觉这对于代码管理非常有帮助，不用再像以前那样复制一堆 "代码_最终版_v2.zip" 了。

---
*环境配置虽然繁琐，但跑通 Hello World (import torch; print(torch.cuda.is_available())) 的那一刻非常有成就感！*