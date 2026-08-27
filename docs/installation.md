# 安装与运行

> 本文档摘录自项目总说明，完整版见 [README](../README.md)。

### 系统要求

- Linux 桌面环境（当前主要支持平台）
- Python 3.10 或 3.11，验证基线为 Python 3.10
- 建议使用 Conda 或 venv 隔离环境
- GPU 训练需要兼容的 NVIDIA 驱动和 PyTorch CUDA 版本
- 源码安装不要求虚拟环境必须命名为 `Files_process`

### 推荐 Conda 安装

```bash
git clone <repository-url> Files_process_QT
cd Files_process_QT

conda env create -n vision-platform -f deployment/environment.yml
conda activate vision-platform
```

安装 CUDA 12.8 PyTorch、平台依赖和 X-AnyLabeling：

```bash
bash deployment/install.sh --torch cu128
```

CPU 环境：

```bash
bash deployment/install.sh --torch cpu
```

如果当前环境已经安装正确的 PyTorch：

```bash
bash deployment/install.sh
```

不安装 X-AnyLabeling：

```bash
bash deployment/install.sh --without-label-tool
```

安装脚本始终使用当前环境的 `python`。也可以显式指定解释器：

```bash
PYTHON_BIN=/path/to/python bash deployment/install.sh --torch auto
```

### 直接安装 requirements

```bash
python -m pip install -r requirements.txt
```

这种方式不会替你选择合适的 PyTorch CUDA 发行包。需要训练时，优先使用 `deployment/install.sh`。

### 启动

推荐启动方式：

```bash
bash deployment/run.sh
```

直接启动：

```bash
python main.py
```

`deployment/run.sh` 会自动设置以下项目内运行目录：

- `MPLCONFIGDIR=.runtime/matplotlib`
- `YOLO_CONFIG_DIR=.runtime/ultralytics`
- `XDG_CACHE_HOME=.runtime/cache`
- `PYTHONPATH=<项目根目录>`

首次启动会要求选择数据目录。之后通过 `QSettings` 自动恢复上次目录、图片选择、窗口几何和分栏尺寸。

### 环境诊断

```bash
python deployment/doctor.py --require-label-tool
```

同时检查 CUDA：

```bash
python deployment/doctor.py --require-label-tool --require-cuda
```

诊断项包括 Python、PyQt5、Qt 平台插件、NumPy、OpenCV、Ultralytics、PyTorch、X-AnyLabeling、CUDA、运行目录写权限和 `pip check`。

### 当前验证依赖

| 依赖 | 版本 |
| --- | --- |
| Python | 3.10（基线） |
| PyQt5 | 5.15.11 |
| NumPy | 1.26.4 |
| OpenCV | 4.11.0.86 |
| Pillow | 11.1.0 |
| PyYAML | 6.0.3 |
| psutil | 7.2.2 |
| Send2Trash | 2.1.0 |
| Ultralytics | 8.4.115 |
| X-AnyLabeling | 3.3.10 |

NumPy 固定在 `1.26.4`，OpenCV 相关发行包固定到同一版本代，主要用于避免 X-AnyLabeling、OpenCV 和 Qt 二进制依赖冲突。

当 X-AnyLabeling 与主程序位于同一个 Python 环境时，不需要额外配置。

如果标注工具位于另一个虚拟环境：

```bash
cp deployment/config.example.env deployment/local.env
```

编辑 `deployment/local.env`，推荐指向标注环境的 Python：

```bash
export VISION_PLATFORM_XANYLABELING_PYTHON="$HOME/miniconda3/envs/labeling/bin/python"
```

也可以直接指定可执行文件：

```bash
export VISION_PLATFORM_XANYLABELING="$HOME/miniconda3/envs/labeling/bin/xanylabeling"
```

环境变量优先级高于自动查找。配置的是解释器或可执行文件路径，不是 Conda 环境名称，因此其他机器不需要创建名为 `Files_process` 的环境。

平台使用独立子进程启动标注工具，并清理可能指向 `cv2/qt/plugins` 的错误 Qt 插件路径，避免常见的 `Could not load the Qt platform plugin "xcb"` 冲突。
