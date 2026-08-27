# 视觉数据管理平台源码部署指南

本目录集中保存源码部署所需的环境定义、依赖约束、安装脚本、启动脚本和环境诊断工具。部署过程不依赖 `Files_process` 或任何固定虚拟环境名称。

## 1. 支持范围

- 操作系统：当前优先支持 Linux 桌面环境。
- Python：支持 3.10 和 3.11，当前验证基线为 Python 3.10。
- 数据、模型仓库和训练结果不随源码分发，需要在应用中重新选择本机目录。
- `models/`、`training/runs/`、`.runtime/` 等运行产物已加入 `.gitignore`。

## 2. 推荐安装

克隆源码后进入项目根目录：

```bash
git clone <repository-url> Files_process_QT
cd Files_process_QT
```

环境名称可以自由指定。下面的 `vision-platform` 只是示例：

```bash
conda env create -n vision-platform -f deployment/environment.yml
conda activate vision-platform
```

安装 NVIDIA CUDA 12.8 训练环境及 X-AnyLabeling：

```bash
bash deployment/install.sh --torch cu128
```

只使用 CPU：

```bash
bash deployment/install.sh --torch cpu
```

如果当前环境已经安装了正确的 PyTorch，可以使用默认模式：

```bash
bash deployment/install.sh
```

安装脚本始终使用当前激活环境中的 `python`，不会执行 `conda run -n Files_process`。

## 3. 启动

```bash
bash deployment/run.sh
```

也可以直接运行：

```bash
python main.py
```

推荐使用 `run.sh`，它会为 Matplotlib、Ultralytics 和缓存文件设置项目内可写目录，并加载本机的标注工具配置。

## 4. 环境诊断

安装后或出现环境异常时执行：

```bash
python deployment/doctor.py --require-label-tool
```

需要确认 GPU 训练能力时执行：

```bash
python deployment/doctor.py --require-label-tool --require-cuda
```

诊断内容包括：

- Python 与当前解释器路径
- PyQt5 和 Qt 平台插件
- NumPy、OpenCV、Ultralytics、PyTorch 版本
- X-AnyLabeling 是否可启动
- CUDA 是否可用
- 项目运行目录是否可写
- `pip check` 依赖一致性

## 5. X-AnyLabeling 独立环境

默认情况下，平台会自动使用当前 Python 环境旁边的 `xanylabeling`。如果 X-AnyLabeling 与主程序放在不同环境，不需要关心那个环境叫什么，只需要指定它的 Python 路径。

```bash
cp deployment/config.example.env deployment/local.env
```

编辑 `deployment/local.env`：

```bash
export VISION_PLATFORM_XANYLABELING_PYTHON="$HOME/miniconda3/envs/my-label-tool/bin/python"
```

也可以指定可执行文件：

```bash
export VISION_PLATFORM_XANYLABELING="$HOME/miniconda3/envs/my-label-tool/bin/xanylabeling"
```

`local.env` 是每台机器自己的配置，不会提交到源码仓库。程序通过对应解释器调用项目内的 Qt 安全启动器，以避免 OpenCV 自带 Qt 插件与 PyQt5 的 `xcb` 插件冲突。

## 6. 依赖文件说明

| 文件 | 作用 |
| --- | --- |
| `environment.yml` | 创建基础 Conda 环境 |
| `requirements-core.txt` | Qt、数据管理、模型管理与训练功能 |
| `requirements-labeling.txt` | X-AnyLabeling 及兼容的 NumPy/OpenCV 约束 |
| `install.sh` | 安装 PyTorch 和项目依赖，并执行诊断 |
| `run.sh` | 从源码启动程序 |
| `doctor.py` | 检查部署环境和常见冲突 |

当前约束将 NumPy 固定为 `1.26.4`，并将两个 OpenCV 发行包固定到相同版本。这是因为 X-AnyLabeling 3.3.10 要求 `numpy<=1.26.4`；混装 NumPy 2.x 或不同代的 OpenCV 是 Qt 插件和二进制兼容问题的重要来源。

## 7. Linux Qt/xcb 排查

如果诊断报告 Qt 平台插件缺少系统动态库，需要根据 Linux 发行版安装对应的 XCB、XKB、OpenGL 系统包。Ubuntu/Debian 常见依赖包括：

```bash
sudo apt install libxcb-cursor0 libxcb-xinerama0 libxkbcommon-x11-0 libgl1 libegl1
```

不要手工把 OpenCV 的 `cv2/qt/plugins` 写入全局 `QT_QPA_PLATFORM_PLUGIN_PATH`。项目的 X-AnyLabeling 启动器会在子进程内选择 PyQt5 自己的插件目录。

## 8. 迁移现有数据和训练任务

源码部署与工作数据迁移是两件事：

1. 源码部署只安装应用，不复制数据集、模型权重和训练输出。
2. 第一次启动时选择本机数据根目录，之后会通过 `QSettings` 记住该目录。
3. 历史训练任务和 `dataset.yaml` 可能包含原机器的绝对路径。复制到另一台机器后，需要重新选择数据项目或重新生成训练批次。
4. 模型仓库和训练输出目录应放在目标机器可写的位置，不应直接提交到源码仓库。

后续如果需要完整迁移历史任务，应再增加“工作区导出/导入”和“路径映射”功能，而不是依赖两台机器拥有相同的 `/home/...` 目录结构。

## 9. 发布前验证清单

```bash
python deployment/doctor.py --require-label-tool
python -m unittest discover -s tests -v
bash deployment/run.sh
```

正式发布前建议在一台没有开发环境残留的机器或全新 Conda 环境中执行一次完整安装、启动、打开标注工具和最小训练任务测试。
