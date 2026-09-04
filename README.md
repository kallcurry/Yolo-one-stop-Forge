# YoloForge · YOLO 一站式视觉数据锻造平台

> 原始数据是原料，审查是精炼，数据准备是配料，训练是铸型，模型是成品。
> 一个本地桌面平台，跑完**视觉数据 → YOLO 模型**的全流程。

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)
![UI](https://img.shields.io/badge/Qt-PyQt5%205.15-41CD52)
![Framework](https://img.shields.io/badge/Ultralytics-8.4.x-FF6A00)
![Tasks](https://img.shields.io/badge/Tasks-Pose%20%7C%20Detection%20%7C%20Segmentation%20%7C%20OBB-00D4FF)

[![GitHub](https://img.shields.io/badge/Repository-GitHub-181717?logo=github&logoColor=white)](https://github.com/kallcurry/Yolo-one-stop-Forge)

## 这是什么

YoloForge 是一个基于 **PyQt5** 的本地计算机视觉数据与模型工作平台，围绕 **X-AnyLabeling JSON 标注**与 **Ultralytics YOLO 训练**流程，把数据工程师最常做的四件事装进一个深色科技风的桌面应用：

| 模块 | 一句话说明 |
| --- | --- |
| 🗂️ **数据管理** | 按项目浏览原始/训练/测试数据，图片与标注可视化（框、关键点、骨架、异常红显），文件操作与标注联动 |
| 🔍 **标注审查** | 内置四任务规则引擎 + 自定义规则 + Python 插件，图表分析、人工结论独立持久化 |
| 🏭 **数据准备** | 多批次收集、按来源分层划分 train/val、可选测试集独立批次、稀有类别保护、动态生成 `dataset.yaml`，全流程可追溯 |
| 🚀 **训练中心** | Ultralytics 子进程训练、任务快照、实时监控（曲线/指标/日志）、断点续训 |
| 📦 **模型管理** | 自动扫描训练仓库为模型卡片，多模型对比、指标折线、最佳/最终值比较 |
| 🎥 **推理中心** | 模型实时推理工作台：摄像头/视频/图片目录/RTSP 输入，四任务标注叠加、FPS/计数板 HUD、截图与录制 |

> ⚠️ 平台**不会把类别数、类别名、关键点数写死**：结构一律运行时从数据、JSON、TXT 和 `dataset.yaml` 推断。
> 当前状态：数据管理、模型管理、训练中心、评估中心、推理中心已接入；跟踪、MoE、模型辅助标注暂不在范围内。

## 快速开始

```bash
# 1. 创建环境（Python 3.10 / 3.11）
conda env create -n vision-platform -f deployment/environment.yml
conda activate vision-platform

# 2. 安装依赖（GPU: cu128 / CPU: cpu / 已有 PyTorch: 不传参）
bash deployment/install.sh --torch cu128

# 3. 启动
bash deployment/run.sh
```

首次启动会要求选择数据目录；之后自动记住目录、图片选择、窗口尺寸与分栏。环境体检：

```bash
python deployment/doctor.py --require-label-tool
```

- 安装细节与 PyTorch 选择：见 [安装指南](docs/installation.md)
- 标注工具独立环境配置：见 [安装指南 · X-AnyLabeling 配置](docs/installation.md)

## 📚 文档中心

专题文档位于 [`docs/`](docs/README.md)，按角色与主题拆分，按需深入：

| 文档 | 内容 | 适合谁 |
| --- | --- | --- |
| [界面与交互设计](docs/ui-design.md) | 信息架构、布局、视觉语言、快捷键 | 使用者 / UI 评审 |
| [核心能力](docs/features.md) | 数据管理、审查、准备、训练、模型的完整能力清单 | 使用者 |
| [数据目录与标注约定](docs/data-conventions.md) | 项目目录约定、X-AnyLabeling JSON 格式 | 使用者 / 数据侧 |
| [训练管线](docs/training-pipeline.md) | 数据生成逻辑、训练参数、任务与运行产物 | 训练 / 算法 |
| [数据审查模板](docs/review-templates.md) | 模板字段、内置规则、自定义规则与插件 | 算法 / 质量 |
| [模型仓库约定](docs/model-repository.md) | 仓库结构、模型识别格式 | 模型管理 |
| [操作中心工具](docs/operation-tools.md) | 统计、匹配检查、重复审查、标签替换等 | 数据侧 |
| [配置与持久化](docs/configuration.md) | 项目内配置、QSettings 说明 | 运维 |
| [开发与测试](docs/development.md) | 代码结构、运行链路、测试指引 | 开发者 |
| [常见问题](docs/troubleshooting.md) | xcb 冲突、缺 TXT、类别不全、断点续训等 | 所有人 |
| [安全与数据保护](docs/security.md) | 删除策略、副本策略、迁移注意 | 所有人 |
| [路线图](docs/roadmap.md) | 当前边界、推荐后续方向、部署资料 | 所有人 |
| [评估中心设计](docs/evaluation-center.md) | 评估中心定稿设计：流程、指标、结果契约、模块联动、实施阶段 | 所有人 |

## 项目结构

```text
Files_process_QT/
├── main.py                        # Qt 应用入口与视图装配
├── app/
│   ├── controllers/               # 控制器：信号联动与状态中枢
│   ├── models/                    # 纯业务逻辑：审查、准备、训练契约、模型解析
│   ├── views/                     # Qt 界面：主窗口、可视化、训练管理、模型管理
│   ├── tools/                     # 操作中心独立工具
│   └── training_runner.py         # Ultralytics 子进程训练入口
├── resources/                     # 深色主题 QSS 与四任务审查/训练模板
├── deployment/                    # 安装、启动、诊断与环境配置
├── models/                        # 本地模型权重缓存（不入库）
├── training/                      # 训练任务快照与运行结果（不入库）
└── docs/                          # 本文档中心
```

## 发布前检查

```bash
# 环境体检（Qt 插件、CUDA、依赖一致性等）
python deployment/doctor.py --require-label-tool

# 启动冒烟
bash deployment/run.sh
```

## 默认配置与全局任务

- **默认配置化**：可调默认值集中管理，复制 `resources/app_config.example.json`
  为 `resources/app_config.json` 即可按需覆盖（个人配置不入库）：
  推理/评估/数据准备超参、统一图片扩展名、各任务标注目录名。
  详见 [docs/configuration.md](docs/configuration.md)。
- **全局任务联动**：顶部任务选择器一次切换，数据管理/训练中心/评估中心/
  工具全部同步，任务类型持久化、重启自动恢复。

## 社区与支持

- 🐛 问题反馈：请在本仓库提交 [Issue](https://github.com/kallcurry/Yolo-one-stop-Forge/issues)
- 📖 完整安装/迁移/发布说明：见 [deployment/README.md](deployment/README.md)
