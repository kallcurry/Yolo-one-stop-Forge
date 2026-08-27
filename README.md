# YoloForge · YOLO 一站式视觉数据锻造平台

> 原始数据是原料，审查是精炼，数据准备是配料，训练是铸型，模型是成品——一个平台跑完视觉数据到 YOLO 模型的全流程。

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)
![UI](https://img.shields.io/badge/Qt-PyQt5%205.15-41CD52)
![Framework](https://img.shields.io/badge/Ultralytics-8.4.x-FF6A00)
![Tasks](https://img.shields.io/badge/Tasks-Pose%20%7C%20Detection%20%7C%20Segmentation%20%7C%20OBB-00D4FF)

[![GitHub](https://img.shields.io/badge/Repository-GitHub-181717?logo=github&logoColor=white)](https://github.com/kallcurry/Yolo-one-stop-Forge)

基于 PyQt5 的本地计算机视觉数据与模型工作平台，围绕 X-AnyLabeling JSON 标注和 Ultralytics YOLO 训练流程，提供数据浏览、标注审查、数据准备、训练任务管理、训练监控、模型仓库解析与模型对比等能力。

当前平台重点支持以下任务：

- 姿态估计（Pose）
- 目标检测（Detection）
- 语义/实例分割（Segmentation）
- 旋转目标检测（OBB）

平台不会把类别数量、类别名称或 Pose 关键点数量写死在业务流程中。运行时会优先从当前数据、X-AnyLabeling JSON、YOLO TXT 和批次 `dataset.yaml` 中推断数据结构；审查模板负责定义规则策略和骨架拓扑，训练模板负责定义 Ultralytics 参数。

> 当前状态：数据管理、模型管理和训练中心已经接入；评估中心保留为后续模块。跟踪、MoE 和模型辅助标注目前不在实现范围内。

## 1. 核心能力

### 1.1 数据管理

- 按数据项目浏览原始数据、训练数据和验证/测试数据。
- 支持 Pose、Detection、Segmentation、OBB 任务切换，并切换对应标注目录和审查策略。
- 展示图片、目标框、关键点、骨架连线及问题高亮。
- 图片支持适应窗口、自由缩放、滚轮缩放和拖拽查看。
- 标注正常元素使用绿色，审查命中的异常点、异常框或异常连线使用红色。
- 支持隐藏标注、仅显示框、显示框和关键点，以及独立开关骨架线。
- 支持复制、移动、删除、重命名和新建目录，并同步处理对应标注文件。
- 记住上次打开的数据根目录、图片位置、窗口尺寸和分栏尺寸。

### 1.2 标注审查

- 当前文件审查和当前目录批量统计。
- 问题文件列表、指标明细列表及关键词搜索。
- 图表分析和 Markdown 文字报告切换。
- 图表柱状项可点击并联动到对应文件明细。
- 支持“人工通过当前文件”“忽略选中问题”“撤销人工结论”。
- 人工结论独立保存，不会伪造或覆盖原始标注内容。
- 支持按模板顺序重排整个文件夹的 Pose 关键点。
- 支持从现有模板复制、修改并另存为新模板。
- 支持参数化自定义规则和 Python 审查插件。
- 未实现的规则会显示“规则未执行”，不会静默当作通过。

### 1.3 标注修改与副本同步

- “修改当前文件标签”按钮可启动 X-AnyLabeling，并直接打开当前图片及其标注目录。
- 标注工具退出后，平台通过文件指纹判断 JSON 是否发生变化。
- 标注变更可同步到原始数据、训练批次和测试批次中的对应副本。
- 同步优先依据 `preparation_manifest.json` 建立可追溯关系，并检查任务标注目录是否一致。

### 1.4 全数据目录重复审查

- 从任意数据项目根目录、`images`/`annotations`/`labels` 目录或其原始批次进入审查。
- 自动解析实际数据项目根目录，不依赖 `ShengSong_Datasets` 等固定项目名称。
- 仅扫描原始数据树，不扫描派生的 `training_data` 和 `test_data`。
- 分别统计图片、JSON 标注和 TXT 标签的重复组、可删除副本和可释放空间。
- 支持删除当前重复组或全部重复副本，并为每组保留一个确定性副本。
- 使用“文件大小 -> 头尾快速指纹 -> 完整 SHA-256”的分阶段流程减少无效磁盘读取。

重复判定依据是**文件字节内容完全相同**，不是文件名相同。两个名称不同的 TXT 文件，如果内容完全一致，仍会被归为重复文件；两个视觉相似但字节不同的图片不会被判为重复。

### 1.5 训练数据准备

- 浏览已有 `training_data` 批次并查看其状态、样本数、划分和来源。
- 已有训练批次支持查看、使用、重命名、删除和打开目录。
- 从多个原始批次收集 `images`、任务 JSON 标注和 YOLO TXT 标签。
- 默认按 `8:2` 划分训练集和验证集，验证比例默认 `0.20`，随机种子默认 `42`。
- 每个原始来源批次独立按比例抽取验证样本，而不是合并全部数据后再进行一次全局随机划分。
- 划分时进行类别感知和稀有类别保护；在样本条件允许时，避免某个类别完全缺失于验证集。
- 默认创建独立副本，也可配置其他写入策略。
- 对缺少 TXT、无效标签、结构列数不一致、同名冲突和训练/验证类别缺失进行预检。
- 待补齐数据可以在明确警告后参与准备；未标注图片默认跳过，避免大批背景图被误收入训练集。
- 只有明确启用“空标注作为背景”时，缺少标签的图片才会作为背景样本进入批次。
- 生成 `preparation_manifest.json`，记录来源批次、复制关系、参数和数据结构。
- 动态生成 `dataset.yaml`，不使用固定类别数或固定关键点数。

### 1.6 训练中心

- 训练工作台包含任务中心、数据准备、数据审查、训练配置和任务监控。
- 训练配置支持核心参数表单与高级 JSON 模板。
- 内置 Pose、Detection、Segmentation、OBB 四套独立训练模板。
- 使用 Ultralytics YOLO 启动本地单任务训练。
- 预训练权重统一下载或存放到项目根目录 `models/`。
- 每个任务在 `training/tasks/<task-id>/` 保存可检查、可复现的任务文件。
- 训练结果默认写入 `training/runs/<项目分组>/<任务名称>/`。
- 任务中心支持草稿、排队、运行、完成、停止、中断和失败等状态。
- 支持新建、编辑草稿、复制配置、重试、重命名、归档、删除、备注和打开任务文件。
- 停止、异常中断和失败任务可使用原任务目录的 `weights/last.pt` 继续训练。
- 任务监控展示 Epoch、进度、运行时间、CPU、内存、GPU、显存、实时日志和训练曲线。
- 训练曲线包含损失、评估指标与学习率等数据。
- 训练完成后可跳转到模型管理查看对应模型。
- 任务列表支持右键跳转到模型管理结果。

### 1.7 模型管理

- 扫描一个模型仓库目录，将一次 Ultralytics 训练结果解析为一张模型卡片。
- 按项目和任务类型筛选 Pose、Detection、Segmentation、OBB 模型。
- 自动读取 `args.yaml`、`results.csv`、`weights/` 和训练结果图片。
- 展示模型架构、训练框架、输入尺寸、Batch、优化器、Epoch、更新时间、权重格式和本地路径。
- 按任务选择主要指标，例如 Pose mAP50-95、Box mAP50-95 或 Mask mAP50-95。
- 训练结果图片通过列表选择后查看。
- 展示训练数据来源、原始批次、图片数、JSON 数、TXT 数及可用状态。
- 点击训练数据来源可跳转到数据管理界面，查看后可以返回模型详情。
- 支持多模型对比、公共指标折线图、最佳值和最终值比较。
- 对比项可跳转到模型详情，并保留返回路径。

## 2. UI 与交互设计

### 2.1 信息架构

顶部一级模块分为：

1. 数据管理
2. 模型管理
3. 训练中心
4. 评估中心（规划中）

数据管理内部再通过任务选择器切换 Pose、Detection、Segmentation 和 OBB。平台模块和视觉任务属于不同层级，不通过审查模板来伪装成不同业务模块。

### 2.2 数据管理布局

- 左侧：数据目录树，区分原始数据、训练数据、验证/测试数据。
- 中间：图片与标注可视化区域。
- 右侧：图片信息、标注树、文件数据和审查面板。
- 底部：图片导航、标注显示控制和文件操作。
- 分栏使用 `QSplitter`，尺寸可拖动并在退出时保存。

### 2.3 视觉语言

- 深色科技风主题，强调青色、绿色和少量红/橙状态色。
- 背景包含动态网格、扫描线和信号轨迹效果。
- 面板使用低透明度背景，让动态底纹可见，同时为表格头和状态栏保留不透明底色保证可读性。
- 按钮带悬停高亮和柔和发光效果。
- 状态采用颜色编码：绿色表示就绪/通过，黄色表示待补齐/警告，红色表示失败/异常，蓝色表示当前选择或主要操作。
- 自定义无边框窗口支持四边和四角缩放、双击标题区域最大化、最大化后恢复原窗口几何。

### 2.4 常用快捷键

| 快捷键 | 功能 |
| --- | --- |
| `Ctrl+O` | 打开数据目录 |
| `Ctrl+Q` | 退出 |
| `Ctrl+C` | 复制当前文件 |
| `Ctrl+X` | 移动当前文件 |
| `Delete` | 删除当前文件 |
| `F2` | 重命名 |
| `Ctrl+Shift+N` | 新建文件夹 |
| `F5` | 刷新 |
| `Left` / `Right` | 上一张 / 下一张图片 |
| `A` | 循环切换标注显示模式 |
| `1` 或 `0` | 适应窗口 |
| `+` / `-` | 放大 / 缩小图片 |
| 鼠标滚轮 | 在图片区域缩放 |
| 鼠标拖拽 | 缩放后平移图片 |

## 3. 数据目录约定

平台面向“项目根目录 + 分离式图片/标注目录”的数据组织方式。项目名和批次名均可自定义。

```text
Dataset_Project/
├── images/                         # 原始图片
│   ├── batch_A/
│   └── batch_B/
├── annotations/                    # Pose，默认名称可由模板修改
│   ├── batch_A/
│   └── batch_B/
├── annotations-det/                # Detection，可选
├── annotations-seg/                # Segmentation，可选
├── annotations-obb/                # OBB，可选
├── labels/                         # YOLO TXT 标签
│   ├── batch_A/
│   └── batch_B/
├── training_data/                  # 平台生成或接管的训练批次
│   └── 2026-08-27/
│       ├── images/
│       ├── annotations/
│       ├── labels/
│       ├── train_data/
│       │   ├── images/train/
│       │   ├── images/val/
│       │   ├── labels/train/
│       │   └── labels/val/
│       ├── dataset.yaml
│       └── preparation_manifest.json
├── test_data/                      # 验证/测试数据
│   ├── images/
│   ├── annotations/
│   └── labels/
└── videos/                         # 可选，当前不进入主要审查流程
```

说明：

- `annotations` 当前默认代表 Pose，但 `annotation_dir` 可在每个任务的审查模板中修改。
- 某个任务对应的标注目录不存在时，界面会明确显示“当前没有该任务的标注”，而不是回退到其他任务目录。
- 图片与 JSON/TXT 通过同名文件主干关联，例如 `image_001.jpg`、`image_001.json` 和 `image_001.txt`。
- 训练和测试目录属于派生数据。原始数据重复审查不会扫描这两个目录。
- 数据副本同步依赖文件名、任务标注目录和 `preparation_manifest.json`，应保留该清单。

## 4. X-AnyLabeling JSON 约定

平台按 X-AnyLabeling/LabelMe 风格 JSON 读取标注，常用字段如下：

```json
{
  "version": "3.x",
  "imagePath": "image_001.jpg",
  "imageWidth": 1280,
  "imageHeight": 720,
  "shapes": [
    {
      "label": "person",
      "shape_type": "rectangle",
      "group_id": 0,
      "points": [[100, 80], [600, 700]]
    },
    {
      "label": "nose",
      "shape_type": "point",
      "group_id": 0,
      "points": [[340, 150]]
    }
  ]
}
```

任务对应的主要形状：

| 任务 | 主要 `shape_type` | 说明 |
| --- | --- | --- |
| Pose | `rectangle` + `point` | 人框和关键点通过 `group_id` 关联 |
| Detection | `rectangle` | 水平目标框 |
| Segmentation | `polygon` | 多边形点集 |
| OBB | `rotation` 等旋转框形状 | 按 X-AnyLabeling 旋转框格式解析 |

## 5. 安装与运行

### 5.1 系统要求

- Linux 桌面环境（当前主要支持平台）
- Python 3.10 或 3.11，验证基线为 Python 3.10
- 建议使用 Conda 或 venv 隔离环境
- GPU 训练需要兼容的 NVIDIA 驱动和 PyTorch CUDA 版本
- 源码安装不要求虚拟环境必须命名为 `Files_process`

### 5.2 推荐 Conda 安装

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

### 5.3 直接安装 requirements

```bash
python -m pip install -r requirements.txt
```

这种方式不会替你选择合适的 PyTorch CUDA 发行包。需要训练时，优先使用 `deployment/install.sh`。

### 5.4 启动

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

### 5.5 环境诊断

```bash
python deployment/doctor.py --require-label-tool
```

同时检查 CUDA：

```bash
python deployment/doctor.py --require-label-tool --require-cuda
```

诊断项包括 Python、PyQt5、Qt 平台插件、NumPy、OpenCV、Ultralytics、PyTorch、X-AnyLabeling、CUDA、运行目录写权限和 `pip check`。

### 5.6 当前验证依赖

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

## 6. X-AnyLabeling 配置

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

## 7. 数据审查模板

内置模板位于：

```text
resources/
├── pose_review_template.json
├── detection_review_template.json
├── segmentation_review_template.json
└── obb_review_template.json
```

模板是审查策略，不是数据类别的永久真相。加载目录后，平台会从实际数据和可用的 `dataset.yaml` 推断类别、关键点、左右对和连接信息，再与当前模板的规则策略合并。

### 7.1 模板字段

```json
{
  "name": "项目 Pose 审查模板",
  "version": 1,
  "description": "模板用途",
  "task_type": "pose",
  "annotation_dir": "annotations",
  "target_classes": ["person_type_a", "person_type_b"],
  "keypoints": ["nose", "left_shoulder", "right_shoulder"],
  "kpt_connections": [[0, 1], [0, 2]],
  "left_right_pairs": [
    ["left_shoulder", "right_shoulder"]
  ],
  "rules": {
    "duplicate_keypoint": true,
    "suspected_left_right_swap": true
  },
  "custom_rules": [],
  "thresholds": {
    "box_margin_min": 4.0,
    "box_margin_ratio": 0.02
  }
}
```

| 字段 | 含义 |
| --- | --- |
| `name` | 模板显示名称 |
| `version` | 模板版本，必须为正整数 |
| `task_type` | `pose`、`detection`、`segmentation` 或 `obb` |
| `annotation_dir` | 数据根目录下该任务的 JSON 标注目录名 |
| `target_classes` | 目标类别；Pose 中通常表示可作为人物框的类别 |
| `keypoints` | Pose 关键点顺序，也是重排序目标顺序 |
| `kpt_connections` | 骨架连接，可写索引或关键点名 |
| `left_right_pairs` | 左右关键点对 |
| `rules` | 内置规则开关 |
| `custom_rules` | 参数化规则或 Python 插件规则 |
| `thresholds` | 几何规则阈值 |

### 7.2 内置审查规则

#### Pose

| 规则 | 检查内容 |
| --- | --- |
| `duplicate_keypoint` | 同一 `group_id` 中同名关键点重复出现 |
| `suspected_left_right_swap` | 根据成对关键点相对关系对疑似左右反标进行启发式判断 |
| `missing_person_box` | 关键点组没有对应目标框 |
| `keypoint_outside_box` | 关键点落在对应框外 |
| `keypoint_wrong_person` | 关键点更接近或落入其他人物框，疑似归属错误 |
| `group_id_missing` | 目标框或关键点缺少 `group_id` |
| `group_id_conflict` | 分组关系冲突或混乱 |
| `image_size_missing` | JSON 缺少图片宽高 |
| `image_size_mismatch` | JSON 宽高与真实图片不一致 |

左右反标是启发式规则。固定机位、侧身、弯腰、遮挡和只出现半身时可能产生误报，因此应结合红色骨架、人工通过和忽略单项问题使用，不能把它当成绝对真值。

#### Detection

空标注、无效矩形、框越界、面积过小、宽高比异常、框重复、未知类别、意外形状类型和图片尺寸问题。

#### Segmentation

空标注、无效多边形、点越界、重复点、自相交、面积过小、未知类别、意外形状类型和图片尺寸问题。

#### OBB

空标注、无效旋转框、框越界、重复点、角点顺序异常、面积过小、宽高比异常、未知类别、意外形状类型和图片尺寸问题。

### 7.3 默认阈值

| 阈值 | 默认值 | 用途 |
| --- | ---: | --- |
| `box_margin_min` | 4.0 | Pose 框边界最小容差 |
| `box_margin_ratio` | 0.02 | Pose 框尺寸比例容差 |
| `left_right_min_points` | 6 | 左右反标判断所需最少有效点数 |
| `left_right_margin_min` | 12.0 | 左右关系最小像素差 |
| `left_right_margin_ratio` | 0.04 | 相对人物框宽度的左右容差 |
| `left_right_score_ratio` | 1.35 | 判定反标所需分数比例 |
| `bbox_min_area` | 4.0 | 水平框最小面积 |
| `bbox_min_side` | 2.0 | 水平框最小边长 |
| `bbox_max_aspect_ratio` | 30.0 | 水平框最大宽高比 |
| `bbox_duplicate_iou` | 0.95 | 重复框 IoU 阈值 |
| `obb_min_area` | 4.0 | OBB 最小面积 |
| `obb_min_edge` | 2.0 | OBB 最小边长 |
| `obb_max_aspect_ratio` | 30.0 | OBB 最大宽高比 |
| `polygon_min_area` | 4.0 | 多边形最小面积 |
| `polygon_min_edge` | 2.0 | 多边形最小边长 |

### 7.4 自定义参数化规则

支持的 `custom_rules[].type`：

- `required_keypoints`：要求指定关键点存在。
- `forbidden_keypoints`：禁止指定关键点出现。
- `paired_keypoints`：成对关键点必须同时存在。
- `relative_position`：两个点必须满足上下左右关系。
- `distance_range`：两个点的像素距离必须位于指定范围。
- `python`：交给 Python 插件执行。

示例：

```json
{
  "id": "helmet_above_nose",
  "name": "头盔应位于鼻子上方",
  "type": "relative_position",
  "severity": "warning",
  "params": {
    "point_a": "top_helmet",
    "point_b": "nose",
    "relation": "above",
    "margin": 2,
    "target_classes": ["person_dress_finish"]
  }
}
```

### 7.5 Python 审查插件

模板规则：

```json
{
  "id": "custom_head_rule",
  "name": "自定义头部规则",
  "type": "python",
  "severity": "warning",
  "path": "plugins/custom_head_rule.py",
  "function": "check"
}
```

插件函数签名为 `check(context, rule)`，返回 issue 字典列表或 `ReviewIssue` 列表：

```python
def check(context, rule):
    issues = []
    for group_id in context.group_ids():
        point = context.point(group_id, "nose")
        if point is None:
            issues.append(context.issue(
                rule_id=rule["id"],
                severity=rule.get("severity", "warning"),
                message=f"group_id={group_id} 缺少 nose",
                group_id=group_id,
                label="nose",
            ))
    return issues
```

插件可以使用：

- `context.data`：完整 JSON object。
- `context.shapes`：原始 shapes 列表。
- `context.boxes` / `context.points`：已解析框和点。
- `context.image_size`：实际图片尺寸。
- `context.group_ids()`：有效分组列表。
- `context.boxes_in_group(group_id)`：组内框。
- `context.points_in_group(group_id, label=None)`：组内关键点。
- `context.point(group_id, label)`：指定组的第一个同名点。
- `context.issue(...)`：构造标准问题字典。

相对插件路径优先相对模板 JSON 所在目录解析，其次相对项目根目录和当前工作目录。也可以使用 `entry: "module.function"` 或 `entry: "path.py:function"`。

## 8. 训练数据生成逻辑

### 8.1 收集规则

对每个选中的原始批次，平台按文件主干匹配：

```text
images/<source>/sample.jpg
<annotation_dir>/<source>/sample.json
labels/<source>/sample.txt
```

默认要求 JSON 和 TXT 均存在且可解析。选择“跳过不完整样本”后，缺失 JSON/TXT 的图片不进入训练批次；选择“空标注作为背景”后，符合背景规则的无标签图片可以进入训练集。

### 8.2 划分规则

- 默认 `val_ratio=0.20`，即每个原始来源约 80% 训练、20% 验证。
- 每个来源单独洗牌并抽取验证样本，防止大批次完全支配验证集。
- 随机种子控制结果可复现。
- 会检查类别分布，并尽可能在验证集中保留每个类别。
- 单例或极稀有类别会优先保留在训练集，具体结果受样本数量约束。
- 测试集中已登记的数据可按准备参数排除，避免数据泄漏。

### 8.3 动态 `dataset.yaml`

平台从实际 TXT 和 JSON 推断：

- 所有出现过的 class id。
- class id 到类别名称的映射。
- Pose 标签每行列数。
- `kpt_shape` 中的关键点数量和维度。
- 关键点名称顺序。
- 根据 `left_`/`right_` 命名推断 `flip_idx`。

一个 Pose 文件可能生成如下配置：

```yaml
path: /absolute/path/to/training_data/batch
train: train_data/images/train
val: train_data/images/val
names:
  0: person_type_a
  1: person_type_b
kpt_shape: [23, 3]
flip_idx: [0, 2, 1]
keypoint_names:
  - top_helmet
  - left_helmet
  - right_helmet
```

上例中的类别数和 23 个关键点只是示例。实际文件由当前批次推断：以后出现第 5、第 8 个类别或不同关键点数量时，不需要修改代码中的固定列表。

## 9. 训练参数配置

内置模板位于：

```text
resources/training_templates/
├── pose_training_template.json
├── detection_training_template.json
├── segmentation_training_template.json
└── obb_training_template.json
```

### 9.1 模板格式

```json
{
  "name": "项目 Pose 训练模板",
  "version": 1,
  "description": "训练用途",
  "task_type": "pose",
  "model": "yolov8n-pose.pt",
  "parameters": {
    "epochs": 100,
    "batch": 16,
    "imgsz": 640,
    "device": "0",
    "optimizer": "auto",
    "lr0": 0.01,
    "amp": true,
    "plots": true
  },
  "ultralytics_version": "8.4.115"
}
```

### 9.2 平台管理参数

以下参数由任务创建、数据批次和输出目录统一管理，不能写入模板 `parameters`：

- `data`
- `project`
- `name`
- `task`
- `mode`
- `model`
- `classes`
- `single_cls`
- `resume`

`resume` 是任务运行状态，不是新任务的普通超参数。停止或异常中断后，在任务中心执行“继续训练”时，平台会自动将模型切换到当前任务目录的 `weights/last.pt` 并启用恢复模式，以恢复 Epoch、优化器和调度器状态。

### 9.3 常用核心参数

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `epochs` | 100 | 最大训练轮次 |
| `patience` | 100 | Early Stopping 等待轮次 |
| `batch` | 16 | Batch size；可使用 Ultralytics 支持的比例/自动形式 |
| `imgsz` | 640 | 输入尺寸 |
| `device` | `"0"` | 设备，例如 `0`、`0,1` 或 `cpu` |
| `workers` | 8 | DataLoader worker 数量 |
| `optimizer` | `auto` | 优化器 |
| `lr0` | 0.01 | 初始学习率 |
| `lrf` | 0.01 | 最终学习率系数 |
| `momentum` | 0.937 | 动量 |
| `weight_decay` | 0.0005 | 权重衰减 |
| `warmup_epochs` | 3.0 | Warmup 轮次 |
| `amp` | `true` | 自动混合精度 |
| `cache` | `false` | 数据缓存，可设 `true`、`ram` 或 `disk` |
| `pretrained` | `true` | 使用预训练参数 |
| `seed` | 0 | 训练随机种子 |
| `deterministic` | `true` | 尽可能确定性训练 |
| `save_period` | -1 | 周期保存；-1 表示仅按默认策略保存 |
| `plots` | `true` | 生成训练图表 |
| `val` | `true` | 训练时执行验证 |

任务专用参数包括 Pose 的 `pose`、`kobj`，Segmentation 的 `mask_ratio`、`overlap_mask`、`copy_paste`，以及各任务的增强和损失权重。完整默认值以对应 JSON 模板为准。

高级模板允许保留 Ultralytics 支持的其他 JSON 可序列化参数。平台会对布尔值、概率、非负数、`batch`、`cache` 和 `freeze` 等常见参数进行类型与范围校验。

## 10. 训练任务与运行产物

项目内运行目录：

```text
Files_process_QT/
├── models/                         # 自动下载和手工放置的模型权重
├── training/
│   ├── task_registry.sqlite3       # 训练任务注册表
│   ├── tasks/
│   │   └── <task-id>/
│   │       ├── dataset.yaml        # 任务数据配置快照
│   │       ├── training_request.json
│   │       ├── run_training.py     # 可单独执行的任务入口
│   │       └── training.log
│   └── runs/
│       └── <project>/<run-name>/   # Ultralytics 训练结果
└── .runtime/
    ├── training_jobs/              # 运行态事件/状态
    ├── matplotlib/
    ├── ultralytics/
    └── cache/
```

训练任务通过独立 Python 进程运行，UI 与训练进程通过结构化事件和任务注册表同步状态。这样训练崩溃不会直接拖垮主窗口，应用重启后也可以恢复任务历史。

`training/tasks/<task-id>/run_training.py` 可在当前应用环境中单独运行：

```bash
python training/tasks/<task-id>/run_training.py
```

任务快照中的 `dataset.yaml` 会把 `path` 绑定到经过校验的原训练批次，因此复制任务目录到其他机器后，如果数据路径改变，需要重新绑定或重新创建任务。

## 11. 模型仓库约定

模型管理支持任意用户选择的仓库根目录，推荐结构：

```text
models_repository/
├── Project_A/
│   ├── yolov8x-pose-2026-07-29/
│   │   ├── args.yaml
│   │   ├── results.csv
│   │   ├── results.png
│   │   └── weights/
│   │       ├── best.pt
│   │       └── last.pt
│   └── another-run/
└── Project_B/
    └── yolov8n-obb-run/
```

每个包含 `args.yaml` 或 `weights/` 的训练结果目录会被识别为一个模型记录。支持识别的常见模型格式包括 `.pt`、`.pth`、`.onnx`、`.engine`、`.plan`、`.xml/.bin`、`.tflite` 和 `.torchscript`。

模型详情中的训练数据来源优先从训练参数、`dataset.yaml` 和 `preparation_manifest.json` 解析。缺少这些文件时，部分字段会显示未知，不会要求用户手工维护一份模型 JSON 注册表。

## 12. 操作中心工具

顶部“操作中心”提供独立工具入口：

- 图片与标注数量统计
- 图片与标注匹配检查
- 训练/测试集重复检查
- 原始数据重复审查
- 数据集统计报告
- 训练/测试集统计
- 查找关键点
- 数据集管理（测试集 + 训练集）
- 标签替换

这些工具保留为独立操作流程；训练中心的数据准备使用共享的数据准备核心，但不会依赖已经废弃的 `data_merge_andsplit_pose` 或 `data_Visualization_pose` 脚本。

## 13. 配置与持久化

### 13.1 项目内配置

| 路径 | 作用 |
| --- | --- |
| `resources/style.qss` | 全局 Qt 样式表 |
| `resources/*_review_template.json` | 四类任务的内置审查模板 |
| `resources/training_templates/*.json` | 四类任务的内置训练模板 |
| `deployment/local.env` | 当前机器的可选本地环境变量，不提交仓库 |
| `training/task_registry.sqlite3` | 训练任务状态与历史 |
| `training/tasks/` | 可检查的训练任务快照 |
| `.runtime/` | 缓存和瞬态运行信息 |

### 13.2 QSettings

应用使用组织名 `FilesProcessQT`、应用名 `ImageManager` 保存本机 UI 状态，主要包括：

- 上次打开的数据目录和选中图片。
- 正常窗口几何及分栏尺寸。
- 各任务的审查模板列表和当前模板。
- 模型仓库路径。
- 训练输出根目录和项目分组。

这些设置属于当前操作系统用户，不在项目仓库中。迁移到新机器后需要重新选择数据和模型路径。

## 14. 项目代码结构

```text
Files_process_QT/
├── main.py                          # Qt 应用入口和视图装配
├── app/
│   ├── controllers/
│   │   └── app_controller.py       # 数据管理主控制器和模块联动
│   ├── models/
│   │   ├── annotation_review.py    # 审查配置、规则和插件执行
│   │   ├── annotation_schema.py    # 从数据动态推断类别/关键点结构
│   │   ├── annotation_sync.py      # JSON 标注副本同步
│   │   ├── dataset_duplicates.py   # 原始数据内容重复审查
│   │   ├── dataset_preparation.py  # 数据收集、划分、YAML 和清单生成
│   │   ├── file_system.py          # 数据目录识别和文件关联
│   │   ├── label_tool.py           # X-AnyLabeling 安全启动
│   │   ├── model_registry.py       # 模型仓库扫描与元数据提取
│   │   ├── operations.py           # 文件增删改移动及标注联动
│   │   ├── review_decisions.py     # 人工审查结论持久化
│   │   ├── training_config.py      # 训练模板加载与校验
│   │   ├── training_job.py         # 训练任务契约
│   │   └── training_task_registry.py
│   ├── views/
│   │   ├── main_window.py          # 无边框主窗口和平台导航
│   │   ├── image_viewer.py         # 图片缩放与标注绘制
│   │   ├── detail_panel.py         # 图片信息、标注树和审查分析
│   │   ├── model_management.py     # 模型卡片和模型详情
│   │   ├── model_comparison.py     # 多模型图表对比
│   │   ├── training_management.py  # 训练全流程 UI
│   │   ├── training_charts.py      # 训练实时图表
│   │   ├── ui_effects.py           # 动态背景和悬停效果
│   │   └── *_template_dialog.py    # 模板高级配置界面
│   ├── tools/                       # 操作中心的独立工具
│   └── training_runner.py           # Ultralytics 子进程训练入口
├── resources/                       # QSS 与 JSON 模板
├── deployment/                      # 安装、启动、诊断和部署文档
├── tests/                           # 单元测试
├── models/                          # 本地模型缓存，默认不提交
├── training/                        # 任务和训练结果，默认不提交
└── .runtime/                        # 运行时缓存，默认不提交
```

### 14.1 主要运行链路

```text
main.py
  -> 创建 MainWindow / Data / Model / Training Views
  -> AppController 连接信号和数据状态
  -> 选择数据目录
  -> file_system 解析数据结构
  -> annotation_schema 推断当前任务结构
  -> annotation_review 执行模板和插件规则
  -> ImageViewer + DetailPanel 展示标注、问题与统计
```

训练链路：

```text
选择原始批次
  -> dataset_preparation 扫描与预检
  -> 按来源分层划分 train/val
  -> 生成 preparation_manifest.json 和 dataset.yaml
  -> training_config 合并模板与界面参数
  -> training_job 固化任务契约
  -> training/tasks/<id>/ 保存任务快照
  -> training_runner 调用 Ultralytics
  -> 结构化事件驱动任务监控与训练图表
  -> 训练结果由模型管理自动发现
```

## 15. 测试与开发验证

运行全部单元测试：

```bash
python -m unittest discover -s tests -v
```

无显示器环境可尝试：

```bash
QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -v
```

提交或发布前建议执行：

```bash
python deployment/doctor.py --require-label-tool
python -m unittest discover -s tests -v
bash deployment/run.sh
```

测试覆盖审查规则、动态数据结构、标注同步、数据准备、重复检测、图片缩放、模型解析、模型对比、训练配置、训练任务、训练监控、窗口几何和 UI 效果等核心模块。

## 16. 常见问题

### 16.1 X-AnyLabeling 报 `xcb` 插件冲突

先运行：

```bash
python deployment/doctor.py --require-label-tool
```

Ubuntu/Debian 常见系统依赖：

```bash
sudo apt install libxcb-cursor0 libxcb-xinerama0 libxkbcommon-x11-0 libgl1 libegl1
```

不要把 OpenCV 的 `cv2/qt/plugins` 手工配置为全局 `QT_QPA_PLATFORM_PLUGIN_PATH`。优先通过 `deployment/local.env` 指定标注工具解释器，并使用 `deployment/run.sh` 启动平台。

### 16.2 数据目录存在，但某任务显示没有标注

检查当前任务模板的 `annotation_dir`。例如 Pose 默认 `annotations`，Detection 默认 `annotations-det`。目录不存在时属于真实的数据状态，不会自动借用其他任务标注。

### 16.3 训练提示缺少 TXT

训练需要 YOLO TXT。若只有 X-AnyLabeling JSON，当前流程会提示缺失，不会假装数据已就绪。可以补齐 TXT，或在明确理解后使用跳过不完整样本/背景样本策略。

### 16.4 明明有更多类别，训练只识别到部分类别

检查以下项目：

1. 新 class id 是否实际出现在 train/val TXT 中。
2. `dataset.yaml` 的 `names` 是否覆盖到最大 class id。
3. TXT 每行列数是否与当前任务结构一致。
4. 验证集是否缺少该类别。
5. 是否在修改数据后复用了旧的 Ultralytics `.cache`。

重新扫描训练批次会根据实际数据重建或校验 `dataset.yaml`。平台不应通过固定类别列表过滤新类别。

### 16.5 Pose 绘图报 `cannot convert float infinity to integer`

这通常表示 YOLO TXT 中存在非有限坐标（`inf`、`-inf` 或 `nan`）或不合法归一化值。数据准备预检会检查数值有限性；应修复或移除对应样本后删除旧标签缓存并重新训练。该异常发生在线程绘图时也可能不立即终止训练，但数据本身必须处理。

### 16.6 训练提前停止，但界面显示满进度

Ultralytics 可能因 Early Stopping 在 `epochs` 前结束。平台完成状态应以 `results.csv` 的实际最后 Epoch 为准，并区分“任务完成”和“执行到最大 Epoch”。历史旧任务如果只保存了计划轮次，可能需要刷新或重新解析结果。

### 16.7 如何继续停止或异常中断的训练

在任务中心选择任务并执行继续训练。只有当前任务目录中的 `weights/last.pt` 可以恢复完整训练状态；`best.pt` 或普通预训练模型只能作为新训练权重，不能恢复原优化器和 Epoch。

### 16.8 文件名不同，为什么被判为重复

重复审查比较文件字节内容。名称、目录和修改时间不参与最终重复判定。界面会为每组保留排序后的第一个文件，只将其余完全相同的成员列为可删除副本。

### 16.9 最大化后无法恢复初始尺寸

当前窗口会在进入最大化前保存 `normalGeometry`，还原时恢复该尺寸。若本机历史设置异常，可清理应用的 `QSettings` 后重新启动；清理会同时丢失上次目录、分栏和模板选择记录。

## 17. 安全与数据保护

- 数据准备默认生成副本，不修改原始数据。
- 文件删除优先使用系统回收站；具体行为取决于功能入口和系统支持。
- 重复审查删除前会验证目标仍位于数据根目录、文件未发生变化，并始终保留每组一个副本。
- 人工通过/忽略结论不会改写 JSON。
- X-AnyLabeling 修改后的副本同步会改写关联 JSON，这是有意行为；修改前建议使用版本控制或数据快照。
- `training_data`、`test_data`、训练任务和模型结果可能包含绝对路径，跨机器迁移后应重新生成或进行路径映射。

## 18. 当前边界与后续方向

当前明确边界：

- 评估中心尚未完成独立工作流。
- 训练中心当前以本地单任务、单运行队列为主。
- 跟踪、多任务 MoE 和模型辅助标注尚未接入。
- 标注标准以 X-AnyLabeling JSON 为主，YOLO TXT 用于 Ultralytics 训练。
- 原始重复审查当前扫描 `images`、`annotations` 和 `labels`；任务专用 JSON 目录可在后续扩展为由任务配置动态列举。
- 左右反标属于几何启发式检查，复杂姿态必须保留人工复核环节。

推荐后续顺序：

1. 建立独立评估目录和评估任务注册表。
2. 让评估结果与模型卡片、测试数据来源双向联动。
3. 增加工作区导出/导入与跨机器路径映射。
4. 将原始重复审查扩展到所有任务模板声明的标注目录。
5. 增加训练队列资源调度和多 GPU 任务策略。

## 19. 部署资料

更聚焦的源码部署说明位于 [deployment/README.md](deployment/README.md)，其中包含：

- 新环境安装流程
- CPU/CUDA PyTorch 选择
- X-AnyLabeling 独立环境配置
- Qt/xcb 排查
- 环境诊断
- 数据和历史任务迁移注意事项
- 发布前验证清单

源码部署只安装应用本身，不自动复制数据集、训练输出或模型仓库。将项目分享给其他用户时，应分别规划源码、工作数据和模型资产的分发方式。
