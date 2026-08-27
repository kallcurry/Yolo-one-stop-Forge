# 训练管线

> 本文档摘录自项目总说明，完整版见 [README](../README.md)。

### 收集规则

对每个选中的原始批次，平台按文件主干匹配：

```text
images/<source>/sample.jpg
<annotation_dir>/<source>/sample.json
labels/<source>/sample.txt
```

默认要求 JSON 和 TXT 均存在且可解析。选择“跳过不完整样本”后，缺失 JSON/TXT 的图片不进入训练批次；选择“空标注作为背景”后，符合背景规则的无标签图片可以进入训练集。

### 划分规则

- 验证比例支持 `0.10 ~ 0.50`（默认 `0.20`），每个原始来源约 80% 训练、20% 验证。
- 测试集比例支持 `0 ~ 0.50`（默认 `0`，即不划出测试集；开启时需填写测试批次名）。
- 每个来源单独洗牌并抽取样本，防止大批次完全支配某个集合。
- 随机种子控制结果可复现。
- 会检查类别分布，并尽可能在每个集合中保留每个类别；单例或极稀有类别优先保留在训练集。
- **测试集优先划分**：先按测试比例逐来源抽取测试样本（类别感知、稀有类保护），剩余样本再按验证比例划分 `train/val`；测试样本独立写入 `test_data/<测试批次名>/`，训练批次中永不混入测试样本。
- **多任务数据划分**：标签目录按任务隔离（`labels` / `labels-det` / `labels-seg` / `labels-obb`，与标注目录约定一致）；每次准备在训练批次写入任务无关的 `split_manifest.json`（stem → train/val/test），勾选“复用上次划分”后，同来源/同比例/同种子的另一任务准备会复用该分配，保证同一张图跨任务都落在同一集合（跨任务比较合法、无数据泄漏）。
- 测试批次包含 `images/annotations/labels`、`dataset.yaml` 与 `test_manifest.json`（记录来源、比例、样本清单与关联的训练批次），供评估中心使用；训练批次 `preparation_manifest.json` 同步记录 `test_batch`/`test_ratio`/`test` 数量。
- 测试集中已登记的数据可按准备参数排除，避免数据泄漏。

### 动态 `dataset.yaml`

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

内置模板位于：

```text
resources/training_templates/
├── pose_training_template.json
├── detection_training_template.json
├── segmentation_training_template.json
└── obb_training_template.json
```

### 模板格式

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

### 平台管理参数

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

### 常用核心参数

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
