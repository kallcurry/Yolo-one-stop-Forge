# 数据目录与标注约定

> 本文档摘录自项目总说明，完整版见 [README](../README.md)。

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
├── test_data/                      # 测试数据（评估用）
│   ├── images/                     # 旧扁平格式（默认测试集），保留
│   ├── annotations/
│   ├── labels/
│   └── <批次名>/                   # 数据准备新生成的测试批次（评估中心使用）
│       ├── images/
│       ├── annotations/
│       ├── labels/
│       ├── dataset.yaml
│       └── test_manifest.json
└── videos/                         # 可选，当前不进入主要审查流程
```

说明：

- `annotations` 当前默认代表 Pose，但 `annotation_dir` 可在每个任务的审查模板中修改。
- 某个任务对应的标注目录不存在时，界面会明确显示“当前没有该任务的标注”，而不是回退到其他任务目录。
- 图片与 JSON/TXT 通过同名文件主干关联，例如 `image_001.jpg`、`image_001.json` 和 `image_001.txt`。
- 训练和测试目录属于派生数据。原始数据重复审查不会扫描这两个目录。
- 数据副本同步依赖文件名、任务标注目录和 `preparation_manifest.json`，应保留该清单。

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
