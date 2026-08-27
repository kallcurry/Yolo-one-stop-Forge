# 数据审查模板

> 本文档摘录自项目总说明，完整版见 [README](../README.md)。

内置模板位于：

```text
resources/
├── pose_review_template.json
├── detection_review_template.json
├── segmentation_review_template.json
└── obb_review_template.json
```

模板是审查策略，不是数据类别的永久真相。加载目录后，平台会从实际数据和可用的 `dataset.yaml` 推断类别、关键点、左右对和连接信息，再与当前模板的规则策略合并。

### 模板字段

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

### 内置审查规则

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

### 默认阈值

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

### 自定义参数化规则

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

### Python 审查插件

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
