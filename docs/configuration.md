# 配置与持久化

> 本文档摘录自项目总说明，完整版见 [README](../README.md)。

### 项目内配置

| 路径 | 作用 |
| --- | --- |
| `resources/app_config.example.json` | 默认配置样板（复制为 `app_config.json` 后按需修改，个人配置不入库） |
| `resources/app_config.json` | 用户默认配置（gitignore），优先级高于 example |
| `resources/style.qss` | 全局 Qt 样式表 |
| `resources/*_review_template.json` | 四类任务的内置审查模板 |
| `resources/training_templates/*.json` | 四类任务的内置训练模板 |
| `deployment/local.env` | 当前机器的可选本地环境变量，不提交仓库 |
| `training/task_registry.sqlite3` | 训练任务状态与历史 |
| `training/tasks/` | 可检查的训练任务快照 |
| `.runtime/` | 缓存和瞬态运行信息 |

### QSettings

应用使用组织名 `FilesProcessQT`、应用名 `ImageManager` 保存本机 UI 状态，主要包括：

- 上次打开的数据目录和选中图片。
- 正常窗口几何及分栏尺寸。
- 各任务的审查模板列表和当前模板。
- 模型仓库路径。
- 训练输出根目录和项目分组。

这些设置属于当前操作系统用户，不在项目仓库中。迁移到新机器后需要重新选择数据和模型路径。

### 默认配置（app_config）

平台的可调默认值集中在 `app/models/app_defaults.py` 加载，配置优先级为：

```
代码内置 fallback  <  resources/app_config.example.json  <  resources/app_config.json
```

覆盖层级深合并，其中：

- `inference`：conf / iou / imgsz / device / half / max_fps / 相机尺寸 / 录制 fps 与编码器。
- `evaluation`：imgsz / batch / conf / iou。
- `preparation`：val_ratio / test_ratio / seed / use_copy。
- `extensions.images`：平台统一的图片扩展名清单（所有模块共用同一份）。
- `task_dirs`：各任务的标注/标签目录名（模板 > app_config > 代码 fallback）。

配置文件损坏时自动回退内置默认并记录日志，不会影响启动。修改后重启应用生效。

### 全局任务联动

顶部任务选择器（数据管理/训练中心/评估中心常驻）用于切换平台级任务
（POSE / DETECTION / SEGMENTATION / OBB）。切换后：

- 数据管理：审查模板、标注集下拉、当前图即时更新。
- 训练中心：数据准备模板与标注/标签目录预设同步。
- 评估中心：头部 `TASK · …` 徽章更新。
- 工具（标注转换与校验、统计工具）：按当前任务解析 schema 并预填标注配置。

任务类型持久化在 `lastTaskType`（QSettings），重启后自动恢复并广播到各中心。
