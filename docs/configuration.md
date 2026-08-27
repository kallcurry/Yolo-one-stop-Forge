# 配置与持久化

> 本文档摘录自项目总说明，完整版见 [README](../README.md)。

### 项目内配置

| 路径 | 作用 |
| --- | --- |
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
