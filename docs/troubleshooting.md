# 常见问题

> 本文档摘录自项目总说明，完整版见 [README](../README.md)。

### X-AnyLabeling 报 `xcb` 插件冲突

先运行：

```bash
python deployment/doctor.py --require-label-tool
```

Ubuntu/Debian 常见系统依赖：

```bash
sudo apt install libxcb-cursor0 libxcb-xinerama0 libxkbcommon-x11-0 libgl1 libegl1
```

不要把 OpenCV 的 `cv2/qt/plugins` 手工配置为全局 `QT_QPA_PLATFORM_PLUGIN_PATH`。优先通过 `deployment/local.env` 指定标注工具解释器，并使用 `deployment/run.sh` 启动平台。

**直接 `python main.py` 启动失败（`Could not load the Qt platform plugin "xcb"`，路径指向 `cv2/qt/plugins`）**：

- 原因：shell 环境残留了指向 OpenCV Qt 插件的 `QT_QPA_PLATFORM_PLUGIN_PATH`/`QT_PLUGIN_PATH`，与 PyQt5 不兼容。
- 立即修复：`unset QT_QPA_PLATFORM_PLUGIN_PATH QT_PLUGIN_PATH`，然后重新启动。
- 自愈机制：平台启动时会自动检测并移除指向 `cv2` 的 Qt 插件路径（`sanitize_qt_environment`），无显示环境时自动回退 offscreen——建议始终使用 `bash deployment/run.sh`，新版本即使 shell 被污染也能启动。

### 数据目录存在，但某任务显示没有标注

检查当前任务模板的 `annotation_dir`。例如 Pose 默认 `annotations`，Detection 默认 `annotations-det`。目录不存在时属于真实的数据状态，不会自动借用其他任务标注。

### 训练提示缺少 TXT

训练需要 YOLO TXT。若只有 X-AnyLabeling JSON，当前流程会提示缺失，不会假装数据已就绪。可以补齐 TXT，或在明确理解后使用跳过不完整样本/背景样本策略。

### 明明有更多类别，训练只识别到部分类别

检查以下项目：

1. 新 class id 是否实际出现在 train/val TXT 中。
2. `dataset.yaml` 的 `names` 是否覆盖到最大 class id。
3. TXT 每行列数是否与当前任务结构一致。
4. 验证集是否缺少该类别。
5. 是否在修改数据后复用了旧的 Ultralytics `.cache`。

重新扫描训练批次会根据实际数据重建或校验 `dataset.yaml`。平台不应通过固定类别列表过滤新类别。

### Pose 绘图报 `cannot convert float infinity to integer`

这通常表示 YOLO TXT 中存在非有限坐标（`inf`、`-inf` 或 `nan`）或不合法归一化值。数据准备预检会检查数值有限性；应修复或移除对应样本后删除旧标签缓存并重新训练。该异常发生在线程绘图时也可能不立即终止训练，但数据本身必须处理。

### 训练提前停止，但界面显示满进度

Ultralytics 可能因 Early Stopping 在 `epochs` 前结束。平台完成状态应以 `results.csv` 的实际最后 Epoch 为准，并区分“任务完成”和“执行到最大 Epoch”。历史旧任务如果只保存了计划轮次，可能需要刷新或重新解析结果。

### 如何继续停止或异常中断的训练

在任务中心选择任务并执行继续训练。只有当前任务目录中的 `weights/last.pt` 可以恢复完整训练状态；`best.pt` 或普通预训练模型只能作为新训练权重，不能恢复原优化器和 Epoch。

### 文件名不同，为什么被判为重复

重复审查比较文件字节内容。名称、目录和修改时间不参与最终重复判定。界面会为每组保留排序后的第一个文件，只将其余完全相同的成员列为可删除副本。

### 最大化后无法恢复初始尺寸

当前窗口会在进入最大化前保存 `normalGeometry`，还原时恢复该尺寸。若本机历史设置异常，可清理应用的 `QSettings` 后重新启动；清理会同时丢失上次目录、分栏和模板选择记录。
