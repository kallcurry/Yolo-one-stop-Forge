# 项目代码结构与开发验证

> 本文档摘录自项目总说明，完整版见 [README](../README.md)。

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

### 主要运行链路

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
