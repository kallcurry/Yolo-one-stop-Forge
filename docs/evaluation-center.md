# 评估中心设计文档

> 状态：**阶段 1（引擎与数据）✅ 阶段 2（评估中心 UI）✅ 已实现并通过端到端验收**；阶段 3（模型卡片/对比联动）待开发
> 定位：对已训练模型在独立测试批次上的客观度量，并与模型管理、模型对比双向联动

## 1. 目标与闭环

评估中心回答一个核心问题：**这个模型在真实场景数据（测试批次）上表现如何？**

```
数据准备（test_data/<测试批次> 已就绪 + test_manifest.json）
   ↓
训练中心（模型出炉）──→ 模型管理（卡片）
   ↓                        ↑
评估中心（测试集客观度量）──┘   ← 结果回写卡片 + 参与模型对比
```

与数据准备的约定已在 `data-conventions.md` 与 `training-pipeline.md` 中固化：
- 测试样本独立写入 `test_data/<测试批次名>/`（images / annotations / labels / dataset.yaml / test_manifest.json）
- 训练批次 `preparation_manifest.json` 记录 `test_batch`、`test_ratio`、`test` 数量
- **评估中心的正式数据源只有测试批次**（训练批次自带 `val` 不参与正式评估；如需快速自检应另建测试批次）

## 2. 核心流程

```
选择模型（.pt）  →  选择测试批次  →  配置评估参数  →  创建任务（快照）
      ↓
任务注册表排队 → evaluation_runner 子进程执行
      ↓
事件流实时监控（进度 / 日志 / 指标） → 完成 → evaluation_result.json
      ↓
结果页展示 → 回写模型卡片 → 参与模型对比
```

## 3. 评价指标

ultralytics `YOLO(val)` 原生产出的指标按任务自动识别：

| 任务 | 主指标 | 明细 |
| --- | --- | --- |
| Detection | mAP50-95(B) / mAP50(B) | Precision / Recall / AR + per-class AP |
| Pose | mAP50-95(P) / mAP50(P) | Precision / Recall + 关键点维度 |
| Segmentation | mAP50-95(M) / mAP50(M) | Precision / Recall + per-class |
| OBB | mAP50-95(O) / mAP50(O) | Precision / Recall + per-class |

**平台附加指标**：

- **泛化差距**：`测试集 mAP50-95 − 训练侧 val mAP50-95`
  （训练侧指标来自模型自身的 `results.csv` / `args.yaml`，从模型卡片解析）
- **推理性能**：`速度 ms/img`、`FPS`（ultralytics 时延统计，设备与模型格式相关，仅供参考）
- **数据指纹**：评估任务记录测试批次 `test_manifest.json` 的 SHA-256——测试集内容变化后旧结果自动标记为"已失效"

## 4. 目录与产物约定

与训练中心同构（`training/tasks|runs` → `evaluation/...`）：

```text
evaluation/
├── task_registry.sqlite3            # 评估任务注册表（状态与历史）
├── tasks/
│   └── <task-id>/
│       ├── evaluation_request.json  # 任务契约快照
│       ├── run_evaluation.py        # 可独立执行的评估入口
│       └── evaluation.log
└── runs/
    └── <项目分组>/<模型名>@<测试批次名>/   # ultralytics val 原生输出
        ├── args.yaml
        ├── results.png / confusion_matrix.png / PR 曲线等
        └── evaluation_result.json   # 平台自建结果契约
```

### 4.1 evaluation_request.json（任务契约）

```json
{
  "task_id": "eval-20260827-0001",
  "task_type": "pose",
  "model_path": "/abs/path/to/weights/best.pt",
  "model_label": "yolov8x-pose-2026-08-25",
  "training_batch": "2026-08-25",
  "test_batch": "2026-08-27-test",
  "test_data_root": "/abs/test_data/2026-08-27-test",
  "test_dataset_yaml": "/abs/test_data/2026-08-27-test/dataset.yaml",
  "test_manifest_sha256": "…",
  "parameters": {"imgsz": 640, "batch": 16, "device": "0", "conf": 0.001, "iou": 0.6},
  "output_dir": "/abs/evaluation/runs/<project>/<model>@<test_batch>",
  "created_at": "…"
}
```

### 4.2 evaluation_result.json（结果契约）

```json
{
  "version": 1,
  "task_id": "eval-20260827-0001",
  "model_path": "…",
  "test_batch": "2026-08-27-test",
  "test_manifest_sha256": "…",
  "task_type": "pose",
  "metrics": {
    "mAP50-95": 0.83,
    "mAP50": 0.91,
    "precision": 0.89,
    "recall": 0.86
  },
  "per_class": {
    "person_dress_middle": {"mAP50-95": 0.84, "mAP50": 0.92}
  },
  "train_metrics": {"mAP50-95": 0.91, "mAP50": 0.95},
  "generalization_gap": -0.08,
  "latency": {"ms_per_image": 23.5, "fps": 42.6},
  "outputs": {
    "results_csv": "…/results.csv",
    "results_png": "…/results.png",
    "confusion_matrix_png": "…/confusion_matrix.png"
  },
  "created_at": "…"
}
```

## 5. 任务注册表与状态机

与训练任务一致，注册表为 `evaluation/task_registry.sqlite3`：

| 状态 | 含义 | 迁移 |
| --- | --- | --- |
| `draft` | 新建未提交 | → queued |
| `queued` | 等待执行（可暂停/取消） | → running / failed |
| `running` | 子进程执行中 | → completed / failed / stopped / interrupted |
| `completed` | 评估完成，结果已落盘 | 可重跑 |
| `stopped` / `interrupted` / `failed` | 终止 / 异常中断 / 失败 | 可重跑 |

**队列策略**：单任务顺序执行（同一时刻一个评估任务在跑）；多任务排入队列，逐个出队。任务快照可独立重跑（模型路径与测试批次路径为绝对路径，跨机需重新绑定）。

## 6. 模块设计（对应代码文件）

| 文件 | 职责 |
| --- | --- |
| `app/models/evaluation_job.py` | 任务契约 dataclass、参数校验、request 序列化 |
| `app/models/evaluation_task_registry.py` | sqlite 注册表（读写、恢复、状态迁移），参考 `training_task_registry` |
| `app/evaluation_runner.py` | 子进程入口：加载 ultralytics `YOLO(model).val()`，解析 `results_dict`，汇总 per-class、泛化差距、时延；结构化事件流（`@@FILESPROCESS_EVAL@@` 前缀 JSON 行） |
| `app/views/evaluation_management.py` | 评估中心 UI（任务中心 / 新建评估 / 监控 / 结果页） |
| `app/views/evaluation_charts.py` | 结果图表（per-class 条形、泛化差距、PR/混淆矩阵展示） |
| `app/models/model_registry.py`（扩展） | 读取 `evaluation_result.json` 写入模型卡片元数据 |
| `app/views/model_comparison.py`（扩展） | 对比表/图新增"测试集 mAP50-95"与"泛化差距" |

### 6.1 UI 结构（顶层第四模块「评估中心」，与训练中心对称）

```
评估中心
├── 任务中心    列表（状态徽章/进度/结果摘要）+ 右键菜单（重试/复制/删除/打开目录）
├── 新建评估    ① 选模型（模型仓库/最近训练） ② 选测试批次（test_data/<批次>）
│               ③ 参数（imgsz/batch/device/conf/iou，默认继承模型 args.yaml）
├── 监控        进度/日志/指标曲线（复用 training_charts 模式）
└── 结果        指标总览 + per-class 表 + 混淆矩阵/PR 图 + 泛化差距 + 关联模型卡片入口
```

## 7. 联动设计

| 联动点 | 行为 |
| --- | --- |
| 模型管理 | 卡片新增「评估结果」区：最近一次测试集 mAP50-95、泛化差距、测试批次；无评估显示"未评估"，点击跳转评估中心 |
| 模型对比 | 对比表新增列「测试集 mAP50-95」「泛化差距」；对比图可切换训练曲线 ⇄ 测试评估指标 |
| 训练中心 | 训练完成的模型在任务中心右键"去评估"（预选该模型） |
| 数据准备 | 测试批次 `test_manifest.json` 为评估任务的数据来源契约；评估任务记录其 SHA-256 |
| 结果可追溯 | 测试批次内容变化（指纹不匹配）→ 历史评估结果标记"数据已变更，结果可能失效" |

## 8. 一期边界与后续方向

**一期实现范围**：

- 仅本地 **Ultralytics `.pt`**（`best.pt` / `last.pt`）模型
- 正式数据源仅 **测试批次**（`test_data/<批次>/`；旧扁平"默认测试集"兼容）
- 单模型 × 单测试集 × 单任务，多任务排队顺序执行；无 GPU 资源调度
- onnx / engine / 自定义推理不在一期

**后续方向**：

1. onnx/engine 推理评估（自定义前向 + 后处理）
2. 多模型 × 多测试集矩阵评估（结果热力图）
3. FP/FN 样例可视化浏览（点击混淆矩阵单元格查看对应图片）
4. 自动"训练 → 评估 → 模型选择"循环（结合训练队列）
5. 多 GPU / 远程执行调度

## 9. 实施阶段与验收

| 阶段 | 内容 | 验收标准 |
| --- | --- | --- |
| 1 引擎与数据 ✅ | `evaluation_job` + 注册表 + `evaluation_runner`（纯模型层） | 单元测试通过；用现有模型 + 测试批次跑通一次子进程评估，产出符合契约的 `evaluation_result.json`（已验收：8.25-pose-2 × 6 张真实样本，mAP50-95=0.900） |
| 2 评估中心 UI ✅ | 顶部第四模块激活（任务中心/新建/监控/结果） | GUI 上完成一次"选模型→选测试→跑完→看结果"全流程（已验收：模型 2026-08-25-pose-2 × e2e 测试批次，UI 驱动全链路 completed） |
| 3 全局联动 | 模型卡片评估区 + 模型对比测试 mAP + 训练"去评估" + 文档 | 模型卡片展示评估结果；对比表出现测试集 mAP 列 |

**验收示例**（当前资产）：`training/runs/ShengSong/2026-08-25-pose-2/weights/best.pt` × 新生成的测试批次（如 `2026-08-27-test`），跑出 mAP/每类/泛化差距，并可在模型对比中与 2026-08-25-pose-2 等历史模型横向比较。

## 10. 安全与数据保护

- 评估为**只读**流程：不修改测试批次、不修改模型文件；输出仅落在 `evaluation/` 下
- 测试批次缺失/指纹不符 → 任务拒绝启动并说明原因
- 评估子进程崩溃不拖垮主窗口；应用重启可恢复任务历史（同训练中心模式）
- 事件流与日志写入任务目录而非全局，便于清理
