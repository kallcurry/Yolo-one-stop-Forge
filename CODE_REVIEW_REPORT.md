# YoloForge 全站代码审查报告

- 审查日期：2026-09-04
- 审查方式：只读审查（未修改任何代码）
- 范围：main.py、app/controllers、app/models、app/views、app/tools、tests、deployment、仓库健康（约 3.8 万行）
- 评级：P0 阻断（崩溃/数据丢失）｜P1 严重（功能失效/高风险）｜P2 建议｜P3 风格

## 总体评价

工程质量高于同类桌面工具的平均水准：MVC 边界基本干净（models 层无 Qt 依赖）、数据准备采用 staging+rename 原子模式、训练子进程有完整的状态机与 terminate→kill 兜底、删除操作层层设防（路径校验 + 运行中任务检查 + send2trash + 二次确认）、依赖全部 pin 版本、247 个测试函数。**未发现 P0 级问题**。主要短板集中在：推理中心线程生命周期管理、一处确定性的功能 Bug（文件夹重命名）、UI 线程同步重计算的冻结风险、以及一处重复定义导致新版 HUD 成为死代码（暴露缺乏 lint/CI）。

## 设计亮点

1. `app/models/annotation_sync.py:293-310` — `_atomic_replace`：tempfile + fsync + chmod + os.replace，教科书级原子写。
2. `app/models/dataset_preparation.py:466-686` — `prepare_dataset`：隐藏 staging 目录构建后 rename，拒绝覆盖既有目标，rmtree 仅清理自己的 staging。
3. `app/models/dataset_preparation.py:933-1001` — 分层划分：确定性种子、按来源独立分层、稀有类（≥2 样本）预留 1 个验证样本、单例类尽量留在 train。
4. `app/views/training_management.py:2148-2166, 2745-2750` — 删除任务的路径防护：`relative_to` + 目录深度校验，明确拒绝删除输出根/任务根。
5. `app/views/training_management.py:3023-3028` — 子进程环境隔离：MPLCONFIGDIR/YOLO_CONFIG_DIR/XDG_CACHE_HOME 全部指向 `.runtime`，不污染用户目录。
6. `app/views/training_management.py:3382-3395` — 训练停止：terminate() → 5 秒宽限 → 强制 kill。
7. `app/views/training_management.py:3156-3171` — 日志协议：JSON 事件前缀 + ANSI 转义清理 + 无效 JSON 容错。
8. `main.py:39-77` — 全局 excepthook 防 core dump；xcb 插件路径环境保护（OpenCV Qt 冲突）。
9. `app/models/inference_worker.py:199-210` — fp16 推理失败自动回退 FP32。
10. 仓库健康：git 仅追踪 90 个文件 / 1.6MB，无权重/数据/运行产物入库；`deployment/requirements-core.txt` 全部 pin 版本。

## 问题清单

### P1 严重

1. **[P1] `app/models/operations.py:132` + `app/controllers/app_controller.py:1871` — 文件夹重命名功能必然失败**
   右键菜单「重命名文件夹」（`app/views/dir_tree.py:302`）发出信号后，控制器调用 `rename_image(p, new_name, False)`，而该函数第 132 行 `if not old.is_file(): return f"'{old}' 不是文件"` —— 目录永远不满足 `is_file()`，用户必然收到「重命名失败: ... 不是文件」。需要独立的 `rename_folder` 实现。

2. **[P1] `app/models/inference_worker.py:490/572、511/593` — `_count_targets` 与 `_draw_hud` 重复定义，新版 HUD 是死代码**
   两个函数各定义两次，Python 后定义覆盖前定义。第一版 `_draw_hud`（511 行，字号随分辨率自适应、两行信息、动态宽度计数面板）永远不执行，实际生效的是 593 行的旧版。典型的合并/重构事故，`flake8` 的 F811 规则可直接抓住。

3. **[P1] 应用退出时推理线程不停止 — 退出路径崩溃风险**
   `app/controllers/app_controller.py:894-899` 的 `about_to_close` 只连接了 `shutdown_training`；`InferenceWorker` 运行中退出应用会触发 "QThread: Destroyed while thread is still running"，导致崩溃或未定义行为。`inference_center.py` 没有 closeEvent/停止钩子。

4. **[P1] `app/models/inference_worker.py:272-280` — 单张损坏图片中断整个图片目录循环**
   `_ImageListSource.read()` 遇损坏文件时 `cv2.imread` 返回 None → `ok=False` → `_run_loop` 直接 `break`。与 `% len` 循环播放的设计矛盾：一张坏图让整个目录浏览停止，而非跳过该图。

5. **[P1] `app/controllers/app_controller.py:1538-1677` — `_scan_review_stats` 在 UI 线程同步全量扫描，复杂度近似 O(n²)**
   逐图调用 `find_annotation(fmt=None)` → 每图触发 `detect_format` → `_has_direct_images` 全目录 `iterdir()+is_file()`。万图目录 = 数千万次 stat，UI 冻结可达分钟级。对比：数据准备已用 `_DatasetPreparationWorker(QThread)`（`training_management.py:137`），审查统计应同样下沉后台线程，且 `detect_format` 结果应按目录缓存。

### P2 建议

6. **[P2] `app/models/annotation_review.py:3194-3199` — Python 插件每次规则求值都重新 exec_module**
   无任何缓存（未注册 sys.modules 也无 dict 缓存）。千张图 × 每图一条 python 规则 = 千次模块重执行（含插件内 import）。另外 `_resolve_custom_plugin_path`（3215-3231）的候选包含 `Path.cwd()`，且模板 JSON 可指向任意 .py——分享模板即执行任意代码，建议首次加载时弹窗确认。

7. **[P2] `app/models/annotation_review.py:851` — `reorder_keypoints_file` 非原子覆写 JSON**
   `path.write_text(...)` 中途崩溃/断电会损坏标注文件。同项目 `annotation_sync._atomic_replace` 已有正确做法，应复用。

8. **[P2] `app/views/inference_center.py:614-618` — `_stop` 在 UI 线程同步 `wait(3000)`**
   停止期间 UI 冻结最多 3 秒；RTSP/慢 GPU 源超时后线程仍在运行，UI 状态与实际不同步。应改信号驱动（已有 `finished.connect(_on_worker_finished)`，只需去掉阻塞 wait）。

9. **[P2] `app/views/evaluation_management.py:1137-1143` — 评估停止只 terminate 无 kill 兜底**
   训练中心有 terminate→5s→kill（`training_management.py:3394-3395`），评估中心只 `terminate()`，且先把 registry 记为 stopped——进程若不响应 SIGTERM，状态与实际相反。closeEvent（1427-1430）同样只有 2 秒 waitForFinished。

10. **[P2] `app/models/inference_worker.py:272-280` — RTSP 断流无重连**
    网络抖动即终止推理循环，监控场景需要重连退避。

11. **[P2] `app/models/dataset_preparation.py:627-675` — 测试批次 rename 成功后主批次 rename 失败会残留孤儿测试批次**
    第 665 行 `test_staging.rename(test_target)` 先于第 671 行 `staging.rename(target)`；后者抛异常时 except 只清理 `staging`，已落位的 `test_target` 残留且引用了不存在的训练批次。

12. **[P2] `app/controllers/app_controller.py:461, 519` — `waitForStarted(4000)` 阻塞 UI 最多 4 秒**
    启动标注工具的两个路径均在 UI 线程同步等待；`errorOccurred` 已有信号处理，这里的同步兜底可去掉或改异步。

13. **[P2] `app/models/dataset_preparation.py:514-517` — 集合推导在过滤条件内重复求值**
    `sample.stem not in {sample.stem for sample in test_samples}` 每个样本重建一次集合，O(n×m)。应先 `test_stem_set = {...}` 再过滤。

14. **[P2] `app/models/file_system.py:393-405` — `create_folder` 未校验名称**
    标注集名有 `_annotation_dir_name` 防路径穿越（199-206），新建文件夹没有：`../x` 可在父目录之外创建目录。建议同样走 `_validate_simple_name` 类校验。

15. **[P2] `app/models/inference_worker.py:329-339` — 录制器无 `isOpened()` 检查、fps 假设固定**
    编码器初始化失败时静默丢帧，用户无感知；录制 fps 用 `min(max_fps, 25)` 常数，实际推理速率慢于该值时回放速度失真。分辨率中途变化也会与 VideoWriter 固定尺寸不匹配。

16. **[P2] `app/models/annotation_review.py:15-21, 211` — 模块级可变全局配置**
    `CURRENT_TASK_TYPE`/`TARGET_CLASSES`/`KEYPOINT_INDEX` 等由 `apply_pose_review_config` 原地突变。当前全部在 UI 线程串行调用是安全的，但任何后台线程一旦调用审查函数即数据竞态（目前数据准备线程未触碰，属预防性风险）。建议配置对象化后随调用显式传递。

17. **[P2] 工程化：无 CI，测试依赖未声明**
    247 个测试函数但没有 `.github/` 工作流；pytest 等测试依赖不在任何 requirements 中。最小动作：GitHub Actions 跑 `pytest + flake8 --select=F811`，可直接防住问题 2 的重复定义事故。

### P3 风格

18. **[P3] `main.py:147, 152`** — `win._on_key_a = ...` 猴子补丁装配、`action_open.triggered.disconnect()` 假设默认连接存在（MainWindow 改动即抛 TypeError）。
19. **[P3] `app/controllers/app_controller.py:131`** — `self._batch_annotate` 仅在 `open_directory` 赋值，`__init__` 未声明。
20. **[P3] `app/tools/dataset_stats.py:356` 等 9 处** — `__import__('PyQt5.QtWidgets', fromlist=[...])` 惰性惯用法应改为正常 import。
21. **[P3] `app/views/training_management.py`（3837 行）、`app/views/detail_panel.py`（1876 行）** — 巨石文件，训练中心可按「任务中心/数据准备/监控器」拆为组件。
22. **[P3] `app/models/operations.py:135`** — 注释声称"保持相同扩展名"但代码未实现；图片移动成功而标注移动失败时留下图文分离状态（已有报错提示，可接受）。

## 测试覆盖缺口

- 文件夹重命名/删除的控制器路径（正好漏掉了 P1-1）
- 推理 worker：录制器初始化失败、损坏图片源、RTSP 断流、退出时线程停止
- 评估中心：进程不响应 terminate 的停止路径、closeEvent
- 插件规则：重复加载行为、路径解析候选（cwd）
- `prepare_dataset` 主批次 rename 失败后的孤儿测试批次场景

## 修复优先级建议

1. P1-1 文件夹重命名（用户可感知、修复成本低）
2. P1-3 退出时停止推理线程（退出崩溃）
3. P1-2 删除重复定义的死代码 + 引入 F811 检查
4. P1-4 坏图跳过、P1-5 审查统计下沉后台线程
5. P2 按序处理；P3 随手清理
