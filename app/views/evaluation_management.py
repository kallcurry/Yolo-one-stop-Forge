"""Evaluation center UI: task center, new evaluation, monitor and results."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from PyQt5.QtCore import QProcess, Qt, QTimer
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.models.evaluation_job import (
    build_evaluation_job,
    load_evaluation_job,
    save_evaluation_job,
)
from app.models.evaluation_task_registry import (
    EvaluationTaskRecord,
    EvaluationTaskRegistry,
)
from app.views.tool_dialog import stored_dataset_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = PROJECT_ROOT / 'evaluation'

STATUS_TONES = {
    'queued': '#FFD07A',
    'running': '#62E8FF',
    'completed': '#45D483',
    'failed': '#FF6B6B',
    'stopped': '#FFB0B0',
    'interrupted': '#FFB0B0',
}


class EvaluationManagementView(QWidget):
    """Top-level evaluation module with its own serial task queue."""

    def __init__(self, parent=None, registry_path=None,
                 runner_script: str | None = None):
        super().__init__(parent)
        self._registry = EvaluationTaskRegistry(
            registry_path or (EVAL_ROOT / 'task_registry.sqlite3')
        )
        self._registry.recover_interrupted()
        self._runner_script = runner_script or str(
            PROJECT_ROOT / 'app' / 'evaluation_runner.py'
        )
        self._process: QProcess | None = None
        self._current_record: EvaluationTaskRecord | None = None
        self._selected_task: EvaluationTaskRecord | None = None

        self._build_ui()
        self.refresh_tasks()
        QTimer.singleShot(300, self._resume_queued)
        QTimer.singleShot(350, self._init_choices)

    def _init_choices(self):
        self.refresh_model_choices()
        self.refresh_test_batches()

    @staticmethod
    def _metric(parent_layout: QHBoxLayout, caption: str) -> QLabel:
        box = QWidget()
        box.setObjectName('trainingMetricCard')
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(10, 7, 10, 7)
        box_layout.setSpacing(1)
        caption_label = QLabel(caption)
        caption_label.setObjectName('trainingMetricCaption')
        value = QLabel('-')
        value.setObjectName('trainingMetricValue')
        box_layout.addWidget(caption_label)
        box_layout.addWidget(value)
        parent_layout.addWidget(box, 1)
        return value

    # ---- UI build ----

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel('评估中心')
        title.setObjectName('trainingTitle')
        header.addWidget(title)
        hint = QLabel('模型 × 测试批次 → 客观度量 → 结果回写模型卡片')
        hint.setObjectName('duplicateScope')
        header.addWidget(hint)
        header.addStretch()
        layout.addLayout(header)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_task_center_tab(), '任务中心')
        self.tabs.addTab(self._build_new_tab(), '新建评估')
        self.tabs.addTab(self._build_monitor_tab(), '监控')
        self.tabs.addTab(self._build_result_tab(), '结果')
        layout.addWidget(self.tabs, 1)

    def _build_task_center_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 8, 0, 0)

        metrics = QHBoxLayout()
        self.metric_total = self._metric(metrics, '总任务')
        self.metric_queued = self._metric(metrics, '排队中')
        self.metric_running = self._metric(metrics, '运行中')
        self.metric_done = self._metric(metrics, '已完成')
        self.metric_failed = self._metric(metrics, '失败')
        layout.addLayout(metrics)

        self.lbl_empty = QLabel('暂无评估任务 — 点击下方「新建评估」选择模型与测试批次')
        self.lbl_empty.setObjectName('evaluationEmpty')
        self.lbl_empty.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_empty)

        action_row = QHBoxLayout()
        btn_new = QPushButton('新建评估')
        btn_new.setObjectName('primaryBtn')
        btn_new.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        action_row.addWidget(btn_new)
        btn_retry = QPushButton('重试所选任务')
        btn_retry.setObjectName('successBtn')
        btn_retry.clicked.connect(self._retry_selected)
        action_row.addWidget(btn_retry)
        btn_open = QPushButton('打开结果目录')
        btn_open.setObjectName('fileOpBtn')
        btn_open.clicked.connect(self._open_selected_result)
        action_row.addWidget(btn_open)
        btn_delete = QPushButton('删除所选任务')
        btn_delete.setObjectName('dangerBtn')
        btn_delete.clicked.connect(self._delete_selected)
        action_row.addWidget(btn_delete)
        btn_refresh = QPushButton('刷新')
        btn_refresh.setObjectName('fileOpBtn')
        btn_refresh.clicked.connect(self.refresh_tasks)
        action_row.addWidget(btn_refresh)
        action_row.addStretch()
        layout.addLayout(action_row)

        self.task_table = QTableWidget(0, 6)
        self.task_table.setHorizontalHeaderLabels(
            ['任务', '模型', '测试批次', '状态', '摘要', '创建时间']
        )
        self.task_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.task_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.task_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.task_table.setAlternatingRowColors(True)
        self.task_table.verticalHeader().setVisible(False)
        header = self.task_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.task_table.itemSelectionChanged.connect(self._on_task_selected)
        self.task_table.itemDoubleClicked.connect(
            lambda _item: self._show_detail()
        )
        layout.addWidget(self.task_table, 1)

        self.task_detail = QLabel('选择任务查看详情；双击在“结果”页打开。')
        self.task_detail.setObjectName('duplicateSummary')
        self.task_detail.setWordWrap(True)
        layout.addWidget(self.task_detail)
        return widget

    def _build_new_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 8, 0, 0)

        section = QLabel('新建评估任务')
        section.setObjectName('trainingSectionTitle')
        layout.addWidget(section)

        form = QGridLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(10)

        form.addWidget(QLabel('数据根目录'), 0, 0)
        self.edit_root = QLineEdit(stored_dataset_path())
        self.edit_root.setObjectName('trainingEdit')
        self.edit_root.setPlaceholderText('测试批次所在的项目根目录')
        form.addWidget(self.edit_root, 0, 1, 1, 3)
        btn_root = QPushButton('选择')
        btn_root.setObjectName('fileOpBtn')
        btn_root.clicked.connect(self._pick_root)
        form.addWidget(btn_root, 0, 4)

        form.addWidget(QLabel('测试批次'), 1, 0)
        self.combo_test = QComboBox()
        self.combo_test.setObjectName('trainingCombo')
        form.addWidget(self.combo_test, 1, 1, 1, 3)
        btn_scan = QPushButton('刷新测试批次')
        btn_scan.setObjectName('fileOpBtn')
        btn_scan.clicked.connect(self.refresh_test_batches)
        form.addWidget(btn_scan, 1, 4)

        form.addWidget(QLabel('评估模型'), 2, 0)
        self.combo_model = QComboBox()
        self.combo_model.setObjectName('trainingCombo')
        self.combo_model.setEditable(True)
        form.addWidget(self.combo_model, 2, 1, 1, 3)
        btn_model = QPushButton('浏览 .pt')
        btn_model.setObjectName('fileOpBtn')
        btn_model.clicked.connect(self._browse_model)
        form.addWidget(btn_model, 2, 4)

        form.addWidget(QLabel('模型名称'), 3, 0)
        self.edit_model_label = QLineEdit()
        self.edit_model_label.setObjectName('trainingEdit')
        self.edit_model_label.setPlaceholderText('显示名，如 2026-08-25-pose-2')
        form.addWidget(self.edit_model_label, 3, 1, 1, 4)

        form.addWidget(QLabel('imgsz'), 4, 0)
        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setObjectName('trainingSpin')
        self.spin_imgsz.setRange(160, 2560)
        self.spin_imgsz.setSingleStep(32)
        self.spin_imgsz.setValue(640)
        form.addWidget(self.spin_imgsz, 4, 1)
        form.addWidget(QLabel('batch'), 4, 2)
        self.spin_batch = QSpinBox()
        self.spin_batch.setObjectName('trainingSpin')
        self.spin_batch.setRange(1, 256)
        self.spin_batch.setValue(16)
        form.addWidget(self.spin_batch, 4, 3)
        form.addWidget(QLabel('device'), 5, 0)
        self.edit_device = QLineEdit('0')
        self.edit_device.setObjectName('trainingEdit')
        self.edit_device.setToolTip('如 0、0,1 或 cpu')
        form.addWidget(self.edit_device, 5, 1)
        form.addWidget(QLabel('conf'), 5, 2)
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setObjectName('trainingSpin')
        self.spin_conf.setRange(0.0001, 1.0)
        self.spin_conf.setDecimals(4)
        self.spin_conf.setValue(0.001)
        form.addWidget(self.spin_conf, 5, 3)
        form.addWidget(QLabel('iou'), 6, 0)
        self.spin_iou = QDoubleSpinBox()
        self.spin_iou.setObjectName('trainingSpin')
        self.spin_iou.setRange(0.1, 1.0)
        self.spin_iou.setDecimals(3)
        self.spin_iou.setValue(0.6)
        form.addWidget(self.spin_iou, 6, 1)

        layout.addLayout(form)

        btn_row = QHBoxLayout()
        btn_create = QPushButton('创建评估任务')
        btn_create.setObjectName('primaryBtn')
        btn_create.clicked.connect(self._create_task)
        btn_row.addWidget(btn_create)
        btn_cancel = QPushButton('清空')
        btn_cancel.setObjectName('fileOpBtn')
        btn_cancel.clicked.connect(self._clear_form)
        btn_row.addWidget(btn_cancel)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.new_status = QLabel('测试批次为空？先在训练中心数据准备页设置“测试集比例”并生成。')
        self.new_status.setObjectName('duplicateScope')
        self.new_status.setWordWrap(True)
        layout.addWidget(self.new_status)
        layout.addStretch()
        return widget

    def _build_monitor_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 8, 0, 0)

        self.monitor_title = QLabel('当前任务：无')
        self.monitor_title.setObjectName('trainingTitle')
        layout.addWidget(self.monitor_title)
        self.monitor_status = QLabel('空闲')
        self.monitor_status.setObjectName('duplicateSummary')
        self.monitor_status.setWordWrap(True)
        layout.addWidget(self.monitor_status)

        btn_row = QHBoxLayout()
        btn_stop = QPushButton('停止当前任务')
        btn_stop.setObjectName('dangerBtn')
        btn_stop.clicked.connect(self._stop_current)
        btn_row.addWidget(btn_stop)
        btn_clear = QPushButton('清空日志')
        btn_clear.setObjectName('fileOpBtn')
        btn_clear.clicked.connect(lambda: self.monitor_log.clear())
        btn_row.addWidget(btn_clear)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.monitor_log = QPlainTextEdit()
        self.monitor_log.setObjectName('trainingLog')
        self.monitor_log.setReadOnly(True)
        layout.addWidget(self.monitor_log, 1)
        return widget

    def _build_result_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 8, 0, 0)

        self.result_title = QLabel('选择“任务中心”中已完成的任务以查看结果。')
        self.result_title.setObjectName('duplicateTitle')
        self.result_title.setWordWrap(True)
        layout.addWidget(self.result_title)

        self.metric_cards = QGridLayout()
        layout.addLayout(self.metric_cards)

        per_class_header = QLabel('按类别指标')
        per_class_header.setObjectName('duplicateDetailTitle')
        layout.addWidget(per_class_header)
        self.result_table = QTableWidget(0, 3)
        self.result_table.setHorizontalHeaderLabels(['类别', 'mAP50-95', 'mAP50'])
        self.result_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.result_table.verticalHeader().setVisible(False)
        header = self.result_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        layout.addWidget(self.result_table, 1)

        btn_row = QHBoxLayout()
        btn_open = QPushButton('打开结果目录')
        btn_open.clicked.connect(self._open_selected_result)
        btn_row.addWidget(btn_open)
        btn_row.addStretch()
        layout.addLayout(btn_row)
        return widget

    # ---- helpers ----

    def _pick_root(self):
        path = QFileDialog.getExistingDirectory(self, '选择数据根目录', self.edit_root.text())
        if path:
            self.edit_root.setText(path)
            self.refresh_test_batches()

    def _browse_model(self):
        path, _f = QFileDialog.getOpenFileName(
            self, '选择模型权重', str(PROJECT_ROOT / 'models'), '模型 (*.pt);;所有文件 (*)'
        )
        if path:
            self.combo_model.setEditText(path)
            if not self.edit_model_label.text().strip():
                self.edit_model_label.setText(Path(path).parent.parent.name)

    def refresh_test_batches(self):
        self.combo_test.clear()
        root = Path(self.edit_root.text().strip()).expanduser()
        if not root.is_dir():
            return
        test_root = root / 'test_data'
        if not test_root.is_dir():
            self.new_status.setText('未找到 test_data 目录：先在训练中心数据准备页设置“测试集比例”并生成测试批次。')
            return
        found = 0
        for candidate in sorted(test_root.iterdir()):
            if not candidate.is_dir():
                continue
            if (candidate / 'dataset.yaml').is_file():
                self.combo_test.addItem(candidate.name, str(candidate))
                found += 1
        self.new_status.setText(
            f'发现 {found} 个测试批次。' if found else
            'test_data 下没有带 dataset.yaml 的测试批次（旧“默认测试集”请先生成新批次）。'
        )

    def prefill_model(self, model_path: str, model_label: str):
        """Prefill the new-evaluation form from a model card (phase 3 link)."""
        resolved = str(Path(str(model_path)).expanduser().resolve())
        self.refresh_model_choices(keep_current=resolved)
        self.combo_model.setEditText(resolved)
        self.edit_model_label.setText(str(model_label))
        self.tabs.setCurrentIndex(1)
        self.new_status.setText('已预选模型，请选择测试批次后创建评估任务。')

    def refresh_model_choices(self, keep_current: str = ''):
        current = keep_current or self.combo_model.currentText()
        self.combo_model.clear()
        candidates: list[Path] = []
        runs_root = PROJECT_ROOT / 'training' / 'runs'
        if runs_root.is_dir():
            for best in runs_root.glob('*/weights/best.pt'):
                candidates.append(best)
            for last in runs_root.glob('*/weights/last.pt'):
                if last not in candidates:
                    candidates.append(last)
        models_root = PROJECT_ROOT / 'models'
        if models_root.is_dir():
            for weight in models_root.glob('*.pt'):
                candidates.append(weight)
        seen = set()
        for path in sorted(candidates, key=lambda p: p.name):
            if str(path) in seen:
                continue
            seen.add(str(path))
            label = path.parent.parent.name if path.parent.name == 'weights' else path.name
            self.combo_model.addItem(label, str(path))
        self.combo_model.setEditText(current)
        if not self.edit_model_label.text().strip() and current:
            try:
                self.edit_model_label.setText(Path(current).parent.parent.name)
            except (OSError, AttributeError):
                pass

    def _selected_model(self) -> tuple[Path, str]:
        text = self.combo_model.currentText().strip()
        if not text:
            raise ValueError('请选择或填写模型 .pt 路径')
        current_data = self.combo_model.currentData()
        if current_data and text == self.combo_model.itemText(self.combo_model.currentIndex()):
            model_path = Path(str(current_data))
        else:
            model_path = Path(text).expanduser()
        if not model_path.is_file():
            raise ValueError(f'模型文件不存在: {model_path}')
        label = self.edit_model_label.text().strip() or model_path.stem
        return model_path, label

    def _clear_form(self):
        self.edit_root.setText('')
        self.combo_test.clear()
        self.combo_model.clearEditText()
        self.edit_model_label.clear()

    # ---- task lifecycle ----

    def _create_task(self):
        try:
            model_path, model_label = self._selected_model()
        except ValueError as exc:
            QMessageBox.warning(self, '无法创建', str(exc))
            return
        test_data = self.combo_test.currentData()
        if not test_data:
            QMessageBox.warning(self, '无法创建', '请选择测试批次。')
            return
        test_root = Path(str(test_data)).expanduser().resolve()
        test_batch = self.combo_test.currentText()

        # 关联训练批次与任务类型：从测试批次 test_manifest.json 解析
        training_batch = ''
        task_type = ''
        training_run_dir = (
            model_path.parent.parent if model_path.parent.name == 'weights' else None
        )
        manifest = test_root / 'test_manifest.json'
        if manifest.is_file():
            try:
                payload = json.loads(manifest.read_text(encoding='utf-8'))
                training_batch = str(payload.get('training_batch') or '')
                task_type = str(payload.get('task_type') or '')
            except (OSError, ValueError):
                pass

        try:
            job = build_evaluation_job(
                model_path=model_path,
                model_label=model_label,
                test_data_root=test_root,
                test_batch=test_batch,
                task_type=task_type or _guess_task_type(
                    str(training_batch or test_batch)
                ),
                project_dir=str(EVAL_ROOT / 'runs'),
                run_name=f'{model_label}@{test_batch}',
                training_run_dir=training_run_dir,
                training_batch=training_batch or None,
                parameters={
                    'imgsz': int(self.spin_imgsz.value()),
                    'batch': int(self.spin_batch.value()),
                    'device': self.edit_device.text().strip() or '0',
                    'conf': float(self.spin_conf.value()),
                    'iou': float(self.spin_iou.value()),
                },
            )
            problems = job.validate()
            if problems:
                raise ValueError('；'.join(problems))
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, '无法创建', str(exc))
            return

        task_dir = EVAL_ROOT / 'tasks' / job.job_id
        task_dir.mkdir(parents=True, exist_ok=True)
        spec_path = save_evaluation_job(job, task_dir / 'evaluation_request.json')
        log_path = task_dir / 'evaluation.log'
        self._registry.create(
            job.to_dict(),
            task_dir=str(task_dir),
            output_dir=str(EVAL_ROOT / 'runs' / job.run_name),
            log_path=str(log_path),
        )
        self.refresh_tasks()
        self.tabs.setCurrentIndex(0)
        self._resume_queued()

    def _resume_queued(self):
        if self._process is not None and self._process.state() != QProcess.NotRunning:
            return
        record = self._registry.next_queued()
        if record is None:
            return
        self._start_record(record)

    def _start_record(self, record: EvaluationTaskRecord):
        spec_path = Path(record.task_dir) / 'evaluation_request.json'
        if not spec_path.is_file():
            self._registry.update(
                record.task_id, status='failed', error=f'任务快照缺失: {spec_path}'
            )
            self.refresh_tasks()
            self._resume_queued()
            return
        self._registry.update(record.task_id, status='running')
        self._current_record = self._registry.get(record.task_id)
        self.monitor_title.setText(f'当前任务：{record.task_id} · {record.spec.get("model_label", "")} @ {record.spec.get("test_batch", "")}')
        self.monitor_status.setText('启动中...')
        self.monitor_log.clear()

        process = QProcess(self)
        process.setWorkingDirectory(str(PROJECT_ROOT))
        env = __import__('PyQt5.QtCore', fromlist=['QProcessEnvironment']).QProcessEnvironment.systemEnvironment()
        runtime = PROJECT_ROOT / '.runtime'
        runtime.mkdir(parents=True, exist_ok=True)
        env.insert('MPLCONFIGDIR', str(runtime / 'matplotlib'))
        env.insert('YOLO_CONFIG_DIR', str(runtime / 'ultralytics'))
        env.insert('XDG_CACHE_HOME', str(runtime / 'cache'))
        env.insert('PYTHONPATH', str(PROJECT_ROOT) + os.pathsep + env.value('PYTHONPATH', ''))
        process.setProcessEnvironment(env)
        process.readyReadStandardOutput.connect(self._read_process_output)
        process.finished.connect(self._process_finished)
        process.start(sys.executable, [
            str(PROJECT_ROOT / 'app' / 'evaluation_runner.py'), str(spec_path),
        ])
        self._process = process

    def _read_process_output(self):
        if self._process is None:
            return
        text = bytes(self._process.readAllStandardOutput()).decode('utf-8', 'replace')
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('@@FILESPROCESS_EVAL@@'):
                try:
                    event = json.loads(stripped[len('@@FILESPROCESS_EVAL@@'):])
                    self._handle_event(event)
                except (ValueError, TypeError):
                    pass
            else:
                self.monitor_log.appendPlainText(stripped)

    def _handle_event(self, event: dict):
        event_type = event.get('type')
        if event_type == 'initializing':
            self.monitor_status.setText('正在初始化评估任务...')
        elif event_type == 'completed':
            metrics = event.get('metrics') or {}
            self.monitor_status.setText(
                f'评估完成（耗时 {event.get("elapsed", "-")}s）：'
                f"mAP50-95={metrics.get('mAP50-95')}  mAP50={metrics.get('mAP50')}"
            )
        elif event_type == 'failed':
            self.monitor_status.setText(f'评估失败：{event.get("message")}')
        elif event_type == 'cancelled':
            self.monitor_status.setText('评估已取消')

    def _process_finished(self, exit_code, _exit_status):
        record = self._current_record
        self._process = None
        if record is None:
            return
        if record.status == 'stopped':
            self._registry.update(
                record.task_id, error=f'用户停止（进程退出码 {exit_code}）'
            )
            self.refresh_tasks()
            self._resume_queued()
            return
        result_path = (
            Path(record.output_dir) / 'evaluation_result.json'
        )
        summary = ''
        if exit_code == 0 and result_path.is_file():
            try:
                payload = json.loads(result_path.read_text(encoding='utf-8'))
                metrics = payload.get('metrics') or {}
                summary = json.dumps({
                    'mAP50-95': metrics.get('mAP50-95'),
                    'mAP50': metrics.get('mAP50'),
                    'gap': payload.get('generalization_gap'),
                }, ensure_ascii=False)
                self._registry.update(record.task_id, status='completed', summary=summary)
            except (OSError, ValueError):
                self._registry.update(
                    record.task_id, status='failed',
                    error='结果文件解析失败',
                )
        else:
            error = '评估进程退出码非 0'
            self._registry.update(record.task_id, status='failed', error=error)
        self.refresh_tasks()
        self._resume_queued()

    def _stop_current(self):
        if self._process is not None and self._process.state() != QProcess.NotRunning:
            self._process.terminate()
            record = self._current_record
            if record is not None:
                self._registry.update(record.task_id, status='stopped', error='用户停止')
            self.refresh_tasks()

    # ---- task center interactions ----

    def refresh_tasks(self):
        records = self._registry.list_all()
        self.task_table.setRowCount(len(records))
        self._records = records
        self.metric_total.setText(str(len(records)))
        self.metric_queued.setText(
            str(sum(1 for r in records if r.status == 'queued'))
        )
        self.metric_running.setText(
            str(sum(1 for r in records if r.status == 'running'))
        )
        self.metric_done.setText(
            str(sum(1 for r in records if r.status == 'completed'))
        )
        self.metric_failed.setText(
            str(sum(1 for r in records if r.status in ('failed', 'interrupted', 'stopped')))
        )
        self.lbl_empty.setVisible(not records)
        for row, record in enumerate(records):
            metrics = record.metrics
            summary = (
                f"mAP50-95={metrics.get('mAP50-95')}" if metrics else (
                    record.error or ''
                )
            )
            values = (
                record.task_id,
                str(record.spec.get('model_label') or record.spec.get('model_path') or ''),
                str(record.spec.get('test_batch') or ''),
                record.status,
                summary,
                record.created_at,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 3:
                    item.setForeground(
                        __import__('PyQt5.QtGui', fromlist=['QColor']).QColor(
                            STATUS_TONES.get(record.status, '#D8E2EF')
                        )
                    )
                item.setData(Qt.UserRole, row)
                self.task_table.setItem(row, column, item)
        self._on_task_selected()

    def _on_task_selected(self):
        rows = self.task_table.selectionModel().selectedRows()
        if not rows or not hasattr(self, '_records'):
            return
        self._selected_task = self._records[rows[0].row()]
        self._show_detail()

    def _show_detail(self):
        record = self._selected_task
        if record is None:
            return
        detail = (
            f'任务 {record.task_id} · {record.status}\n'
            f'模型: {record.spec.get("model_path", "")}\n'
            f'测试批次: {record.spec.get("test_batch", "")} '
            f'（指纹 {str(record.spec.get("test_manifest_sha256", ""))[:16]}...）\n'
            f'输出: {record.output_dir}'
        )
        if record.error:
            detail += f'\n错误: {record.error}'
        self.task_detail.setText(detail)
        self._load_result(record)

    def _load_result(self, record: EvaluationTaskRecord):
        self.result_title.setText(
            f'任务 {record.task_id} · {record.spec.get("model_label", "")} '
            f'@ {record.spec.get("test_batch", "")}'
        )
        while self.metric_cards.count():
            item = self.metric_cards.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.result_table.setRowCount(0)
        if record.status != 'completed':
            self.result_title.setText(
                f'任务 {record.task_id} 尚未完成（{record.status}），无法查看结果。'
            )
            return
        result_file = Path(record.output_dir) / 'evaluation_result.json'
        if not result_file.is_file():
            self.result_title.setText('结果文件缺失。')
            return
        try:
            payload = json.loads(result_file.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            self.result_title.setText(f'结果解析失败：{exc}')
            return
        metrics = payload.get('metrics') or {}
        cards = [
            ('mAP50-95', metrics.get('mAP50-95')),
            ('mAP50', metrics.get('mAP50')),
            ('Precision', metrics.get('precision')),
            ('Recall', metrics.get('recall')),
            ('泛化差距', payload.get('generalization_gap')),
        ]
        for index, (name, value) in enumerate(cards):
            box = QWidget()
            box.setObjectName('trainingMetricCard')
            box_layout = QVBoxLayout(box)
            box_layout.setContentsMargins(10, 7, 10, 7)
            caption = QLabel(name)
            caption.setObjectName('trainingMetricCaption')
            value_label = QLabel(
                '-' if value is None else f'{value:.4f}'
            )
            value_label.setObjectName('trainingMetricValue')
            box_layout.addWidget(caption)
            box_layout.addWidget(value_label)
            self.metric_cards.addWidget(box, 0, index)
        per_class = payload.get('per_class') or {}
        self.result_table.setRowCount(len(per_class))
        for row, (class_name, values) in enumerate(sorted(per_class.items())):
            self.result_table.setItem(row, 0, QTableWidgetItem(str(class_name)))
            self.result_table.setItem(
                row, 1,
                QTableWidgetItem(
                    '-' if values.get('mAP50-95') is None
                    else f"{values['mAP50-95']:.4f}"
                ),
            )
            self.result_table.setItem(
                row, 2,
                QTableWidgetItem(
                    '-' if values.get('mAP50') is None
                    else f"{values['mAP50']:.4f}"
                ),
            )

    def _retry_selected(self):
        record = self._selected_task
        if record is None:
            return
        if record.status in ('queued', 'running'):
            QMessageBox.information(self, '重试', '任务正在执行或排队中。')
            return
        if not (Path(record.task_dir) / 'evaluation_request.json').is_file():
            QMessageBox.warning(self, '重试', '任务快照缺失，无法重试。')
            return
        self._registry.update(record.task_id, status='queued', error='')
        self.refresh_tasks()
        self._resume_queued()

    def _open_selected_result(self):
        record = self._selected_task
        if record is None:
            return
        target = Path(record.output_dir or record.task_dir)
        if not target.is_dir():
            QMessageBox.warning(self, '结果缺失', f'目录不存在: {target}')
            return
        import subprocess
        subprocess.Popen(['xdg-open', str(target)])

    def _delete_selected(self):
        record = self._selected_task
        if record is None:
            return
        if record.status in ('queued', 'running'):
            QMessageBox.warning(self, '删除', '执行中的任务不能删除。')
            return
        answer = QMessageBox.warning(
            self, '确认删除', f'删除任务 {record.task_id} 的注册记录？',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self._registry.delete(record.task_id)
        self._selected_task = None
        self.refresh_tasks()

    def closeEvent(self, event):
        if self._process is not None and self._process.state() != QProcess.NotRunning:
            self._process.terminate()
            self._process.waitForFinished(2000)
        self._registry.recover_interrupted()
        self._registry.close()
        event.accept()


def _guess_task_type(hint: str) -> str:
    """Best-effort task detection from a batch name or training batch."""
    lower = str(hint or '').lower()
    if 'det' in lower:
        return 'detection'
    if 'seg' in lower:
        return 'segmentation'
    if 'obb' in lower or 'rot' in lower:
        return 'obb'
    return 'pose'
