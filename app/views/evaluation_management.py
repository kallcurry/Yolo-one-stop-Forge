"""Evaluation center UI: task center, new evaluation, monitor and results."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from PyQt5.QtCore import QProcess, Qt, QTimer, QEvent
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
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
    QScrollArea,
    QSpinBox,
    QSplitter,
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
        self._records: list[EvaluationTaskRecord] = []
        self._visible_cards: list[QFrame] = []

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
        widget.setObjectName('evalTaskCenter')
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(10)

        metrics = QHBoxLayout()
        self.metric_total = self._metric(metrics, '总任务')
        self.metric_queued = self._metric(metrics, '排队中')
        self.metric_running = self._metric(metrics, '运行中')
        self.metric_done = self._metric(metrics, '已完成')
        self.metric_failed = self._metric(metrics, '失败')
        layout.addLayout(metrics)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel('查看'))
        self.filter_status = QComboBox()
        self.filter_status.setObjectName('trainingCombo')
        self.filter_status.addItem('全部状态', '')
        for label, value in (
            ('排队中', 'queued'),
            ('运行中', 'running'),
            ('已完成', 'completed'),
            ('失败/停止', 'failed'),
            ('中断', 'interrupted'),
            ('已停止', 'stopped'),
        ):
            self.filter_status.addItem(label, value)
        self.filter_status.currentIndexChanged.connect(self.refresh_tasks)
        self.filter_status.setMinimumWidth(140)
        toolbar.addWidget(self.filter_status)
        toolbar.addStretch()
        btn_new = QPushButton('新建评估')
        btn_new.setObjectName('primaryBtn')
        btn_new.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        toolbar.addWidget(btn_new)
        btn_refresh = QPushButton('刷新')
        btn_refresh.setObjectName('fileOpBtn')
        btn_refresh.clicked.connect(self.refresh_tasks)
        toolbar.addWidget(btn_refresh)
        layout.addLayout(toolbar)

        self.lbl_empty = QLabel('暂无评估任务 — 点击「新建评估」选择模型与测试批次')
        self.lbl_empty.setObjectName('evaluationEmpty')
        self.lbl_empty.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_empty)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        self.cards_host = QWidget()
        self.cards_host.setObjectName('evalTaskCardsHost')
        self.cards_grid = QGridLayout(self.cards_host)
        self.cards_grid.setContentsMargins(2, 2, 2, 2)
        self.cards_grid.setSpacing(14)
        self.cards_grid.setAlignment(Qt.AlignTop)
        scroll.setWidget(self.cards_host)
        layout.addWidget(scroll, 1)
        widget.installEventFilter(self)
        return widget

    def eventFilter(self, watched, event):
        if watched is self and event.type() == QEvent.Resize:
            QTimer.singleShot(0, self._relayout_cards)
        return super().eventFilter(watched, event)

    def _relayout_cards(self):
        if not hasattr(self, 'cards_grid') or not self._visible_cards:
            return
        width = self.cards_host.width()
        columns = 2 if width >= 860 else (1 if width >= 0 else 1)
        total = len(self._visible_cards)
        for index, card in enumerate(self._visible_cards):
            row, column = divmod(index, columns)
            self.cards_grid.addWidget(card, row, column)
        for column in range(columns):
            self.cards_grid.setColumnStretch(column, 1)

    def _build_new_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(12)

        section = QLabel('新建评估任务')
        section.setObjectName('trainingSectionTitle')
        layout.addWidget(section)

        content = QHBoxLayout()
        content.setSpacing(16)

        # 左：数据来源（带标题面板）
        source_box = QGroupBox('数据来源')
        source_layout = QGridLayout(source_box)
        source_layout.setHorizontalSpacing(10)
        source_layout.setVerticalSpacing(12)
        source_layout.setContentsMargins(14, 20, 14, 16)

        def _label(text: str) -> QLabel:
            label = QLabel(text)
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            return label

        source_layout.addWidget(_label('数据根目录'), 0, 0)
        self.edit_root = QLineEdit(stored_dataset_path())
        self.edit_root.setObjectName('trainingEdit')
        self.edit_root.setPlaceholderText('测试批次所在的项目根目录')
        source_layout.addWidget(self.edit_root, 0, 1)
        btn_root = QPushButton('选择')
        btn_root.setObjectName('fileOpBtn')
        btn_root.clicked.connect(self._pick_root)
        source_layout.addWidget(btn_root, 0, 2)

        source_layout.addWidget(_label('测试批次'), 1, 0)
        self.combo_test = QComboBox()
        self.combo_test.setObjectName('trainingCombo')
        source_layout.addWidget(self.combo_test, 1, 1)
        btn_scan = QPushButton('刷新测试批次')
        btn_scan.setObjectName('fileOpBtn')
        btn_scan.clicked.connect(self.refresh_test_batches)
        source_layout.addWidget(btn_scan, 1, 2)

        source_layout.addWidget(_label('评估模型'), 2, 0)
        self.combo_model = QComboBox()
        self.combo_model.setObjectName('trainingCombo')
        self.combo_model.setEditable(True)
        source_layout.addWidget(self.combo_model, 2, 1)
        btn_model = QPushButton('浏览 .pt')
        btn_model.setObjectName('fileOpBtn')
        btn_model.clicked.connect(self._browse_model)
        source_layout.addWidget(btn_model, 2, 2)

        source_layout.addWidget(_label('模型名称'), 3, 0)
        self.edit_model_label = QLineEdit()
        self.edit_model_label.setObjectName('trainingEdit')
        self.edit_model_label.setPlaceholderText('显示名，如 2026-08-25-pose-2')
        source_layout.addWidget(self.edit_model_label, 3, 1, 1, 2)
        source_layout.setColumnStretch(1, 1)
        content.addWidget(source_box, 3)

        # 右：评估参数（带标题面板，控件统一宽度）
        param_box = QGroupBox('评估参数')
        param_layout = QGridLayout(param_box)
        param_layout.setHorizontalSpacing(12)
        param_layout.setVerticalSpacing(12)
        param_layout.setContentsMargins(14, 20, 14, 16)

        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setObjectName('trainingSpin')
        self.spin_imgsz.setRange(160, 2560)
        self.spin_imgsz.setSingleStep(32)
        self.spin_imgsz.setValue(640)
        self.spin_batch = QSpinBox()
        self.spin_batch.setObjectName('trainingSpin')
        self.spin_batch.setRange(1, 256)
        self.spin_batch.setValue(16)
        self.edit_device = QLineEdit('0')
        self.edit_device.setObjectName('trainingEdit')
        self.edit_device.setToolTip('如 0、0,1 或 cpu')
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setObjectName('trainingSpin')
        self.spin_conf.setRange(0.0001, 1.0)
        self.spin_conf.setDecimals(4)
        self.spin_conf.setValue(0.001)
        self.spin_iou = QDoubleSpinBox()
        self.spin_iou.setObjectName('trainingSpin')
        self.spin_iou.setRange(0.1, 1.0)
        self.spin_iou.setDecimals(3)
        self.spin_iou.setValue(0.6)
        for control in (
            self.spin_imgsz, self.spin_batch, self.edit_device,
            self.spin_conf, self.spin_iou,
        ):
            control.setFixedWidth(132)

        param_layout.addWidget(_label('imgsz'), 0, 0)
        param_layout.addWidget(self.spin_imgsz, 0, 1)
        param_layout.addWidget(_label('batch'), 0, 2)
        param_layout.addWidget(self.spin_batch, 0, 3)
        param_layout.addWidget(_label('device'), 1, 0)
        param_layout.addWidget(self.edit_device, 1, 1)
        param_layout.addWidget(_label('conf'), 1, 2)
        param_layout.addWidget(self.spin_conf, 1, 3)
        param_layout.addWidget(_label('iou'), 2, 0)
        param_layout.addWidget(self.spin_iou, 2, 1)
        param_layout.setColumnStretch(1, 1)
        param_layout.setColumnStretch(3, 1)
        content.addWidget(param_box, 2)

        layout.addLayout(content)

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
        layout.setSpacing(12)

        panel = QWidget()
        panel.setObjectName('evaluationPanel')
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(16, 14, 16, 14)
        panel_layout.setSpacing(10)

        self.monitor_title = QLabel('当前任务：无')
        self.monitor_title.setObjectName('trainingTitle')
        panel_layout.addWidget(self.monitor_title)
        self.monitor_status = QLabel('空闲')
        self.monitor_status.setObjectName('duplicateSummary')
        self.monitor_status.setWordWrap(True)
        panel_layout.addWidget(self.monitor_status)

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
        panel_layout.addLayout(btn_row)
        layout.addWidget(panel)

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

        self.result_meta = QLabel('')
        self.result_meta.setObjectName('evaluationSectionHint')
        self.result_meta.setWordWrap(True)
        layout.addWidget(self.result_meta)

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
        btn_open.setObjectName('fileOpBtn')
        btn_open.clicked.connect(self._open_selected_result)
        btn_row.addWidget(btn_open)
        self.btn_output_csv = QPushButton('results.csv')
        self.btn_output_csv.setObjectName('fileOpBtn')
        self.btn_output_csv.setEnabled(False)
        self.btn_output_csv.clicked.connect(lambda: self._open_output('csv'))
        btn_row.addWidget(self.btn_output_csv)
        self.btn_output_png = QPushButton('训练曲线 PNG')
        self.btn_output_png.setObjectName('fileOpBtn')
        self.btn_output_png.setEnabled(False)
        self.btn_output_png.clicked.connect(lambda: self._open_output('png'))
        btn_row.addWidget(self.btn_output_png)
        self.btn_output_conf = QPushButton('混淆矩阵')
        self.btn_output_conf.setObjectName('fileOpBtn')
        self.btn_output_conf.setEnabled(False)
        self.btn_output_conf.clicked.connect(lambda: self._open_output('conf'))
        btn_row.addWidget(self.btn_output_conf)
        btn_row.addStretch()
        layout.addLayout(btn_row)
        return widget

    def _open_output(self, kind: str):
        record = self._selected_task
        if record is None or not record.output_dir:
            return
        mapping = {
            'csv': 'results.csv',
            'png': 'results.png',
            'conf': 'confusion_matrix.png',
        }
        target = Path(record.output_dir) / mapping.get(kind, '')
        if target.is_file():
            import subprocess
            subprocess.Popen(['xdg-open', str(target)])

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
        filter_value = self.filter_status.currentData() or ''
        visible = [
            record for record in records
            if not filter_value or record.status == filter_value
        ]
        while self.cards_grid.count():
            item = self.cards_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._visible_cards = [self._build_task_card(record) for record in visible]
        self._relayout_cards()
        self.lbl_empty.setVisible(not visible)

    def _build_task_card(self, record: EvaluationTaskRecord) -> QFrame:
        card = QFrame()
        card.setObjectName('evaluationTaskCard')
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        top = QHBoxLayout()
        status_badge = QLabel({
            'queued': '排队中', 'running': '运行中', 'completed': '已完成',
            'failed': '失败', 'stopped': '已停止', 'interrupted': '中断',
        }.get(record.status, record.status))
        status_badge.setObjectName('evalCardStatus')
        status_badge.setProperty('tone', record.status)
        top.addWidget(status_badge)
        top.addStretch()
        time_label = QLabel(record.created_at or '')
        time_label.setObjectName('evalCardTime')
        top.addWidget(time_label)
        layout.addLayout(top)

        title = QLabel(
            str(record.spec.get('model_label') or record.spec.get('model_path') or record.task_id)
        )
        title.setObjectName('evalCardTitle')
        title.setWordWrap(True)
        layout.addWidget(title)

        scope = QLabel(
            f'测试批次 {record.spec.get("test_batch", "-")} · {record.task_id}'
        )
        scope.setObjectName('evalCardScope')
        scope.setWordWrap(True)
        layout.addWidget(scope)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setObjectName('evalCardDivider')
        layout.addWidget(divider)

        metrics = record.metrics
        stats = QHBoxLayout()
        stats.addWidget(self._card_stat('mAP50-95', metrics.get('mAP50-95')))
        stats.addWidget(self._card_stat('泛化差距', metrics.get('gap')))
        stats.addWidget(self._card_stat('状态', record.error or '正常'))
        stats.addStretch()
        layout.addLayout(stats)

        actions = QHBoxLayout()
        btn_open = QPushButton('查看结果')
        btn_open.setObjectName('primaryBtn')
        btn_open.setFixedHeight(28)
        btn_open.clicked.connect(
            lambda _checked=False, r=record: self._open_result(r)
        )
        actions.addWidget(btn_open)
        btn_dir = QPushButton('结果目录')
        btn_dir.setObjectName('fileOpBtn')
        btn_dir.setFixedHeight(28)
        btn_dir.clicked.connect(
            lambda _checked=False, r=record: self._open_selected_result(r)
        )
        actions.addWidget(btn_dir)
        btn_retry = QPushButton('重试')
        btn_retry.setObjectName('successBtn')
        btn_retry.setFixedHeight(28)
        btn_retry.clicked.connect(
            lambda _checked=False, r=record: self._retry_selected(r)
        )
        actions.addWidget(btn_retry)
        btn_delete = QPushButton('删除')
        btn_delete.setObjectName('dangerBtn')
        btn_delete.setFixedHeight(28)
        btn_delete.clicked.connect(
            lambda _checked=False, r=record: self._delete_selected(r)
        )
        actions.addWidget(btn_delete)
        actions.addStretch()
        layout.addLayout(actions)
        return card

    @staticmethod
    def _card_stat(caption: str, value) -> QWidget:
        box = QWidget()
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(0, 0, 0, 0)
        box_layout.setSpacing(1)
        caption_label = QLabel(caption)
        caption_label.setObjectName('evalCardStatCaption')
        value_label = QLabel(
            '-' if value is None or value == ''
            else (f'{value:.4f}' if isinstance(value, (int, float)) else str(value))
        )
        value_label.setObjectName('evalCardStatValue')
        box_layout.addWidget(caption_label)
        box_layout.addWidget(value_label)
        return box

    def _open_result(self, record: EvaluationTaskRecord):
        self._selected_task = record
        self._load_result(record)
        self.tabs.setCurrentIndex(3)

    def _load_result(self, record: EvaluationTaskRecord):
        self.result_title.setText(
            f'任务 {record.task_id} · {record.spec.get("model_label", "")} '
            f'@ {record.spec.get("test_batch", "")}'
        )
        self.result_meta.setText('')
        for button in (self.btn_output_csv, self.btn_output_png, self.btn_output_conf):
            button.setEnabled(False)
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
        latency = payload.get('latency') or {}
        latency_text = (
            f"时延 {latency.get('ms_per_image')} ms · {latency.get('fps')} FPS"
            if latency.get('ms_per_image') is not None else '时延未记录'
        )
        self.result_meta.setText(
            f'测试批次 {payload.get("test_batch", "-")} · '
            f'训练批次 {payload.get("training_batch") or "-"} · '
            f'{latency_text} · '
            f'数据指纹 {str(payload.get("test_manifest_sha256", ""))[:12]}... · '
            f'{payload.get("created_at", "")}'
        )
        outputs = Path(record.output_dir)
        for button, name in (
            (self.btn_output_csv, 'results.csv'),
            (self.btn_output_png, 'results.png'),
            (self.btn_output_conf, 'confusion_matrix.png'),
        ):
            button.setEnabled((outputs / name).is_file())

    def _retry_selected(self, record: EvaluationTaskRecord | None = None):
        record = record or self._selected_task
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

    def _open_selected_result(self, record: EvaluationTaskRecord | None = None):
        record = record or self._selected_task
        if record is None:
            return
        target = Path(record.output_dir or record.task_dir)
        if not target.is_dir():
            QMessageBox.warning(self, '结果缺失', f'目录不存在: {target}')
            return
        import subprocess
        subprocess.Popen(['xdg-open', str(target)])

    def _delete_selected(self, record: EvaluationTaskRecord | None = None):
        record = record or self._selected_task
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
