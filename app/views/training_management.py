"""Training-center workspace and integrated dataset preparation flow."""

from __future__ import annotations

import csv
import json
import re
import shutil
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import yaml

from PyQt5.QtCore import (
    QProcess,
    QProcessEnvironment,
    QUrl,
    QSettings,
    Qt,
    QThread,
    QTimer,
    pyqtSignal,
)
from PyQt5.QtGui import QBrush, QColor, QDesktopServices
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.models.annotation_review import TASK_PRESETS
from app.models.dataset_preparation import (
    DatasetPreparationError,
    DatasetPreparationRequest,
    DatasetScanResult,
    PreparedDataset,
    TrainingBatchSummary,
    ensure_training_dataset_yaml,
    inspect_training_batch,
    list_source_batches,
    prepare_existing_batch_split,
    prepare_dataset,
    scan_dataset,
)
from app.models.training_config import (
    TrainingConfig,
    custom_training_template_dir,
    default_training_config,
    list_training_template_paths,
    load_training_config,
    training_config_from_dict,
    training_config_to_dict,
)
from app.models.training_job import (
    TRAIN_EVENT_PREFIX,
    TrainingJob,
    TrainingJobError,
    create_training_job,
    write_training_job,
)
from app.models.training_task_registry import (
    ACTIVE_STATUSES,
    TrainingTaskRecord,
    TrainingTaskRegistry,
    TrainingTaskRegistryError,
)
from app.views.training_template_dialog import TrainingTemplateDialog
from app.views.training_charts import METRIC_GROUPS, RealtimeTrainingChart
from app.views.ui_effects import HoverGlow


TASK_LABELS = {
    'pose': '姿态估计',
    'detection': '目标检测',
    'segmentation': '语义分割',
    'obb': '旋转框 OBB',
}
TASK_SHORT_LABELS = {
    'pose': 'Pose',
    'detection': 'Det',
    'segmentation': 'Seg',
    'obb': 'OBB',
}

TASK_CENTER_PAGE = 4
TASK_STATUS_META = {
    'draft': ('草稿', 'neutral'),
    'queued': ('排队中', 'info'),
    'preparing': ('准备中', 'info'),
    'running': ('训练中', 'running'),
    'stopping': ('停止中', 'warning'),
    'completed': ('已完成', 'success'),
    'failed': ('失败', 'danger'),
    'cancelled': ('已停止', 'warning'),
    'interrupted': ('异常中断', 'danger'),
    'archived': ('已归档', 'neutral'),
}
TASK_STATUS_COLORS = {
    'neutral': '#91A8B8',
    'info': '#62D7FF',
    'running': '#36B7FF',
    'success': '#57E397',
    'warning': '#F5C451',
    'danger': '#FF6677',
}


class _DatasetPreparationWorker(QThread):
    scan_ready = pyqtSignal(object)
    preparation_ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, request: DatasetPreparationRequest,
                 prepare: bool, parent=None):
        super().__init__(parent)
        self._request = request
        self._prepare = prepare

    def run(self):
        try:
            scan = scan_dataset(self._request)
            self.scan_ready.emit(scan)
            if not self._prepare or not scan.can_prepare:
                return
            prepared = prepare_dataset(self._request, scan)
            self.preparation_ready.emit(prepared)
        except Exception as exc:
            self.failed.emit(str(exc))


class TrainingManagementView(QWidget):
    """One-stop training workspace with shared operation-center data logic."""

    review_dataset_requested = pyqtSignal(str, str)
    dataset_prepared = pyqtSignal(str, str)
    training_completed = pyqtSignal(str, str)
    model_result_requested = pyqtSignal(str, str)
    status_message = pyqtSignal(str)

    def __init__(self, parent=None, task_registry_path: str | Path | None = None,
                 training_root: str | Path | None = None,
                 models_root: str | Path | None = None):
        super().__init__(parent)
        self.setObjectName('trainingManagementView')
        self._project_root = Path(__file__).resolve().parents[2]
        self._runtime_root = self._project_root / '.runtime'
        self._models_root = Path(
            models_root or self._project_root / 'models'
        ).expanduser().resolve()
        self._training_root = Path(
            training_root or self._project_root / 'training'
        ).expanduser().resolve()
        self._task_files_root = self._training_root / 'tasks'
        self._training_runs_root = self._training_root / 'runs'
        registry_path = (
            task_registry_path
            if task_registry_path is not None
            else self._training_root / 'task_registry.sqlite3'
        )
        self._task_registry = TrainingTaskRegistry(registry_path)
        self._dataset_root: Path | None = None
        self._task_type = 'pose'
        self._annotation_dir = 'annotations'
        self._label_dir = 'labels'
        self._scan_result: DatasetScanResult | None = None
        self._current_batch: Path | None = None
        self._batch_source_names: tuple[str, ...] = ()
        self._worker: _DatasetPreparationWorker | None = None
        self._training_config = default_training_config('pose')
        self._loading_training_template = False
        self._training_process: QProcess | None = None
        self._training_output_buffer = ''
        self._active_training_job: TrainingJob | None = None
        self._active_training_job_path: Path | None = None
        self._active_run_dir: Path | None = None
        self._training_terminal_event = ''
        self._training_stop_requested = False
        self._training_started_at = 0.0
        self._closing = False
        self._task_recovery_complete = False
        self._active_task_id = ''
        self._displayed_task_id = ''
        self._editing_task_id = ''
        self._task_items: dict[str, QTreeWidgetItem] = {}
        self._stats_process = None
        self._nvml = None
        self._nvml_handle = None
        self._metric_items: dict[str, QTreeWidgetItem] = {}
        self._auto_project_name = ''
        self._auto_run_name = ''
        self._build_ui()
        self._initialize_training_output()
        self._set_task('pose')
        self._recover_training_tasks()
        self._show_step(TASK_CENTER_PAGE)
        self._hover_glow = HoverGlow(self)
        self._hover_glow.watch_buttons(self)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(8)

        header = QWidget()
        header.setObjectName('trainingHeader')
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 12, 16, 12)
        header_layout.setSpacing(12)
        heading = QVBoxLayout()
        heading.setSpacing(1)
        eyebrow = QLabel('TRAINING ORCHESTRATOR')
        eyebrow.setObjectName('trainingEyebrow')
        title = QLabel('训练中心')
        title.setObjectName('trainingTitle')
        self.lbl_training_context = QLabel('等待选择数据项目')
        self.lbl_training_context.setObjectName('trainingContext')
        heading.addWidget(eyebrow)
        heading.addWidget(title)
        heading.addWidget(self.lbl_training_context)
        header_layout.addLayout(heading, 1)

        self.lbl_environment = QLabel('LOCAL · SINGLE JOB')
        self.lbl_environment.setObjectName('trainingEnvironmentBadge')
        self.lbl_environment.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(self.lbl_environment)
        root.addWidget(header)

        body = QHBoxLayout()
        body.setSpacing(8)
        self.step_rail = QWidget()
        self.step_rail.setObjectName('trainingStepRail')
        self.step_rail.setMinimumWidth(190)
        self.step_rail.setMaximumWidth(230)
        rail_layout = QVBoxLayout(self.step_rail)
        rail_layout.setContentsMargins(12, 16, 12, 14)
        rail_layout.setSpacing(7)
        rail_title = QLabel('训练工作台')
        rail_title.setObjectName('trainingRailTitle')
        rail_layout.addWidget(rail_title)

        self.step_group = QButtonGroup(self)
        self.step_group.setExclusive(True)
        self.btn_task_center = QPushButton('任务中心')
        self.btn_task_center.setObjectName('trainingTaskCenterBtn')
        self.btn_task_center.setCheckable(True)
        self.btn_task_center.setMinimumHeight(44)
        self.btn_task_center.clicked.connect(
            lambda _checked=False: self._show_step(TASK_CENTER_PAGE)
        )
        self.step_group.addButton(self.btn_task_center, TASK_CENTER_PAGE)
        rail_layout.addWidget(self.btn_task_center)
        flow_caption = QLabel('新建任务')
        flow_caption.setObjectName('trainingRailCaption')
        rail_layout.addWidget(flow_caption)
        self.step_buttons = []
        steps = (
            ('01', '数据准备'),
            ('02', '数据审查'),
            ('03', '训练配置'),
            ('04', '任务监控'),
        )
        for index, (number, label) in enumerate(steps):
            button = QPushButton(f'{number}   {label}')
            button.setObjectName('trainingStepBtn')
            button.setCheckable(True)
            button.setMinimumHeight(42)
            button.clicked.connect(
                lambda _checked=False, page=index: self._show_step(page)
            )
            self.step_group.addButton(button, index)
            self.step_buttons.append(button)
            rail_layout.addWidget(button)
        self.btn_task_center.setChecked(True)
        rail_layout.addStretch()

        self.lbl_task_state = QLabel('数据尚未准备')
        self.lbl_task_state.setObjectName('trainingRailState')
        self.lbl_task_state.setWordWrap(True)
        rail_layout.addWidget(self.lbl_task_state)
        body.addWidget(self.step_rail)

        self.content_stack = QStackedWidget()
        self.content_stack.setObjectName('trainingContentStack')
        self.data_page = self._build_data_page()
        self.review_page = self._build_review_page()
        self.config_page = self._build_config_page()
        self.monitor_page = self._build_monitor_page()
        self.task_center_page = self._build_task_center_page()
        for page in (
            self.data_page, self.review_page,
            self.config_page, self.monitor_page, self.task_center_page,
        ):
            self.content_stack.addWidget(page)
        body.addWidget(self.content_stack, 1)
        root.addLayout(body, 1)

    def _build_data_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName('trainingPage')
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 14, 16, 16)
        layout.setSpacing(12)

        title_row = QHBoxLayout()
        title = QLabel('训练数据准备')
        title.setObjectName('trainingSectionTitle')
        title_row.addWidget(title)
        title_row.addStretch()
        self.btn_refresh_sources = QToolButton()
        self.btn_refresh_sources.setObjectName('trainingIconBtn')
        self.btn_refresh_sources.setText('↻')
        self.btn_refresh_sources.setToolTip('刷新数据项目')
        self.btn_refresh_sources.setFixedSize(34, 34)
        self.btn_refresh_sources.clicked.connect(self.refresh_sources)
        title_row.addWidget(self.btn_refresh_sources)
        layout.addLayout(title_row)

        setup = QWidget()
        setup.setObjectName('trainingSetupPanel')
        setup_layout = QGridLayout(setup)
        setup_layout.setContentsMargins(14, 12, 14, 12)
        setup_layout.setHorizontalSpacing(10)
        setup_layout.setVerticalSpacing(9)

        setup_layout.addWidget(QLabel('任务类型'), 0, 0)
        self.task_combo = QComboBox()
        self.task_combo.setObjectName('trainingCombo')
        for task_type, label in TASK_LABELS.items():
            self.task_combo.addItem(label, task_type)
        self.task_combo.currentIndexChanged.connect(
            lambda: self._set_task(self.task_combo.currentData())
        )
        setup_layout.addWidget(self.task_combo, 0, 1)

        setup_layout.addWidget(QLabel('数据项目'), 0, 2)
        self.dataset_root_edit = QLineEdit()
        self.dataset_root_edit.setObjectName('trainingPathEdit')
        self.dataset_root_edit.setPlaceholderText('选择包含 images 的数据项目')
        self.dataset_root_edit.editingFinished.connect(
            self._on_dataset_root_edited
        )
        setup_layout.addWidget(self.dataset_root_edit, 0, 3)
        self.btn_browse_dataset = QPushButton('选择目录')
        self.btn_browse_dataset.setObjectName('secondaryBtn')
        self.btn_browse_dataset.clicked.connect(self._choose_dataset_root)
        setup_layout.addWidget(self.btn_browse_dataset, 0, 4)
        setup_layout.setColumnStretch(1, 1)
        setup_layout.setColumnStretch(3, 2)
        layout.addWidget(setup)

        self.data_source_splitter = QSplitter(Qt.Horizontal)
        self.data_source_splitter.setObjectName('trainingDataSourceSplitter')
        self.data_source_splitter.setChildrenCollapsible(False)
        self.data_source_splitter.addWidget(self._build_existing_source_panel())
        self.data_source_splitter.addWidget(self._build_raw_source_panel())
        self.data_source_splitter.setStretchFactor(0, 2)
        self.data_source_splitter.setStretchFactor(1, 3)
        self.data_source_splitter.setSizes([420, 540])
        layout.addWidget(self.data_source_splitter, 1)

        footer = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setObjectName('trainingPreparationProgress')
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        footer.addWidget(self.progress, 1)
        self.lbl_preparation_state = QLabel('等待数据扫描')
        self.lbl_preparation_state.setObjectName('trainingPreparationState')
        footer.addWidget(self.lbl_preparation_state)
        layout.addLayout(footer)
        return page

    def _build_raw_source_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName('trainingSourcePanel')
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(9)

        heading = QHBoxLayout()
        raw_title = QLabel('创建训练批次')
        raw_title.setObjectName('trainingPanelTitle')
        heading.addWidget(raw_title)
        heading.addStretch()
        self.btn_select_all_sources = QToolButton()
        self.btn_select_all_sources.setObjectName('trainingBatchSelectionBtn')
        self.btn_select_all_sources.setText('全选')
        self.btn_select_all_sources.setToolTip('选择全部原始批次')
        self.btn_select_all_sources.clicked.connect(
            lambda: self._set_all_source_checks(Qt.Checked)
        )
        heading.addWidget(self.btn_select_all_sources)
        self.btn_clear_sources = QToolButton()
        self.btn_clear_sources.setObjectName('trainingBatchSelectionBtn')
        self.btn_clear_sources.setText('清空')
        self.btn_clear_sources.setToolTip('清除全部原始批次选择')
        self.btn_clear_sources.clicked.connect(
            lambda: self._set_all_source_checks(Qt.Unchecked)
        )
        heading.addWidget(self.btn_clear_sources)
        raw_badge = QLabel('RAW  →  TRAIN / VAL')
        raw_badge.setObjectName('trainingPanelBadge')
        heading.addWidget(raw_badge)
        layout.addLayout(heading)

        self.source_tree = QTreeWidget()
        self.source_tree.setObjectName('trainingSourceTree')
        self.source_tree.setHeaderLabels(
            ['选择', '原始批次', '图片', 'JSON', 'TXT', '状态']
        )
        self.source_tree.setRootIsDecorated(False)
        self.source_tree.setAlternatingRowColors(False)
        self.source_tree.setUniformRowHeights(True)
        # Keep the controls below the table usable in compact windows. The
        # table remains scrollable, so showing fewer rows is preferable to a
        # layout overlap when the main window is restored to its initial size.
        self.source_tree.setMinimumHeight(96)
        self.source_tree.setSelectionMode(QAbstractItemView.NoSelection)
        self.source_tree.itemChanged.connect(self._on_source_selection_changed)
        self.source_tree.itemClicked.connect(self._on_source_row_clicked)
        self.source_tree.setToolTip('点击任意单元格即可选择或取消当前批次')
        header = self.source_tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, header.ResizeToContents)
        header.setSectionResizeMode(1, header.Stretch)
        for column in range(2, 5):
            header.setSectionResizeMode(column, header.ResizeToContents)
        header.setSectionResizeMode(5, header.Fixed)
        self.source_tree.setColumnWidth(5, 78)
        layout.addWidget(self.source_tree, 1)

        self.lbl_batch_source_hint = QLabel('选择已有批次后，这里显示其数据来源')
        self.lbl_batch_source_hint.setObjectName('trainingBatchSourceHint')
        self.lbl_batch_source_hint.setWordWrap(True)
        self.lbl_batch_source_hint.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.lbl_batch_source_hint)

        options = QGridLayout()
        options.setHorizontalSpacing(10)
        options.setVerticalSpacing(8)
        options.addWidget(QLabel('输出批次'), 0, 0)
        self.target_name_edit = QLineEdit(datetime.now().strftime('%Y-%m-%d'))
        self.target_name_edit.setObjectName('trainingEdit')
        options.addWidget(self.target_name_edit, 0, 1, 1, 5)

        options.addWidget(QLabel('验证比例'), 1, 0)
        self.val_ratio_spin = QDoubleSpinBox()
        self.val_ratio_spin.setObjectName('trainingSpin')
        self.val_ratio_spin.setRange(0.05, 0.5)
        self.val_ratio_spin.setSingleStep(0.05)
        self.val_ratio_spin.setValue(0.2)
        self.val_ratio_spin.setSuffix('  / VAL')
        options.addWidget(self.val_ratio_spin, 1, 1)

        options.addWidget(QLabel('随机种子'), 1, 2)
        self.seed_spin = QSpinBox()
        self.seed_spin.setObjectName('trainingSpin')
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        options.addWidget(self.seed_spin, 1, 3)

        options.addWidget(QLabel('写入方式'), 2, 0)
        self.write_mode_combo = QComboBox()
        self.write_mode_combo.setObjectName('trainingCombo')
        self.write_mode_combo.addItem('复制（独立副本）', True)
        self.write_mode_combo.addItem('硬链接（会关联源文件）', False)
        self.write_mode_combo.setToolTip(
            '默认复制以保护原始数据；硬链接节省空间，但修改副本也会修改源文件'
        )
        options.addWidget(self.write_mode_combo, 2, 1, 1, 3)

        self.background_checkbox = QPushButton('空标注作背景')
        self.background_checkbox.setObjectName('trainingOptionBtn')
        self.background_checkbox.setCheckable(True)
        self.background_checkbox.setToolTip(
            '空 JSON 且缺少 TXT 时，确认生成空 TXT 作为背景样本'
        )
        options.addWidget(self.background_checkbox, 2, 4, 1, 2)
        options.setColumnStretch(1, 1)
        options.setColumnStretch(3, 1)
        layout.addLayout(options)

        actions = QHBoxLayout()
        self.lbl_source_summary = QLabel('尚未选择原始批次')
        self.lbl_source_summary.setObjectName('trainingSourceSummary')
        actions.addWidget(self.lbl_source_summary, 1)
        self.btn_scan = QPushButton('扫描与预检')
        self.btn_scan.setObjectName('secondaryBtn')
        self.btn_scan.clicked.connect(lambda: self._start_dataset_job(False))
        actions.addWidget(self.btn_scan)
        self.btn_prepare = QPushButton('生成训练数据')
        self.btn_prepare.setObjectName('primaryBtn')
        self.btn_prepare.setProperty('emphasis', 'strong')
        self.btn_prepare.setEnabled(False)
        self.btn_prepare.setToolTip(
            '自动扫描与预检，只将图片、JSON 和 TXT 配对完整的样本写入训练副本'
        )
        self.btn_prepare.clicked.connect(lambda: self._start_dataset_job(True))
        actions.addWidget(self.btn_prepare)
        layout.addLayout(actions)
        return panel

    def _build_existing_source_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName('trainingSourcePanel')
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(9)

        row = QHBoxLayout()
        existing_title = QLabel('现有训练数据')
        existing_title.setObjectName('trainingPanelTitle')
        row.addWidget(existing_title)
        self.lbl_existing_count = QLabel('0 个批次')
        self.lbl_existing_count.setObjectName('trainingPanelBadge')
        row.addWidget(self.lbl_existing_count)
        row.addStretch()
        self.btn_refresh_batches = QToolButton()
        self.btn_refresh_batches.setObjectName('trainingIconBtn')
        self.btn_refresh_batches.setText('↻')
        self.btn_refresh_batches.setToolTip('刷新现有训练批次')
        self.btn_refresh_batches.setFixedSize(34, 34)
        self.btn_refresh_batches.clicked.connect(self.refresh_existing_batches)
        row.addWidget(self.btn_refresh_batches)
        layout.addLayout(row)

        self.existing_batch_tree = QTreeWidget()
        self.existing_batch_tree.setObjectName('trainingExistingBatchTree')
        self.existing_batch_tree.setHeaderLabels(
            ['训练批次', '任务', '样本', '划分', '状态']
        )
        self.existing_batch_tree.setRootIsDecorated(False)
        self.existing_batch_tree.setAlternatingRowColors(False)
        self.existing_batch_tree.setUniformRowHeights(True)
        self.existing_batch_tree.setSelectionMode(
            QAbstractItemView.SingleSelection
        )
        existing_header = self.existing_batch_tree.header()
        existing_header.setStretchLastSection(False)
        existing_header.setSectionResizeMode(0, existing_header.Stretch)
        for column in range(1, 4):
            existing_header.setSectionResizeMode(
                column, existing_header.ResizeToContents
            )
        existing_header.setSectionResizeMode(4, existing_header.Fixed)
        self.existing_batch_tree.setColumnWidth(4, 72)
        self.existing_batch_tree.currentItemChanged.connect(
            self._inspect_selected_existing_batch
        )
        self.existing_batch_tree.itemDoubleClicked.connect(
            lambda _item, _column: self._use_existing_batch()
        )
        layout.addWidget(self.existing_batch_tree, 1)

        self.lbl_existing_state = QLabel('尚未选择训练批次')
        self.lbl_existing_state.setObjectName('trainingSourceSummary')
        self.lbl_existing_state.setWordWrap(True)
        layout.addWidget(self.lbl_existing_state)

        actions = QHBoxLayout()
        self.btn_view_existing = QPushButton('查看数据')
        self.btn_view_existing.setObjectName('secondaryBtn')
        self.btn_view_existing.setEnabled(False)
        self.btn_view_existing.setToolTip('在数据管理模块中打开当前训练批次')
        self.btn_view_existing.clicked.connect(self._view_existing_batch)
        actions.addWidget(self.btn_view_existing)
        self.btn_open_existing_folder = QPushButton('打开目录')
        self.btn_open_existing_folder.setObjectName('secondaryBtn')
        self.btn_open_existing_folder.setEnabled(False)
        self.btn_open_existing_folder.setToolTip('打开当前训练批次所在目录')
        self.btn_open_existing_folder.clicked.connect(
            self._open_existing_batch_folder
        )
        actions.addWidget(self.btn_open_existing_folder)
        self.btn_rename_existing = QPushButton('重命名')
        self.btn_rename_existing.setObjectName('secondaryBtn')
        self.btn_rename_existing.setEnabled(False)
        self.btn_rename_existing.setToolTip('修改当前训练批次目录名称')
        self.btn_rename_existing.clicked.connect(self._rename_existing_batch)
        actions.addWidget(self.btn_rename_existing)
        self.btn_delete_existing = QPushButton('删除批次')
        self.btn_delete_existing.setObjectName('dangerBtn')
        self.btn_delete_existing.setEnabled(False)
        self.btn_delete_existing.setToolTip(
            '删除当前训练批次及其 train/val、YAML、审查报告，不影响原始数据'
        )
        self.btn_delete_existing.clicked.connect(self._delete_existing_batch)
        actions.addWidget(self.btn_delete_existing)
        actions.addStretch()
        self.btn_use_existing = QPushButton('使用此训练批次')
        self.btn_use_existing.setObjectName('primaryBtn')
        self.btn_use_existing.setEnabled(False)
        self.btn_use_existing.clicked.connect(self._use_existing_batch)
        actions.addWidget(self.btn_use_existing)
        layout.addLayout(actions)
        return panel

    def _build_review_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName('trainingPage')
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 14, 16, 16)
        layout.setSpacing(12)
        title = QLabel('训练数据审查')
        title.setObjectName('trainingSectionTitle')
        layout.addWidget(title)

        self.review_status_panel = QWidget()
        self.review_status_panel.setObjectName('trainingReviewStatusPanel')
        status_layout = QVBoxLayout(self.review_status_panel)
        status_layout.setContentsMargins(18, 16, 18, 16)
        self.lbl_review_status = QLabel('尚未选择训练数据')
        self.lbl_review_status.setObjectName('trainingReviewStatus')
        self.lbl_review_batch = QLabel('-')
        self.lbl_review_batch.setObjectName('trainingReviewPath')
        self.lbl_review_batch.setTextInteractionFlags(Qt.TextSelectableByMouse)
        status_layout.addWidget(self.lbl_review_status)
        status_layout.addWidget(self.lbl_review_batch)
        layout.addWidget(self.review_status_panel)

        self.review_summary_tree = QTreeWidget()
        self.review_summary_tree.setObjectName('trainingBatchSummaryTree')
        self.review_summary_tree.setHeaderLabels(['数据项', '数量', '检查结果'])
        self.review_summary_tree.setRootIsDecorated(False)
        self._configure_summary_header(self.review_summary_tree)
        layout.addWidget(self.review_summary_tree, 1)

        actions = QHBoxLayout()
        self.btn_back_to_data = QPushButton('返回数据准备')
        self.btn_back_to_data.setObjectName('secondaryBtn')
        self.btn_back_to_data.clicked.connect(lambda: self._show_step(0))
        actions.addWidget(self.btn_back_to_data)
        actions.addStretch()
        self.btn_recheck_batch = QPushButton('重新检测标签')
        self.btn_recheck_batch.setObjectName('secondaryBtn')
        self.btn_recheck_batch.setEnabled(False)
        self.btn_recheck_batch.clicked.connect(self._refresh_current_batch)
        actions.addWidget(self.btn_recheck_batch)
        self.btn_open_review = QPushButton('审查训练数据')
        self.btn_open_review.setObjectName('primaryBtn')
        self.btn_open_review.setEnabled(False)
        self.btn_open_review.clicked.connect(self._request_review)
        actions.addWidget(self.btn_open_review)
        layout.addLayout(actions)
        return page

    def _build_config_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName('trainingPage')
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 14, 16, 16)
        layout.setSpacing(10)

        title_row = QHBoxLayout()
        title = QLabel('训练配置')
        title.setObjectName('trainingSectionTitle')
        title_row.addWidget(title)
        title_row.addStretch()
        self.lbl_training_parameter_count = QLabel('0 个参数')
        self.lbl_training_parameter_count.setObjectName('trainingPanelBadge')
        title_row.addWidget(self.lbl_training_parameter_count)
        layout.addLayout(title_row)

        template_bar = QWidget()
        template_bar.setObjectName('trainingTemplateBar')
        template_layout = QHBoxLayout(template_bar)
        template_layout.setContentsMargins(12, 8, 10, 8)
        template_layout.setSpacing(8)
        template_layout.addWidget(QLabel('参数模板'))
        self.training_template_combo = QComboBox()
        self.training_template_combo.setObjectName('trainingTemplateCombo')
        self.training_template_combo.currentIndexChanged.connect(
            self._on_training_template_changed
        )
        template_layout.addWidget(self.training_template_combo, 1)
        self.btn_advanced_config = QPushButton('高级配置')
        self.btn_advanced_config.setObjectName('trainingAdvancedBtn')
        self.btn_advanced_config.clicked.connect(
            self._open_training_advanced_config
        )
        template_layout.addWidget(self.btn_advanced_config)
        layout.addWidget(template_bar)

        dataset_panel = QWidget()
        dataset_panel.setObjectName('trainingDatasetContext')
        dataset_layout = QHBoxLayout(dataset_panel)
        dataset_layout.setContentsMargins(14, 9, 10, 9)
        dataset_text = QVBoxLayout()
        dataset_text.setSpacing(1)
        dataset_caption = QLabel('当前训练数据')
        dataset_caption.setObjectName('trainingConfigCaption')
        self.lbl_config_dataset = QLabel('尚未选择训练批次')
        self.lbl_config_dataset.setObjectName('trainingConfigDatasetName')
        self.lbl_config_dataset_meta = QLabel('-')
        self.lbl_config_dataset_meta.setObjectName('trainingConfigDatasetMeta')
        dataset_text.addWidget(dataset_caption)
        dataset_text.addWidget(self.lbl_config_dataset)
        dataset_text.addWidget(self.lbl_config_dataset_meta)
        dataset_layout.addLayout(dataset_text, 1)
        self.btn_config_view_data = QPushButton('查看数据')
        self.btn_config_view_data.setObjectName('secondaryBtn')
        self.btn_config_view_data.setEnabled(False)
        self.btn_config_view_data.clicked.connect(self._request_review)
        dataset_layout.addWidget(self.btn_config_view_data)
        layout.addWidget(dataset_panel)

        scroll = QScrollArea()
        scroll.setObjectName('trainingConfigScroll')
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        config_body = QWidget()
        config_body.setObjectName('trainingConfigBody')
        config_grid = QGridLayout(config_body)
        config_grid.setContentsMargins(0, 0, 0, 0)
        config_grid.setHorizontalSpacing(10)
        config_grid.setVerticalSpacing(10)

        model_panel = QWidget()
        model_panel.setObjectName('trainingParameterPanel')
        model_layout = QGridLayout(model_panel)
        model_layout.setContentsMargins(14, 11, 14, 12)
        model_layout.setHorizontalSpacing(10)
        model_layout.setVerticalSpacing(8)
        model_title = QLabel('模型与任务')
        model_title.setObjectName('trainingParameterTitle')
        model_layout.addWidget(model_title, 0, 0, 1, 4)
        model_layout.addWidget(QLabel('模型来源'), 1, 0)
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setObjectName('trainingPathEdit')
        self.model_path_edit.setPlaceholderText('选择 .pt 或模型 .yaml')
        model_layout.addWidget(self.model_path_edit, 1, 1, 1, 2)
        self.btn_browse_model = QPushButton('选择模型')
        self.btn_browse_model.setObjectName('secondaryBtn')
        self.btn_browse_model.clicked.connect(self._choose_model)
        model_layout.addWidget(self.btn_browse_model, 1, 3)
        model_layout.addWidget(QLabel('输出根目录'), 2, 0)
        self.output_root_edit = QLineEdit()
        self.output_root_edit.setObjectName('trainingPathEdit')
        self.output_root_edit.setPlaceholderText('训练产物保存根目录')
        model_layout.addWidget(self.output_root_edit, 2, 1, 1, 2)
        self.btn_browse_output = QPushButton('选择目录')
        self.btn_browse_output.setObjectName('secondaryBtn')
        self.btn_browse_output.clicked.connect(self._choose_training_output)
        model_layout.addWidget(self.btn_browse_output, 2, 3)
        model_layout.addWidget(QLabel('项目分组'), 3, 0)
        self.project_name_edit = QLineEdit()
        self.project_name_edit.setObjectName('trainingEdit')
        self.project_name_edit.setPlaceholderText('例如 ShengSong')
        model_layout.addWidget(self.project_name_edit, 3, 1)
        model_layout.addWidget(QLabel('任务名称'), 3, 2)
        self.run_name_edit = QLineEdit()
        self.run_name_edit.setObjectName('trainingEdit')
        self.run_name_edit.setPlaceholderText('训练任务名称')
        model_layout.addWidget(self.run_name_edit, 3, 3)
        self.btn_open_models = QPushButton('打开模型目录')
        self.btn_open_models.setObjectName('trainingTaskActionBtn')
        self.btn_open_models.clicked.connect(self._open_models_directory)
        model_layout.addWidget(self.btn_open_models, 4, 0, 1, 2)
        model_layout.setColumnStretch(1, 1)
        model_layout.setColumnStretch(2, 1)
        config_grid.addWidget(model_panel, 0, 0, 1, 2)

        core_panel = QWidget()
        core_panel.setObjectName('trainingParameterPanel')
        core = QGridLayout(core_panel)
        core.setContentsMargins(14, 11, 14, 12)
        core.setHorizontalSpacing(9)
        core.setVerticalSpacing(8)
        core_title = QLabel('核心参数')
        core_title.setObjectName('trainingParameterTitle')
        core.addWidget(core_title, 0, 0, 1, 4)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setObjectName('trainingParamSpin')
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(100)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setObjectName('trainingParamSpin')
        self.imgsz_spin.setRange(64, 4096)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        self.batch_spin = QDoubleSpinBox()
        self.batch_spin.setObjectName('trainingParamSpin')
        self.batch_spin.setRange(-1, 1024)
        self.batch_spin.setDecimals(2)
        self.batch_spin.setSingleStep(1)
        self.batch_spin.setValue(16)
        self.patience_spin = QSpinBox()
        self.patience_spin.setObjectName('trainingParamSpin')
        self.patience_spin.setRange(0, 10000)
        self.workers_spin = QSpinBox()
        self.workers_spin.setObjectName('trainingParamSpin')
        self.workers_spin.setRange(0, 256)
        self.device_combo = QComboBox()
        self.device_combo.setObjectName('trainingCombo')
        self.device_combo.setEditable(True)
        self.device_combo.addItem('GPU 0', '0')
        self.device_combo.addItem('CPU', 'cpu')
        for row, values in enumerate((
            ('Epochs', self.epochs_spin, 'Image size', self.imgsz_spin),
            ('Batch', self.batch_spin, 'Patience', self.patience_spin),
            ('Workers', self.workers_spin, 'Device', self.device_combo),
        ), start=1):
            core.addWidget(QLabel(values[0]), row, 0)
            core.addWidget(values[1], row, 1)
            core.addWidget(QLabel(values[2]), row, 2)
            core.addWidget(values[3], row, 3)
        core.setColumnStretch(1, 1)
        core.setColumnStretch(3, 1)
        config_grid.addWidget(core_panel, 1, 0)

        optimizer_panel = QWidget()
        optimizer_panel.setObjectName('trainingParameterPanel')
        optimizer = QGridLayout(optimizer_panel)
        optimizer.setContentsMargins(14, 11, 14, 12)
        optimizer.setHorizontalSpacing(9)
        optimizer.setVerticalSpacing(8)
        optimizer_title = QLabel('优化器与学习率')
        optimizer_title.setObjectName('trainingParameterTitle')
        optimizer.addWidget(optimizer_title, 0, 0, 1, 4)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.setObjectName('trainingCombo')
        self.optimizer_combo.setEditable(True)
        self.optimizer_combo.addItems(
            ['auto', 'SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']
        )
        self.lr0_spin = self._double_parameter_spin(0, 10, 0.01, 6)
        self.lrf_spin = self._double_parameter_spin(0, 1, 0.01, 6)
        self.momentum_spin = self._double_parameter_spin(0, 1, 0.937, 4)
        self.weight_decay_spin = self._double_parameter_spin(
            0, 1, 0.0005, 6
        )
        self.warmup_epochs_spin = self._double_parameter_spin(0, 100, 3.0, 2)
        optimization_rows = (
            ('Optimizer', self.optimizer_combo, 'LR0', self.lr0_spin),
            ('LR factor', self.lrf_spin, 'Momentum', self.momentum_spin),
            ('Weight decay', self.weight_decay_spin,
             'Warmup epochs', self.warmup_epochs_spin),
        )
        for row, values in enumerate(optimization_rows, start=1):
            optimizer.addWidget(QLabel(values[0]), row, 0)
            optimizer.addWidget(values[1], row, 1)
            optimizer.addWidget(QLabel(values[2]), row, 2)
            optimizer.addWidget(values[3], row, 3)
        optimizer.setColumnStretch(1, 1)
        optimizer.setColumnStretch(3, 1)
        config_grid.addWidget(optimizer_panel, 1, 1)

        runtime_panel = QWidget()
        runtime_panel.setObjectName('trainingParameterPanel')
        runtime = QGridLayout(runtime_panel)
        runtime.setContentsMargins(14, 11, 14, 12)
        runtime.setHorizontalSpacing(10)
        runtime.setVerticalSpacing(8)
        runtime_title = QLabel('运行策略')
        runtime_title.setObjectName('trainingParameterTitle')
        runtime.addWidget(runtime_title, 0, 0, 1, 4)
        self.cache_combo = QComboBox()
        self.cache_combo.setObjectName('trainingCombo')
        self.cache_combo.addItem('关闭', False)
        self.cache_combo.addItem('RAM', 'ram')
        self.cache_combo.addItem('磁盘', 'disk')
        self.seed_train_spin = QSpinBox()
        self.seed_train_spin.setObjectName('trainingParamSpin')
        self.seed_train_spin.setRange(0, 999999)
        self.close_mosaic_spin = QSpinBox()
        self.close_mosaic_spin.setObjectName('trainingParamSpin')
        self.close_mosaic_spin.setRange(0, 10000)
        self.save_period_spin = QSpinBox()
        self.save_period_spin.setObjectName('trainingParamSpin')
        self.save_period_spin.setRange(-1, 10000)
        runtime_controls = (
            ('Cache', self.cache_combo), ('Seed', self.seed_train_spin),
            ('Close mosaic', self.close_mosaic_spin),
            ('Save period', self.save_period_spin),
        )
        for index, (label, widget) in enumerate(runtime_controls):
            row = 1 + index // 2
            column = (index % 2) * 2
            runtime.addWidget(QLabel(label), row, column)
            runtime.addWidget(widget, row, column + 1)

        self.amp_check = self._training_toggle('AMP')
        self.pretrained_check = self._training_toggle('Pretrained')
        self.deterministic_check = self._training_toggle('Deterministic')
        self.cos_lr_check = self._training_toggle('Cosine LR')
        self.rect_check = self._training_toggle('Rect batches')
        self.plots_check = self._training_toggle('Plots')
        for index, checkbox in enumerate((
            self.amp_check, self.pretrained_check, self.deterministic_check,
            self.cos_lr_check, self.rect_check, self.plots_check,
        )):
            runtime.addWidget(checkbox, 3 + index // 4, index % 4)
        config_grid.addWidget(runtime_panel, 2, 0, 1, 2)
        config_grid.setColumnStretch(0, 1)
        config_grid.setColumnStretch(1, 1)
        config_grid.setRowStretch(3, 1)
        scroll.setWidget(config_body)
        layout.addWidget(scroll, 1)

        action_row = QHBoxLayout()
        self.lbl_training_config_state = QLabel('配置等待训练执行器')
        self.lbl_training_config_state.setObjectName('trainingSourceSummary')
        action_row.addWidget(self.lbl_training_config_state)
        action_row.addStretch()
        self.btn_save_training_draft = QPushButton('保存草稿')
        self.btn_save_training_draft.setObjectName('secondaryBtn')
        self.btn_save_training_draft.setEnabled(False)
        self.btn_save_training_draft.clicked.connect(
            self._save_training_draft
        )
        action_row.addWidget(self.btn_save_training_draft)
        self.btn_start_training = QPushButton('开始训练')
        self.btn_start_training.setObjectName('successBtn')
        self.btn_start_training.setEnabled(False)
        self.btn_start_training.clicked.connect(self._start_training)
        action_row.addWidget(self.btn_start_training)
        layout.addLayout(action_row)
        return page

    def _build_monitor_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName('trainingPage')
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 14, 16, 16)
        layout.setSpacing(10)

        title_row = QHBoxLayout()
        title = QLabel('任务监控')
        title.setObjectName('trainingSectionTitle')
        title_row.addWidget(title)
        self.lbl_monitor_status = QLabel('等待任务')
        self.lbl_monitor_status.setObjectName('trainingRunStatus')
        self.lbl_monitor_status.setProperty('tone', 'idle')
        self.lbl_monitor_status.setAlignment(Qt.AlignCenter)
        title_row.addWidget(self.lbl_monitor_status)
        title_row.addStretch()
        self.btn_stop_training = QPushButton('停止训练')
        self.btn_stop_training.setObjectName('dangerBtn')
        self.btn_stop_training.setEnabled(False)
        self.btn_stop_training.clicked.connect(self._stop_training)
        title_row.addWidget(self.btn_stop_training)
        layout.addLayout(title_row)

        context = QWidget()
        context.setObjectName('trainingRunContext')
        context_layout = QGridLayout(context)
        context_layout.setContentsMargins(14, 10, 14, 10)
        context_layout.setHorizontalSpacing(12)
        context_layout.setVerticalSpacing(4)
        context_layout.addWidget(QLabel('训练任务'), 0, 0)
        self.lbl_monitor_run = QLabel('尚未启动')
        self.lbl_monitor_run.setObjectName('trainingRunPrimary')
        context_layout.addWidget(self.lbl_monitor_run, 0, 1)
        context_layout.addWidget(QLabel('输出目录'), 1, 0)
        self.lbl_monitor_output = QLabel('-')
        self.lbl_monitor_output.setObjectName('trainingRunPath')
        self.lbl_monitor_output.setTextInteractionFlags(Qt.TextSelectableByMouse)
        context_layout.addWidget(self.lbl_monitor_output, 1, 1)
        context_layout.setColumnStretch(1, 1)
        layout.addWidget(context)

        monitor_grid = QGridLayout()
        monitor_grid.setHorizontalSpacing(8)
        monitor_grid.setVerticalSpacing(8)
        self.lbl_monitor_epoch = self._add_monitor_card(
            monitor_grid, 0, 0, 'EPOCH', '0 / 0'
        )
        self.lbl_monitor_elapsed = self._add_monitor_card(
            monitor_grid, 0, 1, '运行时间', '00:00:00'
        )
        self.lbl_monitor_cpu = self._add_monitor_card(
            monitor_grid, 0, 2, '任务 CPU', '-'
        )
        self.lbl_monitor_memory = self._add_monitor_card(
            monitor_grid, 0, 3, '任务内存', '-'
        )
        self.lbl_monitor_gpu = self._add_monitor_card(
            monitor_grid, 0, 4, 'GPU', '-'
        )
        self.lbl_monitor_vram = self._add_monitor_card(
            monitor_grid, 0, 5, '显存', '-'
        )
        layout.addLayout(monitor_grid)

        self.training_run_progress = QProgressBar()
        self.training_run_progress.setObjectName('trainingRunProgress')
        self.training_run_progress.setRange(0, 1000)
        self.training_run_progress.setValue(0)
        self.training_run_progress.setFormat('等待训练任务')
        layout.addWidget(self.training_run_progress)

        monitor_splitter = QSplitter(Qt.Horizontal)
        monitor_splitter.setObjectName('trainingMonitorSplitter')
        monitor_splitter.setChildrenCollapsible(False)
        metric_panel = QWidget()
        metric_panel.setObjectName('trainingMonitorPanel')
        metric_layout = QVBoxLayout(metric_panel)
        metric_layout.setContentsMargins(10, 10, 10, 10)
        metric_title_row = QHBoxLayout()
        metric_title = QLabel('实时指标')
        metric_title.setObjectName('trainingPanelTitle')
        metric_title_row.addWidget(metric_title)
        metric_title_row.addStretch()
        self.training_metric_group_combo = QComboBox()
        self.training_metric_group_combo.setObjectName(
            'trainingMetricGroupCombo'
        )
        self.training_metric_group_combo.setToolTip('切换曲线指标类型')
        for group, label in METRIC_GROUPS:
            self.training_metric_group_combo.addItem(label, group)
        self.training_metric_group_combo.currentIndexChanged.connect(
            self._change_training_metric_group
        )
        metric_title_row.addWidget(self.training_metric_group_combo)
        metric_layout.addLayout(metric_title_row)

        self.training_metric_tabs = QTabWidget()
        self.training_metric_tabs.setObjectName('trainingMetricTabs')
        self.training_curve_chart = RealtimeTrainingChart()
        self.training_metric_tabs.addTab(
            self.training_curve_chart, '趋势图'
        )
        self.training_metric_tree = QTreeWidget()
        self.training_metric_tree.setObjectName('trainingMetricTree')
        self.training_metric_tree.setHeaderLabels(['指标', '当前值'])
        self.training_metric_tree.setRootIsDecorated(False)
        self.training_metric_tree.setUniformRowHeights(True)
        metric_header = self.training_metric_tree.header()
        metric_header.setStretchLastSection(False)
        metric_header.setSectionResizeMode(0, metric_header.Stretch)
        metric_header.setSectionResizeMode(1, metric_header.ResizeToContents)
        self.training_metric_tabs.addTab(
            self.training_metric_tree, '精确值'
        )
        metric_layout.addWidget(self.training_metric_tabs, 1)
        monitor_splitter.addWidget(metric_panel)

        log_panel = QWidget()
        log_panel.setObjectName('trainingMonitorPanel')
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(10, 10, 10, 10)
        log_title_row = QHBoxLayout()
        log_title = QLabel('训练日志')
        log_title.setObjectName('trainingPanelTitle')
        log_title_row.addWidget(log_title)
        log_title_row.addStretch()
        self.btn_clear_training_log = QToolButton()
        self.btn_clear_training_log.setObjectName('trainingBatchSelectionBtn')
        self.btn_clear_training_log.setText('清空日志')
        self.btn_clear_training_log.clicked.connect(
            lambda: self.training_log.clear()
        )
        log_title_row.addWidget(self.btn_clear_training_log)
        log_layout.addLayout(log_title_row)
        self.training_log = QPlainTextEdit()
        self.training_log.setObjectName('trainingLog')
        self.training_log.setReadOnly(True)
        self.training_log.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.training_log.document().setMaximumBlockCount(5000)
        log_layout.addWidget(self.training_log, 1)
        monitor_splitter.addWidget(log_panel)
        monitor_splitter.setStretchFactor(0, 6)
        monitor_splitter.setStretchFactor(1, 5)
        monitor_splitter.setSizes([620, 520])
        layout.addWidget(monitor_splitter, 1)

        action_row = QHBoxLayout()
        self.btn_monitor_back = QPushButton('返回任务中心')
        self.btn_monitor_back.setObjectName('secondaryBtn')
        self.btn_monitor_back.clicked.connect(
            lambda: self._show_step(TASK_CENTER_PAGE)
        )
        action_row.addWidget(self.btn_monitor_back)
        action_row.addStretch()
        self.btn_view_training_result = QPushButton('在模型管理中查看')
        self.btn_view_training_result.setObjectName('primaryBtn')
        self.btn_view_training_result.setEnabled(False)
        self.btn_view_training_result.clicked.connect(
            self._request_trained_model
        )
        action_row.addWidget(self.btn_view_training_result)
        layout.addLayout(action_row)

        self._training_runtime_timer = QTimer(self)
        self._training_runtime_timer.setInterval(1500)
        self._training_runtime_timer.timeout.connect(
            self._update_training_runtime
        )
        return page

    def _build_task_center_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName('trainingPage')
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 14, 16, 16)
        layout.setSpacing(10)

        title_row = QHBoxLayout()
        title_block = QVBoxLayout()
        title_block.setSpacing(3)
        title = QLabel('训练任务')
        title.setObjectName('trainingSectionTitle')
        title.setMinimumHeight(27)
        subtitle = QLabel('统一管理草稿、排队、运行与历史训练记录')
        subtitle.setObjectName('trainingTaskSubtitle')
        subtitle.setMinimumHeight(16)
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        title_row.addLayout(title_block)
        title_row.addStretch()
        self.btn_task_refresh = QToolButton()
        self.btn_task_refresh.setObjectName('trainingIconBtn')
        self.btn_task_refresh.setText('↻')
        self.btn_task_refresh.setToolTip('刷新任务列表')
        self.btn_task_refresh.clicked.connect(self._recover_training_tasks)
        title_row.addWidget(self.btn_task_refresh)
        self.btn_task_new = QPushButton('新建训练任务')
        self.btn_task_new.setObjectName('primaryBtn')
        self.btn_task_new.clicked.connect(self._new_training_task)
        title_row.addWidget(self.btn_task_new)
        layout.addLayout(title_row)

        stats = QHBoxLayout()
        stats.setSpacing(8)
        self.task_stat_values = {}
        for key, caption in (
            ('total', '全部任务'),
            ('active', '正在运行'),
            ('queued', '等待队列'),
            ('completed', '已完成'),
            ('failed', '需要关注'),
        ):
            card = QWidget()
            card.setObjectName('trainingTaskStatCard')
            card_layout = QHBoxLayout(card)
            card_layout.setContentsMargins(12, 8, 12, 8)
            caption_label = QLabel(caption)
            caption_label.setObjectName('trainingTaskStatCaption')
            value_label = QLabel('0')
            value_label.setObjectName('trainingTaskStatValue')
            value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            card_layout.addWidget(caption_label)
            card_layout.addStretch()
            card_layout.addWidget(value_label)
            self.task_stat_values[key] = value_label
            stats.addWidget(card, 1)
        layout.addLayout(stats)

        toolbar = QWidget()
        toolbar.setObjectName('trainingTaskToolbar')
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(10, 7, 10, 7)
        toolbar_layout.setSpacing(8)
        self.task_search_edit = QLineEdit()
        self.task_search_edit.setObjectName('trainingTaskSearch')
        self.task_search_edit.setPlaceholderText('搜索任务、项目、模型或数据集')
        self.task_search_edit.setClearButtonEnabled(True)
        self.task_search_edit.textChanged.connect(self.refresh_training_tasks)
        toolbar_layout.addWidget(self.task_search_edit, 1)
        self.task_filter_combo = QComboBox()
        self.task_filter_combo.setObjectName('trainingTaskFilter')
        self.task_filter_combo.addItem('全部状态', '')
        for status in (
            'draft', 'queued', 'running', 'completed',
            'failed', 'cancelled', 'interrupted', 'archived',
        ):
            self.task_filter_combo.addItem(TASK_STATUS_META[status][0], status)
        self.task_filter_combo.currentIndexChanged.connect(
            self.refresh_training_tasks
        )
        toolbar_layout.addWidget(self.task_filter_combo)
        layout.addWidget(toolbar)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setObjectName('trainingTaskSplitter')
        splitter.setChildrenCollapsible(False)

        list_panel = QWidget()
        list_panel.setObjectName('trainingTaskListPanel')
        list_layout = QVBoxLayout(list_panel)
        list_layout.setContentsMargins(8, 8, 8, 8)
        self.task_tree = QTreeWidget()
        self.task_tree.setObjectName('trainingTaskTree')
        self.task_tree.setHeaderLabels([
            '任务', '状态', '类型', '训练数据', '进度', '创建时间',
        ])
        self.task_tree.setRootIsDecorated(False)
        self.task_tree.setAlternatingRowColors(True)
        self.task_tree.setUniformRowHeights(True)
        self.task_tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.task_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        task_header = self.task_tree.header()
        task_header.setStretchLastSection(False)
        task_header.setSectionResizeMode(0, task_header.Stretch)
        task_header.setSectionResizeMode(1, task_header.ResizeToContents)
        task_header.setSectionResizeMode(2, task_header.ResizeToContents)
        task_header.setSectionResizeMode(3, task_header.Stretch)
        task_header.setSectionResizeMode(4, task_header.ResizeToContents)
        task_header.setSectionResizeMode(5, task_header.ResizeToContents)
        self.task_tree.currentItemChanged.connect(self._on_task_selected)
        self.task_tree.customContextMenuRequested.connect(
            self._show_task_context_menu
        )
        self.task_tree.itemDoubleClicked.connect(
            lambda _item, _column: self._open_selected_task()
        )
        list_layout.addWidget(self.task_tree, 1)
        self.lbl_task_list_summary = QLabel('尚无训练任务')
        self.lbl_task_list_summary.setObjectName('trainingTaskListSummary')
        list_layout.addWidget(self.lbl_task_list_summary)
        splitter.addWidget(list_panel)

        detail = QWidget()
        detail.setObjectName('trainingTaskDetail')
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(14, 12, 14, 12)
        detail_layout.setSpacing(9)
        detail_top = QHBoxLayout()
        detail_caption = QLabel('TASK PROFILE')
        detail_caption.setObjectName('trainingTaskDetailCaption')
        detail_top.addWidget(detail_caption)
        detail_top.addStretch()
        self.lbl_task_detail_status = QLabel('未选择')
        self.lbl_task_detail_status.setObjectName('trainingTaskStatusBadge')
        self.lbl_task_detail_status.setProperty('tone', 'neutral')
        detail_top.addWidget(self.lbl_task_detail_status)
        detail_layout.addLayout(detail_top)
        self.lbl_task_detail_name = QLabel('选择一个训练任务')
        self.lbl_task_detail_name.setObjectName('trainingTaskDetailName')
        self.lbl_task_detail_name.setWordWrap(True)
        detail_layout.addWidget(self.lbl_task_detail_name)
        self.lbl_task_detail_meta = QLabel('-')
        self.lbl_task_detail_meta.setObjectName('trainingTaskDetailMeta')
        self.lbl_task_detail_meta.setWordWrap(True)
        detail_layout.addWidget(self.lbl_task_detail_meta)
        self.task_detail_progress = QProgressBar()
        self.task_detail_progress.setObjectName('trainingTaskProgress')
        self.task_detail_progress.setRange(0, 1000)
        self.task_detail_progress.setValue(0)
        self.task_detail_progress.setFormat('等待选择任务')
        detail_layout.addWidget(self.task_detail_progress)
        self.lbl_task_detail_path = QLabel('-')
        self.lbl_task_detail_path.setObjectName('trainingTaskDetailPath')
        self.lbl_task_detail_path.setWordWrap(True)
        self.lbl_task_detail_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        detail_layout.addWidget(self.lbl_task_detail_path)
        self.lbl_task_detail_error = QLabel('')
        self.lbl_task_detail_error.setObjectName('trainingTaskDetailError')
        self.lbl_task_detail_error.setWordWrap(True)
        self.lbl_task_detail_error.hide()
        detail_layout.addWidget(self.lbl_task_detail_error)

        notes_caption = QLabel('备注')
        notes_caption.setObjectName('trainingTaskDetailCaption')
        detail_layout.addWidget(notes_caption)
        self.task_notes_edit = QPlainTextEdit()
        self.task_notes_edit.setObjectName('trainingTaskNotes')
        self.task_notes_edit.setPlaceholderText('记录实验目的、参数调整或结果结论')
        self.task_notes_edit.setMaximumHeight(86)
        detail_layout.addWidget(self.task_notes_edit)
        self.btn_task_save_notes = QPushButton('保存备注')
        self.btn_task_save_notes.setObjectName('secondaryBtn')
        self.btn_task_save_notes.clicked.connect(self._save_task_notes)
        detail_layout.addWidget(self.btn_task_save_notes)
        detail_layout.addStretch()

        primary_actions = QHBoxLayout()
        self.btn_task_open = QPushButton('查看监控')
        self.btn_task_open.setObjectName('primaryBtn')
        self.btn_task_open.clicked.connect(self._open_selected_task)
        self.btn_task_start = QPushButton('启动任务')
        self.btn_task_start.setObjectName('successBtn')
        self.btn_task_start.clicked.connect(self._start_selected_task)
        primary_actions.addWidget(self.btn_task_open, 1)
        primary_actions.addWidget(self.btn_task_start, 1)
        detail_layout.addLayout(primary_actions)

        secondary_actions = QGridLayout()
        self.btn_task_edit = QPushButton('编辑草稿')
        self.btn_task_clone = QPushButton('复制配置')
        self.btn_task_retry = QPushButton('重新训练')
        self.btn_task_rename = QPushButton('重命名')
        self.btn_task_archive = QPushButton('归档')
        self.btn_task_delete = QPushButton('删除')
        self.btn_task_open_files = QPushButton('打开任务文件')
        for button in (
            self.btn_task_edit, self.btn_task_clone, self.btn_task_retry,
            self.btn_task_rename, self.btn_task_archive, self.btn_task_delete,
            self.btn_task_open_files,
        ):
            button.setObjectName('trainingTaskActionBtn')
        self.btn_task_delete.setProperty('danger', True)
        self.btn_task_edit.clicked.connect(self._edit_selected_task)
        self.btn_task_clone.clicked.connect(self._clone_selected_task)
        self.btn_task_retry.clicked.connect(self._retry_selected_task)
        self.btn_task_rename.clicked.connect(self._rename_selected_task)
        self.btn_task_archive.clicked.connect(self._archive_selected_task)
        self.btn_task_delete.clicked.connect(self._delete_selected_task)
        self.btn_task_open_files.clicked.connect(self._open_selected_task_files)
        for index, button in enumerate((
            self.btn_task_edit, self.btn_task_clone, self.btn_task_retry,
            self.btn_task_rename, self.btn_task_archive, self.btn_task_delete,
            self.btn_task_open_files,
        )):
            secondary_actions.addWidget(button, index // 3, index % 3)
        detail_layout.addLayout(secondary_actions)
        splitter.addWidget(detail)
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([820, 360])
        layout.addWidget(splitter, 1)
        self._set_task_detail(None)
        return page

    def set_dataset_root(self, path: str | Path, refresh: bool = True):
        root = self._project_root_for(path)
        if root is None:
            return False
        self._dataset_root = root
        self.dataset_root_edit.setText(str(root))
        self.lbl_training_context.setText(root.name)
        current_project = self.project_name_edit.text().strip()
        if not current_project or current_project == self._auto_project_name:
            self._auto_project_name = self._default_project_name(root)
            self.project_name_edit.setText(self._auto_project_name)
        QSettings('FilesProcessQT', 'ImageManager').setValue(
            'lastTrainingDatasetRoot', str(root)
        )
        if refresh:
            self.refresh_sources()
            self.refresh_existing_batches()
        return True

    def refresh_sources(self):
        self.source_tree.blockSignals(True)
        self.source_tree.clear()
        if self._dataset_root is None:
            self.source_tree.blockSignals(False)
            self.lbl_source_summary.setText('请选择数据项目')
            return
        rows = list_source_batches(
            self._dataset_root, self._annotation_dir, self._label_dir
        )
        for row in rows:
            image_count = row['image_count']
            annotation_count = row['annotation_count']
            label_count = row['label_count']
            ready = (
                image_count > 0
                and annotation_count == image_count
                and label_count == image_count
            )
            if ready:
                status_text, status_tone = '完整', 'success'
            elif annotation_count == 0 or label_count == 0:
                status_text, status_tone = '缺失', 'danger'
            else:
                status_text, status_tone = '待补齐', 'warning'
            item = QTreeWidgetItem([
                '', row['name'], str(image_count),
                str(annotation_count), str(label_count), status_text,
            ])
            item.setData(0, Qt.UserRole, row['name'])
            item.setCheckState(0, Qt.Unchecked)
            if annotation_count != image_count:
                color = '#FF6B72' if annotation_count == 0 else '#F5B942'
                item.setForeground(3, QBrush(QColor(color)))
            if label_count != image_count:
                color = '#FF6B72' if label_count == 0 else '#F5B942'
                item.setForeground(4, QBrush(QColor(color)))
            self.source_tree.addTopLevelItem(item)
            self._set_tree_status(
                self.source_tree, item, 5, status_text, status_tone
            )
        self.source_tree.blockSignals(False)
        self.lbl_source_summary.setText(f'发现 {len(rows)} 个原始批次')
        self._scan_result = None
        self.btn_prepare.setEnabled(False)
        self._suggest_target_name()
        if self._batch_source_names:
            self._sync_sources_for_batch(self._batch_source_names)

    def refresh_existing_batches(self):
        current = self._selected_existing_batch_path()
        self.existing_batch_tree.blockSignals(True)
        self.existing_batch_tree.clear()
        selected_item = None
        batch_count = 0
        if self._dataset_root is not None:
            training_root = self._dataset_root / 'training_data'
            if training_root.is_dir():
                for batch in sorted(training_root.iterdir(), reverse=True):
                    if batch.is_dir() and not batch.name.startswith('.'):
                        summary = inspect_training_batch(batch)
                        if summary.is_ready:
                            status_text, status_tone = '就绪', 'success'
                        elif summary.image_count <= 0 or summary.missing_top_labels:
                            status_text, status_tone = '阻断', 'danger'
                        else:
                            status_text, status_tone = '待定', 'warning'
                        item = QTreeWidgetItem([
                            batch.name,
                            TASK_SHORT_LABELS.get(
                                summary.task_type, summary.task_type
                            ),
                            str(summary.image_count),
                            f'{summary.train_image_count} / {summary.val_image_count}',
                            status_text,
                        ])
                        item.setData(0, Qt.UserRole, str(batch))
                        item.setToolTip(0, str(batch))
                        item.setToolTip(
                            1, TASK_LABELS.get(
                                summary.task_type, summary.task_type
                            )
                        )
                        self.existing_batch_tree.addTopLevelItem(item)
                        self._set_tree_status(
                            self.existing_batch_tree, item, 4,
                            status_text, status_tone,
                        )
                        batch_count += 1
                        if current and Path(current) == batch:
                            selected_item = item
        self.existing_batch_tree.blockSignals(False)
        self.lbl_existing_count.setText(f'{batch_count} 个批次')
        if selected_item is None and batch_count:
            selected_item = self.existing_batch_tree.topLevelItem(0)
        if selected_item is not None:
            self.existing_batch_tree.setCurrentItem(selected_item)
        self._inspect_selected_existing_batch()

    def _set_task(self, task_type: str):
        task_type = str(task_type or 'pose')
        if task_type not in TASK_LABELS:
            return
        self._task_type = task_type
        self._annotation_dir = str(
            TASK_PRESETS.get(task_type, {}).get(
                'annotation_dir', f'annotations-{task_type}'
            )
        )
        index = self.task_combo.findData(task_type)
        if index >= 0 and self.task_combo.currentIndex() != index:
            self.task_combo.blockSignals(True)
            self.task_combo.setCurrentIndex(index)
            self.task_combo.blockSignals(False)
        if hasattr(self, 'source_tree'):
            self.refresh_sources()
        if hasattr(self, 'training_template_combo'):
            self._reload_training_templates(task_type)

    def _show_step(self, index: int):
        index = max(0, min(index, self.content_stack.count() - 1))
        self.content_stack.setCurrentIndex(index)
        if index == TASK_CENTER_PAGE:
            self.btn_task_center.setChecked(True)
            self.refresh_training_tasks()
        elif index < len(self.step_buttons):
            self.step_buttons[index].setChecked(True)

    def _recover_training_tasks(self, *_args):
        output_root = self.output_root_edit.text().strip()
        persistent_registry = self._task_registry.path != ':memory:'
        result = self._task_registry.recover(
            request_directory=(
                self._runtime_root / 'training_jobs'
                if persistent_registry else self._training_root / 'legacy_requests'
            ),
            extra_request_directories=(self._task_files_root,),
            output_roots=tuple(filter(None, (
                output_root,
                self._training_runs_root,
                self._project_root / 'training_runs'
                if persistent_registry else '',
            ))),
            mark_active_interrupted=not self._task_recovery_complete,
        )
        self._task_recovery_complete = True
        migrated = self._migrate_legacy_task_bundles()
        enriched = self._enrich_failed_task_errors()
        self.refresh_training_tasks()
        recovered = result['imported'] + result['discovered']
        if recovered or migrated or enriched:
            self.status_message.emit(
                f'已恢复 {recovered} 个历史训练任务，迁移 {migrated} 个任务包'
            )

    def _enrich_failed_task_errors(self) -> int:
        updated = 0
        for record in self._task_registry.list_tasks(status='failed'):
            try:
                summary = inspect_training_batch(record.batch_root)
            except (OSError, ValueError):
                continue
            if not summary.invalid_label_count:
                continue
            message = summary.readiness_message()
            if message == record.error_message:
                continue
            self._task_registry.set_status(
                record.task_id, 'failed', error_message=message
            )
            updated += 1
        return updated

    def _migrate_legacy_task_bundles(self) -> int:
        migrated = 0
        task_root = self._task_files_root.resolve()
        records = self._task_registry.list_tasks(include_archived=True)
        for record in records:
            if record.status in ACTIVE_STATUSES:
                continue
            request = Path(record.request_path) if record.request_path else None
            if request is not None:
                try:
                    request.resolve().relative_to(task_root)
                    continue
                except (OSError, ValueError):
                    pass
            if not Path(record.dataset_yaml).is_file():
                continue
            previous_log = Path(record.log_path) if record.log_path else None
            try:
                job, request_path, log_path = self._write_task_bundle(record.job)
                if previous_log is not None and previous_log.is_file():
                    shutil.copy2(previous_log, log_path)
                self._task_registry.relocate_artifacts(
                    record.task_id, job, request_path=request_path,
                    log_path=log_path,
                )
            except (OSError, ValueError, TrainingTaskRegistryError,
                    TrainingJobError):
                continue
            migrated += 1
        return migrated

    def refresh_training_tasks(self, *_args, select_task_id: str = ''):
        if not hasattr(self, 'task_tree'):
            return
        selected = select_task_id or (
            self._selected_task_record().task_id
            if self._selected_task_record() is not None else ''
        )
        status = str(self.task_filter_combo.currentData() or '')
        include_archived = status == 'archived'
        records = self._task_registry.list_tasks(
            status=status,
            search=self.task_search_edit.text(),
            include_archived=include_archived,
        )
        self.task_tree.blockSignals(True)
        self.task_tree.clear()
        self._task_items.clear()
        selected_item = None
        for record in records:
            status_label, tone = TASK_STATUS_META.get(
                record.status, (record.status, 'neutral')
            )
            if record.archived:
                status_label, tone = TASK_STATUS_META['archived']
            progress = self._task_progress_text(record)
            item = QTreeWidgetItem([
                record.display_name,
                status_label,
                TASK_SHORT_LABELS.get(record.task_type, record.task_type),
                Path(record.batch_root).name or '-',
                progress,
                self._format_task_time(record.created_at),
            ])
            item.setData(0, Qt.UserRole, record.task_id)
            item.setToolTip(0, record.run_dir)
            item.setForeground(
                1, QBrush(QColor(TASK_STATUS_COLORS.get(tone, '#91A8B8')))
            )
            if record.status in {'failed', 'interrupted'}:
                item.setForeground(0, QBrush(QColor('#FF8090')))
            self.task_tree.addTopLevelItem(item)
            self._task_items[record.task_id] = item
            if record.task_id == selected:
                selected_item = item
        self.task_tree.blockSignals(False)
        self.lbl_task_list_summary.setText(
            f'当前显示 {len(records)} 个任务'
            + (f'  ·  搜索「{self.task_search_edit.text()}」'
               if self.task_search_edit.text() else '')
        )
        counts = self._task_registry.counts()
        for key, label in self.task_stat_values.items():
            label.setText(str(counts.get(key, 0)))
        if selected_item is not None:
            self.task_tree.setCurrentItem(selected_item)
            self._set_task_detail(self._task_registry.get(selected))
        elif records:
            self.task_tree.setCurrentItem(self.task_tree.topLevelItem(0))
            self._set_task_detail(records[0])
        else:
            self._set_task_detail(None)

    def _selected_task_record(self) -> TrainingTaskRecord | None:
        if not hasattr(self, 'task_tree'):
            return None
        item = self.task_tree.currentItem()
        if item is None:
            return None
        task_id = str(item.data(0, Qt.UserRole) or '')
        return self._task_registry.get(task_id) if task_id else None

    def _on_task_selected(self, current, _previous):
        if current is None:
            self._set_task_detail(None)
            return
        task_id = str(current.data(0, Qt.UserRole) or '')
        self._set_task_detail(self._task_registry.get(task_id))

    def _show_task_context_menu(self, position):
        item = self.task_tree.itemAt(position)
        if item is None:
            return
        self.task_tree.setCurrentItem(item)
        record = self._selected_task_record()
        if record is None:
            return

        menu = QMenu(self.task_tree)
        open_model = menu.addAction('在模型管理中查看')
        open_model.setEnabled(Path(record.run_dir).is_dir())
        selected = menu.exec_(
            self.task_tree.viewport().mapToGlobal(position)
        )
        if selected is open_model:
            self._request_task_model(record)

    def _set_task_detail(self, record: TrainingTaskRecord | None):
        self._displayed_task_id = record.task_id if record is not None else ''
        controls = (
            self.btn_task_open, self.btn_task_start, self.btn_task_edit,
            self.btn_task_clone, self.btn_task_retry, self.btn_task_rename,
            self.btn_task_archive, self.btn_task_delete,
            self.btn_task_open_files, self.btn_task_save_notes,
        )
        if record is None:
            self.lbl_task_detail_name.setText('选择一个训练任务')
            self.lbl_task_detail_meta.setText('-')
            self.lbl_task_detail_path.setText('-')
            self.lbl_task_detail_status.setText('未选择')
            self._set_task_status_badge('neutral')
            self.task_detail_progress.setValue(0)
            self.task_detail_progress.setFormat('等待选择任务')
            self.lbl_task_detail_error.hide()
            self.task_notes_edit.clear()
            self.task_notes_edit.setEnabled(False)
            for control in controls:
                control.setEnabled(False)
            return

        status_label, tone = TASK_STATUS_META.get(
            'archived' if record.archived else record.status,
            (record.status, 'neutral'),
        )
        self.lbl_task_detail_status.setText(status_label)
        self._set_task_status_badge(tone)
        self.lbl_task_detail_name.setText(record.display_name)
        self.lbl_task_detail_meta.setText(
            f'{TASK_LABELS.get(record.task_type, record.task_type)}  ·  '
            f'{record.project_name}\n'
            f'模型: {Path(record.model).name or record.model}\n'
            f'数据: {Path(record.batch_root).name or record.batch_root}'
        )
        self.lbl_task_detail_path.setText(f'输出: {record.run_dir}')
        value = round(max(0.0, min(record.progress, 100.0)) * 10)
        self.task_detail_progress.setValue(value)
        self.task_detail_progress.setFormat(self._task_progress_text(record))
        self.lbl_task_detail_error.setText(record.error_message)
        self.lbl_task_detail_error.setVisible(bool(record.error_message))
        self.task_notes_edit.setEnabled(True)
        self.task_notes_edit.setPlainText(record.notes)

        active = record.status in ACTIVE_STATUSES
        run_exists = Path(record.run_dir).is_dir()
        log_exists = bool(record.log_path and Path(record.log_path).is_file())
        self.btn_task_open.setEnabled(
            active or run_exists or log_exists or record.task_id == self._active_task_id
        )
        resumable = self._resume_checkpoint(record) is not None
        if record.status in {'failed', 'cancelled', 'interrupted'}:
            self.btn_task_start.setEnabled(resumable)
            self.btn_task_start.setText(
                '继续训练' if resumable else '无可用断点'
            )
            self.btn_task_start.setToolTip(
                '从 weights/last.pt 恢复优化器和训练轮次'
                if resumable else '未找到 weights/last.pt，请使用“重新训练”'
            )
        else:
            self.btn_task_start.setEnabled(
                record.status in {'draft', 'queued'}
            )
            self.btn_task_start.setText(
                '已在队列' if record.status == 'queued' and self.is_training()
                else '启动任务'
            )
            self.btn_task_start.setToolTip('')
        self.btn_task_edit.setEnabled(record.status == 'draft')
        self.btn_task_clone.setEnabled(not active)
        self.btn_task_retry.setEnabled(
            record.status in {'failed', 'cancelled', 'interrupted'}
        )
        self.btn_task_retry.setText('重新训练')
        self.btn_task_retry.setToolTip(
            '复制原任务配置，在新目录中从第 0 轮开始训练'
        )
        self.btn_task_rename.setEnabled(not active)
        self.btn_task_archive.setEnabled(not active)
        self.btn_task_archive.setText('取消归档' if record.archived else '归档')
        self.btn_task_delete.setEnabled(not active)
        self.btn_task_open_files.setEnabled(bool(
            record.request_path and Path(record.request_path).parent.is_dir()
        ))
        self.btn_task_save_notes.setEnabled(True)

    def _set_task_status_badge(self, tone: str):
        self.lbl_task_detail_status.setProperty('tone', tone)
        self.lbl_task_detail_status.style().unpolish(
            self.lbl_task_detail_status
        )
        self.lbl_task_detail_status.style().polish(
            self.lbl_task_detail_status
        )

    def _new_training_task(self):
        self._editing_task_id = ''
        self.btn_save_training_draft.setText('保存草稿')
        self.lbl_training_config_state.setText('新任务尚未保存')
        if self._current_batch is not None:
            base = f'{self._current_batch.name}-{self._task_type}'
            self.run_name_edit.setText(self._next_run_name(
                base, self.output_root_edit.text(), self.project_name_edit.text()
            ))
        self._show_step(0)

    def _edit_selected_task(self):
        record = self._selected_task_record()
        if record is None or record.status != 'draft':
            return
        self._editing_task_id = record.task_id
        self._load_task_into_config(record, clone=False)
        self.btn_save_training_draft.setText('保存修改')

    def _clone_selected_task(self):
        record = self._selected_task_record()
        if record is None:
            return
        self._editing_task_id = ''
        self._load_task_into_config(record, clone=True)
        self.btn_save_training_draft.setText('保存草稿')

    def _load_task_into_config(self, record: TrainingTaskRecord,
                               clone: bool):
        self._set_task(record.task_type)
        batch = Path(record.batch_root)
        if batch.is_dir():
            dataset_root = (
                batch.parent.parent
                if batch.parent.name == 'training_data' else batch
            )
            self.set_dataset_root(dataset_root, refresh=False)
            self._update_review_summary(inspect_training_batch(batch))
        base = default_training_config(record.task_type)
        config_data = training_config_to_dict(base)
        config_data['name'] = '任务配置快照'
        config_data['model'] = record.model
        config_data['parameters'] = dict(record.parameters)
        self._apply_training_config(training_config_from_dict(config_data))
        self.output_root_edit.setText(record.output_root)
        self.project_name_edit.setText(record.project_name)
        run_name = record.run_name
        if clone:
            run_name = self._next_run_name(
                run_name, record.output_root, record.project_name
            )
        self.run_name_edit.setText(run_name)
        self.lbl_training_config_state.setText(
            '已复制历史任务配置'
            if clone else f'正在编辑草稿: {record.display_name}'
        )
        self._show_step(2)

    def _save_training_draft(self):
        try:
            record = self._persist_current_task('draft')
        except (ValueError, TrainingJobError, TrainingTaskRegistryError,
                OSError) as exc:
            QMessageBox.warning(self, '无法保存草稿', str(exc))
            return
        self._editing_task_id = record.task_id
        self.btn_save_training_draft.setText('保存修改')
        self.status_message.emit(f'已保存训练草稿: {record.display_name}')
        self._show_task_center(record.task_id)

    def _persist_current_task(self, status: str) -> TrainingTaskRecord:
        job = self._build_training_job()
        if self._editing_task_id:
            existing = self._task_registry.require(self._editing_task_id)
            job = replace(
                job, job_id=existing.task_id, created_at=existing.created_at
            )
        job, request_path, log_path = self._write_task_bundle(job)
        if self._editing_task_id:
            record = self._task_registry.update_job(
                self._editing_task_id, job, request_path
            )
            if status != 'draft':
                record = self._task_registry.set_status(record.task_id, status)
            return record
        return self._task_registry.register_job(
            job, status=status, request_path=request_path,
            log_path=log_path, source='platform',
        )

    def _ensure_training_runtime_dirs(self):
        for directory in (
            self._models_root, self._training_root, self._task_files_root,
            self._training_runs_root, self._runtime_root / 'matplotlib',
            self._runtime_root / 'ultralytics', self._runtime_root / 'cache',
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def _write_task_bundle(self, job: TrainingJob) -> tuple[TrainingJob, Path, Path]:
        """Persist all inspectable task inputs under training/tasks/<id>."""
        self._ensure_training_runtime_dirs()
        task_dir = self._task_files_root / job.job_id
        task_dir.mkdir(parents=True, exist_ok=True)
        dataset_snapshot = task_dir / 'dataset.yaml'
        self._write_dataset_snapshot(Path(job.dataset_yaml), dataset_snapshot)
        bundled_job = replace(job, dataset_yaml=str(dataset_snapshot.resolve()))
        request_path = write_training_job(
            bundled_job, task_dir, filename='training_request.json'
        )
        self._write_task_launcher(task_dir)
        log_path = task_dir / 'training.log'
        log_path.touch(exist_ok=True)
        return bundled_job, request_path, log_path

    @staticmethod
    def _write_dataset_snapshot(source: Path, destination: Path):
        try:
            payload = yaml.safe_load(source.read_text(encoding='utf-8'))
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise TrainingJobError(f'无法写入数据集快照: {exc}') from exc
        if not isinstance(payload, dict):
            raise TrainingJobError('dataset.yaml 根节点必须是 mapping')
        # The task snapshot lives elsewhere, so bind its relative train/val paths
        # to the validated source batch instead of the task directory.
        payload['path'] = str(source.parent.resolve())
        destination.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding='utf-8',
        )

    def _write_task_launcher(self, task_dir: Path):
        project_root = repr(str(self._project_root))
        script = (
            '"""Run this saved training task with the application environment."""\n\n'
            'from pathlib import Path\n'
            'import sys\n\n'
            f'PROJECT_ROOT = Path({project_root})\n'
            'if str(PROJECT_ROOT) not in sys.path:\n'
            '    sys.path.insert(0, str(PROJECT_ROOT))\n\n'
            'from app.training_runner import main\n\n\n'
            "if __name__ == '__main__':\n"
            "    request = Path(__file__).with_name('training_request.json')\n"
            '    raise SystemExit(main([str(request)]))\n'
        )
        (task_dir / 'run_training.py').write_text(script, encoding='utf-8')

    def _start_selected_task(self):
        record = self._selected_task_record()
        if record is None:
            return
        if record.status in {'failed', 'cancelled', 'interrupted'}:
            self._resume_selected_task(record)
            return
        if record.status == 'draft':
            record = self._task_registry.set_status(record.task_id, 'queued')
        if record.status != 'queued':
            return
        if self.is_training():
            self.status_message.emit(f'任务已在队列中: {record.display_name}')
            self.refresh_training_tasks(select_task_id=record.task_id)
            return
        self._launch_task(record)

    def _resume_selected_task(self, record: TrainingTaskRecord):
        checkpoint = self._resume_checkpoint(record)
        if checkpoint is None:
            QMessageBox.information(
                self, '无法继续训练',
                '当前任务没有可用的 weights/last.pt。\n'
                '可以使用“重新训练”按原配置创建一个新任务。',
            )
            return
        if not self._confirm_training_resume(record, checkpoint):
            return
        try:
            self._archive_resume_request(record)
            params = dict(record.parameters)
            params['resume'] = True
            params['exist_ok'] = True
            job = create_training_job(
                task_type=record.task_type,
                model=str(checkpoint),
                batch_root=record.batch_root,
                output_root=record.output_root,
                project_name=record.project_name,
                run_name=record.run_name,
                parameters=params,
            )
            job = replace(
                job, job_id=record.task_id, created_at=record.created_at
            )
            job, request_path, log_path = self._write_task_bundle(job)
            resumed = self._task_registry.relocate_artifacts(
                record.task_id, job, request_path=request_path,
                log_path=log_path,
            )
            resumed = self._task_registry.set_status(
                resumed.task_id, 'queued', error_message=''
            )
            marker = Path(resumed.run_dir) / 'training_complete.json'
            if marker.is_file():
                marker.unlink()
        except (ValueError, TrainingJobError, TrainingTaskRegistryError,
                OSError) as exc:
            QMessageBox.warning(self, '无法继续训练', str(exc))
            return
        if self.is_training():
            self.status_message.emit(
                f'续训任务已加入队列: {resumed.display_name}'
            )
            self._show_task_center(resumed.task_id)
        else:
            self._launch_task(resumed)

    @staticmethod
    def _resume_checkpoint(record: TrainingTaskRecord) -> Path | None:
        if record.status not in {'failed', 'cancelled', 'interrupted'}:
            return None
        checkpoint = Path(record.run_dir) / 'weights' / 'last.pt'
        return checkpoint.resolve() if checkpoint.is_file() else None

    def _confirm_training_resume(self, record: TrainingTaskRecord,
                                 checkpoint: Path) -> bool:
        answer = QMessageBox.question(
            self, '确认继续训练',
            f'任务: {record.display_name}\n'
            f'当前进度: {record.current_epoch}/{record.total_epochs or "?"}\n'
            f'断点: {checkpoint}\n\n'
            '续训将写回原训练目录，并保留已有指标和权重。',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        return answer == QMessageBox.Yes

    def _archive_resume_request(self, record: TrainingTaskRecord):
        request = Path(record.request_path)
        if not request.is_file():
            return
        history = request.parent / 'resume_history'
        history.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        destination = history / (
            f'epoch_{record.current_epoch}_{timestamp}.json'
        )
        shutil.copy2(request, destination)

    def _retry_selected_task(self):
        record = self._selected_task_record()
        if record is None:
            return
        try:
            name = self._next_run_name(
                f'{record.run_name}-retry', record.output_root,
                record.project_name,
            )
            params = dict(record.parameters)
            params['resume'] = False
            params['exist_ok'] = False
            job = create_training_job(
                task_type=record.task_type,
                model=record.model,
                batch_root=record.batch_root,
                output_root=record.output_root,
                project_name=record.project_name,
                run_name=name,
                parameters=params,
            )
            job, request_path, log_path = self._write_task_bundle(job)
            retry = self._task_registry.register_job(
                job, status='queued', request_path=request_path,
                log_path=log_path, source='retry',
            )
        except (ValueError, TrainingJobError, TrainingTaskRegistryError,
                OSError) as exc:
            QMessageBox.warning(self, '无法重试任务', str(exc))
            return
        if self.is_training():
            self._show_task_center(retry.task_id)
        else:
            self._launch_task(retry)

    def _rename_selected_task(self):
        record = self._selected_task_record()
        if record is None:
            return
        name, accepted = QInputDialog.getText(
            self, '重命名训练任务', '显示名称:',
            text=record.display_name,
        )
        if not accepted:
            return
        try:
            self._task_registry.rename(record.task_id, name)
        except TrainingTaskRegistryError as exc:
            QMessageBox.warning(self, '无法重命名', str(exc))
            return
        self.refresh_training_tasks(select_task_id=record.task_id)

    def _save_task_notes(self):
        record = self._selected_task_record()
        if record is None:
            return
        self._task_registry.set_notes(
            record.task_id, self.task_notes_edit.toPlainText()
        )
        self.status_message.emit('任务备注已保存')

    def _archive_selected_task(self):
        record = self._selected_task_record()
        if record is None:
            return
        try:
            updated = self._task_registry.archive(
                record.task_id, not record.archived
            )
        except TrainingTaskRegistryError as exc:
            QMessageBox.warning(self, '无法归档任务', str(exc))
            return
        self.refresh_training_tasks(select_task_id=updated.task_id)

    def _delete_selected_task(self):
        record = self._selected_task_record()
        if record is None or not self._confirm_task_delete(record):
            return
        try:
            run_dir = Path(record.run_dir)
            if run_dir.exists():
                output_root = Path(record.output_root).resolve()
                resolved = run_dir.resolve()
                relative_run = resolved.relative_to(output_root)
                if len(relative_run.parts) < 2:
                    raise ValueError('拒绝删除训练输出根目录或项目根目录')
                shutil.rmtree(resolved)
            task_dir = (
                Path(record.request_path).parent.resolve()
                if record.request_path else None
            )
            if task_dir is not None and task_dir.is_dir():
                task_root = self._task_files_root.resolve()
                task_dir.relative_to(task_root)
                if task_dir == task_root:
                    raise ValueError('拒绝删除训练任务根目录')
                shutil.rmtree(task_dir)
            else:
                for value in (record.request_path, record.log_path):
                    path = Path(value) if value else None
                    if path is not None and path.is_file():
                        path.unlink()
            self._task_registry.delete(record.task_id)
        except (OSError, ValueError, TrainingTaskRegistryError) as exc:
            QMessageBox.critical(self, '删除任务失败', str(exc))
            return
        self.refresh_training_tasks()
        self.status_message.emit(f'已删除训练任务: {record.display_name}')

    def _open_selected_task_files(self):
        record = self._selected_task_record()
        if record is None or not record.request_path:
            return
        folder = Path(record.request_path).parent
        if folder.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _open_selected_task(self):
        record = self._selected_task_record()
        if record is None:
            return
        if self.is_training() and record.task_id != self._active_task_id:
            QMessageBox.information(
                self, '任务正在运行',
                '当前已有训练任务运行。请在其结束后再打开其他任务的历史监控。',
            )
            return
        if record.task_id == self._active_task_id:
            self._show_step(3)
            return
        self._load_task_monitor(record)
        self._show_step(3)

    def _load_task_monitor(self, record: TrainingTaskRecord):
        self._active_training_job = record.job
        self._active_run_dir = Path(record.run_dir)
        self._reset_training_monitor(record.job)
        self.lbl_monitor_run.setText(
            f'{record.project_name} / {record.display_name} · '
            f'{TASK_LABELS.get(record.task_type, record.task_type)}'
        )
        self.lbl_monitor_output.setText(record.run_dir)
        self.lbl_monitor_epoch.setText(
            f'{record.current_epoch} / {record.total_epochs or "-"}'
        )
        self.training_run_progress.setValue(
            round(max(0.0, min(record.progress, 100.0)) * 10)
        )
        self.training_run_progress.setFormat(self._task_progress_text(record))
        status_label, tone = TASK_STATUS_META.get(
            record.status, (record.status, 'neutral')
        )
        self._set_run_status(status_label, tone)
        self.btn_stop_training.setEnabled(False)
        self.btn_view_training_result.setEnabled(Path(record.run_dir).is_dir())

        if record.log_path and Path(record.log_path).is_file():
            try:
                lines = Path(record.log_path).read_text(
                    encoding='utf-8', errors='replace'
                ).splitlines()
                self.training_log.setPlainText('\n'.join(lines[-2000:]))
            except OSError:
                pass
        self._load_saved_task_metrics(record)

    def _load_saved_task_metrics(self, record: TrainingTaskRecord):
        results_path = Path(record.run_dir) / 'results.csv'
        if not results_path.is_file():
            return
        try:
            with results_path.open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
                first_epoch = self._result_row_epoch(rows[0]) if rows else 0
                epoch_offset = 1 if first_epoch == 0 else 0
                for row in rows:
                    raw_epoch = row.get('epoch') or row.get(' epoch') or 0
                    try:
                        epoch = int(float(raw_epoch)) + epoch_offset
                    except (TypeError, ValueError):
                        continue
                    metrics = {}
                    for key, value in row.items():
                        name = str(key or '').strip()
                        if name in {'epoch', 'time'} or not name:
                            continue
                        try:
                            metrics[name] = float(str(value).strip())
                        except (TypeError, ValueError):
                            continue
                    self._update_metric_tree(metrics, epoch)
        except OSError:
            return

    @staticmethod
    def _result_row_epoch(row: dict) -> int:
        raw_epoch = row.get('epoch') or row.get(' epoch') or 0
        try:
            return int(float(raw_epoch))
        except (TypeError, ValueError):
            return 0

    def _confirm_task_delete(self, record: TrainingTaskRecord) -> bool:
        answer = QMessageBox.warning(
            self, '删除训练任务',
            f'将删除任务记录和训练产物:\n{record.run_dir}\n\n'
            '需要只隐藏记录时，请使用“归档”。',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    def _show_task_center(self, task_id: str = ''):
        self._show_step(TASK_CENTER_PAGE)
        self.refresh_training_tasks(select_task_id=task_id)

    def _next_run_name(self, base: str, output_root: str | Path,
                       project_name: str) -> str:
        base = str(base or '训练任务').strip()
        root = Path(output_root).expanduser() / str(project_name).strip()
        candidate = base
        index = 2
        while (
            (root / candidate).exists()
            or self._task_registry.get_by_run_dir(root / candidate) is not None
        ):
            candidate = f'{base}-{index}'
            index += 1
        return candidate

    @staticmethod
    def _task_progress_text(record: TrainingTaskRecord) -> str:
        if record.total_epochs:
            return (
                f'{record.current_epoch} / {record.total_epochs}  ·  '
                f'{record.progress:.1f}%'
            )
        return f'{record.progress:.1f}%'

    @staticmethod
    def _format_task_time(value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
            return parsed.astimezone().strftime('%m-%d %H:%M')
        except (TypeError, ValueError):
            return '-'

    def _on_dataset_root_edited(self):
        self.set_dataset_root(self.dataset_root_edit.text())

    def _choose_dataset_root(self):
        start = str(self._dataset_root or Path.home())
        path = QFileDialog.getExistingDirectory(self, '选择数据项目', start)
        if path:
            self.set_dataset_root(path)

    def _choose_model(self):
        self._models_root.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getOpenFileName(
            self, '选择模型', str(self._models_root),
            'Ultralytics 模型 (*.pt *.yaml *.yml)'
        )
        if path:
            self.model_path_edit.setText(path)

    def _open_models_directory(self):
        self._models_root.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._models_root)))

    def _choose_training_output(self):
        start = self.output_root_edit.text().strip() or str(Path.home())
        path = QFileDialog.getExistingDirectory(
            self, '选择训练输出根目录', start
        )
        if path:
            self.output_root_edit.setText(path)
            QSettings('FilesProcessQT', 'ImageManager').setValue(
                'lastTrainingOutputRoot', path
            )

    def _on_source_selection_changed(self, _item, _column):
        selected = len(self._selected_source_names())
        self.lbl_source_summary.setText(f'已选择 {selected} 个原始批次')
        self._scan_result = None
        self.btn_prepare.setEnabled(
            selected > 0
            and not (self._worker is not None and self._worker.isRunning())
        )

    def _on_source_row_clicked(self, item: QTreeWidgetItem, column: int):
        if column == 0:
            return
        state = (
            Qt.Unchecked
            if item.checkState(0) == Qt.Checked
            else Qt.Checked
        )
        item.setCheckState(0, state)

    def _set_all_source_checks(self, state: Qt.CheckState):
        self.source_tree.blockSignals(True)
        for index in range(self.source_tree.topLevelItemCount()):
            self.source_tree.topLevelItem(index).setCheckState(0, state)
        self.source_tree.blockSignals(False)
        self._on_source_selection_changed(None, 0)

    def _selected_source_names(self) -> tuple[str, ...]:
        names = []
        for index in range(self.source_tree.topLevelItemCount()):
            item = self.source_tree.topLevelItem(index)
            if item.checkState(0) == Qt.Checked:
                names.append(str(item.data(0, Qt.UserRole)))
        return tuple(names)

    def _build_request(self) -> DatasetPreparationRequest:
        if self._dataset_root is None:
            raise DatasetPreparationError('请先选择数据项目')
        return DatasetPreparationRequest(
            dataset_root=self._dataset_root,
            source_names=self._selected_source_names(),
            target_name=self.target_name_edit.text(),
            task_type=self._task_type,
            annotation_dir=self._annotation_dir,
            label_dir=self._label_dir,
            val_ratio=self.val_ratio_spin.value(),
            seed=self.seed_spin.value(),
            use_copy=bool(self.write_mode_combo.currentData()),
            exclude_test=True,
            allow_background_without_label=self.background_checkbox.isChecked(),
            skip_incomplete_samples=True,
            skip_duplicate_samples=True,
            class_names=(),
            keypoints=(),
            left_right_pairs=(),
        )

    def _start_dataset_job(self, prepare: bool):
        if self._worker is not None and self._worker.isRunning():
            return
        try:
            request = self._build_request().normalized()
        except DatasetPreparationError as exc:
            QMessageBox.warning(self, '数据准备', str(exc))
            return

        if prepare and not self._confirm_skipping_incomplete_sources():
            return

        target = request.dataset_root / 'training_data' / request.target_name
        if prepare and target.exists():
            QMessageBox.warning(
                self, '训练批次已存在',
                f'不会覆盖已有训练批次:\n{target}',
            )
            return

        self._set_busy(True, '正在生成训练数据' if prepare else '正在扫描数据')
        self._worker = _DatasetPreparationWorker(request, prepare, self)
        self._worker.scan_ready.connect(self._on_scan_ready)
        self._worker.preparation_ready.connect(self._on_preparation_ready)
        self._worker.failed.connect(self._on_dataset_job_failed)
        self._worker.finished.connect(self._on_dataset_job_finished)
        self._worker.start()

    def _on_scan_ready(self, result: DatasetScanResult):
        self._scan_result = result
        self.btn_prepare.setEnabled(False)
        skipped = len(result.missing_annotations) + len(result.missing_labels)
        duplicates = len(result.duplicate_images)
        status = (
            f'有效 {len(result.samples)} · 测试排除 {len(result.test_excluded)} · '
            f'重名去重 {duplicates} · '
            f'缺 JSON {len(result.missing_annotations)} · '
            f'缺 TXT {len(result.missing_labels)} · '
            f'格式异常 {len(result.invalid_labels)}'
        )
        self.lbl_source_summary.setText(status)
        self.lbl_preparation_state.setText(
            (
                f'预检通过 · 跳过 {skipped} 张未配对图片 · 去重 {duplicates} 张'
                if result.can_prepare and (skipped or duplicates)
                else ('预检通过' if result.can_prepare else '预检未通过')
            )
        )
        if not result.can_prepare:
            QMessageBox.warning(
                self,
                '训练标签预检未通过',
                result.blocking_message(),
            )

    def _on_preparation_ready(self, prepared: PreparedDataset):
        self._current_batch = prepared.batch_root
        self.lbl_task_state.setText(
            f'数据已准备 · {prepared.total_count} 张\n'
            f'TRAIN {prepared.train_count} / VAL {prepared.val_count}'
        )
        self.lbl_preparation_state.setText('训练数据已生成')
        self.progress.setValue(100)
        self.refresh_existing_batches()
        self._update_review_summary(inspect_training_batch(prepared.batch_root))
        self.dataset_prepared.emit(str(prepared.batch_root), self._task_type)
        self.status_message.emit(f'训练数据已生成: {prepared.batch_root.name}')
        self._show_step(1)

    def _on_dataset_job_failed(self, message: str):
        self.lbl_preparation_state.setText('数据准备失败')
        QMessageBox.critical(self, '数据准备失败', message)

    def _on_dataset_job_finished(self):
        self._set_busy(False)
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

    def _set_busy(self, busy: bool, text: str = ''):
        self.btn_scan.setEnabled(not busy)
        self.btn_prepare.setEnabled(
            not busy and bool(self._selected_source_names())
        )
        self.btn_refresh_sources.setEnabled(not busy)
        if busy:
            self.progress.setRange(0, 0)
            self.lbl_preparation_state.setText(text)
        else:
            self.progress.setRange(0, 100)
            if self.progress.value() < 100:
                self.progress.setValue(0)

    def _confirm_skipping_incomplete_sources(self) -> bool:
        incomplete = []
        for index in range(self.source_tree.topLevelItemCount()):
            item = self.source_tree.topLevelItem(index)
            if item.checkState(0) != Qt.Checked:
                continue
            images = int(item.text(2) or 0)
            annotations = int(item.text(3) or 0)
            labels = int(item.text(4) or 0)
            if annotations >= images and labels >= images:
                continue
            incomplete.append(
                f'{item.text(1)}: 图片 {images}, JSON {annotations}, TXT {labels}'
            )
        if not incomplete:
            return True
        preview = '\n'.join(incomplete[:8])
        if len(incomplete) > 8:
            preview += f'\n... 还有 {len(incomplete) - 8} 个批次'
        answer = QMessageBox.question(
            self,
            '使用待补齐数据',
            '选中的批次包含未配对图片。生成时将只保留同时具有图片、JSON '
            '和 YOLO TXT 的有效样本，缺少标注的图片会被跳过。\n\n'
            f'{preview}\n\n是否继续？',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    def _inspect_selected_existing_batch(self, *_args):
        path = self._selected_existing_batch_path()
        self.btn_use_existing.setEnabled(False)
        self.btn_view_existing.setEnabled(False)
        self.btn_open_existing_folder.setEnabled(False)
        self.btn_rename_existing.setEnabled(False)
        self.btn_delete_existing.setEnabled(False)
        if not path:
            self.lbl_existing_state.setText('没有可用训练批次')
            self.lbl_batch_source_hint.setText('选择已有批次后，这里显示其数据来源')
            self._batch_source_names = ()
            return
        summary = inspect_training_batch(path)
        self.btn_use_existing.setEnabled(summary.image_count > 0)
        self.btn_view_existing.setEnabled(summary.image_count > 0)
        self.btn_open_existing_folder.setEnabled(path.is_dir())
        self.btn_rename_existing.setEnabled(path.is_dir())
        self.btn_delete_existing.setEnabled(path.is_dir())
        self._batch_source_names = self._read_batch_source_names(path)
        self._sync_sources_for_batch(self._batch_source_names)
        if self._batch_source_names:
            self.lbl_batch_source_hint.setText(
                '当前训练批次来源：' + '、'.join(self._batch_source_names)
                + '\n已在右侧原始批次列表中同步选中，可继续调整后生成新批次。'
            )
        else:
            self.lbl_batch_source_hint.setText(
                '当前训练批次没有可追溯的来源清单（缺少 preparation_manifest.json）'
            )
        self.lbl_existing_state.setText(
            f'图片 {summary.image_count} · JSON {summary.annotation_count} · '
            f'TXT {summary.label_count} · '
            + ('训练数据可用' if summary.is_ready else summary.readiness_message())
        )

    def _use_existing_batch(self):
        path = self._selected_existing_batch_path()
        if not path:
            return
        summary = inspect_training_batch(path)
        was_pending = not summary.is_ready
        if was_pending:
            answer = QMessageBox.question(
                self,
                '训练批次待定',
                f'批次“{summary.batch_root.name}”当前尚未完成训练数据预检。\n\n'
                f'{summary.readiness_message()}\n\n'
                '仍然使用此批次并进入审查/配置吗？平台会在启动训练前再次检查，'
                '不会跳过结构性错误。',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if answer != QMessageBox.Yes:
                return
        try:
            # This also creates a missing batch-local YAML when train/val
            # directories already exist and repairs an old keypoint schema.
            ensure_training_dataset_yaml(summary.batch_root)
            summary = inspect_training_batch(summary.batch_root)
        except DatasetPreparationError as exc:
            QMessageBox.warning(self, '批次 YAML 分析失败', str(exc))

        if not summary.is_split and summary.image_count > 1:
            split_answer = QMessageBox.question(
                self,
                '批次尚未划分',
                f'批次“{summary.batch_root.name}”还没有 train/val 图片划分。\n\n'
                '是否在当前训练批次目录内按 8:2 生成划分，并重新分析 dataset.yaml？\n'
                '顶层 images/labels 原始文件不会被修改。',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if split_answer == QMessageBox.Yes:
                try:
                    prepare_existing_batch_split(summary.batch_root)
                    ensure_training_dataset_yaml(summary.batch_root)
                    summary = inspect_training_batch(summary.batch_root)
                except DatasetPreparationError as exc:
                    QMessageBox.warning(self, '生成 train/val 失败', str(exc))
        if was_pending:
            self.refresh_existing_batches()
        if summary.task_type in TASK_LABELS and summary.task_type != self._task_type:
            self._set_task(summary.task_type)
        self._current_batch = summary.batch_root
        self._update_review_summary(summary)
        state = '已选择训练批次' if summary.is_ready else '已选择待定批次'
        if was_pending and summary.is_ready:
            state += '（已按实际标签结构更新 YAML）'
        self.lbl_task_state.setText(f'{state}\n{summary.batch_root.name}')
        self._show_step(1)

    def _view_existing_batch(self):
        path = self._selected_existing_batch_path()
        if not path:
            return
        summary = inspect_training_batch(path)
        self._current_batch = summary.batch_root
        if summary.task_type in TASK_LABELS:
            self._task_type = summary.task_type
        self.review_dataset_requested.emit(
            str(summary.batch_root), summary.task_type
        )

    def _open_existing_batch_folder(self):
        path = self._selected_existing_batch_path()
        if path is not None and path.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _rename_existing_batch(self):
        path = self._selected_existing_batch_path()
        training_root = self._dataset_root / 'training_data' if self._dataset_root else None
        if path is None or training_root is None or not path.is_dir():
            return
        try:
            path.resolve().relative_to(training_root.resolve())
        except ValueError:
            QMessageBox.warning(self, '无法重命名', '当前批次不在数据项目的 training_data 目录内。')
            return
        name, accepted = QInputDialog.getText(
            self, '重命名训练批次', '新的批次名称:', text=path.name
        )
        if not accepted:
            return
        name = str(name).strip()
        target = training_root / name
        if not name or name in {'.', '..'} or Path(name).is_absolute() or len(Path(name).parts) != 1:
            QMessageBox.warning(self, '无法重命名', '批次名称必须是单个目录名称。')
            return
        if target.exists() and target.resolve() != path.resolve():
            QMessageBox.warning(self, '无法重命名', f'目标批次已存在：{target.name}')
            return
        if target.resolve() == path.resolve():
            return
        try:
            path.rename(target)
            try:
                self._update_batch_manifest_name(target, name)
            except Exception:
                target.rename(path)
                raise
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            QMessageBox.critical(self, '重命名失败', str(exc))
            return
        if self._current_batch is not None and self._current_batch.resolve() == path.resolve():
            self._current_batch = target
        self.refresh_existing_batches()
        for index in range(self.existing_batch_tree.topLevelItemCount()):
            item = self.existing_batch_tree.topLevelItem(index)
            if Path(str(item.data(0, Qt.UserRole))).resolve() == target.resolve():
                self.existing_batch_tree.setCurrentItem(item)
                break
        self.status_message.emit(f'训练批次已重命名为: {name}')

    @staticmethod
    def _update_batch_manifest_name(batch: Path, name: str):
        manifest = batch / 'preparation_manifest.json'
        if not manifest.is_file():
            return
        payload = json.loads(manifest.read_text(encoding='utf-8'))
        if not isinstance(payload, dict):
            return
        request = payload.get('request')
        if isinstance(request, dict):
            request['target_name'] = name
        manifest.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

    def _delete_existing_batch(self):
        path = self._selected_existing_batch_path()
        training_root = self._dataset_root / 'training_data' if self._dataset_root else None
        if path is None or training_root is None or not path.is_dir():
            return
        try:
            resolved_root = training_root.resolve()
            resolved_path = path.resolve()
            relative = resolved_path.relative_to(resolved_root)
            if len(relative.parts) != 1 or path.name.startswith('.'):
                raise ValueError('拒绝删除 training_data 根目录或临时目录')
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, '无法删除', str(exc))
            return
        linked_active = []
        for record in self._task_registry.list_tasks():
            if record.status not in ACTIVE_STATUSES and record.status != 'queued':
                continue
            try:
                linked = Path(record.batch_root).expanduser().resolve()
            except OSError:
                linked = Path(record.batch_root).expanduser().absolute()
            if linked == resolved_path:
                linked_active.append(record.display_name)
        if linked_active:
            QMessageBox.warning(
                self,
                '无法删除',
                '以下训练任务正在排队或运行，不能删除其数据批次：\n'
                + '、'.join(linked_active),
            )
            return
        answer = QMessageBox.warning(
            self,
            '删除训练批次',
            f'将删除批次目录及其中的 train/val、dataset.yaml、审查报告：\n{resolved_path}\n\n'
            '原始 images、annotations、labels 不会被删除。此操作不可自动恢复，是否继续？',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        if self.is_training() and self._active_training_job is not None:
            active_batch = Path(self._active_training_job.batch_root).resolve()
            if active_batch == resolved_path:
                QMessageBox.warning(self, '无法删除', '当前批次正在被训练任务使用，请先停止训练。')
                return
        try:
            shutil.rmtree(resolved_path)
        except OSError as exc:
            QMessageBox.critical(self, '删除失败', str(exc))
            return
        if self._current_batch is not None and self._current_batch.resolve() == resolved_path:
            self._current_batch = None
            self._batch_source_names = ()
            self.lbl_batch_source_hint.setText('选择已有批次后，这里显示其数据来源')
            self.lbl_existing_state.setText('尚未选择训练批次')
            self.lbl_task_state.setText('数据尚未准备')
        self.refresh_existing_batches()
        self.status_message.emit(f'已删除训练批次: {path.name}')

    def _selected_existing_batch_path(self) -> Path | None:
        item = self.existing_batch_tree.currentItem()
        if item is None:
            return None
        value = item.data(0, Qt.UserRole)
        return Path(str(value)) if value else None

    @staticmethod
    def _read_batch_source_names(path: str | Path) -> tuple[str, ...]:
        manifest = Path(path) / 'preparation_manifest.json'
        try:
            payload = json.loads(manifest.read_text(encoding='utf-8'))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return ()
        request = payload.get('request') if isinstance(payload, dict) else None
        values = request.get('source_names') if isinstance(request, dict) else ()
        if not isinstance(values, (list, tuple)):
            return ()
        return tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))

    def _sync_sources_for_batch(self, source_names: tuple[str, ...]):
        if not hasattr(self, 'source_tree'):
            return
        selected = set(source_names)
        self.source_tree.blockSignals(True)
        for index in range(self.source_tree.topLevelItemCount()):
            item = self.source_tree.topLevelItem(index)
            name = str(item.data(0, Qt.UserRole) or item.text(1))
            item.setCheckState(0, Qt.Checked if name in selected else Qt.Unchecked)
        self.source_tree.blockSignals(False)
        self._on_source_selection_changed(None, 0)

    def _update_review_summary(self, summary: TrainingBatchSummary):
        self._current_batch = summary.batch_root
        self.lbl_review_status.setText(
            '训练数据预检通过' if summary.is_ready else '训练数据存在阻断问题'
        )
        self.lbl_review_status.setProperty('ready', summary.is_ready)
        self.lbl_review_status.style().unpolish(self.lbl_review_status)
        self.lbl_review_status.style().polish(self.lbl_review_status)
        self.lbl_review_batch.setText(str(summary.batch_root))
        self._populate_summary_tree(self.review_summary_tree, summary)
        self.btn_open_review.setEnabled(summary.image_count > 0)
        self.btn_recheck_batch.setEnabled(True)
        self.step_buttons[1].setEnabled(True)
        self.step_buttons[2].setEnabled(summary.is_ready)
        self.btn_start_training.setEnabled(
            summary.is_ready
        )
        self.btn_save_training_draft.setEnabled(summary.is_ready)
        self.lbl_config_dataset.setText(summary.batch_root.name)
        yaml_state = 'YAML 就绪' if summary.has_dataset_yaml else 'YAML 待生成'
        self.lbl_config_dataset_meta.setText(
            f'{TASK_LABELS.get(summary.task_type, summary.task_type)} · '
            f'{summary.image_count} 张 · '
            f'Train {summary.train_image_count} / Val {summary.val_image_count} · '
            f'{yaml_state}'
            + (
                f' · TXT 期望 {summary.expected_label_columns} / '
                f'实际 {", ".join(map(str, summary.observed_label_columns))}'
                if summary.expected_label_columns else ''
            )
        )
        self.btn_config_view_data.setEnabled(summary.image_count > 0)
        suggested_run_name = f'{summary.batch_root.name}-{summary.task_type}'
        current_run_name = self.run_name_edit.text().strip()
        if not current_run_name or current_run_name == self._auto_run_name:
            self._auto_run_name = suggested_run_name
            self.run_name_edit.setText(suggested_run_name)

    def _populate_summary_tree(self, tree: QTreeWidget,
                               summary: TrainingBatchSummary):
        tree.clear()
        rows = (
            ('任务 / 标注目录',
             f'{summary.task_type} / {summary.annotation_dir}',
             bool(summary.annotation_dir)),
            ('合并图片', summary.image_count, not summary.missing_top_labels),
            ('JSON 标注', summary.annotation_count,
             summary.annotation_count == summary.image_count),
            ('YOLO TXT', summary.label_count, not summary.missing_top_labels),
            ('TRAIN 图片 / TXT',
             f'{summary.train_image_count} / {summary.train_label_count}',
             not summary.missing_train_labels and summary.train_image_count > 0),
            ('VAL 图片 / TXT',
             f'{summary.val_image_count} / {summary.val_label_count}',
             not summary.missing_val_labels and summary.val_image_count > 0),
            ('YOLO 标签结构',
             (
                 f'期望 {summary.expected_label_columns} 列 / 实际 '
                 f'{", ".join(map(str, summary.observed_label_columns)) or "-"} 列'
                 if summary.expected_label_columns else '按任务默认格式'
             ),
             not summary.invalid_train_labels and not summary.invalid_val_labels),
            ('dataset.yaml', 1 if summary.has_dataset_yaml else 0,
             summary.has_dataset_yaml),
        )
        for label, value, ok in rows:
            item = QTreeWidgetItem([label, str(value), '通过' if ok else '待处理'])
            tree.addTopLevelItem(item)
            self._set_tree_status(
                tree, item, 2,
                '通过' if ok else '待处理',
                'success' if ok else 'warning',
            )

    def _refresh_current_batch(self):
        if self._current_batch is None:
            return
        self._update_review_summary(inspect_training_batch(self._current_batch))

    def _request_review(self):
        if self._current_batch is None:
            return
        self.review_dataset_requested.emit(
            str(self._current_batch), self._task_type
        )

    def _initialize_training_output(self):
        settings = QSettings('FilesProcessQT', 'ImageManager')
        saved = str(settings.value('lastTrainingOutputRoot') or '').strip()
        default_root = self._training_runs_root
        legacy_root = self._project_root / 'training_runs'
        saved_path = Path(saved).expanduser() if saved else None
        self.output_root_edit.setText(
            str(default_root) if saved_path and saved_path == legacy_root
            else str(saved_path) if saved_path and saved_path.is_dir()
            else str(default_root)
        )
        saved_project = str(
            settings.value('lastTrainingProjectName') or ''
        ).strip()
        if saved_project:
            self.project_name_edit.setText(saved_project)
            self._auto_project_name = saved_project

    def _start_training(self):
        try:
            job = self._build_training_job()
        except (ValueError, TrainingJobError) as exc:
            QMessageBox.warning(self, '训练配置无效', str(exc))
            return
        if not self._confirm_training_start(job):
            return

        try:
            record = self._persist_current_task('queued')
        except (OSError, TrainingTaskRegistryError, TrainingJobError,
                ValueError) as exc:
            QMessageBox.critical(self, '无法创建训练任务', str(exc))
            return

        self._editing_task_id = ''
        if self.is_training():
            self.status_message.emit(f'任务已加入队列: {record.display_name}')
            self._show_task_center(record.task_id)
            return
        self._launch_task(record)

    def _launch_task(self, record: TrainingTaskRecord):
        if self.is_training():
            return
        if record.status == 'draft':
            record = self._task_registry.set_status(record.task_id, 'queued')
        if record.status != 'queued':
            return
        job = record.job
        run_exists = Path(record.run_dir).exists()
        resume_checkpoint = self._job_resume_checkpoint(record)
        if run_exists and resume_checkpoint is None:
            self._task_registry.set_status(
                record.task_id, 'failed',
                error_message='输出目录已存在，请复制配置或重新训练',
            )
            self.refresh_training_tasks(select_task_id=record.task_id)
            QMessageBox.warning(
                self, '无法启动任务',
                '输出目录已存在，但任务没有有效断点。\n'
                '请在任务中心使用“复制配置”或“重新训练”。',
            )
            return
        try:
            self._ensure_training_runtime_dirs()
            request_path = Path(record.request_path)
            if not request_path.is_file():
                raise OSError(f'任务请求文件不存在: {request_path}')
            Path(job.output_root).mkdir(parents=True, exist_ok=True)
        except (OSError, TrainingTaskRegistryError) as exc:
            self._task_registry.set_status(
                record.task_id, 'failed', error_message=str(exc)
            )
            self.refresh_training_tasks(select_task_id=record.task_id)
            return

        settings = QSettings('FilesProcessQT', 'ImageManager')
        settings.setValue('lastTrainingOutputRoot', job.output_root)
        settings.setValue('lastTrainingProjectName', job.project_name)

        self._active_training_job = job
        self._active_training_job_path = request_path
        self._active_run_dir = job.run_dir
        self._active_task_id = record.task_id
        self._training_terminal_event = ''
        self._training_stop_requested = False
        self._training_output_buffer = ''
        self._task_registry.set_status(record.task_id, 'preparing')
        self._reset_training_monitor(job)
        if resume_checkpoint is not None:
            self.lbl_monitor_epoch.setText(
                f'{record.current_epoch} / {record.total_epochs or "-"}'
            )
            self.training_run_progress.setValue(
                round(max(0.0, min(record.progress, 100.0)) * 10)
            )
            self.training_run_progress.setFormat(
                f'正在恢复 · {self._task_progress_text(record)}'
            )
            self._load_saved_task_metrics(record)

        process = QProcess(self)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.setWorkingDirectory(str(self._project_root))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert('PYTHONUNBUFFERED', '1')
        environment.insert('MPLCONFIGDIR', str(self._runtime_root / 'matplotlib'))
        environment.insert('YOLO_CONFIG_DIR', str(self._runtime_root / 'ultralytics'))
        environment.insert('XDG_CACHE_HOME', str(self._runtime_root / 'cache'))
        process.setProcessEnvironment(environment)
        program, arguments = self._training_command(request_path)
        process.setProgram(program)
        process.setArguments(arguments)
        process.started.connect(self._on_training_process_started)
        process.readyReadStandardOutput.connect(self._read_training_output)
        process.errorOccurred.connect(self._on_training_process_error)
        process.finished.connect(self._on_training_process_finished)
        self._training_process = process
        self._set_training_controls(True)
        self._show_step(3)
        process.start()

    @staticmethod
    def _job_resume_checkpoint(record: TrainingTaskRecord) -> Path | None:
        if not record.parameters.get('resume'):
            return None
        checkpoint = Path(record.model).expanduser()
        expected = Path(record.run_dir) / 'weights' / 'last.pt'
        try:
            matches = checkpoint.resolve() == expected.resolve()
        except OSError:
            matches = False
        return checkpoint.resolve() if matches and checkpoint.is_file() else None

    def _start_next_queued_task(self):
        if self.is_training():
            return
        queued = self._task_registry.next_queued()
        if queued is not None:
            self._launch_task(queued)
        else:
            self.refresh_training_tasks()

    def _build_training_job(self) -> TrainingJob:
        if self._current_batch is None:
            raise TrainingJobError('请先选择并审查一个训练批次')
        config = self.current_training_config()
        return create_training_job(
            task_type=self._task_type,
            model=config.model,
            batch_root=self._current_batch,
            output_root=self.output_root_edit.text(),
            project_name=self.project_name_edit.text(),
            run_name=self.run_name_edit.text(),
            parameters=config.parameters,
        )

    def _training_command(self, job_path: Path) -> tuple[str, list[str]]:
        return sys.executable, [
            '-u', '-m', 'app.training_runner', str(job_path)
        ]

    def _confirm_training_start(self, job: TrainingJob) -> bool:
        queue_message = (
            '当前有任务在运行，新任务将加入单任务队列。'
            if self.is_training() else '训练将在独立进程中启动。'
        )
        answer = QMessageBox.question(
            self,
            '确认开始训练',
            f'任务: {job.run_name}\n'
            f'模型: {job.model}\n'
            f'数据: {Path(job.batch_root).name}\n'
            f'Epochs: {job.parameters.get("epochs", "-")}\n'
            f'输出: {job.run_dir}\n\n' + queue_message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        return answer == QMessageBox.Yes

    def _reset_training_monitor(self, job: TrainingJob):
        self.training_log.clear()
        self.training_metric_tree.clear()
        self._metric_items.clear()
        self.training_curve_chart.clear(
            expected_epochs=job.parameters.get('epochs', 0)
        )
        self.training_metric_group_combo.setCurrentIndex(0)
        self.training_metric_tabs.setCurrentIndex(0)
        self.lbl_monitor_run.setText(
            f'{job.project_name} / {job.run_name} · {TASK_LABELS[job.task_type]}'
        )
        self.lbl_monitor_output.setText(str(job.run_dir))
        self.lbl_monitor_epoch.setText(
            f'0 / {job.parameters.get("epochs", 0)}'
        )
        self.lbl_monitor_elapsed.setText('00:00:00')
        for label in (
            self.lbl_monitor_cpu, self.lbl_monitor_memory,
            self.lbl_monitor_gpu, self.lbl_monitor_vram,
        ):
            label.setText('-')
        self.training_run_progress.setValue(0)
        self.training_run_progress.setFormat('正在创建训练进程')
        self.btn_view_training_result.setEnabled(False)
        self._set_run_status('正在启动', 'running')

    def _on_training_process_started(self):
        self._training_started_at = time.monotonic()
        if self._active_task_id:
            self._task_registry.set_status(
                self._active_task_id, 'running',
                pid=int(self._training_process.processId()),
            )
        self._set_run_status('初始化环境', 'running')
        self.training_run_progress.setFormat('正在加载模型与数据')
        self._append_training_log(
            f'[平台] 训练进程已启动: PID {self._training_process.processId()}'
        )
        self._initialize_hardware_sampling()
        self._training_runtime_timer.start()
        self._update_training_runtime()

    def _read_training_output(self):
        process = self._training_process
        if process is None:
            return
        chunk = bytes(process.readAllStandardOutput()).decode(
            'utf-8', errors='replace'
        )
        self._training_output_buffer += chunk.replace('\r', '\n')
        while '\n' in self._training_output_buffer:
            line, self._training_output_buffer = (
                self._training_output_buffer.split('\n', 1)
            )
            self._consume_training_line(line)

    def _consume_training_line(self, line: str):
        line = line.strip()
        if not line:
            return
        if line.startswith(TRAIN_EVENT_PREFIX):
            try:
                event = json.loads(line[len(TRAIN_EVENT_PREFIX):])
            except json.JSONDecodeError:
                self._append_training_log('[平台] 收到无效训练事件')
                return
            if isinstance(event, dict):
                self._handle_training_event(event)
            return
        clean = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', line)
        self._append_training_log(clean)

    def _handle_training_event(self, event: dict):
        event_type = str(event.get('type') or '')
        if event_type == 'initializing':
            if self._active_task_id:
                self._task_registry.set_status(
                    self._active_task_id, 'preparing'
                )
            self._set_run_status('加载模型', 'running')
            self.training_run_progress.setFormat('正在加载模型与数据')
        elif event_type == 'started':
            save_dir = event.get('save_dir')
            if save_dir:
                self._active_run_dir = Path(str(save_dir))
                self.lbl_monitor_output.setText(str(save_dir))
            self._set_run_status('训练中', 'running')
            self.training_run_progress.setFormat('Epoch 0 / %s' % (
                event.get('epochs') or '-'
            ))
            self._append_training_log(
                f'[平台] Ultralytics 已就绪，设备: {event.get("device", "-")}'
            )
        elif event_type == 'epoch':
            epoch = int(event.get('epoch') or 0)
            epochs = int(event.get('epochs') or 0)
            progress = max(0.0, min(float(event.get('progress') or 0), 100.0))
            self.lbl_monitor_epoch.setText(f'{epoch} / {epochs}')
            self.training_run_progress.setValue(round(progress * 10))
            self.training_run_progress.setFormat(
                f'Epoch {epoch} / {epochs}  ·  {progress:.1f}%'
            )
            self._update_metric_tree(event.get('metrics'), epoch)
            if self._active_task_id:
                self._task_registry.update_progress(
                    self._active_task_id, epoch=epoch, epochs=epochs,
                    progress=progress,
                )
                self.refresh_training_tasks(
                    select_task_id=self._active_task_id
                )
        elif event_type == 'finalizing':
            self._set_run_status('正在保存', 'running')
            self.training_run_progress.setFormat('正在保存权重与结果')
        elif event_type == 'completed':
            self._training_terminal_event = 'completed'
            save_dir = event.get('save_dir')
            if save_dir:
                self._active_run_dir = Path(str(save_dir))
                self.lbl_monitor_output.setText(str(save_dir))
            self._show_completed_progress(event)
            if self._active_task_id:
                self._task_registry.set_status(
                    self._active_task_id, 'completed'
                )
        elif event_type == 'cancelled':
            self._training_terminal_event = 'cancelled'
            self._set_run_status('已停止', 'warning')
            self.training_run_progress.setFormat('训练任务已停止')
            self._append_training_log(
                '[平台] ' + str(event.get('message') or '训练任务已停止')
            )
            if self._active_task_id:
                self._task_registry.set_status(
                    self._active_task_id, 'cancelled',
                    error_message=str(event.get('message') or ''),
                )
        elif event_type == 'failed':
            self._training_terminal_event = 'failed'
            self._set_run_status('训练失败', 'danger')
            self.training_run_progress.setFormat('训练失败，请查看日志')
            raw_message = str(event.get('message') or '未知错误')
            self._append_training_log('[平台] 训练失败: ' + raw_message)
            message = self._training_failure_summary(raw_message)
            if self._active_task_id:
                self._task_registry.set_status(
                    self._active_task_id, 'failed',
                    error_message=message,
                )

    def _change_training_metric_group(self, _index=0):
        group = self.training_metric_group_combo.currentData()
        self.training_curve_chart.set_metric_group(str(group or 'loss'))

    def _update_metric_tree(self, metrics, epoch: int = 0):
        if not isinstance(metrics, dict):
            return
        for key, value in metrics.items():
            name = str(key)
            text = self._format_metric_value(value)
            item = self._metric_items.get(name)
            if item is None:
                item = QTreeWidgetItem([name, text])
                self.training_metric_tree.addTopLevelItem(item)
                self._metric_items[name] = item
            else:
                item.setText(1, text)
        self.training_curve_chart.append_metrics(epoch, metrics)

    def _on_training_process_error(self, error):
        if error != QProcess.FailedToStart:
            return
        self._training_terminal_event = 'failed'
        self._set_run_status('启动失败', 'danger')
        self.training_run_progress.setFormat('无法启动训练进程')
        process = self._training_process
        message = process.errorString() if process is not None else str(error)
        self._append_training_log(f'[平台] 无法启动训练进程: {message}')
        if self._active_task_id:
            self._task_registry.set_status(
                self._active_task_id, 'failed', error_message=message
            )
        QTimer.singleShot(
            0,
            lambda: self._on_training_process_finished(
                -1, QProcess.CrashExit
            ),
        )

    def _on_training_process_finished(self, exit_code: int, _exit_status):
        if self._training_process is None:
            return
        if self._training_output_buffer.strip():
            self._consume_training_line(self._training_output_buffer)
        self._training_output_buffer = ''
        if not self._training_terminal_event:
            if self._training_stop_requested:
                self._training_terminal_event = 'cancelled'
                self._set_run_status('已停止', 'warning')
                self.training_run_progress.setFormat('训练任务已停止')
            elif exit_code == 0:
                self._training_terminal_event = 'completed'
                self._show_completed_progress()
            else:
                self._training_terminal_event = 'failed'
                self._set_run_status('训练失败', 'danger')
                self.training_run_progress.setFormat(
                    f'训练进程异常退出 · code {exit_code}'
                )
        self._append_training_log(f'[平台] 训练进程已退出: code {exit_code}')
        process = self._training_process
        self._training_process = None
        if process is not None:
            process.deleteLater()
        self._training_runtime_timer.stop()
        self._update_training_runtime()
        self._shutdown_hardware_sampling()
        self._set_training_controls(False)

        job = self._active_training_job
        task_id = self._active_task_id
        if task_id:
            status = self._training_terminal_event or 'failed'
            error = ''
            if status == 'failed':
                current = self._task_registry.get(task_id)
                error = current.error_message if current is not None else ''
                if not error:
                    lines = self.training_log.toPlainText().splitlines()[-1:]
                    error = lines[0] if lines else f'训练进程退出: {exit_code}'
            self._task_registry.set_status(
                task_id, status, error_message=error
            )
        if self._training_terminal_event == 'completed' and job is not None:
            run_dir = self._active_run_dir or job.run_dir
            self.btn_view_training_result.setEnabled(run_dir.is_dir())
            self.lbl_task_state.setText(f'训练已完成\n{job.run_name}')
            self.status_message.emit(f'训练完成: {job.run_name}')
            self.training_completed.emit(job.output_root, str(run_dir))
        elif self._training_terminal_event == 'cancelled':
            self.lbl_task_state.setText('训练任务已停止')
            self.status_message.emit('训练任务已停止')
        else:
            self.lbl_task_state.setText('训练失败，请查看任务日志')
            self.status_message.emit('训练失败，请查看任务监控日志')
        self.refresh_training_tasks(select_task_id=task_id)
        self._active_task_id = ''
        if not self._closing:
            QTimer.singleShot(0, self._start_next_queued_task)

    def _show_completed_progress(self, event: dict | None = None):
        event = event or {}
        record = (
            self._task_registry.get(self._active_task_id)
            if self._active_task_id else None
        )
        epoch = int(event.get('epoch') or (
            record.current_epoch if record is not None else 0
        ))
        epochs = int(event.get('epochs') or (
            record.total_epochs if record is not None else 0
        ))
        progress = float(event.get('progress') or (
            epoch / epochs * 100.0 if epochs else 0.0
        ))
        progress = max(0.0, min(progress, 100.0))
        if self._active_task_id and (epoch or epochs):
            self._task_registry.update_progress(
                self._active_task_id, epoch=epoch, epochs=epochs,
                progress=progress,
            )
        self.lbl_monitor_epoch.setText(f'{epoch} / {epochs or "-"}')
        self.training_run_progress.setValue(round(progress * 10))
        if epochs and epoch < epochs:
            self.training_run_progress.setFormat(
                f'提前结束 · {epoch} / {epochs}  ·  {progress:.1f}%'
            )
            self._set_run_status('提前结束', 'warning')
        else:
            self.training_run_progress.setFormat('训练完成 · 100%')
            self._set_run_status('训练完成', 'success')

    def _stop_training(self):
        process = self._training_process
        if process is None or process.state() == QProcess.NotRunning:
            return
        if not self._confirm_training_stop():
            return
        self._training_stop_requested = True
        if self._active_task_id:
            self._task_registry.set_status(self._active_task_id, 'stopping')
        self._set_run_status('正在停止', 'warning')
        self.btn_stop_training.setEnabled(False)
        self._append_training_log('[平台] 正在请求训练进程安全停止')
        process.terminate()
        QTimer.singleShot(5000, self._kill_training_if_needed)

    def _confirm_training_stop(self) -> bool:
        answer = QMessageBox.question(
            self, '停止训练',
            '确定停止当前训练任务吗？已保存的周期权重会保留。',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    def _kill_training_if_needed(self):
        process = self._training_process
        if process is not None and process.state() != QProcess.NotRunning:
            self._append_training_log('[平台] 训练进程未响应，正在强制结束')
            process.kill()

    def shutdown_training(self):
        process = self._training_process
        if process is None:
            return
        self._training_stop_requested = True
        process.terminate()
        if not process.waitForFinished(1500):
            process.kill()
            process.waitForFinished(1000)

    def closeEvent(self, event):
        self._closing = True
        self.shutdown_training()
        try:
            self._task_registry.close()
        except Exception:
            pass
        super().closeEvent(event)

    def is_training(self) -> bool:
        return self._training_process is not None

    def _set_training_controls(self, running: bool):
        ready = (
            self._current_batch is not None
            and inspect_training_batch(self._current_batch).is_ready
        )
        self.btn_start_training.setEnabled(ready)
        self.btn_start_training.setText('加入队列' if running else '开始训练')
        self.btn_save_training_draft.setEnabled(ready)
        self.btn_stop_training.setEnabled(running)
        self.task_combo.setEnabled(True)
        self.btn_browse_model.setEnabled(True)
        self.btn_browse_output.setEnabled(True)
        self.btn_advanced_config.setEnabled(True)

    def _set_run_status(self, text: str, tone: str):
        self.lbl_monitor_status.setText(text)
        self.lbl_monitor_status.setProperty('tone', tone)
        self.lbl_monitor_status.style().unpolish(self.lbl_monitor_status)
        self.lbl_monitor_status.style().polish(self.lbl_monitor_status)

    def _append_training_log(self, text: str):
        if text:
            self.training_log.appendPlainText(text)
            if self._active_task_id:
                record = self._task_registry.get(self._active_task_id)
                if record is not None and record.log_path:
                    try:
                        Path(record.log_path).parent.mkdir(
                            parents=True, exist_ok=True
                        )
                        with Path(record.log_path).open(
                            'a', encoding='utf-8'
                        ) as stream:
                            stream.write(text.rstrip() + '\n')
                    except OSError:
                        pass

    def _request_trained_model(self):
        job = self._active_training_job
        run_dir = self._active_run_dir
        if job is None or run_dir is None or not run_dir.is_dir():
            return
        self.model_result_requested.emit(job.output_root, str(run_dir))

    def _request_task_model(self, record: TrainingTaskRecord):
        run_dir = Path(record.run_dir)
        if not run_dir.is_dir():
            return
        self.model_result_requested.emit(record.output_root, str(run_dir))

    def _initialize_hardware_sampling(self):
        self._stats_process = None
        process = self._training_process
        if process is not None:
            try:
                import psutil
                self._stats_process = psutil.Process(int(process.processId()))
                self._stats_process.cpu_percent(None)
            except Exception:
                self._stats_process = None
        self._nvml = None
        self._nvml_handle = None
        job = self._active_training_job
        device = str(job.parameters.get('device', '0') if job else '0')
        if device.lower() in {'cpu', 'mps', 'none', ''}:
            return
        try:
            import pynvml
            index = int(device.split(',', 1)[0].strip())
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except Exception:
            self._nvml = None
            self._nvml_handle = None

    def _update_training_runtime(self):
        if self._training_started_at:
            elapsed = max(0, int(time.monotonic() - self._training_started_at))
            self.lbl_monitor_elapsed.setText(self._format_duration(elapsed))
        if self._stats_process is not None:
            try:
                processes = [self._stats_process]
                processes.extend(self._stats_process.children(recursive=True))
                cpu = sum(proc.cpu_percent(None) for proc in processes)
                memory = sum(proc.memory_info().rss for proc in processes)
                self.lbl_monitor_cpu.setText(f'{cpu:.1f}%')
                self.lbl_monitor_memory.setText(
                    f'{memory / (1024 ** 3):.2f} GB'
                )
            except Exception:
                self._stats_process = None
        if self._nvml is not None and self._nvml_handle is not None:
            try:
                utilization = self._nvml.nvmlDeviceGetUtilizationRates(
                    self._nvml_handle
                )
                memory = self._nvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                self.lbl_monitor_gpu.setText(f'{utilization.gpu}%')
                self.lbl_monitor_vram.setText(
                    f'{memory.used / (1024 ** 3):.1f} / '
                    f'{memory.total / (1024 ** 3):.1f} GB'
                )
            except Exception:
                self._nvml = None
                self._nvml_handle = None

    def _shutdown_hardware_sampling(self):
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
        self._nvml = None
        self._nvml_handle = None
        self._stats_process = None

    @staticmethod
    def _add_monitor_card(layout: QGridLayout, row: int, column: int,
                          caption: str, value: str) -> QLabel:
        card = QWidget()
        card.setObjectName('trainingMetricCard')
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(10, 7, 10, 8)
        card_layout.setSpacing(1)
        caption_label = QLabel(caption)
        caption_label.setObjectName('trainingMetricCaption')
        value_label = QLabel(value)
        value_label.setObjectName('trainingMetricValue')
        value_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        card_layout.addWidget(caption_label)
        card_layout.addWidget(value_label)
        layout.addWidget(card, row, column)
        layout.setColumnStretch(column, 1)
        return value_label

    @staticmethod
    def _format_metric_value(value) -> str:
        if isinstance(value, bool):
            return 'true' if value else 'false'
        if isinstance(value, (int, float)):
            return f'{value:.6g}'
        return str(value)

    def _training_failure_summary(self, message: str) -> str:
        job = self._active_training_job
        if job is not None:
            try:
                summary = inspect_training_batch(job.batch_root)
                if summary.invalid_label_count:
                    return summary.readiness_message()
            except (OSError, ValueError):
                pass
        first_line = str(message or '未知训练错误').strip().splitlines()[0]
        return first_line[:800]

    @staticmethod
    def _format_duration(seconds: int) -> str:
        hours, remainder = divmod(max(0, int(seconds)), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

    @staticmethod
    def _default_project_name(root: Path) -> str:
        name = root.name
        for suffix in ('_Datasets', '-Datasets', '_datasets', '-datasets'):
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break
        return name or 'Project'

    def _suggest_target_name(self):
        if self._dataset_root is None:
            return
        base = datetime.now().strftime('%Y-%m-%d')
        candidate = base
        index = 2
        training_root = self._dataset_root / 'training_data'
        while (training_root / candidate).exists():
            candidate = f'{base}-{index}'
            index += 1
        self.target_name_edit.setText(candidate)

    def _reload_training_templates(self, task_type: str,
                                   selected_path: str | Path | None = None):
        settings = QSettings('FilesProcessQT', 'ImageManager')
        extra_paths = self._settings_path_list(
            settings.value(f'trainingTemplatePaths/{task_type}')
        )
        paths = list_training_template_paths(task_type, extra_paths)
        active_path = str(
            selected_path
            or settings.value(f'trainingTemplateActive/{task_type}')
            or ''
        )
        self._loading_training_template = True
        self.training_template_combo.clear()
        active_index = 0
        for index, path in enumerate(paths):
            try:
                config = load_training_config(path)
            except ValueError:
                continue
            self.training_template_combo.addItem(config.name, str(path))
            if active_path and self._same_path(path, active_path):
                active_index = index
        if self.training_template_combo.count():
            active_index = min(active_index, self.training_template_combo.count() - 1)
            self.training_template_combo.setCurrentIndex(active_index)
        self._loading_training_template = False
        self._on_training_template_changed()

    def _on_training_template_changed(self, *_args):
        if self._loading_training_template:
            return
        path = self.training_template_combo.currentData()
        if not path:
            return
        try:
            config = load_training_config(path)
        except ValueError as exc:
            QMessageBox.warning(self, '训练模板无效', str(exc))
            return
        if config.task_type != self._task_type:
            QMessageBox.warning(
                self, '训练模板不匹配',
                f'当前任务是 {self._task_type}，模板任务是 {config.task_type}',
            )
            return
        QSettings('FilesProcessQT', 'ImageManager').setValue(
            f'trainingTemplateActive/{self._task_type}', str(path)
        )
        self._apply_training_config(config)

    def _apply_training_config(self, config: TrainingConfig):
        self._training_config = config
        params = config.parameters
        self.model_path_edit.setText(config.model)
        self.epochs_spin.setValue(int(params.get('epochs', 100)))
        self.imgsz_spin.setValue(int(params.get('imgsz', 640)))
        self.batch_spin.setValue(float(params.get('batch', 16)))
        self.patience_spin.setValue(int(params.get('patience', 100)))
        self.workers_spin.setValue(int(params.get('workers', 8)))
        self.device_combo.setCurrentText(str(params.get('device', '0')))
        self.optimizer_combo.setCurrentText(str(params.get('optimizer', 'auto')))
        self.lr0_spin.setValue(float(params.get('lr0', 0.01)))
        self.lrf_spin.setValue(float(params.get('lrf', 0.01)))
        self.momentum_spin.setValue(float(params.get('momentum', 0.937)))
        self.weight_decay_spin.setValue(
            float(params.get('weight_decay', 0.0005))
        )
        self.warmup_epochs_spin.setValue(
            float(params.get('warmup_epochs', 3.0))
        )
        cache_index = self.cache_combo.findData(params.get('cache', False))
        self.cache_combo.setCurrentIndex(max(0, cache_index))
        self.seed_train_spin.setValue(int(params.get('seed', 0)))
        self.close_mosaic_spin.setValue(int(params.get('close_mosaic', 10)))
        self.save_period_spin.setValue(int(params.get('save_period', -1)))
        self.amp_check.setChecked(bool(params.get('amp', True)))
        self.pretrained_check.setChecked(bool(params.get('pretrained', True)))
        self.deterministic_check.setChecked(
            bool(params.get('deterministic', True))
        )
        self.cos_lr_check.setChecked(bool(params.get('cos_lr', False)))
        self.rect_check.setChecked(bool(params.get('rect', False)))
        self.plots_check.setChecked(bool(params.get('plots', True)))
        self.lbl_training_parameter_count.setText(
            f'{len(params)} 个参数'
        )
        self.lbl_training_config_state.setText(f'当前模板: {config.name}')

    def current_training_config(self) -> TrainingConfig:
        params = dict(self._training_config.parameters)
        device = self.device_combo.currentData()
        if self.device_combo.isEditable():
            device = self.device_combo.currentText().strip()
        batch_value = self.batch_spin.value()
        if batch_value >= 1 and batch_value.is_integer():
            batch_value = int(batch_value)
        params.update({
            'epochs': self.epochs_spin.value(),
            'imgsz': self.imgsz_spin.value(),
            'batch': batch_value,
            'patience': self.patience_spin.value(),
            'workers': self.workers_spin.value(),
            'device': device,
            'optimizer': self.optimizer_combo.currentText(),
            'lr0': self.lr0_spin.value(),
            'lrf': self.lrf_spin.value(),
            'momentum': self.momentum_spin.value(),
            'weight_decay': self.weight_decay_spin.value(),
            'warmup_epochs': self.warmup_epochs_spin.value(),
            'cache': self.cache_combo.currentData(),
            'seed': self.seed_train_spin.value(),
            'close_mosaic': self.close_mosaic_spin.value(),
            'save_period': self.save_period_spin.value(),
            'amp': self.amp_check.isChecked(),
            'pretrained': self.pretrained_check.isChecked(),
            'deterministic': self.deterministic_check.isChecked(),
            'cos_lr': self.cos_lr_check.isChecked(),
            'rect': self.rect_check.isChecked(),
            'plots': self.plots_check.isChecked(),
        })
        data = training_config_to_dict(self._training_config)
        data['model'] = self.model_path_edit.text().strip()
        data['parameters'] = params
        return training_config_from_dict(data, self._training_config.path)

    def _open_training_advanced_config(self):
        try:
            current = self.current_training_config()
        except ValueError as exc:
            QMessageBox.warning(self, '训练参数无效', str(exc))
            return
        template_dir = custom_training_template_dir(self._task_type)
        dialog = TrainingTemplateDialog(current, template_dir, self)
        if not dialog.exec_():
            return
        saved_path = dialog.saved_path()
        saved_config = dialog.saved_config()
        if saved_path is None or saved_config is None:
            return
        settings = QSettings('FilesProcessQT', 'ImageManager')
        key = f'trainingTemplatePaths/{self._task_type}'
        paths = self._settings_path_list(settings.value(key))
        if not any(self._same_path(path, saved_path) for path in paths):
            paths.append(str(saved_path))
        settings.setValue(key, paths)
        settings.setValue(
            f'trainingTemplateActive/{self._task_type}', str(saved_path)
        )
        self._reload_training_templates(self._task_type, saved_path)
        self.status_message.emit(f'已切换训练模板: {saved_config.name}')

    @staticmethod
    def _double_parameter_spin(minimum: float, maximum: float,
                               value: float, decimals: int) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setObjectName('trainingParamSpin')
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setValue(value)
        spin.setSingleStep(10 ** -min(decimals, 3))
        return spin

    @staticmethod
    def _training_toggle(text: str) -> QCheckBox:
        checkbox = QCheckBox(text)
        checkbox.setObjectName('trainingToggle')
        return checkbox

    @staticmethod
    def _settings_path_list(value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        try:
            return [str(item) for item in value if str(item).strip()]
        except TypeError:
            return []

    @staticmethod
    def _same_path(first: str | Path, second: str | Path) -> bool:
        try:
            return Path(first).expanduser().resolve() == Path(second).expanduser().resolve()
        except OSError:
            return str(first) == str(second)

    @staticmethod
    def _configure_summary_header(tree: QTreeWidget):
        header = tree.header()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, header.ResizeToContents)
        header.setSectionResizeMode(1, header.ResizeToContents)
        header.setSectionResizeMode(2, header.Stretch)

    @staticmethod
    def _set_tree_status(tree: QTreeWidget, item: QTreeWidgetItem,
                         column: int, text: str, tone: str):
        # QTreeWidget still paints the item's text underneath an item widget.
        # Keep the value as item data and let the badge be the only visual text.
        item.setData(column, Qt.UserRole, text)
        item.setText(column, '')
        badge = QLabel(text)
        badge.setObjectName('trainingStatusBadge')
        badge.setProperty('tone', tone)
        badge.setAlignment(Qt.AlignCenter)
        badge.setAttribute(Qt.WA_TransparentForMouseEvents)
        badge.setToolTip(text)
        tree.setItemWidget(item, column, badge)

    @staticmethod
    def _project_root_for(path: str | Path) -> Path | None:
        candidate = Path(path).expanduser()
        try:
            candidate = candidate.resolve()
        except OSError:
            pass
        for root in (candidate, *candidate.parents):
            if (root / 'images').is_dir():
                return root
        return None
