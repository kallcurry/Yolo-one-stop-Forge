"""JSON -> YOLO TXT conversion and consistency validation tool."""

import json
from pathlib import Path

from PyQt5.QtCore import QSettings, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QHeaderView,
    QDialog,
)

from app.models.annotation_review import TASK_PRESETS
from app.models.annotation_converter import (
    ConvertReport,
    LabelConfig,
    ValidationReport,
    convert_json_batch,
    export_csv,
    parse_label_config,
    preview_conversion,
    validate_annotation_tree,
)
from app.views.tool_dialog import stored_dataset_path

CONFIG_PATH_KEY = 'convertLabelConfigPath'


def _stored_config_path() -> str:
    settings = QSettings()
    return str(settings.value(CONFIG_PATH_KEY, '') or '')


def _save_config_path(path: str):
    QSettings().setValue(CONFIG_PATH_KEY, path)


class _WorkerThread(QThread):
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            self.finished_ok.emit(self._fn())
        except Exception as exc:  # noqa: BLE001 - reported to the dialog
            self.failed.emit(str(exc))


class ConvertValidateDialog(QDialog):
    """Two-step workflow: convert JSON to YOLO TXT, then validate TXT."""

    def __init__(self, default_root: str = '', parent=None):
        super().__init__(parent)
        self.setWindowTitle('JSON ⇄ YOLO TXT 转换与校验')
        self.setMinimumSize(900, 700)
        self.resize(1080, 780)
        self._threads: list[QThread] = []
        QTimer.singleShot(200, self._auto_fix_annotation_dir)
        QTimer.singleShot(220, self._refresh_scope_list)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel('标注转换与一致性校验')
        title.setObjectName('duplicateTitle')
        header.addWidget(title)
        header.addStretch()
        hint = QLabel('类顺序与关键点顺序以 X-AnyLabeling 配置为准')
        hint.setObjectName('duplicateScope')
        header.addWidget(hint)
        layout.addLayout(header)

        inputs = QGroupBox('输入')
        input_layout = QVBoxLayout(inputs)
        root_row = QHBoxLayout()
        root_row.addWidget(QLabel('数据根目录'))
        self.edit_root = QLineEdit(default_root or stored_dataset_path())
        self.edit_root.setPlaceholderText('包含 images / annotations / labels 的项目根目录')
        root_row.addWidget(self.edit_root, 1)
        btn_root = QPushButton('选择')
        btn_root.clicked.connect(self._pick_root)
        root_row.addWidget(btn_root)
        root_row.addWidget(QLabel('标注配置'))
        self.edit_config = QLineEdit(_stored_config_path())
        self.edit_config.setPlaceholderText('X-AnyLabeling 标签配置 YAML（如 yolov8m_pose_boyuan.yaml）')
        root_row.addWidget(self.edit_config, 1)
        btn_config = QPushButton('选择')
        btn_config.clicked.connect(self._pick_config)
        root_row.addWidget(btn_config)
        input_layout.addLayout(root_row)

        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel('转换范围'))
        self.combo_scope = QComboBox()
        self.combo_scope.setObjectName('trainingCombo')
        self.combo_scope.addItem('全部数据目录', '')
        self.combo_scope.setToolTip('可选择 annotations 下的指定批次/子目录，只转换所选范围')
        scope_row.addWidget(self.combo_scope)
        btn_scope = QPushButton('刷新批次')
        btn_scope.setObjectName('fileOpBtn')
        btn_scope.clicked.connect(self._refresh_scope_list)
        scope_row.addWidget(btn_scope)
        self.lbl_scope_count = QLabel('')
        self.lbl_scope_count.setObjectName('duplicateScope')
        scope_row.addWidget(self.lbl_scope_count)
        scope_row.addStretch()
        input_layout.addLayout(scope_row)
        layout.addWidget(inputs)

        self.summary = QLabel('请选择数据根目录与标注配置。')
        self.summary.setObjectName('duplicateSummary')
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_convert_tab(), 'JSON → TXT 转换')
        self.tabs.addTab(self._build_validate_tab(), 'TXT ↔ JSON 校验')
        layout.addWidget(self.tabs, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ---- inputs ----

    def _auto_fix_annotation_dir(self):
        """When the annotation dir does not exist, adopt the actual variant."""
        root = Path(self.edit_root.text().strip()).expanduser()
        current = self.edit_annotation_dir.text().strip() or 'annotations'
        if not root.is_dir() or (root / current).is_dir():
            return
        candidates = sorted(
            child.name for child in root.iterdir()
            if child.is_dir() and child.name.lower().startswith('annotation')
        )
        if len(candidates) == 1:
            self.edit_annotation_dir.setText(candidates[0])
            self.summary.setText(f'已自动匹配标注目录：{candidates[0]}')
            return

        def _contains_json(name: str) -> bool:
            directory = root / name
            if not directory.is_dir():
                return False
            try:
                return any(directory.rglob('*.json'))
            except OSError:
                return False

        # 多候选：当前目录无 JSON 时，优先选择真正含 JSON 的标注集
        if not _contains_json(current):
            for candidate in candidates:
                if candidate != current and _contains_json(candidate):
                    self.edit_annotation_dir.setText(candidate)
                    self.summary.setText(
                        f'已自动匹配标注目录（含 JSON）：{candidate}'
                    )
                    return

    def _pick_root(self):
        path = QFileDialog.getExistingDirectory(self, '选择数据根目录', self.edit_root.text())
        if path:
            self.edit_root.setText(path)
            self._auto_fix_annotation_dir()

    def _pick_config(self):
        path, _filter = QFileDialog.getOpenFileName(
            self, '选择 X-AnyLabeling 标注配置',
            self.edit_config.text() or str(Path.home()),
            'YAML 配置 (*.yaml *.yml);;所有文件 (*)',
        )
        if path:
            self.edit_config.setText(path)
            _save_config_path(path)

    def _refresh_scope_list(self):
        self._auto_fix_annotation_dir()
        self.combo_scope.clear()
        self.combo_scope.addItem('全部数据目录', '')
        _root, ann_root, _lbl = self._resolved_dirs()
        if ann_root is None or not ann_root.is_dir():
            self.lbl_scope_count.setText('')
            return
        found = 0
        for child in sorted(ann_root.iterdir()):
            if child.is_dir():
                self.combo_scope.addItem(child.name, child.name)
                found += 1
        self.lbl_scope_count.setText(f'（{found} 个批次）')

    def _scope_value(self) -> str | None:
        value = str(self.combo_scope.currentData() or '')
        return value or None

    def _load_config(self) -> LabelConfig | None:
        config_path = self.edit_config.text().strip()
        if not config_path:
            QMessageBox.warning(self, '缺少配置', '请先选择 X-AnyLabeling 标注配置 YAML。')
            return None
        try:
            config = parse_label_config(config_path)
        except ValueError as exc:
            QMessageBox.warning(self, '配置无效', str(exc))
            return None
        _save_config_path(config_path)
        # 按任务更新默认目录（多任务数据并存：labels / labels-det / ...）
        preset = TASK_PRESETS.get(config.task_type, {})
        if preset.get('annotation_dir'):
            self.edit_annotation_dir.setText(str(preset['annotation_dir']))
        if preset.get('label_dir'):
            self.edit_label_dir.setText(str(preset['label_dir']))
        return config

    def _resolved_dirs(self):
        root = Path(self.edit_root.text().strip()).expanduser()
        if not root.is_dir():
            QMessageBox.warning(self, '目录无效', f'数据根目录不存在: {root}')
            return None, None, None
        annotation_dir = self.edit_annotation_dir.text().strip() or 'annotations'
        label_dir = self.edit_label_dir.text().strip() or 'labels'
        return root, root / annotation_dir, root / label_dir

    # ---- convert tab ----

    def _build_convert_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 8, 0, 0)

        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel('标注目录'))
        self.edit_annotation_dir = QLineEdit('annotations')
        dir_row.addWidget(self.edit_annotation_dir, 1)
        dir_row.addWidget(QLabel('标签目录'))
        self.edit_label_dir = QLineEdit('labels')
        dir_row.addWidget(self.edit_label_dir, 1)
        layout.addLayout(dir_row)

        policy_row = QHBoxLayout()
        policy_row.addWidget(QLabel('已有 TXT'))
        self.radio_policy = QButtonGroup(self)
        self.rb_skip = QRadioButton('跳过（保留历史）')
        self.rb_backup = QRadioButton('备份后覆盖')
        self.rb_overwrite = QRadioButton('直接覆盖')
        self.rb_skip.setChecked(True)
        for radio in (self.rb_skip, self.rb_backup, self.rb_overwrite):
            self.radio_policy.addButton(radio)
            policy_row.addWidget(radio)
        self.check_skip_empty = QCheckBox('跳过空标注（空 JSON 不生成 TXT）')
        self.check_skip_empty.setObjectName('trainingCheck')
        self.check_skip_empty.setToolTip(
            '与 X-AnyLabeling 的 "Skip empty labels" 一致：空标注不生成 TXT；'
            '若目标已存在且为 0 字节空文件则一并清理，非空文件不受影响。'
        )
        policy_row.addWidget(self.check_skip_empty)
        policy_row.addStretch()
        layout.addLayout(policy_row)

        btn_row = QHBoxLayout()
        btn_preview = QPushButton('预览转换内容')
        btn_preview.clicked.connect(self._preview_convert)
        btn_row.addWidget(btn_preview)
        btn_run = QPushButton('开始转换')
        btn_run.setObjectName('primaryBtn')
        btn_run.clicked.connect(self._run_convert)
        btn_row.addWidget(btn_run)
        self.btn_export_convert = QPushButton('导出转换报告 CSV')
        self.btn_export_convert.setEnabled(False)
        self.btn_export_convert.clicked.connect(self._export_convert)
        btn_row.addWidget(self.btn_export_convert)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.convert_preview = QTextEdit()
        self.convert_preview.setReadOnly(True)
        self.convert_preview.setPlaceholderText('预览区：显示每个 JSON 将生成的 YOLO TXT 内容（不会写盘）')
        layout.addWidget(self.convert_preview, 1)

        self.convert_table = QTableWidget(0, 5)
        self.convert_table.setHorizontalHeaderLabels(
            ['JSON 文件', '输出 TXT', '状态', '行数', '说明']
        )
        self.convert_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.convert_table.verticalHeader().setVisible(False)
        header = self.convert_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        self.convert_table.setMaximumHeight(220)
        layout.addWidget(self.convert_table)
        return widget

    def _policy(self) -> str:
        if self.rb_backup.isChecked():
            return 'backup'
        if self.rb_overwrite.isChecked():
            return 'overwrite'
        return 'skip'

    def _start(self, fn, on_ok):
        thread = _WorkerThread(fn, self)
        thread.finished_ok.connect(on_ok)
        thread.failed.connect(self._on_worker_failed)
        thread.finished.connect(lambda: self._cleanup_thread(thread))
        self._threads.append(thread)
        self.setEnabled(False)
        thread.start()

    def _cleanup_thread(self, thread):
        if thread in self._threads:
            self._threads.remove(thread)
        self.setEnabled(True)

    def _on_worker_failed(self, message: str):
        self.summary.setText(f'处理失败：{message}')
        QMessageBox.warning(self, '处理失败', message)

    def _preview_convert(self):
        config = self._load_config()
        if config is None:
            return
        root, ann_root, _lbl_root = self._resolved_dirs()
        if root is None or not ann_root.is_dir():
            self.summary.setText('标注目录不存在，请检查数据根目录与标注目录名。')
            return
        scope = self._scope_value()
        walk_root = (ann_root / scope) if scope else ann_root
        json_files = sorted(walk_root.rglob('*.json'))
        if not json_files:
            self.summary.setText(
                '所选范围内没有 JSON 文件（请检查【标注目录】是否正确，'
                f'当前为：{self.edit_annotation_dir.text().strip() or "annotations"}）。'
            )
            return
        chunks = []
        for json_file in json_files[:60]:
            chunks.append(preview_conversion(json_file, config, max_lines=3))
        if len(json_files) > 60:
            chunks.append(f'... 其余 {len(json_files) - 60} 个文件未预览')
        self.convert_preview.setPlainText('\n' + ('-' * 72) + '\n' + ('-' * 72).join(chunks))
        self.summary.setText(
            f'预览完成：{len(json_files)} 个 JSON'
            f'{"（范围：" + scope + "）" if scope else ""}（仅显示内容，未写盘）。'
        )

    def _run_convert(self):
        config = self._load_config()
        if config is None:
            return
        root, ann_root, lbl_root = self._resolved_dirs()
        if root is None:
            return
        policy = self._policy()
        self.summary.setText(f'正在转换（策略={policy}）...请稍候')
        self.btn_export_convert.setEnabled(False)
        self._start(
            lambda: convert_json_batch(
                ann_root, lbl_root, config,
                exists_policy=policy, scope=self._scope_value(),
                skip_empty=self.check_skip_empty.isChecked(),
            ),
            self._on_convert_done,
        )

    def _on_convert_done(self, report: ConvertReport):
        self.convert_table.setRowCount(len(report.items))
        for row, item in enumerate(report.items):
            values = (
                item.source.name,
                str(item.target),
                item.status,
                str(item.lines),
                item.message,
            )
            for column, value in enumerate(values):
                self.convert_table.setItem(row, column, QTableWidgetItem(value))
        self._report = report
        self.btn_export_convert.setEnabled(bool(report.items))
        self.summary.setText(
            f'转换完成：写入 {report.written_count}（含空标注），跳过 {report.skipped_count}，'
            f'失败 {report.failed_count}，共 {len(report.items)} 个 JSON。'
        )

    def _export_convert(self):
        report = getattr(self, '_report', None)
        if report is None:
            return
        path, _filter = QFileDialog.getSaveFileName(
            self, '导出转换报告', 'convert_report.csv', 'CSV (*.csv)'
        )
        if path:
            export_csv(report.csv_lines(), path)
            self.summary.setText(f'转换报告已导出：{path}')

    # ---- validate tab ----

    def _build_validate_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 8, 0, 0)

        option_row = QHBoxLayout()
        option_row.addWidget(QLabel('数值容差'))
        self.spin_tolerance = QDoubleSpinBox()
        self.spin_tolerance.setDecimals(6)
        self.spin_tolerance.setRange(0.0, 1.0)
        self.spin_tolerance.setSingleStep(0.0001)
        self.spin_tolerance.setValue(0.0001)
        option_row.addWidget(self.spin_tolerance)
        option_row.addStretch()
        layout.addLayout(option_row)

        btn_row = QHBoxLayout()
        btn_run = QPushButton('开始校验')
        btn_run.setObjectName('primaryBtn')
        btn_run.clicked.connect(self._run_validate)
        btn_row.addWidget(btn_run)
        self.btn_export_validate = QPushButton('导出校验报告 CSV')
        self.btn_export_validate.setEnabled(False)
        self.btn_export_validate.clicked.connect(self._export_validate)
        btn_row.addWidget(self.btn_export_validate)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.validate_table = QTableWidget(0, 4)
        self.validate_table.setHorizontalHeaderLabels(
            ['JSON 文件', '对应 TXT', '状态', '问题 / 差异']
        )
        self.validate_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.validate_table.setAlternatingRowColors(True)
        self.validate_table.verticalHeader().setVisible(False)
        header = self.validate_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        layout.addWidget(self.validate_table, 1)
        return widget

    def _run_validate(self):
        config = self._load_config()
        if config is None:
            return
        root, ann_root, lbl_root = self._resolved_dirs()
        if root is None:
            return
        tolerance = self.spin_tolerance.value()
        self.summary.setText(f'正在校验（容差 {tolerance}）...请稍候')
        self.btn_export_validate.setEnabled(False)
        self._start(
            lambda: validate_annotation_tree(
                ann_root, lbl_root, config, tolerance,
                scope=self._scope_value(),
            ),
            self._on_validate_done,
        )

    def _on_validate_done(self, report: ValidationReport):
        self.validate_table.setRowCount(len(report.items))
        for row, item in enumerate(report.items):
            detail = '；'.join(item.issues) or '；'.join(item.diffs)
            values = (
                item.json_path.name,
                str(item.txt_path) if item.txt_path else '(缺失)',
                item.status,
                detail,
            )
            for column, value in enumerate(values):
                self.validate_table.setItem(row, column, QTableWidgetItem(value))
        self._report = report
        self.btn_export_validate.setEnabled(bool(report.items))
        self.summary.setText(
            f'校验完成：通过 {report.ok_count}，异常 {report.bad_count}'
            f'（缺 TXT {report.missing_txt_count}），'
            f'多余 TXT {len(report.extra_txts)}。'
        )

    def _export_validate(self):
        report = getattr(self, '_report', None)
        if report is None:
            return
        path, _filter = QFileDialog.getSaveFileName(
            self, '导出校验报告', 'validation_report.csv', 'CSV (*.csv)'
        )
        if path:
            export_csv(report.csv_lines(), path)
            self.summary.setText(f'校验报告已导出：{path}')

    def closeEvent(self, event):
        for thread in list(self._threads):
            thread.terminate()
            thread.wait(1000)
        event.accept()


def create_dialog(parent=None):
    return ConvertValidateDialog(stored_dataset_path(), parent)
