"""Duplicate and overlap checking tools.

The original training/test overlap checker remains available below.  The
interactive raw-data audit in this module is a separate workflow so the two
different meanings of "duplicate" do not get mixed together.
"""

import time
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

from app.models.dataset_duplicates import (
    DuplicateGroup,
    DuplicateScanResult,
    delete_duplicate_files,
    delete_orphan_files,
    move_to_backup,
    resolve_raw_dataset_root,
    scan_raw_duplicates,
)
from app.views.tool_dialog import ToolDialog, stored_dataset_path

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def _get_stems(directory: Path) -> set[str]:
    if not directory.is_dir():
        return set()
    return {f.stem for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS}


def run_check(train_path: str, test_path: str, show_all: bool):
    train_dir = Path(train_path)
    test_dir = Path(test_path)

    # Find image directories
    train_img = train_dir / "images"
    test_img = test_dir / "images"

    if not train_img.is_dir():
        # Try train_data sub-structure
        for sub in ["train_data/images/train", "train_data/images"]:
            candidate = train_dir / sub
            if candidate.is_dir():
                train_img = candidate
                break

    if not test_img.is_dir():
        print(f"❌ 测试集 images 目录不存在: {test_img}")
        return

    print(f"📊 训练集: {train_img}")
    print(f"📊 测试集: {test_img}")
    print()

    train_stems = _get_stems(train_img)
    test_stems = _get_stems(test_img)

    print(f"   训练集图片: {len(train_stems)}")
    print(f"   测试集图片: {len(test_stems)}")

    overlap = train_stems & test_stems
    only_train = train_stems - test_stems
    only_test = test_stems - train_stems

    print(f"   重复图片:   {len(overlap)}")
    print(f"   仅训练集:   {len(only_train)}")
    print(f"   仅测试集:   {len(only_test)}")

    total_unique = len(train_stems | test_stems)
    print(f"   唯一图片:   {total_unique}")
    print()

    if overlap:
        print(f"{'='*60}")
        print(f"❌ 发现 {len(overlap)} 张重复图片！")
        print(f"{'='*60}")
        show_n = min(len(overlap), 200) if show_all else min(len(overlap), 30)
        for i, stem in enumerate(sorted(overlap)[:show_n], 1):
            print(f"   {i:4d}. {stem}")
        if len(overlap) > show_n:
            print(f"   ... 还有 {len(overlap) - show_n} 张")
        print()
        print(f"⚠️ 建议: 从训练集中删除这些重复图片，或从测试集中移除。")
    else:
        print("✅ 训练集和测试集零交叉，数据隔离良好！")

    # Also check annotations overlap
    train_ann = train_dir / "annotations"
    test_ann = test_dir / "annotations"
    if train_ann.is_dir() and test_ann.is_dir():
        train_ann_stems = _get_stems(train_ann)
        test_ann_stems = _get_stems(test_ann)
        ann_overlap = train_ann_stems & test_ann_stems
        if ann_overlap:
            print(f"\n📝 标注文件也有 {len(ann_overlap)} 个重复")
        else:
            print(f"\n📝 标注文件: 无重复 ✅")

    print("\n✅ 检查完成!")


def create_dialog(parent=None):
    dlg = ToolDialog('训练/测试集重复检查', parent)

    dlg.edit_train = dlg._add_dir_picker('训练集目录:',
        stored_dataset_path('training_data'))
    dlg.edit_test = dlg._add_dir_picker('测试集目录:',
        stored_dataset_path('test_data'))

    row = __import__('PyQt5.QtWidgets', fromlist=['QHBoxLayout']).QHBoxLayout()
    cb_all = QCheckBox('显示全部重复图片（否则只显示前30张）')
    row.addWidget(cb_all)
    dlg.param_widget.addLayout(row)

    dlg.set_runner(lambda: run_check(
        dlg.edit_train.text(), dlg.edit_test.text(), cb_all.isChecked()
    ))
    return dlg


class _RawDuplicateScanThread(QThread):
    """Hash raw files without blocking the main Qt event loop."""

    completed = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)

    def __init__(self, root: str, parent=None):
        super().__init__(parent)
        self._root = root

    def run(self):
        try:
            self.completed.emit(
                scan_raw_duplicates(self._root, progress=self.progress.emit)
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class RawDuplicateDialog(QDialog):
    """Review and remove exact duplicate files from the raw data trees."""

    _KIND_LABELS = {
        'image': '图片',
        'annotation': 'JSON 标注',
        'label': 'TXT 标签',
    }

    def __init__(self, default_root: str = '', parent=None):
        super().__init__(parent)
        self.setWindowTitle('原始数据重复审查')
        self.setMinimumSize(900, 650)
        self.resize(1120, 780)

        self._thread = None
        self._result: DuplicateScanResult | None = None
        self._visible_groups: list[DuplicateGroup] = []
        self._visible_orphans: list = []
        self._visible_conflicts: list = []
        self._backup_dir: Path | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        title_row = QHBoxLayout()
        title = QLabel('全数据目录 · 原始数据审查')
        title.setObjectName('duplicateTitle')
        title_row.addWidget(title)
        title_row.addStretch()
        scope = QLabel('仅扫描 images / annotations / labels')
        scope.setObjectName('duplicateScope')
        title_row.addWidget(scope)
        layout.addLayout(title_row)

        root_row = QHBoxLayout()
        root_row.addWidget(QLabel('数据根目录'))
        self.edit_root = QLineEdit(default_root)
        self.edit_root.setPlaceholderText('选择包含 images、annotations、labels 的数据根目录')
        root_row.addWidget(self.edit_root, 1)
        self.btn_browse = QPushButton('选择目录')
        self.btn_browse.clicked.connect(self._browse_root)
        root_row.addWidget(self.btn_browse)
        self.btn_scan = QPushButton('开始扫描')
        self.btn_scan.setObjectName('primaryBtn')
        self.btn_scan.clicked.connect(self._start_scan)
        root_row.addWidget(self.btn_scan)
        layout.addLayout(root_row)

        metrics = QHBoxLayout()
        self.metric_images = self._metric(metrics, '图片文件')
        self.metric_annotations = self._metric(metrics, 'JSON 标注')
        self.metric_labels = self._metric(metrics, 'TXT 标签')
        self.metric_groups = self._metric(metrics, '重复组')
        self.metric_files = self._metric(metrics, '可删除副本')
        self.metric_bytes = self._metric(metrics, '可释放空间')
        self.metric_orphans = self._metric(metrics, '孤儿文件')
        self.metric_conflicts = self._metric(metrics, '同名多版本')
        layout.addLayout(metrics)

        self.summary = QLabel('尚未扫描')
        self.summary.setObjectName('duplicateSummary')
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_duplicate_tab(), '重复数据')
        self.tabs.addTab(self._build_orphan_tab(), '孤儿标注/标签')
        self.tabs.addTab(self._build_conflict_tab(), '同名多版本')
        layout.addWidget(self.tabs, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        if default_root:
            self._start_scan()

    def _build_duplicate_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 8, 0, 0)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel('查看'))
        self.filter_combo = QComboBox()
        self.filter_combo.addItem('全部类型', '')
        self.filter_combo.addItem('图片重复', 'image')
        self.filter_combo.addItem('JSON 标注重复', 'annotation')
        self.filter_combo.addItem('TXT 标签重复', 'label')
        self.filter_combo.currentIndexChanged.connect(self._populate_table)
        filter_row.addWidget(self.filter_combo)
        filter_row.addStretch()
        self.status = QLabel('等待扫描')
        self.status.setObjectName('duplicateStatus')
        filter_row.addWidget(self.status)
        layout.addLayout(filter_row)

        body = QGroupBox('重复组（每组保留一个副本）')
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(8, 12, 8, 8)
        self.group_table = QTableWidget(0, 5)
        self.group_table.setHorizontalHeaderLabels(
            ['类型', '保留副本', '文件数', '可删除副本', '可释放空间']
        )
        self.group_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.group_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.group_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.group_table.setAlternatingRowColors(True)
        self.group_table.verticalHeader().setVisible(False)
        header = self.group_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.group_table.itemSelectionChanged.connect(self._show_selected_group)
        body_layout.addWidget(self.group_table, 1)

        detail_title = QLabel('重复组详情')
        detail_title.setObjectName('duplicateDetailTitle')
        body_layout.addWidget(detail_title)
        self.detail_list = QListWidget()
        self.detail_list.setMinimumHeight(105)
        body_layout.addWidget(self.detail_list)
        layout.addWidget(body, 1)

        linked_row = QHBoxLayout()
        self.check_linked = QCheckBox(
            '联动删除（删除图片副本时，同步删除已确认重复的对应 JSON/TXT 副本）'
        )
        self.check_linked.setObjectName('duplicateLinked')
        self.check_linked.setToolTip(
            '仅当同名 JSON 标注 / TXT 标签被扫描确认属于重复组且不是保留副本时，'
            '才会随图片副本一并删除；保留副本与未确认的文件不会删除。'
        )
        linked_row.addWidget(self.check_linked)
        linked_row.addWidget(QLabel('（保留副本与未确认文件将跳过并提示）'))
        linked_row.addStretch()
        layout.addLayout(linked_row)

        action_row = QHBoxLayout()
        self.btn_delete_group = QPushButton('删除当前重复副本')
        self.btn_delete_group.setObjectName('dangerBtn')
        self.btn_delete_group.setEnabled(False)
        self.btn_delete_group.clicked.connect(self._delete_selected_group)
        action_row.addWidget(self.btn_delete_group)
        self.btn_delete_all = QPushButton('删除全部重复副本')
        self.btn_delete_all.setObjectName('dangerBtn')
        self.btn_delete_all.setEnabled(False)
        self.btn_delete_all.clicked.connect(self._delete_all)
        action_row.addWidget(self.btn_delete_all)
        action_row.addWidget(QLabel('删除动作只处理每组的重复成员，不会删除保留副本。'))
        action_row.addStretch()
        layout.addLayout(action_row)
        return widget

    def _build_orphan_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 8, 0, 0)
        hint = QLabel(
            '标注 / 标签在同批次目录中找不到对应图片（图片可能在其他批次，'
            '或此副本已无用）。可移出到备份目录，或删除到回收站。'
        )
        hint.setWordWrap(True)
        hint.setObjectName('duplicateHint')
        layout.addWidget(hint)

        self.orphan_table = QTableWidget(0, 3)
        self.orphan_table.setHorizontalHeaderLabels(['类型', '批次', '文件'])
        self.orphan_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.orphan_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.orphan_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.orphan_table.setAlternatingRowColors(True)
        self.orphan_table.verticalHeader().setVisible(False)
        header = self.orphan_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        self.orphan_table.itemSelectionChanged.connect(self._update_orphan_buttons)
        layout.addWidget(self.orphan_table, 1)

        row = QHBoxLayout()
        self.btn_orphan_backup = QPushButton('移出所选到备份目录')
        self.btn_orphan_backup.setEnabled(False)
        self.btn_orphan_backup.clicked.connect(self._move_selected_orphans)
        row.addWidget(self.btn_orphan_backup)
        self.btn_orphan_delete = QPushButton('删除所选到回收站')
        self.btn_orphan_delete.setObjectName('dangerBtn')
        self.btn_orphan_delete.setEnabled(False)
        self.btn_orphan_delete.clicked.connect(self._delete_selected_orphans)
        row.addWidget(self.btn_orphan_delete)
        row.addWidget(QLabel('备份目录保留原相对路径，可随时找回。'))
        row.addStretch()
        layout.addLayout(row)
        return widget

    def _build_conflict_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 8, 0, 0)
        hint = QLabel(
            '同一文件名在多个目录各有一份，且内容不同（例如审查副本与原批次版本）。'
            '请在下方选择要保留的版本，其余可移出到备份或删除。'
        )
        hint.setWordWrap(True)
        hint.setObjectName('duplicateHint')
        layout.addWidget(hint)

        self.conflict_table = QTableWidget(0, 4)
        self.conflict_table.setHorizontalHeaderLabels(
            ['类型', '文件名', '版本数', '所在目录']
        )
        self.conflict_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.conflict_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.conflict_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.conflict_table.setAlternatingRowColors(True)
        self.conflict_table.verticalHeader().setVisible(False)
        header = self.conflict_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.conflict_table.itemSelectionChanged.connect(self._show_selected_conflict)
        layout.addWidget(self.conflict_table, 1)

        detail_title = QLabel('版本详情（选择要保留的版本）')
        detail_title.setObjectName('duplicateDetailTitle')
        layout.addWidget(detail_title)
        self.conflict_detail = QListWidget()
        self.conflict_detail.setMinimumHeight(110)
        self.conflict_detail.itemSelectionChanged.connect(self._update_conflict_buttons)
        layout.addWidget(self.conflict_detail)

        row = QHBoxLayout()
        self.btn_conflict_backup = QPushButton('保留所选版本（其余移出到备份）')
        self.btn_conflict_backup.setEnabled(False)
        self.btn_conflict_backup.clicked.connect(lambda: self._resolve_conflict('backup'))
        row.addWidget(self.btn_conflict_backup)
        self.btn_conflict_delete = QPushButton('保留所选版本（其余删除到回收站）')
        self.btn_conflict_delete.setObjectName('dangerBtn')
        self.btn_conflict_delete.setEnabled(False)
        self.btn_conflict_delete.clicked.connect(lambda: self._resolve_conflict('trash'))
        row.addWidget(self.btn_conflict_delete)
        row.addStretch()
        layout.addLayout(row)
        return widget

    @staticmethod
    def _metric(parent_layout: QHBoxLayout, caption: str) -> QLabel:
        box = QWidget()
        box.setObjectName('duplicateMetric')
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(10, 7, 10, 7)
        box_layout.setSpacing(1)
        label = QLabel(caption)
        label.setObjectName('duplicateMetricCaption')
        value = QLabel('-')
        value.setObjectName('duplicateMetricValue')
        box_layout.addWidget(label)
        box_layout.addWidget(value)
        parent_layout.addWidget(box, 1)
        return value

    def _browse_root(self):
        path = QFileDialog.getExistingDirectory(self, '选择数据根目录', self.edit_root.text())
        if path:
            self.edit_root.setText(path)

    def _start_scan(self):
        if self._thread is not None and self._thread.isRunning():
            return
        selected_root = self.edit_root.text().strip()
        if not selected_root:
            QMessageBox.warning(self, '无法扫描', '请先选择数据根目录。')
            return
        try:
            root = str(resolve_raw_dataset_root(selected_root))
        except ValueError as exc:
            self.status.setText(f'目录无效：{exc}')
            QMessageBox.warning(self, '无法扫描', str(exc))
            return
        if root != selected_root:
            self.edit_root.setText(root)
            self.summary.setText(
                f'已从所选目录识别项目根目录：{root}，将扫描整个原始数据目录。'
            )
        self._result = None
        self._visible_groups = []
        self._visible_orphans = []
        self._visible_conflicts = []
        self.group_table.setRowCount(0)
        self.detail_list.clear()
        self.orphan_table.setRowCount(0)
        self.conflict_table.setRowCount(0)
        self.conflict_detail.clear()
        self._set_action_enabled(False)
        self._update_orphan_buttons()
        self._update_conflict_buttons()
        self.btn_scan.setEnabled(False)
        self.status.setText('正在扫描并计算文件指纹，请稍候...')
        self._thread = _RawDuplicateScanThread(root, self)
        self._thread.progress.connect(self._on_scan_progress)
        self._thread.completed.connect(self._on_scan_completed)
        self._thread.failed.connect(self._on_scan_failed)
        self._thread.finished.connect(lambda: self.btn_scan.setEnabled(True))
        self._thread.start()

    def _on_scan_progress(self, current: int, total: int, message: str):
        if total > 0:
            self.status.setText(f'{message} · {current}/{total}')
        else:
            self.status.setText(message)

    def _on_scan_completed(self, result: DuplicateScanResult):
        self._result = result
        counts = result.scanned_counts
        self.metric_images.setText(str(counts.get('image', 0)))
        self.metric_annotations.setText(str(counts.get('annotation', 0)))
        self.metric_labels.setText(str(counts.get('label', 0)))
        self.metric_groups.setText(str(result.duplicate_group_count))
        self.metric_files.setText(str(result.duplicate_file_count))
        self.metric_bytes.setText(_format_bytes(result.reclaimable_bytes))
        self.metric_orphans.setText(str(result.orphan_count))
        self.metric_conflicts.setText(str(result.name_conflict_count))
        self.summary.setText(
            f'已扫描 {result.root} · 发现 {result.duplicate_group_count} 个重复组，'
            f'共 {result.duplicate_file_count} 个可删除副本；'
            f'孤儿文件 {result.orphan_count} 个，同名多版本 {result.name_conflict_count} 组。'
        )
        if result.errors:
            self.summary.setText(
                self.summary.text() + f' 读取失败 {len(result.errors)} 个文件。'
            )
        self.status.setText(
            '扫描完成：未发现重复文件。' if not result.duplicate_group_count
            else '扫描完成：请选择重复组查看详情。'
        )
        self._populate_table()
        self._populate_orphan_table()
        self._populate_conflict_table()

    def _on_scan_failed(self, message: str):
        self._result = None
        self.status.setText(f'扫描失败：{message}')
        QMessageBox.warning(self, '扫描失败', message)

    def _populate_table(self):
        self.group_table.setRowCount(0)
        self.detail_list.clear()
        if self._result is None:
            self._set_action_enabled(False)
            return
        selected_kind = self.filter_combo.currentData()
        self._visible_groups = [
            group for group in self._result.all_groups
            if not selected_kind or group.kind == selected_kind
        ]
        self.group_table.setRowCount(len(self._visible_groups))
        for row, group in enumerate(self._visible_groups):
            values = (
                self._KIND_LABELS.get(group.kind, group.kind),
                self._relative_path(group.keeper),
                str(len(group.files)),
                str(len(group.duplicates)),
                _format_bytes(group.reclaimable_bytes),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, row)
                self.group_table.setItem(row, column, item)
            kind_color = {
                'image': QColor('#62E8FF'),
                'annotation': QColor('#D8B4FE'),
                'label': QColor('#FFD07A'),
            }.get(group.kind)
            if kind_color:
                self.group_table.item(row, 0).setForeground(QBrush(kind_color))
        self._set_action_enabled(bool(self._visible_groups))

    def _show_selected_group(self):
        rows = self.group_table.selectionModel().selectedRows()
        self.detail_list.clear()
        if not rows:
            self.btn_delete_group.setEnabled(False)
            return
        group = self._visible_groups[rows[0].row()]
        self.detail_list.addItem(f'保留：{self._relative_path(group.keeper)}')
        for path in group.duplicates:
            self.detail_list.addItem(f'删除：{self._relative_path(path)}')
        self.btn_delete_group.setEnabled(True)

    # ---- orphan / name-conflict workflows ----

    def _populate_orphan_table(self):
        self.orphan_table.setRowCount(0)
        self._visible_orphans = list(self._result.orphans) if self._result else []
        self.orphan_table.setRowCount(len(self._visible_orphans))
        for row, record in enumerate(self._visible_orphans):
            values = (
                self._KIND_LABELS.get(record.kind, record.kind),
                record.batch_dir or '(根目录)',
                self._relative_path(record.path),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, row)
                self.orphan_table.setItem(row, column, item)
        self._update_orphan_buttons()

    def _update_orphan_buttons(self):
        has_rows = bool(self.orphan_table.selectionModel().selectedRows())
        self.btn_orphan_backup.setEnabled(has_rows and bool(self._visible_orphans))
        self.btn_orphan_delete.setEnabled(has_rows and bool(self._visible_orphans))

    def _selected_orphan_paths(self) -> list:
        if self._result is None:
            return []
        rows = self.orphan_table.selectionModel().selectedRows()
        return [self._visible_orphans[row.row()].path for row in rows]

    def _move_selected_orphans(self):
        if self._result is None:
            return
        paths = self._selected_orphan_paths()
        if not paths:
            return
        backup = self._ensure_backup_dir()
        if backup is None:
            return
        answer = QMessageBox.warning(
            self,
            '确认移出孤儿文件',
            f'将移出 {len(paths)} 个孤儿文件到备份目录：\n{backup}\n'
            '（保留原相对路径，可随时找回）',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        result = move_to_backup(paths, self._result.root, backup)
        self._report_action(
            f'已移出 {len(result.moved)} 个孤儿文件到备份目录。',
            result.errors,
        )
        self._start_scan()

    def _delete_selected_orphans(self):
        if self._result is None:
            return
        paths = self._selected_orphan_paths()
        if not paths:
            return
        answer = QMessageBox.warning(
            self,
            '确认删除孤儿文件',
            f'将删除 {len(paths)} 个孤儿文件（进入系统回收站，'
            '若系统不支持则直接删除）。\n请确认这些文件已无用。',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        result = delete_orphan_files(paths, self._result.root)
        self._report_action(
            f'已删除 {len(result.deleted)} 个孤儿文件。',
            result.errors,
        )
        self._start_scan()

    def _populate_conflict_table(self):
        self.conflict_table.setRowCount(0)
        self.conflict_detail.clear()
        self._visible_conflicts = (
            list(self._result.name_conflicts) if self._result else []
        )
        self.conflict_table.setRowCount(len(self._visible_conflicts))
        for row, conflict in enumerate(self._visible_conflicts):
            directories = sorted({
                str(member.path.parent)
                for member in conflict.members
            })
            values = (
                self._KIND_LABELS.get(conflict.kind, conflict.kind),
                conflict.stem,
                str(len(conflict.members)),
                '、'.join(self._relative_path(Path(path)) for path in directories),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, row)
                self.conflict_table.setItem(row, column, item)
        self._update_conflict_buttons()

    def _show_selected_conflict(self):
        rows = self.conflict_table.selectionModel().selectedRows()
        self.conflict_detail.clear()
        self._update_conflict_buttons()
        if not rows or self._result is None:
            return
        conflict = self._visible_conflicts[rows[0].row()]
        for index, member in enumerate(conflict.members):
            item = QListWidgetItem(
                f'{index + 1}. [{member.digest[:12]}] '
                f'{self._relative_path(member.path)}（{_format_bytes(member.size)}）'
            )
            item.setData(Qt.UserRole, index)
            self.conflict_detail.addItem(item)
        self._update_conflict_buttons()

    def _update_conflict_buttons(self):
        has_member = bool(self.conflict_detail.selectedItems())
        self.btn_conflict_backup.setEnabled(has_member)
        self.btn_conflict_delete.setEnabled(has_member)

    def _resolve_conflict(self, mode: str):
        if self._result is None:
            return
        rows = self.conflict_table.selectionModel().selectedRows()
        if not rows:
            return
        conflict = self._visible_conflicts[rows[0].row()]
        selected = self.conflict_detail.selectedItems()
        if not selected:
            return
        keeper = conflict.members[selected[0].data(Qt.UserRole)]
        others = [m.path for m in conflict.members if m is not keeper]
        if not others:
            return
        if mode == 'backup':
            backup = self._ensure_backup_dir()
            if backup is None:
                return
            answer = QMessageBox.warning(
                self,
                '确认保留版本',
                f'保留：{self._relative_path(keeper.path)}\n\n'
                f'其余 {len(others)} 个版本将移出到备份目录：\n{backup}\n'
                '（保留原相对路径，可随时找回）',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
            result = move_to_backup(others, self._result.root, backup)
            self._report_action(
                f'已保留所选版本，移出 {len(result.moved)} 个版本到备份目录。',
                result.errors,
            )
        else:
            answer = QMessageBox.warning(
                self,
                '确认保留版本',
                f'保留：{self._relative_path(keeper.path)}\n\n'
                f'其余 {len(others)} 个版本将删除（进入系统回收站，'
                '若系统不支持则直接删除）。',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
            result = delete_orphan_files(others, self._result.root)
            self._report_action(
                f'已保留所选版本，删除 {len(result.deleted)} 个版本。',
                result.errors,
            )
        self._start_scan()

    def _ensure_backup_dir(self):
        if self._result is None:
            return None
        if self._backup_dir is None:
            self._backup_dir = (
                Path(self._result.root)
                / f'duplicate_review_backup_{time.strftime("%Y%m%d-%H%M%S")}'
            )
        return self._backup_dir

    def _report_action(self, message: str, errors: tuple):
        parts = [message]
        if errors:
            parts.append('部分文件未成功：\n' + '\n'.join(errors[:8]))
        QMessageBox.warning(self, '处理完成', '\n'.join(parts))

    def _delete_selected_group(self):
        rows = self.group_table.selectionModel().selectedRows()
        if not rows or self._result is None:
            return
        group = self._visible_groups[rows[0].row()]
        self._confirm_delete(
            f'当前重复组将删除 {len(group.duplicates)} 个副本，保留：\n'
            f'{self._relative_path(group.keeper)}',
            (group,),
        )

    def _delete_all(self):
        if self._result is None:
            return
        self._confirm_delete(
            f'全部重复组将删除 {self._result.duplicate_file_count} 个副本，'
            f'预计释放 {_format_bytes(self._result.reclaimable_bytes)}。',
            self._result.all_groups,
        )

    def _confirm_delete(self, message: str, groups):
        linked = self.check_linked.isChecked()
        answer = QMessageBox.warning(
            self,
            '确认删除重复副本',
            message
            + ('\n\n已启用联动删除：已确认重复的对应 JSON 标注 / TXT 标签副本将一并删除。'
               if linked else '')
            + '\n\n文件将进入系统回收站（若系统不支持则直接删除）。',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes or self._result is None:
            return
        deletion = delete_duplicate_files(
            self._result, groups, use_trash=True,
            delete_companions=linked,
        )
        parts = [f'成功处理 {len(deletion.deleted)} 个文件。']
        if deletion.skipped:
            parts.append('以下文件未联动删除：\n' + '\n'.join(deletion.skipped[:8]))
        if deletion.errors:
            parts.append('部分文件失败：\n' + '\n'.join(deletion.errors[:8]))
        if len(parts) > 1:
            QMessageBox.warning(self, '删除完成', '\n'.join(parts))
        self.status.setText(f'已处理 {len(deletion.deleted)} 个重复副本，正在重新扫描...')
        self._start_scan()

    def _set_action_enabled(self, enabled: bool):
        self.btn_delete_all.setEnabled(enabled)
        self.btn_delete_group.setEnabled(False)

    def _relative_path(self, path: Path) -> str:
        if self._result is None:
            return str(path)
        try:
            return path.relative_to(self._result.root).as_posix()
        except ValueError:
            return str(path)

    def closeEvent(self, event):
        if self._thread is not None and self._thread.isRunning():
            self._thread.terminate()
            self._thread.wait(1000)
        event.accept()


def _format_bytes(value: int) -> str:
    value = float(value)
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if value < 1024 or unit == 'TB':
            return f'{value:.1f} {unit}' if unit != 'B' else f'{int(value)} B'
        value /= 1024
    return '0 B'


def _guess_dataset_root(path: str) -> str:
    """Walk upward so a previously selected raw batch still resolves to root."""
    if not str(path or '').strip():
        return ''
    try:
        return str(resolve_raw_dataset_root(path))
    except ValueError:
        candidate = Path(path).expanduser()
        return str(candidate)


def create_raw_dialog(parent=None):
    return RawDuplicateDialog(_guess_dataset_root(stored_dataset_path()), parent)
