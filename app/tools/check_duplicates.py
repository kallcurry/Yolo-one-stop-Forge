"""Duplicate and overlap checking tools.

The original training/test overlap checker remains available below.  The
interactive raw-data audit in this module is a separate workflow so the two
different meanings of "duplicate" do not get mixed together.
"""

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
    QMessageBox,
    QPushButton,
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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        title_row = QHBoxLayout()
        title = QLabel('全数据目录 · 原始数据重复审查')
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
        layout.addLayout(metrics)

        self.summary = QLabel('尚未扫描')
        self.summary.setObjectName('duplicateSummary')
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

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
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        action_row.addWidget(buttons)
        layout.addLayout(action_row)

        if default_root:
            self._start_scan()

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
        self.group_table.setRowCount(0)
        self.detail_list.clear()
        self._set_action_enabled(False)
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
        self.summary.setText(
            f'已扫描 {result.root} · 发现 {result.duplicate_group_count} 个重复组，'
            f'共 {result.duplicate_file_count} 个可删除副本。'
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
        answer = QMessageBox.warning(
            self,
            '确认删除重复副本',
            message + '\n\n文件将进入系统回收站（若系统不支持则直接删除）。',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes or self._result is None:
            return
        deletion = delete_duplicate_files(self._result, groups)
        if deletion.errors:
            QMessageBox.warning(
                self,
                '删除完成但有失败项',
                f'成功处理 {len(deletion.deleted)} 个文件。\n' +
                '\n'.join(deletion.errors[:8]),
            )
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
