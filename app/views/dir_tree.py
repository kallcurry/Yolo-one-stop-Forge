"""Directory tree panel with context menu and drag-drop support."""

from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QPushButton,
    QComboBox,
    QLabel,
    QHBoxLayout,
    QMenu,
    QMessageBox,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from app.models.file_system import DirFormat, scan_tree, IMAGE_EXTENSIONS


FORMAT_COLORS = {
    DirFormat.DJI_PAIR: QColor('#3BA4FF'),    # blue
    DirFormat.SEPARATED: QColor('#45D483'),   # green
    DirFormat.FLAT: QColor('#93A1B3'),        # gray
}

TOTAL_COLOR = QColor('#93A1B3')              # gray for count text


class DirTreePanel(QWidget):
    annotation_dir_changed = pyqtSignal(str)
    """Left panel: directory tree with format color-coding."""

    directory_selected = pyqtSignal(str, DirFormat)
    new_folder_requested = pyqtSignal(str)
    rename_folder_requested = pyqtSignal(str)
    delete_folder_requested = pyqtSignal(str)
    files_dropped = pyqtSignal(str, list)  # dest_path, list of source Paths

    def __init__(self, parent=None):
        super().__init__(parent)
        self._root_path = ''
        self._checked_path: str | None = None
        self._setting_check = False  # guard against recursive itemChanged

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        selector = QHBoxLayout()
        selector.setContentsMargins(8, 6, 8, 2)
        selector.setSpacing(6)
        caption = QLabel('标注集（点击切换）')
        caption.setObjectName('captionLabel')
        caption.setToolTip('下拉列出当前数据目录的全部标注集，切换即生效')
        selector.addWidget(caption)
        self.annotation_combo = QComboBox()
        self.annotation_combo.setObjectName('smallCombo')
        self.annotation_combo.setEditable(True)
        self.annotation_combo.setToolTip(
            '点击下拉即可切换当前目录的标注集（annotations / annotations-obb …），'
            '选择后立即生效并刷新当前图片'
        )
        self.annotation_combo.currentIndexChanged.connect(
            self._emit_annotation_dir
        )
        selector.addWidget(self.annotation_combo, 1)
        layout.addLayout(selector)

        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels(['目录'])
        self.model.itemChanged.connect(self._on_check_changed)

        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tree.setHeaderHidden(True)
        self.tree.setAnimated(True)
        self.tree.setIndentation(18)
        self.tree.clicked.connect(self._on_clicked)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)

        # Drag-drop: accept file drops
        self.tree.setAcceptDrops(True)
        self.tree.setDragDropMode(QAbstractItemView.DropOnly)
        self.tree.viewport().installEventFilter(self)

        layout.addWidget(self.tree)

    def populate_annotation_dirs(self, root_path: str, current: str = ''):
        """Discover annotation-set directories under the data root."""
        self.annotation_combo.blockSignals(True)
        self.annotation_combo.clear()
        root = Path(str(root_path))
        discovered = []
        if root.is_dir():
            for child in sorted(root.iterdir()):
                if child.is_dir() and 'annotation' in child.name.lower():
                    discovered.append(child.name)
        if not discovered:
            discovered = ['annotations']
        current = str(current or '').strip()
        items = list(dict.fromkeys(discovered + ([current] if current else [])))
        for name in items:
            self.annotation_combo.addItem(name)
        if current and current in items:
            self.annotation_combo.setCurrentText(current)
        elif items:
            self.annotation_combo.setCurrentIndex(0)
        self.annotation_combo.blockSignals(False)

    def _emit_annotation_dir(self):
        name = self.annotation_combo.currentText().strip()
        if name:
            self.annotation_dir_changed.emit(name)

    def load_root(self, root_path: str):
        """Populate the tree from a root directory path."""
        self._root_path = root_path
        self.model.clear()
        root_data = scan_tree(root_path)
        self._populate_tree(self.model.invisibleRootItem(), root_data)
        self.tree.expandAll()
        # Restore previously checked path
        if self._checked_path and not self._restore_check(self._checked_path):
            self._checked_path = None

    def refresh(self):
        """Reload the current root."""
        if self._root_path:
            self.load_root(self._root_path)

    def _populate_tree(self, parent_item, node_data):
        """Recursively add items from scan_tree data, with image counts."""
        if not node_data:
            return
        name = node_data.get('display_name') or Path(node_data['path']).name
        fmt = node_data.get('format')
        selectable = node_data.get('selectable', True)
        kind = node_data.get('kind', 'directory')

        # Count images in this directory
        img_count = self._count_images(node_data['path'], fmt)

        # Build display text with count
        display = f'{name} ({img_count})' if img_count > 0 else name

        item = QStandardItem(display)
        item.setData(node_data['path'], Qt.UserRole)
        item.setData(fmt, Qt.UserRole + 1)
        item.setData(kind, Qt.UserRole + 2)
        item.setCheckable(selectable)
        if kind == 'scope':
            color = QColor('#67D9FF')
            font = item.font()
            font.setBold(True)
            item.setFont(font)
        elif kind == 'project':
            color = QColor('#DCEBFA')
            font = item.font()
            font.setBold(True)
            item.setFont(font)
        else:
            color = FORMAT_COLORS.get(fmt, QColor('#D7DEE8'))
        item.setForeground(color)
        parent_item.appendRow(item)

        for child in node_data.get('children', []):
            self._populate_tree(item, child)

    @staticmethod
    def _count_images(path_str: str, fmt) -> int:
        """Quickly count image files in a directory."""
        from pathlib import Path
        p = Path(path_str)
        if not p.is_dir():
            return 0
        if fmt == DirFormat.DJI_PAIR and (p / 'images').is_dir():
            p = p / 'images'
        try:
            return sum(1 for f in p.iterdir()
                       if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)
        except (OSError, PermissionError):
            return 0

    def _on_clicked(self, index):
        item = self.model.itemFromIndex(index)
        if not item or not item.isCheckable():
            return
        path = item.data(Qt.UserRole)
        fmt = item.data(Qt.UserRole + 1)
        if path:
            # Check this item — _on_check_changed does the rest
            item.setCheckState(Qt.Checked)

    def _on_check_changed(self, item):
        """When an item is checked: uncheck others, emit signal."""
        if self._setting_check or not item.isCheckable():
            return
        path = item.data(Qt.UserRole)
        if not path:
            return
        if item.checkState() == Qt.Checked:
            if path == self._checked_path:
                return  # already checked, no-op
            self._checked_path = path
            self._setting_check = True
            self._uncheck_all_except(item)
            self._setting_check = False
            fmt = item.data(Qt.UserRole + 1)
            self.directory_selected.emit(path, fmt)
        elif item.checkState() == Qt.Unchecked:
            # Prevent unchecking the currently active directory
            if path == self._checked_path:
                self._setting_check = True
                item.setCheckState(Qt.Checked)
                self._setting_check = False

    def _uncheck_all_except(self, keep_item):
        """Recursively uncheck all items except `keep_item`."""
        def _uncheck(parent):
            for row in range(parent.rowCount()):
                child = parent.child(row)
                if child.isCheckable() and child != keep_item:
                    child.setCheckState(Qt.Unchecked)
                if child.hasChildren():
                    _uncheck(child)
        _uncheck(self.model.invisibleRootItem())

    def _restore_check(self, path: str):
        """Find the item with `path` and re-check it after model rebuild."""
        def _find(parent):
            for row in range(parent.rowCount()):
                child = parent.child(row)
                if child.data(Qt.UserRole) == path:
                    self._setting_check = True
                    child.setCheckState(Qt.Checked)
                    self._setting_check = False
                    return True
                if child.hasChildren() and _find(child):
                    return True
            return False
        return _find(self.model.invisibleRootItem())

    def select_path(self, path: str | Path, emit: bool = True) -> bool:
        """Select and check a directory node by its absolute path."""
        try:
            target = str(Path(path).expanduser().resolve())
        except OSError:
            target = str(Path(path).expanduser())

        def _find(parent):
            for row in range(parent.rowCount()):
                child = parent.child(row)
                child_path = child.data(Qt.UserRole)
                if child_path:
                    try:
                        normalized = str(Path(child_path).expanduser().resolve())
                    except OSError:
                        normalized = str(Path(child_path).expanduser())
                    if normalized == target:
                        return child
                if child.hasChildren():
                    found = _find(child)
                    if found is not None:
                        return found
            return None

        item = _find(self.model.invisibleRootItem())
        if item is None or not item.isCheckable():
            return False

        self._setting_check = True
        self._uncheck_all_except(item)
        item.setCheckState(Qt.Checked)
        self._setting_check = False
        self._checked_path = item.data(Qt.UserRole)
        index = item.index()
        self.tree.setCurrentIndex(index)
        self.tree.scrollTo(index, QAbstractItemView.PositionAtCenter)
        if emit:
            self.directory_selected.emit(
                self._checked_path,
                item.data(Qt.UserRole + 1),
            )
        return True

    def _show_context_menu(self, pos):
        index = self.tree.indexAt(pos)
        path = index.data(Qt.UserRole) if index.isValid() else self._root_path
        if not path:
            return

        menu = QMenu(self)
        menu.addAction('📂 新建文件夹', lambda: self.new_folder_requested.emit(path))
        if index.isValid():
            menu.addAction('✏️ 重命名文件夹', lambda: self.rename_folder_requested.emit(path))
            menu.addAction('🗑 删除文件夹', lambda: self.delete_folder_requested.emit(path))
        menu.addSeparator()
        menu.addAction('🔄 刷新', self.refresh)
        menu.exec_(self.tree.viewport().mapToGlobal(pos))

    def selected_path(self) -> str | None:
        """Return the path of the currently selected item."""
        indexes = self.tree.selectedIndexes()
        if indexes:
            return indexes[0].data(Qt.UserRole)
        return None

    def eventFilter(self, obj, event):
        from PyQt5.QtCore import QEvent
        if obj == self.tree.viewport() and event.type() == QEvent.Drop:
            mime = event.mimeData()
            if mime.hasUrls():
                paths = [Path(u.toLocalFile()) for u in mime.urls()]
                # Find target directory from drop position
                index = self.tree.indexAt(event.pos())
                dest = index.data(Qt.UserRole) if index.isValid() else self._root_path
                if dest and paths:
                    self.files_dropped.emit(str(dest), paths)
                    return True
        return super().eventFilter(obj, event)
