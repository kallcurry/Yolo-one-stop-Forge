"""Bottom panel: file list with checkboxes for multi-select.

Supports: individual click, Ctrl+Click, Shift+Click range selection.
"""

from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class FileListPanel(QWidget):
    """Collapsible bottom panel listing all images in the current directory."""

    current_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._syncing = False
        self._checked_count = 0   # cached count, avoids full traversal
        self._current_highlight = -1  # previously highlighted row

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 4)

        top_row = QHBoxLayout()
        self.btn_select_all = QPushButton('☐ 全选')
        self.btn_invert = QPushButton('☑ 反选')
        self.btn_clear_sel = QPushButton('☐ 清空')
        self.lbl_selected = QLabel('已选: 0')

        for btn in [self.btn_select_all, self.btn_invert, self.btn_clear_sel]:
            btn.setMaximumWidth(80)
        top_row.addWidget(self.btn_select_all)
        top_row.addWidget(self.btn_invert)
        top_row.addWidget(self.btn_clear_sel)
        top_row.addWidget(self.lbl_selected)
        top_row.addStretch()
        layout.addLayout(top_row)

        self.list_widget = QListWidget()
        self.list_widget.setAlternatingRowColors(False)
        self.list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_widget.setDragEnabled(True)
        self.list_widget.setUniformItemSizes(True)  # big perf boost for large lists
        self.list_widget.doubleClicked.connect(self._on_double_click)
        self.list_widget.itemChanged.connect(self._on_item_check_changed)
        self.list_widget.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.list_widget)

        self._image_paths: list[Path] = []

        self.btn_select_all.clicked.connect(self._select_all)
        self.btn_invert.clicked.connect(self._invert)
        self.btn_clear_sel.clicked.connect(self._clear)

    def populate(self, images: list[Path]):
        """Fill the list efficiently — batch create items, no per-item signals."""
        self._image_paths = images
        self._current_highlight = -1

        # Freeze UI updates and block signals during bulk population
        self.list_widget.setUpdatesEnabled(False)
        self.list_widget.blockSignals(True)
        self.list_widget.clear()

        for img in images:
            item = QListWidgetItem()
            item.setText(img.name)
            item.setData(Qt.UserRole, str(img))
            item.setCheckState(Qt.Unchecked)
            self.list_widget.addItem(item)

        self.list_widget.blockSignals(False)
        self.list_widget.setUpdatesEnabled(True)
        self._checked_count = 0
        self.lbl_selected.setText('已选: 0')

    def set_current_index(self, index: int):
        """Highlight only the current row — O(1) instead of O(n)."""
        count = self.list_widget.count()
        if count == 0:
            return
        # Un-highlight previous
        if 0 <= self._current_highlight < count:
            self.list_widget.item(self._current_highlight).setBackground(QColor('#12161C'))
        # Highlight new
        if 0 <= index < count:
            self.list_widget.item(index).setBackground(QColor('#24364A'))
        self._current_highlight = index

    def get_selected(self) -> list[Path]:
        """Return paths of all checked images."""
        selected = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                path = item.data(Qt.UserRole)
                if path:
                    selected.append(Path(path))
        return selected

    # --- signal handlers ---

    def _on_double_click(self, index):
        self.current_changed.emit(index.row())

    def _on_item_check_changed(self, item):
        if self._syncing:
            return
        delta = 1 if item.checkState() == Qt.Checked else -1
        self._checked_count += delta
        self.lbl_selected.setText(f'已选: {max(0, self._checked_count)}')

    def _on_selection_changed(self):
        """Sync checkboxes when selection changes (Shift/Ctrl+Click)."""
        if self._syncing:
            return
        self._syncing = True
        new_checked = 0
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            is_sel = item.isSelected()
            target = Qt.Checked if is_sel else Qt.Unchecked
            item.setCheckState(target)
            if is_sel:
                new_checked += 1
        self._syncing = False
        self._checked_count = new_checked
        self.lbl_selected.setText(f'已选: {new_checked}')

    def _select_all(self):
        self._syncing = True
        n = self.list_widget.count()
        self.list_widget.setUpdatesEnabled(False)
        for i in range(n):
            item = self.list_widget.item(i)
            item.setCheckState(Qt.Checked)
            item.setSelected(True)
        self.list_widget.setUpdatesEnabled(True)
        self._syncing = False
        self._checked_count = n
        self.lbl_selected.setText(f'已选: {n}')

    def _invert(self):
        self._syncing = True
        new_checked = 0
        self.list_widget.setUpdatesEnabled(False)
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            is_checked = item.checkState() == Qt.Checked
            new_state = Qt.Unchecked if is_checked else Qt.Checked
            item.setCheckState(new_state)
            item.setSelected(new_state == Qt.Checked)
            if new_state == Qt.Checked:
                new_checked += 1
        self.list_widget.setUpdatesEnabled(True)
        self._syncing = False
        self._checked_count = new_checked
        self.lbl_selected.setText(f'已选: {new_checked}')

    def _clear(self):
        self._syncing = True
        n = self.list_widget.count()
        self.list_widget.setUpdatesEnabled(False)
        for i in range(n):
            item = self.list_widget.item(i)
            item.setCheckState(Qt.Unchecked)
            item.setSelected(False)
        self.list_widget.setUpdatesEnabled(True)
        self._syncing = False
        self._checked_count = 0
        self.lbl_selected.setText('已选: 0')

