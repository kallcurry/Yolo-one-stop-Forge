"""Application dialogs: rename, new folder, confirm delete."""

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)


class RenameDialog(QDialog):
    """Dialog for renaming a single file or folder."""

    def __init__(self, path: Path | str, parent=None):
        super().__init__(parent)
        self.setWindowTitle('重命名')
        self.setMinimumWidth(400)

        self._path = Path(path)
        self._new_name = ''

        layout = QVBoxLayout(self)

        name = self._path.name
        if self._path.is_file():
            self._stem = self._path.stem
            self._suffix = self._path.suffix
            default = self._stem
        else:
            self._stem = name
            self._suffix = ''
            default = name

        layout.addWidget(QLabel(f'原名称: {name}'))
        layout.addWidget(QLabel('新名称:'))

        self.edit = QLineEdit(default)
        self.edit.selectAll()
        layout.addWidget(self.edit)

        if self._suffix:
            layout.addWidget(QLabel(f'扩展名将自动保留: {self._suffix}'))

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self):
        name = self.edit.text().strip()
        if not name:
            return
        self._new_name = name + self._suffix
        self.accept()

    def get_new_name(self) -> str:
        return self._new_name


class NewFolderDialog(QDialog):
    """Dialog for creating a new folder."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('新建文件夹')
        self.setMinimumWidth(350)

        self._folder_name = ''

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('请输入文件夹名称:'))

        self.edit = QLineEdit()
        self.edit.setPlaceholderText('新建文件夹')
        layout.addWidget(self.edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self):
        name = self.edit.text().strip()
        if not name:
            return
        self._folder_name = name
        self.accept()

    def get_folder_name(self) -> str:
        return self._folder_name
