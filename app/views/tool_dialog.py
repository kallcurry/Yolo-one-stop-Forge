"""Generic tool runner dialog: parameter panel + run button + log output."""

import sys
import io
from pathlib import Path

from PyQt5.QtCore import QSettings, Qt, pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.views.ui_effects import HoverGlow


def stored_dataset_path(*relative_parts: str) -> str:
    """Return a portable tool default based on the last opened dataset."""
    value = QSettings('FilesProcessQT', 'ImageManager').value('lastDirectory')
    if not value:
        return ''
    root = Path(str(value)).expanduser()
    return str(root.joinpath(*relative_parts))


class _WorkerThread(QThread):
    """Runs a tool function in a background thread, capturing stdout."""

    log_msg = pyqtSignal(str)
    finished_run = pyqtSignal(bool, str)

    def __init__(self, run_func, parent=None):
        super().__init__(parent)
        self._run_func = run_func

    def run(self):
        buf = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = _TeeWriter(original_stdout, self.log_msg)
        try:
            self._run_func()
            sys.stdout = original_stdout
            self.finished_run.emit(True, '完成')
        except Exception as e:
            sys.stdout = original_stdout
            import traceback
            self.log_msg.emit(traceback.format_exc())
            self.finished_run.emit(False, str(e))


class _TeeWriter:
    """Writes to both original stdout and a Qt signal."""
    def __init__(self, original, signal):
        self._orig = original
        self._sig = signal

    def write(self, text):
        if text and text.strip():
            self._sig.emit(text.rstrip())
        if self._orig:
            self._orig.write(text)

    def flush(self):
        if self._orig:
            self._orig.flush()


class ToolDialog(QDialog):
    """Base dialog for all data processing tools.

    Usage:
        dlg = ToolDialog('工具名称')
        # Add parameter widgets to dlg.param_widget
        dlg.set_runner(your_function)
        dlg.exec_()
    """

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(650, 520)
        self.resize(700, 600)

        layout = QVBoxLayout(self)

        # Parameter area (subclass fills this)
        self.param_widget = QVBoxLayout()
        layout.addLayout(self.param_widget)

        # Run / Stop buttons
        btn_row = QHBoxLayout()
        self.btn_run = QPushButton('▶ 运行')
        self.btn_run.setObjectName('successBtn')
        self.btn_stop = QPushButton('⬛ 停止')
        self.btn_stop.setObjectName('dangerBtn')
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_stop)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Log output
        self.log_view = QTextEdit()
        self.log_view.setObjectName('logView')
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view, 1)

        # Close button
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._thread = None
        self._runner = None
        self._hover_glow = HoverGlow(self)
        self._hover_glow.watch(self.btn_run)
        self._hover_glow.watch(self.btn_stop)

        self.btn_run.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._stop)

    def set_runner(self, func):
        """Set the function to run (takes no arguments, uses captured params)."""
        self._runner = func

    def _start(self):
        if self._runner is None:
            return
        self.log_view.clear()
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._thread = _WorkerThread(self._runner, self)
        self._thread.log_msg.connect(self._append_log)
        self._thread.finished_run.connect(self._on_finished)
        self._thread.start()

    def _stop(self):
        if self._thread and self._thread.isRunning():
            self._thread.terminate()
            self._thread.wait(1000)
        self._append_log('⚠️ 用户停止')
        self._reset_buttons()

    def _on_finished(self, ok: bool, msg: str):
        if ok:
            self._append_log(f'✅ {msg}')
        else:
            self._append_log(f'❌ 运行出错: {msg}')
        self._reset_buttons()

    def _reset_buttons(self):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _append_log(self, text: str):
        self.log_view.append(text)
        # Auto-scroll to bottom
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ---- Helper widgets for parameter panels ----

    def _add_dir_picker(self, label: str, default: str = '') -> QLineEdit:
        """Add a row: label + QLineEdit + [Browse] button."""
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        edit = QLineEdit(default)
        edit.setMinimumWidth(300)
        row.addWidget(edit)
        btn = QPushButton('浏览...')
        btn.clicked.connect(
            lambda: self._browse_dir(edit)
        )
        self._hover_glow.watch(btn)
        row.addWidget(btn)
        self.param_widget.addLayout(row)
        return edit

    def _add_file_picker(self, label: str, default: str = '') -> QLineEdit:
        """Add a row: label + QLineEdit + [Browse] file button."""
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        edit = QLineEdit(default)
        edit.setMinimumWidth(300)
        row.addWidget(edit)
        btn = QPushButton('浏览...')
        btn.clicked.connect(
            lambda: self._browse_file(edit)
        )
        self._hover_glow.watch(btn)
        row.addWidget(btn)
        self.param_widget.addLayout(row)
        return edit

    def _browse_dir(self, edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, '选择目录')
        if path:
            edit.setText(path)

    def _browse_file(self, edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(self, '选择文件')
        if path:
            edit.setText(path)
