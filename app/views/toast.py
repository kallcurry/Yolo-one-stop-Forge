"""Lightweight toast notifications + message center for the platform.

Progressive disclosure: routine feedback lands in the status bar as before;
important state changes surface as transient toasts (bottom-right stack)
and are archived into the message center (persisted via QSettings).
"""

from __future__ import annotations

import json

from PyQt5.QtCore import QSettings, Qt, QTimer, QPropertyAnimation, pyqtSignal
from PyQt5.QtGui import QColor, QFontMetrics
from PyQt5.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.models.app_defaults import QSETTINGS_ORG, QSETTINGS_APP

from app.models.style_tokens import color as _token_color

TONE_COLORS = {
    'info': (_token_color('accent_blue'), '信息'),
    'success': (_token_color('status_success'), '成功'),
    'warning': (_token_color('status_warning'), '警告'),
    'error': (_token_color('status_error'), '错误'),
}

_HISTORY_KEY = 'toastHistory'


class ToastItem(QFrame):
    """Single toast bubble (auto-dismissed)."""

    dismissed = pyqtSignal(object)

    def __init__(self, manager, text: str, tone: str = 'info',
                 duration_ms: int = 3200):
        super().__init__()
        self._manager = manager
        self._tone = tone
        color, label = TONE_COLORS.get(tone, TONE_COLORS['info'])
        self.setObjectName('toastItem')
        self.setProperty('tone', tone)
        self.setFixedWidth(320)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        self._title = QLabel(label)
        self._title.setObjectName('toastTitle')
        self._title.setStyleSheet(
            f'color: {color}; font-weight: 800; font-size: 10px;'
            'border: none; background: transparent;'
        )
        layout.addWidget(self._title, 0)
        self._text = QLabel(text)
        self._text.setObjectName('toastText')
        self._text.setWordWrap(True)
        self._text.setMaximumWidth(230)
        layout.addWidget(self._text, 1)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._dismiss)
        self._timer.start(duration_ms)

    def _dismiss(self):
        self.dismissed.emit(self)
        self.deleteLater()

    def text_value(self) -> str:
        return self._text.text()


class ToastManager(QWidget):
    """Stack of toasts anchored to the bottom-right of its host window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('toastHost')
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self._items: list[ToastItem] = []
        self.hide()

    def toast(self, text: str, tone: str = 'info', duration_ms: int = 3200):
        item = ToastItem(self, str(text), tone, duration_ms)
        item.dismissed.connect(self._on_dismissed)
        self._items.append(item)
        self._relayout()
        self.show()
        self.raise_()

    def _on_dismissed(self, item):
        if item in self._items:
            self._items.remove(item)
        self._relayout()
        if not self._items:
            self.hide()

    def _relayout(self):
        parent = self.parentWidget()
        if parent is None:
            return
        width = 340
        height = len(self._items) * 86 + 8
        self.setGeometry(parent.width() - width - 16,
                         parent.height() - height - 52,
                         width, height)
        for index, item in enumerate(self._items):
            item.move(8, index * 86)
            item.show()

    def resizeEvent(self, _event):
        self._relayout()


def persist_toast(text: str, tone: str):
    """Archive a toast into the message center (bounded history)."""
    settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)
    try:
        history = json.loads(settings.value(_HISTORY_KEY, '[]'))
    except (TypeError, ValueError):
        history = []
    if not isinstance(history, list):
        history = []
    history.append({'text': str(text), 'tone': str(tone)})
    settings.setValue(_HISTORY_KEY, json.dumps(history[-100:], ensure_ascii=False))


def load_toast_history() -> list[dict]:
    settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)
    try:
        history = json.loads(settings.value(_HISTORY_KEY, '[]'))
    except (TypeError, ValueError):
        return []
    return history if isinstance(history, list) else []


class MessageCenterDialog(QDialog):
    """Message center: archived toasts, newest first, clear button."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('消息中心')
        self.resize(460, 480)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        head = QHBoxLayout()
        title = QLabel('消息中心')
        title.setObjectName('duplicateTitle')
        head.addWidget(title)
        head.addStretch()
        btn_clear = QPushButton('清空')
        btn_clear.setObjectName('fileOpBtn')
        btn_clear.clicked.connect(self._clear)
        head.addWidget(btn_clear)
        layout.addLayout(head)

        self.list_widget = QListWidget()
        self.list_widget.setObjectName('trainingLog')
        layout.addWidget(self.list_widget, 1)
        hint = QLabel('重要状态变更（评估完成/转换完成/错误…）会归档在此，重启不丢失')
        hint.setObjectName('evaluationSectionHint')
        layout.addWidget(hint)
        self.reload()

    def reload(self):
        self.list_widget.clear()
        for record in reversed(load_toast_history()):
            color, label = TONE_COLORS.get(
                record.get('tone', 'info'), TONE_COLORS['info']
            )
            item = QListWidgetItem(f'[{label}] {record.get("text", "")}')
            item.setForeground(QColor(color))
            self.list_widget.addItem(item)

    def _clear(self):
        QSettings(QSETTINGS_ORG, QSETTINGS_APP).remove(_HISTORY_KEY)
        self.reload()


def world_toast(host, text: str, tone: str = 'info', duration_ms: int = 3200):
    """Convenience: show toast on a host that has a ToastManager + archive."""
    manager = getattr(host, 'toast_manager', None)
    if manager is not None:
        manager.toast(text, tone, duration_ms)
    persist_toast(text, tone)


def notify_any(widget, text: str, tone: str = 'info'):
    """Notify from any widget: toast on its host window + archive."""
    persist_toast(text, tone)
    host = widget.window() if widget is not None else None
    if host is not None and getattr(host, 'toast_manager', None) is not None:
        host.toast(str(text), tone)
