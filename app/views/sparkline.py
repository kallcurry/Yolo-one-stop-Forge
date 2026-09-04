"""Mini sparkline widget used on model cards (training metric trending)."""

from __future__ import annotations

from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import QWidget


class SparklineWidget(QWidget):
    """Tiny painted trend line for a single metric series."""

    def __init__(self, parent=None, points=None, color: str = '#36B7FF'):
        super().__init__(parent)
        self._points = list(points or [])
        self._color = QColor(color)
        self.setFixedHeight(30)
        self.setMinimumWidth(60)

    def set_series(self, points, color: str = '#36B7FF'):
        self._points = list(points or [])
        self._color = QColor(color)
        self.update()

    def is_empty(self) -> bool:
        return len(self._points) < 2

    def paintEvent(self, _event):
        if self.is_empty():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        width = max(self.width(), 60)
        height = max(self.height(), 24)
        values = [float(point[1]) for point in self._points]
        low, high = min(values), max(values)
        span = (high - low) or 1.0
        points = QPolygonF()
        step = width / (len(values) - 1)
        for index, value in enumerate(values):
            x = index * step
            y = height - 4 - ((value - low) / span) * (height - 8)
            points.append(QPointF(x, y))
        painter.setPen(QPen(self._color, 1.6))
        painter.setBrush(Qt.NoBrush)
        painter.drawPolyline(points)
        # 渐变填充（弱化）
        painter.setPen(Qt.NoPen)
        fill = QColor(self._color)
        fill.setAlpha(36)
        painter.setBrush(fill)
        fill_path = QPolygonF(points)
        fill_path.append(QPointF(width, height))
        fill_path.append(QPointF(0.0, height))
        painter.drawPolygon(fill_path)
        painter.setPen(QPen(self._color, 2.2))
        last = points.at(len(points) - 1)
        painter.drawEllipse(last, 2.4, 2.4)
        painter.end()
