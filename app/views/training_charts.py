"""Lightweight real-time charts used by the training center."""

from __future__ import annotations

import math

from PyQt5.QtCore import QEasingCurve, QPointF, QRectF, Qt, QVariantAnimation
from PyQt5.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import QSizePolicy, QStyle, QStyleOption, QWidget


SERIES_COLORS = (
    '#36B7FF',
    '#45D483',
    '#F5A524',
    '#FF6677',
    '#B88CFF',
    '#62E8FF',
    '#F28AC8',
    '#A4D65E',
)

METRIC_GROUPS = (
    ('loss', '损失曲线'),
    ('quality', '质量指标'),
    ('learning_rate', '学习率'),
    ('other', '其他指标'),
)


def metric_group(metric_name: str) -> str:
    """Map an Ultralytics metric name to a chart with a shared scale."""
    name = str(metric_name or '').strip().lower()
    if 'loss' in name:
        return 'loss'
    if name.startswith('lr/') or name.startswith('lr') or 'learning_rate' in name:
        return 'learning_rate'
    if (
        name.startswith('metrics/')
        or 'map' in name
        or 'precision' in name
        or 'recall' in name
        or name == 'fitness'
    ):
        return 'quality'
    return 'other'


def _format_value(value: float) -> str:
    if value != 0 and abs(value) < 0.001:
        return f'{value:.2e}'
    return f'{value:.4f}'


class RealtimeTrainingChart(QWidget):
    """Animated multi-series chart fed incrementally by epoch events."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('trainingCurveChart')
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMouseTracking(True)
        self.setMinimumSize(360, 250)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._series: dict[str, list[tuple[int, float]]] = {}
        self._group = 'loss'
        self._expected_epochs = 0
        self._latest_epoch = 0
        self._animation_progress = 1.0
        self._painted_points = []
        self._hovered = None
        self._animation = QVariantAnimation(self)
        self._animation.setDuration(260)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)
        self._animation.valueChanged.connect(self._set_animation_progress)

    @property
    def series(self) -> dict[str, tuple[tuple[int, float], ...]]:
        return {name: tuple(points) for name, points in self._series.items()}

    def clear(self, expected_epochs: int = 0):
        self._animation.stop()
        self._series.clear()
        self._expected_epochs = max(0, int(expected_epochs or 0))
        self._latest_epoch = 0
        self._animation_progress = 1.0
        self._painted_points = []
        self._hovered = None
        self.update()

    def set_metric_group(self, group: str):
        valid = {key for key, _label in METRIC_GROUPS}
        self._group = group if group in valid else 'loss'
        self._hovered = None
        self._animate(420)

    def append_metrics(self, epoch: int, metrics):
        if not isinstance(metrics, dict):
            return
        epoch = max(0, int(epoch or 0))
        changed = False
        for key, raw_value in metrics.items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            name = str(key)
            points = self._series.setdefault(name, [])
            if points and points[-1][0] == epoch:
                points[-1] = (epoch, value)
            else:
                points.append((epoch, value))
            changed = True
        if changed:
            self._latest_epoch = max(self._latest_epoch, epoch)
            self._hovered = None
            self._animate(260)

    def available_groups(self) -> set[str]:
        return {metric_group(name) for name in self._series}

    def _animate(self, duration: int):
        self._animation.stop()
        self._animation.setDuration(duration)
        self._animation.setStartValue(0.0)
        self._animation.setEndValue(1.0)
        self._animation.start()

    def _set_animation_progress(self, value):
        self._animation_progress = float(value)
        self.update()

    def _visible_series(self):
        return [
            (name, points)
            for name, points in self._series.items()
            if metric_group(name) == self._group and points
        ]

    def _plot_rect(self, legend_rows: int) -> QRectF:
        top = 52 + legend_rows * 24
        return QRectF(
            62,
            top,
            max(60, self.width() - 84),
            max(55, self.height() - top - 42),
        )

    @staticmethod
    def _bounds(visible):
        values = [value for _name, points in visible for _epoch, value in points]
        epochs = [epoch for _name, points in visible for epoch, _value in points]
        if not values or not epochs:
            return None
        x_min, x_max = min(epochs), max(epochs)
        if x_min == x_max:
            x_min = max(0, x_min - 1)
            x_max += 1
        y_min, y_max = min(values), max(values)
        if y_min == y_max:
            padding = max(abs(y_min) * 0.1, 0.01)
        else:
            padding = max((y_max - y_min) * 0.12, 0.0001)
        return x_min, x_max, y_min - padding, y_max + padding

    def paintEvent(self, _event):
        option = QStyleOption()
        option.initFrom(self)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        self.style().drawPrimitive(QStyle.PE_Widget, option, painter, self)
        self._painted_points = []

        visible = self._visible_series()
        if not visible:
            group_label = dict(METRIC_GROUPS).get(self._group, '当前分组')
            painter.setPen(QColor('#71879A'))
            painter.drawText(
                self.rect(), Qt.AlignCenter,
                f'等待 {group_label} 数据',
            )
            return

        legend_columns = max(1, min(3, max(1, (self.width() - 36) // 190)))
        legend_rows = math.ceil(len(visible) / legend_columns)
        rect = self._plot_rect(legend_rows)
        bounds = self._bounds(visible)
        if bounds is None:
            return
        x_min, x_max, y_min, y_max = bounds

        painter.setPen(QColor('#7897AA'))
        painter.drawText(
            QRectF(14, 10, max(120, self.width() - 28), 24),
            Qt.AlignLeft | Qt.AlignVCenter,
            f'实时曲线  ·  Epoch {self._latest_epoch}'
            + (f' / {self._expected_epochs}' if self._expected_epochs else ''),
        )

        cell_width = max(120.0, (self.width() - 28.0) / legend_columns)
        for index, (name, _points) in enumerate(visible):
            row, column = divmod(index, legend_columns)
            color = QColor(SERIES_COLORS[index % len(SERIES_COLORS)])
            cell = QRectF(
                14 + column * cell_width,
                34 + row * 24,
                cell_width - 8,
                20,
            )
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(QPointF(cell.left() + 5, cell.center().y()), 3.5, 3.5)
            painter.setPen(QColor('#CFE3EE'))
            text_rect = cell.adjusted(14, 0, -2, 0)
            text = painter.fontMetrics().elidedText(
                name, Qt.ElideMiddle, int(text_rect.width())
            )
            painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, text)

        painter.setPen(QPen(QColor(88, 126, 153, 55), 1))
        for step in range(6):
            ratio = step / 5
            y = rect.bottom() - ratio * rect.height()
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            value = y_min + ratio * (y_max - y_min)
            painter.setPen(QColor('#71879A'))
            painter.drawText(
                QRectF(3, y - 9, 53, 18),
                Qt.AlignRight | Qt.AlignVCenter,
                _format_value(value),
            )
            painter.setPen(QPen(QColor(88, 126, 153, 55), 1))
        for step in range(6):
            ratio = step / 5
            x = rect.left() + ratio * rect.width()
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            epoch = x_min + ratio * (x_max - x_min)
            painter.setPen(QColor('#71879A'))
            painter.drawText(
                QRectF(x - 26, rect.bottom() + 7, 52, 18),
                Qt.AlignCenter,
                f'{epoch:.0f}',
            )
            painter.setPen(QPen(QColor(88, 126, 153, 55), 1))

        for index, (name, points) in enumerate(visible):
            color = QColor(SERIES_COLORS[index % len(SERIES_COLORS)])
            path = QPainterPath()
            screen_points = []
            for point_index, (epoch, value) in enumerate(points):
                x_ratio = (epoch - x_min) / (x_max - x_min)
                y_ratio = (value - y_min) / (y_max - y_min)
                screen = QPointF(
                    rect.left() + x_ratio * rect.width(),
                    rect.bottom() - y_ratio * rect.height(),
                )
                screen_points.append((screen, epoch, value, name, color))
                if point_index == 0:
                    path.moveTo(screen)
                else:
                    path.lineTo(screen)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(color, 2.2))
            painter.drawPath(path)
            if screen_points:
                latest = screen_points[-1][0]
                halo = 3.5 + (1.0 - self._animation_progress) * 4.0
                halo_color = QColor(color)
                halo_color.setAlpha(round(95 * self._animation_progress))
                painter.setPen(Qt.NoPen)
                painter.setBrush(halo_color)
                painter.drawEllipse(latest, halo, halo)
                painter.setBrush(color)
                painter.drawEllipse(latest, 3.0, 3.0)
            self._painted_points.extend(screen_points)

        if self._hovered is not None:
            screen, epoch, value, name, color = self._hovered
            painter.setPen(QPen(QColor(98, 232, 255, 60), 1, Qt.DashLine))
            painter.drawLine(QPointF(screen.x(), rect.top()), QPointF(screen.x(), rect.bottom()))
            painter.setPen(QPen(color, 2))
            painter.setBrush(QColor('#07131D'))
            painter.drawEllipse(screen, 5, 5)
            text = f'{name}\nEpoch {epoch}  ·  {_format_value(value)}'
            tooltip = QRectF(screen.x() + 12, screen.y() - 53, 230, 45)
            if tooltip.right() > self.width() - 8:
                tooltip.moveRight(screen.x() - 12)
            if tooltip.top() < rect.top():
                tooltip.moveTop(screen.y() + 12)
            painter.setPen(QPen(QColor(98, 232, 255, 105), 1))
            painter.setBrush(QColor(8, 19, 29, 244))
            painter.drawRoundedRect(tooltip, 6, 6)
            painter.setPen(QColor('#E7F4FB'))
            painter.drawText(
                tooltip.adjusted(9, 4, -8, -4),
                Qt.AlignLeft | Qt.AlignVCenter,
                text,
            )

    def _nearest_point(self, position, radius: float = 12.0):
        nearest = None
        nearest_distance = radius * radius
        for item in self._painted_points:
            screen = item[0]
            distance = (
                (screen.x() - position.x()) ** 2
                + (screen.y() - position.y()) ** 2
            )
            if distance < nearest_distance:
                nearest = item
                nearest_distance = distance
        return nearest

    def mouseMoveEvent(self, event):
        hovered = self._nearest_point(event.pos())
        if hovered != self._hovered:
            self._hovered = hovered
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self._hovered = None
        self.update()
        super().leaveEvent(event)
