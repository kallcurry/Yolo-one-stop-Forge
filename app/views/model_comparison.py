"""Interactive comparison workspace for Ultralytics training runs."""

from __future__ import annotations

from PyQt5.QtCore import (
    QEasingCurve,
    QPointF,
    QRectF,
    Qt,
    QVariantAnimation,
    pyqtSignal,
)
from PyQt5.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStyle,
    QStyleOption,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.models.model_registry import MetricPoint, MetricSeries, ModelRecord


SERIES_COLORS = (
    '#36B7FF',
    '#45D483',
    '#F5A524',
    '#FF6677',
    '#B88CFF',
    '#62E8FF',
)

TASK_PRIMARY_SERIES = {
    'pose': 'metrics/mAP50-95(P)',
    'detection': 'metrics/mAP50-95(B)',
    'segmentation': 'metrics/mAP50-95(M)',
    'obb': 'metrics/mAP50-95(B)',
}


def _series_for(record: ModelRecord, key: str) -> MetricSeries | None:
    return next((series for series in record.metric_series if series.key == key), None)


def _series_best(series: MetricSeries | None) -> MetricPoint | None:
    if series is None or not series.points:
        return None
    selector = min if series.higher_is_better is False else max
    return selector(series.points, key=lambda point: point.value)


def _format_value(value: float | None) -> str:
    if value is None:
        return '-'
    if value != 0 and abs(value) < 0.001:
        return f'{value:.2e}'
    return f'{value:.4f}'


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds or 0)))
    if total <= 0:
        return '-'
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f'{hours}h {minutes}m'
    if minutes:
        return f'{minutes}m {secs}s'
    return f'{secs}s'


def _smoothed_points(points: tuple[MetricPoint, ...], window: int = 5):
    if len(points) < 3:
        return points
    result = []
    radius = max(1, window // 2)
    for index, point in enumerate(points):
        start = max(0, index - radius)
        end = min(len(points), index + radius + 1)
        average = sum(item.value for item in points[start:end]) / (end - start)
        result.append(MetricPoint(point.epoch, average))
    return tuple(result)


class TrainingCurveChart(QWidget):
    """Animated multi-model line chart with point inspection and navigation."""

    model_activated = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('modelCurveChart')
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMouseTracking(True)
        self.setMinimumHeight(290)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._records: tuple[ModelRecord, ...] = ()
        self._metric_key = ''
        self._x_mode = 'epoch'
        self._smoothing = False
        self._progress = 1.0
        self._highlighted_id = ''
        self._hovered = None
        self._painted_points = []
        self._legend_hits = []
        self._animation = QVariantAnimation(self)
        self._animation.setDuration(520)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)
        self._animation.valueChanged.connect(self._set_progress)

    def set_data(self, records, metric_key: str, x_mode='epoch', smoothing=False):
        self._records = tuple(records)
        self._metric_key = str(metric_key or '')
        self._x_mode = x_mode if x_mode in {'epoch', 'progress'} else 'epoch'
        self._smoothing = bool(smoothing)
        if self._highlighted_id not in {record.model_id for record in self._records}:
            self._highlighted_id = ''
        self._hovered = None
        self._animation.stop()
        self._animation.setStartValue(0.0)
        self._animation.setEndValue(1.0)
        self._animation.start()
        self.update()

    def _set_progress(self, value):
        self._progress = float(value)
        self.update()

    def _plot_rect(self) -> QRectF:
        return QRectF(64, 62, max(40, self.width() - 88), max(40, self.height() - 106))

    def _layout_series(self):
        available = []
        all_values = []
        all_x = []
        for index, record in enumerate(self._records):
            series = _series_for(record, self._metric_key)
            if series is None or not series.points:
                continue
            points = _smoothed_points(series.points) if self._smoothing else series.points
            if self._x_mode == 'progress':
                first = points[0].epoch
                span = max(1, points[-1].epoch - first)
                values = [((point.epoch - first) * 100.0 / span, point) for point in points]
            else:
                values = [(float(point.epoch), point) for point in points]
            all_x.extend(value[0] for value in values)
            all_values.extend(value[1].value for value in values)
            available.append((record, series, values, QColor(SERIES_COLORS[index % len(SERIES_COLORS)])))
        if not available:
            return [], None

        x_min, x_max = min(all_x), max(all_x)
        if x_min == x_max:
            x_max = x_min + 1.0
        y_min, y_max = min(all_values), max(all_values)
        if y_min == y_max:
            padding = max(abs(y_min) * 0.08, 0.01)
        else:
            padding = (y_max - y_min) * 0.1
        y_min -= padding
        y_max += padding
        return available, (x_min, x_max, y_min, y_max)

    def paintEvent(self, _event):
        option = QStyleOption()
        option.initFrom(self)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        self.style().drawPrimitive(QStyle.PE_Widget, option, painter, self)
        rect = self._plot_rect()
        available, bounds = self._layout_series()
        self._painted_points = []
        self._legend_hits = []

        if not available or bounds is None:
            painter.setPen(QColor('#71879A'))
            painter.drawText(self.rect(), Qt.AlignCenter, '当前模型没有可共同对比的训练曲线')
            return

        x_min, x_max, y_min, y_max = bounds
        painter.setPen(QPen(QColor(88, 126, 153, 55), 1))
        for step in range(6):
            ratio = step / 5
            y = rect.bottom() - ratio * rect.height()
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            value = y_min + ratio * (y_max - y_min)
            painter.setPen(QColor('#71879A'))
            painter.drawText(QRectF(4, y - 9, 54, 18), Qt.AlignRight | Qt.AlignVCenter,
                             _format_value(value))
            painter.setPen(QPen(QColor(88, 126, 153, 55), 1))
        for step in range(6):
            ratio = step / 5
            x = rect.left() + ratio * rect.width()
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            value = x_min + ratio * (x_max - x_min)
            suffix = '%' if self._x_mode == 'progress' else ''
            painter.setPen(QColor('#71879A'))
            painter.drawText(QRectF(x - 30, rect.bottom() + 7, 60, 18), Qt.AlignCenter,
                             f'{value:.0f}{suffix}')
            painter.setPen(QPen(QColor(88, 126, 153, 55), 1))

        legend_width = max(92.0, (self.width() - 34.0) / max(1, len(self._records)))
        for index, record in enumerate(self._records):
            color = QColor(SERIES_COLORS[index % len(SERIES_COLORS)])
            hit = QRectF(14 + index * legend_width, 14, legend_width - 5, 30)
            self._legend_hits.append((hit, record))
            painter.setBrush(color)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(hit.left() + 7, hit.center().y()), 4, 4)
            painter.setPen(QColor('#E3EFF7'))
            text_rect = hit.adjusted(16, 0, -4, 0)
            text = painter.fontMetrics().elidedText(record.name, Qt.ElideRight, int(text_rect.width()))
            painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, text)

        for record, _series, values, color in available:
            dimmed = self._highlighted_id and record.model_id != self._highlighted_id
            color.setAlpha(64 if dimmed else 230)
            painter.setPen(QPen(color, 1.3 if dimmed else 2.4))
            painter.setBrush(Qt.NoBrush)
            path = QPainterPath()
            screen_points = []
            for point_index, (x_value, point) in enumerate(values):
                x_ratio = (x_value - x_min) / (x_max - x_min)
                y_ratio = (point.value - y_min) / (y_max - y_min)
                x = rect.left() + x_ratio * rect.width() * self._progress
                y = rect.bottom() - y_ratio * rect.height()
                screen = QPointF(x, y)
                screen_points.append((screen, point, x_value))
                if point_index == 0:
                    path.moveTo(screen)
                else:
                    path.lineTo(screen)
            painter.drawPath(path)
            if screen_points:
                painter.setBrush(color)
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(screen_points[-1][0], 3.5, 3.5)
            for screen, point, x_value in screen_points:
                self._painted_points.append((screen, record, point, x_value))

        if self._hovered is not None:
            screen, record, point, x_value = self._hovered
            index = self._records.index(record)
            color = QColor(SERIES_COLORS[index % len(SERIES_COLORS)])
            painter.setBrush(QColor('#09131D'))
            painter.setPen(QPen(color, 2))
            painter.drawEllipse(screen, 5, 5)
            x_label = f'{x_value:.1f}%' if self._x_mode == 'progress' else f'epoch {point.epoch}'
            text = f'{record.name}\n{x_label}  ·  {_format_value(point.value)}'
            metrics = painter.fontMetrics()
            width = min(300, max(160, metrics.horizontalAdvance(record.name) + 28))
            tooltip = QRectF(screen.x() + 12, screen.y() - 54, width, 46)
            if tooltip.right() > self.width() - 8:
                tooltip.moveRight(screen.x() - 12)
            if tooltip.top() < 49:
                tooltip.moveTop(screen.y() + 12)
            painter.setPen(QPen(QColor(98, 232, 255, 105), 1))
            painter.setBrush(QColor(8, 19, 29, 242))
            painter.drawRoundedRect(tooltip, 6, 6)
            painter.setPen(QColor('#E7F4FB'))
            painter.drawText(tooltip.adjusted(9, 5, -8, -4), Qt.AlignLeft | Qt.AlignVCenter, text)

    def _nearest_point(self, position, radius=12.0):
        best = None
        best_distance = radius * radius
        for item in self._painted_points:
            screen = item[0]
            distance = (screen.x() - position.x()) ** 2 + (screen.y() - position.y()) ** 2
            if distance < best_distance:
                best = item
                best_distance = distance
        return best

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

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            for rect, record in self._legend_hits:
                if rect.contains(event.pos()):
                    self.model_activated.emit(record)
                    event.accept()
                    return
            nearest = self._nearest_point(event.pos(), 16.0)
            if nearest is not None:
                record = nearest[1]
                self._highlighted_id = (
                    '' if self._highlighted_id == record.model_id else record.model_id
                )
                self.update()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        nearest = self._nearest_point(event.pos(), 18.0)
        if nearest is not None:
            self.model_activated.emit(nearest[1])
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class MetricSummaryBarChart(QWidget):
    """Animated best-value bars; every bar opens its model profile."""

    model_activated = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('modelMetricBarChart')
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMouseTracking(True)
        self.setMinimumSize(320, 220)
        self._records = ()
        self._metric_key = ''
        self._progress = 1.0
        self._bar_hits = []
        self._hovered_id = ''
        self._animation = QVariantAnimation(self)
        self._animation.setDuration(460)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)
        self._animation.valueChanged.connect(self._set_progress)

    def set_data(self, records, metric_key: str):
        self._records = tuple(records)
        self._metric_key = str(metric_key or '')
        self._animation.stop()
        self._animation.setStartValue(0.0)
        self._animation.setEndValue(1.0)
        self._animation.start()

    def _set_progress(self, value):
        self._progress = float(value)
        self.update()

    def paintEvent(self, _event):
        option = QStyleOption()
        option.initFrom(self)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        self.style().drawPrimitive(QStyle.PE_Widget, option, painter, self)
        values = []
        for index, record in enumerate(self._records):
            best = _series_best(_series_for(record, self._metric_key))
            if best is not None:
                values.append((record, best, QColor(SERIES_COLORS[index % len(SERIES_COLORS)])))
        self._bar_hits = []
        if not values:
            painter.setPen(QColor('#71879A'))
            painter.drawText(self.rect(), Qt.AlignCenter, '没有最佳值数据')
            return

        rect = QRectF(36, 28, max(40, self.width() - 54), max(40, self.height() - 67))
        max_value = max(item[1].value for item in values)
        min_value = min(0.0, min(item[1].value for item in values))
        span = max(1e-12, max_value - min_value)
        slot = rect.width() / len(values)
        bar_width = min(54.0, slot * 0.5)
        painter.setPen(QPen(QColor(88, 126, 153, 58), 1))
        painter.drawLine(rect.bottomLeft(), rect.bottomRight())
        for index, (record, point, color) in enumerate(values):
            ratio = (point.value - min_value) / span
            height = max(2.0, ratio * rect.height() * self._progress)
            bar = QRectF(
                rect.left() + slot * index + (slot - bar_width) / 2,
                rect.bottom() - height,
                bar_width,
                height,
            )
            self._bar_hits.append((bar.adjusted(-8, -8, 8, 28), record))
            if record.model_id == self._hovered_id:
                color = color.lighter(125)
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawRoundedRect(bar, 5, 5)
            painter.setPen(QColor('#F0F8FC'))
            painter.drawText(QRectF(bar.left() - 22, bar.top() - 23, bar.width() + 44, 20),
                             Qt.AlignCenter, _format_value(point.value))
            painter.setPen(QColor('#8FA6B8'))
            label_rect = QRectF(rect.left() + slot * index + 3, rect.bottom() + 7, slot - 6, 22)
            label = painter.fontMetrics().elidedText(record.name, Qt.ElideRight,
                                                     int(label_rect.width()))
            painter.drawText(label_rect, Qt.AlignTop | Qt.AlignHCenter, label)

    def mouseMoveEvent(self, event):
        hovered = ''
        for rect, record in self._bar_hits:
            if rect.contains(event.pos()):
                hovered = record.model_id
                break
        if hovered != self._hovered_id:
            self._hovered_id = hovered
            self.setCursor(Qt.PointingHandCursor if hovered else Qt.ArrowCursor)
            self.update()
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            for rect, record in self._bar_hits:
                if rect.contains(event.pos()):
                    self.model_activated.emit(record)
                    event.accept()
                    return
        super().mousePressEvent(event)


class ModelComparisonPage(QWidget):
    """Comparison page shared by model selection and profile navigation."""

    back_requested = pyqtSignal()
    model_requested = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('modelComparisonPage')
        self._records: tuple[ModelRecord, ...] = ()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 16)
        layout.setSpacing(10)

        heading = QHBoxLayout()
        self.btn_back = QToolButton()
        self.btn_back.setObjectName('modelBackBtn')
        self.btn_back.setText('‹')
        self.btn_back.setToolTip('返回模型库')
        self.btn_back.setFixedSize(36, 36)
        self.btn_back.clicked.connect(self.back_requested)
        heading.addWidget(self.btn_back)
        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        eyebrow = QLabel('MODEL COMPARISON')
        eyebrow.setObjectName('modelEyebrow')
        title = QLabel('训练表现对比')
        title.setObjectName('modelComparisonTitle')
        title_box.addWidget(eyebrow)
        title_box.addWidget(title)
        heading.addLayout(title_box, 1)
        self.lbl_task = QLabel('-')
        self.lbl_task.setObjectName('modelComparisonTask')
        heading.addWidget(self.lbl_task)
        self.lbl_count = QLabel('0 MODELS')
        self.lbl_count.setObjectName('modelTotalPill')
        heading.addWidget(self.lbl_count)
        layout.addLayout(heading)

        self.model_strip = QWidget()
        self.model_strip.setObjectName('modelComparisonStrip')
        self.model_strip_layout = QHBoxLayout(self.model_strip)
        self.model_strip_layout.setContentsMargins(8, 7, 8, 7)
        self.model_strip_layout.setSpacing(7)
        layout.addWidget(self.model_strip)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        controls.addWidget(QLabel('指标'))
        self.metric_combo = QComboBox()
        self.metric_combo.setObjectName('modelComparisonCombo')
        self.metric_combo.setMinimumWidth(280)
        self.metric_combo.currentIndexChanged.connect(self._refresh_charts)
        controls.addWidget(self.metric_combo)
        controls.addWidget(QLabel('横轴'))
        self.x_axis_combo = QComboBox()
        self.x_axis_combo.setObjectName('modelComparisonCombo')
        self.x_axis_combo.addItem('训练轮次', 'epoch')
        self.x_axis_combo.addItem('训练进度', 'progress')
        self.x_axis_combo.currentIndexChanged.connect(self._refresh_charts)
        controls.addWidget(self.x_axis_combo)
        self.chk_smoothing = QCheckBox('平滑曲线')
        self.chk_smoothing.setObjectName('modelComparisonSmoothing')
        self.chk_smoothing.toggled.connect(self._refresh_charts)
        controls.addWidget(self.chk_smoothing)
        controls.addStretch()
        self.lbl_dataset_note = QLabel('-')
        self.lbl_dataset_note.setObjectName('modelComparisonDatasetNote')
        controls.addWidget(self.lbl_dataset_note)
        layout.addLayout(controls)

        body = QSplitter(Qt.Vertical)
        body.setObjectName('modelComparisonSplitter')
        body.setChildrenCollapsible(False)
        body.setHandleWidth(7)

        curve_panel = QWidget()
        curve_panel.setObjectName('modelComparisonPanel')
        curve_layout = QVBoxLayout(curve_panel)
        curve_layout.setContentsMargins(10, 9, 10, 10)
        curve_layout.setSpacing(6)
        curve_title = QLabel('逐轮训练曲线')
        curve_title.setObjectName('modelComparisonSectionTitle')
        curve_layout.addWidget(curve_title)
        self.curve_chart = TrainingCurveChart()
        self.curve_chart.model_activated.connect(self.model_requested)
        curve_layout.addWidget(self.curve_chart, 1)
        body.addWidget(curve_panel)

        lower = QSplitter(Qt.Horizontal)
        lower.setObjectName('modelComparisonLowerSplitter')
        lower.setChildrenCollapsible(False)
        lower.setHandleWidth(7)
        bar_panel = QWidget()
        bar_panel.setObjectName('modelComparisonPanel')
        bar_layout = QVBoxLayout(bar_panel)
        bar_layout.setContentsMargins(10, 9, 10, 10)
        bar_title = QLabel('最佳指标')
        bar_title.setObjectName('modelComparisonSectionTitle')
        bar_layout.addWidget(bar_title)
        self.bar_chart = MetricSummaryBarChart()
        self.bar_chart.model_activated.connect(self.model_requested)
        bar_layout.addWidget(self.bar_chart, 1)
        lower.addWidget(bar_panel)

        table_panel = QWidget()
        table_panel.setObjectName('modelComparisonPanel')
        table_layout = QVBoxLayout(table_panel)
        table_layout.setContentsMargins(10, 9, 10, 10)
        table_title = QLabel('配置与结果差异')
        table_title.setObjectName('modelComparisonSectionTitle')
        table_layout.addWidget(table_title)
        self.table = QTreeWidget()
        self.table.setObjectName('modelComparisonTable')
        self.table.setRootIsDecorated(False)
        self.table.setAlternatingRowColors(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setTextElideMode(Qt.ElideRight)
        self.table.header().setSectionResizeMode(QHeaderView.Interactive)
        table_layout.addWidget(self.table, 1)
        lower.addWidget(table_panel)
        lower.setSizes([410, 760])
        body.addWidget(lower)
        body.setSizes([390, 280])
        layout.addWidget(body, 1)

    def set_models(self, records):
        self._records = tuple(records)
        self.lbl_count.setText(f'{len(self._records)} MODELS')
        task_type = self._records[0].task_type if self._records else 'other'
        task_labels = {
            'pose': '姿态估计',
            'detection': '目标检测',
            'segmentation': '语义分割',
            'obb': '旋转框 OBB',
        }
        self.lbl_task.setText(task_labels.get(task_type, task_type))
        self._rebuild_model_strip()
        self._populate_metric_combo()
        self._update_dataset_note()
        self._refresh_charts()

    def _rebuild_model_strip(self):
        while self.model_strip_layout.count():
            item = self.model_strip_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for index, record in enumerate(self._records):
            button = QPushButton(f'{index + 1}  {record.name}  ›')
            button.setObjectName('modelComparisonChip')
            button.setProperty('colorIndex', index)
            button.setToolTip(f'打开模型详情\n{record.path}')
            button.clicked.connect(
                lambda _checked=False, selected=record:
                self.model_requested.emit(selected)
            )
            self.model_strip_layout.addWidget(button)
        self.model_strip_layout.addStretch()

    def _populate_metric_combo(self):
        previous = self.metric_combo.currentData()
        series_by_key = {}
        key_sets = []
        for record in self._records:
            mapping = {series.key: series for series in record.metric_series}
            key_sets.append(set(mapping))
            series_by_key.update(mapping)
        common = set.intersection(*key_sets) if key_sets else set()
        keys = common or set(series_by_key)
        order = {'evaluation': 0, 'loss': 1, 'learning_rate': 2, 'other': 3}
        task_type = self._records[0].task_type if self._records else 'other'
        primary_key = TASK_PRIMARY_SERIES.get(task_type, '')
        keys = sorted(
            keys,
            key=lambda key: (
                0 if key == primary_key else 1,
                order.get(series_by_key[key].category, 9),
                series_by_key[key].label.lower(),
            ),
        )
        self.metric_combo.blockSignals(True)
        self.metric_combo.clear()
        for key in keys:
            series = series_by_key[key]
            suffix = '' if key in common else '  /  部分模型'
            self.metric_combo.addItem(f'{series.label}{suffix}', key)
        if previous:
            index = self.metric_combo.findData(previous)
            if index >= 0:
                self.metric_combo.setCurrentIndex(index)
        self.metric_combo.blockSignals(False)

    def _update_dataset_note(self):
        datasets = set()
        missing = False
        for record in self._records:
            train_sources = [source for source in record.data_sources if source.role == 'train']
            if not train_sources:
                missing = True
            else:
                datasets.update(
                    (source.dataset_name, source.batch_name) for source in train_sources
                )
        warning = missing or len(datasets) > 1
        self.lbl_dataset_note.setProperty('warning', warning)
        if missing:
            note = '部分模型训练数据未解析'
        elif len(datasets) > 1:
            note = '训练数据不同，指标仅供参考'
        else:
            note = '训练数据来源一致'
        self.lbl_dataset_note.setText(note)
        self.lbl_dataset_note.style().unpolish(self.lbl_dataset_note)
        self.lbl_dataset_note.style().polish(self.lbl_dataset_note)

    def _refresh_charts(self, *_args):
        key = self.metric_combo.currentData() or ''
        self.curve_chart.set_data(
            self._records,
            key,
            self.x_axis_combo.currentData() or 'epoch',
            self.chk_smoothing.isChecked(),
        )
        self.bar_chart.set_data(self._records, key)
        self._populate_table(key)

    def _populate_table(self, metric_key: str):
        self.table.clear()
        headers = ['对比项', *[record.name for record in self._records]]
        self.table.setHeaderLabels(headers)
        self.table.setColumnWidth(0, 150)
        for column in range(1, len(headers)):
            self.table.setColumnWidth(column, 190)

        def metric_value(record, part):
            series = _series_for(record, metric_key)
            best = _series_best(series)
            if part == 'best':
                return _format_value(best.value if best else None)
            if part == 'last':
                return _format_value(series.points[-1].value if series and series.points else None)
            return str(best.epoch) if best else '-'

        def train_source(record):
            sources = [source for source in record.data_sources if source.role == 'train']
            if not sources:
                return '-'
            return ' / '.join(
                f'{source.dataset_name}:{source.batch_name}' for source in sources
            )

        rows = (
            ('最佳指标', lambda record: metric_value(record, 'best')),
            ('最终指标', lambda record: metric_value(record, 'last')),
            ('最佳轮次', lambda record: metric_value(record, 'epoch')),
            ('训练轮次', lambda record: f'{record.actual_epochs}/{record.planned_epochs or "-"}'),
            ('训练时长', lambda record: _format_duration(record.training_seconds)),
            ('模型架构', lambda record: record.architecture),
            ('输入尺寸', lambda record: record.input_size),
            ('Batch', lambda record: record.batch_size),
            ('优化器', lambda record: record.optimizer),
            ('训练数据', train_source),
        )
        for label, getter in rows:
            values = [str(getter(record) or '-') for record in self._records]
            item = QTreeWidgetItem([label, *values])
            for column, value in enumerate(values, 1):
                item.setToolTip(column, value)
            self.table.addTopLevelItem(item)
