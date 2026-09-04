"""Right panel: image metadata and annotation JSON tree display.

Per-point checkboxes let the user toggle individual key points on/off.
"""

import json
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import (
    QEasingCurve,
    QPointF,
    QRectF,
    Qt,
    QVariantAnimation,
    pyqtSignal,
)
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPen,
)
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QHBoxLayout,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QToolTip,
    QTreeWidget,
    QTreeWidgetItem,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.models.annotation_review import review_annotation_file
from app.views.ui_effects import HoverGlow


def _format_size(bytes_val: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024:
            return f'{bytes_val:.1f} {unit}'
        bytes_val /= 1024
    return f'{bytes_val:.1f} TB'


def _format_date(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')


def _issue_color(issue) -> QColor:
    if getattr(issue, 'severity', '') == 'warning':
        return QColor('#F5A524')
    return QColor('#FF4D4F')


ISSUE_TITLES = {
    'duplicate_keypoint': '重复关键点',
    'suspected_left_right_swap': '疑似左右反标',
    'missing_person_box': '缺失人框',
    'keypoint_outside_box': '点在人框外',
    'keypoint_wrong_person': '疑似归属错误',
    'group_id_missing': 'group_id缺失',
    'group_id_conflict': 'group_id混乱',
    'empty_annotation': '空标注',
    'invalid_rectangle': '无效矩形框',
    'bbox_outside_image': '目标框越界',
    'bbox_small_area': '检测框过小',
    'bbox_bad_aspect_ratio': '检测框长宽比异常',
    'bbox_duplicate': '疑似重复检测框',
    'invalid_rotation_box': '无效旋转框',
    'obb_outside_image': '旋转框越界',
    'obb_duplicate_points': '旋转框顶点异常',
    'obb_corner_order': '旋转框点序异常',
    'obb_small_area': '旋转框面积过小',
    'obb_bad_aspect_ratio': '旋转框长宽比异常',
    'invalid_polygon': '无效分割多边形',
    'polygon_outside_image': '分割多边形越界',
    'polygon_duplicate_points': '分割多边形顶点异常',
    'polygon_self_intersection': '分割多边形自交',
    'polygon_small_area': '分割多边形面积过小',
    'unknown_class': '未知类别',
    'unexpected_shape_type': '非预期标注类型',
    'image_size_missing': '尺寸字段缺失',
    'image_size_mismatch': '图片尺寸不一致',
    'unavailable_rule': '规则未执行',
    'custom_rule_error': '自定义规则错误',
}


def _issue_title_for_rule(rule: str) -> str:
    return ISSUE_TITLES.get(rule, rule)


def _issue_title(issue) -> str:
    return _issue_title_for_rule(issue.rule)


def _format_issue_counts(issues: list) -> str:
    counts = {}
    for issue in issues:
        title = _issue_title(issue)
        counts[title] = counts.get(title, 0) + 1
    return ', '.join(f'{title}×{count}' for title, count in counts.items())


def _obb_angle_deg(points) -> float:
    """OBB orientation: tilt of the first edge vs horizontal (0-180°)."""
    try:
        import math
        x0, y0 = float(points[0][0]), float(points[0][1])
        x1, y1 = float(points[1][0]), float(points[1][1])
        return math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180.0
    except (TypeError, ValueError, IndexError):
        return 0.0


def _to_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


class _ReviewChartsWidget(QWidget):
    """Folder review chart dashboard: two charts on top, one wide chart below."""

    metric_selected = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(6)

        self.overview_chart = _BarChartWidget(
            '图片 / 标注 / 类别样本', chart_id='overview'
        )
        self.quality_chart = _BarChartWidget(
            '标注质量', chart_id='quality'
        )
        self.keypoint_chart = _BarChartWidget(
            '关键点数量统计', chart_id='keypoints', dense=True
        )
        self._charts = (
            self.overview_chart,
            self.quality_chart,
            self.keypoint_chart,
        )
        self._selected_metric_key = None
        for chart in self._charts:
            chart.bar_activated.connect(
                lambda payload, source=chart: self._on_bar_activated(
                    source, payload
                )
            )

        self.overview_chart.setMinimumHeight(108)
        self.quality_chart.setMinimumHeight(108)
        self.keypoint_chart.setMinimumHeight(148)

        top_row.addWidget(self.overview_chart, 1)
        top_row.addWidget(self.quality_chart, 1)
        layout.addLayout(top_row)
        layout.addWidget(self.keypoint_chart)

    def set_summary(self, total_images: int, summary: dict):
        self.clear_selection(emit=False)
        task_type = str(summary.get('task_type') or 'pose')
        annotation_files = _to_int(summary.get('annotation_files'))
        missing_annotations = _to_int(summary.get('missing_annotations'))
        invalid_annotations = _to_int(summary.get('invalid_annotations'))
        issue_files = _to_int(summary.get('issue_files'))
        manual_pass_files = _to_int(summary.get('manual_pass_files'))

        target_class_counts = summary.get('target_class_counts', {}) or {}
        overview_data = [
            self._bar(
                'overview:images', '图片数量', total_images, '#36B7FF',
                'images', total_images,
                '当前文件夹中的全部图片',
            ),
            self._bar(
                'overview:annotations', 'JSON标注', annotation_files,
                '#45D483', 'annotations', annotation_files,
                '已找到对应 JSON 标注文件的图片',
            ),
        ]
        class_file_counts = summary.get('target_class_file_counts', {}) or {}
        for label, count in target_class_counts.items():
            overview_data.append(self._bar(
                f'class:{label}', label, _to_int(count), '#F5A524',
                'class', _to_int(class_file_counts.get(label)),
                f'{label} 类别的实例与文件覆盖情况',
            ))

        issue_images = min(
            total_images,
            issue_files + missing_annotations + invalid_annotations,
        )
        auto_ok_images = max(
            0, total_images - issue_images - manual_pass_files
        )
        quality_data = [
            self._bar(
                'quality:issue', '待处理', issue_images, '#FF4D4F',
                'quality_issue', issue_images,
                '包含规则问题、缺失标注或无效 JSON 的图片',
            ),
            self._bar(
                'quality:manual', '人工通过', manual_pass_files, '#36CFC9',
                'quality_manual', manual_pass_files,
                '人工确认算法问题为误报的图片',
            ),
            self._bar(
                'quality:ok', '规则通过', auto_ok_images, '#45D483',
                'quality_ok', auto_ok_images,
                '当前已执行规则未发现问题的图片',
            ),
        ]

        keypoint_counts = summary.get('keypoint_counts', {}) or {}
        keypoint_file_counts = summary.get('keypoint_file_counts', {}) or {}
        keypoint_data = []
        if keypoint_counts:
            self.keypoint_chart.set_title('关键点数量统计')
            keypoint_data = [
                self._bar(
                    f'keypoint:{label}', label, _to_int(count), '#36B7FF',
                    'keypoint', _to_int(keypoint_file_counts.get(label)),
                    f'{label} 的实际数量与异常文件',
                    expected=_to_int(summary.get('person_boxes')),
                )
                for label, count in keypoint_counts.items()
            ]
        else:
            shape_type_counts = summary.get('shape_type_counts', {}) or {}
            shape_file_counts = summary.get('shape_type_file_counts', {}) or {}
            if shape_type_counts:
                self.keypoint_chart.set_title('shape 类型统计')
                keypoint_data = [
                    self._bar(
                        f'shape:{label}', label or 'unknown', _to_int(count),
                        '#36B7FF', 'shape_type',
                        _to_int(shape_file_counts.get(label)),
                        f'{label or "unknown"} shape 的实例与文件覆盖情况',
                    )
                    for label, count in shape_type_counts.items()
                ]
            else:
                rule_counts = summary.get('rule_counts', {}) or {}
                self.keypoint_chart.set_title('问题类型统计')
                keypoint_data = [
                    self._bar(
                        f'rule:{rule}', _issue_title_for_rule(rule),
                        _to_int(count), '#F5A524', 'rule',
                        len((summary.get('metric_files', {}) or {}).get(
                            f'rule:{rule}', []
                        )),
                        f'{_issue_title_for_rule(rule)}涉及的文件',
                    )
                    for rule, count in rule_counts.items()
                ]

        self.overview_chart.set_data(overview_data)
        self.quality_chart.set_data(quality_data)
        self.keypoint_chart.set_data(keypoint_data)

        object_name = '人框' if task_type == 'pose' else '目标框'
        self.overview_chart.setToolTip(
            f'图片数量、JSON 标注文件数量，以及各{object_name}类别样本数量。'
        )
        self.quality_chart.setToolTip(
            f'待处理={issue_images}，人工通过={manual_pass_files}，'
            f'规则通过={auto_ok_images}；待处理包含规则问题、缺失标注和无效 JSON。'
        )
        if keypoint_counts:
            self.keypoint_chart.setToolTip(
                '当前模板中每个关键点标签在该文件夹内出现的次数。'
            )
        else:
            self.keypoint_chart.setToolTip(
                '非 Pose 任务显示 shape_type 或问题类型分布。'
            )

    def clear(self):
        self.clear_selection(emit=False)
        self.overview_chart.clear()
        self.quality_chart.clear()
        self.keypoint_chart.clear()

    @staticmethod
    def _bar(key: str, label: str, value: int, color: str, kind: str,
             file_count: int, description: str,
             expected: int | None = None) -> dict:
        result = {
            'key': key,
            'label': label,
            'value': max(0, _to_int(value)),
            'color': color,
            'kind': kind,
            'file_count': max(0, _to_int(file_count)),
            'description': description,
        }
        if expected is not None:
            result['expected'] = max(0, _to_int(expected))
        return result

    def _on_bar_activated(self, source, payload: dict):
        key = payload.get('key')
        if key == self._selected_metric_key:
            self.clear_selection(emit=True)
            return
        self._selected_metric_key = key
        for chart in self._charts:
            chart.set_selected_key(key if chart is source else None)
        self.metric_selected.emit(dict(payload))

    def clear_selection(self, emit: bool = False):
        self._selected_metric_key = None
        for chart in getattr(self, '_charts', ()):
            chart.set_selected_key(None)
        if emit:
            self.metric_selected.emit(None)


class _BarChartWidget(QWidget):
    """Animated, keyboard-accessible bar chart painted with Qt."""

    bar_activated = pyqtSignal(object)

    def __init__(self, title: str, dense: bool = False, parent=None,
                 chart_id: str = ''):
        super().__init__(parent)
        self._title = title
        self._chart_id = chart_id
        self._dense = dense
        self._data: list[dict] = []
        self._bar_hit_rects: list[QRectF] = []
        self._animation_progress = 1.0
        self._hovered_index = -1
        self._keyboard_index = -1
        self._selected_key = None
        self._animation = QVariantAnimation(self)
        self._animation.setDuration(580 if dense else 480)
        self._animation.setStartValue(0.0)
        self._animation.setEndValue(1.0)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)
        self._animation.valueChanged.connect(self._set_animation_progress)
        self.setMinimumHeight(84)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName(title)

    def set_title(self, title: str):
        if self._title != title:
            self._title = title
            self.update()

    def set_data(self, data: list[dict]):
        normalized = []
        for item in data:
            entry = dict(item)
            entry['key'] = str(entry.get('key') or entry.get('label') or '')
            entry['label'] = str(entry.get('label') or '')
            entry['value'] = max(0, _to_int(entry.get('value')))
            entry['color'] = QColor(entry.get('color') or '#36B7FF')
            normalized.append(entry)
        self._data = normalized
        self._hovered_index = -1
        self._keyboard_index = 0 if self._data else -1
        if self._selected_key not in {
            item.get('key') for item in self._data
        }:
            self._selected_key = None
        self._animation.stop()
        self._animation_progress = 0.0
        self._animation.start()
        self.update()

    def clear(self):
        self._animation.stop()
        self._data = []
        self._bar_hit_rects = []
        self._hovered_index = -1
        self._keyboard_index = -1
        self._selected_key = None
        self._animation_progress = 1.0
        QToolTip.hideText()
        self.update()

    def set_selected_key(self, key):
        if self._selected_key != key:
            self._selected_key = key
            self.update()

    def _set_animation_progress(self, value):
        self._animation_progress = float(value)
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        area = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        painter.setPen(QPen(QColor('#314154'), 1))
        painter.setBrush(QBrush(QColor('#0D141C')))
        painter.drawRoundedRect(area, 8, 8)

        title_font = QFont(painter.font())
        title_font.setBold(True)
        title_font.setPointSize(9)
        painter.setFont(title_font)
        painter.setPen(QColor('#DDF2FF'))
        painter.drawText(QRectF(10, 7, self.width() - 20, 18), Qt.AlignLeft, self._title)

        if not self._data:
            painter.setPen(QColor('#7F8DA3'))
            painter.drawText(area, Qt.AlignCenter, '点击统计后显示图表')
            painter.end()
            return

        # Dense keypoint labels are rotated; reserve enough room so the
        # splitter never clips their lower edge at compact panel heights.
        bottom_margin = 70 if self._dense else 42
        left_margin = 34
        right_margin = 10
        top_margin = 34
        chart = QRectF(
            left_margin,
            top_margin,
            max(20, self.width() - left_margin - right_margin),
            max(28, self.height() - top_margin - bottom_margin),
        )
        max_value = max(max(item['value'] for item in self._data), 1)

        grid_pen = QPen(QColor('#223041'), 1)
        painter.setPen(grid_pen)
        for ratio in (0.25, 0.5, 0.75, 1.0):
            y = chart.bottom() - chart.height() * ratio
            painter.drawLine(QPointF(chart.left(), y), QPointF(chart.right(), y))

        label_font = QFont(painter.font())
        label_font.setBold(False)
        label_font.setPointSize(7 if self._dense else 8)
        painter.setFont(label_font)
        painter.setPen(QColor('#8EA0B6'))
        painter.drawText(
            QRectF(4, chart.bottom() - 8, left_margin - 8, 14),
            Qt.AlignRight | Qt.AlignVCenter,
            '0',
        )

        count = len(self._data)
        slot = chart.width() / max(count, 1)
        bar_width = max(4.0, min(slot * 0.62, 28.0 if self._dense else 44.0))
        self._bar_hit_rects = []
        stagger = min(0.018, 0.24 / max(1, count - 1))

        for idx, item in enumerate(self._data):
            label = item['label']
            value = item['value']
            color = QColor(item['color'])
            center_x = chart.left() + slot * idx + slot / 2
            height_ratio = value / max_value if max_value else 0
            delay = idx * stagger
            local_progress = min(
                1.0,
                max(0.0, (self._animation_progress - delay) / max(0.01, 1.0 - delay)),
            )
            target_height = chart.height() * height_ratio
            bar_h = max(
                2.0 if value > 0 and local_progress > 0 else 0.0,
                target_height * local_progress,
            )
            rect = QRectF(
                center_x - bar_width / 2,
                chart.bottom() - bar_h,
                bar_width,
                bar_h,
            )
            self._bar_hit_rects.append(QRectF(
                center_x - slot / 2,
                chart.top(),
                slot,
                chart.height() + bottom_margin - 8,
            ))

            selected = item.get('key') == self._selected_key
            hovered = idx == self._hovered_index
            keyboard_focus = self.hasFocus() and idx == self._keyboard_index
            if self._selected_key and not selected:
                color.setAlpha(92)
            if selected or hovered:
                glow = QColor(color)
                glow.setAlpha(48 if selected else 32)
                painter.setPen(Qt.NoPen)
                painter.setBrush(glow)
                painter.drawRoundedRect(rect.adjusted(-4, -4, 4, 2), 5, 5)
                color = color.lighter(118 if selected else 110)

            gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
            gradient.setColorAt(0.0, color.lighter(122))
            gradient.setColorAt(1.0, color.darker(108))
            if selected:
                painter.setPen(QPen(QColor('#9EE7FF'), 1.5))
            elif keyboard_focus:
                painter.setPen(QPen(QColor('#DDF2FF'), 1, Qt.DotLine))
            else:
                painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(gradient))
            draw_rect = rect.adjusted(
                -1 if hovered else 0,
                -2 if hovered else 0,
                1 if hovered else 0,
                0,
            )
            painter.drawRoundedRect(draw_rect, 3, 3)

            value_font = QFont(label_font)
            value_font.setPointSize(7)
            value_font.setBold(True)
            painter.setFont(value_font)
            value_text = str(round(value * local_progress))
            if value > 0 and bar_h >= 18:
                text_rect = QRectF(rect.left() - 4, rect.top() + 2, rect.width() + 8, 14)
                painter.setPen(QColor('#FFFFFF'))
            else:
                text_rect = QRectF(
                    center_x - slot / 2,
                    max(chart.top(), rect.top() - 14),
                    slot,
                    12,
                )
                painter.setPen(QColor('#D8E2EF'))
            painter.drawText(
                text_rect,
                Qt.AlignCenter,
                value_text,
            )

            painter.setFont(label_font)
            painter.setPen(QColor('#AAB7C8'))
            if self._dense:
                painter.save()
                painter.translate(center_x - 2, chart.bottom() + 8)
                painter.rotate(-62)
                painter.drawText(QRectF(-72, -8, 72, 16), Qt.AlignRight, label)
                painter.restore()
            else:
                elided = painter.fontMetrics().elidedText(
                    label, Qt.ElideRight, max(20, int(slot - 4))
                )
                painter.drawText(
                    QRectF(center_x - slot / 2, chart.bottom() + 6, slot, 30),
                    Qt.AlignHCenter | Qt.AlignTop,
                    elided,
                )

        painter.end()

    def _bar_index_at(self, pos) -> int:
        point = QPointF(pos)
        for idx, rect in enumerate(self._bar_hit_rects):
            if rect.contains(point):
                return idx
        return -1

    def _tooltip_for(self, item: dict) -> str:
        lines = [f"{item.get('label', '-')}", f"数量: {item.get('value', 0)}"]
        file_count = _to_int(item.get('file_count'))
        if file_count:
            lines.append(f'涉及文件: {file_count}')
        expected = item.get('expected')
        if expected is not None:
            lines.append(f'参考期望: {_to_int(expected)}')
        description = str(item.get('description') or '')
        if description:
            lines.append(description)
        lines.append('点击查看指标明细')
        return '\n'.join(lines)

    def mouseMoveEvent(self, event):
        index = self._bar_index_at(event.pos())
        if index != self._hovered_index:
            self._hovered_index = index
            self.setCursor(
                Qt.PointingHandCursor if index >= 0 else Qt.ArrowCursor
            )
            if index >= 0:
                QToolTip.showText(
                    event.globalPos(), self._tooltip_for(self._data[index]), self
                )
            else:
                QToolTip.hideText()
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self._hovered_index = -1
        self.setCursor(Qt.ArrowCursor)
        QToolTip.hideText()
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            index = self._bar_index_at(event.pos())
            if index >= 0:
                self._keyboard_index = index
                self.setFocus(Qt.MouseFocusReason)
                payload = dict(self._data[index])
                payload['chart_id'] = self._chart_id
                payload['chart_title'] = self._title
                self.bar_activated.emit(payload)
                event.accept()
                return
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        if not self._data:
            super().keyPressEvent(event)
            return
        if event.key() in {Qt.Key_Left, Qt.Key_Up}:
            self._keyboard_index = (self._keyboard_index - 1) % len(self._data)
            self.update()
            event.accept()
            return
        if event.key() in {Qt.Key_Right, Qt.Key_Down}:
            self._keyboard_index = (self._keyboard_index + 1) % len(self._data)
            self.update()
            event.accept()
            return
        if event.key() in {Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space}:
            index = max(0, self._keyboard_index)
            payload = dict(self._data[index])
            payload['chart_id'] = self._chart_id
            payload['chart_title'] = self._title
            self.bar_activated.emit(payload)
            event.accept()
            return
        super().keyPressEvent(event)


class DetailPanel(QWidget):
    """Right-side panel showing file metadata and annotation tree with point controls."""

    point_toggled = pyqtSignal(int, int, bool)  # shape_idx, point_idx, visible
    all_points_toggled = pyqtSignal(bool)       # visible
    review_issue_selected = pyqtSignal(object, object)  # shape_indices, point_indices
    review_stats_requested = pyqtSignal()
    review_file_selected = pyqtSignal(int)
    manual_accept_current_requested = pyqtSignal()
    manual_ignore_issue_requested = pyqtSignal(object)
    manual_restore_current_requested = pyqtSignal()
    folder_keypoints_reorder_requested = pyqtSignal()
    pose_config_import_requested = pyqtSignal()
    pose_config_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('detailPanel')
        self._review_issues = []
        self._review_raw_issues = []
        self._review_accepted_issues = []
        self._review_decision_stale = False
        self._review_folder_summary = {}
        self._review_total_images = 0
        self._review_stats_rows = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(7, 7, 7, 7)
        layout.setSpacing(7)

        # Image info group
        info_group = QGroupBox('图片信息')
        info_group.setObjectName('imageInfoGroup')
        info_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self._info_layout = QGridLayout(info_group)
        self._info_layout.setContentsMargins(12, 13, 12, 9)
        self._info_layout.setHorizontalSpacing(8)
        self._info_layout.setVerticalSpacing(6)

        self.lbl_filename = QLabel('-')
        self.lbl_filename.setObjectName('imageInfoFilename')
        self.lbl_filename.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_dimensions = QLabel('-')
        self.lbl_filesize = QLabel('-')
        self.lbl_modified = QLabel('-')
        self.lbl_annotation = QLabel('-')
        for label in (
            self.lbl_dimensions, self.lbl_filesize, self.lbl_modified,
        ):
            label.setObjectName('imageInfoValue')
        self.lbl_annotation.setObjectName('imageInfoPath')
        self.lbl_annotation.setWordWrap(False)
        self.lbl_annotation.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred
        )
        self.lbl_annotation.setTextInteractionFlags(Qt.TextSelectableByMouse)

        info_labels = []
        for text in ('当前文件', '尺寸', '大小', '修改时间', '标注文件'):
            label = QLabel(text)
            label.setObjectName('imageInfoCaption')
            info_labels.append(label)
        self._info_layout.addWidget(info_labels[0], 0, 0)
        self._info_layout.addWidget(self.lbl_filename, 0, 1, 1, 5)
        self._info_layout.addWidget(info_labels[1], 1, 0)
        self._info_layout.addWidget(self.lbl_dimensions, 1, 1)
        self._info_layout.addWidget(info_labels[2], 1, 2)
        self._info_layout.addWidget(self.lbl_filesize, 1, 3)
        self._info_layout.addWidget(info_labels[3], 1, 4)
        self._info_layout.addWidget(self.lbl_modified, 1, 5)
        self._info_layout.addWidget(info_labels[4], 2, 0)
        self._info_layout.addWidget(self.lbl_annotation, 2, 1, 1, 5)
        self._info_layout.setColumnStretch(1, 1)
        self._info_layout.setColumnStretch(3, 1)
        self._info_layout.setColumnStretch(5, 2)

        layout.addWidget(info_group)

        # Annotation review + tree group
        ann_group = QGroupBox('标注审查')
        ann_group.setObjectName('annotationReviewGroup')
        ann_layout = QVBoxLayout(ann_group)
        ann_layout.setContentsMargins(8, 8, 8, 8)
        ann_layout.setSpacing(8)

        self.tabs = QTabWidget()
        self.tabs.setObjectName('annotationTabs')
        ann_layout.addWidget(self.tabs)

        review_tab = QWidget()
        review_layout = QVBoxLayout(review_tab)
        review_layout.setContentsMargins(6, 6, 6, 6)
        review_layout.setSpacing(6)

        # Keep the state available for logic/tests without duplicating the
        # same issue summary above the detailed issue list.
        self.lbl_review_summary = QLabel('未加载标注', review_tab)
        self.lbl_review_summary.setObjectName('reviewSummary')
        self.lbl_review_summary.setProperty('tone', 'neutral')
        self.lbl_review_summary.setWordWrap(True)
        self.lbl_review_summary.hide()

        self.review_config_bar = QWidget()
        self.review_config_bar.setObjectName('reviewConfigBar')
        review_config_layout = QVBoxLayout(self.review_config_bar)
        review_config_layout.setContentsMargins(8, 7, 8, 7)
        review_config_layout.setSpacing(4)
        config_row = QHBoxLayout()
        config_row.setSpacing(6)
        self.lbl_pose_config = QLabel('当前任务模板:')
        self.lbl_pose_config.setObjectName('reviewStats')
        self.pose_config_combo = QComboBox()
        self.pose_config_combo.setMinimumHeight(28)
        self.pose_config_combo.setMaximumHeight(32)
        self.pose_config_combo.currentIndexChanged.connect(
            self._on_pose_config_combo_changed
        )
        self.btn_import_pose_config = QPushButton('高级配置')
        self.btn_import_pose_config.setObjectName('toggleBtn')
        self.btn_import_pose_config.setMinimumHeight(28)
        self.btn_import_pose_config.setMaximumHeight(32)
        self.btn_import_pose_config.clicked.connect(
            lambda _checked=False: self.pose_config_import_requested.emit()
        )
        config_row.addWidget(self.lbl_pose_config)
        config_row.addWidget(self.pose_config_combo, 1)
        config_row.addWidget(self.btn_import_pose_config)
        review_config_layout.addLayout(config_row)

        self.lbl_task_context = QLabel('任务: pose | 标注集: annotations')
        self.lbl_task_context.setObjectName('reviewStats')
        self.lbl_task_context.setWordWrap(True)
        review_config_layout.addWidget(self.lbl_task_context)
        review_layout.addWidget(self.review_config_bar)

        self.review_analysis_panel = QWidget()
        self.review_analysis_panel.setObjectName('analysisPanel')
        analysis_layout = QVBoxLayout(self.review_analysis_panel)
        analysis_layout.setContentsMargins(0, 0, 0, 0)
        analysis_layout.setSpacing(6)

        analysis_mode_row = QHBoxLayout()
        analysis_mode_row.setContentsMargins(0, 0, 0, 0)
        analysis_mode_row.setSpacing(6)
        self.lbl_analysis_mode = QLabel('分析视图:')
        self.lbl_analysis_mode.setObjectName('reviewStats')
        self.btn_analysis_charts = QPushButton('图表分析')
        self.btn_analysis_text = QPushButton('文字描述')
        for button in (self.btn_analysis_charts, self.btn_analysis_text):
            button.setObjectName('toggleBtn')
            button.setCheckable(True)
            button.setMinimumHeight(26)
            button.setMaximumHeight(30)
        self._analysis_mode_group = QButtonGroup(self)
        self._analysis_mode_group.setExclusive(True)
        self._analysis_mode_group.addButton(self.btn_analysis_charts, 0)
        self._analysis_mode_group.addButton(self.btn_analysis_text, 1)
        self.btn_analysis_charts.clicked.connect(lambda: self._set_analysis_mode(0))
        self.btn_analysis_text.clicked.connect(lambda: self._set_analysis_mode(1))
        analysis_mode_row.addWidget(self.lbl_analysis_mode)
        analysis_mode_row.addWidget(self.btn_analysis_charts)
        analysis_mode_row.addWidget(self.btn_analysis_text)
        analysis_mode_row.addStretch()
        analysis_layout.addLayout(analysis_mode_row)

        self.review_charts = _ReviewChartsWidget()
        self.review_charts.metric_selected.connect(
            self._on_chart_metric_selected
        )
        self.review_text_report = QTextEdit()
        self.review_text_report.setObjectName('reviewTextReport')
        self.review_text_report.setReadOnly(True)
        self._set_review_markdown('点击 **统计当前文件夹** 生成文字分析。')

        self.review_analysis_stack = QStackedWidget()
        self.review_analysis_stack.setMinimumHeight(280)
        self.review_analysis_stack.addWidget(self.review_charts)
        self.review_analysis_stack.addWidget(self.review_text_report)
        analysis_layout.addWidget(self.review_analysis_stack)
        self._set_analysis_mode(0)

        self.review_analysis_panel.setVisible(False)

        self.review_tree = QTreeWidget()
        self.review_tree.setObjectName('currentIssueTree')
        self.review_tree.setHeaderLabels(['审查项', '详情'])
        self.review_tree.setRootIsDecorated(False)
        self.review_tree.setMinimumHeight(92)
        self.review_tree.itemClicked.connect(self._on_review_item_clicked)
        self.review_tree.itemSelectionChanged.connect(
            self._update_current_review_actions
        )
        self.review_tree.setVisible(False)

        self.review_issue_panel = QWidget()
        self.review_issue_panel.setObjectName('currentIssuePanel')
        current_issue_layout = QVBoxLayout(self.review_issue_panel)
        current_issue_layout.setContentsMargins(7, 7, 7, 7)
        current_issue_layout.setSpacing(5)
        current_issue_toolbar = QHBoxLayout()
        current_issue_toolbar.setContentsMargins(0, 0, 0, 0)
        current_issue_toolbar.setSpacing(6)
        self.btn_manual_accept_current = QPushButton('人工通过当前文件')
        self.btn_manual_ignore_issue = QPushButton('忽略选中问题')
        self.btn_manual_restore_current = QPushButton('撤销人工结论')
        self.btn_manual_accept_current.setObjectName('successBtn')
        self.btn_manual_ignore_issue.setObjectName('toggleBtn')
        self.btn_manual_restore_current.setObjectName('toggleBtn')
        self.btn_manual_accept_current.setProperty('reviewAction', 'accept')
        self.btn_manual_ignore_issue.setProperty('reviewAction', 'ignore')
        self.btn_manual_restore_current.setProperty('reviewAction', 'restore')
        for button in (
            self.btn_manual_accept_current,
            self.btn_manual_ignore_issue,
            self.btn_manual_restore_current,
        ):
            button.setMinimumHeight(26)
            button.setMaximumHeight(30)
        self.btn_manual_accept_current.clicked.connect(
            lambda _checked=False: self.manual_accept_current_requested.emit()
        )
        self.btn_manual_ignore_issue.clicked.connect(
            self._emit_ignore_selected_issue
        )
        self.btn_manual_restore_current.clicked.connect(
            lambda _checked=False: self.manual_restore_current_requested.emit()
        )
        current_issue_toolbar.addWidget(self.btn_manual_accept_current)
        current_issue_toolbar.addWidget(self.btn_manual_ignore_issue)
        current_issue_toolbar.addWidget(self.btn_manual_restore_current)
        current_issue_toolbar.addStretch()
        current_issue_layout.addLayout(current_issue_toolbar)
        current_issue_layout.addWidget(self.review_tree, 1)
        self.review_issue_panel.setVisible(False)

        self.review_stats_bar = QWidget()
        self.review_stats_bar.setObjectName('reviewStatsBar')
        stats_row = QHBoxLayout(self.review_stats_bar)
        stats_row.setContentsMargins(7, 6, 7, 6)
        stats_row.setSpacing(6)
        self.btn_review_stats = QPushButton('统计当前文件夹')
        self.btn_review_stats.setObjectName('primaryBtn')
        self.btn_review_stats.setMinimumHeight(28)
        self.btn_review_stats.setMaximumHeight(32)
        self.btn_review_stats.clicked.connect(
            lambda _checked=False: self.review_stats_requested.emit()
        )
        self.btn_reorder_folder_keypoints = QPushButton('重排当前文件夹')
        self.btn_reorder_folder_keypoints.setObjectName('successBtn')
        self.btn_reorder_folder_keypoints.setMinimumHeight(28)
        self.btn_reorder_folder_keypoints.setMaximumHeight(32)
        self.btn_reorder_folder_keypoints.setEnabled(False)
        self.btn_reorder_folder_keypoints.clicked.connect(
            lambda _checked=False: self.folder_keypoints_reorder_requested.emit()
        )
        self.lbl_review_stats = QLabel('目录统计: -')
        self.lbl_review_stats.setObjectName('reviewDirectorySummary')
        self.lbl_review_stats.setProperty('tone', 'neutral')
        self.lbl_review_stats.setWordWrap(True)
        self.lbl_review_stats.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        stats_row.addWidget(self.btn_review_stats)
        stats_row.addWidget(self.btn_reorder_folder_keypoints)
        stats_row.addWidget(self.lbl_review_stats, 1)
        review_layout.addWidget(self.review_stats_bar)

        self.review_stats_tree = QTreeWidget()
        self.review_stats_tree.setObjectName('reviewProblemTree')
        self.review_stats_tree.setHeaderLabels(['文件', '问题'])
        self.review_stats_tree.setRootIsDecorated(False)
        self.review_stats_tree.itemClicked.connect(
            self._on_review_stats_item_activated
        )

        self.problem_files_page = QWidget()
        problem_files_layout = QVBoxLayout(self.problem_files_page)
        problem_files_layout.setContentsMargins(4, 4, 4, 4)
        problem_files_layout.setSpacing(5)
        problem_search_row = QHBoxLayout()
        problem_search_row.setContentsMargins(0, 0, 0, 0)
        problem_search_row.setSpacing(6)
        self.review_file_search = QLineEdit()
        self.review_file_search.setObjectName('reviewFileSearch')
        self.review_file_search.setPlaceholderText('搜索文件名或问题，空格分隔多个关键词')
        self.review_file_search.setClearButtonEnabled(True)
        self.review_file_search.setEnabled(False)
        self.review_file_search.textChanged.connect(
            self._filter_review_stats_rows
        )
        self.review_file_search.returnPressed.connect(
            self._open_first_review_search_result
        )
        self.lbl_review_search_count = QLabel('共 0 项')
        self.lbl_review_search_count.setObjectName('reviewFileSearchCount')
        self.lbl_review_search_count.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        problem_search_row.addWidget(self.review_file_search, 1)
        problem_search_row.addWidget(self.lbl_review_search_count)
        problem_files_layout.addLayout(problem_search_row)
        problem_files_layout.addWidget(self.review_stats_tree, 1)

        self.metric_detail_page = QWidget()
        metric_detail_layout = QVBoxLayout(self.metric_detail_page)
        metric_detail_layout.setContentsMargins(4, 4, 4, 4)
        metric_detail_layout.setSpacing(5)
        self.lbl_metric_detail = QLabel('选择图表中的柱状查看文件明细')
        self.lbl_metric_detail.setObjectName('metricDetailSummary')
        self.lbl_metric_detail.setWordWrap(True)
        metric_detail_layout.addWidget(self.lbl_metric_detail)
        metric_search_row = QHBoxLayout()
        metric_search_row.setContentsMargins(0, 0, 0, 0)
        metric_search_row.setSpacing(6)
        self.metric_detail_search = QLineEdit()
        self.metric_detail_search.setObjectName('metricDetailSearch')
        self.metric_detail_search.setPlaceholderText(
            '搜索文件名、数量状态或说明'
        )
        self.metric_detail_search.setClearButtonEnabled(True)
        self.metric_detail_search.setEnabled(False)
        self.metric_detail_search.textChanged.connect(
            self._filter_metric_detail_rows
        )
        self.metric_detail_search.returnPressed.connect(
            self._open_first_metric_search_result
        )
        self.lbl_metric_search_count = QLabel('共 0 项')
        self.lbl_metric_search_count.setObjectName('metricDetailSearchCount')
        self.lbl_metric_search_count.setAlignment(
            Qt.AlignRight | Qt.AlignVCenter
        )
        metric_search_row.addWidget(self.metric_detail_search, 1)
        metric_search_row.addWidget(self.lbl_metric_search_count)
        metric_detail_layout.addLayout(metric_search_row)
        self.metric_detail_tree = QTreeWidget()
        self.metric_detail_tree.setObjectName('metricDetailTree')
        self.metric_detail_tree.setHeaderLabels(['文件', '数量 / 状态', '说明'])
        self.metric_detail_tree.setRootIsDecorated(False)
        self.metric_detail_tree.itemDoubleClicked.connect(
            self._on_review_stats_item_activated
        )
        metric_detail_layout.addWidget(self.metric_detail_tree, 1)

        # 信息架构减法：双 tab（审查结果 / 问题文件 / 指标明细）合并为
        # 标注审查组内的单一 QTabWidget；拆分分栏（analysisSplitter）移除。
        self.review_results_tabs = None
        self._review_stats_splitter_initialized = True  # 不再有分栏要初始化
        self.review_stats_splitter = None

        self.review_body_splitter = QSplitter(Qt.Vertical)
        self.review_body_splitter.setObjectName('reviewBodySplitter')
        self.review_body_splitter.setChildrenCollapsible(False)
        self.review_body_splitter.setHandleWidth(6)
        self.review_body_splitter.addWidget(self.review_analysis_panel)
        self.review_body_splitter.addWidget(self.review_issue_panel)
        self.review_body_splitter.setStretchFactor(0, 1)
        self.review_body_splitter.setStretchFactor(1, 0)
        self.review_body_splitter.setSizes([620, 145])
        self._review_body_splitter_initialized = False
        review_layout.addWidget(self.review_body_splitter, 2)
        self.tabs.addTab(review_tab, '审查')
        self.tabs.addTab(self.problem_files_page, '问题文件')
        self.tabs.addTab(self.metric_detail_page, '指标明细')
        self.tabs.setTabEnabled(2, False)

        tree_tab = QWidget()
        tree_layout = QVBoxLayout(tree_tab)
        tree_layout.setContentsMargins(6, 6, 6, 6)
        tree_layout.setSpacing(6)

        # All-points toggle buttons
        pts_bar = QWidget()
        pts_bar.setObjectName('inlineToolbar')
        pts_bar.setFixedHeight(38)
        pts_toggle = QHBoxLayout(pts_bar)
        pts_toggle.setContentsMargins(4, 4, 4, 4)
        pts_toggle.setSpacing(6)
        self.btn_all_pts_on = QPushButton('☑ 全部显示')
        self.btn_all_pts_off = QPushButton('☐ 全部隐藏')
        self.btn_all_pts_on.setObjectName('toggleBtn')
        self.btn_all_pts_off.setObjectName('toggleBtn')
        self.btn_all_pts_on.setMinimumHeight(28)
        self.btn_all_pts_on.setMaximumHeight(30)
        self.btn_all_pts_off.setMinimumHeight(28)
        self.btn_all_pts_off.setMaximumHeight(30)
        self.btn_all_pts_on.clicked.connect(lambda: self.all_points_toggled.emit(True))
        self.btn_all_pts_off.clicked.connect(lambda: self.all_points_toggled.emit(False))
        pts_toggle.addWidget(self.btn_all_pts_on)
        pts_toggle.addWidget(self.btn_all_pts_off)
        pts_toggle.addStretch()
        tree_layout.addWidget(pts_bar, 0)

        self.ann_tree = QTreeWidget()
        self.ann_tree.setHeaderLabels(['字段', '值'])
        self.ann_tree.setAlternatingRowColors(False)
        tree_layout.addWidget(self.ann_tree, 1)
        self.tabs.addTab(tree_tab, '标注树')

        file_tab = QWidget()
        self._file_list_container = QVBoxLayout(file_tab)
        self._file_list_container.setContentsMargins(6, 6, 6, 6)
        self._file_list_container.setSpacing(6)
        self.tabs.addTab(file_tab, '文件')

        layout.addWidget(ann_group)

        self._hover_glow = HoverGlow(self)
        self._hover_glow.watch_buttons(self)

    def set_file_list(self, file_list_panel):
        """Embed the file list panel below the annotation tree."""
        self._file_list_container.addWidget(file_list_panel)
        self._hover_glow.watch_buttons(file_list_panel)

    def clear(self):
        """Reset the panel to default empty state."""
        self.lbl_filename.setText('-')
        self.lbl_dimensions.setText('-')
        self.lbl_filesize.setText('-')
        self.lbl_modified.setText('-')
        self.lbl_annotation.setText('-')
        self.btn_reorder_folder_keypoints.setEnabled(False)
        self._clear_review()
        self._clear_review_stats()
        self.ann_tree.clear()

    def clear_review_stats(self):
        """Clear folder-level review statistics."""
        self._clear_review_stats()

    def set_pose_review_templates(self, templates: list[dict], active_id: str):
        """Refresh the current task review template list."""
        self.pose_config_combo.blockSignals(True)
        self.pose_config_combo.clear()

        active_index = 0
        for idx, template in enumerate(templates):
            template_id = str(template.get('id', ''))
            name = str(template.get('name', template_id or '-'))
            source = str(template.get('path') or template.get('source') or name)
            task_type = str(template.get('task_type') or '')
            annotation_dir = str(template.get('annotation_dir') or '')
            if task_type or annotation_dir:
                source = (
                    f'任务: {task_type or "-"} | 标注集: {annotation_dir or "-"}\n'
                    f'{source}'
                )
            self.pose_config_combo.addItem(name, template_id)
            self.pose_config_combo.setItemData(idx, source, Qt.ToolTipRole)
            if template_id == active_id:
                active_index = idx

        if self.pose_config_combo.count() > 0:
            self.pose_config_combo.setCurrentIndex(active_index)
            self.pose_config_combo.setToolTip(
                self.pose_config_combo.itemData(active_index, Qt.ToolTipRole) or ''
            )
        self.pose_config_combo.blockSignals(False)

    def set_pose_review_config(self, name: str,
                               path: str | Path | None = None,
                               task_type: str = 'pose',
                               annotation_dir: str = 'annotations'):
        """Show the active external review template."""
        self.pose_config_combo.setToolTip(str(path or name or ''))
        self.lbl_task_context.setText(
            f'任务: {task_type or "-"} | 标注集: {annotation_dir or "-"}'
        )

    def _on_pose_config_combo_changed(self, index: int):
        template_id = self.pose_config_combo.itemData(index)
        self.pose_config_combo.setToolTip(
            self.pose_config_combo.itemData(index, Qt.ToolTipRole) or ''
        )
        if template_id:
            self.pose_config_selected.emit(str(template_id))

    def show_image(self, image_path: str | Path,
                   annotation_path: str | Path | None,
                   expected_annotation_path: str | Path | None = None,
                   task_type: str = 'pose',
                   annotation_dir: str = 'annotations',
                   review_result=None):
        """Update the panel for the given image."""
        img = Path(image_path)
        self.lbl_filename.setText(img.name)

        # Use Pillow for dimensions
        try:
            from PIL import Image
            with Image.open(img) as im:
                w, h = im.size
            self.lbl_dimensions.setText(f'{w} × {h}')
        except Exception:
            self.lbl_dimensions.setText('-')

        if img.is_file():
            stat = img.stat()
            self.lbl_filesize.setText(_format_size(stat.st_size))
            self.lbl_modified.setText(_format_date(stat.st_mtime))
        else:
            self.lbl_filesize.setText('-')
            self.lbl_modified.setText('-')

        # Annotation — show actual path
        if annotation_path and Path(annotation_path).is_file():
            ann = Path(annotation_path)
            # Show relative path if possible, otherwise full path
            self.lbl_annotation.setText(str(ann))
            self.lbl_annotation.setToolTip(str(ann))
            self.btn_reorder_folder_keypoints.setEnabled(task_type == 'pose')
            self._load_review(annotation_path, img, review_result)
            self._load_annotation_tree(annotation_path)
        else:
            expected = (
                Path(expected_annotation_path)
                if expected_annotation_path is not None
                else None
            )
            if expected is not None and not expected.parent.is_dir():
                text = f'当前任务没有对应标注集: {annotation_dir}'
                tooltip = str(expected.parent)
            elif expected is not None:
                text = f'当前任务标注缺失: {expected.name}'
                tooltip = str(expected)
            else:
                text = '（无）'
                tooltip = ''
            self.lbl_annotation.setText(text)
            self.lbl_annotation.setToolTip(tooltip)
            self.btn_reorder_folder_keypoints.setEnabled(False)
            self._clear_review()
            self.ann_tree.clear()

    def _clear_review(self):
        """Clear annotation review status and issue list."""
        self._review_issues = []
        self._review_raw_issues = []
        self._review_accepted_issues = []
        self._review_decision_stale = False
        self._set_review_summary('未加载标注', 'neutral')
        self.review_tree.clear()
        self.review_tree.setVisible(False)
        self.review_issue_panel.setVisible(False)
        self._update_current_review_actions()
        self._review_body_splitter_initialized = False

    def _load_review(self, ann_path: str | Path,
                     image_path: str | Path | None = None,
                     review_result=None):
        """Populate the review issue list for the current annotation."""
        self.review_tree.clear()
        self.review_tree.setVisible(False)
        if review_result is None:
            raw_issues = list(review_annotation_file(ann_path, image_path))
            active_issues = list(raw_issues)
            accepted_issues = []
            stale = False
        else:
            raw_issues = list(review_result.raw_issues)
            active_issues = list(review_result.active_issues)
            accepted_issues = list(review_result.accepted_issues)
            stale = bool(review_result.stale)
        self._review_raw_issues = raw_issues
        self._review_issues = active_issues
        self._review_accepted_issues = accepted_issues
        self._review_decision_stale = stale

        if not raw_issues:
            self._set_review_summary('规则检查通过 · 未发现已配置问题', 'success')
            self.review_issue_panel.setVisible(False)
            self._update_current_review_actions()
            return

        self.review_issue_panel.setVisible(True)
        self.review_tree.setVisible(True)
        if not self._review_body_splitter_initialized:
            self.review_body_splitter.setSizes([620, 145])
            self._review_body_splitter_initialized = True
        active_count = len(active_issues)
        accepted_count = len(accepted_issues)
        if not active_issues:
            self._set_review_summary(
                f'人工复核通过: 已确认 {accepted_count} 个算法问题为误报',
                'manual',
            )
        elif stale:
            self._set_review_summary(
                f'标注或模板已变化，需要重新复核 {active_count} 个问题',
                'warning',
            )
        else:
            suffix = f'，已人工忽略 {accepted_count} 个' if accepted_count else ''
            self._set_review_summary(
                f'发现 {active_count} 个问题: '
                f'{_format_issue_counts(active_issues)}{suffix}',
                'danger',
            )

        for issue_idx, issue in enumerate(raw_issues):
            is_accepted = issue in accepted_issues
            shape_text = ', '.join(f'[{idx}]' for idx in issue.shape_indices)
            detail = f'{issue.message}; shapes={shape_text}'
            title = _issue_title(issue)
            if is_accepted:
                title = f'{title}（人工忽略）'
            item = QTreeWidgetItem(self.review_tree, [title, detail])
            item.setData(0, Qt.UserRole, issue_idx)
            item.setData(0, Qt.UserRole + 1, is_accepted)
            item.setToolTip(
                0,
                '已人工确认该问题为误报，可撤销人工结论'
                if is_accepted else '点击后在图片中高亮问题点',
            )
            item.setToolTip(1, detail)
            brush = QBrush(
                QColor('#36CFC9') if is_accepted else _issue_color(issue)
            )
            item.setForeground(0, brush)
            item.setForeground(1, brush)

        self.review_tree.resizeColumnToContents(0)
        self._update_current_review_actions()

    def _set_review_summary(self, text: str, tone: str):
        self.lbl_review_summary.setText(text)
        self.lbl_review_summary.setProperty('tone', tone)
        self.lbl_review_summary.style().unpolish(self.lbl_review_summary)
        self.lbl_review_summary.style().polish(self.lbl_review_summary)

    def _on_review_item_clicked(self, item, _column):
        issue_idx = item.data(0, Qt.UserRole)
        if issue_idx is None:
            self.review_issue_selected.emit([], [])
            return
        if 0 <= issue_idx < len(self._review_raw_issues):
            issue = self._review_raw_issues[issue_idx]
            if item.data(0, Qt.UserRole + 1):
                self.review_issue_selected.emit([], [])
                self._update_current_review_actions()
                return
            self.review_issue_selected.emit(
                issue.shape_indices, issue.point_indices
            )
        self._update_current_review_actions()

    def _emit_ignore_selected_issue(self, _checked=False):
        item = self.review_tree.currentItem()
        if item is None or item.data(0, Qt.UserRole + 1):
            return
        issue_idx = item.data(0, Qt.UserRole)
        if isinstance(issue_idx, int) and 0 <= issue_idx < len(self._review_raw_issues):
            self.manual_ignore_issue_requested.emit(
                self._review_raw_issues[issue_idx]
            )

    def _update_current_review_actions(self):
        if not hasattr(self, 'btn_manual_accept_current'):
            return
        selected = self.review_tree.currentItem()
        selected_active = bool(
            selected is not None
            and not selected.data(0, Qt.UserRole + 1)
        )
        self.btn_manual_accept_current.setEnabled(bool(self._review_issues))
        self.btn_manual_ignore_issue.setEnabled(selected_active)
        self.btn_manual_restore_current.setEnabled(
            bool(self._review_accepted_issues) or self._review_decision_stale
        )

    def review_highlights(self) -> tuple[list[int], list[tuple[int, int]]]:
        """Return all current review highlight targets."""
        shape_indices = []
        point_indices = []
        for issue in self._review_issues:
            shape_indices.extend(issue.shape_indices)
            point_indices.extend(issue.point_indices)
        return (
            list(dict.fromkeys(shape_indices)),
            list(dict.fromkeys(point_indices)),
        )

    def show_review_stats(self, total_images: int, rows: list[dict],
                          folder_summary: dict | None = None):
        """Show folder-level review statistics."""
        self.review_stats_tree.clear()
        summary = dict(folder_summary or {})
        if folder_summary is None and rows:
            all_issues = []
            for row in rows:
                if row.get('status') != 'manual':
                    all_issues.extend(row.get('issues', []))
            summary['issue_files'] = len(rows)
            summary['issue_count'] = len(all_issues)
            summary['rule_counts'] = {}
            summary['severity_counts'] = {}
            for issue in all_issues:
                summary['rule_counts'][issue.rule] = summary['rule_counts'].get(issue.rule, 0) + 1
                summary['severity_counts'][issue.severity] = summary['severity_counts'].get(issue.severity, 0) + 1
            self.lbl_review_stats.setToolTip(_format_issue_counts(all_issues))
        elif folder_summary is None:
            summary['issue_files'] = 0
            summary['issue_count'] = 0
            summary['rule_counts'] = {}
            summary['severity_counts'] = {}

        self._review_total_images = total_images
        self._review_folder_summary = summary
        self._review_stats_rows = list(rows)
        self.review_file_search.setEnabled(bool(rows))
        self._reset_metric_detail()

        missing_annotations = summary.get('missing_annotations', 0)
        invalid_annotations = summary.get('invalid_annotations', 0)
        manual_pass_files = _to_int(summary.get('manual_pass_files'))
        problem_images = min(
            total_images,
            summary['issue_files'] + missing_annotations + invalid_annotations,
        )
        if summary.get('annotation_set_missing'):
            annotation_dir = summary.get('annotation_dir', '-')
            self._set_review_directory_summary(
                f'当前任务没有对应标注集: {annotation_dir}; '
                f'{missing_annotations} / {total_images} 张缺失',
                'warning',
            )
        elif missing_annotations or invalid_annotations:
            parts = [f'目录统计: {problem_images} / {total_images} 张待处理']
            if missing_annotations:
                parts.append(f'{missing_annotations} 张缺失标注')
            if invalid_annotations:
                parts.append(f'{invalid_annotations} 张 JSON 无效')
            if manual_pass_files:
                parts.append(f'{manual_pass_files} 张人工通过')
            self._set_review_directory_summary(', '.join(parts), 'warning')
        elif summary['issue_files'] == 0:
            self._set_review_directory_summary(
                f'目录统计: 0 / {total_images} 张待处理'
                + (f', {manual_pass_files} 张人工通过' if manual_pass_files else ''),
                'manual' if manual_pass_files else 'success',
            )
        else:
            self._set_review_directory_summary(
                f'目录统计: {summary["issue_files"]} / {total_images} 张待处理, '
                f'{summary["issue_count"]} 个问题'
                + (f', {manual_pass_files} 张人工通过' if manual_pass_files else ''),
                'danger',
            )

        self._populate_review_stats_rows()
        self.review_analysis_panel.setVisible(True)
        self.review_charts.set_summary(total_images, summary)
        self._set_review_markdown(self._build_review_report(total_images, summary))
        if not self._review_stats_splitter_initialized:
            self.review_stats_splitter.setSizes([310, 430])
            self._review_stats_splitter_initialized = True

    def _clear_review_stats(self):
        self._set_review_directory_summary('目录统计: -', 'neutral')
        self.review_stats_tree.clear()
        self.review_file_search.blockSignals(True)
        self.review_file_search.clear()
        self.review_file_search.blockSignals(False)
        self.review_file_search.setEnabled(False)
        self.lbl_review_search_count.setText('共 0 项')
        self._review_stats_rows = []
        self._review_folder_summary = {}
        self._review_total_images = 0
        self._reset_metric_detail()
        self.review_charts.clear()
        self._set_review_markdown('点击 **统计当前文件夹** 生成文字分析。')
        self.review_analysis_panel.setVisible(False)

    def _set_review_directory_summary(self, text: str, tone: str):
        self.lbl_review_stats.setText(text)
        self.lbl_review_stats.setProperty('tone', tone)
        self.lbl_review_stats.style().unpolish(self.lbl_review_stats)
        self.lbl_review_stats.style().polish(self.lbl_review_stats)

    def has_review_stats(self) -> bool:
        return self._review_total_images > 0

    def _populate_review_stats_rows(self):
        if not hasattr(self, 'review_stats_tree'):
            return
        tree = self.review_stats_tree
        tree.setUpdatesEnabled(False)
        tree.clear()
        for row in self._review_stats_rows:
            status = str(row.get('status') or 'problem')
            issues = list(row.get('issues', []) or [])
            accepted = list(row.get('accepted_issues', []) or [])
            if status == 'manual':
                issue_text = f'人工通过 · {_format_issue_counts(accepted)}'
                color = QColor('#36CFC9')
            elif status == 'stale':
                issue_text = f'需重新复核 · {_format_issue_counts(issues)}'
                color = QColor('#F5A524')
            else:
                issue_text = _format_issue_counts(issues)
                if accepted:
                    issue_text += f' · 已忽略 {len(accepted)}'
                color = QColor('#FF4D4F')
            item = QTreeWidgetItem(tree, [
                row.get('filename', '-'),
                issue_text,
            ])
            item.setData(0, Qt.UserRole, row.get('index', -1))
            item.setData(0, Qt.UserRole + 1, status)
            item.setToolTip(0, '单击切换到该图片并在下方处理')
            detail_issues = accepted if status == 'manual' else issues
            item.setToolTip(
                1, '\n'.join(issue.message for issue in detail_issues)
            )
            brush = QBrush(color)
            item.setForeground(0, brush)
            item.setForeground(1, brush)
        tree.setUpdatesEnabled(True)
        tree.resizeColumnToContents(0)
        self._filter_review_stats_rows(self.review_file_search.text())

    def _filter_review_stats_rows(self, query: str):
        if not hasattr(self, 'review_stats_tree'):
            return
        visible, total, filtered = self._filter_tree_items(
            self.review_stats_tree, query
        )
        self.lbl_review_search_count.setText(
            f'匹配 {visible} / {total}' if filtered else f'共 {total} 项'
        )

    def _open_first_review_search_result(self):
        self._open_first_visible_tree_item(self.review_stats_tree)

    def _on_review_stats_item_activated(self, item, _column):
        image_index = item.data(0, Qt.UserRole)
        if isinstance(image_index, int) and image_index >= 0:
            self.review_file_selected.emit(image_index)

    def _reset_metric_detail(self):
        if not hasattr(self, 'metric_detail_tree'):
            return
        self.metric_detail_tree.clear()
        self.metric_detail_search.blockSignals(True)
        self.metric_detail_search.clear()
        self.metric_detail_search.blockSignals(False)
        self.metric_detail_search.setEnabled(False)
        self.lbl_metric_search_count.setText('共 0 项')
        self.lbl_metric_detail.setText('选择图表中的柱状查看文件明细')
        self.tabs.setCurrentIndex(1)
        self.tabs.setTabEnabled(2, False)

    def _on_chart_metric_selected(self, payload):
        if not payload:
            self._reset_metric_detail()
            return

        key = str(payload.get('key') or '')
        metric_files = self._review_folder_summary.get('metric_files', {}) or {}
        records = list(metric_files.get(key, []))
        records.sort(key=lambda record: (
            -_to_int(record.get('count')),
            str(record.get('filename') or ''),
        ))

        self.lbl_metric_detail.setText(
            self._build_metric_detail_summary(payload, records)
        )
        self.metric_detail_search.blockSignals(True)
        self.metric_detail_search.clear()
        self.metric_detail_search.blockSignals(False)
        self.metric_detail_search.setEnabled(bool(records))
        tree = self.metric_detail_tree
        tree.setUpdatesEnabled(False)
        tree.clear()
        items = []
        for record in records:
            count = _to_int(record.get('count'))
            expected = record.get('expected')
            status = str(record.get('status') or '')
            if expected is not None:
                value_text = f'{count} / {_to_int(expected)}'
                if status:
                    value_text = f'{value_text}  {status}'
            else:
                value_text = status or str(count)
            detail = str(record.get('detail') or '')
            item = QTreeWidgetItem([
                str(record.get('filename') or '-'),
                value_text,
                detail,
            ])
            item.setData(0, Qt.UserRole, record.get('index', -1))
            item.setToolTip(0, '双击跳转到该图片')
            item.setToolTip(2, detail)
            if any(token in status for token in ('问题', '缺失', '无效', '多出', '异常')):
                brush = QBrush(QColor('#FF6B6B'))
            elif '通过' in status or '有效' in status:
                brush = QBrush(QColor('#45D483'))
            else:
                brush = QBrush(QColor('#9EE7FF'))
            item.setForeground(0, brush)
            item.setForeground(1, brush)
            items.append(item)
        tree.addTopLevelItems(items)
        tree.setUpdatesEnabled(True)
        tree.resizeColumnToContents(0)
        tree.resizeColumnToContents(1)
        self._filter_metric_detail_rows('')
        self.tabs.setTabEnabled(2, True)
        self.tabs.setCurrentIndex(2)

    def _filter_metric_detail_rows(self, query: str):
        if not hasattr(self, 'metric_detail_tree'):
            return
        visible, total, filtered = self._filter_tree_items(
            self.metric_detail_tree, query
        )
        self.lbl_metric_search_count.setText(
            f'匹配 {visible} / {total}' if filtered else f'共 {total} 项'
        )

    def _open_first_metric_search_result(self):
        self._open_first_visible_tree_item(self.metric_detail_tree)

    def _filter_tree_items(self, tree: QTreeWidget,
                           query: str) -> tuple[int, int, bool]:
        tokens = [
            token.casefold() for token in str(query or '').split() if token
        ]
        visible = 0
        for index in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(index)
            searchable_parts = []
            for column in range(tree.columnCount()):
                searchable_parts.extend((
                    item.text(column), item.toolTip(column),
                ))
            searchable = ' '.join(searchable_parts).casefold()
            matched = all(token in searchable for token in tokens)
            item.setHidden(not matched)
            visible += int(matched)
        current = tree.currentItem()
        if current is not None and current.isHidden():
            tree.setCurrentItem(None)
        return visible, tree.topLevelItemCount(), bool(tokens)

    def _open_first_visible_tree_item(self, tree: QTreeWidget):
        for index in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(index)
            if item.isHidden():
                continue
            tree.setCurrentItem(item)
            tree.scrollToItem(item)
            self._on_review_stats_item_activated(item, 0)
            return

    def _build_metric_detail_summary(self, payload: dict,
                                     records: list[dict]) -> str:
        label = str(payload.get('label') or '-')
        value = _to_int(payload.get('value'))
        file_count = _to_int(payload.get('file_count'))
        total = max(0, self._review_total_images)
        coverage = (file_count / total * 100.0) if total else 0.0
        kind = str(payload.get('kind') or '')

        if kind in {'class', 'shape_type'}:
            average = value / file_count if file_count else 0.0
            return (
                f'{label}: {value} 个实例，覆盖 {file_count} / {total} 张图片 '
                f'({coverage:.1f}%)，覆盖图片平均 {average:.2f} 个。'
            )
        if kind == 'keypoint':
            expected = _to_int(payload.get('expected'))
            missing = sum(max(0, _to_int(record.get('expected')) - _to_int(record.get('count')))
                          for record in records)
            extra = sum(max(0, _to_int(record.get('count')) - _to_int(record.get('expected')))
                        for record in records)
            return (
                f'{label}: 实际 {value}，参考期望 {expected}，覆盖 '
                f'{file_count} / {total} 张图片；发现 {len(records)} 张数量异常，'
                f'缺失 {missing}，多出 {extra}。'
            )
        if kind in {'quality_issue', 'quality_ok'}:
            return f'{label}: {value} / {total} 张图片，占比 {coverage:.1f}%。'
        if kind == 'rule':
            return f'{label}: 共 {value} 个问题，涉及 {file_count} 张图片。'
        return (
            f'{label}: {value}，涉及 {file_count} / {total} 张图片 '
            f'({coverage:.1f}%)。'
        )

    def _set_analysis_mode(self, mode: int):
        if hasattr(self, 'review_analysis_stack'):
            self.review_analysis_stack.setCurrentIndex(mode)
            self.review_analysis_stack.setMinimumHeight(
                280 if mode == 0 else 165
            )
        if hasattr(self, 'btn_analysis_charts'):
            self.btn_analysis_charts.setChecked(mode == 0)
        if hasattr(self, 'btn_analysis_text'):
            self.btn_analysis_text.setChecked(mode == 1)

    def _set_review_markdown(self, markdown: str):
        if hasattr(self.review_text_report, 'setMarkdown'):
            self.review_text_report.setMarkdown(markdown)
        else:
            self.review_text_report.setPlainText(markdown)

    def _build_review_report(self, total_images: int, summary: dict) -> str:
        issue_files = _to_int(summary.get('issue_files'))
        issue_count = _to_int(summary.get('issue_count'))
        raw_issue_files = _to_int(summary.get('raw_issue_files'))
        raw_issue_count = _to_int(summary.get('raw_issue_count'))
        manual_pass_files = _to_int(summary.get('manual_pass_files'))
        accepted_issue_count = _to_int(summary.get('accepted_issue_count'))
        stale_review_files = _to_int(summary.get('stale_review_files'))
        annotation_files = _to_int(summary.get('annotation_files'))
        missing_annotations = _to_int(summary.get('missing_annotations'))
        invalid_annotations = _to_int(summary.get('invalid_annotations'))
        total_shapes = _to_int(summary.get('total_shapes'))
        person_boxes = _to_int(summary.get('person_boxes'))
        keypoints = _to_int(summary.get('keypoints'))
        other_shapes = _to_int(summary.get('other_shapes'))
        checked_true = _to_int(summary.get('checked_true'))
        checked_false = _to_int(summary.get('checked_false'))
        checked_unknown = _to_int(summary.get('checked_unknown'))
        task_type = str(summary.get('task_type') or '-')
        annotation_dir = str(summary.get('annotation_dir') or '-')
        annotation_set_dir = str(summary.get('annotation_set_dir') or '')
        annotation_set_missing = bool(summary.get('annotation_set_missing'))
        object_label = '人框' if task_type == 'pose' else '目标框'
        auto_ok_images = max(
            0,
            total_images - min(
                total_images,
                issue_files + missing_annotations + invalid_annotations
                + manual_pass_files,
            ),
        )
        problem_images = min(
            total_images,
            issue_files + missing_annotations + invalid_annotations,
        )

        lines = [
            '# 审查文字分析',
            '',
            '## 当前任务',
            f'- 任务类型: **{task_type}**',
            f'- 标注集: **{annotation_dir}**',
        ]
        if annotation_set_dir:
            lines.append(f'- 标注集路径: `{annotation_set_dir}`')
        if annotation_set_missing:
            lines.append('- 标注集状态: **当前任务没有对应标注集**')

        lines.extend([
            '',
            '## 数据完整性',
            f'- 图片数量: **{total_images}**',
            f'- JSON 标注: **{annotation_files}**',
            f'- 缺失标注: **{missing_annotations}**',
            f'- JSON 无效: **{invalid_annotations}**',
            '',
            '## 标签统计',
            f'- shape 总数: **{total_shapes}**',
            f'- {object_label}: **{person_boxes}**',
            f'- 关键点: **{keypoints}**' if task_type == 'pose' else f'- 关键点: **{keypoints}**（当前任务通常不使用）',
            f'- 其他标签: **{other_shapes}**',
            f'- checked 状态: True={checked_true}，False={checked_false}，未填写={checked_unknown}',
            '',
            '## 质量结论',
            f'- 待处理图片: **{problem_images}**',
            f'- 人工通过图片: **{manual_pass_files}**',
            f'- 规则检查通过图片: **{auto_ok_images}**',
            f'- 当前待处理问题: **{issue_count}**',
            f'- 算法原始问题: **{raw_issue_count}**，涉及 **{raw_issue_files}** 张图片',
            f'- 人工确认误报: **{accepted_issue_count}**',
        ])
        if stale_review_files:
            lines.append(f'- 需要重新复核: **{stale_review_files}** 张')

        rule_counts = summary.get('rule_counts', {}) or {}
        if rule_counts:
            top_rules = sorted(rule_counts.items(), key=lambda item: (-item[1], item[0]))
            lines.append(
                '- 主要问题: '
                + '，'.join(
                    f'`{_issue_title_for_rule(rule)}` × {count}'
                    for rule, count in top_rules[:6]
                )
            )
        elif missing_annotations or invalid_annotations:
            lines.append('- 当前规则未发现具体标注问题，但需要先补齐缺失或无效 JSON 后重新统计。')
        else:
            lines.append(
                '- 当前已执行规则未发现标注问题；这不等价于人工确认标注绝对正确。'
            )

        target_class_counts = summary.get('target_class_counts', {}) or {}
        class_parts = [
            f'`{label}`={_to_int(count)}'
            for label, count in target_class_counts.items()
            if _to_int(count) > 0
        ]
        if class_parts:
            lines.extend([
                '',
                f'## {object_label}类别样本',
                '- ' + '，'.join(class_parts),
            ])

        keypoint_counts = summary.get('keypoint_counts', {}) or {}
        nonzero_keypoints = [
            (label, _to_int(count))
            for label, count in keypoint_counts.items()
            if _to_int(count) > 0
        ]
        if nonzero_keypoints:
            top_keypoints = sorted(
                nonzero_keypoints,
                key=lambda item: (-item[1], item[0]),
            )[:6]
            low_keypoints = sorted(nonzero_keypoints, key=lambda item: (item[1], item[0]))[:6]
            lines.extend([
                '',
                '## 关键点分布',
                '- 数量较多: '
                + '，'.join(f'`{label}`={count}' for label, count in top_keypoints),
                '- 数量较少: '
                + '，'.join(f'`{label}`={count}' for label, count in low_keypoints),
            ])

        shape_type_counts = summary.get('shape_type_counts', {}) or {}
        shape_type_parts = [
            f'`{label}`={_to_int(count)}'
            for label, count in shape_type_counts.items()
            if _to_int(count) > 0
        ]
        if shape_type_parts and task_type != 'pose':
            lines.extend([
                '',
                '## shape 类型分布',
                '- ' + '，'.join(shape_type_parts),
            ])

        lines.extend([
            '',
            '> 下方问题列表保持不变，可双击具体文件跳转查看。',
        ])
        return '\n'.join(lines)

    def _load_annotation_tree(self, ann_path: str | Path):
        """Populate the annotation tree with per-point QCheckBox widgets."""
        self.ann_tree.clear()
        try:
            data = json.loads(Path(ann_path).read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            return

        self._add_tree_item('version', data.get('version', '-'))
        self._add_tree_item('checked', str(data.get('checked', False)))

        shapes = data.get('shapes', [])
        shapes_root = QTreeWidgetItem(
            self.ann_tree, ['shapes', f'{len(shapes)} 个标注']
        )
        shapes_root.setExpanded(True)

        for si, shape in enumerate(shapes):
            label = shape.get('label', '-')
            shape_item = QTreeWidgetItem(shapes_root, [f'[{si}]', str(label)])
            self._add_tree_child(
                shape_item, 'group_id', str(shape.get('group_id', '-'))
            )
            self._add_tree_child(
                shape_item, 'difficult', str(shape.get('difficult', False))
            )
            shape_type = str(
                shape.get('shape_type') or shape.get('type') or ''
            ).strip()
            if shape_type:
                self._add_tree_child(shape_item, '类型', shape_type)

            # Per-point items with QCheckBox widgets
            points = shape.get('points', [])
            if shape_type == 'rotation' and len(points) >= 3:
                self._add_tree_child(
                    shape_item, '旋转角', f'{_obb_angle_deg(points):.1f}°'
                )
            pts_item = QTreeWidgetItem(shape_item, ['关键点', f'{len(points)} 个'])
            pts_item.setExpanded(len(points) <= 4)
            for pi, pt in enumerate(points):
                pt_item = QTreeWidgetItem(pts_item)
                # Column 0: checkbox + label
                cb = QCheckBox(f'点 {pi + 1}')
                cb.setChecked(True)
                cb.toggled.connect(
                    lambda checked, s=si, p=pi: self.point_toggled.emit(s, p, checked)
                )
                self.ann_tree.setItemWidget(pt_item, 0, cb)
                # Column 1: coordinates
                pt_item.setText(1, f'({pt[0]:.0f}, {pt[1]:.0f})')

    def _add_tree_item(self, key: str, value: str):
        QTreeWidgetItem(self.ann_tree, [key, value])

    def _add_tree_child(self, parent, key: str, value: str):
        QTreeWidgetItem(parent, [key, value])
