"""Full-image display widget with zoom, pan, and annotation overlay.

Annotation modes cycle (A key or button):
  0 = hidden
  1 = rectangles + labels
  2 = rectangles + labels + key points
"""

import json
from pathlib import Path

from PyQt5.QtCore import Qt, QPoint, QPointF, QRectF, pyqtSignal
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt5.QtWidgets import QLabel, QScrollArea

from app.models.annotation_review import KPT_CONNECTION_LABELS, KEYPOINT_SET
from app.utils import log

ANNOTATION_MODES = {
    0: '隐藏标注',
    1: '仅矩形框',
    2: '矩形框 + 关键点',
}
POINT_RADIUS = 4
ANNOTATION_COLOR = QColor('#45D483')        # green
ANNOTATION_BORDER_COLOR = QColor('#E6EDF3') # light border for keypoints
REVIEW_HIGHLIGHT_COLOR = QColor('#FF4D4F')  # red
SKELETON_LINE_WIDTH = 2
SKELETON_HIGHLIGHT_WIDTH = 3
MIN_SCALE = 0.02
MAX_SCALE = 20.0
WHEEL_ZOOM_FACTOR = 1.15


class ImageViewer(QScrollArea):
    """Center panel: displays full image with configurable annotation overlay."""

    annotation_mode_changed = pyqtSignal(int)
    skeleton_visibility_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('imageViewer')
        self.setAlignment(Qt.AlignCenter)
        self.setWidgetResizable(False)
        self.setFocusPolicy(Qt.StrongFocus)
        self.viewport().setObjectName('imageViewport')
        self.viewport().setMouseTracking(True)

        self._label = _ImageLabel()
        self._label.setObjectName('imageCanvasLabel')
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setWidget(self._label)

        self._scale = 1.0
        self._fit_to_window = True
        self._annotation_mode = 1  # 0=off, 1=rect, 2=rect+points
        self._pixmap: QPixmap | None = None
        self._annotation_shapes: list[dict] = []
        self._current_path: str = ''
        self._hidden_points: set[str] = set()
        self._highlighted_shapes: set[int] = set()
        self._highlighted_points: set[str] = set()
        self._show_skeleton = False
        self._pan_active = False
        self._pan_start = QPoint()
        self._pan_scroll_start = QPoint()

        log('ImageViewer 初始化完成')

    # --- public API ---

    def load_image(self, path: str | Path):
        self._current_path = str(path)
        self._pixmap = QPixmap(str(path))
        if self._pixmap.isNull():
            log(f'⚠ 无法加载图片: {path}')
            self._label.setText('无法加载图片')
            return
        pw, ph = self._pixmap.width(), self._pixmap.height()
        log(f'📷 加载图片: {Path(path).name} ({pw}×{ph})')
        self._annotation_shapes = []
        self._hidden_points.clear()
        self.clear_review_highlights(update=False)
        self._fit_scale()
        self._update_display()

    def load_annotation(self, annotation_path: str | Path | None):
        self._annotation_shapes = []
        self._hidden_points.clear()
        self.clear_review_highlights(update=False)
        if annotation_path is not None:
            try:
                data = json.loads(Path(annotation_path).read_text(encoding='utf-8'))
                self._annotation_shapes = data.get('shapes', [])
                total_pts = sum(len(s.get('points', [])) for s in self._annotation_shapes)
                log(f'📝 加载标注: {len(self._annotation_shapes)} shapes, {total_pts} points')
            except (json.JSONDecodeError, OSError) as e:
                log(f'⚠ 标注加载失败: {e}')
        else:
            log('📝 无标注文件')
        self._update_display()

    def current_path(self) -> str:
        return self._current_path

    # --- annotation mode ---

    def annotation_mode(self) -> int:
        return self._annotation_mode

    def annotation_mode_name(self) -> str:
        return ANNOTATION_MODES.get(self._annotation_mode, '')

    def set_annotation_mode(self, mode: int):
        self._annotation_mode = mode % 3
        log(f'🏷 标注模式: {self.annotation_mode_name()}')
        self._update_display()
        self.annotation_mode_changed.emit(self._annotation_mode)

    def cycle_annotation_mode(self):
        self.set_annotation_mode(self._annotation_mode + 1)

    def toggle_annotations(self):
        self.cycle_annotation_mode()

    # --- point visibility ---

    def is_point_visible(self, shape_idx: int, point_idx: int) -> bool:
        return f'{shape_idx},{point_idx}' not in self._hidden_points

    def set_point_visible(self, shape_idx: int, point_idx: int, visible: bool):
        key = f'{shape_idx},{point_idx}'
        if visible:
            self._hidden_points.discard(key)
        else:
            self._hidden_points.add(key)
        log(f'🔵 点 [{shape_idx}][{point_idx}] visible={visible} (hidden={len(self._hidden_points)})')
        self._update_display()

    def all_points_visible(self) -> bool:
        return len(self._hidden_points) == 0

    def set_all_points_visible(self, visible: bool):
        if visible:
            self._hidden_points.clear()
        else:
            for si, shape in enumerate(self._annotation_shapes):
                for pi in range(len(shape.get('points', []))):
                    self._hidden_points.add(f'{si},{pi}')
        log(f'🔵 全部关键点 visible={visible}')
        self._update_display()

    def get_shapes(self) -> list[dict]:
        return self._annotation_shapes

    # --- skeleton overlay ---

    def skeleton_visible(self) -> bool:
        return self._show_skeleton

    def set_skeleton_visible(self, visible: bool):
        self._show_skeleton = bool(visible)
        if self._show_skeleton and self._annotation_mode == 0:
            self.set_annotation_mode(2)
        self._update_display()
        self.skeleton_visibility_changed.emit(self._show_skeleton)

    def toggle_skeleton(self):
        self.set_skeleton_visible(not self._show_skeleton)

    # --- review highlights ---

    def set_review_highlights(self, shape_indices: list[int] | None = None,
                              point_indices: list[tuple[int, int]] | None = None):
        """Mark review issues in red. `point_indices` are (shape_idx, point_idx)."""
        self._highlighted_shapes = set(shape_indices or [])
        self._highlighted_points = {
            f'{shape_idx},{point_idx}'
            for shape_idx, point_idx in (point_indices or [])
        }
        self._update_display()

    def clear_review_highlights(self, update: bool = True):
        self._highlighted_shapes.clear()
        self._highlighted_points.clear()
        if update:
            self._update_display()

    def _is_shape_highlighted(self, shape_idx: int) -> bool:
        return shape_idx in self._highlighted_shapes

    def _is_point_highlighted(self, shape_idx: int, point_idx: int) -> bool:
        return (self._is_shape_highlighted(shape_idx)
                or f'{shape_idx},{point_idx}' in self._highlighted_points)

    # --- zoom ---

    def toggle_fit(self):
        if self._fit_to_window:
            self._fit_to_window = False
            self._scale = 1.0
            self._update_display()
        else:
            self.fit_to_window()

    def zoom_in(self, factor: float = WHEEL_ZOOM_FACTOR):
        self._zoom_at(self.viewport().rect().center(), factor)

    def zoom_out(self, factor: float = WHEEL_ZOOM_FACTOR):
        self._zoom_at(self.viewport().rect().center(), 1.0 / factor)

    def zoom_scale(self) -> float:
        return self._scale

    def fit_to_window(self):
        self._fit_to_window = True
        self._fit_scale()
        self._update_display()

    def _zoom_at(self, viewport_pos: QPoint, factor: float):
        if self._pixmap is None or self._pixmap.isNull() or factor <= 0:
            return
        old_width = max(1, self._label.width())
        old_height = max(1, self._label.height())
        image_pos = self._label.mapFrom(self.viewport(), viewport_pos)
        ratio_x = min(max(image_pos.x() / old_width, 0.0), 1.0)
        ratio_y = min(max(image_pos.y() / old_height, 0.0), 1.0)
        new_scale = min(max(self._scale * factor, MIN_SCALE), MAX_SCALE)
        if abs(new_scale - self._scale) < 1e-9:
            return

        self._scale = new_scale
        self._fit_to_window = False
        self._update_display()

        new_width = self._label.width()
        new_height = self._label.height()
        self.horizontalScrollBar().setValue(
            round(ratio_x * new_width - viewport_pos.x())
        )
        self.verticalScrollBar().setValue(
            round(ratio_y * new_height - viewport_pos.y())
        )
        self._update_pan_cursor()

    def _fit_scale(self):
        if self._pixmap is None or self._pixmap.isNull():
            return
        viewport = self.viewport()
        if not viewport:
            return
        vw = viewport.width() - 10
        vh = viewport.height() - 10
        pw = self._pixmap.width()
        ph = self._pixmap.height()
        if pw > 0 and ph > 0:
            self._scale = max(MIN_SCALE, min(vw / pw, vh / ph, 1.0))

    # --- rendering ---

    def _update_display(self):
        if self._pixmap is None or self._pixmap.isNull():
            return
        pw = int(self._pixmap.width() * self._scale)
        ph = int(self._pixmap.height() * self._scale)
        if pw <= 0 or ph <= 0:
            return

        scaled = self._pixmap.scaled(
            pw, ph, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        # Draw annotations
        n_shapes = len(self._annotation_shapes)
        if self._annotation_mode > 0 and n_shapes > 0:
            painter = QPainter(scaled)
            painter.setRenderHint(QPainter.Antialiasing)
            sx = pw / self._pixmap.width()
            sy = ph / self._pixmap.height()

            rect_pen = QPen(ANNOTATION_COLOR, 2)
            point_border = QPen(ANNOTATION_BORDER_COLOR, 1.5)
            point_fill = QBrush(ANNOTATION_COLOR)
            text_color = ANNOTATION_COLOR
            highlight_rect_pen = QPen(REVIEW_HIGHLIGHT_COLOR, 3)
            highlight_point_border = QPen(REVIEW_HIGHLIGHT_COLOR, 2)
            highlight_point_fill = QBrush(REVIEW_HIGHLIGHT_COLOR)

            n_bbox = 0
            n_kpt = 0
            n_vtx_drawn = 0
            n_skeleton, n_skeleton_red = self._draw_skeleton(painter, sx, sy)

            for si, shape in enumerate(self._annotation_shapes):
                points = shape.get('points', [])
                label = shape.get('label', '')
                if not points:
                    continue

                if len(points) == 1:
                    # Single-point shape → draw as a keypoint dot (check visibility)
                    highlighted = self._is_point_highlighted(si, 0)
                    if not self.is_point_visible(si, 0) and not highlighted:
                        continue
                    px, py = points[0][0] * sx, points[0][1] * sy
                    painter.setPen(highlight_point_border if highlighted else point_border)
                    painter.setBrush(highlight_point_fill if highlighted else point_fill)
                    painter.drawEllipse(
                        QPointF(px, py), POINT_RADIUS + 1, POINT_RADIUS + 1
                    )
                    painter.setPen(REVIEW_HIGHLIGHT_COLOR if highlighted else text_color)
                    painter.setBrush(Qt.NoBrush)
                    painter.drawText(QPointF(px + 6, py - 6), str(label))
                    n_kpt += 1

                else:
                    # Multi-point shape → draw bounding box + label
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    x1, x2 = min(xs) * sx, max(xs) * sx
                    y1, y2 = min(ys) * sy, max(ys) * sy

                    highlighted = self._is_shape_highlighted(si)
                    painter.setPen(highlight_rect_pen if highlighted else rect_pen)
                    painter.setBrush(Qt.NoBrush)
                    painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
                    painter.setPen(REVIEW_HIGHLIGHT_COLOR if highlighted else text_color)
                    painter.drawText(QPointF(x1, y1 - 4), str(label))
                    n_bbox += 1

                    # Per-vertex points (mode 2)
                    if self._annotation_mode >= 2:
                        for pi, pt in enumerate(points):
                            pt_highlighted = self._is_point_highlighted(si, pi)
                            if not self.is_point_visible(si, pi) and not pt_highlighted:
                                continue
                            px, py = pt[0] * sx, pt[1] * sy
                            painter.setPen(
                                highlight_point_border if pt_highlighted else point_border
                            )
                            painter.setBrush(
                                highlight_point_fill if pt_highlighted else point_fill
                            )
                            painter.drawEllipse(
                                QPointF(px, py), POINT_RADIUS, POINT_RADIUS
                            )
                            painter.setPen(
                                REVIEW_HIGHLIGHT_COLOR if pt_highlighted else text_color
                            )
                            painter.setBrush(Qt.NoBrush)
                            painter.drawText(
                                QPointF(px + 5, py - 5), str(pi + 1)
                            )
                            n_vtx_drawn += 1

            painter.end()
            log(f'🎨 渲染: {n_bbox} bbox + {n_kpt} keypoints + {n_vtx_drawn} vertices + {n_skeleton} skeleton ({n_skeleton_red} red) (mode={self._annotation_mode}, scale={self._scale:.2f})')

        self._label.setPixmap(scaled)
        self._label.resize(scaled.size())
        self._update_pan_cursor()

    def _draw_skeleton(self, painter: QPainter, sx: float, sy: float) -> tuple[int, int]:
        if not self._show_skeleton or self._annotation_mode == 0:
            return 0, 0

        normal_pen = QPen(ANNOTATION_COLOR, SKELETON_LINE_WIDTH)
        highlight_pen = QPen(REVIEW_HIGHLIGHT_COLOR, SKELETON_HIGHLIGHT_WIDTH)
        grouped = self._keypoint_shapes_by_group()
        line_count = 0
        red_count = 0

        for points_by_label in grouped.values():
            for label_a, label_b in KPT_CONNECTION_LABELS:
                entry_a = self._first_drawable_point(points_by_label.get(label_a, []))
                entry_b = self._first_drawable_point(points_by_label.get(label_b, []))
                if entry_a is None or entry_b is None:
                    continue

                shape_a, point_a = entry_a
                shape_b, point_b = entry_b
                highlighted = (
                    self._is_point_highlighted(shape_a, 0)
                    or self._is_point_highlighted(shape_b, 0)
                )
                painter.setPen(highlight_pen if highlighted else normal_pen)
                painter.drawLine(
                    QPointF(point_a[0] * sx, point_a[1] * sy),
                    QPointF(point_b[0] * sx, point_b[1] * sy),
                )
                line_count += 1
                if highlighted:
                    red_count += 1

        return line_count, red_count

    def _keypoint_shapes_by_group(self) -> dict[object, dict[str, list[tuple[int, tuple[float, float]]]]]:
        grouped: dict[object, dict[str, list[tuple[int, tuple[float, float]]]]] = {}
        for shape_idx, shape in enumerate(self._annotation_shapes):
            label = str(shape.get('label', '')).strip()
            if label not in KEYPOINT_SET:
                continue
            points = shape.get('points', [])
            shape_type = str(
                shape.get('shape_type', shape.get('shape_type ', ''))
            ).strip()
            if not points or (shape_type != 'point' and len(points) != 1):
                continue
            try:
                point = (float(points[0][0]), float(points[0][1]))
            except (TypeError, ValueError, IndexError):
                continue
            group_id = shape.get('group_id')
            grouped.setdefault(group_id, {}).setdefault(label, []).append(
                (shape_idx, point)
            )
        return grouped

    def _first_drawable_point(self, entries: list[tuple[int, tuple[float, float]]]
                              ) -> tuple[int, tuple[float, float]] | None:
        for shape_idx, point in entries:
            highlighted = self._is_point_highlighted(shape_idx, 0)
            if self.is_point_visible(shape_idx, 0) or highlighted:
                return shape_idx, point
        return None

    # --- events ---

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0 or self._pixmap is None or self._pixmap.isNull():
            super().wheelEvent(event)
            return
        steps = delta / 120.0
        self._zoom_at(event.pos(), WHEEL_ZOOM_FACTOR ** steps)
        event.accept()

    def mousePressEvent(self, event):
        can_pan = (
            self.horizontalScrollBar().maximum() > 0
            or self.verticalScrollBar().maximum() > 0
        )
        if event.button() == Qt.LeftButton and can_pan:
            self._pan_active = True
            self._pan_start = event.pos()
            self._pan_scroll_start = QPoint(
                self.horizontalScrollBar().value(),
                self.verticalScrollBar().value(),
            )
            self.viewport().setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan_active and event.buttons() & Qt.LeftButton:
            delta = event.pos() - self._pan_start
            self.horizontalScrollBar().setValue(
                self._pan_scroll_start.x() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self._pan_scroll_start.y() - delta.y()
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._pan_active:
            self._pan_active = False
            self._update_pan_cursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit_to_window and self._pixmap is not None:
            self._fit_scale()
            self._update_display()

    def _update_pan_cursor(self):
        if self._pan_active:
            return
        can_pan = (
            self.horizontalScrollBar().maximum() > 0
            or self.verticalScrollBar().maximum() > 0
        )
        self.viewport().setCursor(
            Qt.OpenHandCursor if can_pan else Qt.ArrowCursor
        )

    def enterEvent(self, event):
        self.setFocus()
        super().enterEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_1:
            self.fit_to_window()
        elif event.key() in {Qt.Key_Plus, Qt.Key_Equal}:
            self.zoom_in()
        elif event.key() == Qt.Key_Minus:
            self.zoom_out()
        elif event.key() == Qt.Key_0:
            self.fit_to_window()
        elif event.key() == Qt.Key_A:
            self.cycle_annotation_mode()
        else:
            super().keyPressEvent(event)


class _ImageLabel(QLabel):
    pass
