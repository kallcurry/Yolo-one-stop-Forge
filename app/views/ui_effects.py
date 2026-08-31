"""Small visual effects for the PyQt interface."""

import math

from PyQt5.QtCore import (
    QEasingCurve,
    QElapsedTimer,
    QEvent,
    QObject,
    QPropertyAnimation,
    QTimer,
    Qt,
)
from PyQt5.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPen
from PyQt5.QtWidgets import (
    QAbstractButton,
    QApplication,
    QGraphicsDropShadowEffect,
    QWidget,
)


ACCENT_BLUE = QColor('#36B7FF')

# ---- Ambient animation scheduling (inference can pause the backdrop) ----
_AMBIENT_TIMERS: list = []
_AMBIENT_ENABLED = True


def register_ambient_timer(timer):
    """Register a repeating animation timer for global pause/restore."""
    _AMBIENT_TIMERS.append(timer)
    if _AMBIENT_ENABLED:
        timer.start()


def set_ambient_animation_enabled(enabled: bool):
    """Pause/resume background animations (releases CPU while inferring)."""
    global _AMBIENT_ENABLED
    _AMBIENT_ENABLED = bool(enabled)
    for timer in list(_AMBIENT_TIMERS):
        if _AMBIENT_ENABLED:
            timer.start()
        else:
            timer.stop()

ACCENT_GREEN = QColor('#45D483')
ACCENT_RED = QColor('#FF4D4F')


class HoverGlow(QObject):
    """Adds subtle neon hover and checked-state glow to buttons."""

    def __init__(self, parent=None):
        # Event filters can receive late events while Qt is constructing or
        # destroying the owning widget. Establish Python state first.
        self._effects = {}
        self._animations = []
        self._stopping = False
        super().__init__(parent)
        self._pulse_phase = 0.0
        self._effects_enabled = QApplication.platformName() != 'offscreen'
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(90)
        self._pulse_timer.timeout.connect(self._pulse_checked_buttons)
        if self._effects_enabled:
            self._pulse_timer.start()
        register_ambient_timer(self._pulse_timer)

    def watch_buttons(self, root: QWidget):
        for button in root.findChildren(QAbstractButton):
            self.watch(button)

    def watch(self, button: QAbstractButton):
        effects = getattr(self, '_effects', None)
        if not self._effects_enabled or effects is None or self._stopping:
            return
        if button in effects:
            return
        effect = QGraphicsDropShadowEffect(button)
        effect.setOffset(0, 0)
        effect.setBlurRadius(0)
        effect.setColor(_accent_for_button(button))
        button.setGraphicsEffect(effect)
        effects[button] = effect
        button.installEventFilter(self)

    def stop(self):
        """Stop timers before an owning window starts destroying its buttons."""
        self._stopping = True
        timer = getattr(self, '_pulse_timer', None)
        if timer is not None:
            timer.stop()
        animations = getattr(self, '_animations', None)
        for animation in list(animations or ()):
            animation.stop()
        if animations is not None:
            animations.clear()
        effects = getattr(self, '_effects', {})
        for button in list(effects):
            try:
                button.removeEventFilter(self)
            except RuntimeError:
                pass
        effects.clear()

    def eventFilter(self, obj, event):
        effects = getattr(self, '_effects', None)
        if not effects or getattr(self, '_stopping', False):
            return False
        effect = effects.get(obj)
        if effect is None:
            return super().eventFilter(obj, event)

        if event.type() == QEvent.Enter and obj.isEnabled():
            self._set_effect_color(effect, obj, 210)
            self._animate(effect, 16)
        elif event.type() == QEvent.Leave:
            self._animate(effect, 8 if _button_checked(obj) else 0)
        elif event.type() == QEvent.MouseButtonPress and obj.isEnabled():
            self._set_effect_color(effect, obj, 180)
            self._animate(effect, 5, duration=80)
        elif event.type() == QEvent.MouseButtonRelease and obj.isEnabled():
            self._animate(
                effect,
                16 if obj.underMouse() else (8 if _button_checked(obj) else 0),
                duration=120,
            )

        return super().eventFilter(obj, event)

    def _animate(self, effect: QGraphicsDropShadowEffect,
                 end_value: int, duration: int = 160):
        anim = QPropertyAnimation(effect, b'blurRadius', self)
        anim.setDuration(duration)
        anim.setStartValue(effect.blurRadius())
        anim.setEndValue(end_value)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        self._animations.append(anim)

        def _cleanup():
            if anim in self._animations:
                self._animations.remove(anim)

        anim.finished.connect(_cleanup)
        anim.start()

    def _pulse_checked_buttons(self):
        self._pulse_phase = (self._pulse_phase + 0.24) % (math.pi * 2)
        effects = getattr(self, '_effects', {})
        for button, effect in list(effects.items()):
            if not button or not button.isEnabled() or not _button_checked(button):
                continue
            if button.underMouse():
                continue
            pulse = 8.0 + 3.5 * (0.5 + 0.5 * math.sin(self._pulse_phase))
            effect.setBlurRadius(pulse)
            self._set_effect_color(effect, button, 145)

    @staticmethod
    def _set_effect_color(effect: QGraphicsDropShadowEffect,
                          button: QAbstractButton,
                          alpha: int):
        color = _accent_for_button(button)
        color.setAlpha(alpha)
        effect.setColor(color)


class TechBackdrop(QWidget):
    """Low-contrast animated grid background inspired by the reference effects."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAutoFillBackground(False)
        self._elapsed = QElapsedTimer()
        self._elapsed.start()
        self._timer = QTimer(self)
        self._timer.setInterval(45)
        self._timer.timeout.connect(self.update)
        self._timer.start()
        register_ambient_timer(self._timer)

    def paintEvent(self, _event):
        rect = self.rect()
        if rect.isEmpty():
            return

        t = self._elapsed.elapsed() / 1000.0
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        base = QLinearGradient(0, 0, rect.width(), rect.height())
        base.setColorAt(0.0, QColor(6, 12, 18))
        base.setColorAt(0.46, QColor(10, 19, 29))
        base.setColorAt(1.0, QColor(6, 10, 16))
        painter.fillRect(rect, QBrush(base))

        self._paint_depth_sheen(painter, rect.width(), rect.height(), t)
        self._paint_grid(painter, rect.width(), rect.height(), t)
        self._paint_scan_sweep(painter, rect.width(), rect.height(), t)
        self._paint_signal_ticks(painter, rect.width(), rect.height(), t)
        self._paint_trace_lines(painter, rect.width(), rect.height(), t)
        self._paint_edge_glow(painter, rect.width(), rect.height(), t)

    @staticmethod
    def _paint_depth_sheen(painter: QPainter, width: int, height: int, t: float):
        if width <= 0 or height <= 0:
            return
        diagonal = QLinearGradient(0, 0, width, height)
        diagonal.setColorAt(0.0, QColor(54, 183, 255, 18))
        diagonal.setColorAt(0.34, QColor(54, 183, 255, 0))
        diagonal.setColorAt(0.62, QColor(69, 212, 131, 10))
        diagonal.setColorAt(1.0, QColor(69, 212, 131, 0))
        painter.fillRect(0, 0, width, height, QBrush(diagonal))

        band_x = int((math.sin(t * 0.38) * 0.5 + 0.5) * max(1, width))
        band = QLinearGradient(band_x - 180, 0, band_x + 180, 0)
        band.setColorAt(0.0, QColor(54, 183, 255, 0))
        band.setColorAt(0.5, QColor(54, 183, 255, 12))
        band.setColorAt(1.0, QColor(54, 183, 255, 0))
        painter.fillRect(0, 0, width, height, QBrush(band))

    @staticmethod
    def _paint_grid(painter: QPainter, width: int, height: int, t: float):
        grid = 42
        offset = int((t * 18) % grid)
        fine_pen = QPen(QColor(54, 183, 255, 18), 1)
        strong_pen = QPen(QColor(69, 212, 131, 20), 1)

        painter.setPen(fine_pen)
        for x in range(-grid + offset, width + grid, grid):
            painter.drawLine(x, 0, x, height)
        for y in range(-grid + offset, height + grid, grid):
            painter.drawLine(0, y, width, y)

        painter.setPen(strong_pen)
        for x in range(-grid + offset * 2, width + grid, grid * 4):
            painter.drawLine(x, 0, x, height)
        for y in range(-grid + offset * 2, height + grid, grid * 4):
            painter.drawLine(0, y, width, y)

    @staticmethod
    def _paint_scan_sweep(painter: QPainter, width: int, height: int, t: float):
        if width <= 0 or height <= 0:
            return
        sweep = (t * 120) % (height + 220) - 110
        grad = QLinearGradient(0, sweep - 70, width, sweep + 70)
        grad.setColorAt(0.0, QColor(54, 183, 255, 0))
        grad.setColorAt(0.5, QColor(54, 183, 255, 24))
        grad.setColorAt(1.0, QColor(69, 212, 131, 0))
        painter.fillRect(0, 0, width, height, QBrush(grad))

    @staticmethod
    def _paint_signal_ticks(painter: QPainter, width: int, height: int, t: float):
        if width <= 0 or height <= 0:
            return
        painter.save()
        grid = 42
        for idx, y in enumerate(range(72, height, grid * 3)):
            phase = (t * 90 + idx * 117) % (width + 180) - 90
            alpha = 34 + int(18 * (0.5 + 0.5 * math.sin(t * 1.7 + idx)))
            pen = QPen(QColor(98, 232, 255, alpha), 2)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.drawLine(int(phase), y, int(phase + 38), y)
            painter.drawLine(int(phase + 50), y, int(phase + 64), y)

        for idx, x in enumerate(range(84, width, grid * 4)):
            phase = (t * 70 + idx * 83) % (height + 140) - 70
            pen = QPen(QColor(69, 212, 131, 24), 1)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.drawLine(x, int(phase), x, int(phase + 44))
        painter.restore()

    @staticmethod
    def _paint_trace_lines(painter: QPainter, width: int, height: int, t: float):
        if width <= 0 or height <= 0:
            return
        pen = QPen(QColor(101, 232, 255, 34), 1)
        painter.setPen(pen)
        rows = 7
        for idx in range(rows):
            y = int((height * (idx + 1) / (rows + 1)) + math.sin(t + idx) * 12)
            start = int((t * 55 + idx * 97) % (width + 260) - 180)
            painter.drawLine(start, y, min(width, start + 120), y)
            painter.drawLine(
                min(width, start + 120),
                y,
                min(width, start + 156),
                max(0, y - 20),
            )
            painter.drawLine(
                min(width, start + 156),
                max(0, y - 20),
                min(width, start + 260),
                max(0, y - 20),
            )

    @staticmethod
    def _paint_edge_glow(painter: QPainter, width: int, height: int, t: float):
        if width <= 0 or height <= 0:
            return
        alpha = 26 + int(12 * (0.5 + 0.5 * math.sin(t * 1.2)))
        top = QLinearGradient(0, 0, width, 0)
        top.setColorAt(0.0, QColor(54, 183, 255, 0))
        top.setColorAt(0.24, QColor(54, 183, 255, alpha))
        top.setColorAt(0.72, QColor(69, 212, 131, alpha))
        top.setColorAt(1.0, QColor(69, 212, 131, 0))
        painter.fillRect(0, 0, width, 2, QBrush(top))

        bottom = QLinearGradient(0, height - 2, width, height - 2)
        bottom.setColorAt(0.0, QColor(69, 212, 131, 0))
        bottom.setColorAt(0.5, QColor(54, 183, 255, alpha // 2))
        bottom.setColorAt(1.0, QColor(69, 212, 131, 0))
        painter.fillRect(0, max(0, height - 2), width, 2, QBrush(bottom))


class _BackdropResizer(QObject):
    def __init__(self, backdrop: TechBackdrop, parent=None):
        super().__init__(parent)
        self._backdrop = backdrop

    def eventFilter(self, obj, event):
        if event.type() in {QEvent.Resize, QEvent.Show}:
            self._backdrop.setGeometry(obj.rect())
            self._backdrop.lower()
        return super().eventFilter(obj, event)


def attach_ambient_backdrop(container: QWidget) -> TechBackdrop:
    """Attach an animated backdrop behind a layout-managed widget tree."""
    backdrop = TechBackdrop(container)
    backdrop.setGeometry(container.rect())
    backdrop.lower()
    backdrop.show()
    resizer = _BackdropResizer(backdrop, container)
    container.installEventFilter(resizer)
    container._ambient_backdrop = backdrop
    container._ambient_backdrop_resizer = resizer
    return backdrop


def fade_in_window(widget: QWidget, duration: int = 220):
    """Fade in a top-level window after it is shown."""
    widget.setWindowOpacity(0.0)
    anim = QPropertyAnimation(widget, b'windowOpacity', widget)
    anim.setDuration(duration)
    anim.setStartValue(0.0)
    anim.setEndValue(1.0)
    anim.setEasingCurve(QEasingCurve.OutCubic)
    widget._intro_animation = anim
    QTimer.singleShot(0, anim.start)


def _accent_for_button(button: QAbstractButton) -> QColor:
    name = button.objectName()
    if name in {'dangerBtn', 'windowCloseBtn'}:
        return ACCENT_RED
    if name in {'toggleBtn', 'successBtn', 'taskNavBtn', 'taskPickerBtn'}:
        return ACCENT_GREEN
    return ACCENT_BLUE


def _button_checked(button: QAbstractButton) -> bool:
    try:
        return bool(button.isCheckable() and button.isChecked())
    except RuntimeError:
        return False
