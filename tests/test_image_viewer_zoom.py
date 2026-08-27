import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PyQt5.QtCore import QPoint, QPointF, Qt
from PyQt5.QtGui import QColor, QPixmap, QWheelEvent
from PyQt5.QtWidgets import QApplication

from app.views.image_viewer import ImageViewer, MAX_SCALE, MIN_SCALE


class ImageViewerZoomTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.viewer = ImageViewer()
        self.viewer.resize(520, 420)
        self.viewer.show()
        self.app.processEvents()
        pixmap = QPixmap(1200, 800)
        pixmap.fill(QColor('#243447'))
        self.viewer._pixmap = pixmap
        self.viewer.fit_to_window()
        self.app.processEvents()

    def tearDown(self):
        self.viewer.hide()
        self.viewer.deleteLater()
        self.app.processEvents()

    def test_zoom_at_cursor_leaves_fit_mode_and_enables_panning(self):
        fit_scale = self.viewer.zoom_scale()
        center = self.viewer.viewport().rect().center()

        self.viewer._zoom_at(center, 3.0)
        self.app.processEvents()

        self.assertGreater(self.viewer.zoom_scale(), fit_scale)
        self.assertFalse(self.viewer._fit_to_window)
        self.assertTrue(
            self.viewer.horizontalScrollBar().maximum() > 0
            or self.viewer.verticalScrollBar().maximum() > 0
        )

    def test_plain_mouse_wheel_zooms_without_control_modifier(self):
        old_scale = self.viewer.zoom_scale()
        center = self.viewer.viewport().rect().center()
        event = QWheelEvent(
            QPointF(center),
            QPointF(self.viewer.viewport().mapToGlobal(center)),
            QPoint(),
            QPoint(0, 120),
            Qt.NoButton,
            Qt.NoModifier,
            Qt.NoScrollPhase,
            False,
        )

        QApplication.sendEvent(self.viewer.viewport(), event)

        self.assertGreater(self.viewer.zoom_scale(), old_scale)
        self.assertTrue(event.isAccepted())

    def test_zoom_scale_is_bounded(self):
        center = QPoint(100, 100)

        self.viewer._zoom_at(center, 1_000_000.0)
        self.assertEqual(self.viewer.zoom_scale(), MAX_SCALE)

        self.viewer._zoom_at(center, 0.0000001)
        self.assertEqual(self.viewer.zoom_scale(), MIN_SCALE)

    def test_fit_scale_tracks_viewport_resize(self):
        first_scale = self.viewer.zoom_scale()

        self.viewer.resize(820, 620)
        self.app.processEvents()

        self.assertGreater(self.viewer.zoom_scale(), first_scale)


if __name__ == '__main__':
    unittest.main()
