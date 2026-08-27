import os
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PyQt5.QtCore import QPoint, QRect, QSettings, Qt
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication

from app.controllers.app_controller import AppController
from app.views.detail_panel import DetailPanel
from app.views.dir_tree import DirTreePanel
from app.views.file_list_panel import FileListPanel
from app.views.image_viewer import ImageViewer
from app.views.main_window import MainWindow


class _Screen:
    def __init__(self, available_geometry):
        self._available_geometry = QRect(available_geometry)

    def availableGeometry(self):
        return QRect(self._available_geometry)


class MainWindowGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings_dir = tempfile.TemporaryDirectory()
        QSettings.setPath(
            QSettings.NativeFormat,
            QSettings.UserScope,
            cls.settings_dir.name,
        )
        QSettings.setPath(
            QSettings.IniFormat,
            QSettings.UserScope,
            cls.settings_dir.name,
        )
        cls.app = QApplication.instance() or QApplication([])

    @classmethod
    def tearDownClass(cls):
        cls.settings_dir.cleanup()

    def setUp(self):
        settings = QSettings('FilesProcessQT', 'ImageManager')
        settings.clear()
        settings.sync()

    def _create_window(self):
        window = MainWindow()
        window.show()
        self.app.processEvents()
        self.app.processEvents()
        self.addCleanup(self._dispose_window, window)
        return window

    def _dispose_window(self, window):
        try:
            window.hide()
            window.deleteLater()
        except RuntimeError:
            return
        self.app.processEvents()

    def test_maximize_then_restore_returns_to_exact_normal_geometry(self):
        window = self._create_window()
        screen = _Screen(QRect(0, 0, 1920, 1080))
        window._current_screen = lambda: screen
        original = QRect(120, 140, 1320, 780)
        window.setGeometry(original)
        self.app.processEvents()
        window._capture_normal_geometry()

        window._toggle_maximized()
        self.app.processEvents()
        self.assertTrue(window.isMaximized())

        window._toggle_maximized()
        self.app.processEvents()
        self.app.processEvents()

        self.assertFalse(window.isMaximized())
        self.assertEqual(window.geometry(), original)

    def test_full_screen_geometry_is_not_valid_startup_geometry(self):
        window = self._create_window()
        available = QRect(0, 0, 2560, 1440)

        self.assertFalse(window._is_usable_restore_geometry(
            QRect(0, 0, 2560, 1440), available
        ))
        self.assertFalse(window._is_usable_restore_geometry(
            QRect(0, 0, 2464, 1344), available
        ))
        self.assertTrue(window._is_usable_restore_geometry(
            QRect(580, 295, 1400, 850), available
        ))

    def test_startup_rejects_saved_full_screen_geometry(self):
        settings = QSettings('FilesProcessQT', 'ImageManager')
        settings.setValue('normalGeometry', QRect(0, 0, 1920, 1080))
        settings.sync()
        screen = _Screen(QRect(0, 0, 1920, 1080))

        with patch.object(MainWindow, '_current_screen', return_value=screen):
            window = self._create_window()

        self.assertEqual(window.geometry(), QRect(260, 115, 1400, 850))

    def test_frameless_resize_fallback_supports_edges_and_corners(self):
        window = self._create_window()
        start = QRect(100, 100, 1400, 850)

        window._resize_from_handle(
            Qt.TopEdge | Qt.LeftEdge,
            start,
            QPoint(100, 50),
        )
        self.assertEqual(window.geometry(), QRect(200, 150, 1300, 800))

        window._resize_from_handle(
            Qt.BottomEdge | Qt.RightEdge,
            start,
            QPoint(100, 80),
        )
        self.assertEqual(window.geometry(), QRect(100, 100, 1500, 930))

        window._resize_from_handle(Qt.LeftEdge, start, QPoint(500, 0))
        self.assertEqual(window.width(), window.minimumWidth())

    def test_last_data_selection_is_restored_after_reopen(self):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        data_root = os.path.join(root.name, 'dataset')
        image_dir = os.path.join(data_root, 'images')
        os.makedirs(image_dir)
        for name in ('frame_001.png', 'frame_002.png'):
            QImage(4, 4, QImage.Format_RGB32).save(
                os.path.join(image_dir, name)
            )

        def create_session():
            window = MainWindow()
            tree = DirTreePanel()
            viewer = ImageViewer()
            detail = DetailPanel()
            file_list = FileListPanel()
            window.set_dir_tree(tree)
            window.set_image_viewer(viewer)
            window.set_detail_panel(detail)
            controller = AppController(
                window, tree, viewer, detail, file_list
            )
            window.show()
            self.app.processEvents()
            return window, tree, controller

        window, tree, controller = create_session()
        self.addCleanup(self._dispose_window, window)
        controller.open_directory(data_root)
        self.assertTrue(tree.select_path(image_dir, emit=True))
        controller._jump_to_image(1)
        window.close()
        self.app.processEvents()

        reopened, reopened_tree, reopened_controller = create_session()
        self.addCleanup(self._dispose_window, reopened)
        reopened_controller.open_directory(reopened_controller.last_directory())

        self.assertTrue(reopened_controller.restore_last_selection())
        self.assertEqual(
            os.path.realpath(reopened_tree.selected_path()),
            os.path.realpath(image_dir),
        )
        self.assertEqual(reopened_controller._current_index, 1)


if __name__ == '__main__':
    unittest.main()
