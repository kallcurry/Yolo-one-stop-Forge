import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PyQt5.QtCore import QEvent
from PyQt5.QtWidgets import QApplication, QPushButton

from app.views.ui_effects import HoverGlow


class HoverGlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_late_event_is_safe_after_python_state_is_unavailable(self):
        glow = HoverGlow()
        button = QPushButton('test')
        del glow._effects

        self.assertFalse(glow.eventFilter(button, QEvent(QEvent.Enter)))


if __name__ == '__main__':
    unittest.main()
