"""Launch X-AnyLabeling with PyQt5's plugin tree, not OpenCV's bundled Qt."""

from __future__ import annotations

import os
import sys


def configure_qt_plugins():
    """Undo OpenCV wheel's Linux Qt plugin environment override.

    The non-headless OpenCV wheel sets ``QT_QPA_PLATFORM_PLUGIN_PATH`` when it
    is imported. X-AnyLabeling uses PyQt5, so loading OpenCV's bundled Qt
    platform plugin mixes two Qt distributions and makes xcb fail to start.
    """
    import cv2  # noqa: F401 - intentionally triggers OpenCV's environment setup.
    from PyQt5.QtCore import QLibraryInfo

    plugins = QLibraryInfo.location(QLibraryInfo.PluginsPath)
    if not plugins:
        raise RuntimeError('无法定位 PyQt5 Qt 插件目录')
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugins
    os.environ.pop('QT_PLUGIN_PATH', None)


def main(argv: list[str] | None = None) -> int:
    configure_qt_plugins()
    from anylabeling.app import main as anylabeling_main

    if argv is not None:
        sys.argv = [sys.argv[0], *argv]
    return anylabeling_main() or 0


if __name__ == '__main__':
    raise SystemExit(main())
