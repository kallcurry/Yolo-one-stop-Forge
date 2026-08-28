#!/usr/bin/env python3
"""Image File Manager — PyQt5 application entry point."""

import os
import sys

# Make sure the project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from PyQt5.QtWidgets import QApplication, QFileDialog, QVBoxLayout, QWidget

from app.views.main_window import MainWindow
from app.views.model_management import ModelManagementView
from app.views.training_management import TrainingManagementView
from app.views.evaluation_management import EvaluationManagementView
from app.views.inference_center import InferenceCenterView
from app.views.dir_tree import DirTreePanel
from app.views.image_viewer import ImageViewer
from app.views.detail_panel import DetailPanel
from app.views.file_list_panel import FileListPanel
from app.views.ui_effects import fade_in_window
from app.controllers.app_controller import AppController
from app.tools import (
    dataset_stats, train_val_stats, find_keypoint,
    merge_and_split, swap_labels, file_count, file_match,
    check_duplicates, convert_validate,
)


def _load_stylesheet() -> str:
    path = os.path.join(PROJECT_ROOT, 'resources', 'style.qss')
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''


def _install_excepthook():
    """Unhandled exceptions must not abort the Qt app with a core dump."""
    import traceback

    def _hook(exc_type, exc_value, exc_tb):
        traceback.print_exc()
        try:
            from PyQt5.QtWidgets import QApplication, QMessageBox
            message = (
                f'发生未处理错误，应用将继续运行。\n\n'
                f'{exc_type.__name__}: {exc_value}\n\n'
                f'详细信息见终端日志。'
            )
            if QApplication.instance() is not None:
                QMessageBox.critical(None, '未处理错误', message)
        except Exception:  # noqa: BLE001
            pass

    sys.excepthook = _hook


def sanitize_qt_environment():
    """Drop OpenCV's incompatible Qt plugin paths before QApplication boots.

    OpenCV wheels ship their own Qt plugins under ``cv2/qt/plugins``; if a
    shell/launcher has exported ``QT_QPA_PLATFORM_PLUGIN_PATH`` pointing there,
    the Qt platform plugin fails to load (xcb incompatibility) and the app
    aborts at startup.  This mirrors the cleanup the label-tool launcher does
    for its subprocess and makes ``python main.py`` safe in polluted shells.
    """
    for key in ('QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_PLUGIN_PATH'):
        value = os.environ.get(key, '')
        if 'cv2' in value.replace('\\', '/'):
            os.environ.pop(key, None)
            print(f'[启动保护] 已移除不兼容的 OpenCV Qt 插件路径: {key}')
    # 无头环境提示（若用户在无显示器终端运行）
    if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        print('[启动保护] 未检测到显示环境，已启用 offscreen 模式')


def main():
    sanitize_qt_environment()
    _install_excepthook()
    app = QApplication(sys.argv)
    app.setApplicationName('ImageFileManager')
    app.setStyleSheet(_load_stylesheet())

    # Create views
    win = MainWindow()
    dir_tree = DirTreePanel()
    viewer = ImageViewer()
    detail = DetailPanel()
    file_list = FileListPanel()
    model_manager = ModelManagementView()
    training_manager = TrainingManagementView()
    evaluation_manager = EvaluationManagementView()
    inference_center = InferenceCenterView()

    # Embed file list into detail panel (right side)
    detail.set_file_list(file_list)

    # Wrap detail + file list as right panel
    right_panel = QWidget()
    right_layout = QVBoxLayout(right_panel)
    right_layout.setContentsMargins(0, 0, 0, 0)
    right_layout.addWidget(detail)

    # Insert panels into main window
    win.set_dir_tree(dir_tree)
    win.set_image_viewer(viewer)
    win.set_detail_panel(right_panel)
    win.set_model_manager(model_manager)
    win.set_training_manager(training_manager)
    win.set_evaluation_manager(evaluation_manager)
    win.set_inference_center(inference_center)

    # Set default splitter ratios: tree 20%, image 55%, detail 25%
    win.top_splitter.setSizes([280, 770, 350])

    # Create controller — wires everything
    ctrl = AppController(
        win, dir_tree, viewer, detail, file_list, model_manager,
        training_manager, evaluation_manager, inference_center,
    )
    model_manager.evaluate_requested.connect(ctrl.open_evaluation)
    model_manager.inference_requested.connect(ctrl.open_inference)
    # 模型管理仓库与评估/推理中心联动：仓库切换后模型列表同步
    model_manager.directory_changed.connect(
        lambda repo: (
            evaluation_manager.set_model_repository(repo),
            inference_center.set_model_repository(repo),
        )
    )

    # Annotation mode: sync button + A-key shortcut ↔ viewer
    def _on_annotation_cycle():
        viewer.cycle_annotation_mode()
        win.status_bar.showMessage(
            f'标注显示: {viewer.annotation_mode_name()}', 2000
        )

    viewer.annotation_mode_changed.connect(
        lambda _mode: win.set_annotation_btn_text(viewer.annotation_mode_name())
    )
    viewer.skeleton_visibility_changed.connect(win.set_skeleton_btn_text)
    win.btn_annotation.clicked.connect(_on_annotation_cycle)
    win.btn_skeleton.clicked.connect(lambda _checked=False: viewer.toggle_skeleton())
    win._on_key_a = _on_annotation_cycle

    # Connect menu / shortcut stubs on MainWindow
    win._on_dir_opened = ctrl.open_directory
    win._on_key_1 = viewer.toggle_fit
    win.action_open.triggered.disconnect()
    win.action_open.triggered.connect(ctrl.open_directory_dialog)
    win.action_copy.triggered.connect(ctrl.on_copy)
    win.action_move.triggered.connect(ctrl.on_move)
    win.action_delete.triggered.connect(ctrl.on_delete)
    win.action_rename.triggered.connect(ctrl.on_rename)
    win.action_new_folder.triggered.connect(
        lambda: ctrl.on_new_folder(dir_tree.selected_path() or '')
    )
    win.action_refresh.triggered.connect(ctrl.refresh)

    # Tool menu actions
    win.action_tool_count.triggered.connect(lambda: file_count.create_dialog(win).exec_())
    win.action_tool_match.triggered.connect(lambda: file_match.create_dialog(win).exec_())
    win.action_tool_dupcheck.triggered.connect(lambda: check_duplicates.create_dialog(win).exec_())
    win.action_tool_raw_dupcheck.triggered.connect(
        lambda: check_duplicates.create_raw_dialog(win).exec_()
    )
    win.action_tool_convert.triggered.connect(
        lambda: convert_validate.create_dialog(win).exec_()
    )
    win.action_tool_stats.triggered.connect(lambda: dataset_stats.create_dialog(win).exec_())
    win.action_tool_trainval.triggered.connect(lambda: train_val_stats.create_dialog(win).exec_())
    win.action_tool_findkp.triggered.connect(lambda: find_keypoint.create_dialog(win).exec_())
    win.action_tool_merge.triggered.connect(lambda: merge_and_split.create_dialog(win).exec_())
    win.action_tool_swap.triggered.connect(lambda: swap_labels.create_dialog(win).exec_())

    # Show
    win.show()
    fade_in_window(win)

    # Auto-open last directory, or prompt
    last_dir = ctrl.last_directory()
    if last_dir and os.path.isdir(last_dir):
        ctrl.open_directory(last_dir)
        ctrl.restore_last_selection()
    else:
        ctrl.open_directory_dialog()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
