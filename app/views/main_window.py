"""Main application window — assembles the 4-panel layout with menu and bottom bar."""

from pathlib import Path

from PyQt5.QtCore import (
    QEvent,
    QPoint,
    QRect,
    QRectF,
    QSettings,
    QSize,
    QTimer,
    Qt,
    pyqtSignal,
)
from PyQt5.QtGui import QColor, QGuiApplication, QKeySequence, QPainterPath, QRegion
from PyQt5.QtWidgets import (
    QAction,
    QButtonGroup,
    QFileDialog,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.views.ui_effects import HoverGlow, attach_ambient_backdrop


class _WindowResizeHandle(QWidget):
    """Transparent border handle for resizing a frameless top-level window."""

    def __init__(self, window, edges, cursor):
        super().__init__(window)
        self._window = window
        self._edges = edges
        self._press_pos = None
        self._press_geometry = None
        self._native_resize = False
        self.setCursor(cursor)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton or self._window._is_window_maximized():
            event.ignore()
            return
        self._press_pos = event.globalPos()
        self._press_geometry = QRect(self._window.geometry())
        handle = self._window.windowHandle()
        self._native_resize = bool(
            handle is not None
            and handle.startSystemResize(self._edges)
        )
        event.accept()

    def mouseMoveEvent(self, event):
        if (
            not self._native_resize
            and self._press_pos is not None
            and event.buttons() & Qt.LeftButton
        ):
            self._window._resize_from_handle(
                self._edges,
                self._press_geometry,
                event.globalPos() - self._press_pos,
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._press_pos = None
        self._press_geometry = None
        self._native_resize = False
        super().mouseReleaseEvent(event)


class MainWindow(QMainWindow):
    """Top-level window with menu bar, 4-panel layout, and bottom action bar."""

    task_selected = pyqtSignal(str)
    module_selected = pyqtSignal(str)
    model_return_requested = pyqtSignal()
    training_return_requested = pyqtSignal()
    about_to_close = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle('视觉数据管理平台')
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setMinimumSize(1200, 700)
        self._restore_window_size = QSize(1400, 850)
        self._window_drag_pos = None
        self._window_drag_targets = set()
        self._current_module = 'data'
        self._current_task_type = 'pose'
        self._data_context_kind = None
        self._data_actions_available = False
        self._data_navigation_available = False
        self._corner_radius = 18
        self._panel_glows = []
        self._normal_geometry_before_maximize = None
        self._last_chrome_maximized_style = None
        self._max_restore_drag_ratio = 0.5
        self._max_restore_drag_y = 18
        self._window_transition = None
        self._pending_restore_geometry = None
        self._pending_restore_drag_pos = None
        self._restore_completion_scheduled = False
        self._applying_restore_geometry = False
        self._resize_handles = []
        self._resize_handle_thickness = 8
        self._resize_corner_size = 16
        self.resize(self._restore_window_size)
        # Build actions first; the native menu bar is hidden and exposed in header.
        self._setup_menus()

        # -- Central container --
        central = QWidget()
        central.setObjectName('appShell')
        central.setAttribute(Qt.WA_StyledBackground, True)
        self._central_shell = central
        self.setCentralWidget(central)
        self._ambient_backdrop = attach_ambient_backdrop(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 4)
        root_layout.setSpacing(6)
        self._root_layout = root_layout

        self._setup_platform_header(root_layout)

        # -- Module workspaces --
        self.workspace_stack = QStackedWidget()
        self.workspace_stack.setObjectName('workspaceStack')
        self.data_workspace = QWidget()
        self.data_workspace.setObjectName('dataWorkspace')
        self.data_workspace_layout = QVBoxLayout(self.data_workspace)
        self.data_workspace_layout.setContentsMargins(0, 0, 0, 0)
        self.data_workspace_layout.setSpacing(6)

        self.data_source_context_bar = QWidget()
        self.data_source_context_bar.setObjectName('dataSourceContextBar')
        context_layout = QHBoxLayout(self.data_source_context_bar)
        context_layout.setContentsMargins(10, 5, 8, 5)
        context_layout.setSpacing(9)
        self.btn_return_to_model = QPushButton('‹  返回模型详情')
        self.btn_return_to_model.setObjectName('returnToModelBtn')
        self.btn_return_to_model.setToolTip('返回刚才查看的模型训练记录')
        self.btn_return_to_model.clicked.connect(self._emit_data_context_return)
        context_layout.addWidget(self.btn_return_to_model)
        context_badge = QLabel('MODEL DATA')
        context_badge.setObjectName('dataSourceContextBadge')
        context_layout.addWidget(context_badge)
        self.lbl_data_source_context = QLabel('-')
        self.lbl_data_source_context.setObjectName('dataSourceContextText')
        self.lbl_data_source_context.setTextInteractionFlags(Qt.TextSelectableByMouse)
        context_layout.addWidget(self.lbl_data_source_context, 1)
        self.btn_close_data_context = QToolButton()
        self.btn_close_data_context.setObjectName('dataSourceContextCloseBtn')
        self.btn_close_data_context.setText('×')
        self.btn_close_data_context.setToolTip('关闭来源提示')
        self.btn_close_data_context.clicked.connect(self.hide_data_source_context)
        context_layout.addWidget(self.btn_close_data_context)
        self.data_source_context_bar.hide()
        self.data_workspace_layout.addWidget(self.data_source_context_bar, 0)

        # -- Data workspace: horizontal splitter (tree | image | detail) --
        self.top_splitter = QSplitter(Qt.Horizontal)
        self.top_splitter.setObjectName('mainContentSplitter')
        self.data_workspace_layout.addWidget(self.top_splitter, 1)

        # -- Bottom bar: navigation + actions --
        self.bottom_bar = QWidget()
        self.bottom_bar.setObjectName('bottomBar')
        self.bottom_bar.setMinimumHeight(44)
        self.bottom_bar.setMaximumHeight(52)
        bottom = QHBoxLayout(self.bottom_bar)
        bottom.setContentsMargins(10, 5, 10, 5)
        bottom.setSpacing(8)

        # Navigation (left)
        self.btn_prev = QPushButton('◀')
        self.btn_prev.setObjectName('navBtn')
        self.btn_prev.setToolTip('上一张 (←)')
        self.btn_prev.setEnabled(False)

        self.btn_next = QPushButton('▶')
        self.btn_next.setObjectName('navBtn')
        self.btn_next.setToolTip('下一张 (→)')
        self.btn_next.setEnabled(False)

        self.lbl_counter = QLabel('没有图片')
        self.lbl_counter.setObjectName('imageCount')

        self.btn_open_label_tool = QPushButton('修改当前文件标签')
        self.btn_open_label_tool.setObjectName('primaryBtn')
        self.btn_open_label_tool.setToolTip('用 X-AnyLabeling 修改当前图片和标注文件')

        bottom.addWidget(self.btn_prev)
        bottom.addWidget(self.btn_next)
        bottom.addWidget(self.lbl_counter)
        bottom.addWidget(self.btn_open_label_tool)
        self.btn_batch_annotate = QPushButton('标注本文件夹')
        self.btn_batch_annotate.setObjectName('fileOpBtn')
        self.btn_batch_annotate.setToolTip(
            '一次打开 X-AnyLabeling 窗口，左侧图库加载本文件夹全部图片；'
            '点击任意图片即可切换标注，保存的 JSON 写入标注集目录。'
        )
        bottom.addWidget(self.btn_batch_annotate)

        sep_display = QWidget()
        sep_display.setObjectName('barSeparator')
        sep_display.setFixedWidth(1)
        bottom.addWidget(sep_display)

        # Annotation mode button
        self.btn_annotation = QPushButton('🏷 仅矩形框')
        self.btn_annotation.setObjectName('toggleBtn')
        self.btn_annotation.setToolTip(
            '切换标注显示模式 (A键)\n隐藏 → 仅矩形框 → 矩形框 + 关键点'
        )
        bottom.addWidget(self.btn_annotation)

        self.btn_skeleton = QPushButton('骨架线: 关')
        self.btn_skeleton.setObjectName('toggleBtn')
        self.btn_skeleton.setToolTip('显示/隐藏关键点骨架连线')
        bottom.addWidget(self.btn_skeleton)
        bottom.addStretch()

        # Separator
        sep = QWidget()
        sep.setObjectName('barSeparator')
        sep.setFixedWidth(1)
        bottom.addWidget(sep)

        # Action buttons (right)
        self.btn_copy = QPushButton('📋 复制')
        self.btn_copy.setObjectName('fileOpBtn')
        self.btn_copy.setToolTip('复制图片到其他目录')
        self.btn_move = QPushButton('📁 移动')
        self.btn_move.setObjectName('fileOpBtn')
        self.btn_move.setToolTip('移动图片到其他目录')
        self.btn_delete = QPushButton('🗑 删除')
        self.btn_delete.setObjectName('dangerBtn')
        self.btn_delete.setToolTip('删除选中图片')
        self.btn_rename = QPushButton('✏️ 重命名')
        self.btn_rename.setObjectName('fileOpBtn')
        self.btn_rename.setToolTip('重命名当前图片 (F2)')
        self.btn_new_folder = QPushButton('📂 新建文件夹')
        self.btn_new_folder.setObjectName('fileOpBtn')
        self._set_actions_enabled(False)

        for btn in [
            self.btn_prev,
            self.btn_next,
            self.btn_annotation,
            self.btn_skeleton,
            self.btn_open_label_tool,
            self.btn_copy,
            self.btn_move,
            self.btn_delete,
            self.btn_rename,
            self.btn_new_folder,
        ]:
            btn.setMinimumHeight(28)
            btn.setMaximumHeight(34)

        bottom.addWidget(self.btn_copy)
        bottom.addWidget(self.btn_move)
        bottom.addWidget(self.btn_delete)
        bottom.addWidget(self.btn_rename)
        bottom.addWidget(self.btn_new_folder)

        self.data_workspace_layout.addWidget(self.bottom_bar, 0)
        self.workspace_stack.addWidget(self.data_workspace)
        self._module_pages = {'data': self.data_workspace}
        root_layout.addWidget(self.workspace_stack, 1)

        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.setSizeGripEnabled(False)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('就绪 — 按 Ctrl+O 打开数据目录')

        # Shortcuts
        self._setup_shortcuts()

        # Restore geometry
        self._restore_state()

        self._setup_visual_effects()
        self._setup_resize_handles()
        QTimer.singleShot(0, self._normalize_initial_geometry)

    def _setup_platform_header(self, root_layout: QVBoxLayout):
        self.platform_bar = QWidget()
        self.platform_bar.setObjectName('platformBar')
        self.platform_bar.setMinimumHeight(58)
        self.platform_bar.setMaximumHeight(66)
        platform = QVBoxLayout(self.platform_bar)
        platform.setContentsMargins(12, 8, 12, 8)
        platform.setSpacing(0)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(12)

        self.lbl_window_title = QLabel('视觉数据管理平台', self.platform_bar)
        self.lbl_window_title.setObjectName('platformTitle')
        self.lbl_window_title.setAlignment(Qt.AlignCenter)
        self.lbl_window_title.setFixedSize(220, 34)

        self.module_band = QWidget()
        self.module_band.setObjectName('moduleNavBand')
        self.module_band.setMinimumHeight(36)
        self.module_band.setMaximumHeight(38)
        module_row = QHBoxLayout(self.module_band)
        module_row.setContentsMargins(0, 0, 0, 0)
        module_row.setSpacing(4)

        self.module_button_group = QButtonGroup(self)
        self.module_button_group.setExclusive(True)
        self.module_buttons = {}
        modules = [
            ('data', '数据管理', '数据管理', True),
            ('model', '模型管理', '模型管理', True),
            ('train', '训练中心', '训练中心', True),
            ('eval', '评估中心', '评估中心', True),
            ('infer', '推理中心', '推理中心', True),
        ]
        for idx, (module_id, label, tooltip, enabled) in enumerate(modules):
            btn = QPushButton(label)
            btn.setObjectName('moduleNavBtn')
            btn.setCheckable(True)
            btn.setFixedWidth(104)
            btn.setMinimumHeight(32)
            btn.setMaximumHeight(34)
            btn.setEnabled(enabled)
            if not enabled:
                btn.setToolTip(f'{tooltip}规划中，当前先完成数据管理闭环')
            else:
                btn.setToolTip(tooltip)
            self.module_button_group.addButton(btn, idx)
            self.module_buttons[module_id] = btn
            module_row.addWidget(btn)
            if enabled:
                btn.clicked.connect(
                    lambda _checked=False, module=module_id:
                    self._select_module(module)
                )
        self.module_buttons['data'].setChecked(True)
        top_row.addWidget(self.module_band)

        self.task_menu = QMenu(self)
        self.task_actions = {}
        self._task_labels = {}
        tasks = [
            ('pose', '姿态'),
            ('detection', '检测'),
            ('segmentation', '分割'),
            ('obb', 'OBB'),
        ]
        for task_type, label in tasks:
            action = QAction(label, self)
            action.setCheckable(True)
            action.setToolTip(self._task_display_name(task_type))
            action.triggered.connect(
                lambda _checked=False, task=task_type: self._select_task(task)
            )
            self.task_menu.addAction(action)
            self.task_actions[task_type] = action
            self._task_labels[task_type] = label

        self.btn_task_picker = QToolButton()
        self.btn_task_picker.setObjectName('taskPickerBtn')
        self.btn_task_picker.setToolTip('切换数据任务')
        self.btn_task_picker.setPopupMode(QToolButton.InstantPopup)
        self.btn_task_picker.setMenu(self.task_menu)
        self.btn_task_picker.setMinimumWidth(112)
        self.btn_task_picker.setMaximumWidth(124)
        self.btn_task_picker.setMinimumHeight(32)
        self.btn_task_picker.setMaximumHeight(34)
        self._select_task('pose', emit=False)

        top_row.addStretch(1)
        self.action_dock = QWidget()
        self.action_dock.setObjectName('headerToolsDock')
        action_row = QHBoxLayout(self.action_dock)
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(6)
        self.workspace_menu = QMenu(self)
        self.workspace_menu.addMenu(self.file_menu)
        self.workspace_menu.addMenu(self.edit_menu)
        self.workspace_menu.addMenu(self.view_menu)
        self.workspace_menu.addMenu(self.tools_menu)
        self.btn_workspace_menu = QToolButton()
        self.btn_workspace_menu.setObjectName('headerMenuBtn')
        self.btn_workspace_menu.setText('操作中心')
        self.btn_workspace_menu.setToolTip('打开文件、编辑、查看与工具菜单')
        self.btn_workspace_menu.setPopupMode(QToolButton.InstantPopup)
        self.btn_workspace_menu.setMenu(self.workspace_menu)
        self.btn_workspace_menu.setMinimumWidth(96)
        self.btn_workspace_menu.setMinimumHeight(32)
        self.btn_workspace_menu.setMaximumHeight(34)
        action_row.addWidget(self.btn_task_picker)
        action_row.addWidget(self.btn_workspace_menu)
        top_row.addWidget(self.action_dock)

        self.window_dock = QWidget()
        self.window_dock.setObjectName('windowChromeDock')
        window_row = QHBoxLayout(self.window_dock)
        window_row.setContentsMargins(0, 0, 0, 0)
        window_row.setSpacing(3)

        self.btn_window_min = self._make_window_button('−', '最小化')
        self.btn_window_max = self._make_window_button('▢', '最大化 / 还原')
        self.btn_window_close = self._make_window_button('×', '关闭')
        self.btn_window_close.setObjectName('windowCloseBtn')

        self.btn_window_min.clicked.connect(self.showMinimized)
        self.btn_window_max.clicked.connect(self._toggle_maximized)
        self.btn_window_close.clicked.connect(self.close)
        window_row.addWidget(self.btn_window_min)
        window_row.addWidget(self.btn_window_max)
        window_row.addWidget(self.btn_window_close)
        top_row.addWidget(self.window_dock)
        platform.addLayout(top_row)

        self._register_window_drag_target(self.platform_bar)
        self._register_window_drag_target(self.lbl_window_title)

        root_layout.addWidget(self.platform_bar, 0)

    # ---- Panel insertion ----

    def set_dir_tree(self, widget: QWidget):
        self.top_splitter.insertWidget(0, widget)

    def set_image_viewer(self, widget: QWidget):
        self.top_splitter.insertWidget(1, widget)

    def set_detail_panel(self, widget: QWidget):
        self.top_splitter.insertWidget(2, widget)

    def set_model_manager(self, widget: QWidget):
        """Install the model-management workspace as a peer of data review."""
        previous = self._module_pages.get('model')
        if previous is not None:
            self.workspace_stack.removeWidget(previous)
            previous.setParent(None)
        self._module_pages['model'] = widget
        self.workspace_stack.addWidget(widget)
        if self._current_module == 'model':
            self.workspace_stack.setCurrentWidget(widget)

    def set_training_manager(self, widget: QWidget):
        """Install the training workspace as a peer platform module."""
        previous = self._module_pages.get('train')
        if previous is not None:
            self.workspace_stack.removeWidget(previous)
            previous.setParent(None)
        self._module_pages['train'] = widget
        self.workspace_stack.addWidget(widget)
        if self._current_module == 'train':
            self.workspace_stack.setCurrentWidget(widget)

    def set_evaluation_manager(self, widget: QWidget):
        """Install the evaluation workspace as a peer platform module."""
        previous = self._module_pages.get('eval')
        if previous is not None:
            self.workspace_stack.removeWidget(previous)
            previous.setParent(None)
        self._module_pages['eval'] = widget
        self.workspace_stack.addWidget(widget)
        if self._current_module == 'eval':
            self.workspace_stack.setCurrentWidget(widget)

    def set_inference_center(self, widget: QWidget):
        """Install the inference workbench as a peer platform module."""
        previous = self._module_pages.get('infer')
        if previous is not None:
            self.workspace_stack.removeWidget(previous)
            previous.setParent(None)
        self._module_pages['infer'] = widget
        self.workspace_stack.addWidget(widget)
        if self._current_module == 'infer':
            self.workspace_stack.setCurrentWidget(widget)

    # ---- Menus ----

    def _setup_menus(self):
        menu = self.menuBar()

        # File
        file_menu = menu.addMenu('&文件')
        act_open = QAction('&打开目录...', self)
        act_open.setShortcut(QKeySequence('Ctrl+O'))
        act_open.triggered.connect(self._on_open_directory)
        file_menu.addAction(act_open)
        file_menu.addSeparator()
        act_exit = QAction('&退出', self)
        act_exit.setShortcut(QKeySequence('Ctrl+Q'))
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # Edit
        edit_menu = menu.addMenu('&编辑')
        act_copy = QAction('&复制', self)
        act_copy.setShortcut(QKeySequence('Ctrl+C'))
        edit_menu.addAction(act_copy)

        act_move = QAction('移动到...', self)
        act_move.setShortcut(QKeySequence('Ctrl+X'))
        edit_menu.addAction(act_move)

        act_delete = QAction('&删除', self)
        act_delete.setShortcut(QKeySequence('Delete'))
        edit_menu.addAction(act_delete)

        act_rename = QAction('重命名', self)
        act_rename.setShortcut(QKeySequence('F2'))
        edit_menu.addAction(act_rename)
        edit_menu.addSeparator()

        act_new_folder = QAction('新建文件夹', self)
        act_new_folder.setShortcut(QKeySequence('Ctrl+Shift+N'))
        edit_menu.addAction(act_new_folder)

        # View
        view_menu = menu.addMenu('&查看')
        act_refresh = QAction('刷新', self)
        act_refresh.setShortcut(QKeySequence('F5'))
        view_menu.addAction(act_refresh)

        # Tools
        tools_menu = menu.addMenu('&工具')
        self.action_open_label_tool = QAction('修改当前文件标签', self)
        self.action_open_label_tool.setEnabled(False)
        tools_menu.addAction(self.action_open_label_tool)
        tools_menu.addSeparator()
        self.action_tool_count = QAction('图片与标注数量统计', self)
        tools_menu.addAction(self.action_tool_count)
        self.action_tool_match = QAction('图片与标注匹配检查', self)
        tools_menu.addAction(self.action_tool_match)
        self.action_tool_dupcheck = QAction('训练/测试集重复检查', self)
        tools_menu.addAction(self.action_tool_dupcheck)
        self.action_tool_raw_dupcheck = QAction('原始数据重复审查', self)
        tools_menu.addAction(self.action_tool_raw_dupcheck)
        self.action_tool_convert = QAction('标注转换与校验（JSON ⇄ YOLO TXT）', self)
        tools_menu.addAction(self.action_tool_convert)
        self.action_tool_model_convert = QAction('模型转换（.pt → ONNX）', self)
        tools_menu.addAction(self.action_tool_model_convert)
        tools_menu.addSeparator()
        self.action_tool_stats = QAction('数据集统计报告', self)
        tools_menu.addAction(self.action_tool_stats)
        self.action_tool_trainval = QAction('训练/测试集统计', self)
        tools_menu.addAction(self.action_tool_trainval)
        self.action_tool_findkp = QAction('查找关键点', self)
        tools_menu.addAction(self.action_tool_findkp)
        tools_menu.addSeparator()
        self.action_tool_merge = QAction('数据集管理（测试集+训练集）', self)
        tools_menu.addAction(self.action_tool_merge)
        self.action_tool_swap = QAction('标签替换', self)
        tools_menu.addAction(self.action_tool_swap)

        # Store references for controller
        self.action_open = act_open
        self.action_copy = act_copy
        self.action_move = act_move
        self.action_delete = act_delete
        self.action_rename = act_rename
        self.action_new_folder = act_new_folder
        self.action_refresh = act_refresh

        for action in [
            self.action_copy,
            self.action_move,
            self.action_delete,
            self.action_rename,
        ]:
            action.setEnabled(False)

        self.file_menu = file_menu
        self.edit_menu = edit_menu
        self.view_menu = view_menu
        self.tools_menu = tools_menu
        menu.setVisible(False)

    def _on_open_directory(self):
        dlg = QFileDialog(self, '选择数据目录')
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setOption(QFileDialog.ShowDirsOnly, True)
        if dlg.exec_():
            path = dlg.selectedFiles()[0]
            self.status_bar.showMessage(f'已打开: {path}')
            # The controller connects to this signal
            self._on_dir_opened(path)

    def _on_dir_opened(self, path: str):
        """Stub — connected by controller in main.py."""
        pass

    # ---- Shortcuts ----

    def _setup_shortcuts(self):
        self.shortcut_next = QS(self, QKeySequence('Right'), self.btn_next.click)
        self.shortcut_prev = QS(self, QKeySequence('Left'), self.btn_prev.click)
        self.shortcut_fit = QS(self, Qt.Key_1, self._on_key_1)
        self.shortcut_annotation = QS(self, Qt.Key_A, self._on_key_a)
        self._data_shortcuts = (
            self.shortcut_next,
            self.shortcut_prev,
            self.shortcut_fit,
            self.shortcut_annotation,
        )

    # Stubs connected by controller
    def _on_key_1(self): pass
    def _on_key_a(self): pass

    def eventFilter(self, watched, event):
        if watched in getattr(self, '_window_drag_targets', set()):
            if (
                event.type() == QEvent.MouseButtonDblClick
                and event.button() == Qt.LeftButton
            ):
                self._toggle_maximized()
                return True
            if (
                event.type() == QEvent.MouseButtonPress
                and event.button() == Qt.LeftButton
            ):
                if self._is_window_maximized():
                    geometry = self.geometry()
                    self._max_restore_drag_ratio = (
                        (event.globalPos().x() - geometry.x())
                        / max(1, geometry.width())
                    )
                    self._max_restore_drag_y = min(
                        max(12, event.pos().y()), max(24, geometry.height())
                    )
                    self._window_drag_pos = None
                else:
                    self._window_drag_pos = (
                        event.globalPos() - self.frameGeometry().topLeft()
                    )
                return False
            if (
                event.type() == QEvent.MouseMove
                and event.buttons() & Qt.LeftButton
            ):
                if self._is_window_maximized():
                    self._restore_for_drag(event.globalPos())
                    return True
                if self._window_drag_pos is not None:
                    self.move(event.globalPos() - self._window_drag_pos)
                return True
            if event.type() == QEvent.MouseButtonRelease:
                self._window_drag_pos = None
        return super().eventFilter(watched, event)

    def _setup_resize_handles(self):
        specs = [
            (Qt.TopEdge, Qt.SizeVerCursor),
            (Qt.BottomEdge, Qt.SizeVerCursor),
            (Qt.LeftEdge, Qt.SizeHorCursor),
            (Qt.RightEdge, Qt.SizeHorCursor),
            (Qt.TopEdge | Qt.LeftEdge, Qt.SizeFDiagCursor),
            (Qt.TopEdge | Qt.RightEdge, Qt.SizeBDiagCursor),
            (Qt.BottomEdge | Qt.LeftEdge, Qt.SizeBDiagCursor),
            (Qt.BottomEdge | Qt.RightEdge, Qt.SizeFDiagCursor),
        ]
        self._resize_handles = [
            _WindowResizeHandle(self, edges, cursor)
            for edges, cursor in specs
        ]
        self._layout_resize_handles()

    def _layout_resize_handles(self):
        handles = getattr(self, '_resize_handles', [])
        if not handles:
            return
        enabled = not self._is_window_maximized() and not self.isFullScreen()
        if not enabled:
            for handle in handles:
                handle.hide()
            return

        width = self.width()
        height = self.height()
        thickness = min(self._resize_handle_thickness, width, height)
        corner = min(self._resize_corner_size, width // 2, height // 2)
        rects = [
            QRect(corner, 0, max(0, width - 2 * corner), thickness),
            QRect(corner, height - thickness, max(0, width - 2 * corner), thickness),
            QRect(0, corner, thickness, max(0, height - 2 * corner)),
            QRect(width - thickness, corner, thickness, max(0, height - 2 * corner)),
            QRect(0, 0, corner, corner),
            QRect(width - corner, 0, corner, corner),
            QRect(0, height - corner, corner, corner),
            QRect(width - corner, height - corner, corner, corner),
        ]
        for handle, rect in zip(handles, rects):
            handle.setGeometry(rect)
            handle.show()
            handle.raise_()

    def _resize_from_handle(self, edges, start: QRect, delta: QPoint):
        if not isinstance(start, QRect) or not start.isValid():
            return
        left, top = start.left(), start.top()
        right, bottom = start.right(), start.bottom()
        min_width = self.minimumWidth()
        min_height = self.minimumHeight()

        if edges & Qt.LeftEdge:
            left = min(left + delta.x(), right - min_width + 1)
        if edges & Qt.RightEdge:
            right = max(right + delta.x(), left + min_width - 1)
        if edges & Qt.TopEdge:
            top = min(top + delta.y(), bottom - min_height + 1)
        if edges & Qt.BottomEdge:
            bottom = max(bottom + delta.y(), top + min_height - 1)
        self.setGeometry(QRect(QPoint(left, top), QPoint(right, bottom)))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._layout_resize_handles()
        self._capture_normal_geometry()
        self._sync_chrome_geometry()

    def moveEvent(self, event):
        super().moveEvent(event)
        self._capture_normal_geometry()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.WindowStateChange:
            if self.isMaximized():
                self._window_transition = None
                self.btn_window_max.setText('▣')
                self.clearMask()
            elif self._window_transition == 'restoring':
                self._schedule_restore_completion()
            else:
                self._window_transition = None
                self.btn_window_max.setText('▢')
            self._sync_chrome_geometry()

    def _register_window_drag_target(self, widget: QWidget):
        widget.installEventFilter(self)
        self._window_drag_targets.add(widget)

    def _make_window_button(self, text: str, tooltip: str) -> QToolButton:
        button = QToolButton()
        button.setObjectName('windowChromeBtn')
        button.setText(text)
        button.setToolTip(tooltip)
        button.setFixedSize(28, 28)
        return button

    def _toggle_maximized(self):
        if self._is_window_maximized():
            self._restore_window()
        else:
            self._maximize_window()
        self._sync_chrome_geometry()

    def _sync_chrome_geometry(self):
        self._position_window_title()
        self._sync_shell_margins()
        self._update_window_shape()
        self._layout_resize_handles()

    def _position_window_title(self):
        if not hasattr(self, 'lbl_window_title') or not hasattr(self, 'platform_bar'):
            return
        bar_rect = self.platform_bar.rect()
        if bar_rect.isEmpty():
            return
        title_w = self.lbl_window_title.width()
        title_h = self.lbl_window_title.height()
        desired_x = max(12, (bar_rect.width() - title_w) // 2)
        left_bound = 12
        right_bound = bar_rect.width() - 12
        if hasattr(self, 'module_band') and self.module_band.geometry().isValid():
            left_bound = max(left_bound, self.module_band.geometry().right() + 14)
        if hasattr(self, 'action_dock') and self.action_dock.geometry().isValid():
            right_bound = min(right_bound, self.action_dock.geometry().left() - 14)
        if right_bound - left_bound >= title_w:
            title_x = max(left_bound, min(desired_x, right_bound - title_w))
        else:
            title_x = desired_x
        title_y = max(0, (bar_rect.height() - title_h) // 2)
        self.lbl_window_title.setGeometry(title_x, title_y, title_w, title_h)
        self.lbl_window_title.raise_()

    def _is_window_maximized(self) -> bool:
        return bool(
            self.isMaximized() or self._window_transition == 'maximizing'
        )

    def _maximize_window(self):
        if self.isMaximized() or self._window_transition == 'maximizing':
            return
        screen = self._current_screen()
        available = screen.availableGeometry() if screen else self.geometry()
        current = QRect(self.geometry())
        if current.isValid():
            self._normal_geometry_before_maximize = current
        else:
            self._normal_geometry_before_maximize = (
                self._fixed_restore_geometry(available)
            )
        self._window_transition = 'maximizing'
        self.clearMask()
        self.btn_window_max.setText('▣')
        self.showMaximized()

    def _restore_window(self):
        target = self._restore_target_geometry()
        self._begin_restore(target)

    def _begin_restore(self, target: QRect, drag_pos=None):
        self._pending_restore_geometry = QRect(target)
        self._pending_restore_drag_pos = (
            QPoint(drag_pos) if drag_pos is not None else None
        )
        self._window_transition = 'restoring'
        self.btn_window_max.setText('▢')
        self.showNormal()
        if not self.isMaximized():
            self._schedule_restore_completion()

    def _schedule_restore_completion(self):
        if self._restore_completion_scheduled:
            return
        self._restore_completion_scheduled = True
        QTimer.singleShot(0, self._complete_restore)

    def _complete_restore(self):
        self._restore_completion_scheduled = False
        if self._window_transition != 'restoring' or self.isMaximized():
            return
        target = self._pending_restore_geometry
        if not isinstance(target, QRect) or not target.isValid():
            target = self._restore_target_geometry()
        target = QRect(target)
        drag_pos = self._pending_restore_drag_pos

        self._applying_restore_geometry = True
        self.setGeometry(target)
        self._applying_restore_geometry = False

        self._normal_geometry_before_maximize = QRect(target)
        self._pending_restore_geometry = None
        self._pending_restore_drag_pos = None
        self._window_transition = None
        if drag_pos is not None:
            self._window_drag_pos = drag_pos - target.topLeft()
        self._sync_chrome_geometry()

    def _restore_for_drag(self, global_pos):
        target = self._restore_target_geometry()
        ratio = min(max(self._max_restore_drag_ratio, 0.12), 0.88)
        x = global_pos.x() - int(target.width() * ratio)
        y = global_pos.y() - min(max(self._max_restore_drag_y, 12), 42)
        target.moveTopLeft(self._clamp_top_left(x, y, target))
        self._begin_restore(target, global_pos)

    def _restore_target_geometry(self) -> QRect:
        screen = self._current_screen()
        available = screen.availableGeometry() if screen else self.geometry()
        target = self._normal_geometry_before_maximize
        if isinstance(target, QRect) and target.isValid():
            return self._clamp_rect_to_available(QRect(target), available)
        return self._fixed_restore_geometry(available)

    def _fixed_restore_geometry(self, available: QRect) -> QRect:
        if available.isEmpty():
            return QRect(self.pos(), self._restore_window_size)
        width = min(self._restore_window_size.width(), available.width() - 96)
        height = min(self._restore_window_size.height(), available.height() - 96)
        width = max(self.minimumWidth(), width)
        height = max(self.minimumHeight(), height)
        width = min(width, available.width())
        height = min(height, available.height())
        x = available.x() + max(0, (available.width() - width) // 2)
        y = available.y() + max(0, (available.height() - height) // 2)
        return QRect(x, y, width, height)

    def _is_usable_restore_geometry(self, target: QRect, available: QRect) -> bool:
        if not target.isValid() or available.isEmpty():
            return target.isValid()
        if self._is_near_available_geometry(target, available):
            return False
        return (
            target.width() <= available.width() - 96
            and target.height() <= available.height() - 72
        )

    def _clamp_rect_to_available(self, rect: QRect, available: QRect) -> QRect:
        if available.isEmpty():
            return rect

        # 确保宽度和高度不会占满整个屏幕（留出边距）
        max_width = available.width() - 96
        max_height = available.height() - 96

        width = min(max(rect.width(), self.minimumWidth()), max_width)
        height = min(max(rect.height(), self.minimumHeight()), max_height)

        rect.setWidth(width)
        rect.setHeight(height)
        rect.moveTopLeft(self._clamp_top_left(rect.x(), rect.y(), rect))
        return rect

    def _clamp_top_left(self, x: int, y: int, rect: QRect):
        screen = self._current_screen()
        available = screen.availableGeometry() if screen else self.geometry()
        if available.isEmpty():
            return rect.topLeft()
        max_x = available.right() - rect.width() + 1
        max_y = available.bottom() - rect.height() + 1
        x = min(max(available.x(), x), max(available.x(), max_x))
        y = min(max(available.y(), y), max(available.y(), max_y))
        return QPoint(x, y)

    @staticmethod
    def _is_near_available_geometry(target: QRect, available: QRect) -> bool:
        if available.isEmpty():
            return False
        width_ratio = target.width() / max(1, available.width())
        height_ratio = target.height() / max(1, available.height())
        return width_ratio >= 0.90 and height_ratio >= 0.88

    def _current_screen(self):
        handle = self.windowHandle()
        if handle is not None and handle.screen() is not None:
            return handle.screen()
        screen = QGuiApplication.screenAt(self.frameGeometry().center())
        return screen or QGuiApplication.primaryScreen()

    def _sync_shell_margins(self):
        if not hasattr(self, '_root_layout'):
            return
        maximized = self._is_window_maximized()
        if maximized:
            self._root_layout.setContentsMargins(0, 0, 0, 0)
        else:
            self._root_layout.setContentsMargins(8, 8, 8, 4)
        self._refresh_chrome_state_styles(maximized)

    def _refresh_chrome_state_styles(self, maximized: bool):
        if self._last_chrome_maximized_style is maximized:
            return
        self._last_chrome_maximized_style = maximized
        widgets = [
            self,
            getattr(self, '_central_shell', None),
            getattr(self, 'platform_bar', None),
            getattr(self, 'bottom_bar', None),
        ]
        for widget in widgets:
            if widget is None:
                continue
            widget.setProperty('pseudoMaximized', maximized)
            widget.style().unpolish(widget)
            widget.style().polish(widget)
            widget.update()

    def _update_window_shape(self):
        if self._is_window_maximized() or self.isFullScreen():
            self.clearMask()
            return
        rect = self.rect()
        if rect.isEmpty():
            return
        path = QPainterPath()
        path.addRoundedRect(QRectF(rect), self._corner_radius, self._corner_radius)
        self.setMask(QRegion(path.toFillPolygon().toPolygon()))

    def _capture_normal_geometry(self):
        if (
            self.isMaximized()
            or self.isMinimized()
            or self._window_transition is not None
            or self._applying_restore_geometry
        ):
            return
        current = QRect(self.geometry())
        if not current.isValid():
            return

        # 只保存非全屏的几何信息
        screen = self._current_screen()
        available = screen.availableGeometry() if screen else current
        if not self._is_near_available_geometry(current, available):
            self._normal_geometry_before_maximize = current

    def _normalize_initial_geometry(self):
        screen = self._current_screen()
        available = screen.availableGeometry() if screen else self.geometry()
        target = self._normal_geometry_before_maximize
        if not isinstance(target, QRect) or not self._is_usable_restore_geometry(
            target, available
        ):
            target = self._fixed_restore_geometry(available)
        else:
            target = self._clamp_rect_to_available(QRect(target), available)

        self._normal_geometry_before_maximize = QRect(target)
        if self.isMaximized():
            self._begin_restore(target)
            return
        if target != self.geometry():
            self.setGeometry(target)
        self._window_transition = None

        if hasattr(self, 'btn_window_max'):
            self.btn_window_max.setText('▢')

        # 强制更新chrome几何
        QTimer.singleShot(0, self._sync_chrome_geometry)

    def _select_task(self, task_type: str, emit: bool = True):
        task_type = str(task_type or 'pose')
        if task_type not in getattr(self, '_task_labels', {}):
            return
        unchanged = task_type == getattr(self, '_current_task_type', 'pose')
        self._current_task_type = task_type
        label = self._task_labels[task_type]
        self.btn_task_picker.setText(f'{label}  ▾')
        self.btn_task_picker.setToolTip(self._task_display_name(task_type))
        for key, action in self.task_actions.items():
            action.setChecked(key == task_type)
        if emit and not unchanged:
            self.task_selected.emit(task_type)

    def _select_module(self, module_id: str, emit: bool = True):
        module_id = str(module_id or 'data')
        page = getattr(self, '_module_pages', {}).get(module_id)
        button = getattr(self, 'module_buttons', {}).get(module_id)
        if page is None or button is None or not button.isEnabled():
            return

        unchanged = module_id == self._current_module
        self._current_module = module_id
        for key, module_button in self.module_buttons.items():
            module_button.setChecked(key == module_id)
        self.workspace_stack.setCurrentWidget(page)
        self.btn_task_picker.setVisible(module_id == 'data')
        data_active = module_id == 'data'
        for shortcut in getattr(self, '_data_shortcuts', ()):
            shortcut.setEnabled(data_active)
        self._apply_data_action_state()
        self.btn_prev.setEnabled(data_active and self._data_navigation_available)
        self.btn_next.setEnabled(data_active and self._data_navigation_available)
        if not unchanged:
            module_names = {
                'data': '数据管理',
                'model': '模型管理',
                'train': '训练中心',
                'eval': '评估中心',
                'infer': '推理中心',
            }
            self.status_bar.showMessage(
                f'已切换到{module_names.get(module_id, module_id)}', 1800
            )
        if emit and not unchanged:
            self.module_selected.emit(module_id)

    def current_module(self) -> str:
        return self._current_module

    # ---- Public helpers ----

    def select_module(self, module_id: str):
        """Switch platform modules from a controller workflow."""
        self._select_module(module_id)

    def select_task(self, task_type: str, emit: bool = True):
        """Switch the active data task from a controller workflow."""
        self._select_task(task_type, emit=emit)

    def show_data_source_context(self, model, source):
        """Show the model provenance for a model-to-data navigation."""
        self._data_context_kind = 'model'
        self.btn_return_to_model.setText('‹  返回模型详情')
        role_label = {
            'train': '训练集',
            'val': '验证 / 测试集',
            'test': '测试集',
        }.get(getattr(source, 'role', ''), getattr(source, 'role', '-'))
        model_name = getattr(model, 'name', '未知模型')
        dataset_name = getattr(source, 'dataset_name', '-')
        batch_name = getattr(source, 'batch_name', '-')
        self.lbl_data_source_context.setText(
            f'{model_name}  /  {role_label}  /  {dataset_name} · {batch_name}'
        )
        self.lbl_data_source_context.setToolTip(
            str(getattr(source, 'image_path', ''))
        )
        self.data_source_context_bar.show()

    def show_training_data_context(self, batch_path: str | Path,
                                   task_type: str):
        """Show a reversible training-task context in data review."""
        batch = Path(batch_path)
        self._data_context_kind = 'training'
        self.btn_return_to_model.setText('‹  返回训练任务')
        self.lbl_data_source_context.setText(
            f'训练数据审查  /  {self._task_display_name(task_type)}  /  {batch.name}'
        )
        self.lbl_data_source_context.setToolTip(str(batch))
        self.data_source_context_bar.show()

    def _emit_data_context_return(self):
        if self._data_context_kind == 'training':
            self.training_return_requested.emit()
        else:
            self.model_return_requested.emit()

    def hide_data_source_context(self):
        self.data_source_context_bar.hide()
        self._data_context_kind = None

    def set_counter_text(self, current: int, total: int):
        self.lbl_counter.setText(
            f'图片 {current} / {total}' if total > 0 else '没有图片'
        )

    def set_annotation_btn_text(self, text: str):
        self.btn_annotation.setText(f'🏷 {text}')

    def set_skeleton_btn_text(self, visible: bool):
        self.btn_skeleton.setText('骨架线: 开' if visible else '骨架线: 关')

    def set_task_context(self, task_type: str, annotation_dir: str):
        self._select_task(task_type, emit=False)

    @staticmethod
    def _task_display_name(task_type: str) -> str:
        task_names = {
            'pose': 'Pose 姿态',
            'detection': '目标检测',
            'segmentation': '语义分割',
            'obb': '旋转框 OBB',
        }
        return task_names.get(task_type, task_type or '-')

    def set_nav_enabled(self, enabled: bool):
        self._data_navigation_available = bool(enabled)
        active = enabled and self._current_module == 'data'
        self.btn_prev.setEnabled(active)
        self.btn_next.setEnabled(active)

    def set_action_enabled(self, enabled: bool):
        self._data_actions_available = bool(enabled)
        self._apply_data_action_state()

    def _apply_data_action_state(self):
        enabled = (
            self._data_actions_available
            and self._current_module == 'data'
        )
        self._set_actions_enabled(enabled)
        self.action_copy.setEnabled(enabled)
        self.action_move.setEnabled(enabled)
        self.action_delete.setEnabled(enabled)
        self.action_rename.setEnabled(enabled)
        self.action_open_label_tool.setEnabled(enabled)

    def _set_actions_enabled(self, enabled: bool):
        for btn in [
            self.btn_open_label_tool,
            self.btn_copy,
            self.btn_move,
            self.btn_delete,
            self.btn_rename,
        ]:
            btn.setEnabled(enabled)

    def _setup_visual_effects(self):
        self._hover_glow = HoverGlow(self)
        self._hover_glow.watch_buttons(self.bottom_bar)
        self._hover_glow.watch_buttons(self.platform_bar)
        self._attach_soft_shadow(self.lbl_window_title, '#36B7FF', 22, 90)

    def _attach_soft_shadow(self, widget: QWidget, color: str,
                            blur: int, alpha: int):
        effect = QGraphicsDropShadowEffect(widget)
        effect.setOffset(0, 0)
        effect.setBlurRadius(blur)
        glow = QColor(color)
        glow.setAlpha(alpha)
        effect.setColor(glow)
        widget.setGraphicsEffect(effect)
        self._panel_glows.append(effect)

    # ---- Persistence ----

    def _restore_state(self):
        settings = QSettings('FilesProcessQT', 'ImageManager')
        screen = self._current_screen()
        available = screen.availableGeometry() if screen else self.geometry()
        geo = settings.value('normalGeometry')
        if (
            isinstance(geo, QRect)
            and self._is_usable_restore_geometry(QRect(geo), available)
        ):
            target = self._clamp_rect_to_available(QRect(geo), available)
        else:
            target = self._fixed_restore_geometry(available)
        self.setGeometry(target)
        self._normal_geometry_before_maximize = QRect(target)
        # restore splitter sizes
        sizes = settings.value('splitterSizes')
        if sizes:
            try:
                self.top_splitter.setSizes([int(s) for s in sizes])
            except (TypeError, ValueError):
                pass

    def closeEvent(self, event):
        # Let the controller persist the currently opened data root before the
        # window and its child views are torn down.
        self.about_to_close.emit()
        for glow in self.findChildren(HoverGlow):
            glow.stop()
        if hasattr(self, '_ambient_backdrop'):
            self._ambient_backdrop._timer.stop()
        settings = QSettings('FilesProcessQT', 'ImageManager')
        screen = self._current_screen()
        available = screen.availableGeometry() if screen else self.geometry()
        if self._window_transition == 'restoring':
            candidate = self._pending_restore_geometry
        elif self.isMaximized() or self._window_transition == 'maximizing':
            candidate = self._normal_geometry_before_maximize
        else:
            candidate = QRect(self.geometry())
        if not isinstance(candidate, QRect) or not self._is_usable_restore_geometry(
            candidate, available
        ):
            candidate = self._normal_geometry_before_maximize
        if not isinstance(candidate, QRect) or not self._is_usable_restore_geometry(
            candidate, available
        ):
            candidate = self._fixed_restore_geometry(available)
        normal_geometry = self._clamp_rect_to_available(
            QRect(candidate), available
        )
        settings.setValue('normalGeometry', normal_geometry)
        settings.setValue('splitterSizes', self.top_splitter.sizes())
        settings.sync()
        super().closeEvent(event)


def QS(parent, keys, callback):
    """Create a QShortcut and connect its activated signal."""
    from PyQt5.QtWidgets import QShortcut as _QS
    seq = keys if isinstance(keys, QKeySequence) else QKeySequence(keys)
    sc = _QS(seq, parent)
    sc.activated.connect(callback)
    return sc
