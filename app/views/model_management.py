"""Model registry workspace with task filters, cards, and inline details."""

from __future__ import annotations

import re
from pathlib import Path

from PyQt5.QtCore import (
    QEasingCurve,
    QEvent,
    QPropertyAnimation,
    QSettings,
    Qt,
    QVariantAnimation,
    pyqtSignal,
)
from PyQt5.QtGui import (
    QColor,
    QKeyEvent,
    QLinearGradient,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QStyle,
    QStyleOption,
    QToolButton,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.views.ui_effects import HoverGlow
from app.views.model_comparison import ModelComparisonPage
from app.models.model_registry import (
    DatasetSource,
    ModelRecord,
    scan_model_repository,
)


TASK_META = {
    'all': ('全部模型', '#36B7FF'),
    'detection': ('目标检测', '#36B7FF'),
    'segmentation': ('语义分割', '#45D483'),
    'pose': ('姿态估计', '#B88CFF'),
    'obb': ('旋转框 OBB', '#F5A524'),
    'other': ('未分类', '#8EA0B6'),
}

def _task_label(task_type: str) -> str:
    return TASK_META.get(task_type, TASK_META['other'])[0]


def _format_bytes(size: int) -> str:
    value = max(0, int(size))
    units = ('B', 'KB', 'MB', 'GB', 'TB')
    number = float(value)
    for unit in units:
        if number < 1024 or unit == units[-1]:
            return f'{number:.1f} {unit}' if unit != 'B' else f'{int(number)} B'
        number /= 1024
    return f'{value} B'


def _dataset_role_label(role: str) -> str:
    return {
        'train': '训练集',
        'val': '验证 / 测试集',
        'test': '测试集',
    }.get(role, role or '-')


def _format_metric(value: float | None) -> str:
    return f'{value:.4f}' if value is not None else '-'


class _ModelCard(QFrame):
    """Clickable model card with a restrained animated focus glow."""

    activated = pyqtSignal(object)
    data_source_requested = pyqtSignal(object)
    comparison_toggled = pyqtSignal(object, bool)

    def __init__(self, record: ModelRecord, parent=None):
        super().__init__(parent)
        self.record = record
        self._glow = 0.0
        self._pressed = False
        self._comparison_mode = False
        self.setObjectName('modelCard')
        self.setProperty('taskType', record.task_type)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName(f'{record.name}, {_task_label(record.task_type)}')
        self.setMinimumWidth(270)
        self.setMinimumHeight(210)
        self.setMaximumHeight(224)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self._hover_animation = QVariantAnimation(self)
        self._hover_animation.setDuration(180)
        self._hover_animation.setEasingCurve(QEasingCurve.OutCubic)
        self._hover_animation.valueChanged.connect(self._set_glow)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(17, 15, 17, 14)
        layout.setSpacing(10)

        header = QHBoxLayout()
        header.setSpacing(10)
        monogram = QLabel(self._monogram(record.name))
        monogram.setObjectName('modelMonogram')
        monogram.setProperty('taskType', record.task_type)
        monogram.setAlignment(Qt.AlignCenter)
        monogram.setFixedSize(38, 38)
        header.addWidget(monogram)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel(record.name)
        title.setObjectName('modelCardTitle')
        title.setToolTip(record.name)
        subtitle = QLabel(f'{record.framework}  /  {record.file_format}')
        subtitle.setObjectName('modelCardSubtitle')
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box, 1)

        self.compare_button = QToolButton()
        self.compare_button.setObjectName('modelCardCompareBtn')
        self.compare_button.setText('+')
        self.compare_button.setToolTip('加入模型对比')
        self.compare_button.setCheckable(True)
        self.compare_button.setFixedSize(27, 27)
        self.compare_button.toggled.connect(self._on_compare_toggled)
        self.compare_button.hide()
        header.addWidget(self.compare_button, 0, Qt.AlignTop)

        status = QLabel('DEMO' if record.is_demo else record.status)
        status.setObjectName('modelStatusBadge')
        status.setProperty('demo', record.is_demo)
        status.setAlignment(Qt.AlignCenter)
        header.addWidget(status, 0, Qt.AlignTop)
        layout.addLayout(header)

        badges = QHBoxLayout()
        badges.setSpacing(6)
        task_badge = QLabel(_task_label(record.task_type))
        task_badge.setObjectName('modelTaskBadge')
        task_badge.setProperty('taskType', record.task_type)
        format_badge = QLabel(record.precision)
        format_badge.setObjectName('modelMetaBadge')
        badges.addWidget(task_badge)
        badges.addWidget(format_badge)
        badges.addStretch()
        layout.addLayout(badges)

        self.source_button = QToolButton()
        self.source_button.setObjectName('modelDataSourceBtn')
        self.source_button.setPopupMode(QToolButton.InstantPopup)
        self.source_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.source_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.source_button.setMinimumHeight(28)
        self._setup_source_menu(record.data_sources)
        layout.addWidget(self.source_button)

        divider = QFrame()
        divider.setObjectName('modelCardDivider')
        divider.setFixedHeight(1)
        layout.addWidget(divider)

        footer = QHBoxLayout()
        footer.setSpacing(14)
        epoch_text = (
            f'{record.actual_epochs}/{record.planned_epochs}'
            if record.planned_epochs else str(record.actual_epochs or '-')
        )
        footer.addWidget(self._metric('轮次', epoch_text))
        footer.addWidget(self._metric(
            record.primary_metric_name,
            _format_metric(record.primary_metric_value),
        ))
        footer.addWidget(self._metric('更新', record.modified_at))
        footer.addStretch()
        arrow = QLabel('›')
        arrow.setObjectName('modelCardArrow')
        footer.addWidget(arrow, 0, Qt.AlignBottom)
        layout.addLayout(footer)

    def set_comparison_mode(self, enabled: bool, selected: bool = False):
        self._comparison_mode = bool(enabled)
        self.compare_button.blockSignals(True)
        self.compare_button.setChecked(bool(selected))
        self.compare_button.setText('✓' if selected else '+')
        self.compare_button.blockSignals(False)
        self.compare_button.setVisible(self._comparison_mode)
        self.setProperty('compareSelected', bool(selected))
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def set_compare_selected(self, selected: bool):
        self.set_comparison_mode(self._comparison_mode, selected)

    def _on_compare_toggled(self, checked: bool):
        self.compare_button.setText('✓' if checked else '+')
        self.compare_button.setToolTip(
            '移出模型对比' if checked else '加入模型对比'
        )
        self.comparison_toggled.emit(self.record, checked)

    def _setup_source_menu(self, sources: tuple[DatasetSource, ...]):
        if not sources:
            self.source_button.setText('训练数据  未解析')
            self.source_button.setEnabled(False)
            return

        train_sources = [source for source in sources if source.role == 'train']
        other_sources = [source for source in sources if source.role != 'train']
        if len(train_sources) == 1:
            source = train_sources[0]
            summary = f'{source.dataset_name} / {source.batch_name}'
        elif train_sources:
            summary = f'{train_sources[0].dataset_name} / {len(train_sources)} 个训练批次'
        else:
            summary = f'{sources[0].dataset_name} / {len(sources)} 个数据源'
        self.source_button.setText(f'训练数据  {summary}  ›')

        menu = QMenu(self.source_button)
        for source in [*train_sources, *other_sources]:
            action = menu.addAction(
                f'{_dataset_role_label(source.role)}  /  {source.batch_name}'
            )
            action.setEnabled(source.available)
            action.triggered.connect(
                lambda _checked=False, selected=source:
                self.data_source_requested.emit(selected)
            )
        self.source_button.setMenu(menu)

    @staticmethod
    def _monogram(name: str) -> str:
        pieces = re.findall(r'[A-Za-z0-9]+', name)
        if not pieces:
            return 'M'
        if len(pieces) == 1:
            return pieces[0][:2].upper()
        return ''.join(piece[0] for piece in pieces[:2]).upper()

    @staticmethod
    def _metric(label: str, value: str) -> QWidget:
        widget = QWidget()
        widget.setObjectName('modelCardMetric')
        box = QVBoxLayout(widget)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(1)
        caption = QLabel(label)
        caption.setObjectName('modelMetricLabel')
        content = QLabel(value or '-')
        content.setObjectName('modelMetricValue')
        box.addWidget(caption)
        box.addWidget(content)
        return widget

    def _animate_glow(self, target: float):
        self._hover_animation.stop()
        self._hover_animation.setStartValue(self._glow)
        self._hover_animation.setEndValue(target)
        self._hover_animation.start()

    def _set_glow(self, value):
        self._glow = float(value)
        self.update()

    def enterEvent(self, event):
        self._animate_glow(1.0)
        super().enterEvent(event)

    def leaveEvent(self, event):
        if not self.hasFocus():
            self._animate_glow(0.0)
        super().leaveEvent(event)

    def focusInEvent(self, event):
        self._animate_glow(1.0)
        super().focusInEvent(event)

    def focusOutEvent(self, event):
        if not self.underMouse():
            self._animate_glow(0.0)
        super().focusOutEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._pressed = True
            self.update()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        was_pressed = self._pressed
        self._pressed = False
        self.update()
        if was_pressed and event.button() == Qt.LeftButton and self.rect().contains(event.pos()):
            if self._comparison_mode:
                self.compare_button.click()
            else:
                self.activated.emit(self.record)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in {Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space}:
            if self._comparison_mode:
                self.compare_button.click()
            else:
                self.activated.emit(self.record)
            event.accept()
            return
        super().keyPressEvent(event)

    def paintEvent(self, _event):
        option = QStyleOption()
        option.initFrom(self)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        self.style().drawPrimitive(QStyle.PE_Widget, option, painter, self)

        accent = QColor(TASK_META.get(self.record.task_type, TASK_META['other'])[1])
        accent.setAlpha(90 + int(115 * self._glow))
        painter.setPen(QPen(accent, 1.0 + self._glow))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -2, -2), 8, 8)

        if self._glow > 0:
            sheen = QLinearGradient(0, 0, self.width(), 0)
            glow = QColor(accent)
            glow.setAlpha(int(34 * self._glow))
            sheen.setColorAt(0.0, glow)
            sheen.setColorAt(0.46, QColor(accent.red(), accent.green(), accent.blue(), 0))
            sheen.setColorAt(1.0, QColor(accent.red(), accent.green(), accent.blue(), 0))
            painter.fillRect(self.rect().adjusted(2, 2, -2, -2), sheen)

        if self._pressed:
            painter.fillRect(self.rect().adjusted(2, 2, -2, -2), QColor(0, 0, 0, 32))


class ModelManagementView(QWidget):
    """Two-column model registry scaffold used by the model platform module."""

    model_selected = pyqtSignal(object)
    dataset_source_requested = pyqtSignal(object, object)
    directory_changed = pyqtSignal(str)
    evaluate_requested = pyqtSignal(str, str)
    inference_requested = pyqtSignal(str, str)

    def __init__(self, parent=None, load_saved_directory: bool = True):
        super().__init__(parent)
        self.setObjectName('modelManagementView')
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._source_directory: Path | None = None
        self._all_models: list[ModelRecord] = []
        self._visible_models: list[ModelRecord] = []
        self._cards: list[_ModelCard] = []
        self._current_record: ModelRecord | None = None
        self._selected_task = 'all'
        self._comparison_mode = False
        self._comparison_models: list[ModelRecord] = []
        self._comparison_message = ''
        self._detail_origin = 'library'
        self._grid_columns = 0
        self._page_animation = None
        self._build_ui()
        self._hover_glow = HoverGlow(self)
        self._hover_glow.watch_buttons(self)

        if load_saved_directory:
            saved = QSettings('FilesProcessQT', 'ImageManager').value(
                'lastModelDirectory'
            )
            if saved and Path(str(saved)).expanduser().is_dir():
                self.set_model_directory(str(saved), persist=False)
            else:
                self._show_design_preview()
        else:
            self._show_design_preview()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(8)

        header = QWidget()
        header.setObjectName('modelLibraryHeader')
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 13, 16, 13)
        header_layout.setSpacing(12)

        heading = QVBoxLayout()
        heading.setSpacing(2)
        eyebrow = QLabel('MODEL REGISTRY')
        eyebrow.setObjectName('modelEyebrow')
        title = QLabel('模型资产库')
        title.setObjectName('modelLibraryTitle')
        self.lbl_source = QLabel('尚未导入模型目录')
        self.lbl_source.setObjectName('modelLibrarySource')
        self.lbl_source.setTextInteractionFlags(Qt.TextSelectableByMouse)
        heading.addWidget(eyebrow)
        heading.addWidget(title)
        heading.addWidget(self.lbl_source)
        header_layout.addLayout(heading, 1)

        self.lbl_model_total = QLabel('0 MODELS')
        self.lbl_model_total.setObjectName('modelTotalPill')
        self.lbl_model_total.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(self.lbl_model_total)

        self.btn_refresh = QToolButton()
        self.btn_refresh.setObjectName('modelIconBtn')
        self.btn_refresh.setText('↻')
        self.btn_refresh.setToolTip('刷新当前模型目录')
        self.btn_refresh.setFixedSize(34, 34)
        self.btn_refresh.clicked.connect(self.refresh_models)
        header_layout.addWidget(self.btn_refresh)

        self.btn_import = QPushButton('＋  导入模型目录')
        self.btn_import.setObjectName('primaryBtn')
        self.btn_import.setMinimumHeight(34)
        self.btn_import.clicked.connect(self.choose_model_directory)
        header_layout.addWidget(self.btn_import)
        root.addWidget(header)

        self.workspace_splitter = QSplitter(Qt.Horizontal)
        self.workspace_splitter.setObjectName('modelWorkspaceSplitter')
        self.workspace_splitter.setChildrenCollapsible(False)
        self.workspace_splitter.setHandleWidth(7)

        self.type_rail = QWidget()
        self.type_rail.setObjectName('modelTypeRail')
        self.type_rail.setMinimumWidth(190)
        self.type_rail.setMaximumWidth(245)
        rail_layout = QVBoxLayout(self.type_rail)
        rail_layout.setContentsMargins(13, 17, 13, 15)
        rail_layout.setSpacing(7)
        rail_title = QLabel('模型类型')
        rail_title.setObjectName('modelRailTitle')
        rail_subtitle = QLabel('按视觉任务组织')
        rail_subtitle.setObjectName('modelRailSubtitle')
        rail_layout.addWidget(rail_title)
        rail_layout.addWidget(rail_subtitle)
        rail_layout.addSpacing(8)

        self.type_button_group = QButtonGroup(self)
        self.type_button_group.setExclusive(True)
        self.type_buttons = {}
        for idx, task_type in enumerate(TASK_META):
            button = QPushButton()
            button.setObjectName('modelTypeBtn')
            button.setProperty('taskType', task_type)
            button.setCheckable(True)
            button.setMinimumHeight(42)
            button.clicked.connect(
                lambda _checked=False, task=task_type: self._select_task(task)
            )
            self.type_button_group.addButton(button, idx)
            self.type_buttons[task_type] = button
            rail_layout.addWidget(button)
        self.type_buttons['all'].setChecked(True)
        rail_layout.addStretch()

        self.lbl_registry_state = QLabel('设计预览')
        self.lbl_registry_state.setObjectName('modelRegistryState')
        self.lbl_registry_state.setWordWrap(True)
        rail_layout.addWidget(self.lbl_registry_state)
        self.workspace_splitter.addWidget(self.type_rail)

        self.content_stack = QStackedWidget()
        self.content_stack.setObjectName('modelContentStack')
        self.library_page = self._build_library_page()
        self.details_page = self._build_details_page()
        self.comparison_page = ModelComparisonPage()
        self.comparison_page.back_requested.connect(self.show_library)
        self.comparison_page.model_requested.connect(
            self._open_comparison_model_details
        )
        self.content_stack.addWidget(self.library_page)
        self.content_stack.addWidget(self.details_page)
        self.content_stack.addWidget(self.comparison_page)
        self.workspace_splitter.addWidget(self.content_stack)
        self.workspace_splitter.setStretchFactor(0, 0)
        self.workspace_splitter.setStretchFactor(1, 1)
        self.workspace_splitter.setSizes([212, 1000])
        root.addWidget(self.workspace_splitter, 1)

    def _build_library_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName('modelLibraryPage')
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 14, 14, 14)
        layout.setSpacing(12)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        self.lbl_results = QLabel('全部模型')
        self.lbl_results.setObjectName('modelResultsTitle')
        toolbar.addWidget(self.lbl_results)
        toolbar.addStretch()

        self.btn_compare_mode = QPushButton('模型对比')
        self.btn_compare_mode.setObjectName('modelCompareModeBtn')
        self.btn_compare_mode.setCheckable(True)
        self.btn_compare_mode.setToolTip('选择两个或多个同任务模型进行训练指标对比')
        self.btn_compare_mode.toggled.connect(self._set_comparison_mode)
        toolbar.addWidget(self.btn_compare_mode)

        self.search_edit = QLineEdit()
        self.search_edit.setObjectName('modelSearchEdit')
        self.search_edit.setPlaceholderText('搜索模型、格式或路径')
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setMinimumWidth(220)
        self.search_edit.setMaximumWidth(360)
        self.search_edit.textChanged.connect(self._apply_filters)
        toolbar.addWidget(self.search_edit)

        self.sort_combo = QComboBox()
        self.sort_combo.setObjectName('modelSortCombo')
        self.sort_combo.addItem('最近更新', 'updated')
        self.sort_combo.addItem('名称', 'name')
        self.sort_combo.addItem('文件大小', 'size')
        self.sort_combo.currentIndexChanged.connect(self._apply_filters)
        toolbar.addWidget(self.sort_combo)
        layout.addLayout(toolbar)

        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName('modelCardsScroll')
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.cards_container = QWidget()
        self.cards_container.setObjectName('modelCardsContainer')
        self.cards_grid = QGridLayout(self.cards_container)
        self.cards_grid.setContentsMargins(2, 2, 8, 8)
        self.cards_grid.setHorizontalSpacing(12)
        self.cards_grid.setVerticalSpacing(12)
        self.cards_grid.setAlignment(Qt.AlignTop)
        self.scroll_area.setWidget(self.cards_container)
        self.scroll_area.viewport().installEventFilter(self)
        layout.addWidget(self.scroll_area, 1)

        self.comparison_tray = QWidget()
        self.comparison_tray.setObjectName('modelComparisonTray')
        tray_layout = QHBoxLayout(self.comparison_tray)
        tray_layout.setContentsMargins(10, 7, 8, 7)
        tray_layout.setSpacing(8)
        self.lbl_comparison_state = QLabel('请选择 2-6 个同任务模型')
        self.lbl_comparison_state.setObjectName('modelComparisonState')
        tray_layout.addWidget(self.lbl_comparison_state)
        self.comparison_chips = QWidget()
        self.comparison_chips.setObjectName('modelComparisonChips')
        self.comparison_chips_layout = QHBoxLayout(self.comparison_chips)
        self.comparison_chips_layout.setContentsMargins(0, 0, 0, 0)
        self.comparison_chips_layout.setSpacing(5)
        tray_layout.addWidget(self.comparison_chips, 1)
        self.btn_clear_comparison = QToolButton()
        self.btn_clear_comparison.setObjectName('modelComparisonClearBtn')
        self.btn_clear_comparison.setText('清空')
        self.btn_clear_comparison.clicked.connect(self._clear_comparison)
        tray_layout.addWidget(self.btn_clear_comparison)
        self.btn_start_comparison = QPushButton('开始对比')
        self.btn_start_comparison.setObjectName('modelComparisonStartBtn')
        self.btn_start_comparison.setEnabled(False)
        self.btn_start_comparison.clicked.connect(self._start_comparison)
        tray_layout.addWidget(self.btn_start_comparison)
        self.comparison_tray.hide()
        layout.addWidget(self.comparison_tray, 0)

        self.empty_state = QWidget(self.cards_container)
        empty_layout = QVBoxLayout(self.empty_state)
        empty_layout.setAlignment(Qt.AlignCenter)
        empty_code = QLabel('NO MODEL')
        empty_code.setObjectName('modelEmptyCode')
        empty_code.setAlignment(Qt.AlignCenter)
        self.lbl_empty_title = QLabel('未找到模型')
        self.lbl_empty_title.setObjectName('modelEmptyTitle')
        self.lbl_empty_title.setAlignment(Qt.AlignCenter)
        self.lbl_empty_detail = QLabel('请调整筛选条件或导入其他模型目录')
        self.lbl_empty_detail.setObjectName('modelEmptyDetail')
        self.lbl_empty_detail.setAlignment(Qt.AlignCenter)
        empty_layout.addWidget(empty_code)
        empty_layout.addWidget(self.lbl_empty_title)
        empty_layout.addWidget(self.lbl_empty_detail)
        self.empty_state.hide()
        return page

    def _build_details_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName('modelDetailsPage')
        layout = QVBoxLayout(page)
        layout.setContentsMargins(22, 16, 22, 16)
        layout.setSpacing(10)

        heading = QHBoxLayout()
        self.btn_back = QToolButton()
        self.btn_back.setObjectName('modelBackBtn')
        self.btn_back.setText('‹')
        self.btn_back.setToolTip('返回模型库')
        self.btn_back.setFixedSize(36, 36)
        self.btn_back.clicked.connect(self._return_from_details)
        heading.addWidget(self.btn_back)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        detail_eyebrow = QLabel('MODEL PROFILE')
        detail_eyebrow.setObjectName('modelEyebrow')
        self.lbl_detail_title = QLabel('-')
        self.lbl_detail_title.setObjectName('modelDetailTitle')
        title_box.addWidget(detail_eyebrow)
        title_box.addWidget(self.lbl_detail_title)
        heading.addLayout(title_box, 1)
        self.lbl_detail_status = QLabel('就绪')
        self.lbl_detail_status.setObjectName('modelDetailStatus')
        heading.addWidget(self.lbl_detail_status)
        self.btn_go_eval = QPushButton('去评估')
        self.btn_go_eval.setObjectName('primaryBtn')
        self.btn_go_eval.setToolTip('在评估中心为该模型选择测试批次并运行评估')
        self.btn_go_eval.clicked.connect(self._emit_evaluate)
        heading.addWidget(self.btn_go_eval)
        self.btn_go_infer = QPushButton('实时推理')
        self.btn_go_infer.setObjectName('successBtn')
        self.btn_go_infer.setToolTip('在推理中心用该模型连接实时画面预览')
        self.btn_go_infer.clicked.connect(self._emit_inference)
        heading.addWidget(self.btn_go_infer)
        self.btn_convert = QPushButton('模型转换')
        self.btn_convert.setObjectName('fileOpBtn')
        self.btn_convert.setToolTip('将 .pt 导出为 ONNX（输出保存在模型同目录）')
        self.btn_convert.clicked.connect(self._open_model_convert)
        heading.addWidget(self.btn_convert)
        self.lbl_evaluation = QLabel('')
        self.lbl_evaluation.setObjectName('duplicateScope')
        self.lbl_evaluation.setWordWrap(True)
        heading.addWidget(self.lbl_evaluation)
        layout.addLayout(heading)

        line = QFrame()
        line.setObjectName('modelDetailDivider')
        line.setFixedHeight(1)
        layout.addWidget(line)

        self.detail_body_splitter = QSplitter(Qt.Vertical)
        self.detail_body_splitter.setObjectName('modelDetailBodySplitter')
        self.detail_body_splitter.setChildrenCollapsible(False)
        self.detail_body_splitter.setHandleWidth(7)

        overview_panel = QWidget()
        overview_panel.setObjectName('modelOverviewPanel')
        overview_panel.setMinimumHeight(228)
        overview_panel_layout = QVBoxLayout(overview_panel)
        overview_panel_layout.setContentsMargins(0, 4, 0, 4)
        overview_panel_layout.setSpacing(0)
        overview = QHBoxLayout()
        overview.setSpacing(24)
        identity = QWidget()
        identity.setObjectName('modelDetailIdentity')
        identity_layout = QVBoxLayout(identity)
        identity_layout.setContentsMargins(0, 4, 22, 4)
        identity_layout.setSpacing(8)
        self.lbl_detail_monogram = QLabel('M')
        self.lbl_detail_monogram.setObjectName('modelDetailMonogram')
        self.lbl_detail_monogram.setAlignment(Qt.AlignCenter)
        self.lbl_detail_monogram.setFixedSize(82, 82)
        self.lbl_detail_task = QLabel('-')
        self.lbl_detail_task.setObjectName('modelDetailTask')
        self.lbl_detail_task.setAlignment(Qt.AlignCenter)
        self.lbl_detail_summary = QLabel('-')
        self.lbl_detail_summary.setObjectName('modelDetailSummary')
        self.lbl_detail_summary.setWordWrap(True)
        identity_layout.addWidget(self.lbl_detail_monogram, 0, Qt.AlignHCenter)
        identity_layout.addWidget(self.lbl_detail_task)
        identity_layout.addWidget(self.lbl_detail_summary)
        identity_layout.addStretch()
        overview.addWidget(identity, 0)

        metadata = QWidget()
        metadata.setObjectName('modelMetadataGrid')
        metadata_grid = QGridLayout(metadata)
        metadata_grid.setContentsMargins(0, 0, 0, 0)
        metadata_grid.setHorizontalSpacing(24)
        metadata_grid.setVerticalSpacing(10)
        for column in range(3):
            metadata_grid.setColumnStretch(column, 1)
        self._detail_values = {}

        def add_field(key, label, row, column, span=1, path_style=False):
            cell = QWidget()
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(2)
            caption = QLabel(label)
            caption.setObjectName('modelDetailFieldLabel')
            value = QLabel('-')
            value.setObjectName(
                'modelDetailPath' if path_style else 'modelDetailFieldValue'
            )
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            value.setWordWrap(path_style)
            cell_layout.addWidget(caption)
            cell_layout.addWidget(value)
            metadata_grid.addWidget(cell, row, column, 1, span)
            self._detail_values[key] = value

        add_field('project', '所属项目', 0, 0)
        add_field('architecture', '模型架构', 0, 1)
        add_field('modified', '更新时间', 0, 2)
        add_field('path', '本地路径', 1, 0, 3, path_style=True)
        add_field('framework', '训练框架', 2, 0)
        add_field('input', '输入尺寸', 2, 1)
        add_field('precision', '模型精度', 2, 2)
        add_field('epochs', '训练轮次', 3, 0)
        add_field('batch', 'Batch', 3, 1)
        add_field('optimizer', '优化器', 3, 2)
        add_field('metric', '任务主指标', 4, 0, 2)
        add_field('device', '训练设备', 4, 2)
        add_field('duration', '累计训练时间', 5, 0)
        add_field('dataset_config', '数据配置', 5, 1, 2, path_style=True)
        self.lbl_detail_path = self._detail_values['path']
        overview.addWidget(metadata, 1)
        overview_panel_layout.addLayout(overview)
        self.detail_body_splitter.addWidget(overview_panel)

        self.detail_tabs = QTabWidget()
        self.detail_tabs.setObjectName('modelDetailTabs')
        self.detail_tabs.setMinimumHeight(230)

        source_page = QWidget()
        source_page.setObjectName('modelDetailTabPage')
        source_layout = QVBoxLayout(source_page)
        source_layout.setContentsMargins(8, 8, 8, 8)
        source_layout.setSpacing(7)

        source_header = QHBoxLayout()
        source_title = QLabel('训练数据来源')
        source_title.setObjectName('modelSourceTitle')
        self.lbl_detail_source_count = QLabel('0 个数据源')
        self.lbl_detail_source_count.setObjectName('modelSourceCount')
        source_header.addWidget(source_title)
        source_header.addStretch()
        source_header.addWidget(self.lbl_detail_source_count)
        source_layout.addLayout(source_header)

        self.source_tree = QTreeWidget()
        self.source_tree.setObjectName('modelDataSourceTree')
        self.source_tree.setHeaderLabels([
            '用途', '数据集', '批次', '图片', 'JSON', 'YOLO 标签', '状态', '操作'
        ])
        self.source_tree.setRootIsDecorated(False)
        self.source_tree.setAlternatingRowColors(False)
        self._configure_detail_tree(
            self.source_tree,
            [105, 185, 145, 72, 72, 100, 82, 72],
        )
        self.source_tree.header().setStretchLastSection(True)
        self.source_tree.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Maximum
        )
        self.source_tree.setMinimumHeight(74)
        self.source_tree.itemDoubleClicked.connect(
            lambda item, _column: self._request_source_from_item(item)
        )
        source_layout.addWidget(self.source_tree, 0)
        self.lbl_source_summary = QLabel('-')
        self.lbl_source_summary.setObjectName('modelSourceSummary')
        self.lbl_source_summary.setWordWrap(True)
        source_layout.addWidget(self.lbl_source_summary, 0)
        source_layout.addStretch(1)
        self.detail_tabs.addTab(source_page, '数据来源')

        self.metric_tree = QTreeWidget()
        self.metric_tree.setObjectName('modelDetailTree')
        self.metric_tree.setHeaderLabels([
            '指标', '最佳值', '最佳轮次', '最后一轮', 'Ultralytics 字段'
        ])
        self._configure_detail_tree(
            self.metric_tree, [220, 110, 100, 110, 300]
        )
        self.detail_tabs.addTab(
            self._tree_tab_page('训练过程中记录的任务评估指标', self.metric_tree),
            '评估指标',
        )

        self.artifact_tree = QTreeWidget()
        self.artifact_tree.setObjectName('modelDetailTree')
        self.artifact_tree.setHeaderLabels([
            '用途', '文件', '格式', '运行框架', '大小', '更新时间', '完整路径'
        ])
        self._configure_detail_tree(
            self.artifact_tree, [90, 190, 85, 130, 95, 105, 360]
        )
        self.detail_tabs.addTab(
            self._tree_tab_page('本次训练产生的权重、检查点和导出模型', self.artifact_tree),
            '模型产物',
        )

        self.config_tree = QTreeWidget()
        self.config_tree.setObjectName('modelDetailTree')
        self.config_tree.setHeaderLabels(['参数', '值'])
        self._configure_detail_tree(self.config_tree, [220, 720])
        self.detail_tabs.addTab(
            self._tree_tab_page('从 args.yaml 自动读取的完整训练配置', self.config_tree),
            '训练配置',
        )

        result_page = QWidget()
        result_page.setObjectName('modelDetailTabPage')
        result_layout = QVBoxLayout(result_page)
        result_layout.setContentsMargins(8, 8, 8, 8)
        result_layout.setSpacing(7)
        result_picker = QHBoxLayout()
        result_title = QLabel('训练结果图片')
        result_title.setObjectName('modelSourceTitle')
        result_picker.addWidget(result_title)
        self.result_asset_combo = QComboBox()
        self.result_asset_combo.setObjectName('modelResultCombo')
        self.result_asset_combo.setMinimumWidth(260)
        self.result_asset_combo.currentIndexChanged.connect(
            self._show_selected_result_asset
        )
        result_picker.addWidget(self.result_asset_combo, 1)
        self.btn_result_prev = QToolButton()
        self.btn_result_prev.setObjectName('modelResultNavBtn')
        self.btn_result_prev.setText('‹')
        self.btn_result_prev.setToolTip('上一张训练结果图片')
        self.btn_result_prev.clicked.connect(lambda: self._step_result_asset(-1))
        result_picker.addWidget(self.btn_result_prev)
        self.btn_result_next = QToolButton()
        self.btn_result_next.setObjectName('modelResultNavBtn')
        self.btn_result_next.setText('›')
        self.btn_result_next.setToolTip('下一张训练结果图片')
        self.btn_result_next.clicked.connect(lambda: self._step_result_asset(1))
        result_picker.addWidget(self.btn_result_next)
        result_layout.addLayout(result_picker)

        self.result_preview = QScrollArea()
        self.result_preview.setObjectName('modelResultPreview')
        self.result_preview.setWidgetResizable(True)
        self.result_preview.setAlignment(Qt.AlignCenter)
        self.result_preview_label = QLabel('没有可预览的训练结果图片')
        self.result_preview_label.setObjectName('modelResultPreviewLabel')
        self.result_preview_label.setAlignment(Qt.AlignCenter)
        self.result_preview_label.setMinimumSize(240, 150)
        self.result_preview.setWidget(self.result_preview_label)
        self.result_preview.viewport().installEventFilter(self)
        result_layout.addWidget(self.result_preview, 1)
        self.lbl_result_asset_info = QLabel('-')
        self.lbl_result_asset_info.setObjectName('modelResultInfo')
        self.lbl_result_asset_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_result_asset_info.setWordWrap(True)
        result_layout.addWidget(self.lbl_result_asset_info)
        self.detail_tabs.addTab(result_page, '训练结果')

        self.detail_body_splitter.addWidget(self.detail_tabs)
        self.detail_body_splitter.setStretchFactor(0, 0)
        self.detail_body_splitter.setStretchFactor(1, 1)
        self.detail_body_splitter.setSizes([250, 430])
        self.detail_body_splitter.setMinimumHeight(465)
        layout.addWidget(self.detail_body_splitter, 1)
        self._current_result_pixmap = QPixmap()
        return page

    @staticmethod
    def _configure_detail_tree(tree: QTreeWidget, widths: list[int]):
        tree.setRootIsDecorated(False)
        tree.setAlternatingRowColors(False)
        tree.setUniformRowHeights(True)
        tree.setSelectionBehavior(QAbstractItemView.SelectRows)
        tree.setSelectionMode(QAbstractItemView.SingleSelection)
        tree.setTextElideMode(Qt.ElideRight)
        tree.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        header = tree.header()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(48)
        for column, width in enumerate(widths):
            tree.setColumnWidth(column, width)

    @staticmethod
    def _tree_tab_page(description: str, tree: QTreeWidget) -> QWidget:
        page = QWidget()
        page.setObjectName('modelDetailTabPage')
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(7)
        label = QLabel(description)
        label.setObjectName('modelDetailTabHint')
        layout.addWidget(label)
        layout.addWidget(tree, 1)
        return page

    def choose_model_directory(self):
        start = str(self._source_directory or Path.home())
        path = QFileDialog.getExistingDirectory(self, '选择模型目录', start)
        if path:
            self.set_model_directory(path)

    def set_model_directory(self, path: str, persist: bool = True):
        directory = Path(path).expanduser()
        if not directory.is_dir():
            return
        if hasattr(self, 'btn_compare_mode'):
            self.btn_compare_mode.setChecked(False)
        self._source_directory = directory.resolve()
        self._all_models = self.scan_model_directory(self._source_directory)
        self.lbl_source.setText(str(self._source_directory))
        self.lbl_source.setToolTip(str(self._source_directory))
        self.lbl_registry_state.setText(
            f'本地仓库\n{len(self._all_models)} 次训练记录'
        )
        if persist:
            QSettings('FilesProcessQT', 'ImageManager').setValue(
                'lastModelDirectory', str(self._source_directory)
            )
        self._update_category_counts()
        self._apply_filters()
        self.directory_changed.emit(str(self._source_directory))

    def refresh_models(self):
        if self._source_directory and self._source_directory.is_dir():
            self.set_model_directory(str(self._source_directory), persist=False)
        else:
            self._show_design_preview()

    def model_directory(self) -> Path | None:
        return self._source_directory

    def show_training_run(self, repository_root: str | Path,
                          run_directory: str | Path) -> bool:
        """Refresh a repository and open the model produced by one run."""
        repository = Path(repository_root).expanduser()
        target = Path(run_directory).expanduser()
        if not repository.is_dir() or not target.is_dir():
            return False
        self.set_model_directory(str(repository))
        try:
            target = target.resolve()
        except OSError:
            pass
        record = next(
            (
                model for model in self._all_models
                if self._same_model_path(model.path, target)
            ),
            None,
        )
        if record is None:
            return False
        self._select_task(
            record.task_type if record.task_type in TASK_META else 'all'
        )
        self.show_model_details(record)
        return True

    @staticmethod
    def _same_model_path(first: str | Path, second: str | Path) -> bool:
        try:
            return Path(first).expanduser().resolve() == Path(second).expanduser().resolve()
        except OSError:
            return str(first) == str(second)

    @classmethod
    def scan_model_directory(cls, directory: str | Path) -> list[ModelRecord]:
        return scan_model_repository(directory)

    def _show_design_preview(self):
        if hasattr(self, 'btn_compare_mode'):
            self.btn_compare_mode.setChecked(False)
        self._source_directory = None
        self._all_models = self._preview_models()
        self.lbl_source.setText('框架预览  /  导入目录后将替换为本地模型')
        self.lbl_registry_state.setText('设计预览\n非本地模型')
        self._update_category_counts()
        self._apply_filters()

    @staticmethod
    def _preview_models() -> list[ModelRecord]:
        return [
            ModelRecord(
                model_id='demo:det', name='TowerGuard-DET-v3',
                project_name='ShengSong', task_type='detection',
                framework='Ultralytics YOLO', file_format='PT + ONNX',
                path='设计预览，不对应本地文件', size_bytes=38_600_000,
                modified_at='2026-08-02', precision='FP32', input_size='1280 x 1280',
                status='预览', is_demo=True, architecture='yolov8x',
                planned_epochs=100, actual_epochs=100,
                primary_metric_name='Box mAP50-95', primary_metric_value=0.8721,
            ),
            ModelRecord(
                model_id='demo:pose', name='ShengSong-Pose-23',
                project_name='ShengSong', task_type='pose',
                framework='Ultralytics YOLO', file_format='PT',
                path='设计预览，不对应本地文件', size_bytes=52_400_000,
                modified_at='2026-07-29', precision='FP32', input_size='640 x 640',
                status='预览', is_demo=True, architecture='yolov8m-pose',
                planned_epochs=100, actual_epochs=98,
                primary_metric_name='Pose mAP50-95', primary_metric_value=0.8039,
            ),
            ModelRecord(
                model_id='demo:seg', name='Workwear-SEG-v2',
                project_name='ShengSong', task_type='segmentation',
                framework='Ultralytics YOLO', file_format='PT + ENGINE',
                path='设计预览，不对应本地文件', size_bytes=71_800_000,
                modified_at='2026-07-18', precision='FP16', input_size='1024 x 1024',
                status='预览', is_demo=True, architecture='yolov8x-seg',
                planned_epochs=120, actual_epochs=120,
                primary_metric_name='Mask mAP50-95', primary_metric_value=0.7412,
            ),
            ModelRecord(
                model_id='demo:obb', name='Equipment-OBB-v1',
                project_name='YanCheng', task_type='obb',
                framework='Ultralytics YOLO', file_format='PT + ONNX',
                path='设计预览，不对应本地文件', size_bytes=44_900_000,
                modified_at='2026-07-11', precision='FP32', input_size='1024 x 1024',
                status='预览', is_demo=True, architecture='yolov8n-obb',
                planned_epochs=100, actual_epochs=100,
                primary_metric_name='OBB mAP50-95', primary_metric_value=0.8244,
            ),
        ]

    def _select_task(self, task_type: str):
        self._selected_task = task_type if task_type in TASK_META else 'all'
        for key, button in self.type_buttons.items():
            button.setChecked(key == self._selected_task)
        self._apply_filters()

    def _apply_filters(self):
        query = self.search_edit.text().strip().lower()
        models = [
            model for model in self._all_models
            if (self._selected_task == 'all' or model.task_type == self._selected_task)
            and (
                not query
                or query in model.name.lower()
                or query in model.framework.lower()
                or query in model.file_format.lower()
                or query in model.project_name.lower()
                or query in model.architecture.lower()
                or query in model.path.lower()
            )
        ]
        sort_mode = self.sort_combo.currentData()
        if sort_mode == 'name':
            models.sort(key=lambda item: item.name.lower())
        elif sort_mode == 'size':
            models.sort(key=lambda item: (-item.size_bytes, item.name.lower()))
        else:
            models.sort(key=lambda item: (item.modified_at, item.name.lower()), reverse=True)
        self._visible_models = models
        self._render_cards()

    def _update_category_counts(self):
        counts = {task_type: 0 for task_type in TASK_META}
        counts['all'] = len(self._all_models)
        for model in self._all_models:
            counts[model.task_type if model.task_type in counts else 'other'] += 1
        for task_type, button in self.type_buttons.items():
            label = TASK_META[task_type][0]
            button.setText(f'{label}\n{counts.get(task_type, 0)}')
        suffix = ' DEMO' if any(model.is_demo for model in self._all_models) else ' MODELS'
        self.lbl_model_total.setText(f'{len(self._all_models)}{suffix}')

    def _render_cards(self):
        for card in self._cards:
            self.cards_grid.removeWidget(card)
            card.deleteLater()
        self._cards = []

        task_label = TASK_META[self._selected_task][0]
        self.lbl_results.setText(f'{task_label}  /  {len(self._visible_models)}')
        self.empty_state.setVisible(not self._visible_models)
        if not self._visible_models:
            self.empty_state.setGeometry(self.cards_container.rect())
            self.empty_state.raise_()
            return

        for record in self._visible_models:
            card = _ModelCard(record)
            card.activated.connect(self._open_library_model_details)
            card.comparison_toggled.connect(self._toggle_model_comparison)
            card.data_source_requested.connect(
                lambda source, model=record:
                self.dataset_source_requested.emit(model, source)
            )
            card.set_comparison_mode(
                self._comparison_mode,
                self._is_comparison_selected(record),
            )
            self._cards.append(card)
        self._grid_columns = 0
        self._relayout_cards()

    def _relayout_cards(self):
        if not self._cards:
            return
        available = max(300, self.scroll_area.viewport().width() - 12)
        card_min_width = 310
        card_gap = self.cards_grid.horizontalSpacing()
        columns = max(
            1,
            min(4, (available + card_gap) // (card_min_width + card_gap)),
        )
        if columns == self._grid_columns and self.cards_grid.count() == len(self._cards):
            return
        self._grid_columns = columns
        while self.cards_grid.count():
            self.cards_grid.takeAt(0)
        for idx, card in enumerate(self._cards):
            self.cards_grid.addWidget(card, idx // columns, idx % columns)
        for column in range(columns):
            self.cards_grid.setColumnStretch(column, 1)

    def _set_comparison_mode(self, enabled: bool):
        self._comparison_mode = bool(enabled)
        self.btn_compare_mode.setText(
            '退出对比' if self._comparison_mode else '模型对比'
        )
        self.comparison_tray.setVisible(self._comparison_mode)
        if not self._comparison_mode:
            self._comparison_models.clear()
            self._comparison_message = ''
        for card in self._cards:
            card.set_comparison_mode(
                self._comparison_mode,
                self._is_comparison_selected(card.record),
            )
        self._update_comparison_tray()

    def _toggle_model_comparison(self, record: ModelRecord, selected: bool):
        if not self._comparison_mode:
            return
        existing = next(
            (item for item in self._comparison_models
             if item.model_id == record.model_id),
            None,
        )
        if not selected:
            if existing is not None:
                self._comparison_models.remove(existing)
            self._comparison_message = ''
            self._update_comparison_tray()
            return
        if existing is not None:
            return
        if (
            self._comparison_models
            and record.task_type != self._comparison_models[0].task_type
        ):
            self._comparison_message = '只能选择相同任务类型的模型'
            self._set_card_comparison_state(record, False)
            self._update_comparison_tray(error=True)
            return
        if len(self._comparison_models) >= 6:
            self._comparison_message = '最多同时对比 6 个模型'
            self._set_card_comparison_state(record, False)
            self._update_comparison_tray(error=True)
            return
        self._comparison_models.append(record)
        self._comparison_message = ''
        self._update_comparison_tray()

    def _is_comparison_selected(self, record: ModelRecord) -> bool:
        return any(
            item.model_id == record.model_id
            for item in self._comparison_models
        )

    def _set_card_comparison_state(self, record: ModelRecord, selected: bool):
        for card in self._cards:
            if card.record.model_id == record.model_id:
                card.set_compare_selected(selected)
                return

    def _remove_comparison_model(self, model_id: str):
        removed = next(
            (record for record in self._comparison_models
             if record.model_id == model_id),
            None,
        )
        if removed is not None:
            self._comparison_models.remove(removed)
            self._set_card_comparison_state(removed, False)
        self._comparison_message = ''
        self._update_comparison_tray()

    def _clear_comparison(self):
        selected_ids = {record.model_id for record in self._comparison_models}
        self._comparison_models.clear()
        self._comparison_message = ''
        for card in self._cards:
            if card.record.model_id in selected_ids:
                card.set_compare_selected(False)
        self._update_comparison_tray()

    def _update_comparison_tray(self, error: bool = False):
        while self.comparison_chips_layout.count():
            item = self.comparison_chips_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for index, record in enumerate(self._comparison_models):
            chip = QToolButton()
            chip.setObjectName('modelComparisonTrayChip')
            chip.setProperty('colorIndex', index)
            chip.setText(f'{record.name}  ×')
            chip.setToolTip('从对比列表移除')
            chip.clicked.connect(
                lambda _checked=False, model_id=record.model_id:
                self._remove_comparison_model(model_id)
            )
            self.comparison_chips_layout.addWidget(chip)
        self.comparison_chips_layout.addStretch()

        count = len(self._comparison_models)
        if self._comparison_message:
            state = self._comparison_message
        elif count == 0:
            state = '请选择 2-6 个同任务模型'
        elif count == 1:
            state = '已选择 1 个，还需要 1 个模型'
        else:
            state = f'已选择 {count} 个模型'
        self.lbl_comparison_state.setText(state)
        self.lbl_comparison_state.setProperty(
            'error', bool(error or self._comparison_message)
        )
        self.lbl_comparison_state.style().unpolish(self.lbl_comparison_state)
        self.lbl_comparison_state.style().polish(self.lbl_comparison_state)
        self.btn_clear_comparison.setEnabled(count > 0)
        self.btn_start_comparison.setEnabled(count >= 2)

    def _start_comparison(self):
        if len(self._comparison_models) < 2:
            return
        self.comparison_page.set_models(self._comparison_models)
        self._fade_to(self.comparison_page)

    def _open_library_model_details(self, record: ModelRecord):
        self._show_model_details(record, origin='library')

    def _open_comparison_model_details(self, record: ModelRecord):
        self._show_model_details(record, origin='comparison')

    def _return_from_details(self):
        if self._detail_origin == 'comparison' and len(self._comparison_models) >= 2:
            self.comparison_page.set_models(self._comparison_models)
            self._fade_to(self.comparison_page)
        else:
            self.show_library()

    def _emit_evaluate(self):
        weight_path = self._resolve_weight_path()
        if weight_path:
            self.evaluate_requested.emit(weight_path, self._current_record.name)

    def _emit_inference(self):
        weight_path = self._resolve_weight_path()
        if weight_path:
            self.inference_requested.emit(weight_path, self._current_record.name)

    def _open_model_convert(self):
        weight_path = self._resolve_weight_path()
        if not weight_path:
            return
        try:
            from app.tools.model_convert import create_dialog as create_convert
        except ImportError as exc:
            QMessageBox.warning(self, '模型转换', f'工具加载失败: {exc}')
            return
        dialog = create_convert(self, default_model=weight_path)
        dialog.exec_()

    def _resolve_weight_path(self) -> str:
        record = getattr(self, '_current_record', None)
        if record is None:
            return ''
        for artifact in record.artifacts:
            if str(artifact.path).endswith('.pt'):
                return str(artifact.path)
        candidates = [
            str(path) for path in Path(record.path).rglob('*.pt')
            if path.is_file()
        ]
        return candidates[-1] if candidates else record.path

    def _show_model_details(self, record: ModelRecord, origin: str | None = None):
        if origin in {'library', 'comparison'}:
            self._detail_origin = origin
        self._current_record = record
        self.btn_back.setToolTip(
            '返回模型对比' if self._detail_origin == 'comparison' else '返回模型库'
        )
        self.lbl_detail_title.setText(record.name)
        self.lbl_detail_monogram.setText(_ModelCard._monogram(record.name))
        self.lbl_detail_task.setText(_task_label(record.task_type))
        self.lbl_detail_summary.setText(
            f'{record.file_format} / {record.precision}\n{record.input_size}'
        )
        self.lbl_detail_status.setText('DEMO' if record.is_demo else record.status)
        evaluation = record.evaluation or {}
        eval_metrics = evaluation.get('metrics') or {}
        if 'test_batch' in evaluation:
            self.lbl_evaluation.setText(
                f'测试评估 [{evaluation.get("test_batch", "")}] | '
                f"mAP50-95 {_format_metric(eval_metrics.get('mAP50-95'))} | "
                f"泛化差距 {_format_metric(evaluation.get('generalization_gap'))}"
            )
        else:
            self.lbl_evaluation.setText('尚未评估 — 点击“去评估”选择测试批次')
        self._detail_values['project'].setText(record.project_name)
        self._detail_values['architecture'].setText(record.architecture)
        self._detail_values['modified'].setText(record.modified_at)
        self._detail_values['path'].setText(record.path)
        self._detail_values['path'].setToolTip(record.path)
        self._detail_values['framework'].setText(record.framework)
        self._detail_values['input'].setText(record.input_size)
        self._detail_values['precision'].setText(record.precision)
        self._detail_values['epochs'].setText(
            f'{record.actual_epochs} / {record.planned_epochs}'
            if record.planned_epochs else str(record.actual_epochs or '-')
        )
        self._detail_values['batch'].setText(record.batch_size)
        self._detail_values['optimizer'].setText(record.optimizer)
        metric_text = _format_metric(record.primary_metric_value)
        if record.primary_metric_epoch:
            metric_text += f'  @ epoch {record.primary_metric_epoch}'
        self._detail_values['metric'].setText(
            f'{record.primary_metric_name}: {metric_text}'
        )
        self._detail_values['device'].setText(record.device)
        self._detail_values['duration'].setText(
            self._format_duration(record.training_seconds)
        )
        dataset_config = record.dataset_config or '-'
        self._detail_values['dataset_config'].setText(dataset_config)
        self._detail_values['dataset_config'].setToolTip(dataset_config)
        self._populate_data_sources(record)
        self._populate_metrics(record)
        self._populate_artifacts(record)
        self._populate_training_args(record)
        self._populate_result_assets(record)
        self.detail_tabs.setTabText(
            0, f'数据来源  ·  {len(record.data_sources)}'
        )
        self.detail_tabs.setTabText(1, f'评估指标  ·  {len(record.metrics)}')
        self.detail_tabs.setTabText(2, f'模型产物  ·  {len(record.artifacts)}')
        self.detail_tabs.setTabText(3, f'训练配置  ·  {len(record.training_args)}')
        self.detail_tabs.setTabText(4, f'训练结果  ·  {len(record.result_assets)}')
        self._set_task_property(self.lbl_detail_monogram, record.task_type)
        self._set_task_property(self.lbl_detail_task, record.task_type)
        self._fade_to(self.details_page)
        self.model_selected.emit(record)

    def _populate_data_sources(self, record: ModelRecord):
        self.source_tree.clear()
        self.lbl_detail_source_count.setText(
            f'{len(record.data_sources)} 个数据源'
        )
        total_images = 0
        total_annotations = 0
        total_labels = 0
        available_count = 0
        for source in record.data_sources:
            total_images += source.image_count
            total_annotations += source.annotation_count
            total_labels += source.label_count
            available_count += int(source.available)
            if not source.available:
                status = '路径失效'
            elif source.annotation_count == 0:
                status = '无任务 JSON'
            else:
                status = '可用'
            item = QTreeWidgetItem([
                _dataset_role_label(source.role),
                source.dataset_name,
                source.batch_name,
                str(source.image_count),
                str(source.annotation_count),
                str(source.label_count),
                status,
                '',
            ])
            item.setData(0, Qt.UserRole, source)
            item.setToolTip(2, source.image_path)
            item.setToolTip(4, f'{source.annotation_dir}: {source.annotation_path}')
            self.source_tree.addTopLevelItem(item)

            button = QPushButton('查看')
            button.setObjectName('sourceViewBtn')
            button.setEnabled(source.available)
            button.setToolTip(source.image_path)
            button.setMinimumWidth(48)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.clicked.connect(
                lambda _checked=False, selected=source:
                self._request_dataset_source(selected)
            )
            self.source_tree.setItemWidget(item, 7, button)

        row_count = max(1, self.source_tree.topLevelItemCount())
        row_height = max(28, self.source_tree.sizeHintForRow(0))
        content_height = self.source_tree.header().height() + row_height * row_count + 12
        self.source_tree.setFixedHeight(min(300, max(74, content_height)))
        self.lbl_source_summary.setText(
            f'可用数据源 {available_count}/{len(record.data_sources)}  ·  '
            f'图片 {total_images}  ·  JSON {total_annotations}  ·  '
            f'YOLO 标签 {total_labels}'
        )

    def _populate_metrics(self, record: ModelRecord):
        self.metric_tree.clear()
        for metric in record.metrics:
            item = QTreeWidgetItem([
                metric.label,
                _format_metric(metric.best_value),
                str(metric.best_epoch),
                _format_metric(metric.last_value),
                metric.key,
            ])
            item.setToolTip(4, metric.key)
            self.metric_tree.addTopLevelItem(item)

    def _populate_artifacts(self, record: ModelRecord):
        role_names = {
            'best': '最佳权重',
            'last': '最终权重',
            'checkpoint': '训练检查点',
            'export': '导出模型',
        }
        self.artifact_tree.clear()
        for artifact in record.artifacts:
            item = QTreeWidgetItem([
                role_names.get(artifact.role, artifact.role),
                artifact.name,
                artifact.file_format,
                artifact.framework,
                _format_bytes(artifact.size_bytes),
                artifact.modified_at,
                artifact.path,
            ])
            item.setToolTip(1, artifact.name)
            item.setToolTip(6, artifact.path)
            self.artifact_tree.addTopLevelItem(item)

    def _populate_training_args(self, record: ModelRecord):
        self.config_tree.clear()
        for key, value in record.training_args:
            item = QTreeWidgetItem([key, value])
            item.setToolTip(1, value)
            self.config_tree.addTopLevelItem(item)

    def _populate_result_assets(self, record: ModelRecord):
        self.result_asset_combo.blockSignals(True)
        self.result_asset_combo.clear()
        for asset in record.result_assets:
            path = Path(asset)
            self.result_asset_combo.addItem(
                f'{self._result_asset_type(path)}  /  {path.name}',
                str(path),
            )
        self.result_asset_combo.blockSignals(False)
        has_results = self.result_asset_combo.count() > 0
        self.result_asset_combo.setEnabled(has_results)
        self.btn_result_prev.setEnabled(has_results)
        self.btn_result_next.setEnabled(has_results)
        if has_results:
            self.result_asset_combo.setCurrentIndex(0)
            self._show_selected_result_asset(0)
        else:
            self._current_result_pixmap = QPixmap()
            self.result_preview_label.clear()
            self.result_preview_label.setText('没有可预览的训练结果图片')
            self.lbl_result_asset_info.setText('-')

    def _show_selected_result_asset(self, index: int):
        path_text = self.result_asset_combo.itemData(index)
        if not path_text:
            return
        path = Path(str(path_text))
        pixmap = QPixmap(str(path))
        self._current_result_pixmap = pixmap
        if pixmap.isNull():
            self.result_preview_label.clear()
            self.result_preview_label.setText('训练结果图片无法加载')
            self.lbl_result_asset_info.setText(str(path))
            return
        self.result_preview_label.setText('')
        self._scale_result_preview()
        self.lbl_result_asset_info.setText(
            f'{self._result_asset_type(path)}  ·  '
            f'{pixmap.width()} x {pixmap.height()}  ·  {path}'
        )
        self.lbl_result_asset_info.setToolTip(str(path))

    def _scale_result_preview(self):
        pixmap = getattr(self, '_current_result_pixmap', QPixmap())
        if pixmap.isNull():
            return
        viewport = self.result_preview.viewport().size()
        width = max(160, viewport.width() - 18)
        height = max(120, viewport.height() - 18)
        self.result_preview_label.setPixmap(
            pixmap.scaled(
                width,
                height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def _step_result_asset(self, offset: int):
        count = self.result_asset_combo.count()
        if count:
            self.result_asset_combo.setCurrentIndex(
                (self.result_asset_combo.currentIndex() + offset) % count
            )

    @staticmethod
    def _result_asset_type(path: Path) -> str:
        name = path.stem.lower()
        if 'confusion_matrix' in name:
            return '混淆矩阵'
        if name == 'results':
            return '训练曲线总览'
        if 'curve' in name:
            return '评估曲线'
        if name.startswith('train_batch'):
            return '训练批次'
        if name.startswith('val_batch') and name.endswith('_pred'):
            return '验证预测'
        if name.startswith('val_batch'):
            return '验证标签'
        if name == 'labels':
            return '标签分布'
        return '训练结果'

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total = max(0, int(round(seconds or 0)))
        if total <= 0:
            return '-'
        days, remainder = divmod(total, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, secs = divmod(remainder, 60)
        parts = []
        if days:
            parts.append(f'{days} 天')
        if hours:
            parts.append(f'{hours} 小时')
        if minutes:
            parts.append(f'{minutes} 分')
        if not parts:
            parts.append(f'{secs} 秒')
        return ' '.join(parts)

    def _request_source_from_item(self, item):
        source = item.data(0, Qt.UserRole) if item is not None else None
        if isinstance(source, DatasetSource) and source.available:
            self._request_dataset_source(source)

    def _request_dataset_source(self, source: DatasetSource):
        if self._current_record is not None:
            self.dataset_source_requested.emit(self._current_record, source)

    def show_library(self):
        self._fade_to(self.library_page)

    def show_model_details(self, record: ModelRecord):
        """Restore a model profile after a cross-module navigation."""
        if isinstance(record, ModelRecord):
            self._show_model_details(record)

    def _fade_to(self, page: QWidget):
        self.content_stack.setCurrentWidget(page)
        effect = QGraphicsOpacityEffect(page)
        page.setGraphicsEffect(effect)
        animation = QPropertyAnimation(effect, b'opacity', self)
        animation.setDuration(190)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.OutCubic)
        animation.finished.connect(lambda: page.setGraphicsEffect(None))
        self._page_animation = animation
        animation.start()

    @staticmethod
    def _set_task_property(widget: QWidget, task_type: str):
        widget.setProperty('taskType', task_type)
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    def eventFilter(self, watched, event):
        result_preview = getattr(self, 'result_preview', None)
        if (
            result_preview is not None
            and watched is result_preview.viewport()
            and event.type() in {QEvent.Resize, QEvent.Show}
        ):
            self._scale_result_preview()
        if (
            watched is self.scroll_area.viewport()
            and event.type() in {QEvent.Resize, QEvent.Show}
        ):
            self.empty_state.setGeometry(self.cards_container.rect())
            self._relayout_cards()
        return super().eventFilter(watched, event)

    def showEvent(self, event):
        super().showEvent(event)
        self._grid_columns = 0
        self._relayout_cards()
