"""Inference center: real-time model inference workbench (5th module)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QPalette, QPixmap, QPainter
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.models.inference_worker import (
    InferenceWorker,
    KEYPOINT_COLORS,
)
from app.utils import discover_available_models

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_ROOT = PROJECT_ROOT / 'reports' / 'inference'

SOURCE_KINDS = (
    ('camera', '实时摄像头'),
    ('video', '视频文件'),
    ('images', '图片目录'),
    ('rtsp', 'RTSP 网络流'),
)


class InferenceCanvas(QWidget):
    """Dark canvas that renders the latest annotated frame aspect-fit."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('inferenceCanvas')
        self._pixmap: QPixmap | None = None
        self.setMinimumSize(480, 300)

    def set_frame(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self.update()

    def clear_frame(self):
        self._pixmap = None
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(8, 16, 25))
        if self._pixmap is None or self._pixmap.isNull():
            painter.setPen(QColor(126, 156, 173))
            painter.drawText(
                self.rect(), Qt.AlignCenter,
                '选择输入源并点击「开始」\n'
                '支持：实时摄像头 / 视频文件 / 图片目录 / RTSP',
            )
            painter.end()
            return
        target = self.rect()
        scaled = self._pixmap.scaled(
            target.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        x = target.x() + (target.width() - scaled.width()) // 2
        y = target.y() + (target.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)
        painter.end()


class InferenceCenterView(QWidget):
    """Top-level inference module with its own worker lifecycle."""

    def __init__(self, parent=None, model_factory=None):
        super().__init__(parent)
        self._worker: InferenceWorker | None = None
        self._model_factory = model_factory  # injected for tests
        self._model = None
        self._record_path: Path | None = None
        self._extra_repo = ''

        palette = self.palette()
        from PyQt5.QtGui import QPalette
        dark = QColor(8, 16, 25)
        palette.setColor(QPalette.Window, dark)
        palette.setColor(QPalette.Base, dark)
        palette.setColor(QPalette.WindowText, QColor(216, 226, 239))
        palette.setColor(QPalette.Text, QColor(216, 226, 239))
        palette.setColor(QPalette.Button, QColor(17, 24, 33))
        palette.setColor(QPalette.ButtonText, QColor(216, 226, 239))
        palette.setColor(QPalette.Highlight, QColor(36, 104, 150))
        self.setPalette(palette)

        self._build_ui()
        self._refresh_model_choices()
        for child in self.findChildren(QWidget):
            child.setPalette(self.palette())

    # ---- UI ----

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(12)

        header = QWidget()
        header.setObjectName('trainingHeader')
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(4, 2, 4, 2)
        header_layout.setSpacing(2)
        eyebrow = QLabel('INFERENCE WORKBENCH')
        eyebrow.setObjectName('trainingEyebrow')
        header_layout.addWidget(eyebrow)
        title_row = QHBoxLayout()
        title = QLabel('推理中心')
        title.setObjectName('trainingTitle')
        title_row.addWidget(title)
        hint = QLabel('模型 × 实时画面 → 现场诊断')
        hint.setObjectName('duplicateScope')
        title_row.addWidget(hint)
        title_row.addStretch()
        self.header_badge = QLabel('REALTIME · PREVIEW')
        self.header_badge.setObjectName('trainingEnvironmentBadge')
        title_row.addWidget(self.header_badge)
        header_layout.addLayout(title_row)
        layout.addWidget(header)

        panel = QWidget()
        panel.setObjectName('evaluationPanel')
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(16, 14, 16, 14)
        panel_layout.setSpacing(10)

        model_row = QHBoxLayout()
        model_caption = QLabel('模型')
        model_caption.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        model_row.addWidget(model_caption)
        self.combo_model = QComboBox()
        self.combo_model.setObjectName('trainingCombo')
        self.combo_model.setEditable(True)
        model_row.addWidget(self.combo_model, 1)
        btn_model = QPushButton('浏览 .pt')
        btn_model.setObjectName('fileOpBtn')
        btn_model.clicked.connect(self._browse_model)
        model_row.addWidget(btn_model)
        panel_layout.addLayout(model_row)

        source_row = QHBoxLayout()
        source_caption = QLabel('输入源')
        source_caption.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        source_row.addWidget(source_caption)

        # 输入源控件容器：隐藏项不参与布局，始终左对齐
        self.source_container = QWidget()
        self.source_container.setObjectName('inferSourceGroup')
        container_layout = QHBoxLayout(self.source_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(8)
        self.combo_source = QComboBox()
        self.combo_source.setObjectName('trainingCombo')
        for kind, label in SOURCE_KINDS:
            self.combo_source.addItem(label, kind)
        self.combo_source.currentIndexChanged.connect(self._on_source_changed)
        container_layout.addWidget(self.combo_source)
        self.spin_camera = QSpinBox()
        self.spin_camera.setObjectName('trainingSpin')
        self.spin_camera.setRange(0, 8)
        self.spin_camera.setFixedWidth(120)
        container_layout.addWidget(self.spin_camera)
        self.edit_source_path = QLineEdit()
        self.edit_source_path.setObjectName('trainingEdit')
        self.edit_source_path.setPlaceholderText('选择视频文件 / 图片目录 / RTSP 地址')
        container_layout.addWidget(self.edit_source_path, 1)
        btn_source = QPushButton('选择')
        btn_source.setObjectName('fileOpBtn')
        btn_source.clicked.connect(self._pick_source_path)
        self.btn_source = btn_source
        container_layout.addWidget(btn_source)
        container_layout.addStretch(1)
        source_row.addWidget(self.source_container, 1)
        panel_layout.addLayout(source_row)

        param_row = QHBoxLayout()
        self.spin_conf = self._spin(0.0001, 1.0, 0.25)
        self.spin_iou = self._spin(0.1, 1.0, 0.6)
        for caption, control in (
            ('conf', self.spin_conf),
            ('iou', self.spin_iou),
        ):
            label = QLabel(caption)
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            param_row.addWidget(label)
            param_row.addWidget(control)
        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setObjectName('trainingSpin')
        self.spin_imgsz.setRange(160, 2560)
        self.spin_imgsz.setSingleStep(32)
        self.spin_imgsz.setValue(480)
        self.spin_imgsz.setFixedWidth(110)
        imgsz_label = QLabel('imgsz')
        imgsz_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        param_row.addWidget(imgsz_label)
        param_row.addWidget(self.spin_imgsz)
        self.edit_device = QLineEdit('auto')
        self.edit_device.setObjectName('trainingEdit')
        self.edit_device.setToolTip('如 0、0,1 或 cpu；auto 自动选择')
        self.edit_device.setFixedWidth(110)
        device_label = QLabel('device')
        device_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        param_row.addWidget(device_label)
        param_row.addWidget(self.edit_device)
        self.check_half = QCheckBox('半精度 FP16（速度优先）')
        self.check_half.setObjectName('trainingCheck')
        self.check_half.setToolTip(
            '使用 Ultralytics 官方 quantize=16 机制；若不兼容会自动回退 FP32，'
            '稳定性优先于速度时请保持关闭。'
        )
        param_row.addWidget(self.check_half)
        param_row.addStretch()
        panel_layout.addLayout(param_row)
        layout.addWidget(panel)

        actions = QHBoxLayout()
        self.btn_start = QPushButton('开始')
        self.btn_start.setObjectName('primaryBtn')
        self.btn_start.clicked.connect(self._start)
        actions.addWidget(self.btn_start)
        self.btn_pause = QPushButton('暂停')
        self.btn_pause.setObjectName('fileOpBtn')
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self._toggle_pause)
        actions.addWidget(self.btn_pause)
        self.btn_stop = QPushButton('停止')
        self.btn_stop.setObjectName('dangerBtn')
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        actions.addWidget(self.btn_stop)
        self.btn_snapshot = QPushButton('截图')
        self.btn_snapshot.setObjectName('successBtn')
        self.btn_snapshot.setEnabled(False)
        self.btn_snapshot.clicked.connect(self._snapshot)
        actions.addWidget(self.btn_snapshot)
        self.btn_record = QPushButton('开始录制')
        self.btn_record.setObjectName('fileOpBtn')
        self.btn_record.setEnabled(False)
        self.btn_record.clicked.connect(self._toggle_record)
        actions.addWidget(self.btn_record)
        actions.addStretch()
        self.status_label = QLabel('就绪')
        self.status_label.setObjectName('duplicateScope')
        actions.addWidget(self.status_label)
        layout.addLayout(actions)

        self.canvas = InferenceCanvas()
        body = QSplitter(Qt.Horizontal)
        body.setObjectName('evaluationBodySplitter')
        body.setChildrenCollapsible(False)
        body.setHandleWidth(7)
        body.addWidget(self.canvas)
        body.setStretchFactor(0, 1)

        self.legend_panel = QWidget()
        self.legend_panel.setObjectName('evaluationPanel')
        self.legend_panel.setMinimumWidth(260)
        self.legend_panel.setMaximumWidth(340)
        legend_layout = QVBoxLayout(self.legend_panel)
        legend_layout.setContentsMargins(14, 12, 14, 12)
        legend_layout.setSpacing(10)

        self.legend_title = QLabel('关键点图例')
        self.legend_title.setObjectName('trainingSectionTitle')
        legend_layout.addWidget(self.legend_title)
        self.legend_hint = QLabel('点击名称控制该关键点标签是否显示在画面中')
        self.legend_hint.setObjectName('evaluationSectionHint')
        self.legend_hint.setWordWrap(True)
        legend_layout.addWidget(self.legend_hint)

        legend_buttons = QHBoxLayout()
        btn_all_on = QPushButton('全部显示')
        btn_all_on.setObjectName('fileOpBtn')
        btn_all_on.clicked.connect(lambda: self._toggle_all_labels(True))
        legend_buttons.addWidget(btn_all_on)
        btn_all_off = QPushButton('全部隐藏')
        btn_all_off.setObjectName('fileOpBtn')
        btn_all_off.clicked.connect(lambda: self._toggle_all_labels(False))
        legend_buttons.addWidget(btn_all_off)
        legend_buttons.addStretch()
        legend_layout.addLayout(legend_buttons)

        self.legend_scroll = QScrollArea()
        self.legend_scroll.setObjectName('evalTaskScroll')
        self.legend_scroll.setWidgetResizable(True)
        self.legend_scroll.setFrameShape(QFrame.NoFrame)
        self.legend_scroll.viewport().setAutoFillBackground(False)
        legend_dark = QPalette(self.palette())
        legend_dark.setColor(QPalette.Window, QColor(8, 16, 25))
        legend_dark.setColor(QPalette.Base, QColor(8, 16, 25))
        legend_dark.setColor(QPalette.AlternateBase, QColor(18, 37, 52))
        self.legend_scroll.setPalette(legend_dark)
        self.legend_host = QWidget()
        self.legend_host.setObjectName('inferLegendHost')
        self.legend_host.setAutoFillBackground(False)
        self.legend_host.setPalette(legend_dark)
        self.legend_grid = QGridLayout(self.legend_host)
        self.legend_grid.setContentsMargins(0, 2, 0, 2)
        self.legend_grid.setSpacing(6)
        self.legend_scroll.setWidget(self.legend_host)
        legend_layout.addWidget(self.legend_scroll, 1)

        self._legend_checks: list[QCheckBox] = []
        self._kpt_names: list[str] = []
        self._class_names: list[str] = []
        body.addWidget(self.legend_panel)
        layout.addWidget(body, 1)

        self._apply_source_mode()

    @staticmethod
    def _spin(minimum: float, maximum: float, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setObjectName('trainingSpin')
        spin.setRange(minimum, maximum)
        spin.setDecimals(4)
        spin.setValue(value)
        spin.setFixedWidth(110)
        return spin

    def _apply_source_mode(self):
        kind = self.combo_source.currentData()
        self.spin_camera.setVisible(kind == 'camera')
        self.edit_source_path.setVisible(kind != 'camera')
        self.btn_source.setVisible(kind != 'camera')
        self.btn_browse_hint()

    def btn_browse_hint(self):
        kind = self.combo_source.currentData()
        if kind == 'video':
            self.edit_source_path.setPlaceholderText('选择视频文件（mp4/avi/mkv/mov）')
        elif kind == 'images':
            self.edit_source_path.setPlaceholderText('选择图片目录')
        elif kind == 'rtsp':
            self.edit_source_path.setPlaceholderText('如 rtsp://192.168.1.10:554/stream')
        else:
            self.edit_source_path.setPlaceholderText('')

    def _on_source_changed(self):
        self._apply_source_mode()

    # ---- model / source helpers ----

    def _refresh_model_choices(self, keep_current: str = ''):
        current = keep_current or self.combo_model.currentText()
        self.combo_model.clear()
        for path in discover_available_models(self._extra_repo):
            label = (
                path.parent.parent.name if path.parent.name == 'weights'
                else path.name
            )
            self.combo_model.addItem(label, str(path))
        self.combo_model.setEditText(current)

    def set_model_repository(self, repo_path: str):
        """Link the model-management repository into this view's choices."""
        self._extra_repo = str(repo_path or '')
        if self._worker is None:
            self._refresh_model_choices()

    def _browse_model(self):
        start_dir = self._extra_repo or str(PROJECT_ROOT / 'models')
        path, _f = QFileDialog.getOpenFileName(
            self, '选择模型权重', start_dir,
            '模型 (*.pt);;所有文件 (*)',
        )
        if path:
            self.combo_model.setEditText(path)

    def _pick_source_path(self):
        kind = self.combo_source.currentData()
        if kind == 'video':
            path, _f = QFileDialog.getOpenFileName(
                self, '选择视频文件', str(PROJECT_ROOT),
                '视频 (*.mp4 *.avi *.mkv *.mov *.webm);;所有文件 (*)',
            )
        elif kind == 'images':
            path = QFileDialog.getExistingDirectory(self, '选择图片目录')
        else:
            return
        if path:
            self.edit_source_path.setText(path)

    def _current_source(self) -> dict:
        kind = self.combo_source.currentData()
        if kind == 'camera':
            return {'kind': 'camera', 'value': int(self.spin_camera.value())}
        return {'kind': kind, 'value': self.edit_source_path.text().strip()}

    def _current_parameters(self) -> dict:
        return {
            'conf': float(self.spin_conf.value()),
            'iou': float(self.spin_iou.value()),
            'imgsz': int(self.spin_imgsz.value()),
            'device': self.edit_device.text().strip() or 'auto',
            'half': self.check_half.isChecked(),
        }

    # ---- lifecycle ----

    def _start(self):
        source = self._current_source()
        if source.get('kind') != 'camera' and not source.get('value'):
            QMessageBox.warning(self, '无法开始', '请选择输入源路径或地址。')
            return
        model_path = Path(self.combo_model.currentText()).expanduser()
        if not model_path.is_file():
            QMessageBox.warning(self, '无法开始', f'模型文件不存在: {model_path}')
            return
        try:
            if self._model_factory is not None:
                predictor = self._model_factory(str(model_path))
            else:
                from ultralytics import YOLO  # lazy import
                predictor = YOLO(str(model_path))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, '模型加载失败', f'{type(exc).__name__}: {exc}')
            return
        self._model = predictor
        self._worker = InferenceWorker(
            predictor, source, self._current_parameters(), parent=self,
        )
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.stats_ready.connect(self._on_stats)
        self._worker.status_changed.connect(
            lambda text: self.status_label.setText(text)
        )
        self._worker.error_occurred.connect(self._on_worker_error)
        self._worker.keypoints_ready.connect(self._on_keypoints_ready)
        self._worker.classes_ready.connect(self._on_classes_ready)
        self._worker.task_ready.connect(self._on_task_ready)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()
        self._set_running(True)
        self._reset_legend()
        self._show_device_info(predictor)

    def _reset_legend(self):
        while self.legend_grid.count():
            item = self.legend_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._legend_checks = []
        self._kpt_names = []
        self._class_names = []
        self.legend_title.setText('模型结构图例')
        hint = QLabel('等待模型识别结构…')
        hint.setObjectName('evaluationSectionHint')
        self.legend_grid.addWidget(hint, 0, 0)

    def _build_legend_items(self, names, default_checked: bool):
        while self.legend_grid.count():
            item = self.legend_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._legend_checks = []
        for index, name in enumerate(names):
            color = KEYPOINT_COLORS[index % len(KEYPOINT_COLORS)]
            chip = QLabel('  ')
            chip.setFixedSize(14, 14)
            chip.setStyleSheet(
                f'background-color: rgb({color[0]},{color[1]},{color[2]});'
                'border-radius: 4px;'
            )
            check = QCheckBox(str(name))
            check.setObjectName('trainingCheck')
            check.setChecked(default_checked)
            check.toggled.connect(self._on_legend_toggled)
            row, column = divmod(index, 2)
            cell = QWidget()
            cell_layout = QHBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(6)
            cell_layout.addWidget(chip)
            cell_layout.addWidget(check)
            cell_layout.addStretch()
            self.legend_grid.addWidget(cell, row, column)
            self._legend_checks.append(check)

    def _on_keypoints_ready(self, names):
        self._kpt_names = list(names or [])
        self.legend_title.setText('关键点图例')
        self.legend_hint.setText('点击名称控制该关键点标签是否显示在画面中')
        self._build_legend_items(self._kpt_names, default_checked=False)

    _TASK_LABELS = {
        'pose': 'POSE', 'detect': 'DETECTION',
        'segment': 'SEGMENTATION', 'obb': 'OBB',
    }

    def _on_task_ready(self, task: str):
        label = self._TASK_LABELS.get(str(task), str(task).upper())
        self.header_badge.setText(f'TASK · {label} · REALTIME')

    def _on_classes_ready(self, names):
        self._class_names = list(names or [])
        self.legend_title.setText('类别图例')
        self.legend_hint.setText('点击名称控制该类别的框 / 掩码 / 旋转框显示')
        self._build_legend_items(self._class_names, default_checked=True)

    def _on_legend_toggled(self, *_args):
        enabled = {
            check.text()
            for check in self._legend_checks if check.isChecked()
        }
        if self._worker is None:
            return
        if self._kpt_names:
            self._worker.set_keypoint_labels(enabled)
        elif self._class_names:
            if len(enabled) == len(self._class_names):
                self._worker.set_visible_classes(None)
            else:
                self._worker.set_visible_classes(enabled)

    def _toggle_all_labels(self, checked: bool):
        for check in self._legend_checks:
            check.setChecked(checked)

    def _on_frame(self, image):
        pixmap = QPixmap.fromImage(image)
        self.canvas.set_frame(pixmap)
        if not self.btn_snapshot.isEnabled():
            self.btn_snapshot.setEnabled(True)
        if not self.btn_record.isEnabled():
            self.btn_record.setEnabled(True)

    def _on_stats(self, stats):
        self.status_label.setText(
            f"FPS {stats.get('fps', 0)} · "
            f"pre {stats.get('pre_ms', 0)}ms / infer {stats.get('infer_ms', 0)}ms / "
            f"post {stats.get('post_ms', 0)}ms · "
            f'帧 {stats.get("frame", 0)} · 目标 {sum((stats.get("counts") or {}).values())}'
        )

    def _on_worker_error(self, message: str):
        self.status_label.setText(f'错误：{message}')
        QMessageBox.warning(self, '推理错误', message)

    def _on_worker_finished(self):
        self._set_running(False)
        self.status_label.setText('已停止')
        self._worker = None

    def _set_running(self, running: bool):
        self.btn_start.setEnabled(not running)
        self.btn_pause.setEnabled(running)
        self.btn_stop.setEnabled(running)
        self.btn_snapshot.setEnabled(running)
        self.btn_record.setEnabled(running)
        self.combo_model.setEnabled(not running)
        self.combo_source.setEnabled(not running)
        # 运行中锁定推理参数（运行中修改不会生效，明确禁用避免误解）
        for control in (
            self.spin_conf, self.spin_iou, self.spin_imgsz, self.edit_device,
            self.check_half,
        ):
            control.setEnabled(not running)
        if not running:
            self.btn_pause.setText('暂停')
            self.btn_record.setText('开始录制')

    def _show_device_info(self, predictor):
        try:
            import torch
            device_label = '未知'
            if torch.cuda.is_available():
                device_label = f'CUDA · {torch.cuda.get_device_name(0)}'
            else:
                device_label = 'CPU'
            dtype = 'fp32'
            try:
                params = list(next(getattr(predictor, 'model', None)
                                   .parameters()) if getattr(predictor, 'model', None) else [])
                if params:
                    dtype = str(params.dtype).replace('torch.', '')
            except Exception:  # noqa: BLE001
                pass
            self.status_label.setText(
                f'推理设备: {device_label}（{dtype}） · 请选择输入源并开始'
            )
        except Exception:  # noqa: BLE001
            pass

    def _toggle_pause(self):
        if self._worker is None:
            return
        paused = self.btn_pause.text() != '继续'
        self._worker.set_paused(paused)
        self.btn_pause.setText('继续' if paused else '暂停')

    def _stop(self):
        if self._worker is None:
            return
        self._worker.request_stop()
        self._worker.wait(3000)

    def _snapshot(self):
        if self._worker is None:
            return
        INFERENCE_ROOT.mkdir(parents=True, exist_ok=True)
        name = time.strftime('snapshot_%Y%m%d_%H%M%S')
        if self._worker.save_snapshot(INFERENCE_ROOT / f'{name}.png'):
            self.status_label.setText(f'已截图：{INFERENCE_ROOT / name}.png')

    def _toggle_record(self):
        if self._worker is None:
            return
        if self._worker.is_recording:
            path = self._worker.stop_recording()
            self.btn_record.setText('开始录制')
            if path:
                self.status_label.setText(f'录制完成：{path.name}')
        else:
            INFERENCE_ROOT.mkdir(parents=True, exist_ok=True)
            path = INFERENCE_ROOT / time.strftime('record_%Y%m%d_%H%M%S.avi')
            self._record_path = path
            self._worker.start_recording(path)
            self.btn_record.setText('停止录制')
            self.status_label.setText(f'正在录制：{path.name}')

    def closeEvent(self, event):
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()
            if not self._worker.wait(3000):
                self._worker.terminate()
                self._worker.wait(1000)
        super().closeEvent(event)

    def prefill_model(self, model_path: str, model_label: str):
        resolved = str(Path(model_path).expanduser().resolve())
        self._refresh_model_choices(keep_current=resolved)
        self.combo_model.setEditText(resolved)
        self.status_label.setText(f'已选择模型：{model_label} · 请选择输入源并开始')
