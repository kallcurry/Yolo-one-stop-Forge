"""JSON editor for task-specific Ultralytics training templates."""

from __future__ import annotations

import json
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from app.models.training_config import (
    TrainingConfig,
    training_config_from_dict,
    training_config_to_dict,
)


class TrainingTemplateDialog(QDialog):
    """Edit, validate, and save a training template as JSON."""

    def __init__(self, current_config: TrainingConfig,
                 default_template_dir: str | Path, parent=None):
        super().__init__(parent)
        self._current_config = current_config
        self._task_type = current_config.task_type
        self._default_template_dir = Path(default_template_dir)
        self._saved_config: TrainingConfig | None = None
        self._saved_path: Path | None = None
        self.setWindowTitle('训练高级配置')
        self.resize(900, 700)
        self.setObjectName('trainingTemplateDialog')
        self._build_ui()
        self._load_current()

    def saved_config(self) -> TrainingConfig | None:
        return self._saved_config

    def saved_path(self) -> Path | None:
        return self._saved_path

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 14)
        layout.setSpacing(10)

        heading = QHBoxLayout()
        title_box = QVBoxLayout()
        eyebrow = QLabel('TRAINING TEMPLATE')
        eyebrow.setObjectName('trainingDialogEyebrow')
        title = QLabel('高级训练参数')
        title.setObjectName('trainingDialogTitle')
        title_box.addWidget(eyebrow)
        title_box.addWidget(title)
        heading.addLayout(title_box)
        heading.addStretch()
        task = QLabel(self._task_type.upper())
        task.setObjectName('trainingDialogTaskBadge')
        heading.addWidget(task)
        layout.addLayout(heading)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(7)
        self.btn_load_current = QPushButton('载入当前配置')
        self.btn_open_json = QPushButton('打开 JSON')
        self.btn_format = QPushButton('格式化 / 校验')
        self.btn_save = QPushButton('保存为新模板并切换')
        self.btn_save.setObjectName('primaryBtn')
        self.btn_load_current.clicked.connect(self._load_current)
        self.btn_open_json.clicked.connect(self._open_json)
        self.btn_format.clicked.connect(self._format_json)
        self.btn_save.clicked.connect(self._save_as)
        toolbar.addWidget(self.btn_load_current)
        toolbar.addWidget(self.btn_open_json)
        toolbar.addWidget(self.btn_format)
        toolbar.addStretch()
        toolbar.addWidget(self.btn_save)
        layout.addLayout(toolbar)

        self.editor = QPlainTextEdit()
        self.editor.setObjectName('trainingTemplateEditor')
        self.editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.editor.setTabStopDistance(28)
        layout.addWidget(self.editor, 1)

        footer = QHBoxLayout()
        self.status_label = QLabel('就绪')
        self.status_label.setObjectName('trainingDialogStatus')
        footer.addWidget(self.status_label, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.button(QDialogButtonBox.Close).setText('关闭')
        buttons.rejected.connect(self.reject)
        footer.addWidget(buttons)
        layout.addLayout(footer)

    def _load_current(self):
        self.editor.setPlainText(self._dump(self._current_config))
        self.status_label.setText('已载入当前表单参数')

    def _open_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            '打开训练模板',
            str(self._default_template_dir),
            '训练模板 (*.json);;JSON 文件 (*.json);;所有文件 (*)',
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding='utf-8'))
            config = training_config_from_dict(data, path)
            self._ensure_task(config)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            QMessageBox.warning(self, '打开失败', str(exc))
            return
        self.editor.setPlainText(self._dump(config))
        self.status_label.setText(f'已打开: {path}')

    def _format_json(self):
        try:
            config = self._read_config()
        except (json.JSONDecodeError, ValueError) as exc:
            QMessageBox.warning(self, '校验失败', str(exc))
            return
        self.editor.setPlainText(self._dump(config))
        self.status_label.setText(
            f'校验通过 · {len(config.parameters)} 个训练参数'
        )

    def _save_as(self):
        try:
            config = self._read_config()
        except (json.JSONDecodeError, ValueError) as exc:
            QMessageBox.warning(self, '保存失败', str(exc))
            return
        self._default_template_dir.mkdir(parents=True, exist_ok=True)
        default_name = self._safe_name(config.name) + '.json'
        path, _ = QFileDialog.getSaveFileName(
            self,
            '保存训练模板',
            str(self._default_template_dir / default_name),
            '训练模板 (*.json);;JSON 文件 (*.json)',
        )
        if not path:
            return
        save_path = Path(path)
        if save_path.suffix.lower() != '.json':
            save_path = save_path.with_suffix('.json')
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(self._dump(config) + '\n', encoding='utf-8')
            saved = training_config_from_dict(
                training_config_to_dict(config), save_path
            )
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, '保存失败', str(exc))
            return
        self._saved_config = saved
        self._saved_path = save_path
        self.accept()

    def _read_config(self) -> TrainingConfig:
        data = json.loads(self.editor.toPlainText())
        config = training_config_from_dict(data)
        self._ensure_task(config)
        return config

    def _ensure_task(self, config: TrainingConfig):
        if config.task_type != self._task_type:
            raise ValueError(
                f'当前正在编辑 {self._task_type}，模板中的 task_type 是 '
                f'{config.task_type}'
            )

    @staticmethod
    def _dump(config: TrainingConfig) -> str:
        return json.dumps(
            training_config_to_dict(config), ensure_ascii=False, indent=2
        )

    @staticmethod
    def _safe_name(value: str) -> str:
        name = ''.join(
            character if character.isalnum() or character in '-_' else '_'
            for character in value
        )
        return name.strip('_') or 'training_template'
