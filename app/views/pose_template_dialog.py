"""Annotation review template editor dialog."""

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
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.models.annotation_review import (
    pose_review_config_from_dict,
    pose_review_config_to_dict,
)


PLUGIN_TEMPLATE = '''"""Custom annotation review plugin.

Return a list of dict issues. You do not need to import Qt classes here.
The context helper provides:
  - context.data
  - context.shapes
  - context.image_size
  - context.group_ids()
  - context.points_in_group(group_id)
  - context.boxes_in_group(group_id)
  - context.issue(...)
"""


def check(context, rule):
    issues = []
    params = rule.get("params", {})
    required_label = params.get("required_label", "")

    for shape_idx, shape in enumerate(context.shapes):
        label = str(shape.get("label", ""))
        points = shape.get("points") or []

        if required_label and label != required_label:
            continue

        # Replace this example condition with your own scene rule.
        if not points:
            issues.append(context.issue(
                rule_id=rule.get("id", "python_custom_rule"),
                severity=rule.get("severity", "warning"),
                message=f"{label or 'unnamed'} has no points",
                group_id=shape.get("group_id"),
                label=label,
                shape_indices=[shape_idx],
                point_indices=[],
            ))

    return issues
'''


class PoseTemplateDialog(QDialog):
    """Edit/save review templates and copy Python plugin code."""

    def __init__(self, current_template: dict,
                 default_template_dir: str | Path,
                 default_plugin_dir: str | Path,
                 project_root: str | Path,
                 parent=None):
        super().__init__(parent)
        self._current_template = current_template
        self._default_template_dir = Path(default_template_dir)
        self._default_plugin_dir = Path(default_plugin_dir)
        self._project_root = Path(project_root)
        self._task_type = self._template_task_type(current_template)
        self._task_name = self._task_display_name(self._task_type)
        self._saved_template_path: Path | None = None
        self._plugin_source_name = f'custom_{self._safe_filename(self._task_type)}_rule.py'

        self.setWindowTitle(f'{self._task_name} 审查模板管理')
        self.resize(920, 720)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        title = QLabel(
            f'正在编辑 {self._task_name} 任务模板。可以基于当前 JSON 修改，'
            '也可以复制 Python 插件后另存为新模板。'
        )
        title.setObjectName('reviewStats')
        title.setWordWrap(True)
        layout.addWidget(title)

        tabs = QTabWidget()
        tabs.addTab(self._build_template_tab(), '模板 JSON')
        tabs.addTab(self._build_plugin_tab(), 'Python 插件')
        layout.addWidget(tabs, 1)

        self.status_label = QLabel('就绪')
        self.status_label.setObjectName('reviewStats')
        layout.addWidget(self.status_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._load_current_template()

    def saved_template_path(self) -> Path | None:
        return self._saved_template_path

    def _build_template_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 8, 6, 6)
        layout.setSpacing(8)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)
        self.btn_load_current = QPushButton('载入当前模板')
        self.btn_open_json = QPushButton('打开 JSON')
        self.btn_format_json = QPushButton('格式化/校验')
        self.btn_save_template = QPushButton('保存为新模板并切换')
        self.btn_save_template.setObjectName('primaryBtn')
        for button in (
            self.btn_load_current,
            self.btn_open_json,
            self.btn_format_json,
            self.btn_save_template,
        ):
            button.setMinimumHeight(30)
            button.setMaximumHeight(34)
        self.btn_load_current.clicked.connect(self._load_current_template)
        self.btn_open_json.clicked.connect(self._open_json_file)
        self.btn_format_json.clicked.connect(self._format_json)
        self.btn_save_template.clicked.connect(self._save_template_as)
        toolbar.addWidget(self.btn_load_current)
        toolbar.addWidget(self.btn_open_json)
        toolbar.addWidget(self.btn_format_json)
        toolbar.addStretch()
        toolbar.addWidget(self.btn_save_template)
        layout.addLayout(toolbar)

        self.template_editor = QPlainTextEdit()
        self.template_editor.setObjectName('templateEditor')
        self.template_editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        layout.addWidget(self.template_editor, 1)
        return tab

    def _build_plugin_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 8, 6, 6)
        layout.setSpacing(8)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)
        self.btn_plugin_template = QPushButton('插入插件模板')
        self.btn_open_plugin = QPushButton('打开 .py')
        self.btn_save_plugin = QPushButton('复制到插件目录并引用')
        self.btn_save_plugin.setObjectName('successBtn')
        for button in (
            self.btn_plugin_template,
            self.btn_open_plugin,
            self.btn_save_plugin,
        ):
            button.setMinimumHeight(30)
            button.setMaximumHeight(34)
        self.btn_plugin_template.clicked.connect(self._load_plugin_template)
        self.btn_open_plugin.clicked.connect(self._open_plugin_file)
        self.btn_save_plugin.clicked.connect(self._save_plugin_and_reference)
        toolbar.addWidget(self.btn_plugin_template)
        toolbar.addWidget(self.btn_open_plugin)
        toolbar.addStretch()
        toolbar.addWidget(self.btn_save_plugin)
        layout.addLayout(toolbar)

        hint = QLabel(
            '插件会保存为 .py 文件，并自动在 JSON 的 custom_rules 中插入 python 规则引用。'
        )
        hint.setObjectName('reviewStats')
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.plugin_editor = QPlainTextEdit()
        self.plugin_editor.setObjectName('templateEditor')
        self.plugin_editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        layout.addWidget(self.plugin_editor, 1)
        self._load_plugin_template()
        return tab

    def _load_current_template(self):
        self.template_editor.setPlainText(self._dump_json(self._current_template))
        self._set_status('已载入当前模板，可直接编辑后另存。')

    def _open_json_file(self):
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            '打开审查模板',
            str(self._default_template_dir),
            '审查模板 (*.json);;JSON 文件 (*.json);;所有文件 (*)',
        )
        if not path:
            return
        try:
            text = Path(path).read_text(encoding='utf-8')
            data = json.loads(text)
            config = self._validate_template_data(data, path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            QMessageBox.warning(self, '打开失败', f'无法打开模板:\n{path}\n\n{exc}')
            return
        self.template_editor.setPlainText(
            self._dump_json(pose_review_config_to_dict(config))
        )
        self._set_status(f'已打开模板: {path}')

    def _format_json(self):
        try:
            data = self._read_template_data()
            config = self._validate_template_data(data)
        except (json.JSONDecodeError, ValueError) as exc:
            QMessageBox.warning(self, '校验失败', str(exc))
            return
        self.template_editor.setPlainText(
            self._dump_json(pose_review_config_to_dict(config))
        )
        self._set_status('JSON 校验通过。')

    def _save_template_as(self):
        try:
            data = self._read_template_data()
            config = self._validate_template_data(data)
            data = pose_review_config_to_dict(config)
        except (json.JSONDecodeError, ValueError) as exc:
            QMessageBox.warning(self, '保存失败', f'模板校验失败:\n\n{exc}')
            return

        self._default_template_dir.mkdir(parents=True, exist_ok=True)
        default_name = self._safe_filename(
            config.name or f'{config.task_type}_review_template'
        )
        default_path = self._default_template_dir / f'{default_name}.json'
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            '保存为新审查模板',
            str(default_path),
            '审查模板 (*.json);;JSON 文件 (*.json);;所有文件 (*)',
        )
        if not path:
            return

        save_path = Path(path)
        if save_path.suffix.lower() != '.json':
            save_path = save_path.with_suffix('.json')
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(self._dump_json(data) + '\n', encoding='utf-8')
        except OSError as exc:
            QMessageBox.warning(self, '保存失败', f'无法写入模板:\n{save_path}\n\n{exc}')
            return

        self._saved_template_path = save_path
        self._set_status(f'已保存新模板: {save_path}')
        self.accept()

    def _load_plugin_template(self):
        self.plugin_editor.setPlainText(PLUGIN_TEMPLATE)
        self._plugin_source_name = (
            f'custom_{self._safe_filename(self._task_type)}_rule.py'
        )
        self._set_status('已插入 Python 插件模板。')

    def _open_plugin_file(self):
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            '打开 Python 插件',
            str(self._default_plugin_dir),
            'Python 文件 (*.py);;所有文件 (*)',
        )
        if not path:
            return
        try:
            self.plugin_editor.setPlainText(Path(path).read_text(encoding='utf-8'))
        except OSError as exc:
            QMessageBox.warning(self, '打开失败', f'无法打开插件:\n{path}\n\n{exc}')
            return
        self._plugin_source_name = Path(path).name
        self._set_status(f'已打开插件: {path}')

    def _save_plugin_and_reference(self):
        self._default_plugin_dir.mkdir(parents=True, exist_ok=True)
        default_path = self._default_plugin_dir / self._plugin_source_name
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            '复制 Python 插件',
            str(default_path),
            'Python 文件 (*.py);;所有文件 (*)',
        )
        if not path:
            return

        plugin_path = Path(path)
        if plugin_path.suffix.lower() != '.py':
            plugin_path = plugin_path.with_suffix('.py')
        try:
            plugin_path.parent.mkdir(parents=True, exist_ok=True)
            plugin_path.write_text(self.plugin_editor.toPlainText(), encoding='utf-8')
            self._insert_python_rule_reference(plugin_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            QMessageBox.warning(
                self,
                '插件保存失败',
                f'无法保存插件或写入模板引用:\n{plugin_path}\n\n{exc}',
            )
            return

        self._set_status(f'已复制插件并写入 custom_rules: {plugin_path}')

    def _insert_python_rule_reference(self, plugin_path: Path):
        data = self._read_template_data()
        rules = data.setdefault('custom_rules', [])
        if not isinstance(rules, list):
            raise ValueError('custom_rules 必须是数组')

        rel_path = self._relative_path(plugin_path)
        rule_id = self._unique_rule_id(
            plugin_path.stem,
            {
                str(rule.get('id', '')).strip()
                for rule in rules
                if isinstance(rule, dict)
            },
        )
        rules.append({
            'id': rule_id,
            'name': f'{plugin_path.stem} Python 规则',
            'type': 'python',
            'path': rel_path,
            'function': 'check',
            'severity': 'warning',
            'params': {},
        })
        self._validate_template_data(data)
        self.template_editor.setPlainText(self._dump_json(data))

    def _read_template_data(self) -> dict:
        data = json.loads(self.template_editor.toPlainText())
        if not isinstance(data, dict):
            raise ValueError('模板根节点必须是 JSON object')
        return data

    def _validate_template_data(self, data: dict,
                                path: str | Path | None = None):
        config = pose_review_config_from_dict(data, path)
        if config.task_type != self._task_type:
            raise ValueError(
                f'当前窗口正在编辑 {self._task_name} 模板，但 JSON 中 '
                f'task_type="{config.task_type}"。请先切换到对应任务再编辑。'
            )
        return config

    @staticmethod
    def _dump_json(data: dict) -> str:
        return json.dumps(data, ensure_ascii=False, indent=2)

    def _relative_path(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self._project_root.resolve()))
        except ValueError:
            return str(path)

    @staticmethod
    def _safe_filename(name: str) -> str:
        safe = ''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in name)
        return safe.strip('_') or 'review_template'

    @staticmethod
    def _template_task_type(template: dict) -> str:
        task_type = str(template.get('task_type') or 'pose').strip()
        return task_type or 'pose'

    @staticmethod
    def _task_display_name(task_type: str) -> str:
        names = {
            'pose': '姿态',
            'detection': '目标检测',
            'segmentation': '分割',
            'obb': 'OBB',
        }
        return names.get(task_type, task_type or '当前任务')

    @staticmethod
    def _unique_rule_id(base: str, used: set[str]) -> str:
        rule_id = ''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in base)
        rule_id = rule_id.strip('_') or 'python_custom_rule'
        candidate = rule_id
        suffix = 2
        while candidate in used:
            candidate = f'{rule_id}_{suffix}'
            suffix += 1
        return candidate

    def _set_status(self, text: str):
        if hasattr(self, 'status_label'):
            self.status_label.setText(text)
