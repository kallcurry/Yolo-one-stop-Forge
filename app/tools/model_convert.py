"""Model conversion tool: .pt -> ONNX via the project's model_trans script.

The original script uses hard-coded constants; this dialog renders a temporary
copy with the chosen parameters (the user's script is never modified) and runs
it in a subprocess so the UI stays responsive.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

from PyQt5.QtCore import QProcess, QSettings, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.utils import discover_available_models

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCRIPT = (
    '/home/lian-david/Project/ultralytics-main/model_trans.py'
)
SCRIPT_KEY = 'modelConvertScriptPath'

CONFIG_PATTERNS = (
    (r'^MODEL_PATH\s*=.*$', 'MODEL_PATH'),
    (r'^OUTPUT_ONNX\s*=.*$', 'OUTPUT_ONNX'),
    (r'^IMGSZ\s*=.*$', 'IMGSZ'),
    (r'^OPSET\s*=.*$', 'OPSET'),
    (r'^DYNAMIC\s*=.*$', 'DYNAMIC'),
    (r'^DEVICE\s*=.*$', 'DEVICE'),
)


FP16_BLOCK = """
# ---- fp16 转换（GPU 友好：权重+输入半精度，配合 onnxruntime-gpu / TensorRT）----
try:
    import onnx
    from onnxconverter_common import float16
    model = onnx.load(output_path)
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=False)
    onnx.save(model_fp16, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ fp16 转换完成: {output_path}  ({size_mb:.1f} MB)")
except Exception as exc:
    print(f"⚠️ fp16 转换失败（已保留 fp32 ONNX）: {exc}")
"""


def render_temp_script(template: str, model_path: str,
                       imgsz: int, opset: int, dynamic: bool,
                       device: str, fp16: bool = False) -> str:
    """Replace the configuration block in the template script."""
    lines = template.splitlines()
    replacements = {
        'MODEL_PATH': f'MODEL_PATH = r"{model_path}"',
        'OUTPUT_ONNX': 'OUTPUT_ONNX = None   # 与模型同目录同名 .onnx',
        'IMGSZ': f'IMGSZ = ({imgsz}, {imgsz})',
        'OPSET': f'OPSET = {opset}',
        'DYNAMIC': f'DYNAMIC = {dynamic}',
        'DEVICE': f'DEVICE = {device!r}',
    }
    replaced = set()
    output = []
    for line in lines:
        for pattern, key in CONFIG_PATTERNS:
            if re.match(pattern, line.strip()):
                output.append(replacements[key])
                replaced.add(key)
                break
        else:
            output.append(line)
    missing = {'MODEL_PATH', 'IMGSZ', 'OP SET'.replace(' ', ''), 'DEVICE'} - replaced
    missing = {'MODEL_PATH', 'IMGSZ', 'OPSET', 'DEVICE'} - replaced
    if missing:
        raise ValueError(f'脚本配置区缺少字段: {", ".join(sorted(missing))}')
    rendered = '\n'.join(output)
    if fp16:
        rendered += FP16_BLOCK
    return rendered


class ModelConvertDialog(QDialog):
    conversion_finished = pyqtSignal()

    def __init__(self, parent=None, default_model: str = ''):
        super().__init__(parent)
        self.setWindowTitle('模型转换（.pt → ONNX）')
        self.setMinimumSize(760, 560)
        self._process: QProcess | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel('模型 .pt'))
        self.combo_model = QComboBox()
        self.combo_model.setObjectName('trainingCombo')
        self.combo_model.setEditable(True)
        for path in discover_available_models():
            label = (
                path.parent.parent.name if path.parent.name == 'weights'
                else path.name
            )
            self.combo_model.addItem(label, str(path))
        if default_model:
            self.combo_model.setEditText(default_model)
        model_row.addWidget(self.combo_model, 1)
        btn_browse = QPushButton('浏览')
        btn_browse.setObjectName('fileOpBtn')
        btn_browse.clicked.connect(self._browse_model)
        model_row.addWidget(btn_browse)
        layout.addLayout(model_row)

        script_row = QHBoxLayout()
        script_row.addWidget(QLabel('转换脚本'))
        self.edit_script = QLineEdit(
            str(QSettings().value(SCRIPT_KEY, DEFAULT_SCRIPT) or DEFAULT_SCRIPT)
        )
        self.edit_script.setObjectName('trainingEdit')
        script_row.addWidget(self.edit_script, 1)
        btn_script = QPushButton('浏览')
        btn_script.setObjectName('fileOpBtn')
        btn_script.clicked.connect(self._browse_script)
        script_row.addWidget(btn_script)
        layout.addLayout(script_row)

        params_row = QHBoxLayout()
        params_row.addWidget(QLabel('imgsz'))
        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setObjectName('trainingSpin')
        self.spin_imgsz.setRange(160, 2560)
        self.spin_imgsz.setSingleStep(32)
        self.spin_imgsz.setValue(640)
        params_row.addWidget(self.spin_imgsz)
        params_row.addWidget(QLabel('opset'))
        self.spin_opset = QSpinBox()
        self.spin_opset.setObjectName('trainingSpin')
        self.spin_opset.setRange(11, 21)
        self.spin_opset.setValue(18)
        params_row.addWidget(self.spin_opset)
        params_row.addWidget(QLabel('device'))
        self.combo_device = QComboBox()
        self.combo_device.setObjectName('trainingCombo')
        self.combo_device.addItem('GPU（0）', 0)
        self.combo_device.addItem('CPU（cpu）', 'cpu')
        params_row.addWidget(self.combo_device)
        self.check_dynamic = __import__(
            'PyQt5.QtWidgets', fromlist=['QCheckBox']
        ).QCheckBox('动态 batch')
        self.check_dynamic.setObjectName('trainingCheck')
        params_row.addWidget(self.check_dynamic)
        self.check_fp16 = __import__(
            'PyQt5.QtWidgets', fromlist=['QCheckBox']
        ).QCheckBox('fp16 ONNX（GPU 友好）')
        self.check_fp16.setObjectName('trainingCheck')
        self.check_fp16.setToolTip(
            '导出后自动转半精度：配合 onnxruntime-gpu / TensorRT 使用时速度更快、'
            '显存更低；转换失败会保留 fp32 版本。'
        )
        params_row.addWidget(self.check_fp16)
        params_row.addStretch()
        layout.addLayout(params_row)

        btn_row = QHBoxLayout()
        btn_run = QPushButton('开始转换')
        btn_run.setObjectName('primaryBtn')
        btn_run.clicked.connect(self._run)
        btn_row.addWidget(btn_run)
        btn_open = QPushButton('打开输出目录')
        btn_open.setObjectName('fileOpBtn')
        btn_open.clicked.connect(self._open_output_dir)
        btn_row.addWidget(btn_open)
        btn_row.addStretch()
        self.status = QLabel('输出默认保存在 .pt 同目录（同名 .onnx）')
        self.status.setObjectName('duplicateScope')
        btn_row.addWidget(self.status)
        layout.addLayout(btn_row)

        self.log = QPlainTextEdit()
        self.log.setObjectName('trainingLog')
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)

    def _browse_model(self):
        path, _f = QFileDialog.getOpenFileName(
            self, '选择模型权重', str(PROJECT_ROOT),
            '模型 (*.pt);;所有文件 (*)',
        )
        if path:
            self.combo_model.setEditText(path)

    def _browse_script(self):
        path, _f = QFileDialog.getOpenFileName(
            self, '选择转换脚本', str(Path(self.edit_script.text()).parent),
            'Python (*.py);;所有文件 (*)',
        )
        if path:
            self.edit_script.setText(path)

    def _run(self):
        model_text = self.combo_model.currentText().strip()
        if not model_text:
            QMessageBox.warning(self, '无法转换', '请选择模型 .pt')
            return
        model_path = Path(model_text).expanduser()
        if not model_path.is_file():
            QMessageBox.warning(self, '无法转换', f'模型文件不存在: {model_path}')
            return
        script_path = Path(self.edit_script.text().strip()).expanduser()
        if not script_path.is_file():
            QMessageBox.warning(self, '无法转换', f'转换脚本不存在: {script_path}')
            return
        try:
            template = script_path.read_text(encoding='utf-8')
            rendered = render_temp_script(
                template, str(model_path), int(self.spin_imgsz.value()),
                int(self.spin_opset.value()),
                self.check_dynamic.isChecked(),
                self.combo_device.currentData(),
                self.check_fp16.isChecked(),
            )
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, '无法转换', str(exc))
            return
        QSettings().setValue(SCRIPT_KEY, str(script_path))
        # 依赖检查：torch.onnx.export 需要 onnx 包
        try:
            import onnx  # noqa: F401
        except ImportError:
            answer = QMessageBox.question(
                self, '缺少依赖',
                '模型转换需要 onnx 包，当前环境未安装。\n'
                '是否现在自动安装？（pip install onnx）',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
            )
            if answer != QMessageBox.Yes:
                return
            self.status.setText('正在安装 onnx 依赖...')
            self.log.appendPlainText('▶ pip install onnx ...')
            import subprocess
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', 'onnx'],
                capture_output=True, text=True,
            )
            self.log.appendPlainText(result.stdout[-2000:] or result.stderr[-2000:])
            if result.returncode != 0:
                QMessageBox.warning(
                    self, '安装失败',
                    f'onnx 安装失败，请手动执行: pip install onnx\n{result.stderr[-300:]}',
                )
                return
        if self.check_fp16.isChecked():
            try:
                import onnxconverter_common  # noqa: F401
            except ImportError:
                self.status.setText('正在安装 onnxconverter-common ...')
                import subprocess
                result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'install',
                     'onnxconverter-common'],
                    capture_output=True, text=True,
                )
                if result.returncode != 0:
                    QMessageBox.warning(
                        self, '安装失败',
                        'onnxconverter-common 安装失败，请手动执行: '
                        f'pip install onnxconverter-common\n{result.stderr[-300:]}',
                    )
                    return
        temp_dir = PROJECT_ROOT / '.runtime'
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_script = temp_dir / f'model_trans_{int(time.time())}.py'
        temp_script.write_text(rendered, encoding='utf-8')

        self.log.clear()
        self.log.appendPlainText(f'▶ 使用脚本: {script_path}')
        self.log.appendPlainText(f'▶ 模型: {model_path}')
        process = QProcess(self)
        process.readyReadStandardOutput.connect(self._read_output)
        process.readyReadStandardError.connect(self._read_error)
        process.finished.connect(
            lambda code, _s: self._on_finished(code, temp_script)
        )
        env = __import__(
            'PyQt5.QtCore', fromlist=['QProcessEnvironment']
        ).QProcessEnvironment.systemEnvironment()
        env.insert('PYTHONPATH', str(PROJECT_ROOT) + ':' + env.value('PYTHONPATH', ''))
        process.setProcessEnvironment(env)
        process.start(sys.executable, [str(temp_script)])
        self._process = process
        self.status.setText('转换中...')

    def _read_output(self):
        if self._process:
            self.log.appendPlainText(
                bytes(self._process.readAllStandardOutput()).decode(
                    'utf-8', 'replace'
                ).rstrip()
            )

    def _read_error(self):
        if self._process:
            self.log.appendPlainText(
                bytes(self._process.readAllStandardError()).decode(
                    'utf-8', 'replace'
                ).rstrip()
            )

    def _on_finished(self, exit_code: int, temp_script: Path):
        try:
            temp_script.unlink(missing_ok=True)
        except OSError:
            pass
        if exit_code == 0:
            self.status.setText('✅ 转换完成，输出保存在 .pt 同目录')
            self.conversion_finished.emit()
        else:
            self.status.setText(f'❌ 转换失败（退出码 {exit_code}），详见日志')
        self._process = None

    def _open_output_dir(self):
        model_text = self.combo_model.currentText().strip()
        if not model_text:
            return
        target = Path(model_text).expanduser().parent
        if target.is_dir():
            import subprocess
            subprocess.Popen(['xdg-open', str(target)])

    def closeEvent(self, event):
        if self._process is not None and self._process.state() != QProcess.NotRunning:
            self._process.terminate()
            self._process.waitForFinished(2000)
        event.accept()


def create_dialog(parent=None, default_model: str = ''):
    return ModelConvertDialog(parent, default_model=default_model)
