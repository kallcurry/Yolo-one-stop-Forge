#!/usr/bin/env python3
"""Validate a source deployment without depending on an environment name."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Result:
    level: str
    name: str
    detail: str


class Doctor:
    def __init__(self):
        self.results: list[Result] = []

    def ok(self, name: str, detail: str):
        self.results.append(Result('OK', name, detail))

    def warn(self, name: str, detail: str):
        self.results.append(Result('WARN', name, detail))

    def error(self, name: str, detail: str):
        self.results.append(Result('ERROR', name, detail))

    def package(self, distribution: str, module: str | None = None):
        module = module or distribution.replace('-', '_')
        try:
            package_version = version(distribution)
        except PackageNotFoundError:
            self.error(distribution, 'not installed')
            return
        if importlib.util.find_spec(module) is None:
            self.error(distribution, f'{package_version}, module {module!r} not importable')
            return
        self.ok(distribution, package_version)

    def summary(self) -> int:
        widths = max((len(item.level) for item in self.results), default=5)
        for item in self.results:
            print(f'[{item.level:<{widths}}] {item.name}: {item.detail}')
        errors = sum(item.level == 'ERROR' for item in self.results)
        warnings = sum(item.level == 'WARN' for item in self.results)
        print(f'\nSummary: {errors} error(s), {warnings} warning(s)')
        return 1 if errors else 0


def _check_python(doctor: Doctor):
    current = sys.version_info[:2]
    detail = f'{sys.version.split()[0]} ({sys.executable})'
    if (3, 10) <= current < (3, 12):
        doctor.ok('Python', detail)
    else:
        doctor.error('Python', detail + '; supported: 3.10-3.11, tested: 3.10')


def _check_qt_plugins(doctor: Doctor):
    try:
        from PyQt5.QtCore import QLibraryInfo
        plugins = Path(QLibraryInfo.location(QLibraryInfo.PluginsPath))
    except Exception as exc:
        doctor.error('Qt plugins', str(exc))
        return
    platform_dir = plugins / 'platforms'
    candidates = ('libqxcb.so', 'qwindows.dll', 'libqcocoa.dylib')
    if plugins.is_dir() and any((platform_dir / name).is_file() for name in candidates):
        doctor.ok('Qt plugins', str(plugins))
    else:
        doctor.error('Qt plugins', f'platform plugin not found under {plugins}')


def _check_opencv_pair(doctor: Doctor):
    installed = {}
    for package in ('opencv-python', 'opencv-contrib-python-headless'):
        try:
            installed[package] = version(package)
        except PackageNotFoundError:
            continue
    if len(installed) < 2:
        doctor.warn('OpenCV pairing', f'installed distributions: {installed or "none"}')
        return
    versions = {value.split('.')[:3][0] + '.' + value.split('.')[:3][1]
                for value in installed.values()}
    if len(versions) == 1:
        doctor.ok('OpenCV pairing', ', '.join(f'{k}={v}' for k, v in installed.items()))
    else:
        doctor.error('OpenCV pairing', 'version families differ: ' + str(installed))


def _check_xanylabeling(doctor: Doctor, required: bool):
    python_override = os.environ.get('VISION_PLATFORM_XANYLABELING_PYTHON', '').strip()
    tool_override = os.environ.get('VISION_PLATFORM_XANYLABELING', '').strip()
    target_python = Path(python_override).expanduser() if python_override else None

    if target_python is not None:
        if not target_python.is_file():
            doctor.error('X-AnyLabeling', f'configured Python does not exist: {target_python}')
            return
        check = subprocess.run(
            [str(target_python), '-c', 'import anylabeling, cv2, PyQt5'],
            capture_output=True, text=True, timeout=30,
        )
        if check.returncode == 0:
            doctor.ok('X-AnyLabeling', f'external interpreter: {target_python}')
        else:
            doctor.error('X-AnyLabeling', check.stderr.strip() or 'external import failed')
        return

    if tool_override:
        resolved = Path(tool_override).expanduser()
        if resolved.is_file() or shutil.which(tool_override):
            doctor.ok('X-AnyLabeling', f'external executable: {tool_override}')
        else:
            doctor.error('X-AnyLabeling', f'configured executable not found: {tool_override}')
        return

    current_tool = Path(sys.executable).with_name('xanylabeling')
    available = importlib.util.find_spec('anylabeling') is not None
    if available and (current_tool.is_file() or shutil.which('xanylabeling')):
        try:
            tool_version = version('x-anylabeling-cvhub')
        except PackageNotFoundError:
            tool_version = 'installed'
        doctor.ok('X-AnyLabeling', f'{tool_version}, current environment')
    elif required:
        doctor.error('X-AnyLabeling', 'not installed or executable missing')
    else:
        doctor.warn('X-AnyLabeling', 'optional component is not configured')


def _check_torch(doctor: Doctor, require_cuda: bool):
    try:
        import torch
    except Exception as exc:
        doctor.error('PyTorch', str(exc))
        return
    detail = f'{torch.__version__}; CUDA available={torch.cuda.is_available()}'
    if require_cuda and not torch.cuda.is_available():
        doctor.error('CUDA', detail)
    elif torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        doctor.ok('PyTorch/CUDA', f'{detail}; device={device}')
    else:
        doctor.warn('PyTorch/CUDA', detail + '; CPU training remains available')


def _check_writable_workspace(doctor: Doctor):
    for relative in ('.runtime', 'models', 'training'):
        directory = PROJECT_ROOT / relative
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=directory, prefix='.write-test-'):
                pass
        except OSError as exc:
            doctor.error(f'Writable {relative}', str(exc))
        else:
            doctor.ok(f'Writable {relative}', str(directory))


def _check_pip(doctor: Doctor):
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'check'],
        capture_output=True, text=True, timeout=60,
    )
    output = (result.stdout or result.stderr).strip()
    if result.returncode == 0:
        doctor.ok('Dependency consistency', output or 'pip check passed')
    else:
        doctor.error('Dependency consistency', output or 'pip check failed')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-label-tool', action='store_true')
    parser.add_argument('--require-cuda', action='store_true')
    args = parser.parse_args()

    doctor = Doctor()
    _check_python(doctor)
    for distribution, module in (
        ('numpy', 'numpy'),
        ('PyQt5', 'PyQt5'),
        ('Pillow', 'PIL'),
        ('Send2Trash', 'send2trash'),
        ('PyYAML', 'yaml'),
        ('psutil', 'psutil'),
        ('ultralytics', 'ultralytics'),
        ('opencv-python', 'cv2'),
    ):
        doctor.package(distribution, module)
    _check_qt_plugins(doctor)
    _check_opencv_pair(doctor)
    _check_xanylabeling(doctor, args.require_label_tool)
    _check_torch(doctor, args.require_cuda)
    _check_writable_workspace(doctor)
    _check_pip(doctor)
    return doctor.summary()


if __name__ == '__main__':
    raise SystemExit(main())
