"""Helpers for launching external annotation tools."""

from __future__ import annotations

import os
import shlex
import shutil
import sys
from pathlib import Path


LABEL_TOOL_LAUNCHER = Path(__file__).resolve().parents[1] / 'xanylabeling_launcher.py'
XANYLABELING_EXECUTABLE_ENV = 'VISION_PLATFORM_XANYLABELING'
XANYLABELING_PYTHON_ENV = 'VISION_PLATFORM_XANYLABELING_PYTHON'


def _configured_file(value: str) -> Path | None:
    """Resolve a configured path or command name to an existing file."""
    text = str(value or '').strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    found = shutil.which(text)
    return Path(found).resolve() if found else None


def _python_for_tool(tool: Path) -> Path | None:
    """Find the Python interpreter belonging to an X-AnyLabeling script."""
    current_python = Path(sys.executable).resolve()
    if tool.parent == current_python.parent:
        return current_python

    names = ('python.exe', 'python3.exe') if sys.platform == 'win32' else (
        'python', 'python3',
    )
    for name in names:
        candidate = tool.with_name(name)
        if candidate.is_file():
            return candidate.resolve()
    return None


def _qt_safe_command(python: Path) -> list[str]:
    return [str(python), str(LABEL_TOOL_LAUNCHER)]


def resolve_xanylabeling_program() -> list[str]:
    """Return a Qt-safe command without depending on a Conda env name."""
    configured_python_value = os.environ.get(XANYLABELING_PYTHON_ENV, '')
    configured_python = _configured_file(configured_python_value)
    if configured_python is not None:
        return _qt_safe_command(configured_python)
    if configured_python_value:
        raise FileNotFoundError(
            f'{XANYLABELING_PYTHON_ENV} 指向的文件不存在: '
            f'{configured_python_value}'
        )

    configured_tool_value = os.environ.get(XANYLABELING_EXECUTABLE_ENV, '')
    if configured_tool_value:
        configured_tool = _configured_file(configured_tool_value)
        if configured_tool is None:
            raise FileNotFoundError(
                f'{XANYLABELING_EXECUTABLE_ENV} 指向的文件不存在: '
                f'{configured_tool_value}'
            )
        configured_tool_python = _python_for_tool(configured_tool)
        if configured_tool_python is not None:
            return _qt_safe_command(configured_tool_python)
        return [str(configured_tool)]

    current_env_tool = Path(sys.executable).with_name('xanylabeling')
    if current_env_tool.is_file():
        return _qt_safe_command(Path(sys.executable).resolve())

    found = shutil.which('xanylabeling')
    if found:
        tool = Path(found).resolve()
        tool_python = _python_for_tool(tool)
        if tool_python is not None:
            return _qt_safe_command(tool_python)
        return [str(tool)]

    raise FileNotFoundError(
        '未找到 X-AnyLabeling。请将其安装到当前 Python 环境，或设置 '
        f'{XANYLABELING_PYTHON_ENV} / {XANYLABELING_EXECUTABLE_ENV}。'
    )


def build_xanylabeling_command(
    image_path: str | Path,
    annotation_path: str | Path,
    program: list[str] | None = None,
) -> list[str]:
    """Build a command that opens the image and its paired JSON label file."""
    image = Path(image_path).resolve()
    annotation = Path(annotation_path).resolve()
    prefix = list(program) if program is not None else resolve_xanylabeling_program()

    return prefix + [
        '--filename',
        str(image),
        '--output',
        str(annotation.parent),
        '--nodata',
        '--no-auto-update-check',
    ]


def build_xanylabeling_folder_command(
    image_dir: str | Path,
    annotation_dir: str | Path,
    program: list[str] | None = None,
) -> list[str]:
    """Open a whole image folder in X-AnyLabeling (one window, gallery).

    The ``--filename`` flag accepts a directory and loads every matching
    image; ``--output`` receives the annotation directory so JSONs stay
    next to their siblings (not the image folder).
    """
    image_dir_path = Path(image_dir).resolve()
    annotation_dir_path = Path(annotation_dir).resolve()
    prefix = list(program) if program is not None else resolve_xanylabeling_program()

    return prefix + [
        '--filename',
        str(image_dir_path),
        '--output',
        str(annotation_dir_path),
        '--nodata',
        '--no-auto-update-check',
    ]


def command_to_text(command: list[str]) -> str:
    """Format a command for logs and error messages."""
    return ' '.join(shlex.quote(part) for part in command)
