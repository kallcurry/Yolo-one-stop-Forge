import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.models.label_tool import (
    LABEL_TOOL_LAUNCHER,
    XANYLABELING_EXECUTABLE_ENV,
    XANYLABELING_PYTHON_ENV,
    build_xanylabeling_command,
    command_to_text,
    resolve_xanylabeling_program,
)


class LabelToolCommandTest(unittest.TestCase):
    def test_build_command_uses_image_and_annotation_directory(self):
        command = build_xanylabeling_command(
            '/data/images/frame_001.png',
            '/data/annotations/frame_001.json',
            program=['xanylabeling'],
        )

        self.assertEqual(command, [
            'xanylabeling',
            '--filename',
            '/data/images/frame_001.png',
            '--output',
            '/data/annotations',
            '--nodata',
            '--no-auto-update-check',
        ])

    def test_command_text_quotes_paths(self):
        text = command_to_text(['xanylabeling', '--filename', '/data/a b.png'])

        self.assertEqual(text, "xanylabeling --filename '/data/a b.png'")

    def test_current_environment_uses_qt_safe_launcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp) / 'bin'
            bin_dir.mkdir()
            python = bin_dir / 'python'
            tool = bin_dir / 'xanylabeling'
            python.touch()
            tool.touch()
            with patch.dict(os.environ, {}, clear=True), patch(
                'app.models.label_tool.sys.executable', str(python)
            ):
                self.assertEqual(
                    resolve_xanylabeling_program(),
                    [str(python.resolve()), str(LABEL_TOOL_LAUNCHER)],
                )

    def test_configured_python_does_not_depend_on_environment_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            python = Path(tmp) / 'custom-env' / 'bin' / 'python'
            python.parent.mkdir(parents=True)
            python.touch()
            with patch.dict(os.environ, {
                XANYLABELING_PYTHON_ENV: str(python),
            }, clear=True):
                self.assertEqual(
                    resolve_xanylabeling_program(),
                    [str(python.resolve()), str(LABEL_TOOL_LAUNCHER)],
                )

    def test_invalid_configured_executable_is_reported(self):
        with patch.dict(os.environ, {
            XANYLABELING_EXECUTABLE_ENV: '/missing/xanylabeling',
        }, clear=True):
            with self.assertRaisesRegex(FileNotFoundError, '指向的文件不存在'):
                resolve_xanylabeling_program()

    def test_invalid_configured_python_is_reported(self):
        with patch.dict(os.environ, {
            XANYLABELING_PYTHON_ENV: '/missing/python',
        }, clear=True):
            with self.assertRaisesRegex(FileNotFoundError, '指向的文件不存在'):
                resolve_xanylabeling_program()


if __name__ == '__main__':
    unittest.main()
