"""Thin read-only facade over the active review template for tools.

Tools (dataset stats, convert/validate, ...) must follow the platform-wide
task instead of hard coding 'pose'.  This module reads the active template
configuration, which is the single source of truth after task switching.
"""

from __future__ import annotations

from pathlib import Path

from app.models.annotation_review import current_pose_review_config

_TASK_LABELS = {
    'pose': 'POSE',
    'detection': 'DETECTION',
    'segment': 'SEGMENTATION',
    'segmentation': 'SEGMENTATION',
    'obb': 'OBB',
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def current_task_type() -> str:
    return str(current_pose_review_config().task_type or 'pose')


def current_annotation_dir() -> str:
    return str(current_pose_review_config().annotation_dir or 'annotations')


def task_label(task_type: str) -> str:
    return _TASK_LABELS.get(str(task_type), str(task_type).upper())


def current_task_label() -> str:
    return task_label(current_task_type())


def default_label_config_for_task(task_type: str | None = None) -> str | None:
    """Best-effort X-AnyLabeling config file for a task (used as a default).

    Candidates: the project X-AnyLabeling checkout, project configs/ and
    resources/ directories — any YAML matching the task keyword.
    """
    task = str(task_type or current_task_type()).lower()
    keywords = {
        'pose': ('pose',), 'detection': ('detect',), 'segment': ('seg',),
        'segmentation': ('seg',), 'obb': ('obb', 'rotation'),
    }.get(task, (task,))
    search_dirs = [
        Path('/home/lian-david/Project/X-AnyLabeling-main/X-AnyLabeling'
             '/anylabeling/configs/auto_labeling'),
        PROJECT_ROOT / 'configs',
        PROJECT_ROOT / 'resources',
    ]
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        try:
            matches = sorted(
                candidate for candidate in directory.iterdir()
                if candidate.is_file() and candidate.suffix.lower() == '.yaml'
                and any(keyword in candidate.name.lower() for keyword in keywords)
            )
        except OSError:
            continue
        if matches:
            return str(matches[0])
    return None
