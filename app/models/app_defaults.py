"""Central application defaults.

Single source of truth for tunable defaults (inference / evaluation /
preparation / image extensions / task directories).

Load order (deep merge, later wins):

1. built-in fallback constants (never fail)
2. ``resources/app_config.example.json`` (shipped template)
3. ``resources/app_config.json`` (user editable, git-ignored)

A corrupt ``app_config.json`` falls back to the example/built-ins with a
logged warning instead of breaking startup.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESOURCES_DIR = PROJECT_ROOT / 'resources'

CONFIG_NAME = 'app_config.json'
EXAMPLE_NAME = 'app_config.example.json'

# --------------------------------------------------------------------------
# Built-in fallback constants (identical to what the previous hard-coded
# defaults were).
# --------------------------------------------------------------------------

_BUILT_IN: dict = {
    'inference': {
        'conf': 0.25,
        'iou': 0.6,
        'imgsz': 640,
        'device': 'auto',
        'half': False,
        'max_fps': 30,
        'camera_width': 1280,
        'camera_height': 720,
        'record_fps': 25,
        'record_codec': 'mp4v',
    },
    'evaluation': {
        'imgsz': 640,
        'batch': 16,
        'conf': 0.001,
        'iou': 0.6,
    },
    'preparation': {
        'val_ratio': 0.2,
        'test_ratio': 0.0,
        'seed': 42,
        'use_copy': False,
    },
    'extensions': {
        'images': [
            '.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp',
        ],
    },
    'task_dirs': {
        'pose': {'annotation_dir': 'annotations', 'label_dir': 'labels'},
        'detection': {
            'annotation_dir': 'annotations-det', 'label_dir': 'labels-det',
        },
        'segmentation': {
            'annotation_dir': 'annotations-seg', 'label_dir': 'labels-seg',
        },
        'obb': {
            'annotation_dir': 'annotations-obb', 'label_dir': 'labels-obb',
        },
    },
}

_logger = logging.getLogger(__name__)

# Cache key prefixes used by qsettings consumers (single place).
QSETTINGS_ORG = 'FilesProcessQT'
QSETTINGS_APP = 'ImageManager'
KEY_TASK_TYPE = 'lastTaskType'

_config_cache: dict | None = None


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_json(path: Path) -> dict | None:
    try:
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(payload, dict):
            return payload
        _logger.warning('app_config: %s 不是 JSON 对象，已忽略', path)
        return None
    except (OSError, ValueError) as exc:
        _logger.warning('app_config: %s 解析失败（%s），已回退默认', path, exc)
        return None


def load_app_config(config_path: Path | None = None,
                    example_path: Path | None = None) -> dict:
    """Load the merged configuration (built-ins <- example <- config)."""
    config = dict(_BUILT_IN)
    example = _load_json(example_path or (RESOURCES_DIR / EXAMPLE_NAME))
    if example:
        config = _deep_merge(config, example)
    custom = _load_json(config_path or (RESOURCES_DIR / CONFIG_NAME))
    if custom:
        config = _deep_merge(config, custom)
    return config


def get_config() -> dict:
    global _config_cache
    if _config_cache is None:
        _config_cache = load_app_config()
    return _config_cache


def reset_config_cache():
    global _config_cache
    _config_cache = None


# --------------------------------------------------------------------------
# Typed accessors
# --------------------------------------------------------------------------

def _section(section: str) -> dict:
    value = get_config().get(section, {})
    return value if isinstance(value, dict) else {}


def inference_default(key: str, fallback=None):
    return _section('inference').get(key, fallback)


def evaluation_default(key: str, fallback=None):
    return _section('evaluation').get(key, fallback)


def preparation_default(key: str, fallback=None):
    return _section('preparation').get(key, fallback)


def clamp_number(value, low: float | int, high: float | int,
                 fallback, integer: bool = False):
    """Validate + clamp a config value into a spin range."""
    try:
        if integer:
            number = int(value)
        else:
            number = float(value)
    except (TypeError, ValueError):
        return fallback
    number = max(low, min(high, number))
    return int(number) if integer else number


def image_extensions() -> set[str]:
    """Unified image extension set (lower-case, with leading dot)."""
    extensions = []
    for item in _section('extensions').get('images', []):
        value = str(item).strip().lower()
        if value and not value.startswith('.'):
            value = '.' + value
        if value:
            extensions.append(value)
    return set(extensions)


def task_dir(task_type: str, kind: str, fallback: str) -> str:
    """Resolve annotation/label dir for a task (config overridable)."""
    section = _section('task_dirs')
    preset = section.get(str(task_type), {})
    if isinstance(preset, dict):
        value = preset.get(kind)
        if value:
            return str(value)
    return fallback
