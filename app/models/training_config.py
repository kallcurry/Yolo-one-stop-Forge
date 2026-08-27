"""Task-specific Ultralytics training template loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


TRAINING_TASK_TYPES = ('pose', 'detection', 'segmentation', 'obb')
RUNTIME_PARAMETER_NAMES = {'resume'}
MANAGED_PARAMETER_NAMES = {
    'data', 'project', 'name', 'task', 'mode', 'model',
    'classes', 'single_cls',
}
BOOLEAN_PARAMETERS = {
    'save', 'pretrained', 'verbose', 'deterministic', 'rect',
    'cos_lr', 'amp', 'profile', 'plots', 'val', 'save_json',
    'overlap_mask', 'exist_ok',
}
PROBABILITY_PARAMETERS = {
    'fraction', 'hsv_h', 'hsv_s', 'hsv_v', 'translate', 'perspective',
    'flipud', 'fliplr', 'bgr', 'mosaic', 'mixup', 'cutmix', 'copy_paste',
    'erasing',
}
NONNEGATIVE_PARAMETERS = {
    'time', 'patience', 'workers', 'seed', 'close_mosaic', 'multi_scale',
    'lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs',
    'warmup_momentum', 'warmup_bias_lr', 'box', 'cls', 'dfl', 'pose',
    'kobj', 'nbs', 'degrees', 'scale', 'shear', 'dropout', 'iou',
    'max_det', 'mask_ratio',
}
POSITIVE_INTEGER_PARAMETERS = {'epochs', 'imgsz'}


@dataclass(frozen=True)
class TrainingConfig:
    name: str
    task_type: str
    model: str
    parameters: dict[str, Any]
    description: str = ''
    version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def builtin_training_template_path(task_type: str) -> Path:
    task_type = _validate_task_type(task_type)
    return project_root() / 'resources' / 'training_templates' / (
        f'{task_type}_training_template.json'
    )


def custom_training_template_dir(task_type: str) -> Path:
    task_type = _validate_task_type(task_type)
    return project_root() / 'resources' / 'training_templates' / task_type


def default_training_config(task_type: str) -> TrainingConfig:
    path = builtin_training_template_path(task_type)
    return load_training_config(path)


def load_training_config(path: str | Path) -> TrainingConfig:
    config_path = Path(path).expanduser()
    try:
        data = json.loads(config_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ValueError(f'训练模板不是有效 JSON: {exc}') from exc
    except OSError as exc:
        raise ValueError(f'无法读取训练模板: {exc}') from exc
    return training_config_from_dict(data, config_path)


def training_config_from_dict(data: dict,
                              path: str | Path | None = None) -> TrainingConfig:
    if not isinstance(data, dict):
        raise ValueError('训练模板根节点必须是 JSON object')
    source_path = Path(path) if path is not None else None
    name = str(
        data.get('name')
        or (source_path.stem if source_path is not None else '自定义训练模板')
    ).strip()
    if not name:
        raise ValueError('name 不能为空')
    task_type = _validate_task_type(data.get('task_type', 'pose'))
    model = str(data.get('model') or '').strip()
    if not model:
        raise ValueError('model 不能为空，应为 .pt 或模型 .yaml 路径')
    parameters = data.get('parameters')
    if not isinstance(parameters, dict):
        raise ValueError('parameters 必须是 JSON object')
    parameters = {
        key: value for key, value in parameters.items()
        if str(key).strip() not in RUNTIME_PARAMETER_NAMES
    }
    parameters = _validate_parameters(parameters)
    version = data.get('version', 1)
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError('version 必须是大于等于 1 的整数')
    description = str(data.get('description') or '').strip()
    known = {'name', 'version', 'description', 'task_type', 'model', 'parameters'}
    metadata = {key: value for key, value in data.items() if key not in known}
    _validate_json_value(metadata, '模板元数据')
    return TrainingConfig(
        name=name,
        task_type=task_type,
        model=model,
        parameters=parameters,
        description=description,
        version=version,
        metadata=metadata,
        path=source_path,
    )


def training_config_to_dict(config: TrainingConfig) -> dict:
    data = {
        'name': config.name,
        'version': config.version,
        'description': config.description,
        'task_type': config.task_type,
        'model': config.model,
        'parameters': dict(config.parameters),
    }
    for key, value in config.metadata.items():
        if key not in data:
            data[key] = value
    return data


def list_training_template_paths(task_type: str,
                                 extra_paths=()) -> list[Path]:
    task_type = _validate_task_type(task_type)
    paths = [builtin_training_template_path(task_type)]
    custom_dir = custom_training_template_dir(task_type)
    if custom_dir.is_dir():
        paths.extend(sorted(custom_dir.glob('*.json')))
    paths.extend(Path(path).expanduser() for path in extra_paths if str(path).strip())
    unique = []
    seen = set()
    for path in paths:
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key in seen or not path.is_file():
            continue
        try:
            config = load_training_config(path)
        except ValueError:
            continue
        if config.task_type != task_type:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _validate_task_type(value: Any) -> str:
    task_type = str(value or '').strip()
    if task_type not in TRAINING_TASK_TYPES:
        raise ValueError(
            f'task_type 必须是 {", ".join(TRAINING_TASK_TYPES)} 之一'
        )
    return task_type


def _validate_parameters(parameters: dict) -> dict[str, Any]:
    validated = {}
    for raw_name, value in parameters.items():
        name = str(raw_name or '').strip()
        if not name or any(char.isspace() for char in name):
            raise ValueError(f'参数名称无效: {raw_name!r}')
        if name in MANAGED_PARAMETER_NAMES:
            raise ValueError(f'参数 {name} 由平台管理，不能写入 parameters')
        _validate_json_value(value, f'参数 {name}')
        if name in BOOLEAN_PARAMETERS and not isinstance(value, bool):
            raise ValueError(f'参数 {name} 必须是 true 或 false')
        if name in POSITIVE_INTEGER_PARAMETERS:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'参数 {name} 必须是正整数')
        if name in NONNEGATIVE_PARAMETERS and value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f'参数 {name} 必须是数字')
            if value < 0:
                raise ValueError(f'参数 {name} 不能小于 0')
        if name in PROBABILITY_PARAMETERS:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f'参数 {name} 必须是 0 到 1 之间的数字')
            if not 0 <= float(value) <= 1:
                raise ValueError(f'参数 {name} 必须位于 0 到 1 之间')
        if name == 'batch':
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError('参数 batch 必须是整数、比例或 -1')
            if value == 0 or value < -1:
                raise ValueError('参数 batch 不能为 0 或小于 -1')
        if name == 'optimizer' and not isinstance(value, str):
            raise ValueError('参数 optimizer 必须是字符串')
        if name == 'cache' and not isinstance(value, (bool, str)):
            raise ValueError('参数 cache 必须是布尔值、ram 或 disk')
        if name == 'freeze' and value is not None:
            valid_freeze = (
                isinstance(value, int) and not isinstance(value, bool)
            ) or (
                isinstance(value, list)
                and all(isinstance(item, int) and not isinstance(item, bool)
                        for item in value)
            )
            if not valid_freeze:
                raise ValueError('参数 freeze 必须是整数、整数数组或 null')
        validated[name] = value
    return validated


def _validate_json_value(value: Any, field_name: str):
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f'{field_name}[{index}]')
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f'{field_name} 的 object key 必须是字符串')
            _validate_json_value(item, f'{field_name}.{key}')
        return
    raise ValueError(f'{field_name} 不是可序列化的 JSON 值')
