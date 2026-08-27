"""Validated training-job contract shared by the Qt UI and runner process."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.models.dataset_preparation import (
    DatasetPreparationError,
    ensure_training_dataset_yaml,
    inspect_training_batch,
)
from app.models.training_config import MANAGED_PARAMETER_NAMES


TRAIN_EVENT_PREFIX = '@@FILESPROCESS_TRAIN@@'
TASK_TO_ULTRALYTICS = {
    'pose': 'pose',
    'detection': 'detect',
    'segmentation': 'segment',
    'obb': 'obb',
}


class TrainingJobError(ValueError):
    """Raised when a training request is unsafe or incomplete."""


@dataclass(frozen=True)
class TrainingJob:
    job_id: str
    created_at: str
    task_type: str
    ultralytics_task: str
    model: str
    batch_root: str
    dataset_yaml: str
    output_root: str
    project_name: str
    run_name: str
    parameters: dict[str, Any]

    @property
    def project_dir(self) -> Path:
        return Path(self.output_root) / self.project_name

    @property
    def run_dir(self) -> Path:
        return self.project_dir / self.run_name


def create_training_job(*, task_type: str, model: str,
                        batch_root: str | Path,
                        output_root: str | Path,
                        project_name: str, run_name: str,
                        parameters: dict[str, Any]) -> TrainingJob:
    task_type = str(task_type or '').strip()
    if task_type not in TASK_TO_ULTRALYTICS:
        raise TrainingJobError(f'不支持的训练任务: {task_type}')

    batch = Path(batch_root).expanduser().resolve()
    if not batch.is_dir():
        raise TrainingJobError(f'训练批次不存在: {batch}')
    try:
        dataset_yaml = ensure_training_dataset_yaml(batch)
    except DatasetPreparationError as exc:
        raise TrainingJobError(str(exc)) from exc
    summary = inspect_training_batch(batch)
    if not summary.is_ready:
        raise TrainingJobError(summary.readiness_message())
    if summary.task_type != task_type:
        raise TrainingJobError(
            f'训练批次任务为 {summary.task_type}，当前配置任务为 {task_type}'
        )

    model_value = _normalize_model(model)
    output_value = str(output_root or '').strip()
    if not output_value:
        raise TrainingJobError('训练输出根目录不能为空')
    output = Path(output_value).expanduser()
    try:
        output = output.resolve()
    except OSError:
        output = output.absolute()
    _validate_directory_name(project_name, '项目分组')
    _validate_directory_name(run_name, '任务名称')

    if not isinstance(parameters, dict):
        raise TrainingJobError('训练参数必须是 JSON object')
    managed = sorted(set(parameters).intersection(MANAGED_PARAMETER_NAMES))
    if managed:
        raise TrainingJobError(
            '以下参数由平台管理，不能由模板覆盖: ' + ', '.join(managed)
        )
    try:
        normalized_parameters = json.loads(json.dumps(parameters))
    except (TypeError, ValueError) as exc:
        raise TrainingJobError(f'训练参数无法序列化: {exc}') from exc

    run_dir = output / project_name / run_name
    if normalized_parameters.get('resume'):
        checkpoint = Path(model_value).expanduser()
        expected_checkpoint = run_dir / 'weights' / 'last.pt'
        try:
            valid_checkpoint = (
                checkpoint.is_file()
                and checkpoint.resolve() == expected_checkpoint.resolve()
            )
        except OSError:
            valid_checkpoint = False
        if not valid_checkpoint:
            raise TrainingJobError(
                '断点续训必须使用当前任务目录中的 weights/last.pt；'
                '普通预训练权重不能恢复 epoch 和优化器状态'
            )
    allow_existing = bool(
        normalized_parameters.get('resume')
        or normalized_parameters.get('exist_ok')
    )
    if run_dir.exists() and not allow_existing:
        raise TrainingJobError(
            f'训练任务目录已存在，请修改任务名称: {run_dir}'
        )

    return TrainingJob(
        job_id=uuid.uuid4().hex,
        created_at=datetime.now(timezone.utc).isoformat(),
        task_type=task_type,
        ultralytics_task=TASK_TO_ULTRALYTICS[task_type],
        model=model_value,
        batch_root=str(batch),
        dataset_yaml=str(dataset_yaml.resolve()),
        output_root=str(output),
        project_name=project_name.strip(),
        run_name=run_name.strip(),
        parameters=normalized_parameters,
    )


def training_job_to_dict(job: TrainingJob) -> dict[str, Any]:
    return asdict(job)


def training_job_from_dict(payload: dict[str, Any]) -> TrainingJob:
    if not isinstance(payload, dict):
        raise TrainingJobError('训练任务文件根节点必须是 JSON object')
    required = {
        'job_id', 'created_at', 'task_type', 'ultralytics_task', 'model',
        'batch_root', 'dataset_yaml', 'output_root', 'project_name',
        'run_name', 'parameters',
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise TrainingJobError('训练任务缺少字段: ' + ', '.join(missing))
    task_type = str(payload['task_type'])
    if TASK_TO_ULTRALYTICS.get(task_type) != payload['ultralytics_task']:
        raise TrainingJobError('训练任务类型映射无效')
    parameters = payload['parameters']
    if not isinstance(parameters, dict):
        raise TrainingJobError('parameters 必须是 JSON object')
    return TrainingJob(
        job_id=str(payload['job_id']),
        created_at=str(payload['created_at']),
        task_type=task_type,
        ultralytics_task=str(payload['ultralytics_task']),
        model=str(payload['model']),
        batch_root=str(payload['batch_root']),
        dataset_yaml=str(payload['dataset_yaml']),
        output_root=str(payload['output_root']),
        project_name=str(payload['project_name']),
        run_name=str(payload['run_name']),
        parameters=dict(parameters),
    )


def write_training_job(job: TrainingJob, directory: str | Path,
                       filename: str = '') -> Path:
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / (filename or f'{job.job_id}.json')
    path.write_text(
        json.dumps(training_job_to_dict(job), ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    return path


def load_training_job(path: str | Path) -> TrainingJob:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingJobError(f'无法读取训练任务文件: {exc}') from exc
    return training_job_from_dict(payload)


def _normalize_model(value: str) -> str:
    model = str(value or '').strip()
    if not model:
        raise TrainingJobError('模型来源不能为空')
    path = Path(model).expanduser()
    is_explicit_path = path.is_absolute() or len(path.parts) > 1
    if is_explicit_path:
        if not path.is_file():
            raise TrainingJobError(f'模型文件不存在: {path}')
        return str(path.resolve())
    if path.is_file():
        return str(path.resolve())
    if path.suffix.lower() == '.pt':
        models_dir = Path(__file__).resolve().parents[2] / 'models'
        return str((models_dir / path.name).resolve())
    return model


def _validate_directory_name(value: str, field_name: str):
    name = str(value or '').strip()
    path = Path(name)
    if (
        not name
        or path.is_absolute()
        or len(path.parts) != 1
        or name in {'.', '..'}
    ):
        raise TrainingJobError(f'{field_name}必须是单个目录名称')
