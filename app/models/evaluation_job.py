"""Evaluation task contract (evaluation center, phase 1)."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

EVAL_EVENT_PREFIX = '@@FILESPROCESS_EVAL@@'

# Ultralytics results_dict metric name families per task
METRIC_FAMILIES = {
    'detection': '(B)',
    'obb': '(B)',
    'segmentation': '(M)',
    'pose': '(P)',
}


@dataclass(frozen=True)
class EvaluationJob:
    """Immutable evaluation task contract snapshot."""

    job_id: str
    model_path: str
    model_label: str
    task_type: str
    training_run_dir: str | None
    training_batch: str | None
    test_batch: str
    test_data_root: str
    test_dataset_yaml: str
    test_manifest_path: str
    test_manifest_sha256: str
    project_dir: str
    run_name: str
    parameters: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec='seconds'))

    @property
    def task_dir(self) -> Path:
        return Path(self.project_dir).parent / 'tasks' / self.job_id

    @property
    def output_dir(self) -> Path:
        return Path(self.project_dir) / self.run_name

    def validate(self) -> list[str]:
        problems = []
        model = Path(self.model_path)
        if not model.is_file():
            problems.append(f'模型文件不存在: {model}')
        yaml_path = Path(self.test_dataset_yaml)
        if not yaml_path.is_file():
            problems.append(f'测试数据集配置不存在: {yaml_path}')
        if not Path(self.test_data_root).is_dir():
            problems.append(f'测试数据目录不存在: {self.test_data_root}')
        if self.run_name.strip() != str(self.run_name).strip() or '/' in self.run_name:
            problems.append(f'运行名称无效: {self.run_name!r}')
        if self.task_type not in METRIC_FAMILIES:
            problems.append(f'未知任务类型: {self.task_type}')
        return problems

    def to_dict(self) -> dict:
        return asdict(self)


def build_evaluation_job(
    *,
    model_path: str | Path,
    model_label: str,
    test_data_root: str | Path,
    test_batch: str,
    task_type: str,
    project_dir: str | Path,
    run_name: str,
    training_run_dir: str | Path | None = None,
    training_batch: str | None = None,
    parameters: dict | None = None,
    test_manifest_path: str | Path | None = None,
    test_manifest_sha256: str = '',
) -> EvaluationJob:
    root = Path(test_data_root).expanduser().resolve()
    yaml_path = root / 'dataset.yaml'
    manifest_path = (
        Path(test_manifest_path).expanduser().resolve()
        if test_manifest_path else root / 'test_manifest.json'
    )
    job = EvaluationJob(
        job_id=f'eval-{uuid.uuid4().hex[:10]}',
        model_path=str(Path(model_path).expanduser().resolve()),
        model_label=str(model_label),
        task_type=str(task_type),
        training_run_dir=(
            str(Path(training_run_dir).expanduser().resolve())
            if training_run_dir else None
        ),
        training_batch=training_batch,
        test_batch=str(test_batch),
        test_data_root=str(root),
        test_dataset_yaml=str(yaml_path),
        test_manifest_path=str(manifest_path),
        test_manifest_sha256=str(test_manifest_sha256),
        project_dir=str(Path(project_dir).expanduser().resolve()),
        run_name=str(run_name),
        parameters=dict(parameters or {}),
    )
    return job


def save_evaluation_job(job: EvaluationJob, path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(job.to_dict(), ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    return target


def load_evaluation_job(path: str | Path) -> EvaluationJob:
    data = json.loads(Path(path).expanduser().read_text(encoding='utf-8'))
    return EvaluationJob(**data)
