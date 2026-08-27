"""Read-only discovery of Ultralytics training runs and their datasets."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml


MODEL_FORMATS = {
    '.pt': ('PyTorch', 'PT'),
    '.pth': ('PyTorch', 'PTH'),
    '.onnx': ('ONNX Runtime', 'ONNX'),
    '.engine': ('TensorRT', 'ENGINE'),
    '.plan': ('TensorRT', 'PLAN'),
    '.xml': ('OpenVINO', 'XML'),
    '.bin': ('OpenVINO', 'BIN'),
    '.tflite': ('TensorFlow Lite', 'TFLITE'),
    '.torchscript': ('TorchScript', 'TORCHSCRIPT'),
}

TASK_ANNOTATION_DIRS = {
    'pose': 'annotations',
    'detection': 'annotations-det',
    'segmentation': 'annotations-seg',
    'obb': 'annotations-obb',
}

PRIMARY_METRICS = {
    'pose': ('metrics/mAP50-95(P)', 'Pose mAP50-95'),
    'detection': ('metrics/mAP50-95(B)', 'Box mAP50-95'),
    'segmentation': ('metrics/mAP50-95(M)', 'Mask mAP50-95'),
    'obb': ('metrics/mAP50-95(B)', 'OBB mAP50-95'),
}

METRIC_LABELS = {
    'metrics/precision(B)': 'Box Precision',
    'metrics/recall(B)': 'Box Recall',
    'metrics/mAP50(B)': 'Box mAP50',
    'metrics/mAP50-95(B)': 'Box mAP50-95',
    'metrics/precision(P)': 'Pose Precision',
    'metrics/recall(P)': 'Pose Recall',
    'metrics/mAP50(P)': 'Pose mAP50',
    'metrics/mAP50-95(P)': 'Pose mAP50-95',
    'metrics/precision(M)': 'Mask Precision',
    'metrics/recall(M)': 'Mask Recall',
    'metrics/mAP50(M)': 'Mask mAP50',
    'metrics/mAP50-95(M)': 'Mask mAP50-95',
}

SERIES_LABELS = {
    **METRIC_LABELS,
    'train/box_loss': 'Train Box Loss',
    'train/cls_loss': 'Train Class Loss',
    'train/dfl_loss': 'Train DFL Loss',
    'train/pose_loss': 'Train Pose Loss',
    'train/kobj_loss': 'Train Keypoint Objectness Loss',
    'train/seg_loss': 'Train Segmentation Loss',
    'val/box_loss': 'Validation Box Loss',
    'val/cls_loss': 'Validation Class Loss',
    'val/dfl_loss': 'Validation DFL Loss',
    'val/pose_loss': 'Validation Pose Loss',
    'val/kobj_loss': 'Validation Keypoint Objectness Loss',
    'val/seg_loss': 'Validation Segmentation Loss',
    'lr/pg0': 'Learning Rate / Group 0',
    'lr/pg1': 'Learning Rate / Group 1',
    'lr/pg2': 'Learning Rate / Group 2',
}

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


@dataclass(frozen=True)
class ModelArtifact:
    path: str
    name: str
    file_format: str
    framework: str
    size_bytes: int
    modified_at: str
    role: str


@dataclass(frozen=True)
class MetricSummary:
    key: str
    label: str
    best_value: float
    best_epoch: int
    last_value: float


@dataclass(frozen=True)
class MetricPoint:
    epoch: int
    value: float


@dataclass(frozen=True)
class MetricSeries:
    key: str
    label: str
    category: str
    higher_is_better: bool | None
    points: tuple[MetricPoint, ...]


@dataclass(frozen=True)
class DatasetSource:
    role: str
    dataset_name: str
    batch_name: str
    dataset_root: str
    batch_root: str
    image_path: str
    annotation_path: str
    labels_path: str
    annotation_dir: str
    image_count: int
    annotation_count: int
    label_count: int
    available: bool


@dataclass(frozen=True)
class ModelRecord:
    model_id: str
    name: str
    project_name: str
    task_type: str
    framework: str
    file_format: str
    path: str
    size_bytes: int = 0
    modified_at: str = '-'
    precision: str = '-'
    version: str = '-'
    input_size: str = '-'
    status: str = '就绪'
    is_demo: bool = False
    architecture: str = '-'
    planned_epochs: int = 0
    actual_epochs: int = 0
    batch_size: str = '-'
    optimizer: str = '-'
    device: str = '-'
    dataset_config: str = ''
    primary_metric_name: str = '-'
    primary_metric_value: float | None = None
    primary_metric_epoch: int = 0
    training_seconds: float = 0.0
    artifacts: tuple[ModelArtifact, ...] = ()
    metrics: tuple[MetricSummary, ...] = ()
    metric_series: tuple[MetricSeries, ...] = ()
    data_sources: tuple[DatasetSource, ...] = ()
    result_assets: tuple[str, ...] = ()
    training_args: tuple[tuple[str, str], ...] = ()


def scan_model_repository(directory: str | Path) -> list[ModelRecord]:
    """Discover one model record per Ultralytics run directory."""
    root = Path(directory).expanduser()
    if not root.is_dir():
        return []
    root = root.resolve()

    run_dirs = {path.parent.resolve() for path in root.rglob('args.yaml')}
    for weights_dir in root.rglob('weights'):
        if weights_dir.is_dir():
            run_dirs.add(weights_dir.parent.resolve())

    records = [
        _parse_training_run(run_dir, root)
        for run_dir in sorted(run_dirs)
    ]
    return [record for record in records if record is not None]


def _parse_training_run(run_dir: Path, repository_root: Path) -> ModelRecord | None:
    args_path = run_dir / 'args.yaml'
    args = _load_yaml(args_path) if args_path.is_file() else {}
    weights_dir = run_dir / 'weights'
    artifacts = _scan_artifacts(weights_dir)
    if not args and not artifacts:
        return None

    task_type = _normalize_task(args.get('task'))
    if task_type == 'other':
        task_type = _infer_task_type(run_dir, repository_root)

    relative_parts = _relative_parts(run_dir, repository_root)
    project_name = (
        relative_parts[0]
        if len(relative_parts) > 1
        else repository_root.name
    )
    architecture = _architecture_name(args.get('model'))
    dataset_config = _resolve_config_path(args.get('data'), run_dir, args)
    metrics, metric_series, actual_epochs, training_seconds = _read_results(
        run_dir / 'results.csv'
    )
    primary_name, primary_value, primary_epoch = _primary_metric(
        task_type, metrics
    )
    data_sources = _read_dataset_sources(dataset_config, task_type)
    primary_artifact = _primary_artifact(artifacts)
    formats = []
    frameworks = []
    for artifact in artifacts:
        if artifact.file_format not in formats:
            formats.append(artifact.file_format)
        if artifact.framework not in frameworks:
            frameworks.append(artifact.framework)

    modified_timestamp = max(
        [path.stat().st_mtime for path in (args_path, run_dir / 'results.csv') if path.is_file()]
        + [Path(artifact.path).stat().st_mtime for artifact in artifacts]
        + [run_dir.stat().st_mtime],
    )
    result_assets = tuple(
        str(path.resolve())
        for path in sorted(run_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    planned_epochs = _as_int(args.get('epochs'))
    status = '完整' if args and metrics and artifacts else '部分'
    return ModelRecord(
        model_id=str(run_dir.resolve()),
        name=run_dir.name,
        project_name=project_name,
        task_type=task_type,
        framework='Ultralytics YOLO' if args else (frameworks[0] if frameworks else '-'),
        file_format=' + '.join(formats) if formats else '-',
        path=str(run_dir.resolve()),
        size_bytes=primary_artifact.size_bytes if primary_artifact else 0,
        modified_at=datetime.fromtimestamp(modified_timestamp).strftime('%Y-%m-%d'),
        precision=_artifact_precision(primary_artifact),
        version=_infer_version(run_dir.name),
        input_size=_format_input_size(args.get('imgsz')),
        status=status,
        architecture=architecture,
        planned_epochs=planned_epochs,
        actual_epochs=actual_epochs,
        batch_size=str(args.get('batch', '-')),
        optimizer=str(args.get('optimizer', '-')),
        device=str(args.get('device', '-')),
        dataset_config=str(dataset_config) if dataset_config else '',
        primary_metric_name=primary_name,
        primary_metric_value=primary_value,
        primary_metric_epoch=primary_epoch,
        training_seconds=training_seconds,
        artifacts=tuple(artifacts),
        metrics=tuple(metrics),
        metric_series=tuple(metric_series),
        data_sources=tuple(data_sources),
        result_assets=result_assets,
        training_args=tuple(
            (str(key), _config_value_text(value))
            for key, value in args.items()
        ),
    )


def _load_yaml(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def _config_value_text(value) -> str:
    if value is None:
        return '-'
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, separators=(',', ': '))
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _scan_artifacts(weights_dir: Path) -> list[ModelArtifact]:
    if not weights_dir.is_dir():
        return []
    artifacts = []
    for path in sorted(weights_dir.iterdir()):
        suffix = path.suffix.lower()
        if not path.is_file() or suffix not in MODEL_FORMATS:
            continue
        if suffix == '.bin' and path.with_suffix('.xml').is_file():
            continue
        framework, file_format = MODEL_FORMATS[suffix]
        stat = path.stat()
        artifacts.append(ModelArtifact(
            path=str(path.resolve()),
            name=path.name,
            file_format=file_format,
            framework=framework,
            size_bytes=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d'),
            role=_artifact_role(path),
        ))
    return artifacts


def _artifact_role(path: Path) -> str:
    name = path.stem.lower()
    if name == 'best':
        return 'best'
    if name == 'last':
        return 'last'
    if re.fullmatch(r'epoch\d+', name):
        return 'checkpoint'
    return 'export'


def _primary_artifact(artifacts: list[ModelArtifact]) -> ModelArtifact | None:
    role_order = {'best': 0, 'last': 1, 'export': 2, 'checkpoint': 3}
    return min(
        artifacts,
        key=lambda item: (role_order.get(item.role, 9), item.name.lower()),
        default=None,
    )


def _read_results(path: Path) -> tuple[
    list[MetricSummary], list[MetricSeries], int, float
]:
    if not path.is_file():
        return [], [], 0, 0.0
    try:
        with path.open('r', encoding='utf-8-sig', newline='') as stream:
            rows = [
                {str(key).strip(): value for key, value in row.items()}
                for row in csv.DictReader(stream)
            ]
    except (OSError, UnicodeError, csv.Error):
        return [], [], 0, 0.0
    if not rows:
        return [], [], 0, 0.0

    summaries = []
    for key in rows[0]:
        if key not in METRIC_LABELS:
            continue
        values = []
        for row in rows:
            value = _as_float(row.get(key))
            epoch = _as_int(row.get('epoch'))
            if value is not None:
                values.append((value, epoch))
        if not values:
            continue
        best_value, best_epoch = max(values, key=lambda item: item[0])
        summaries.append(MetricSummary(
            key=key,
            label=METRIC_LABELS[key],
            best_value=best_value,
            best_epoch=best_epoch,
            last_value=values[-1][0],
        ))

    series = []
    for key in rows[0]:
        if not _is_comparable_series(key):
            continue
        points = []
        for row_index, row in enumerate(rows):
            value = _as_float(row.get(key))
            if value is None:
                continue
            epoch_value = row.get('epoch')
            epoch = _as_int(epoch_value) if epoch_value not in (None, '') else row_index
            points.append(MetricPoint(epoch=epoch, value=value))
        if not points:
            continue
        category, higher_is_better = _series_semantics(key)
        series.append(MetricSeries(
            key=key,
            label=_series_label(key),
            category=category,
            higher_is_better=higher_is_better,
            points=tuple(points),
        ))
    return summaries, series, len(rows), _as_float(rows[-1].get('time')) or 0.0


def _is_comparable_series(key: str) -> bool:
    return (
        key in METRIC_LABELS
        or key.startswith('train/')
        or key.startswith('val/')
        or key.startswith('lr/')
    )


def _series_semantics(key: str) -> tuple[str, bool | None]:
    if key.startswith('metrics/'):
        return 'evaluation', True
    if key.endswith('_loss'):
        return 'loss', False
    if key.startswith('lr/'):
        return 'learning_rate', None
    return 'other', None


def _series_label(key: str) -> str:
    if key in SERIES_LABELS:
        return SERIES_LABELS[key]
    prefix, _, name = key.partition('/')
    readable = name.replace('_', ' ').strip().title() or key
    return f'{prefix.title()} {readable}' if name else readable


def _primary_metric(task_type: str, metrics: list[MetricSummary]):
    key, label = PRIMARY_METRICS.get(task_type, ('', '-'))
    for metric in metrics:
        if metric.key == key:
            return label, metric.best_value, metric.best_epoch
    return label, None, 0


def _resolve_config_path(value, run_dir: Path, args: dict) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve() if path.exists() else path

    candidates = [run_dir / path, run_dir.parent / path]
    save_dir = args.get('save_dir')
    if save_dir:
        save_path = Path(str(save_dir)).expanduser()
        candidates.extend(parent / path for parent in save_path.parents[:3])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def _read_dataset_sources(config_path: Path | None,
                          task_type: str) -> list[DatasetSource]:
    if config_path is None or not config_path.is_file():
        return []
    config = _load_yaml(config_path)
    base = config_path.parent
    dataset_base = config.get('path')
    if dataset_base:
        root = Path(str(dataset_base)).expanduser()
        base = root if root.is_absolute() else (base / root)

    sources = []
    for role in ('train', 'val', 'test'):
        for value in _split_values(config.get(role)):
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                path = base / path
            sources.append(_dataset_source(role, path, task_type))
    return sources


def _split_values(value) -> list[str]:
    if value is None or value == '':
        return []
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_split_values(item))
        return result
    return [str(value)]


def _dataset_source(role: str, image_path: Path,
                    task_type: str) -> DatasetSource:
    image_path = image_path.expanduser()
    if image_path.exists():
        image_path = image_path.resolve()
    scope_root = None
    scope = 'raw'
    for ancestor in [image_path, *image_path.parents]:
        if ancestor.name == 'training_data':
            scope_root, scope = ancestor, 'train'
            break
        if ancestor.name == 'test_data':
            scope_root, scope = ancestor, 'test'
            break

    if scope_root is not None:
        dataset_root = scope_root.parent
        try:
            relative = image_path.relative_to(scope_root)
        except ValueError:
            relative = Path()
        first = relative.parts[0] if relative.parts else ''
        if first == 'images' or not first:
            batch_root = scope_root
            batch_name = '默认测试集' if scope == 'test' else '默认训练集'
        else:
            batch_root = scope_root / first
            batch_name = first
    else:
        batch_root = _batch_root_for_images(image_path)
        dataset_root = _dataset_root_for_raw(batch_root)
        try:
            relative = image_path.relative_to(dataset_root / 'images')
        except ValueError:
            relative = Path()
        batch_name = relative.parts[0] if relative.parts else '默认原始集'

    annotation_dir = TASK_ANNOTATION_DIRS.get(task_type, 'annotations')
    annotation_path = _peer_data_path(image_path, batch_root, annotation_dir)
    labels_path = _peer_data_path(image_path, batch_root, 'labels')
    return DatasetSource(
        role=role,
        dataset_name=dataset_root.name,
        batch_name=batch_name,
        dataset_root=str(dataset_root),
        batch_root=str(batch_root),
        image_path=str(image_path),
        annotation_path=str(annotation_path),
        labels_path=str(labels_path),
        annotation_dir=annotation_dir,
        image_count=_count_files(image_path, IMAGE_EXTENSIONS),
        annotation_count=_count_files(annotation_path, {'.json'}),
        label_count=_count_files(labels_path, {'.txt'}),
        available=image_path.is_dir(),
    )


def _batch_root_for_images(image_path: Path) -> Path:
    for ancestor in [image_path, *image_path.parents]:
        if ancestor.name == 'images':
            return ancestor.parent
    return image_path.parent if image_path.suffix else image_path


def _dataset_root_for_raw(batch_root: Path) -> Path:
    if batch_root.parent.name == 'images':
        return batch_root.parent.parent
    return batch_root


def _peer_data_path(image_path: Path, batch_root: Path,
                    peer_name: str) -> Path:
    images_root = batch_root / 'images'
    try:
        relative = image_path.relative_to(images_root)
    except ValueError:
        relative = Path()
    return batch_root / peer_name / relative


def _count_files(path: Path, extensions: set[str]) -> int:
    if not path.is_dir():
        return 0
    try:
        return sum(
            1 for item in path.rglob('*')
            if item.is_file() and item.suffix.lower() in extensions
        )
    except (OSError, PermissionError):
        return 0


def _normalize_task(value) -> str:
    token = str(value or '').strip().lower()
    return {
        'detect': 'detection',
        'detection': 'detection',
        'segment': 'segmentation',
        'segmentation': 'segmentation',
        'pose': 'pose',
        'obb': 'obb',
    }.get(token, 'other')


def _infer_task_type(path: Path, root: Path) -> str:
    token = '/'.join(_relative_parts(path, root)).lower().replace('-', '_')
    checks = (
        ('obb', ('obb', 'rotated', 'rotation', 'oriented')),
        ('segmentation', ('segment', 'segmentation', 'mask', 'sam')),
        ('pose', ('pose', 'keypoint', 'kpt', 'skeleton')),
        ('detection', ('detect', 'detection', 'detector', 'yolo')),
    )
    for task_type, keywords in checks:
        if any(keyword in token for keyword in keywords):
            return task_type
    return 'other'


def _relative_parts(path: Path, root: Path) -> tuple[str, ...]:
    try:
        return path.relative_to(root).parts
    except ValueError:
        return path.parts


def _architecture_name(value) -> str:
    if not value:
        return '-'
    return Path(str(value)).stem


def _format_input_size(value) -> str:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return f'{value[0]} x {value[1]}'
    if value not in (None, ''):
        return f'{value} x {value}'
    return '-'


def _artifact_precision(artifact: ModelArtifact | None) -> str:
    if artifact is None:
        return '-'
    token = artifact.name.lower()
    if 'int8' in token:
        return 'INT8'
    if 'fp16' in token or 'half' in token:
        return 'FP16'
    if 'bf16' in token:
        return 'BF16'
    return 'FP32'


def _infer_version(name: str) -> str:
    match = re.search(r'(?i)(?:^|[_\-.])(v\d+(?:\.\d+){0,2})(?:$|[_\-.])', name)
    return match.group(1) if match else '-'


def _as_int(value) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
