"""Shared dataset preparation for the operation and training centers."""

from __future__ import annotations

import errno
import json
import math
import os
import random
import shutil
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import yaml

from app.models.annotation_schema import (
    infer_annotation_schema,
    infer_left_right_pairs,
)


IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp',
}
TEST_LIST_FILE = 'test_list.txt'


class DatasetPreparationError(RuntimeError):
    """Raised when a dataset cannot safely be prepared."""


@dataclass(frozen=True)
class DatasetPreparationRequest:
    dataset_root: Path
    source_names: tuple[str, ...]
    target_name: str
    task_type: str = 'pose'
    annotation_dir: str = 'annotations'
    label_dir: str = 'labels'
    val_ratio: float = 0.2
    seed: int = 42
    test_ratio: float = 0.0
    test_batch_name: str = ''
    reuse_split: bool = False
    use_copy: bool = False
    exclude_test: bool = True
    allow_background_without_label: bool = False
    skip_incomplete_samples: bool = False
    skip_duplicate_samples: bool = False
    class_names: tuple[str, ...] = ()
    keypoints: tuple[str, ...] = ()
    left_right_pairs: tuple[tuple[str, str], ...] = ()

    def normalized(self) -> 'DatasetPreparationRequest':
        root = Path(self.dataset_root).expanduser().resolve()
        sources = tuple(dict.fromkeys(
            str(name).strip() for name in self.source_names if str(name).strip()
        ))
        target_name = str(self.target_name or '').strip()
        _validate_simple_name(target_name, '训练批次名称')
        _validate_simple_name(self.annotation_dir, '标注目录')
        _validate_simple_name(self.label_dir, '标签目录')
        if not 0.0 < float(self.val_ratio) < 1.0:
            raise DatasetPreparationError('验证集比例必须大于 0 且小于 1')
        test_ratio = float(self.test_ratio)
        if not 0.0 <= test_ratio < 1.0:
            raise DatasetPreparationError('测试集比例必须大于等于 0 且小于 1')
        test_batch_name = str(self.test_batch_name or '').strip()
        if test_ratio > 0:
            _validate_simple_name(test_batch_name, '测试批次名称')
        if not sources:
            raise DatasetPreparationError('至少选择一个原始数据批次')
        return DatasetPreparationRequest(
            dataset_root=root,
            source_names=sources,
            target_name=target_name,
            task_type=str(self.task_type or 'pose').strip(),
            annotation_dir=str(self.annotation_dir).strip(),
            label_dir=str(self.label_dir).strip(),
            val_ratio=float(self.val_ratio),
            seed=int(self.seed),
            test_ratio=test_ratio,
            test_batch_name=test_batch_name,
            reuse_split=bool(self.reuse_split),
            use_copy=bool(self.use_copy),
            exclude_test=bool(self.exclude_test),
            allow_background_without_label=bool(
                self.allow_background_without_label
            ),
            skip_incomplete_samples=bool(self.skip_incomplete_samples),
            skip_duplicate_samples=bool(self.skip_duplicate_samples),
            class_names=tuple(str(name) for name in self.class_names),
            keypoints=tuple(str(name) for name in self.keypoints),
            left_right_pairs=tuple(
                (str(left), str(right))
                for left, right in self.left_right_pairs
            ),
        )


@dataclass(frozen=True)
class DatasetSample:
    stem: str
    source_name: str
    image_path: Path
    annotation_path: Path
    label_path: Path | None
    background_without_label: bool = False


@dataclass
class DatasetScanResult:
    request: DatasetPreparationRequest
    samples: list[DatasetSample] = field(default_factory=list)
    source_image_counts: dict[str, int] = field(default_factory=dict)
    source_ready_counts: dict[str, int] = field(default_factory=dict)
    test_excluded: list[Path] = field(default_factory=list)
    duplicate_images: list[Path] = field(default_factory=list)
    missing_annotations: list[Path] = field(default_factory=list)
    invalid_annotations: list[Path] = field(default_factory=list)
    missing_labels: list[Path] = field(default_factory=list)
    invalid_labels: list[Path] = field(default_factory=list)
    observed_label_columns: set[int] = field(default_factory=set)
    background_without_labels: list[Path] = field(default_factory=list)
    missing_sources: list[str] = field(default_factory=list)

    @property
    def blocking_count(self) -> int:
        missing_labels = len(self.missing_labels)
        if self.request.allow_background_without_label:
            missing_labels -= len(self.background_without_labels)
        missing_annotations = len(self.missing_annotations)
        if self.request.skip_incomplete_samples:
            missing_annotations = 0
            missing_labels = 0
        duplicate_images = (
            0 if self.request.skip_duplicate_samples
            else len(self.duplicate_images)
        )
        return (
            len(self.missing_sources)
            + duplicate_images
            + missing_annotations
            + len(self.invalid_annotations)
            + len(self.invalid_labels)
            + max(0, missing_labels)
        )

    @property
    def can_prepare(self) -> bool:
        return self.blocking_count == 0 and len(self.samples) >= 2

    def blocking_message(self, max_files: int = 12) -> str:
        parts = []
        if self.missing_sources:
            parts.append('不存在的来源批次: ' + ', '.join(self.missing_sources))
        if self.duplicate_images and not self.request.skip_duplicate_samples:
            parts.append(
                f'不同来源存在同名图片 {len(self.duplicate_images)} 个，合并后会冲突:\n'
                + _path_preview(self.duplicate_images, max_files)
            )
        if (
            self.missing_annotations
            and not self.request.skip_incomplete_samples
        ):
            parts.append(
                f'缺少 JSON 标注 {len(self.missing_annotations)} 个:\n'
                + _path_preview(self.missing_annotations, max_files)
            )
        if self.invalid_annotations:
            parts.append(
                f'无效 JSON 标注 {len(self.invalid_annotations)} 个:\n'
                + _path_preview(self.invalid_annotations, max_files)
            )
        missing_labels = list(self.missing_labels)
        if self.request.allow_background_without_label:
            background = set(self.background_without_labels)
            missing_labels = [path for path in missing_labels if path not in background]
        if missing_labels and not self.request.skip_incomplete_samples:
            parts.append(
                f'缺少 YOLO TXT 标签 {len(missing_labels)} 个:\n'
                + _path_preview(missing_labels, max_files)
            )
        if self.invalid_labels:
            observed = ', '.join(
                str(value) for value in sorted(self.observed_label_columns)
            ) or '无法解析'
            if len(self.observed_label_columns) > 1:
                expected = min(self.observed_label_columns)
                detail = f'每行期望 {expected} 列，实际发现 {observed} 列；'
                detail += '同一批次存在多种标签结构，不能混合训练'
            else:
                detail = f'无法从标签推断统一的关键点结构，实际发现 {observed} 列'
            parts.append(
                f'YOLO TXT 标签结构不匹配 {len(self.invalid_labels)} 个：'
                f'{detail}。\n'
                + _path_preview(self.invalid_labels, max_files)
            )
        if len(self.samples) < 2:
            parts.append('有效训练样本少于 2 个，无法划分 train/val')
        return '\n\n'.join(parts) or '数据尚未通过训练准备检查'


@dataclass(frozen=True)
class PreparedDataset:
    batch_root: Path
    dataset_yaml: Path
    manifest_path: Path
    train_count: int
    val_count: int
    total_count: int
    test_batch_root: Path | None = None
    test_count: int = 0


@dataclass(frozen=True)
class TrainingBatchSummary:
    batch_root: Path
    task_type: str
    annotation_dir: str
    image_count: int
    annotation_count: int
    label_count: int
    train_image_count: int
    train_label_count: int
    val_image_count: int
    val_label_count: int
    missing_top_labels: tuple[str, ...]
    missing_train_labels: tuple[str, ...]
    missing_val_labels: tuple[str, ...]
    has_dataset_yaml: bool
    keypoint_shape: tuple[int, int] = ()
    expected_label_columns: int = 0
    observed_label_columns: tuple[int, ...] = ()
    invalid_train_labels: tuple[str, ...] = ()
    invalid_val_labels: tuple[str, ...] = ()
    yaml_schema_matches: bool = True
    class_names: dict[int, str] = field(default_factory=dict)
    train_class_counts: dict[int, int] = field(default_factory=dict)
    val_class_counts: dict[int, int] = field(default_factory=dict)
    class_map_matches: bool = True
    missing_val_class_ids: tuple[int, ...] = ()

    @property
    def is_split(self) -> bool:
        return self.train_image_count > 0 and self.val_image_count > 0

    @property
    def is_ready(self) -> bool:
        return (
            self.is_split
            and self.has_dataset_yaml
            and not self.missing_train_labels
            and not self.missing_val_labels
            and not self.invalid_train_labels
            and not self.invalid_val_labels
            and self.yaml_schema_matches
            and self.class_map_matches
            and not self.missing_val_class_ids
        )

    @property
    def invalid_label_count(self) -> int:
        return len(self.invalid_train_labels) + len(self.invalid_val_labels)

    def readiness_message(self, max_files: int = 8) -> str:
        if self.invalid_label_count:
            actual = ', '.join(
                str(value) for value in self.observed_label_columns
            ) or '无法解析'
            preview = list(self.invalid_train_labels + self.invalid_val_labels)
            suffix = ''
            if preview:
                suffix = '\n异常文件：' + '、'.join(preview[:max_files])
                if len(preview) > max_files:
                    suffix += f' 等 {len(preview)} 个'
            return (
                f'Pose TXT 标签结构不匹配：当前批次应统一为 '
                f'{self.expected_label_columns or "-"} 列，实际发现 {actual} 列；'
                f'共有 {self.invalid_label_count} 个标签文件不符合要求。'
                f'请按当前 {self.keypoint_shape[0] if self.keypoint_shape else "-"} '
                f'个关键点重新生成 YOLO Pose TXT。{suffix}'
            )
        if not self.yaml_schema_matches:
            actual = ', '.join(
                str(value) for value in self.observed_label_columns
            ) or '未发现有效 TXT 结构'
            return (
                f'dataset.yaml 的关键点结构与批次实际标签不一致：'
                f'实际 TXT 为 {actual} 列。可以通过重新分析批次自动生成 YAML。'
            )
        if not self.class_map_matches:
            observed = sorted(
                set(self.train_class_counts) | set(self.val_class_counts)
            )
            return (
                f'dataset.yaml 类别映射与 TXT 不一致：实际类别 ID 为 '
                f'{observed}，YAML names 为 {sorted(self.class_names)}。'
            )
        if self.missing_val_class_ids:
            labels = [
                f'{class_id}:{self.class_names.get(class_id, "未命名")}'
                for class_id in self.missing_val_class_ids
            ]
            return (
                '验证集缺少训练集中存在的类别：' + '、'.join(labels)
                + '。请按来源重新划分训练/验证集。'
            )
        if not self.has_dataset_yaml:
            return '训练批次缺少 dataset.yaml'
        if not self.is_split:
            return '训练批次尚未完成 train/val 划分'
        if self.missing_train_labels or self.missing_val_labels:
            return '训练批次存在缺失的 TRAIN/VAL TXT 标签'
        return '训练数据预检通过'


def list_source_batches(dataset_root: str | Path,
                        annotation_dir: str = 'annotations',
                        label_dir: str = 'labels') -> list[dict]:
    """Return raw image batches with matching JSON/TXT counts."""
    root = Path(dataset_root).expanduser().resolve()
    images_root = root / 'images'
    if not images_root.is_dir():
        return []

    rows = []
    for image_dir in sorted(images_root.iterdir()):
        if not image_dir.is_dir():
            continue
        ann_dir = _find_source_peer(root, annotation_dir, image_dir)
        lbl_dir = _find_source_peer(root, label_dir, image_dir)
        image_count = _count_images(image_dir)
        rows.append({
            'name': image_dir.name,
            'image_path': str(image_dir),
            'annotation_path': str(ann_dir) if ann_dir else '',
            'label_path': str(lbl_dir) if lbl_dir else '',
            'image_count': image_count,
            'annotation_count': _count_suffix(ann_dir, '.json'),
            'label_count': _count_suffix(lbl_dir, '.txt'),
            'has_annotation_set': ann_dir is not None,
            'has_label_set': lbl_dir is not None,
        })
    return rows


def scan_dataset(request: DatasetPreparationRequest) -> DatasetScanResult:
    """Build a checked list of image/JSON/TXT triples without writing files."""
    request = request.normalized()
    result = DatasetScanResult(request=request)
    root = request.dataset_root
    images_root = root / 'images'
    if not images_root.is_dir():
        raise DatasetPreparationError(f'images 目录不存在: {images_root}')

    test_stems = _load_test_stems(root / 'test_data') if request.exclude_test else set()
    seen_stems: set[str] = set()

    for source_name in request.source_names:
        image_dir = images_root / source_name
        if not image_dir.is_dir():
            result.missing_sources.append(source_name)
            continue
        annotation_dir = _find_source_peer(
            root, request.annotation_dir, image_dir
        )
        label_dir = _find_source_peer(root, request.label_dir, image_dir)
        image_files = _list_images(image_dir)
        result.source_image_counts[source_name] = len(image_files)
        ready_count = 0

        for image_path in image_files:
            stem = image_path.stem
            if stem in test_stems:
                result.test_excluded.append(image_path)
                continue
            if stem in seen_stems:
                result.duplicate_images.append(image_path)
                continue
            seen_stems.add(stem)

            annotation_path = (
                annotation_dir / f'{stem}.json' if annotation_dir else None
            )
            if annotation_path is None or not annotation_path.is_file():
                result.missing_annotations.append(image_path)
                continue

            try:
                annotation_data = json.loads(
                    annotation_path.read_text(encoding='utf-8')
                )
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                result.invalid_annotations.append(annotation_path)
                continue
            if not isinstance(annotation_data, dict):
                result.invalid_annotations.append(annotation_path)
                continue

            label_path = label_dir / f'{stem}.txt' if label_dir else None
            background_without_label = False
            if label_path is None or not label_path.is_file():
                result.missing_labels.append(image_path)
                shapes = annotation_data.get('shapes')
                if isinstance(shapes, list) and not shapes:
                    result.background_without_labels.append(image_path)
                    background_without_label = True
                if not (
                    background_without_label
                    and request.allow_background_without_label
                ):
                    continue
                label_path = None

            if label_path is not None:
                valid, observed = _inspect_label_structure(
                    label_path, request.task_type
                )
                result.observed_label_columns.update(observed)
                if not valid:
                    result.invalid_labels.append(label_path)
                    continue

            result.samples.append(DatasetSample(
                stem=stem,
                source_name=source_name,
                image_path=image_path,
                annotation_path=annotation_path,
                label_path=label_path,
                background_without_label=background_without_label,
            ))
            ready_count += 1

        result.source_ready_counts[source_name] = ready_count

    # A batch must use one label schema. The schema is inferred from the
    # labels themselves; the review template is not allowed to redefine it.
    valid_columns = {}
    for sample in result.samples:
        if sample.label_path is None:
            continue
        valid, observed = _inspect_label_structure(
            sample.label_path, request.task_type
        )
        if valid and len(observed) == 1:
            valid_columns[sample.label_path] = next(iter(observed))
    if valid_columns:
        dominant = Counter(valid_columns.values()).most_common(1)[0][0]
        for label_path, columns in valid_columns.items():
            if columns != dominant and label_path not in result.invalid_labels:
                result.invalid_labels.append(label_path)

    return result


def prepare_dataset(request: DatasetPreparationRequest,
                    scan: DatasetScanResult | None = None) -> PreparedDataset:
    """Create the merged batch and deterministic train/val split."""
    request = request.normalized()
    scan = scan or scan_dataset(request)
    if scan.request != request:
        scan = scan_dataset(request)
    if not scan.can_prepare:
        raise DatasetPreparationError(scan.blocking_message())

    training_root = request.dataset_root / 'training_data'
    target = training_root / request.target_name
    if target.exists():
        raise DatasetPreparationError(f'训练批次已存在，不能覆盖: {target}')

    # 划分：复用已有同参数划分（跨任务对齐：同一张图在不同任务的
    # train/val/test 集合保持一致），或按测试/验证比例重新分层抽取。
    reuse_source: Path | None = None
    split_assign: dict[str, str] = {}
    if request.reuse_split:
        reused = _find_reusable_split(request)
        if reused is None:
            raise DatasetPreparationError(
                '未找到可复用的同参数划分（来源/验证比例/测试比例/种子需一致），'
                '请取消“复用上次划分”后重试'
            )
        missing = {sample.stem for sample in scan.samples} - set(reused)
        if missing:
            raise DatasetPreparationError(
                f'可复用划分与当前样本不一致（缺少 {len(missing)} 个样本记录），'
                '请检查数据或取消“复用上次划分”'
            )
        split_assign = reused
        reuse_source = Path(reused.get('_manifest_path', '')) or None
    else:
        test_samples: list[DatasetSample] = []
        if request.test_ratio > 0:
            test_stems = _source_stratified_val_stems(
                [
                    (sample.source_name, sample.stem, sample.label_path)
                    for sample in scan.samples
                ],
                request.test_ratio,
                request.seed,
            )
            test_samples = [
                sample for sample in scan.samples if sample.stem in test_stems
            ]
        remaining = [
            sample for sample in scan.samples
            if sample.stem not in {sample.stem for sample in test_samples}
        ]
        val_stems = _source_stratified_val_stems(
            [
                (sample.source_name, sample.stem, sample.label_path)
                for sample in remaining
            ],
            request.val_ratio,
            request.seed,
        )
        for sample in remaining:
            split_assign[sample.stem] = (
                'val' if sample.stem in val_stems else 'train'
            )
        for sample in test_samples:
            split_assign[sample.stem] = 'test'

    train_samples = [
        sample for sample in scan.samples if split_assign[sample.stem] == 'train'
    ]
    val_samples = [
        sample for sample in scan.samples if split_assign[sample.stem] == 'val'
    ]
    test_samples = [
        sample for sample in scan.samples if split_assign[sample.stem] == 'test'
    ]
    remaining = train_samples + val_samples

    training_root.mkdir(parents=True, exist_ok=True)
    staging = training_root / (
        f'.{request.target_name}.preparing-{uuid.uuid4().hex[:8]}'
    )
    test_target: Path | None = None
    try:
        _create_output_tree(staging, request.annotation_dir)
        split_by_stem = {
            sample.stem: 'train' for sample in train_samples
        }
        split_by_stem.update({sample.stem: 'val' for sample in val_samples})
        split_by_stem.update({sample.stem: 'test' for sample in test_samples})

        for sample in remaining:
            split = split_by_stem[sample.stem]
            merged_image = staging / 'images' / sample.image_path.name
            merged_annotation = (
                staging / request.annotation_dir / sample.annotation_path.name
            )
            merged_label = staging / 'labels' / f'{sample.stem}.txt'
            _materialize(sample.image_path, merged_image, request.use_copy)
            _materialize(sample.annotation_path, merged_annotation, request.use_copy)
            if sample.label_path is not None:
                _materialize(sample.label_path, merged_label, request.use_copy)
            else:
                merged_label.write_text('', encoding='utf-8')

            _materialize(
                merged_image,
                staging / 'train_data' / 'images' / split / sample.image_path.name,
                request.use_copy,
            )
            _materialize(
                merged_label,
                staging / 'train_data' / 'labels' / split / f'{sample.stem}.txt',
                request.use_copy,
            )

        dataset_yaml = staging / 'dataset.yaml'
        _write_dataset_yaml(dataset_yaml, request, scan.samples)
        manifest_path = staging / 'preparation_manifest.json'
        _write_manifest(
            manifest_path, request, scan, split_by_stem,
            len(train_samples), len(val_samples),
            test_batch=request.test_batch_name if test_samples else '',
            test_ratio=request.test_ratio if test_samples else 0.0,
            test_count=len(test_samples),
        )
        (staging / 'review_report.json').write_text(
            json.dumps({
                'status': 'pending',
                'task_type': request.task_type,
                'created_at': _utc_now(),
                'issue_files': 0,
                'issue_count': 0,
            }, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

        # 划分清单：任务无关的 stem→集合 分配记录，供跨任务复用
        split_payload = {
            'version': 1,
            'created_at': _utc_now(),
            'task_type': request.task_type,
            'source_batches': list(request.source_names),
            'seed': request.seed,
            'val_ratio': request.val_ratio,
            'test_ratio': request.test_ratio,
            'reused_from': (
                str(reuse_source) if reuse_source is not None else ''
            ),
            'split': {
                stem: value
                for stem, value in split_assign.items()
                if not stem.startswith('_')
            },
        }
        (staging / 'split_manifest.json').write_text(
            json.dumps(split_payload, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

        # 测试批次独立写入 test_data/<测试批次名>/，与训练批次同一次准备原子完成
        if test_samples:
            test_root = request.dataset_root / 'test_data'
            test_target = test_root / request.test_batch_name
            if test_target.exists():
                raise DatasetPreparationError(
                    f'测试批次已存在，不能覆盖: {test_target}'
                )
            test_root.mkdir(parents=True, exist_ok=True)
            test_staging = test_root / (
                f'.{request.test_batch_name}.preparing-{uuid.uuid4().hex[:8]}'
            )
            try:
                _create_test_output_tree(test_staging, request.annotation_dir)
                for sample in test_samples:
                    _materialize(
                        sample.image_path,
                        test_staging / 'images' / sample.image_path.name,
                        request.use_copy,
                    )
                    _materialize(
                        sample.annotation_path,
                        test_staging / request.annotation_dir
                        / sample.annotation_path.name,
                        request.use_copy,
                    )
                    test_label = test_staging / 'labels' / f'{sample.stem}.txt'
                    if sample.label_path is not None:
                        _materialize(sample.label_path, test_label,
                                     request.use_copy)
                    else:
                        test_label.write_text('', encoding='utf-8')
                _write_test_dataset_yaml(
                    test_staging / 'dataset.yaml', request, test_samples
                )
                _write_test_manifest(
                    test_staging / 'test_manifest.json', request, scan,
                    test_samples, request.target_name,
                )
                test_staging.rename(test_target)
            except Exception:
                if test_staging.exists():
                    shutil.rmtree(test_staging, ignore_errors=True)
                raise

        staging.rename(target)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    return PreparedDataset(
        batch_root=target,
        dataset_yaml=target / 'dataset.yaml',
        manifest_path=target / 'preparation_manifest.json',
        train_count=len(train_samples),
        val_count=len(val_samples),
        total_count=len(scan.samples),
        test_batch_root=test_target,
        test_count=len(test_samples),
    )


def inspect_training_batch(batch_root: str | Path,
                           annotation_dir: str | None = None
                           ) -> TrainingBatchSummary:
    """Inspect an existing operation-center batch for training readiness."""
    root = Path(batch_root).expanduser().resolve()
    metadata = _load_batch_metadata(root)
    task_type = str(metadata.get('task_type') or 'pose')
    annotation_dir = str(
        annotation_dir or metadata.get('annotation_dir') or 'annotations'
    )
    top_images = _list_images(root / 'images')
    top_annotations = _list_suffix(root / annotation_dir, '.json')
    top_labels = _list_suffix(root / 'labels', '.txt')
    train_images = _list_images(root / 'train_data' / 'images' / 'train')
    val_images = _list_images(root / 'train_data' / 'images' / 'val')
    train_labels = _list_suffix(
        root / 'train_data' / 'labels' / 'train', '.txt'
    )
    val_labels = _list_suffix(
        root / 'train_data' / 'labels' / 'val', '.txt'
    )
    dataset_path = root / 'dataset.yaml'
    dataset_payload = _load_dataset_yaml(dataset_path)
    yaml_keypoint_shape = _dataset_keypoint_shape(dataset_payload, task_type)
    split_labels = train_labels + val_labels
    observed_columns = _observed_label_columns(split_labels, task_type)
    inferred_keypoint_shape = _schema_from_columns(observed_columns, task_type)
    keypoint_shape = inferred_keypoint_shape or yaml_keypoint_shape
    expected_columns = _expected_pose_label_columns(
        task_type,
        keypoint_shape[0] if keypoint_shape else 0,
        keypoint_shape[1] if keypoint_shape else 0,
    )
    invalid_train, train_columns = _inspect_label_files(
        train_labels, expected_columns
    )
    invalid_val, val_columns = _inspect_label_files(
        val_labels, expected_columns
    )
    class_names = _normalize_class_names(dataset_payload.get('names'))
    train_class_counts = _label_class_counts(train_labels)
    val_class_counts = _label_class_counts(val_labels)
    observed_class_ids = set(train_class_counts) | set(val_class_counts)
    class_map_matches = observed_class_ids.issubset(class_names)
    missing_val_class_ids = tuple(sorted(
        class_id for class_id, count in train_class_counts.items()
        if count > 0 and val_class_counts.get(class_id, 0) == 0
    ))
    return TrainingBatchSummary(
        batch_root=root,
        task_type=task_type,
        annotation_dir=annotation_dir,
        image_count=len(top_images),
        annotation_count=len(top_annotations),
        label_count=len(top_labels),
        train_image_count=len(train_images),
        train_label_count=len(train_labels),
        val_image_count=len(val_images),
        val_label_count=len(val_labels),
        missing_top_labels=_missing_label_names(top_images, top_labels),
        missing_train_labels=_missing_label_names(train_images, train_labels),
        missing_val_labels=_missing_label_names(val_images, val_labels),
        has_dataset_yaml=(root / 'dataset.yaml').is_file(),
        keypoint_shape=keypoint_shape,
        expected_label_columns=expected_columns,
        observed_label_columns=tuple(sorted(train_columns | val_columns)),
        invalid_train_labels=invalid_train,
        invalid_val_labels=invalid_val,
        yaml_schema_matches=(
            task_type != 'pose'
            or not inferred_keypoint_shape
            or inferred_keypoint_shape == yaml_keypoint_shape
        ),
        class_names=class_names,
        train_class_counts=train_class_counts,
        val_class_counts=val_class_counts,
        class_map_matches=class_map_matches,
        missing_val_class_ids=missing_val_class_ids,
    )


def ensure_training_dataset_yaml(batch_root: str | Path) -> Path:
    """Infer/repair a batch-local YAML, then validate train/val image roots.

    The existing file is preserved as ``dataset.yaml.bak`` before an inferred
    schema replaces it. This is important for old batches whose labels were
    generated with a different keypoint count.
    """
    root = Path(batch_root).expanduser().resolve()
    path = root / 'dataset.yaml'
    try:
        payload = _load_dataset_yaml_strict(path) if path.is_file() else {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise DatasetPreparationError(f'dataset.yaml 无法读取: {exc}') from exc

    local_train = root / 'train_data' / 'images' / 'train'
    local_val = root / 'train_data' / 'images' / 'val'
    if not isinstance(payload, dict):
        raise DatasetPreparationError('dataset.yaml 根节点必须是 mapping')
    original_class_names = _normalize_class_names(payload.get('names'))

    if local_train.is_dir() and local_val.is_dir():
        inferred = _infer_batch_dataset_payload(root, payload)
        changed = inferred != payload
        payload = inferred
    else:
        changed = False
        configured_root = str(payload.get('path') or '')
        legacy_staging_path = '.preparing-' in configured_root
        if legacy_staging_path and local_train.is_dir() and local_val.is_dir():
            payload.pop('path', None)
            payload['train'] = 'train_data/images/train'
            payload['val'] = 'train_data/images/val'
            changed = True

    if not path.is_file() and not payload:
        raise DatasetPreparationError(f'dataset.yaml 不存在且无法从批次推断: {path}')
    if changed:
        if path.is_file():
            _backup_dataset_yaml(path)
        path.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding='utf-8',
        )
        if original_class_names != _normalize_class_names(payload.get('names')):
            _invalidate_ultralytics_label_cache(root)

    if not path.is_file():
        raise DatasetPreparationError(f'dataset.yaml 不存在: {path}')

    missing = []
    for split in ('train', 'val'):
        resolved = _resolve_dataset_split(path, payload, split)
        if resolved is None or not resolved.exists():
            missing.append(f'{split}: {resolved or "未配置"}')
    if missing:
        raise DatasetPreparationError(
            'dataset.yaml 中的图片路径无效: ' + '; '.join(missing)
        )
    return path


def prepare_existing_batch_split(batch_root: str | Path,
                                 val_ratio: float = 0.2,
                                 seed: int = 42) -> tuple[int, int]:
    """Create a deterministic split for a batch that only has top-level data.

    This operates inside the selected training batch and copies files into its
    ``train_data`` directory. The original ``images`` and ``labels`` remain
    untouched, which makes the operation reversible by deleting only the
    generated split directory.
    """
    root = Path(batch_root).expanduser().resolve()
    images = _list_images(root / 'images')
    labels = _list_suffix(root / 'labels', '.txt')
    missing = _missing_label_names(images, labels)
    if len(images) < 2:
        raise DatasetPreparationError('有效图片少于 2 个，无法划分 train/val')
    if missing:
        preview = '、'.join(missing[:8])
        raise DatasetPreparationError(
            f'顶层图片缺少 TXT 标签 {len(missing)} 个：{preview}'
        )
    if not 0.0 < float(val_ratio) < 1.0:
        raise DatasetPreparationError('验证集比例必须大于 0 且小于 1')

    train_dir = root / 'train_data' / 'images' / 'train'
    val_dir = root / 'train_data' / 'images' / 'val'
    existing_train = _list_images(train_dir)
    existing_val = _list_images(val_dir)
    if existing_train or existing_val:
        if existing_train and existing_val:
            return len(existing_train), len(existing_val)
        raise DatasetPreparationError(
            '当前批次只有部分 train/val 划分，请先在数据管理中整理后再继续'
        )

    label_lookup = {path.stem: path for path in labels}
    source_lookup = _load_batch_source_lookup(root)
    val_stems = _source_stratified_val_stems(
        [
            (
                source_lookup.get(path.stem, root.name),
                path.stem,
                label_lookup.get(path.stem),
            )
            for path in images
        ],
        float(val_ratio),
        int(seed),
    )
    val_count = len(val_stems)
    for image in images:
        split = 'val' if image.stem in val_stems else 'train'
        destination_image = root / 'train_data' / 'images' / split / image.name
        destination_label = root / 'train_data' / 'labels' / split / f'{image.stem}.txt'
        _materialize(image, destination_image, use_copy=True)
        _materialize(label_lookup[image.stem], destination_label, use_copy=True)
    return len(images) - val_count, val_count


def _source_stratified_val_stems(
    samples: Sequence[tuple[str, str, Path | None]],
    val_ratio: float,
    seed: int,
) -> set[str]:
    """Split each original source independently, then merge the results."""
    groups: dict[str, list[tuple[str, Path | None]]] = {}
    for source_name, stem, label_path in samples:
        groups.setdefault(source_name or '__batch__', []).append(
            (stem, label_path)
        )

    selected: set[str] = set()
    for index, source_name in enumerate(sorted(groups)):
        group = groups[source_name]
        if len(group) < 2:
            continue
        val_count = int(round(len(group) * float(val_ratio)))
        val_count = max(1, val_count)
        val_count = min(val_count, len(group) - 1)
        selected.update(
            _stratified_val_stems(group, val_count, int(seed) + index)
        )
    return selected


def _load_batch_source_lookup(root: Path) -> dict[str, str]:
    """Read source provenance when an existing batch has a manifest."""
    manifest = root / 'preparation_manifest.json'
    try:
        payload = json.loads(manifest.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    records = payload.get('records', []) if isinstance(payload, dict) else []
    if not isinstance(records, list):
        return {}
    return {
        str(row.get('stem')): str(row.get('source_name') or root.name)
        for row in records
        if isinstance(row, dict) and row.get('stem')
    }


def _stratified_val_stems(
    samples: Sequence[tuple[str, Path | None]],
    val_count: int,
    seed: int,
) -> set[str]:
    """Choose a deterministic validation split with class coverage.

    A plain random split can put every example of a rare class in train,
    making Ultralytics omit that class from validation metrics. Classes with
    at least two samples get one reserved validation sample first. Singleton
    classes stay in train whenever the requested validation size allows it.
    """
    if val_count <= 0 or not samples:
        return set()

    shuffled = list(samples)
    random.Random(int(seed)).shuffle(shuffled)
    val_count = min(int(val_count), len(shuffled) - 1)
    if val_count <= 0:
        return set()

    class_samples: dict[int, list[str]] = {}
    for stem, label_path in shuffled:
        if label_path is None:
            continue
        for class_id in set(_label_class_ids(label_path)):
            class_samples.setdefault(class_id, []).append(stem)

    selected: list[str] = []
    selected_set: set[str] = set()
    singleton_stems = {
        stem
        for class_stems in class_samples.values()
        if len(class_stems) == 1
        for stem in class_stems
    }

    # Reserve rare-but-repeatable classes first. Sorting by frequency makes
    # the guarantee useful for the smallest classes when val_count is tight.
    for _class_id, class_stems in sorted(
        class_samples.items(), key=lambda item: (len(item[1]), item[0])
    ):
        if len(class_stems) < 2 or len(selected) >= val_count:
            continue
        stem = next(
            (candidate for candidate in class_stems
             if candidate not in selected_set),
            None,
        )
        if stem is not None:
            selected.append(stem)
            selected_set.add(stem)

    # Fill the remainder randomly, keeping singleton-only examples in train
    # unless they are needed to reach the requested validation size.
    for allow_singletons in (False, True):
        for stem, _label_path in shuffled:
            if len(selected) >= val_count:
                break
            if stem in selected_set:
                continue
            if not allow_singletons and stem in singleton_stems:
                continue
            selected.append(stem)
            selected_set.add(stem)
        if len(selected) >= val_count:
            break

    return selected_set


def _validate_simple_name(value: str, field_name: str):
    value = str(value or '').strip()
    path = Path(value)
    if not value or path.is_absolute() or len(path.parts) != 1 or value in {'.', '..'}:
        raise DatasetPreparationError(f'{field_name}必须是单个目录名称')


def _find_source_peer(root: Path, peer_name: str,
                      image_dir: Path) -> Path | None:
    standard = root / peer_name / image_dir.name
    if standard.is_dir():
        return standard
    nested = image_dir / peer_name
    if nested.is_dir():
        return nested
    return None


def _find_reusable_split(request: DatasetPreparationRequest) -> dict | None:
    """Find an existing task-agnostic split matching sources/ratios/seed."""
    training_root = request.dataset_root / 'training_data'
    if not training_root.is_dir():
        return None
    for manifest in sorted(training_root.glob('*/split_manifest.json')):
        try:
            data = json.loads(manifest.read_text(encoding='utf-8'))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        if set(data.get('source_batches', ())) != set(request.source_names):
            continue
        if abs(float(data.get('val_ratio', -1)) - float(request.val_ratio)) > 1e-9:
            continue
        if abs(float(data.get('test_ratio', -1)) - float(request.test_ratio)) > 1e-9:
            continue
        if int(data.get('seed', -1)) != int(request.seed):
            continue
        split = data.get('split')
        if not isinstance(split, dict) or not split:
            continue
        resolved = {
            str(stem): str(value)
            for stem, value in split.items()
            if value in ('train', 'val', 'test')
        }
        if len(resolved) != len(split):
            continue
        resolved['_manifest_path'] = str(manifest)
        return resolved
    return None


def _load_test_stems(test_root: Path) -> set[str]:
    stems = set()
    images = test_root / 'images'
    if images.is_dir():
        stems.update(path.stem for path in images.rglob('*') if _is_image(path))
    test_list = test_root / TEST_LIST_FILE
    if test_list.is_file():
        try:
            for line in test_list.read_text(encoding='utf-8').splitlines():
                stem = line.split('#', 1)[0].strip()
                if stem:
                    stems.add(stem)
        except OSError:
            pass
    return stems


def _create_output_tree(root: Path, annotation_dir: str):
    for relative in (
        'images', annotation_dir, 'labels',
        'train_data/images/train', 'train_data/images/val',
        'train_data/labels/train', 'train_data/labels/val',
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)


def _materialize(source: Path, destination: Path, use_copy: bool):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if use_copy:
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError as exc:
        if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES}:
            raise
        shutil.copy2(source, destination)


def _write_dataset_yaml(path: Path, request: DatasetPreparationRequest,
                        samples: Sequence[DatasetSample]):
    label_paths = [
        sample.label_path for sample in samples if sample.label_path is not None
    ]
    annotations = {sample.stem: sample.annotation_path for sample in samples}
    class_names = _resolved_class_names(
        request.class_names, label_paths, annotations,
    )
    payload = {
        'train': 'train_data/images/train',
        'val': 'train_data/images/val',
        'names': class_names,
    }
    if request.task_type == 'pose':
        columns = _observed_label_columns(
            [sample.label_path for sample in samples if sample.label_path],
            request.task_type,
        )
        schema = _schema_from_columns(columns, request.task_type)
        keypoint_count, dimensions = schema or (
            (len(request.keypoints), 3) if request.keypoints else (0, 0)
        )
        if keypoint_count:
            keypoint_names = _infer_keypoint_names_from_samples(
                samples, request, keypoint_count
            )
            payload['kpt_shape'] = [keypoint_count, dimensions]
            payload['keypoint_names'] = keypoint_names
            payload['flip_idx'] = _pose_flip_indices(
                keypoint_names,
                infer_left_right_pairs(keypoint_names),
            )
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding='utf-8',
    )


def _resolve_dataset_split(yaml_path: Path, payload: dict,
                           split: str) -> Path | None:
    value = payload.get(split)
    if not isinstance(value, str) or not value.strip():
        return None
    split_path = Path(value).expanduser()
    if split_path.is_absolute():
        return split_path.resolve()
    root_value = payload.get('path')
    if root_value:
        dataset_root = Path(str(root_value)).expanduser()
        if not dataset_root.is_absolute():
            dataset_root = yaml_path.parent / dataset_root
    else:
        dataset_root = yaml_path.parent
    return (dataset_root / split_path).resolve()


def _write_manifest(path: Path, request: DatasetPreparationRequest,
                    scan: DatasetScanResult, split_by_stem: dict[str, str],
                    train_count: int, val_count: int,
                    test_batch: str = '', test_ratio: float = 0.0,
                    test_count: int = 0):
    request_data = asdict(request)
    request_data['dataset_root'] = str(request.dataset_root)
    records = []
    for sample in scan.samples:
        records.append({
            'stem': sample.stem,
            'source_name': sample.source_name,
            'image_path': str(sample.image_path),
            'annotation_path': str(sample.annotation_path),
            'label_path': str(sample.label_path) if sample.label_path else None,
            'generated_empty_label': sample.label_path is None,
            'split': split_by_stem.get(sample.stem, 'test'),
        })
    payload = {
        'version': 1,
        'created_at': _utc_now(),
        'request': request_data,
        'summary': {
            'total': len(scan.samples),
            'train': train_count,
            'val': val_count,
            'test': test_count,
            'test_batch': test_batch,
            'test_ratio': test_ratio,
            'test_excluded': len(scan.test_excluded),
            'duplicates': len(scan.duplicate_images),
            'skipped_missing_annotations': len(scan.missing_annotations),
            'skipped_missing_labels': len(scan.missing_labels),
        },
        'records': records,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


def _create_test_output_tree(root: Path, annotation_dir: str):
    for relative in ('images', annotation_dir, 'labels'):
        (root / relative).mkdir(parents=True, exist_ok=True)


def _write_test_dataset_yaml(path: Path, request: DatasetPreparationRequest,
                             samples: Sequence[DatasetSample]):
    """Write the dataset.yaml for an evaluation test batch.

    ``train`` and ``val`` both point at the single flat ``images`` directory
    so the text file can be consumed by Ultralytics-style evaluation flows
    (the evaluation center will drive the actual metric computation).
    """
    label_paths = [
        sample.label_path for sample in samples if sample.label_path is not None
    ]
    annotations = {sample.stem: sample.annotation_path for sample in samples}
    class_names = _resolved_class_names(
        request.class_names, label_paths, annotations,
    )
    payload = {
        'train': 'images',
        'val': 'images',
        'names': class_names,
    }
    if request.task_type == 'pose':
        columns = _observed_label_columns(
            [sample.label_path for sample in samples if sample.label_path],
            request.task_type,
        )
        schema = _schema_from_columns(columns, request.task_type)
        keypoint_count, dimensions = schema or (
            (len(request.keypoints), 3) if request.keypoints else (0, 0)
        )
        if keypoint_count:
            keypoint_names = _infer_keypoint_names_from_samples(
                samples, request, keypoint_count
            )
            payload['kpt_shape'] = [keypoint_count, dimensions]
            payload['keypoint_names'] = keypoint_names
            payload['flip_idx'] = _pose_flip_indices(
                keypoint_names,
                infer_left_right_pairs(keypoint_names),
            )
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding='utf-8',
    )


def _write_test_manifest(path: Path, request: DatasetPreparationRequest,
                         scan: DatasetScanResult,
                         samples: Sequence[DatasetSample],
                         training_batch: str):
    records = []
    for sample in samples:
        records.append({
            'stem': sample.stem,
            'source_name': sample.source_name,
            'image_path': str(sample.image_path),
            'annotation_path': str(sample.annotation_path),
            'label_path': str(sample.label_path) if sample.label_path else None,
            'generated_empty_label': sample.label_path is None,
        })
    payload = {
        'version': 1,
        'created_at': _utc_now(),
        'task_type': request.task_type,
        'annotation_dir': request.annotation_dir,
        'label_dir': request.label_dir,
        'training_batch': training_batch,
        'test_ratio': request.test_ratio,
        'seed': request.seed,
        'use_copy': request.use_copy,
        'source_batches': list(request.source_names),
        'summary': {'total': len(samples)},
        'records': records,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


def _pose_flip_indices(keypoints: Sequence[str],
                       pairs: Iterable[tuple[str, str]]) -> list[int]:
    indices = list(range(len(keypoints)))
    lookup = {name: index for index, name in enumerate(keypoints)}
    for left, right in pairs:
        left_index = lookup.get(left)
        right_index = lookup.get(right)
        if left_index is None or right_index is None:
            continue
        indices[left_index] = right_index
        indices[right_index] = left_index
    return indices


def _resolved_class_names(
    configured: object,
    label_paths: Sequence[Path],
    annotations: dict[str, Path],
) -> dict[int, str]:
    """Resolve a contiguous class map from config and observed batch data."""
    configured_names = _normalize_class_names(configured)
    observed_ids = set()
    votes: dict[int, Counter] = {}
    annotation_names: list[str] = []

    for path in label_paths:
        class_ids = _label_class_ids(path)
        observed_ids.update(class_ids)
        annotation_path = annotations.get(path.stem)
        rectangle_names = (
            _annotation_rectangle_names(annotation_path)
            if annotation_path is not None else []
        )
        for name in rectangle_names:
            if name not in annotation_names:
                annotation_names.append(name)
        if len(class_ids) != len(rectangle_names):
            continue
        for class_id, name in zip(class_ids, rectangle_names):
            votes.setdefault(class_id, Counter())[name] += 1

    if observed_ids:
        highest_index = max(observed_ids)
    elif configured_names:
        highest_index = max(configured_names)
    else:
        return {}

    names: dict[int, str] = {}
    used_names: set[str] = set()
    for class_id in range(highest_index + 1):
        candidate = ''
        if class_id in votes:
            for value, _count in votes[class_id].most_common():
                if value not in used_names:
                    candidate = value
                    break
        if not candidate:
            configured_name = configured_names.get(class_id, '')
            if configured_name and configured_name not in used_names:
                candidate = configured_name
        if not candidate:
            candidate = next(
                (value for value in annotation_names if value not in used_names),
                '',
            )
        candidate = candidate or f'class_{class_id}'
        names[class_id] = candidate
        used_names.add(candidate)
    return {index: names[index] for index in range(highest_index + 1)}


def _normalize_class_names(value: object) -> dict[int, str]:
    names: dict[int, str] = {}
    items = value.items() if isinstance(value, dict) else enumerate(
        value if isinstance(value, (list, tuple)) else ()
    )
    for raw_index, raw_name in items:
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            continue
        name = str(raw_name or '').strip()
        if index >= 0 and name:
            names[index] = name
    return names


def _label_class_ids(path: Path) -> list[int]:
    class_ids = []
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except (OSError, UnicodeError):
        return class_ids
    for line in lines:
        fields = line.strip().split()
        if not fields:
            continue
        try:
            value = float(fields[0])
            class_id = int(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if value == class_id and class_id >= 0:
            class_ids.append(class_id)
    return class_ids


def _label_class_counts(paths: Sequence[Path]) -> dict[int, int]:
    counts: Counter = Counter()
    for path in paths:
        counts.update(_label_class_ids(path))
    return dict(sorted(counts.items()))


def _annotation_rectangle_names(path: Path) -> list[str]:
    try:
        document = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return []
    shapes = document.get('shapes', []) if isinstance(document, dict) else []
    return [
        str(shape.get('label') or '').strip()
        for shape in shapes
        if (
            isinstance(shape, dict)
            and shape.get('shape_type') == 'rectangle'
            and str(shape.get('label') or '').strip()
        )
    ]


def _missing_label_names(images: Sequence[Path],
                         labels: Sequence[Path]) -> tuple[str, ...]:
    label_stems = {path.stem for path in labels}
    return tuple(path.name for path in images if path.stem not in label_stems)


def _load_dataset_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, yaml.YAMLError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_dataset_yaml_strict(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise yaml.YAMLError('根节点必须是 mapping')
    return payload


def _infer_batch_dataset_payload(root: Path, existing: dict) -> dict:
    """Build a batch-local YAML from the labels and batch metadata."""
    payload = dict(existing)
    local_train = root / 'train_data' / 'images' / 'train'
    local_val = root / 'train_data' / 'images' / 'val'
    if local_train.is_dir() and local_val.is_dir():
        payload.pop('path', None)
        payload['train'] = 'train_data/images/train'
        payload['val'] = 'train_data/images/val'

    metadata = _load_batch_metadata(root)
    task_type = str(metadata.get('task_type') or 'pose')
    payload['names'] = _batch_class_names(root, payload, metadata)

    if task_type == 'pose':
        labels = _list_suffix(root / 'train_data' / 'labels' / 'train', '.txt')
        labels += _list_suffix(root / 'train_data' / 'labels' / 'val', '.txt')
        if not labels:
            labels = _list_suffix(root / 'labels', '.txt')
        observed = _observed_label_columns(labels, task_type)
        schema = _schema_from_columns(observed, task_type)
        if schema:
            keypoint_count, dimensions = schema
            names = _batch_keypoint_names(
                root, payload, metadata, keypoint_count
            )
            payload['kpt_shape'] = [keypoint_count, dimensions]
            payload['keypoint_names'] = names
            pairs = infer_left_right_pairs(names)
            payload['flip_idx'] = _pose_flip_indices(names, pairs)
    return payload


def _backup_dataset_yaml(path: Path):
    backup = path.with_name(path.name + '.bak')
    if backup.exists():
        index = 1
        while path.with_name(f'{path.name}.bak.{index}').exists():
            index += 1
        backup = path.with_name(f'{path.name}.bak.{index}')
    shutil.copy2(path, backup)


def _invalidate_ultralytics_label_cache(root: Path):
    for path in (
        root / 'labels.cache',
        root / 'train_data' / 'labels' / 'train.cache',
        root / 'train_data' / 'labels' / 'val.cache',
    ):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _batch_class_names(root: Path, payload: dict, metadata: dict):
    configured = payload.get('names') or metadata.get('class_names') or ()
    labels = _list_suffix(root / 'labels', '.txt')
    if not labels:
        labels = _list_suffix(
            root / 'train_data' / 'labels' / 'train', '.txt'
        )
        labels += _list_suffix(
            root / 'train_data' / 'labels' / 'val', '.txt'
        )
    annotation_dir = (
        root / str(metadata.get('annotation_dir') or 'annotations')
    )
    annotations = {
        path.stem: path for path in sorted(annotation_dir.glob('*.json'))
    }
    return _resolved_class_names(configured, labels, annotations)


def _batch_keypoint_names(root: Path, payload: dict, metadata: dict,
                          count: int) -> list[str]:
    configured = payload.get('keypoint_names')
    if isinstance(configured, (list, tuple)) and len(configured) == count:
        return [str(value) for value in configured]
    annotation_dir = root / str(metadata.get('annotation_dir') or 'annotations')
    names = list(infer_annotation_schema(
        sorted(annotation_dir.glob('*.json')),
        task_type='pose',
    ).keypoints)
    if len(names) == count:
        return names
    raise DatasetPreparationError(
        f'无法从批次 JSON 推断 {count} 个关键点名称，实际发现 {len(names)} 个'
    )


def _annotation_point_names(annotation_dir: Path) -> list[str]:
    names = []
    if not annotation_dir.is_dir():
        return names
    for path in sorted(annotation_dir.glob('*.json')):
        try:
            document = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        shapes = document.get('shapes', []) if isinstance(document, dict) else []
        for shape in shapes:
            if not isinstance(shape, dict) or shape.get('shape_type') != 'point':
                continue
            label = str(shape.get('label') or '').strip()
            if label and label not in names:
                names.append(label)
    return names


def _batch_keypoint_schema(labels: Sequence[Path], task_type: str):
    return _schema_from_columns(
        _observed_label_columns(labels, task_type), task_type
    )


def _dataset_keypoint_shape(payload: dict, task_type: str) -> tuple[int, int]:
    if task_type != 'pose':
        return ()
    value = payload.get('kpt_shape')
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return ()
    try:
        count, dimensions = int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return ()
    if count <= 0 or dimensions not in {2, 3}:
        return ()
    return count, dimensions


def _expected_pose_label_columns(task_type: str, keypoint_count: int,
                                 dimensions: int) -> int:
    if task_type != 'pose' or keypoint_count <= 0 or dimensions <= 0:
        return 0
    return 5 + keypoint_count * dimensions


def _schema_from_columns(columns: Iterable[int], task_type: str) -> tuple[int, int]:
    if task_type != 'pose':
        return ()
    values = sorted(set(int(value) for value in columns if int(value) >= 5))
    if not values:
        return ()
    # Ultralytics Pose labels are bbox(5) + keypoints * (x, y, visibility).
    # Accept 2D labels as well because some imported datasets omit visibility.
    value = values[0]
    remainder = value - 5
    if remainder > 0 and remainder % 3 == 0:
        return remainder // 3, 3
    if remainder > 0 and remainder % 2 == 0:
        return remainder // 2, 2
    return ()


def _observed_label_columns(labels: Sequence[Path], task_type: str) -> set[int]:
    observed = set()
    for path in labels:
        _valid, columns = _inspect_label_structure(path, task_type)
        observed.update(columns)
    return observed


def _inspect_label_structure(path: Path, task_type: str
                             ) -> tuple[bool, set[int]]:
    """Validate numeric label rows without assuming a template keypoint count."""
    observed = set()
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except (OSError, UnicodeError):
        return False, observed
    valid = True
    for line in lines:
        fields = line.strip().split()
        if not fields:
            continue
        try:
            values = tuple(float(value) for value in fields)
        except ValueError:
            valid = False
            continue
        if not _label_values_are_valid(values, task_type):
            valid = False
            continue
        if task_type == 'pose':
            if len(fields) == 5:
                continue
            schema = _schema_from_columns((len(fields),), task_type)
            if not schema:
                valid = False
                continue
        observed.add(len(fields))
    if len(observed) > 1:
        valid = False
    return valid, observed


def _inspect_label_files(labels: Sequence[Path], expected_columns: int
                         ) -> tuple[tuple[str, ...], set[int]]:
    if not expected_columns:
        return (), set()
    invalid = []
    observed = set()
    for path in labels:
        valid, columns = _inspect_label_file(path, expected_columns)
        observed.update(columns)
        if not valid:
            invalid.append(path.name)
    return tuple(invalid), observed


def _inspect_label_file(path: Path, expected_columns: int
                        ) -> tuple[bool, set[int]]:
    observed = set()
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except (OSError, UnicodeError):
        return False, observed
    valid = True
    for line in lines:
        fields = line.strip().split()
        if not fields:
            continue
        observed.add(len(fields))
        if len(fields) != expected_columns:
            valid = False
            continue
        try:
            values = tuple(float(value) for value in fields)
        except ValueError:
            valid = False
            continue
        if not _label_values_are_valid(
            values, 'pose' if expected_columns > 5 else ''
        ):
            valid = False
    return valid, observed


def _label_values_are_valid(values: Sequence[float], task_type: str) -> bool:
    """Reject values that Ultralytics cannot safely train or plot."""
    if not values or not all(math.isfinite(value) for value in values):
        return False
    class_id = values[0]
    if class_id < 0 or not class_id.is_integer():
        return False
    if len(values) < 5:
        return False
    x_center, y_center, width, height = values[1:5]
    if not (
        0.0 <= x_center <= 1.0
        and 0.0 <= y_center <= 1.0
        and 0.0 < width <= 1.0
        and 0.0 < height <= 1.0
    ):
        return False
    if task_type != 'pose' or len(values) == 5:
        return True

    schema = _schema_from_columns((len(values),), 'pose')
    if not schema:
        return False
    keypoint_count, dimensions = schema
    for index in range(keypoint_count):
        offset = 5 + index * dimensions
        x_coord, y_coord = values[offset:offset + 2]
        if not (0.0 <= x_coord <= 1.0 and 0.0 <= y_coord <= 1.0):
            return False
        if dimensions == 3 and values[offset + 2] not in (0.0, 1.0, 2.0):
            return False
    return True


def _infer_keypoint_names_from_samples(
    samples: Sequence[DatasetSample], request: DatasetPreparationRequest,
    count: int,
) -> list[str]:
    names = list(infer_annotation_schema(
        [sample.annotation_path for sample in samples],
        task_type='pose',
    ).keypoints)
    if len(names) == count:
        return names
    raise DatasetPreparationError(
        f'无法从标注 JSON 推断 {count} 个关键点名称，实际发现 {len(names)} 个'
    )


def _annotation_point_names_for_samples(
    samples: Sequence[DatasetSample],
) -> list[str]:
    names = []
    for sample in samples:
        try:
            document = json.loads(
                sample.annotation_path.read_text(encoding='utf-8')
            )
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        shapes = document.get('shapes', []) if isinstance(document, dict) else []
        for shape in shapes:
            if not isinstance(shape, dict) or shape.get('shape_type') != 'point':
                continue
            label = str(shape.get('label') or '').strip()
            if label and label not in names:
                names.append(label)
    return names


def _load_batch_metadata(root: Path) -> dict:
    manifest = root / 'preparation_manifest.json'
    if not manifest.is_file():
        return {}
    try:
        payload = json.loads(manifest.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    request = payload.get('request') if isinstance(payload, dict) else None
    return request if isinstance(request, dict) else {}


def _list_images(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(item for item in path.iterdir() if _is_image(item))


def _list_suffix(path: Path, suffix: str) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(
        item for item in path.iterdir()
        if item.is_file() and item.suffix.lower() == suffix
    )


def _count_images(path: Path) -> int:
    return len(_list_images(path))


def _count_suffix(path: Path | None, suffix: str) -> int:
    return len(_list_suffix(path, suffix)) if path is not None else 0


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _path_preview(paths: Sequence[Path], limit: int) -> str:
    preview = '\n'.join(str(path) for path in paths[:limit])
    remaining = len(paths) - limit
    if remaining > 0:
        preview += f'\n... 还有 {remaining} 个'
    return preview


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')
