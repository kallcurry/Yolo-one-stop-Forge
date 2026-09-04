"""X-AnyLabeling JSON <-> YOLO TXT conversion and consistency validation.

Conversion rules mirror X-AnyLabeling's official exporter
(``anylabeling/views/labeling/label_converter.py``) line by line, so the
labels regenerated here are equivalent to what the label tool itself would
export: clamped points, diagonal-corner rectangles, ``int()`` pose keypoints
with 6-decimal rounding, visibility from ``difficult``, zero-filled missing
keypoints, and four-corner OBB output.

The label configuration file (an X-AnyLabeling auto-labeling YAML, e.g.
``yolov8m_pose_boyuan.yaml``) is the authoritative source for class ids and
keypoint order.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from app.models.app_defaults import image_extensions as _image_extensions

IMAGE_EXTENSIONS = _image_extensions()

TASK_FROM_TYPE = {
    'pose': 'pose',
    'detect': 'detection',
    'seg': 'segmentation',
    'obb': 'obb',
}


@dataclass(frozen=True)
class LabelConfig:
    """Parsed X-AnyLabeling label configuration.

    For pose the keypoint order per class lives in ``pose_keypoints``; every
    other task only needs the ordered ``classes`` tuple.
    """

    task_type: str
    has_visible: bool = True
    classes: tuple[str, ...] = ()
    pose_keypoints: tuple[tuple[str, tuple[str, ...]], ...] = ()
    path: Path | None = None
    source_name: str = ''

    @property
    def max_keypoints(self) -> int:
        return max((len(kpts) for _name, kpts in self.pose_keypoints), default=0)

    def keypoints_for(self, class_name: str) -> tuple[str, ...]:
        for name, kpts in self.pose_keypoints:
            if name == class_name:
                return kpts
        return ()

    def class_index(self, class_name: str) -> int | None:
        try:
            return self.classes.index(class_name)
        except ValueError:
            return None

    def expected_columns(self) -> int | None:
        """Fixed column count per row, or None for variable-length formats."""
        if self.task_type == 'detection':
            return 5
        if self.task_type == 'pose':
            dim = 3 if self.has_visible else 2
            return 5 + self.max_keypoints * dim
        if self.task_type == 'obb':
            return 9
        return None


def parse_label_config(path: str | Path) -> LabelConfig:
    """Parse an X-AnyLabeling label config YAML into :class:`LabelConfig`."""
    config_path = Path(path).expanduser()
    try:
        data = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    except OSError as exc:
        raise ValueError(f'无法读取配置: {exc}') from exc
    except yaml.YAMLError as exc:
        raise ValueError(f'配置不是有效 YAML: {exc}') from exc
    if not isinstance(data, dict):
        raise ValueError('配置根节点必须是 YAML object')

    type_text = str(data.get('type') or '').strip().lower()
    task_type = ''
    for keyword, task in TASK_FROM_TYPE.items():
        if keyword in type_text:
            task_type = task
            break
    if not task_type:
        raise ValueError(
            f'无法从 type={type_text!r} 推断任务类型（预期 pose/detect/seg/obb）'
        )

    has_visible = bool(data.get('has_visible', True))
    raw_classes = data.get('classes')
    classes: tuple[str, ...] = ()
    pose_keypoints: tuple[tuple[str, tuple[str, ...]], ...] = ()
    if isinstance(raw_classes, dict):
        items = []
        for class_name, kpt_names in raw_classes.items():
            if isinstance(kpt_names, list):
                items.append((str(class_name), tuple(str(k) for k in kpt_names)))
        if not items:
            raise ValueError('配置中缺少有效的 classes 关键点定义')
        classes = tuple(name for name, _kpts in items)
        pose_keypoints = tuple(items)
    elif isinstance(raw_classes, list):
        if not raw_classes or not all(isinstance(c, str) for c in raw_classes):
            raise ValueError('配置 classes 必须是字符串列表')
        classes = tuple(str(c) for c in raw_classes)
    else:
        raise ValueError('配置中缺少 classes 定义')

    if task_type != 'pose' and not classes:
        raise ValueError('配置中缺少 classes 类别定义')
    return LabelConfig(
        task_type=task_type,
        has_visible=has_visible,
        classes=classes,
        pose_keypoints=pose_keypoints,
        path=config_path.resolve(),
        source_name=config_path.name,
    )


# --- helpers mirroring the official converter ---

def clamp_points(points: list, image_width: float, image_height: float):
    """Clamp points to the image boundaries (official: size - 1)."""
    clamped = []
    for point in points:
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError, IndexError):
            continue
        clamped.append([
            max(0, min(x, image_width - 1)),
            max(0, min(y, image_height - 1)),
        ])
    return clamped


def rectangle_from_diagonal(diagonal: list) -> list:
    """Two diagonal points -> [tl, tr, br, bl] (official order)."""
    (x1, y1), (x2, y2) = diagonal[0], diagonal[1]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _r6(value: float) -> str:
    """Official pose rounding: round(..., 6) printed without padding."""
    return str(round(value, 6))


def _norm(value: float, size: float) -> float:
    return value / size if size else 0.0


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


# --- JSON -> YOLO TXT ---

def json_to_yolo_lines(
    data: dict,
    config: LabelConfig,
) -> tuple[list[str], list[str]]:
    """Convert one X-AnyLabeling JSON to YOLO TXT lines.

    Returns ``(lines, issues)``; issues never raise for a single bad shape
    (except a missing image size, which makes conversion impossible).
    """
    lines: list[str] = []
    issues: list[str] = []

    width = data.get('imageWidth')
    height = data.get('imageHeight')
    if width is None or height is None or width <= 0 or height <= 0:
        return [], ['JSON 缺少有效的 imageWidth/imageHeight']

    shapes = data.get('shapes') or []
    if config.task_type == 'pose':
        lines = _pose_lines(shapes, config, float(width), float(height), issues)
    elif config.task_type == 'detection':
        lines = _detection_lines(shapes, config, float(width), float(height), issues)
    elif config.task_type == 'segmentation':
        lines = _segmentation_lines(shapes, config, float(width), float(height), issues)
    elif config.task_type == 'obb':
        lines = _obb_lines(shapes, config, float(width), float(height), issues)
    return lines, issues


def _detection_lines(shapes, config, width, height, issues) -> list[str]:
    lines = []
    for shape in shapes:
        if not isinstance(shape, dict) or shape.get('shape_type') != 'rectangle':
            continue
        label = str(shape.get('label') or '').strip()
        class_index = config.class_index(label)
        if class_index is None:
            issues.append(f'跳过未知类别 rectangle: {label!r}')
            continue
        points = clamp_points(shape.get('points') or [], width, height)
        if len(points) == 2:
            points = rectangle_from_diagonal(points)
        if len(points) < 2:
            issues.append(f'无效矩形 {label!r}: 坐标不足')
            continue
        x1, y1 = points[0]
        x2, y2 = points[1] if len(points) == 2 else points[2]
        lines.append(
            f'{class_index} {_norm(x1 + x2, 2 * width):.6f} '
            f'{_norm(y1 + y2, 2 * height):.6f} '
            f'{abs(x2 - x1) / width:.6f} {abs(y2 - y1) / height:.6f}'
        )
    return lines


def _segmentation_lines(shapes, config, width, height, issues) -> list[str]:
    lines = []
    for shape in shapes:
        if not isinstance(shape, dict) or shape.get('shape_type') != 'polygon':
            continue
        label = str(shape.get('label') or '').strip()
        class_index = config.class_index(label)
        if class_index is None:
            issues.append(f'跳过未知类别 polygon: {label!r}')
            continue
        points = clamp_points(shape.get('points') or [], width, height)
        if len(points) < 3:
            issues.append(f'多边形 {label!r}: 点数不足 3')
            continue
        parts = [str(class_index)]
        for x, y in points:
            parts.append(str(_norm(x, width)))
            parts.append(str(_norm(y, height)))
        lines.append(' '.join(parts))
    return lines


def _obb_lines(shapes, config, width, height, issues) -> list[str]:
    lines = []
    for shape in shapes:
        if not isinstance(shape, dict) or shape.get('shape_type') != 'rotation':
            continue
        label = str(shape.get('label') or '').strip()
        class_index = config.class_index(label)
        if class_index is None:
            issues.append(f'跳过未知类别 rotation: {label!r}')
            continue
        points = shape.get('points') or []
        if len(points) != 4:
            issues.append(f'旋转框 {label!r}: 顶点数不是 4')
            continue
        try:
            flat = [float(v) for point in points for v in point]
        except (TypeError, ValueError, IndexError):
            issues.append(f'旋转框 {label!r}: 坐标无效')
            continue
        corners = [(flat[2 * i], flat[2 * i + 1]) for i in range(4)]
        out_of_bounds = any(
            cx < 0 or cx > width or cy < 0 or cy > height
            for cx, cy in corners
        )
        if out_of_bounds:
            # 轻微越界（越界距离 ≤5% 图幅）clamp 到边界后继续；
            # 严重越界或 clamp 后退化才跳过——避免边界 1-2px 的框
            # 被丢弃成背景（ultralytics 视空文件为背景，损失 GT）。
            max_side = max(width, height, 1)
            margin = 0.05 * max_side
            severe = any(
                cx < -margin or cx > width + margin
                or cy < -margin or cy > height + margin
                for cx, cy in corners
            )
            if severe:
                issues.append(f'旋转框 {label!r}: 越界超过 5% 图幅，已跳过')
                continue
            clamped = [
                (min(max(cx, 0), width), min(max(cy, 0), height))
                for cx, cy in corners
            ]
            if _polygon_area(clamped) < 0.5:
                issues.append(f'旋转框 {label!r}: 越界且退化，已跳过')
                continue
            corners = clamped
            flat = [value for corner in corners for value in corner]
            issues.append(f'旋转框 {label!r}: 轻微越界，已按边界裁剪')
        normalized = [
            flat[i] / width if i % 2 == 0 else flat[i] / height
            for i in range(8)
        ]
        lines.append(f'{class_index} ' + ' '.join(str(v) for v in normalized))
    return lines


def _pose_lines(shapes, config, width, height, issues) -> list[str]:
    groups: dict[int, dict] = {}
    order: list[int] = []
    for shape_idx, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            continue
        shape_type = shape.get('shape_type')
        if shape_type not in ('rectangle', 'point'):
            continue
        raw_group = shape.get('group_id')
        if raw_group is None:
            issues.append(f'shape[{shape_idx}] group_id 为空，跳过')
            continue
        try:
            group_id = int(raw_group)
        except (TypeError, ValueError):
            issues.append(f'shape[{shape_idx}] group_id 无效: {raw_group!r}')
            continue
        if group_id not in groups:
            groups[group_id] = {'rectangle': None, 'box_label': '', 'keypoints': {}}
            order.append(group_id)
        points = clamp_points(shape.get('points') or [], width, height)
        if shape_type == 'rectangle':
            if len(points) == 2:
                points = rectangle_from_diagonal(points)
            if len(points) < 2:
                issues.append(f'group_id={group_id} 矩形坐标不足')
                continue
            groups[group_id]['rectangle'] = points
            groups[group_id]['box_label'] = str(shape.get('label') or '').strip()
        else:
            label = str(shape.get('label') or '').strip()
            if not points:
                continue
            x, y = points[0]
            difficult = shape.get('difficult', False)
            visibility = 1 if difficult is True else 2
            groups[group_id]['keypoints'][label] = (x, y, visibility)

    max_k = config.max_keypoints
    dim = 3 if config.has_visible else 2
    lines = []
    for group_id in order:
        data = groups[group_id]
        box_label = data['box_label']
        rectangle = data['rectangle']
        if not box_label or rectangle is None:
            issues.append(f'group_id={group_id} 缺少人物框，跳过')
            continue
        class_index = config.class_index(box_label)
        if class_index is None:
            issues.append(f'group_id={group_id} 未知人物类别 {box_label!r}，跳过')
            continue
        kpt_names = config.keypoints_for(box_label)
        if len(points := rectangle) >= 2:
            x1, y1 = rectangle[0]
            x2, y2 = rectangle[1] if len(rectangle) == 2 else rectangle[2]
            row_parts = [
                str(class_index),
                _r6(_norm(x1 + x2, 2 * width)),
                _r6(_norm(y1 + y2, 2 * height)),
                _r6(abs(x2 - x1) / width),
                _r6(abs(y2 - y1) / height),
            ]
        keypoints = data['keypoints']
        for name in kpt_names:
            if name not in keypoints:
                row_parts.extend(['0', '0', '0'] if config.has_visible else ['0', '0'])
            else:
                x, y, visibility = keypoints[name]
                row_parts.append(_r6((int(x)) / width))
                row_parts.append(_r6((int(y)) / height))
                if config.has_visible:
                    row_parts.append(str(visibility))
        for _ in range(max_k - len(kpt_names)):
            row_parts.extend(['0', '0', '0'] if config.has_visible else ['0', '0'])
        lines.append(' '.join(row_parts))
    return lines


# --- TXT parsing / validation ---

def parse_txt_rows(path: str | Path) -> tuple[list[list[float]], list[str]]:
    """Parse a YOLO TXT into float rows; non-finite values are reported."""
    rows: list[list[float]] = []
    errors: list[str] = []
    txt_path = Path(path)
    try:
        text = txt_path.read_text(encoding='utf-8')
    except OSError as exc:
        return [], [f'无法读取 {txt_path}: {exc}']
    for line_no, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        try:
            values = [float(value) for value in stripped.split()]
        except ValueError as exc:
            errors.append(f'第{line_no}行包含非数值: {exc}')
            continue
        if not _is_finite(values[0]):
            errors.append(f'第{line_no}行 class id 不是有限数值')
            continue
        rows.append(values)
    return rows, errors


@dataclass(frozen=True)
class FileValidation:
    """Outcome of validating one JSON/TXT pair."""

    json_path: Path
    txt_path: Path | None
    status: str            # ok | missing-txt | structure | mismatch | error
    issues: tuple[str, ...] = ()
    diffs: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status == 'ok'


def validate_txt_against_json(
    json_path: str | Path,
    txt_path: str | Path | None,
    config: LabelConfig,
    tolerance: float = 1e-4,
) -> FileValidation:
    """Compare an existing YOLO TXT with the JSON it should represent."""
    json_file = Path(json_path)
    if txt_path is None:
        return FileValidation(json_file, None, 'missing-txt', ('缺少对应 TXT 文件',))
    txt_file = Path(txt_path)

    try:
        data = json.loads(json_file.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        return FileValidation(json_file, txt_file, 'error', (f'JSON 读取失败: {exc}',))

    expected_lines, convert_issues = json_to_yolo_lines(data, config)
    actual_rows, parse_errors = parse_txt_rows(txt_file)
    issues = list(parse_errors) + list(convert_issues)

    expected_columns = config.expected_columns()
    if not parse_errors:
        for idx, row in enumerate(actual_rows, 1):
            if expected_columns is not None and len(row) != expected_columns:
                issues.append(
                    f'第{idx}行列数 {len(row)}，期望 {expected_columns}'
                )
            elif config.task_type == 'segmentation' and (len(row) < 3 or len(row) % 2 == 0):
                issues.append(
                    f'第{idx}行多边形列数异常: {len(row)}'
                )
            if not 0 <= row[0] < len(config.classes):
                issues.append(
                    f'第{idx}行 class id {int(row[0])} 超出配置类别范围'
                )
            if not all(_is_finite(v) for v in row):
                issues.append(f'第{idx}行存在非有限数值')

    diffs: list[str] = []
    if issues:
        return FileValidation(json_file, txt_file, 'structure', tuple(issues))
    if len(expected_lines) != len(actual_rows):
        diffs.append(
            f'行数不一致: 期望 {len(expected_lines)} 行，实际 {len(actual_rows)} 行'
        )
    for idx, (expected_line, actual_row) in enumerate(
        zip(expected_lines, actual_rows), 1
    ):
        expected_values = [float(v) for v in expected_line.split()]
        if len(expected_values) != len(actual_row):
            diffs.append(f'第{idx}行结构不一致，期望 {len(expected_values)} 项，实际 {len(actual_row)} 项')
            continue
        bad = [
            (name, exp, act)
            for name, exp, act in zip(
                ('class', 'cx', 'cy', 'w', 'h', 'kpt'), expected_values, actual_row
            )
            if abs(exp - act) > tolerance
        ]
        if bad:
            snippet = ', '.join(
                f'{n}: 期望{e:.6f} 实际{a:.6f}' for n, e, a in bad[:4]
            )
            diffs.append(f'第{idx}行数值差异（容差 {tolerance}）: {snippet}')

    if diffs:
        return FileValidation(json_file, txt_file, 'mismatch', tuple(issues), tuple(diffs))
    return FileValidation(json_file, txt_file, 'ok')


@dataclass(frozen=True)
class ValidationReport:
    root: Path
    items: tuple[FileValidation, ...]
    extra_txts: tuple[Path, ...] = ()

    @property
    def ok_count(self) -> int:
        return sum(1 for item in self.items if item.status == 'ok')

    @property
    def bad_count(self) -> int:
        return sum(1 for item in self.items if item.status != 'ok')

    @property
    def missing_txt_count(self) -> int:
        return sum(1 for item in self.items if item.status == 'missing-txt')

    def csv_lines(self) -> list[str]:
        lines = ['JSON 文件,对应 TXT,状态,问题,数值差异']
        for item in self.items:
            lines.append(
                f'{item.json_path},{item.txt_path or ""},{item.status},'
                f'{"; ".join(item.issues)},{"; ".join(item.diffs)}'
            )
        for extra in self.extra_txts:
            lines.append(f'{extra},,extra-txt,缺少对应 JSON,')
        return lines


def validate_annotation_tree(
    annotations_root: str | Path,
    labels_root: str | Path,
    config: LabelConfig,
    tolerance: float = 1e-4,
    scope: str | None = None,
) -> ValidationReport:
    """Validate every JSON/TXT pair across the batch directories.

    ``scope`` optionally restricts the walk to a relative subdirectory of the
    annotation root (e.g. a batch name).
    """
    ann_root = Path(annotations_root).resolve()
    lbl_root = Path(labels_root).resolve()
    scope_rel = _normalize_scope(scope)
    walk_root = ann_root / scope_rel if scope_rel else ann_root
    items: list[FileValidation] = []
    json_paths: set[Path] = set()
    txt_paths: set[Path] = set()
    for json_path in sorted(walk_root.rglob('*.json')):
        json_paths.add(json_path)
        try:
            rel = json_path.relative_to(ann_root)
        except ValueError:
            rel = json_path
        txt_path = (lbl_root / rel).with_name(json_path.stem + '.txt')
        if txt_path.exists():
            txt_paths.add(txt_path)
        else:
            txt_path = None
        items.append(validate_txt_against_json(json_path, txt_path, config, tolerance))

    extra_txts = []
    extra_walk = lbl_root / scope_rel if scope_rel else lbl_root
    for txt_path in sorted(extra_walk.rglob('*.txt')):
        if txt_path in txt_paths:
            continue
        try:
            rel = txt_path.relative_to(lbl_root)
        except ValueError:
            rel = txt_path
        json_candidate = (ann_root / rel).with_name(txt_path.stem + '.json')
        if not json_candidate.exists():
            extra_txts.append(txt_path)
    return ValidationReport(ann_root, tuple(items), tuple(extra_txts))


# --- JSON -> TXT batch conversion ---

@dataclass(frozen=True)
class ConvertItem:
    source: Path
    target: Path
    status: str       # written | exists-skip | empty | error
    lines: int
    message: str = ''


@dataclass(frozen=True)
class ConvertReport:
    root: Path
    items: tuple[ConvertItem, ...]

    @property
    def written_count(self) -> int:
        return sum(1 for item in self.items if item.status in ('written', 'empty'))

    @property
    def skipped_count(self) -> int:
        return sum(1 for item in self.items if item.status == 'exists-skip')

    @property
    def failed_count(self) -> int:
        return sum(1 for item in self.items if item.status == 'error')

    def csv_lines(self) -> list[str]:
        lines = ['JSON 文件,输出 TXT,状态,行数,说明']
        for item in self.items:
            lines.append(
                f'{item.source},{item.target},{item.status},{item.lines},{item.message}'
            )
        return lines


def convert_json_batch(
    annotations_root: str | Path,
    labels_root: str | Path,
    config: LabelConfig,
    exists_policy: str = 'skip',
    dry_run: bool = False,
    max_errors: int = 100,
    scope: str | None = None,
    skip_empty: bool = False,
) -> ConvertReport:
    """Convert every JSON under ``annotations_root`` into YOLO TXT files.

    ``scope`` optionally restricts the walk to a relative subdirectory of the
    annotation root (e.g. a batch name); other directories stay untouched.
    ``exists_policy``: ``skip`` keeps existing TXT files; ``overwrite``
    rewrites them in place; ``backup`` moves the existing TXT into a timestamp
    backup directory before writing the new one.
    """
    ann_root = Path(annotations_root).resolve()
    lbl_root = Path(labels_root).resolve()
    scope_rel = _normalize_scope(scope)
    backup_dir: Path | None = None
    if exists_policy == 'backup':
        backup_dir = lbl_root.parent / f'convert_backup_{time.strftime("%Y%m%d-%H%M%S")}'

    items: list[ConvertItem] = []
    walk_root = ann_root / scope_rel if scope_rel else ann_root
    for json_path in sorted(walk_root.rglob('*.json')):
        try:
            import json as _json
            data = _json.loads(json_path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            items.append(ConvertItem(json_path, None, 'error', 0, f'JSON 读取失败: {exc}'))
            continue

        try:
            rel = json_path.relative_to(ann_root)
        except ValueError:
            rel = json_path
        target = (lbl_root / rel).with_name(json_path.stem + '.txt')

        lines, issues = json_to_yolo_lines(data, config)
        if dry_run:
            items.append(ConvertItem(
                json_path, target, 'written', len(lines),
                '预览（未写盘）; ' + ('; '.join(issues[:3]) if issues else '')
            ))
            continue

        if skip_empty and not lines:
            # 空标注：不生成空 TXT；已有空 TXT（0 字节）一并清理
            if target.exists():
                try:
                    if target.stat().st_size == 0:
                        target.unlink()
                except OSError:
                    pass
            items.append(ConvertItem(
                json_path, target, 'empty-skipped', 0,
                '空标注，已跳过' + ('; ' + '; '.join(issues[:2]) if issues else '')
            ))
            continue

        if target.exists() and exists_policy == 'skip':
            items.append(ConvertItem(json_path, target, 'exists-skip', 0, '目标已存在，跳过'))
            continue
        try:
            if target.exists() and backup_dir is not None:
                backup_target = backup_dir / rel
                backup_target.parent.mkdir(parents=True, exist_ok=True)
                if not backup_target.exists():
                    shutil.move(str(target), str(backup_target))
                else:
                    items.append(ConvertItem(
                        json_path, target, 'error', 0,
                        f'备份目标已存在: {backup_target}'
                    ))
                    continue
            dirname = target.parent if target.parent != Path('.') else lbl_root
            dirname.mkdir(parents=True, exist_ok=True)
            target.write_text(
                ('\n'.join(lines) + '\n') if lines else '',
                encoding='utf-8',
            )
            items.append(ConvertItem(
                json_path, target,
                'written' if lines else 'empty',
                len(lines),
                '; '.join(issues[:3]) if issues else '',
            ))
        except OSError as exc:
            items.append(ConvertItem(json_path, target, 'error', 0, f'写入失败: {exc}'))
    return ConvertReport(ann_root, tuple(items))


def preview_conversion(
    json_path: str | Path,
    config: LabelConfig,
    max_lines: int = 20,
) -> str:
    """Render the first rows a JSON would convert to, without writing."""
    json_file = Path(json_path)
    try:
        import json as _json
        data = _json.loads(json_file.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        return f'读取失败: {exc}'
    lines, issues = json_to_yolo_lines(data, config)
    heading = (
        f'任务={config.task_type} 配置={config.source_name or "内置"}\n'
        f'文件={json_file.name}  将生成 {len(lines)} 行数据\n'
    )
    body = '\n'.join(lines[:max_lines])
    if len(lines) > max_lines:
        body += f'\n... 还有 {len(lines) - max_lines} 行'
    if issues:
        body += '\n' + '\n'.join('⚠ ' + issue for issue in issues[:10])
    return heading + body


def _polygon_area(points) -> float:
    """Shoelace polygon area (pixel units)."""
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += (x1 * y2 - x2 * y1)
    return abs(area) / 2.0


def _normalize_scope(scope: str | None) -> str:
    """Sanitize a scope subdirectory; empty means the whole tree."""
    value = str(scope or '').strip().strip('/')
    if not value or value in {'.', '..'}:
        return ''
    if '/' in value and '..' in value.split('/'):
        raise ValueError(f'转换范围包含非法路径: {value!r}')
    return value


def export_csv(lines: Iterable[str], path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return target
