"""Annotation review helpers for LabelMe JSON files."""

import importlib
import importlib.util
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from app.models.annotation_schema import infer_annotation_schema


DEFAULT_KEYPOINTS = []
KEYPOINTS = list(DEFAULT_KEYPOINTS)
KEYPOINT_SET = set(KEYPOINTS)
KEYPOINT_INDEX = {label: idx for idx, label in enumerate(KEYPOINTS)}

DEFAULT_TARGET_CLASSES = []
TARGET_CLASSES = list(DEFAULT_TARGET_CLASSES)
TARGET_CLASS_SET = set(TARGET_CLASSES)

DEFAULT_KPT_CONNECTIONS = []
KPT_CONNECTIONS = [list(connection) for connection in DEFAULT_KPT_CONNECTIONS]
KPT_CONNECTION_LABELS = [
    (KEYPOINTS[a], KEYPOINTS[b]) for a, b in KPT_CONNECTIONS
]

DEFAULT_LEFT_RIGHT_PAIRS = []
LEFT_RIGHT_PAIRS = list(DEFAULT_LEFT_RIGHT_PAIRS)

DEFAULT_TASK_TYPE = "pose"
DEFAULT_ANNOTATION_DIR = "annotations"

SUPPORTED_REVIEW_RULES = (
    "duplicate_keypoint",
    "suspected_left_right_swap",
    "missing_person_box",
    "keypoint_outside_box",
    "keypoint_wrong_person",
    "group_id_missing",
    "group_id_conflict",
    "empty_annotation",
    "invalid_rectangle",
    "bbox_outside_image",
    "bbox_small_area",
    "bbox_bad_aspect_ratio",
    "bbox_duplicate",
    "invalid_rotation_box",
    "obb_outside_image",
    "obb_duplicate_points",
    "obb_corner_order",
    "obb_small_area",
    "obb_bad_aspect_ratio",
    "invalid_polygon",
    "polygon_outside_image",
    "polygon_duplicate_points",
    "polygon_self_intersection",
    "polygon_small_area",
    "unknown_class",
    "unexpected_shape_type",
    "image_size_missing",
    "image_size_mismatch",
)
TASK_REVIEW_RULES = {
    "pose": (
        "duplicate_keypoint",
        "suspected_left_right_swap",
        "missing_person_box",
        "keypoint_outside_box",
        "keypoint_wrong_person",
        "group_id_missing",
        "group_id_conflict",
        "image_size_missing",
        "image_size_mismatch",
    ),
    "detection": (
        "image_size_missing",
        "image_size_mismatch",
        "empty_annotation",
        "invalid_rectangle",
        "bbox_outside_image",
        "bbox_small_area",
        "bbox_bad_aspect_ratio",
        "bbox_duplicate",
        "unknown_class",
        "unexpected_shape_type",
    ),
    "segmentation": (
        "image_size_missing",
        "image_size_mismatch",
        "empty_annotation",
        "invalid_polygon",
        "polygon_outside_image",
        "polygon_duplicate_points",
        "polygon_self_intersection",
        "polygon_small_area",
        "unknown_class",
        "unexpected_shape_type",
    ),
    "obb": (
        "image_size_missing",
        "image_size_mismatch",
        "empty_annotation",
        "invalid_rotation_box",
        "obb_outside_image",
        "obb_duplicate_points",
        "obb_corner_order",
        "obb_small_area",
        "obb_bad_aspect_ratio",
        "unknown_class",
        "unexpected_shape_type",
    ),
}
DEFAULT_REVIEW_RULES = {
    rule: rule in TASK_REVIEW_RULES[DEFAULT_TASK_TYPE]
    for rule in SUPPORTED_REVIEW_RULES
}
CONFIGURED_REVIEW_RULES = dict(DEFAULT_REVIEW_RULES)
ENABLED_REVIEW_RULES = {
    rule for rule, enabled in DEFAULT_REVIEW_RULES.items() if enabled
}
CUSTOM_REVIEW_RULES: list[dict[str, Any]] = []

DEFAULT_REVIEW_THRESHOLDS = {
    "box_margin_min": 4.0,
    "box_margin_ratio": 0.02,
    "left_right_min_points": 6.0,
    "left_right_margin_min": 12.0,
    "left_right_margin_ratio": 0.04,
    "left_right_score_ratio": 1.35,
    "bbox_min_area": 4.0,
    "bbox_min_side": 2.0,
    "bbox_max_aspect_ratio": 30.0,
    "bbox_duplicate_iou": 0.95,
    "obb_min_area": 4.0,
    "obb_min_edge": 2.0,
    "obb_max_aspect_ratio": 30.0,
    "polygon_min_area": 4.0,
    "polygon_min_edge": 2.0,
}
REVIEW_THRESHOLDS = dict(DEFAULT_REVIEW_THRESHOLDS)

TASK_REVIEW_THRESHOLDS = {
    "pose": (
        "box_margin_min",
        "box_margin_ratio",
        "left_right_min_points",
        "left_right_margin_min",
        "left_right_margin_ratio",
        "left_right_score_ratio",
    ),
    "detection": (
        "bbox_min_area",
        "bbox_min_side",
        "bbox_max_aspect_ratio",
        "bbox_duplicate_iou",
    ),
    "segmentation": (
        "polygon_min_area",
        "polygon_min_edge",
    ),
    "obb": (
        "obb_min_area",
        "obb_min_edge",
        "obb_max_aspect_ratio",
    ),
}

TASK_PRESETS = {
    "pose": {
        "name": "Pose 姿态估计数据审查",
        "annotation_dir": "annotations",
        "label_dir": "labels",
        "template_file": "pose_review_template.json",
    },
    "detection": {
        "name": "目标检测数据审查",
        "annotation_dir": "annotations-det",
        "label_dir": "labels-det",
        "template_file": "detection_review_template.json",
    },
    "segmentation": {
        "name": "语义/实例分割数据审查",
        "annotation_dir": "annotations-seg",
        "label_dir": "labels-seg",
        "template_file": "segmentation_review_template.json",
    },
    "obb": {
        "name": "旋转目标检测数据审查",
        "annotation_dir": "annotations-obb",
        "label_dir": "labels-obb",
        "template_file": "obb_review_template.json",
    },
}
CURRENT_POSE_CONFIG_NAME = "ShengSong Pose 23点默认模板"
CURRENT_POSE_CONFIG_PATH: Path | None = None
CURRENT_TASK_TYPE = DEFAULT_TASK_TYPE
CURRENT_ANNOTATION_DIR = DEFAULT_ANNOTATION_DIR


@dataclass(frozen=True)
class PoseReviewConfig:
    """External pose-review template loaded from JSON."""

    name: str
    task_type: str
    annotation_dir: str
    keypoints: tuple[str, ...]
    target_classes: tuple[str, ...]
    kpt_connections: tuple[tuple[int, int], ...]
    left_right_pairs: tuple[tuple[str, str], ...]
    rules: dict[str, bool]
    thresholds: dict[str, float]
    custom_rules: tuple[dict[str, Any], ...] = ()
    path: Path | None = None


@dataclass(frozen=True)
class ReviewIssue:
    """A structured annotation review issue."""

    rule: str
    severity: str
    message: str
    group_id: Any
    label: str
    shape_indices: list[int]
    point_indices: list[tuple[int, int]]


@dataclass(frozen=True)
class AnnotationSummary:
    """Count labels and review-relevant shapes in one annotation file."""

    valid: bool
    checked: bool | None = None
    shapes: int = 0
    person_boxes: int = 0
    keypoints: int = 0
    other_shapes: int = 0
    target_class_counts: dict[str, int] | None = None
    keypoint_counts: dict[str, int] | None = None
    shape_type_counts: dict[str, int] | None = None
    error: str = ''


@dataclass(frozen=True)
class _PointShape:
    shape_idx: int
    label: str
    group_id: Any
    point: tuple[float, float]


@dataclass(frozen=True)
class _BoxShape:
    shape_idx: int
    label: str
    group_id: Any
    rect: tuple[float, float, float, float]


@dataclass(frozen=True)
class ReorderResult:
    """Result of reordering keypoint shapes in an annotation file."""

    changed: bool
    groups: int
    keypoints: int
    path: Path | None = None


@dataclass(frozen=True)
class ReviewContext:
    """Context object passed to custom review rules and Python plugins."""

    data: dict
    shapes: list[dict]
    boxes: list[_BoxShape]
    points: list[_PointShape]
    image_size: tuple[int, int] | None = None
    config_path: Path | None = None

    def group_ids(self) -> list[Any]:
        ids = {box.group_id for box in self.boxes}
        ids.update(point.group_id for point in self.points)
        return sorted(
            (group_id for group_id in ids if not _is_missing_group_id(group_id)),
            key=_group_sort_key,
        )

    def boxes_in_group(self, group_id: Any) -> list[_BoxShape]:
        return [box for box in self.boxes if box.group_id == group_id]

    def points_in_group(self, group_id: Any,
                        label: str | None = None) -> list[_PointShape]:
        points = [point for point in self.points if point.group_id == group_id]
        if label is not None:
            points = [point for point in points if point.label == label]
        return points

    def point(self, group_id: Any, label: str) -> _PointShape | None:
        points = self.points_in_group(group_id, label)
        return points[0] if points else None

    def issue(self, rule_id: str, message: str,
              severity: str = 'error',
              group_id: Any = None,
              label: str = '',
              shape_indices: list[int] | None = None,
              point_indices: list[tuple[int, int]] | None = None) -> dict:
        """Return a JSON-like issue object from a plugin without imports."""
        return {
            'rule': rule_id,
            'severity': severity,
            'message': message,
            'group_id': group_id,
            'label': label,
            'shape_indices': shape_indices or [],
            'point_indices': point_indices or [],
        }


def default_pose_review_config() -> PoseReviewConfig:
    """Return schema-free review policy until dataset analysis runs."""
    return PoseReviewConfig(
        name="Pose 数据自动分析",
        task_type=DEFAULT_TASK_TYPE,
        annotation_dir=DEFAULT_ANNOTATION_DIR,
        keypoints=tuple(DEFAULT_KEYPOINTS),
        target_classes=tuple(DEFAULT_TARGET_CLASSES),
        kpt_connections=tuple(
            (int(a), int(b)) for a, b in DEFAULT_KPT_CONNECTIONS
        ),
        left_right_pairs=tuple(DEFAULT_LEFT_RIGHT_PAIRS),
        rules=dict(DEFAULT_REVIEW_RULES),
        thresholds=dict(DEFAULT_REVIEW_THRESHOLDS),
        custom_rules=(),
    )


def default_task_review_config(task_type: str) -> PoseReviewConfig:
    """Return a built-in review template for a top-level task."""
    task_type = str(task_type or DEFAULT_TASK_TYPE).strip()
    if task_type == DEFAULT_TASK_TYPE:
        return default_pose_review_config()

    preset = TASK_PRESETS.get(task_type)
    if preset is None:
        raise ValueError(f'未知任务类型: {task_type}')

    return PoseReviewConfig(
        name=str(preset['name']),
        task_type=task_type,
        annotation_dir=str(preset['annotation_dir']),
        keypoints=(),
        target_classes=(),
        kpt_connections=(),
        left_right_pairs=(),
        rules=_default_rules_for_task(task_type),
        thresholds=dict(DEFAULT_REVIEW_THRESHOLDS),
        custom_rules=(),
    )


def current_pose_review_config() -> PoseReviewConfig:
    """Return the active pose review template."""
    return PoseReviewConfig(
        name=CURRENT_POSE_CONFIG_NAME,
        task_type=CURRENT_TASK_TYPE,
        annotation_dir=CURRENT_ANNOTATION_DIR,
        keypoints=tuple(KEYPOINTS),
        target_classes=tuple(TARGET_CLASSES),
        kpt_connections=tuple((int(a), int(b)) for a, b in KPT_CONNECTIONS),
        left_right_pairs=tuple(tuple(pair) for pair in LEFT_RIGHT_PAIRS),
        rules=dict(CONFIGURED_REVIEW_RULES),
        thresholds=dict(REVIEW_THRESHOLDS),
        custom_rules=_copy_custom_rules(CUSTOM_REVIEW_RULES),
        path=CURRENT_POSE_CONFIG_PATH,
    )


def review_config_from_data(
    config: PoseReviewConfig,
    annotation_paths: list[str | Path] | tuple[str | Path, ...],
    dataset_yaml: str | Path | None = None,
) -> PoseReviewConfig:
    """Overlay dataset-owned class/keypoint schema on review policy."""
    schema = infer_annotation_schema(
        annotation_paths,
        task_type=config.task_type,
        dataset_yaml=dataset_yaml,
    )
    # Class names and the observed point set belong to the dataset. The review
    # template still owns its policy and skeleton topology; JSON annotations do
    # not normally contain enough information to infer the latter.
    connections = _remap_connections_by_name(
        config.kpt_connections,
        config.keypoints,
        schema.keypoints,
    )
    configured_pairs = _remap_pairs_by_name(
        config.left_right_pairs,
        config.keypoints,
        schema.keypoints,
    )
    return replace(
        config,
        target_classes=schema.target_classes,
        keypoints=schema.keypoints,
        left_right_pairs=configured_pairs or schema.left_right_pairs,
        kpt_connections=connections,
    )


def _remap_connections_by_name(
    connections: tuple[tuple[int, int], ...],
    configured_keypoints: tuple[str, ...],
    data_keypoints: tuple[str, ...],
) -> tuple[tuple[int, int], ...]:
    """Map template skeleton indices onto the data-owned point order."""
    if not connections:
        return ()
    data_index = {name: index for index, name in enumerate(data_keypoints)}
    remapped = []
    for left, right in connections:
        if not (
            0 <= left < len(configured_keypoints)
            and 0 <= right < len(configured_keypoints)
        ):
            continue
        left_index = data_index.get(configured_keypoints[left])
        right_index = data_index.get(configured_keypoints[right])
        if left_index is None or right_index is None:
            continue
        remapped.append((left_index, right_index))
    return tuple(remapped)


def _remap_pairs_by_name(
    pairs: tuple[tuple[str, str], ...],
    configured_keypoints: tuple[str, ...],
    data_keypoints: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    """Keep template-defined semantic pairs present in the current data."""
    configured = set(configured_keypoints)
    data_names = set(data_keypoints)
    return tuple(
        (left, right)
        for left, right in pairs
        if left in configured and right in configured
        and left in data_names and right in data_names
    )


def _template_description_for_task(task_type: str) -> str:
    task_type = str(task_type or DEFAULT_TASK_TYPE)
    descriptions = {
        'pose': (
            '导入此 JSON 后，Pose 任务的审查规则、标注目录、骨架线和关键点'
            '重排序都会按这里执行。'
        ),
        'detection': (
            '导入此 JSON 后，目标检测任务会按这里的矩形框、面积、长宽比、'
            '重复框、类别、图片尺寸和自定义规则进行审查。'
        ),
        'segmentation': (
            '导入此 JSON 后，分割任务会按这里的 polygon 点集、自交、面积、'
            '类别、图片尺寸和自定义规则进行审查。'
        ),
        'obb': (
            '导入此 JSON 后，OBB 任务会按这里的 rotation 四点框、类别、'
            '图片尺寸和自定义规则进行审查。'
        ),
    }
    return descriptions.get(
        task_type,
        '导入此 JSON 后，当前任务会按这里的标注目录、规则和插件进行审查。',
    )


def _field_tips_for_task(task_type: str) -> dict[str, str]:
    task_type = str(task_type or DEFAULT_TASK_TYPE)
    common = {
        'task_type': '任务类型，例如 pose、detection、segmentation、obb。',
        'annotation_dir': '当前任务使用的 X-AnyLabeling JSON 标注目录名称。',
        'target_classes': '类别白名单；留空表示不检查未知类别。',
        'rules': '当前任务的内置规则开关；未知规则会提示未执行。',
        'custom_rules': (
            '特定场景规则可通过参数化规则或 Python 插件扩展。'
            'Python 插件使用 type=python。'
        ),
    }
    if task_type == DEFAULT_TASK_TYPE:
        common.update({
            'target_classes': '哪些 rectangle 标签被视为人的框。',
            'keypoints': '关键点标准顺序；重排序功能会按这个顺序排列。',
            'kpt_connections': '骨架线连接关系，可写索引，也可写关键点名称。',
            'left_right_pairs': '左右反标检查用的成对关键点。',
            'thresholds': 'Pose 几何检查的容忍阈值。',
        })
    elif task_type == 'detection':
        common.update({
            'target_classes': '检测类别白名单；留空表示不检查未知类别。',
            'rules': (
                '检测规则开关，例如矩形框合法性、面积、长宽比、重复框、'
                '越界、空标注和类别检查。'
            ),
            'thresholds': (
                '检测框几何阈值。bbox_min_area 为最小面积，bbox_min_side 为'
                '最小边长，bbox_max_aspect_ratio 为最大长宽比，'
                'bbox_duplicate_iou 为重复框 IoU 阈值。'
            ),
        })
    elif task_type == 'segmentation':
        common.update({
            'target_classes': '分割类别白名单；留空表示不检查未知类别。',
            'rules': (
                '分割规则开关，例如 polygon 合法性、重复点/短边、自交、'
                '面积、越界、空标注和类别检查。'
            ),
            'thresholds': (
                '分割 polygon 几何阈值。polygon_min_area 为最小面积，'
                'polygon_min_edge 为最小边长。'
            ),
        })
    elif task_type == 'obb':
        common.update({
            'target_classes': 'OBB 类别白名单；留空表示不检查未知类别。',
            'rules': (
                'OBB 规则开关，例如 rotation 四点框合法性、角点顺序、'
                '面积、长宽比、越界、空标注和类别检查。'
            ),
            'thresholds': (
                'OBB 几何阈值。obb_min_area 为最小面积，obb_min_edge 为'
                '最小边长，obb_max_aspect_ratio 为最大长宽比。'
            ),
        })
    return common


def _export_thresholds_for_task(config: PoseReviewConfig) -> dict[str, float]:
    threshold_names = TASK_REVIEW_THRESHOLDS.get(str(config.task_type or ''))
    if not threshold_names:
        return {}
    return {
        name: float(config.thresholds.get(
            name,
            DEFAULT_REVIEW_THRESHOLDS.get(name, 0.0),
        ))
        for name in threshold_names
    }


def _export_rules_for_task(config: PoseReviewConfig) -> dict[str, bool]:
    task_type = str(config.task_type or DEFAULT_TASK_TYPE)
    task_rule_names = TASK_REVIEW_RULES.get(task_type)
    if task_rule_names is None:
        task_rule_names = tuple(
            rule for rule, enabled in _default_rules_for_task(task_type).items()
            if enabled
        )

    rules = {
        rule: bool(config.rules.get(rule, False))
        for rule in task_rule_names
    }
    for rule, enabled in config.rules.items():
        rule_name = str(rule)
        if rule_name in SUPPORTED_REVIEW_RULES or rule_name in rules:
            continue
        rules[rule_name] = bool(enabled)
    return rules


def pose_review_config_to_dict(config: PoseReviewConfig) -> dict:
    """Return a JSON-serializable representation of a pose review template."""
    data = {
        'name': config.name,
        'version': 1,
        'description': _template_description_for_task(config.task_type),
        'task_type': config.task_type,
        'annotation_dir': config.annotation_dir,
        'target_classes': list(config.target_classes),
        'rules': _export_rules_for_task(config),
        'custom_rules': [dict(rule) for rule in config.custom_rules],
        'field_tips': _field_tips_for_task(config.task_type),
    }
    if config.task_type == DEFAULT_TASK_TYPE:
        data.update({
            'keypoints': list(config.keypoints),
            'kpt_connections': [
                list(connection) for connection in config.kpt_connections
            ],
            'left_right_pairs': [list(pair) for pair in config.left_right_pairs],
        })
    thresholds = _export_thresholds_for_task(config)
    if thresholds:
        data['thresholds'] = thresholds
    if not data['custom_rules']:
        data['custom_rules'] = []
    return data


def load_pose_review_config(path: str | Path) -> PoseReviewConfig:
    """Load a pose review template from a JSON file."""
    config_path = Path(path)
    try:
        data = json.loads(config_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ValueError(f'模板不是有效 JSON: {exc}') from exc
    except OSError as exc:
        raise ValueError(f'无法读取模板文件: {exc}') from exc

    if not isinstance(data, dict):
        raise ValueError('模板根节点必须是 JSON object')
    return pose_review_config_from_dict(data, config_path)


def pose_review_config_from_dict(data: dict,
                                 path: str | Path | None = None
                                 ) -> PoseReviewConfig:
    """Build a pose review template from parsed JSON data."""
    source_path = Path(path) if path is not None else None
    name = str(
        data.get('name') or data.get('title')
        or (source_path.stem if source_path else '自定义审查模板')
    ).strip()
    if not name:
        name = '自定义审查模板'

    task_type = _simple_name(
        data.get('task_type', DEFAULT_TASK_TYPE),
        'task_type',
    )
    annotation_dir = _annotation_dir_name(
        data.get('annotation_dir', DEFAULT_ANNOTATION_DIR),
    )

    default_keypoints = DEFAULT_KEYPOINTS if task_type == DEFAULT_TASK_TYPE else []
    keypoints = _string_tuple(
        data.get('keypoints', default_keypoints), 'keypoints'
    )
    if task_type == DEFAULT_TASK_TYPE and not keypoints:
        raise ValueError('keypoints 不能为空')
    if len(set(keypoints)) != len(keypoints):
        raise ValueError('keypoints 中存在重复名称')

    default_target_classes = (
        DEFAULT_TARGET_CLASSES if task_type == DEFAULT_TASK_TYPE else []
    )
    target_classes = _string_tuple(
        data.get('target_classes', data.get('classes', default_target_classes)),
        'target_classes',
    )
    if task_type == DEFAULT_TASK_TYPE and not target_classes:
        raise ValueError('target_classes 不能为空')

    keypoint_index = {label: idx for idx, label in enumerate(keypoints)}
    default_connections = (
        DEFAULT_KPT_CONNECTIONS if task_type == DEFAULT_TASK_TYPE else []
    )
    raw_connections = data.get(
        'kpt_connections',
        data.get('connections', data.get('skeleton', default_connections)),
    )
    connections = _connection_tuple(raw_connections, keypoint_index)

    default_pairs = DEFAULT_LEFT_RIGHT_PAIRS if task_type == DEFAULT_TASK_TYPE else []
    raw_pairs = data.get('left_right_pairs', default_pairs)
    left_right_pairs = _left_right_pair_tuple(raw_pairs, keypoint_index)

    rules = _default_rules_for_task(task_type)
    raw_rules = data.get('rules', data.get('enabled_rules'))
    if raw_rules is not None:
        rules = _rule_dict(raw_rules, rules)

    thresholds = dict(DEFAULT_REVIEW_THRESHOLDS)
    raw_thresholds = data.get('thresholds')
    if raw_thresholds is not None:
        thresholds.update(_threshold_dict(raw_thresholds))

    custom_rules = _custom_rule_tuple(
        data.get('custom_rules', []),
        keypoint_index,
        set(target_classes),
    )

    return PoseReviewConfig(
        name=name,
        task_type=task_type,
        annotation_dir=annotation_dir,
        keypoints=keypoints,
        target_classes=target_classes,
        kpt_connections=connections,
        left_right_pairs=left_right_pairs,
        rules=rules,
        thresholds=thresholds,
        custom_rules=custom_rules,
        path=source_path,
    )


def apply_pose_review_config(config: PoseReviewConfig) -> None:
    """Apply a pose review template to review, reorder, and drawing helpers."""
    global CURRENT_POSE_CONFIG_NAME, CURRENT_POSE_CONFIG_PATH
    global CURRENT_TASK_TYPE, CURRENT_ANNOTATION_DIR

    KEYPOINTS[:] = list(config.keypoints)
    TARGET_CLASSES[:] = list(config.target_classes)
    KPT_CONNECTIONS[:] = [list(connection) for connection in config.kpt_connections]
    LEFT_RIGHT_PAIRS[:] = [tuple(pair) for pair in config.left_right_pairs]

    _rebuild_pose_indexes()
    CONFIGURED_REVIEW_RULES.clear()
    CONFIGURED_REVIEW_RULES.update(config.rules)
    ENABLED_REVIEW_RULES.clear()
    ENABLED_REVIEW_RULES.update(
        rule for rule, enabled in config.rules.items() if enabled
    )
    CUSTOM_REVIEW_RULES[:] = [dict(rule) for rule in config.custom_rules]
    REVIEW_THRESHOLDS.clear()
    REVIEW_THRESHOLDS.update(DEFAULT_REVIEW_THRESHOLDS)
    REVIEW_THRESHOLDS.update(config.thresholds)

    CURRENT_POSE_CONFIG_NAME = config.name
    CURRENT_POSE_CONFIG_PATH = config.path
    CURRENT_TASK_TYPE = config.task_type
    CURRENT_ANNOTATION_DIR = config.annotation_dir


def load_and_apply_pose_review_config(path: str | Path) -> PoseReviewConfig:
    """Load a JSON template and make it the active pose review template."""
    config = load_pose_review_config(path)
    apply_pose_review_config(config)
    return config


def reset_pose_review_config() -> None:
    """Restore the built-in pose review template."""
    apply_pose_review_config(default_pose_review_config())


def review_annotation_file(annotation_path: str | Path | None,
                           image_path: str | Path | None = None
                           ) -> list[ReviewIssue]:
    """Load a LabelMe JSON file and return review issues."""
    if not annotation_path:
        return []

    path = Path(annotation_path)
    if not path.is_file():
        return []

    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        return []

    image_size = _read_image_size(image_path)
    if image_size is None:
        image_size = _read_image_size(_resolve_image_from_annotation(path, data))

    return review_annotation_data(data, image_size=image_size)


def summarize_annotation_file(annotation_path: str | Path | None) -> AnnotationSummary:
    """Return label counts for one LabelMe JSON annotation file."""
    if not annotation_path:
        return AnnotationSummary(valid=False, error='标注文件缺失')

    path = Path(annotation_path)
    if not path.is_file():
        return AnnotationSummary(valid=False, error='标注文件缺失')

    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        return AnnotationSummary(valid=False, error=f'JSON 解析失败: {exc}')
    except OSError as exc:
        return AnnotationSummary(valid=False, error=f'文件读取失败: {exc}')

    shapes = data.get('shapes', data.get('shapes ', []))
    if not isinstance(shapes, list):
        return AnnotationSummary(valid=False, error='shapes 字段缺失或无效')

    target_class_counts = {label: 0 for label in TARGET_CLASSES}
    keypoint_counts = {label: 0 for label in KEYPOINTS}
    shape_type_counts: dict[str, int] = {}
    person_boxes = 0
    keypoints = 0
    other_shapes = 0

    for shape_idx, shape in enumerate(shapes):
        if isinstance(shape, dict):
            shape_type = _shape_type(shape) or 'unknown'
            shape_type_counts[shape_type] = shape_type_counts.get(shape_type, 0) + 1

        target_label = _target_object_label_from_json(shape_idx, shape)
        if target_label:
            person_boxes += 1
            target_class_counts[target_label] = (
                target_class_counts.get(target_label, 0) + 1
            )
            continue

        point_shape = _point_shape_from_json(shape_idx, shape)
        if point_shape is not None:
            keypoints += 1
            keypoint_counts[point_shape.label] = (
                keypoint_counts.get(point_shape.label, 0) + 1
            )
            continue

        other_shapes += 1

    checked = data.get('checked')
    return AnnotationSummary(
        valid=True,
        checked=checked if isinstance(checked, bool) else None,
        shapes=len(shapes),
        person_boxes=person_boxes,
        keypoints=keypoints,
        other_shapes=other_shapes,
        target_class_counts=target_class_counts,
        keypoint_counts=keypoint_counts,
        shape_type_counts=shape_type_counts,
    )


def reorder_keypoints_file(annotation_path: str | Path) -> ReorderResult:
    """Reorder keypoint shapes in a LabelMe JSON file and save it."""
    path = Path(annotation_path)
    data = json.loads(path.read_text(encoding='utf-8'))
    result = reorder_keypoints_data(data)
    if result.changed:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
    return ReorderResult(
        changed=result.changed,
        groups=result.groups,
        keypoints=result.keypoints,
        path=path,
    )


def reorder_keypoints_data(data: dict) -> ReorderResult:
    """Reorder point shapes by group_id and the configured KEYPOINTS order."""
    shapes_key = _shapes_key(data)
    if shapes_key is None:
        return ReorderResult(False, 0, 0)

    shapes = data[shapes_key]
    if not isinstance(shapes, list):
        return ReorderResult(False, 0, 0)

    point_entries = []
    box_groups = set()
    for shape_idx, shape in enumerate(shapes):
        point_shape = _point_shape_from_json(shape_idx, shape)
        if point_shape is not None:
            point_entries.append((shape_idx, point_shape.group_id, point_shape.label, shape))
            continue

        box_shape = _box_shape_from_json(shape_idx, shape)
        if box_shape is not None:
            box_groups.add(box_shape.group_id)

    if not point_entries:
        return ReorderResult(False, 0, 0)

    grouped_points: dict[Any, list[tuple[int, str, dict]]] = {}
    for shape_idx, group_id, label, shape in point_entries:
        grouped_points.setdefault(group_id, []).append((shape_idx, label, shape))

    ordered_groups = {
        group_id: [
            shape for _idx, _label, shape in sorted(
                entries,
                key=lambda item: (KEYPOINT_INDEX[item[1]], item[0]),
            )
        ]
        for group_id, entries in grouped_points.items()
    }

    inserted_groups = set()
    reordered_shapes = []
    for shape_idx, shape in enumerate(shapes):
        point_shape = _point_shape_from_json(shape_idx, shape)
        if point_shape is not None:
            group_id = point_shape.group_id
            if group_id not in box_groups and group_id not in inserted_groups:
                reordered_shapes.extend(ordered_groups[group_id])
                inserted_groups.add(group_id)
            continue

        reordered_shapes.append(shape)

        box_shape = _box_shape_from_json(shape_idx, shape)
        if (
            box_shape is not None
            and box_shape.group_id in ordered_groups
            and box_shape.group_id not in inserted_groups
        ):
            reordered_shapes.extend(ordered_groups[box_shape.group_id])
            inserted_groups.add(box_shape.group_id)

    for group_id, entries in sorted(
        grouped_points.items(), key=lambda item: min(entry[0] for entry in item[1])
    ):
        if group_id not in inserted_groups:
            reordered_shapes.extend(ordered_groups[group_id])
            inserted_groups.add(group_id)

    changed = reordered_shapes != shapes
    if changed:
        data[shapes_key] = reordered_shapes

    return ReorderResult(
        changed=changed,
        groups=len(grouped_points),
        keypoints=len(point_entries),
    )


def review_annotation_data(data: dict,
                           image_size: tuple[int, int] | None = None
                           ) -> list[ReviewIssue]:
    """Review parsed LabelMe JSON data."""
    shapes = data.get('shapes', data.get('shapes ', []))
    if not isinstance(shapes, list):
        return []

    boxes, points = _collect_person_boxes_and_points(shapes)
    context = ReviewContext(
        data=data,
        shapes=shapes,
        boxes=boxes,
        points=points,
        image_size=image_size,
        config_path=CURRENT_POSE_CONFIG_PATH,
    )
    issues = []
    issues.extend(_find_image_size_issues(data, image_size))
    if CURRENT_TASK_TYPE == DEFAULT_TASK_TYPE:
        issues.extend(_find_group_id_issues(boxes, points))
        issues.extend(_find_missing_person_boxes(boxes, points))
        issues.extend(_find_duplicate_keypoints(shapes))
        issues.extend(_find_keypoint_box_issues(boxes, points))
        issues.extend(_find_suspected_left_right_swaps(shapes))
    elif CURRENT_TASK_TYPE == 'detection':
        issues.extend(_find_detection_issues(shapes, boxes, image_size))
    elif CURRENT_TASK_TYPE == 'obb':
        issues.extend(_find_obb_issues(shapes, image_size))
    elif CURRENT_TASK_TYPE == 'segmentation':
        issues.extend(_find_segmentation_issues(shapes, image_size))
    issues.extend(_find_custom_rule_issues(context))
    issues.extend(_find_unavailable_configured_rules())
    return issues


def _find_detection_issues(shapes: list[dict],
                           boxes: list[_BoxShape],
                           image_size: tuple[int, int] | None
                           ) -> list[ReviewIssue]:
    issues = []
    issues.extend(_find_empty_annotation_issue(shapes))
    issues.extend(_find_unexpected_shape_type_issues(
        shapes, {'rectangle'}, '检测矩形'
    ))
    issues.extend(_find_invalid_rectangle_issues(shapes))
    issues.extend(_find_bbox_small_area_issues(boxes))
    issues.extend(_find_bbox_bad_aspect_ratio_issues(boxes))
    issues.extend(_find_bbox_duplicate_issues(boxes))
    issues.extend(_find_unknown_class_issues(shapes, {'rectangle'}))
    issues.extend(_find_bbox_outside_image_issues(boxes, image_size))
    return issues


def _find_obb_issues(shapes: list[dict],
                     image_size: tuple[int, int] | None) -> list[ReviewIssue]:
    issues = []
    issues.extend(_find_empty_annotation_issue(shapes))
    issues.extend(_find_unexpected_shape_type_issues(
        shapes, {'rotation'}, '旋转框'
    ))
    issues.extend(_find_invalid_rotation_box_issues(shapes))
    issues.extend(_find_obb_duplicate_points_issues(shapes))
    issues.extend(_find_obb_corner_order_issues(shapes))
    issues.extend(_find_obb_small_area_issues(shapes))
    issues.extend(_find_obb_bad_aspect_ratio_issues(shapes))
    issues.extend(_find_unknown_class_issues(shapes, {'rotation'}))
    issues.extend(_find_obb_outside_image_issues(shapes, image_size))
    return issues


def _find_segmentation_issues(shapes: list[dict],
                              image_size: tuple[int, int] | None
                              ) -> list[ReviewIssue]:
    issues = []
    issues.extend(_find_empty_annotation_issue(shapes))
    issues.extend(_find_unexpected_shape_type_issues(
        shapes, {'polygon'}, '分割多边形'
    ))
    issues.extend(_find_invalid_polygon_issues(shapes))
    issues.extend(_find_polygon_duplicate_points_issues(shapes))
    issues.extend(_find_polygon_self_intersection_issues(shapes))
    issues.extend(_find_polygon_small_area_issues(shapes))
    issues.extend(_find_unknown_class_issues(shapes, {'polygon'}))
    issues.extend(_find_polygon_outside_image_issues(shapes, image_size))
    return issues


def _find_empty_annotation_issue(shapes: list[dict]) -> list[ReviewIssue]:
    if not _rule_enabled('empty_annotation') or shapes:
        return []
    return [
        ReviewIssue(
            rule='empty_annotation',
            severity='warning',
            message='空标注: JSON 中没有任何 shape 标注',
            group_id=None,
            label='shapes',
            shape_indices=[],
            point_indices=[],
        )
    ]


def _find_unexpected_shape_type_issues(shapes: list[dict],
                                       expected_shape_types: set[str],
                                       expected_name: str) -> list[ReviewIssue]:
    if not _rule_enabled('unexpected_shape_type'):
        return []

    issues = []
    for shape_idx, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            continue
        shape_type = _shape_type(shape)
        if shape_type in expected_shape_types:
            continue

        label = str(shape.get('label', '')).strip() or '-'
        issues.append(ReviewIssue(
            rule='unexpected_shape_type',
            severity='warning',
            message=(
                f'非{expected_name}: shape[{shape_idx}] label={label} '
                f'shape_type={shape_type or "-"}'
            ),
            group_id=shape.get('group_id'),
            label=label,
            shape_indices=[shape_idx],
            point_indices=[],
        ))
    return issues


def _find_invalid_rectangle_issues(shapes: list[dict]) -> list[ReviewIssue]:
    if not _rule_enabled('invalid_rectangle'):
        return []

    issues = []
    for shape_idx, shape in enumerate(shapes):
        if not isinstance(shape, dict) or _shape_type(shape) != 'rectangle':
            continue

        rect, error = _rectangle_from_shape(shape)
        if error is None:
            continue

        label = str(shape.get('label', '')).strip() or '-'
        point_indices = []
        points = shape.get('points', [])
        if isinstance(points, list):
            point_indices = [(shape_idx, idx) for idx in range(len(points))]
        issues.append(ReviewIssue(
            rule='invalid_rectangle',
            severity='error',
            message=f'无效矩形框: shape[{shape_idx}] label={label}; {error}',
            group_id=shape.get('group_id'),
            label=label,
            shape_indices=[shape_idx],
            point_indices=point_indices,
        ))
    return issues


def _find_unknown_class_issues(shapes: list[dict],
                               expected_shape_types: set[str]) -> list[ReviewIssue]:
    if not _rule_enabled('unknown_class') or not TARGET_CLASS_SET:
        return []

    issues = []
    for shape_idx, shape in enumerate(shapes):
        if (
            not isinstance(shape, dict)
            or _shape_type(shape) not in expected_shape_types
        ):
            continue

        label = str(shape.get('label', '')).strip()
        if label in TARGET_CLASS_SET:
            continue
        issues.append(ReviewIssue(
            rule='unknown_class',
            severity='error',
            message=(
                f'未知检测类别: shape[{shape_idx}] label={label or "-"} '
                f'不在当前模板 classes 中'
            ),
            group_id=shape.get('group_id'),
            label=label,
            shape_indices=[shape_idx],
            point_indices=[],
        ))
    return issues


def _find_bbox_small_area_issues(boxes: list[_BoxShape]) -> list[ReviewIssue]:
    if not _rule_enabled('bbox_small_area'):
        return []

    min_area = _threshold('bbox_min_area')
    min_side = _threshold('bbox_min_side')
    if min_area <= 0 and min_side <= 0:
        return []

    issues = []
    for box in boxes:
        x1, y1, x2, y2 = box.rect
        width, height = x2 - x1, y2 - y1
        area = _rect_area(box.rect)
        area_bad = min_area > 0 and area < min_area
        side_bad = min_side > 0 and min(width, height) < min_side
        if not area_bad and not side_bad:
            continue

        details = []
        if area_bad:
            details.append(f'area={area:.2f} < {min_area:.2f}')
        if side_bad:
            details.append(f'min_side={min(width, height):.2f} < {min_side:.2f}')
        issues.append(ReviewIssue(
            rule='bbox_small_area',
            severity='warning',
            message=(
                f'检测框过小: shape[{box.shape_idx}] label={box.label}; '
                + '；'.join(details)
            ),
            group_id=box.group_id,
            label=box.label,
            shape_indices=[box.shape_idx],
            point_indices=[],
        ))
    return issues


def _find_bbox_bad_aspect_ratio_issues(boxes: list[_BoxShape]) -> list[ReviewIssue]:
    if not _rule_enabled('bbox_bad_aspect_ratio'):
        return []

    max_ratio = _threshold('bbox_max_aspect_ratio')
    if max_ratio <= 0:
        return []

    issues = []
    for box in boxes:
        x1, y1, x2, y2 = box.rect
        width, height = x2 - x1, y2 - y1
        short_side = min(width, height)
        if short_side <= 1e-6:
            continue
        ratio = max(width, height) / short_side
        if ratio <= max_ratio:
            continue

        issues.append(ReviewIssue(
            rule='bbox_bad_aspect_ratio',
            severity='warning',
            message=(
                f'检测框长宽比异常: shape[{box.shape_idx}] label={box.label}; '
                f'ratio={ratio:.2f} > {max_ratio:.2f}'
            ),
            group_id=box.group_id,
            label=box.label,
            shape_indices=[box.shape_idx],
            point_indices=[],
        ))
    return issues


def _find_bbox_duplicate_issues(boxes: list[_BoxShape]) -> list[ReviewIssue]:
    if not _rule_enabled('bbox_duplicate'):
        return []

    threshold = _threshold('bbox_duplicate_iou')
    if threshold <= 0:
        return []

    issues = []
    for left_idx, left_box in enumerate(boxes):
        for right_box in boxes[left_idx + 1:]:
            if left_box.label != right_box.label:
                continue
            iou = _rect_iou(left_box.rect, right_box.rect)
            if iou < threshold:
                continue

            issues.append(ReviewIssue(
                rule='bbox_duplicate',
                severity='warning',
                message=(
                    f'疑似重复检测框: shape[{left_box.shape_idx}] 和 '
                    f'shape[{right_box.shape_idx}] label={left_box.label}; '
                    f'IoU={iou:.3f} >= {threshold:.3f}'
                ),
                group_id=left_box.group_id,
                label=left_box.label,
                shape_indices=[left_box.shape_idx, right_box.shape_idx],
                point_indices=[],
            ))
    return issues


def _find_invalid_rotation_box_issues(shapes: list[dict]) -> list[ReviewIssue]:
    if not _rule_enabled('invalid_rotation_box'):
        return []

    issues = []
    for shape_idx, shape in enumerate(shapes):
        if not isinstance(shape, dict) or _shape_type(shape) != 'rotation':
            continue

        points, error = _rotation_points_from_shape(shape)
        if error is None:
            continue

        label = str(shape.get('label', '')).strip() or '-'
        point_indices = []
        raw_points = shape.get('points', [])
        if isinstance(raw_points, list):
            point_indices = [(shape_idx, idx) for idx in range(len(raw_points))]
        issues.append(ReviewIssue(
            rule='invalid_rotation_box',
            severity='error',
            message=f'无效旋转框: shape[{shape_idx}] label={label}; {error}',
            group_id=shape.get('group_id'),
            label=label,
            shape_indices=[shape_idx],
            point_indices=point_indices,
        ))
    return issues


def _find_obb_duplicate_points_issues(shapes: list[dict]) -> list[ReviewIssue]:
    if not _rule_enabled('obb_duplicate_points'):
        return []

    min_edge = _threshold('obb_min_edge')
    issues = []
    for shape_idx, shape in enumerate(shapes):
        if not isinstance(shape, dict) or _shape_type(shape) != 'rotation':
            continue

        points, error = _rotation_points_from_shape(shape)
        if error is not None or points is None:
            continue

        duplicate_pairs = [
            (left_idx, right_idx)
            for left_idx in range(len(points))
            for right_idx in range(left_idx + 1, len(points))
            if _distance(points[left_idx], points[right_idx]) <= 1e-6
        ]
        short_edges = [
            idx for idx, length in enumerate(_edge_lengths(points))
            if min_edge > 0 and length < min_edge
        ]
        if not duplicate_pairs and not short_edges:
            continue

        point_indexes = sorted({
            point_idx
            for pair in duplicate_pairs
            for point_idx in pair
        } | {
            edge_idx
            for edge_idx in short_edges
        } | {
            (edge_idx + 1) % len(points)
            for edge_idx in short_edges
        })
        label = str(shape.get('label', '')).strip() or '-'
        parts = []
        if duplicate_pairs:
            parts.append(
                '重复顶点 ' + ', '.join(
                    f'{left}-{right}' for left, right in duplicate_pairs
                )
            )
        if short_edges:
            parts.append(
                f'短边 {", ".join(str(idx) for idx in short_edges)} '
                f'小于 {min_edge:.1f}px'
            )
        issues.append(ReviewIssue(
            rule='obb_duplicate_points',
            severity='error',
            message=(
                f'旋转框顶点异常: shape[{shape_idx}] label={label}; '
                + '；'.join(parts)
            ),
            group_id=shape.get('group_id'),
            label=label,
            shape_indices=[shape_idx],
            point_indices=[(shape_idx, idx) for idx in point_indexes],
        ))
    return issues


def _find_obb_corner_order_issues(shapes: list[dict]) -> list[ReviewIssue]:
    if not _rule_enabled('obb_corner_order'):
        return []

    issues = []
    for shape_idx, shape in enumerate(shapes):
        if not isinstance(shape, dict) or _shape_type(shape) != 'rotation':
            continue

        points, error = _rotation_points_from_shape(shape)
        if error is not None or points is None:
            continue
        if _obb_has_duplicate_points(points) or _quad_corner_order_valid(points):
            continue

        label = str(shape.get('label', '')).strip() or '-'
        issues.append(ReviewIssue(
            rule='obb_corner_order',
            severity='error',
            message=(
                f'旋转框角点顺序异常: shape[{shape_idx}] label={label}; '
                '顶点连线存在交叉或不是凸四边形'
            ),
            group_id=shape.get('group_id'),
            label=label,
            shape_indices=[shape_idx],
            point_indices=[(shape_idx, idx) for idx in range(len(points))],
        ))
    return issues


def _find_obb_small_area_issues(shapes: list[dict]) -> list[ReviewIssue]:
    if not _rule_enabled('obb_small_area'):
        return []

    min_area = _threshold('obb_min_area')
    if min_area <= 0:
        return []

    issues = []
    for shape_idx, shape in enumerate(shapes):
        if not isinstance(shape, dict) or _shape_type(shape) != 'rotation':
            continue

        points, error = _rotation_points_from_shape(shape)
        if error is not None or points is None:
            continue
        area = _polygon_area(points)
        if area >= min_area:
            continue

        label = str(shape.get('label', '')).strip() or '-'
        issues.append(ReviewIssue(
            rule='obb_small_area',
            severity='warning',
            message=(
                f'旋转框面积过小: shape[{shape_idx}] label={label}; '
                f'area={area:.2f} < {min_area:.2f}'
            ),
            group_id=shape.get('group_id'),
            label=label,
            shape_indices=[shape_idx],
            point_indices=[(shape_idx, idx) for idx in range(len(points))],
        ))
    return issues


def _find_obb_bad_aspect_ratio_issues(shapes: list[dict]) -> list[ReviewIssue]:
    if not _rule_enabled('obb_bad_aspect_ratio'):
        return []

    max_ratio = _threshold('obb_max_aspect_ratio')
    if max_ratio <= 0:
        return []

    issues = []
    for shape_idx, shape in enumerate(shapes):
        if not isinstance(shape, dict) or _shape_type(shape) != 'rotation':
            continue

        points, error = _rotation_points_from_shape(shape)
        if error is not None or points is None:
            continue
        if not _quad_corner_order_valid(points):
            continue

        width, height = _obb_side_lengths(points)
        short_side = min(width, height)
        if short_side <= 1e-6:
            continue
        ratio = max(width, height) / short_side
        if ratio <= max_ratio:
            continue

        label = str(shape.get('label', '')).strip() or '-'
        issues.append(ReviewIssue(
            rule='obb_bad_aspect_ratio',
            severity='warning',
            message=(
                f'旋转框长宽比异常: shape[{shape_idx}] label={label}; '
                f'ratio={ratio:.2f} > {max_ratio:.2f}'
            ),
            group_id=shape.get('group_id'),
            label=label,
            shape_indices=[shape_idx],
            point_indices=[(shape_idx, idx) for idx in range(len(points))],
        ))
    return issues


def _find_invalid_polygon_issues(shapes: list[dict]) -> list[ReviewIssue]:
    if not _rule_enabled('invalid_polygon'):
        return []

    issues = []
    for shape_idx, shape in enumerate(shapes):
        if not isinstance(shape, dict) or _shape_type(shape) != 'polygon':
            continue

        points, error = _polygon_points_from_shape(shape)
        if error is None:
            continue

        label = str(shape.get('label', '')).strip() or '-'
        point_indices = []
        raw_points = shape.get('points', [])
        if isinstance(raw_points, list):
            point_indices = [(shape_idx, idx) for idx in range(len(raw_points))]
        issues.append(ReviewIssue(
            rule='invalid_polygon',
            severity='error',
            message=f'无效分割多边形: shape[{shape_idx}] label={label}; {error}',
            group_id=shape.get('group_id'),
            label=label,
            shape_indices=[shape_idx],
            point_indices=point_indices,
        ))
    return issues


def _find_polygon_duplicate_points_issues(shapes: list[dict]) -> list[ReviewIssue]:
    if not _rule_enabled('polygon_duplicate_points'):
        return []

    min_edge = _threshold('polygon_min_edge')
    issues = []
    for shape_idx, shape in enumerate(shapes):
        if not isinstance(shape, dict) or _shape_type(shape) != 'polygon':
            continue

        points, error = _polygon_coords_for_quality(shape)
        if error is not None or points is None:
            continue

        duplicate_pairs = _duplicate_point_pairs(points)
        short_edges = [
            idx for idx, length in enumerate(_edge_lengths(points))
            if min_edge > 0 and length < min_edge
        ]
        if not duplicate_pairs and not short_edges:
            continue

        point_indexes = sorted({
            point_idx
            for pair in duplicate_pairs
            for point_idx in pair
        } | {
            edge_idx
            for edge_idx in short_edges
        } | {
            (edge_idx + 1) % len(points)
            for edge_idx in short_edges
        })
        label = str(shape.get('label', '')).strip() or '-'
        parts = []
        if duplicate_pairs:
            parts.append(
                '重复顶点 ' + ', '.join(
                    f'{left}-{right}' for left, right in duplicate_pairs
                )
            )
        if short_edges:
            parts.append(
                f'短边 {", ".join(str(idx) for idx in short_edges)} '
                f'小于 {min_edge:.1f}px'
            )
        issues.append(ReviewIssue(
            rule='polygon_duplicate_points',
            severity='error',
            message=(
                f'分割多边形顶点异常: shape[{shape_idx}] label={label}; '
                + '；'.join(parts)
            ),
            group_id=shape.get('group_id'),
            label=label,
            shape_indices=[shape_idx],
            point_indices=[(shape_idx, idx) for idx in point_indexes],
        ))
    return issues


def _find_polygon_self_intersection_issues(shapes: list[dict]) -> list[ReviewIssue]:
    if not _rule_enabled('polygon_self_intersection'):
        return []

    issues = []
    for shape_idx, shape in enumerate(shapes):
        if not isinstance(shape, dict) or _shape_type(shape) != 'polygon':
            continue

        points, error = _polygon_coords_for_quality(shape)
        if error is not None or points is None:
            continue
        if _points_have_duplicates(points):
            continue

        intersections = _polygon_self_intersections(points)
        if not intersections:
            continue

        point_indexes = sorted({
            point_idx
            for edge_a, edge_b in intersections
            for edge_idx in (edge_a, edge_b)
            for point_idx in (edge_idx, (edge_idx + 1) % len(points))
        })
        label = str(shape.get('label', '')).strip() or '-'
        pairs_text = ', '.join(
            f'{edge_a}-{edge_b}' for edge_a, edge_b in intersections[:4]
        )
        more = '...' if len(intersections) > 4 else ''
        issues.append(ReviewIssue(
            rule='polygon_self_intersection',
            severity='error',
            message=(
                f'分割多边形自交: shape[{shape_idx}] label={label}; '
                f'相交边={pairs_text}{more}'
            ),
            group_id=shape.get('group_id'),
            label=label,
            shape_indices=[shape_idx],
            point_indices=[(shape_idx, idx) for idx in point_indexes],
        ))
    return issues


def _find_polygon_small_area_issues(shapes: list[dict]) -> list[ReviewIssue]:
    if not _rule_enabled('polygon_small_area'):
        return []

    min_area = _threshold('polygon_min_area')
    if min_area <= 0:
        return []

    issues = []
    for shape_idx, shape in enumerate(shapes):
        if not isinstance(shape, dict) or _shape_type(shape) != 'polygon':
            continue

        points, error = _polygon_coords_for_quality(shape)
        if error is not None or points is None:
            continue
        area = _polygon_area(points)
        if area <= 1e-6 or area >= min_area:
            continue

        label = str(shape.get('label', '')).strip() or '-'
        issues.append(ReviewIssue(
            rule='polygon_small_area',
            severity='warning',
            message=(
                f'分割多边形面积过小: shape[{shape_idx}] label={label}; '
                f'area={area:.2f} < {min_area:.2f}'
            ),
            group_id=shape.get('group_id'),
            label=label,
            shape_indices=[shape_idx],
            point_indices=[(shape_idx, idx) for idx in range(len(points))],
        ))
    return issues


def _find_bbox_outside_image_issues(boxes: list[_BoxShape],
                                    image_size: tuple[int, int] | None
                                    ) -> list[ReviewIssue]:
    if not _rule_enabled('bbox_outside_image') or image_size is None:
        return []

    width, height = image_size
    issues = []
    for box in boxes:
        x1, y1, x2, y2 = box.rect
        if 0 <= x1 <= width and 0 <= x2 <= width and 0 <= y1 <= height and 0 <= y2 <= height:
            continue

        issues.append(ReviewIssue(
            rule='bbox_outside_image',
            severity='error',
            message=(
                f'目标框越界: shape[{box.shape_idx}] {box.label} '
                f'box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}) '
                f'超出图片范围 0,0,{width},{height}'
            ),
            group_id=box.group_id,
            label=box.label,
            shape_indices=[box.shape_idx],
            point_indices=[],
        ))
    return issues


def _find_obb_outside_image_issues(shapes: list[dict],
                                   image_size: tuple[int, int] | None
                                   ) -> list[ReviewIssue]:
    if not _rule_enabled('obb_outside_image') or image_size is None:
        return []

    width, height = image_size
    issues = []
    for shape_idx, shape in enumerate(shapes):
        if not isinstance(shape, dict) or _shape_type(shape) != 'rotation':
            continue

        points, error = _rotation_points_from_shape(shape)
        if error is not None or points is None:
            continue
        outside = [
            idx for idx, (x, y) in enumerate(points)
            if x < 0 or x > width or y < 0 or y > height
        ]
        if not outside:
            continue

        label = str(shape.get('label', '')).strip() or '-'
        issues.append(ReviewIssue(
            rule='obb_outside_image',
            severity='error',
            message=(
                f'旋转框越界: shape[{shape_idx}] {label} '
                f'有 {len(outside)} 个顶点超出图片范围 0,0,{width},{height}'
            ),
            group_id=shape.get('group_id'),
            label=label,
            shape_indices=[shape_idx],
            point_indices=[(shape_idx, idx) for idx in outside],
        ))
    return issues


def _find_polygon_outside_image_issues(shapes: list[dict],
                                       image_size: tuple[int, int] | None
                                       ) -> list[ReviewIssue]:
    if not _rule_enabled('polygon_outside_image') or image_size is None:
        return []

    width, height = image_size
    issues = []
    for shape_idx, shape in enumerate(shapes):
        if not isinstance(shape, dict) or _shape_type(shape) != 'polygon':
            continue

        points, error = _polygon_points_from_shape(shape)
        if error is not None or points is None:
            continue
        outside = [
            idx for idx, (x, y) in enumerate(points)
            if x < 0 or x > width or y < 0 or y > height
        ]
        if not outside:
            continue

        label = str(shape.get('label', '')).strip() or '-'
        issues.append(ReviewIssue(
            rule='polygon_outside_image',
            severity='error',
            message=(
                f'分割多边形越界: shape[{shape_idx}] {label} '
                f'有 {len(outside)} 个顶点超出图片范围 0,0,{width},{height}'
            ),
            group_id=shape.get('group_id'),
            label=label,
            shape_indices=[shape_idx],
            point_indices=[(shape_idx, idx) for idx in outside],
        ))
    return issues


def _find_duplicate_keypoints(shapes: list[dict]) -> list[ReviewIssue]:
    if not _rule_enabled('duplicate_keypoint'):
        return []

    grouped: dict[tuple[Any, str], list[int]] = {}

    for shape_idx, shape in enumerate(shapes):
        point_shape = _point_shape_from_json(shape_idx, shape)
        if point_shape is None:
            continue
        grouped.setdefault(
            (point_shape.group_id, point_shape.label), []
        ).append(shape_idx)

    issues = []
    for (group_id, label), shape_indices in sorted(
        grouped.items(), key=lambda item: (_group_sort_key(item[0][0]), item[0][1])
    ):
        if len(shape_indices) <= 1:
            continue
        group_text = _format_group_id(group_id)
        issues.append(ReviewIssue(
            rule='duplicate_keypoint',
            severity='error',
            message=(
                f'重复关键点: group_id={group_text} 的 {label} '
                f'出现 {len(shape_indices)} 次'
            ),
            group_id=group_id,
            label=label,
            shape_indices=shape_indices,
            point_indices=[(idx, 0) for idx in shape_indices],
        ))

    return issues


def _find_image_size_issues(data: dict,
                            image_size: tuple[int, int] | None
                            ) -> list[ReviewIssue]:
    if image_size is None:
        return []

    actual_w, actual_h = image_size
    json_w = _as_positive_int(data.get('imageWidth'))
    json_h = _as_positive_int(data.get('imageHeight'))
    if json_w is None or json_h is None:
        if not _rule_enabled('image_size_missing'):
            return []
        return [
            ReviewIssue(
                rule='image_size_missing',
                severity='warning',
                message=(
                    '图片尺寸字段缺失或无效: '
                    f'JSON imageWidth/imageHeight={data.get("imageWidth")}/'
                    f'{data.get("imageHeight")}, 实际={actual_w}×{actual_h}'
                ),
                group_id=None,
                label='imageWidth/imageHeight',
                shape_indices=[],
                point_indices=[],
            )
        ]

    if json_w == actual_w and json_h == actual_h:
        return []

    if not _rule_enabled('image_size_mismatch'):
        return []

    return [
        ReviewIssue(
            rule='image_size_mismatch',
            severity='error',
            message=(
                f'图片尺寸不一致: JSON={json_w}×{json_h}, '
                f'实际={actual_w}×{actual_h}'
            ),
            group_id=None,
            label='imageWidth/imageHeight',
            shape_indices=[],
            point_indices=[],
        )
    ]


def _find_group_id_issues(boxes: list[_BoxShape],
                          points: list[_PointShape]) -> list[ReviewIssue]:
    issues = []

    if _rule_enabled('group_id_missing'):
        for box in boxes:
            if _is_missing_group_id(box.group_id):
                issues.append(ReviewIssue(
                    rule='group_id_missing',
                    severity='error',
                    message=f'group_id 缺失: 人框 {box.label} 未分组',
                    group_id=box.group_id,
                    label=box.label,
                    shape_indices=[box.shape_idx],
                    point_indices=[],
                ))

        for point in points:
            if _is_missing_group_id(point.group_id):
                issues.append(ReviewIssue(
                    rule='group_id_missing',
                    severity='error',
                    message=f'group_id 缺失: 关键点 {point.label} 未分组',
                    group_id=point.group_id,
                    label=point.label,
                    shape_indices=[point.shape_idx],
                    point_indices=[(point.shape_idx, 0)],
                ))

    boxes_by_group = _boxes_by_group(boxes)
    if _rule_enabled('group_id_conflict'):
        for group_id, group_boxes in sorted(
            boxes_by_group.items(), key=lambda item: _group_sort_key(item[0])
        ):
            if _is_missing_group_id(group_id) or len(group_boxes) <= 1:
                continue
            group_text = _format_group_id(group_id)
            labels = ', '.join(box.label for box in group_boxes)
            issues.append(ReviewIssue(
                rule='group_id_conflict',
                severity='error',
                message=(
                    f'group_id 混乱: group_id={group_text} 下有 '
                    f'{len(group_boxes)} 个人框 ({labels})'
                ),
                group_id=group_id,
                label='person_box',
                shape_indices=[box.shape_idx for box in group_boxes],
                point_indices=[],
            ))

    return issues


def _find_missing_person_boxes(boxes: list[_BoxShape],
                               points: list[_PointShape]) -> list[ReviewIssue]:
    if not _rule_enabled('missing_person_box'):
        return []

    boxes_by_group = _boxes_by_group(boxes)
    points_by_group = _points_by_group(points)
    issues = []

    for group_id, group_points in sorted(
        points_by_group.items(), key=lambda item: _group_sort_key(item[0])
    ):
        if _is_missing_group_id(group_id) or group_id in boxes_by_group:
            continue

        group_text = _format_group_id(group_id)
        labels = ', '.join(point.label for point in group_points[:6])
        if len(group_points) > 6:
            labels += ', ...'
        issues.append(ReviewIssue(
            rule='missing_person_box',
            severity='error',
            message=(
                f'缺失人框: group_id={group_text} 有 '
                f'{len(group_points)} 个关键点，但没有对应人框 ({labels})'
            ),
            group_id=group_id,
            label='person_box',
            shape_indices=[point.shape_idx for point in group_points],
            point_indices=[(point.shape_idx, 0) for point in group_points],
        ))

    return issues


def _find_keypoint_box_issues(boxes: list[_BoxShape],
                              points: list[_PointShape]) -> list[ReviewIssue]:
    if (
        not _rule_enabled('keypoint_outside_box')
        and not _rule_enabled('keypoint_wrong_person')
    ):
        return []

    boxes_by_group = _boxes_by_group(boxes)
    issues = []

    for point in points:
        if _is_missing_group_id(point.group_id):
            continue

        own_boxes = boxes_by_group.get(point.group_id, [])
        other_boxes = [
            box for box in boxes
            if box.group_id != point.group_id
            and _point_in_rect(point.point, box.rect, _rect_margin(box.rect))
        ]

        if not own_boxes:
            if other_boxes and _rule_enabled('keypoint_wrong_person'):
                nearest = _nearest_box(point.point, other_boxes)
                issues.append(_wrong_person_issue(point, None, nearest))
            continue

        if len(own_boxes) > 1:
            continue

        own_box = own_boxes[0]
        own_margin = _rect_margin(own_box.rect)
        if _point_in_rect(point.point, own_box.rect, own_margin):
            continue

        if other_boxes and _rule_enabled('keypoint_wrong_person'):
            issues.append(_wrong_person_issue(point, own_box, _nearest_box(
                point.point, other_boxes
            )))
        elif _rule_enabled('keypoint_outside_box'):
            group_text = _format_group_id(point.group_id)
            issues.append(ReviewIssue(
                rule='keypoint_outside_box',
                severity='error',
                message=(
                    f'关键点在人框外: group_id={group_text} 的 '
                    f'{point.label} 不在对应人框内'
                ),
                group_id=point.group_id,
                label=point.label,
                shape_indices=[own_box.shape_idx, point.shape_idx],
                point_indices=[(point.shape_idx, 0)],
            ))

    return issues


def _wrong_person_issue(point: _PointShape,
                        own_box: _BoxShape | None,
                        other_box: _BoxShape) -> ReviewIssue:
    group_text = _format_group_id(point.group_id)
    other_group_text = _format_group_id(other_box.group_id)
    shape_indices = [point.shape_idx, other_box.shape_idx]
    if own_box is not None:
        shape_indices.insert(0, own_box.shape_idx)

    return ReviewIssue(
        rule='keypoint_wrong_person',
        severity='error',
        message=(
            f'关键点疑似归属错误: group_id={group_text} 的 {point.label} '
            f'落在 group_id={other_group_text} 的人框内'
        ),
        group_id=point.group_id,
        label=point.label,
        shape_indices=shape_indices,
        point_indices=[(point.shape_idx, 0)],
    )


def _find_suspected_left_right_swaps(shapes: list[dict]) -> list[ReviewIssue]:
    if not _rule_enabled('suspected_left_right_swap'):
        return []

    instances: dict[Any, dict[str, list[_PointShape]]] = {}
    for shape_idx, shape in enumerate(shapes):
        point_shape = _point_shape_from_json(shape_idx, shape)
        if point_shape is None:
            continue
        instances.setdefault(point_shape.group_id, {}).setdefault(
            point_shape.label, []
        ).append(point_shape)

    issues = []
    for group_id, points_by_label in instances.items():
        unique_points = {
            label: point_shapes[0]
            for label, point_shapes in points_by_label.items()
            if len(point_shapes) == 1
        }
        if len(unique_points) < int(_threshold('left_right_min_points')):
            continue

        bbox_diag = _instance_diag(unique_points)
        margin = max(
            _threshold('left_right_margin_min'),
            bbox_diag * _threshold('left_right_margin_ratio'),
        )

        for left_label, right_label in LEFT_RIGHT_PAIRS:
            left_point = unique_points.get(left_label)
            right_point = unique_points.get(right_label)
            if left_point is None or right_point is None:
                continue

            left_neighbors = _side_neighbors(left_label, unique_points)
            right_neighbors = _side_neighbors(right_label, unique_points)
            if len(left_neighbors) < 2 or len(right_neighbors) < 2:
                continue

            current = (
                _neighbor_score(left_point.point, left_neighbors)
                + _neighbor_score(right_point.point, right_neighbors)
            )
            swapped = (
                _neighbor_score(right_point.point, left_neighbors)
                + _neighbor_score(left_point.point, right_neighbors)
            )

            score_ratio = _threshold('left_right_score_ratio')
            if current > swapped * score_ratio and current - swapped > margin:
                group_text = _format_group_id(group_id)
                issues.append(ReviewIssue(
                    rule='suspected_left_right_swap',
                    severity='warning',
                    message=(
                        f'疑似左右反标: group_id={group_text} 的 '
                        f'{left_label} / {right_label} 可能互换'
                    ),
                    group_id=group_id,
                    label=f'{left_label}/{right_label}',
                    shape_indices=[left_point.shape_idx, right_point.shape_idx],
                    point_indices=[
                        (left_point.shape_idx, 0),
                        (right_point.shape_idx, 0),
                    ],
                ))

    return issues


def _find_custom_rule_issues(context: ReviewContext) -> list[ReviewIssue]:
    issues = []
    for rule in CUSTOM_REVIEW_RULES:
        if not bool(rule.get('enabled', True)):
            continue

        rule_type = str(rule.get('type', '')).strip()
        handler = CUSTOM_RULE_HANDLERS.get(rule_type)
        if handler is None:
            if rule_type == 'python':
                issues.extend(_find_python_custom_rule_issues(context, rule))
            else:
                issues.append(_unavailable_custom_rule_issue(rule))
            continue

        try:
            issues.extend(handler(context, rule))
        except Exception as exc:
            issues.append(_custom_rule_error_issue(rule, exc))
    return issues


def _find_unavailable_configured_rules() -> list[ReviewIssue]:
    custom_rule_ids = {
        _custom_rule_id(rule)
        for rule in CUSTOM_REVIEW_RULES
        if bool(rule.get('enabled', True))
    }
    issues = []
    for rule in sorted(ENABLED_REVIEW_RULES):
        if rule in SUPPORTED_REVIEW_RULES or rule in custom_rule_ids:
            continue
        issues.append(ReviewIssue(
            rule='unavailable_rule',
            severity='warning',
            message=(
                f'规则未执行: {rule} 没有对应的内置算法。'
                '请将它写入 custom_rules，或添加 Python 插件执行器。'
            ),
            group_id=None,
            label=rule,
            shape_indices=[],
            point_indices=[],
        ))

    if (
        _rule_enabled('suspected_left_right_swap')
        and LEFT_RIGHT_PAIRS
        and not KPT_CONNECTION_LABELS
    ):
        issues.append(ReviewIssue(
            rule='unavailable_rule',
            severity='warning',
            message=(
                '规则未执行: suspected_left_right_swap 缺少可用的骨架连接，'
                '请在当前 Pose 模板中配置 kpt_connections。'
            ),
            group_id=None,
            label='suspected_left_right_swap',
            shape_indices=[],
            point_indices=[],
        ))
    return issues


def _custom_required_keypoints(context: ReviewContext,
                               rule: dict[str, Any]) -> list[ReviewIssue]:
    labels = _custom_labels(rule, 'labels')
    if not labels:
        return []

    issues = []
    for group_id in _custom_rule_group_ids(context, rule):
        existing = {point.label for point in context.points_in_group(group_id)}
        missing = [label for label in labels if label not in existing]
        if not missing:
            continue

        group_text = _format_group_id(group_id)
        boxes = context.boxes_in_group(group_id)
        issues.append(ReviewIssue(
            rule=_custom_rule_id(rule),
            severity=_custom_severity(rule),
            message=(
                f'{_custom_rule_name(rule)}: group_id={group_text} '
                f'缺失关键点 {", ".join(missing)}'
            ),
            group_id=group_id,
            label=','.join(missing),
            shape_indices=[box.shape_idx for box in boxes],
            point_indices=[],
        ))
    return issues


def _custom_forbidden_keypoints(context: ReviewContext,
                                rule: dict[str, Any]) -> list[ReviewIssue]:
    labels = set(_custom_labels(rule, 'labels'))
    if not labels:
        return []

    issues = []
    for group_id in _custom_rule_group_ids(context, rule):
        for point in context.points_in_group(group_id):
            if point.label not in labels:
                continue
            group_text = _format_group_id(group_id)
            issues.append(ReviewIssue(
                rule=_custom_rule_id(rule),
                severity=_custom_severity(rule),
                message=(
                    f'{_custom_rule_name(rule)}: group_id={group_text} '
                    f'不应出现关键点 {point.label}'
                ),
                group_id=group_id,
                label=point.label,
                shape_indices=[point.shape_idx],
                point_indices=[(point.shape_idx, 0)],
            ))
    return issues


def _custom_paired_keypoints(context: ReviewContext,
                             rule: dict[str, Any]) -> list[ReviewIssue]:
    pairs = _custom_pairs(rule)
    if not pairs:
        return []

    issues = []
    for group_id in _custom_rule_group_ids(context, rule):
        labels = {point.label for point in context.points_in_group(group_id)}
        for left_label, right_label in pairs:
            has_left = left_label in labels
            has_right = right_label in labels
            if has_left == has_right:
                continue
            present_label = left_label if has_left else right_label
            present = context.point(group_id, present_label)
            group_text = _format_group_id(group_id)
            issues.append(ReviewIssue(
                rule=_custom_rule_id(rule),
                severity=_custom_severity(rule),
                message=(
                    f'{_custom_rule_name(rule)}: group_id={group_text} '
                    f'{left_label}/{right_label} 需要成对出现'
                ),
                group_id=group_id,
                label=f'{left_label}/{right_label}',
                shape_indices=[present.shape_idx] if present else [],
                point_indices=[(present.shape_idx, 0)] if present else [],
            ))
    return issues


def _custom_relative_position(context: ReviewContext,
                              rule: dict[str, Any]) -> list[ReviewIssue]:
    point_a_label = str(_custom_value(rule, 'point_a', '')).strip()
    point_b_label = str(_custom_value(rule, 'point_b', '')).strip()
    relation = str(_custom_value(rule, 'relation', '')).strip()
    margin = _custom_float(rule, 'margin', 0.0)
    if not point_a_label or not point_b_label or not relation:
        return []

    issues = []
    for group_id in _custom_rule_group_ids(context, rule):
        point_a = context.point(group_id, point_a_label)
        point_b = context.point(group_id, point_b_label)
        if point_a is None or point_b is None:
            continue

        if _relative_position_ok(point_a.point, point_b.point, relation, margin):
            continue

        group_text = _format_group_id(group_id)
        issues.append(ReviewIssue(
            rule=_custom_rule_id(rule),
            severity=_custom_severity(rule),
            message=(
                f'{_custom_rule_name(rule)}: group_id={group_text} '
                f'{point_a_label} 未满足相对位置 {relation} {point_b_label}'
            ),
            group_id=group_id,
            label=f'{point_a_label}/{point_b_label}',
            shape_indices=[point_a.shape_idx, point_b.shape_idx],
            point_indices=[(point_a.shape_idx, 0), (point_b.shape_idx, 0)],
        ))
    return issues


def _custom_distance_range(context: ReviewContext,
                           rule: dict[str, Any]) -> list[ReviewIssue]:
    point_a_label = str(_custom_value(rule, 'point_a', '')).strip()
    point_b_label = str(_custom_value(rule, 'point_b', '')).strip()
    min_distance = _custom_optional_float(rule, 'min_distance')
    max_distance = _custom_optional_float(rule, 'max_distance')
    if not point_a_label or not point_b_label:
        return []

    issues = []
    for group_id in _custom_rule_group_ids(context, rule):
        point_a = context.point(group_id, point_a_label)
        point_b = context.point(group_id, point_b_label)
        if point_a is None or point_b is None:
            continue

        distance = _distance(point_a.point, point_b.point)
        too_short = min_distance is not None and distance < min_distance
        too_long = max_distance is not None and distance > max_distance
        if not too_short and not too_long:
            continue

        group_text = _format_group_id(group_id)
        limits = []
        if min_distance is not None:
            limits.append(f'最小 {min_distance:g}')
        if max_distance is not None:
            limits.append(f'最大 {max_distance:g}')
        issues.append(ReviewIssue(
            rule=_custom_rule_id(rule),
            severity=_custom_severity(rule),
            message=(
                f'{_custom_rule_name(rule)}: group_id={group_text} '
                f'{point_a_label}/{point_b_label} 距离 {distance:.1f} '
                f'超出范围 ({", ".join(limits)})'
            ),
            group_id=group_id,
            label=f'{point_a_label}/{point_b_label}',
            shape_indices=[point_a.shape_idx, point_b.shape_idx],
            point_indices=[(point_a.shape_idx, 0), (point_b.shape_idx, 0)],
        ))
    return issues


def _find_python_custom_rule_issues(context: ReviewContext,
                                    rule: dict[str, Any]) -> list[ReviewIssue]:
    try:
        func = _load_python_rule_callable(rule, context.config_path)
        raw_issues = func(context, rule) or []
    except Exception as exc:
        return [_custom_rule_error_issue(rule, exc)]

    issues = []
    for raw_issue in raw_issues:
        try:
            issues.append(_coerce_custom_issue(raw_issue, rule))
        except Exception as exc:
            issues.append(_custom_rule_error_issue(rule, exc))
    return issues


CUSTOM_RULE_HANDLERS = {
    'required_keypoints': _custom_required_keypoints,
    'forbidden_keypoints': _custom_forbidden_keypoints,
    'paired_keypoints': _custom_paired_keypoints,
    'relative_position': _custom_relative_position,
    'distance_range': _custom_distance_range,
}


def _point_shape_from_json(shape_idx: int, shape: Any) -> _PointShape | None:
    if not isinstance(shape, dict):
        return None

    label = str(shape.get('label', '')).strip()
    if label not in KEYPOINT_SET:
        return None

    points = shape.get('points', [])
    shape_type = str(
        shape.get('shape_type', shape.get('shape_type ', ''))
    ).strip()
    is_point = shape_type == 'point' or len(points) == 1
    if not is_point or not points:
        return None

    try:
        x = float(points[0][0])
        y = float(points[0][1])
    except (TypeError, ValueError, IndexError):
        return None

    return _PointShape(
        shape_idx=shape_idx,
        label=label,
        group_id=shape.get('group_id'),
        point=(x, y),
    )


def _box_shape_from_json(shape_idx: int, shape: Any) -> _BoxShape | None:
    if not isinstance(shape, dict):
        return None

    label = str(shape.get('label', '')).strip()
    if not _box_label_enabled(label):
        return None

    if _shape_type(shape) != 'rectangle':
        return None

    rect, error = _rectangle_from_shape(shape)
    if error is not None or rect is None:
        return None

    return _BoxShape(
        shape_idx=shape_idx,
        label=label,
        group_id=shape.get('group_id'),
        rect=rect,
    )


def _target_object_label_from_json(shape_idx: int, shape: Any) -> str | None:
    if not isinstance(shape, dict):
        return None

    if CURRENT_TASK_TYPE in {DEFAULT_TASK_TYPE, 'detection'}:
        box_shape = _box_shape_from_json(shape_idx, shape)
        return box_shape.label if box_shape is not None else None

    label = str(shape.get('label', '')).strip()
    if not label:
        return None

    if CURRENT_TASK_TYPE == 'obb':
        if _shape_type(shape) != 'rotation':
            return None
        _points, error = _rotation_points_from_shape(shape)
        return label if error is None else None

    if CURRENT_TASK_TYPE == 'segmentation':
        if _shape_type(shape) != 'polygon':
            return None
        _points, error = _polygon_points_from_shape(shape)
        return label if error is None else None

    return None


def _box_label_enabled(label: str) -> bool:
    if CURRENT_TASK_TYPE == DEFAULT_TASK_TYPE:
        return label in TARGET_CLASS_SET
    if CURRENT_TASK_TYPE == 'detection':
        return bool(label)
    return bool(TARGET_CLASS_SET and label in TARGET_CLASS_SET)


def _shape_type(shape: dict) -> str:
    return str(shape.get('shape_type', shape.get('shape_type ', ''))).strip()


def _rectangle_from_shape(shape: dict
                          ) -> tuple[tuple[float, float, float, float] | None,
                                     str | None]:
    points = shape.get('points', [])
    if not isinstance(points, list):
        return None, 'points 不是数组'
    if len(points) < 2:
        return None, f'points 数量不足: {len(points)}'

    coords = []
    for idx, point in enumerate(points):
        try:
            coords.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError, IndexError):
            return None, f'points[{idx}] 坐标无效'

    xs = [point[0] for point in coords]
    ys = [point[1] for point in coords]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x1 == x2 or y1 == y2:
        return None, '矩形宽度或高度为 0'

    return (x1, y1, x2, y2), None


def _rotation_points_from_shape(shape: dict
                                ) -> tuple[list[tuple[float, float]] | None,
                                           str | None]:
    points = shape.get('points', [])
    if not isinstance(points, list):
        return None, 'points 不是数组'
    if len(points) != 4:
        return None, f'旋转框必须有 4 个顶点，当前={len(points)}'

    coords = []
    for idx, point in enumerate(points):
        try:
            coords.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError, IndexError):
            return None, f'points[{idx}] 坐标无效'

    if _polygon_area(coords) <= 1e-6:
        return None, '旋转框面积为 0'
    return coords, None


def _polygon_points_from_shape(shape: dict
                               ) -> tuple[list[tuple[float, float]] | None,
                                          str | None]:
    points = shape.get('points', [])
    if not isinstance(points, list):
        return None, 'points 不是数组'
    if len(points) < 3:
        return None, f'分割多边形至少需要 3 个点，当前={len(points)}'

    coords = []
    for idx, point in enumerate(points):
        try:
            coords.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError, IndexError):
            return None, f'points[{idx}] 坐标无效'

    if _polygon_area(coords) <= 1e-6:
        return None, '分割多边形面积为 0'
    return coords, None


def _polygon_coords_for_quality(shape: dict
                                ) -> tuple[list[tuple[float, float]] | None,
                                           str | None]:
    points = shape.get('points', [])
    if not isinstance(points, list):
        return None, 'points 不是数组'
    if len(points) < 3:
        return None, f'分割多边形至少需要 3 个点，当前={len(points)}'

    coords = []
    for idx, point in enumerate(points):
        try:
            coords.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError, IndexError):
            return None, f'points[{idx}] 坐标无效'
    return coords, None


def _polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _edge_lengths(points: list[tuple[float, float]]) -> list[float]:
    return [
        _distance(point, points[(idx + 1) % len(points)])
        for idx, point in enumerate(points)
    ]


def _obb_has_duplicate_points(points: list[tuple[float, float]]) -> bool:
    return _points_have_duplicates(points)


def _points_have_duplicates(points: list[tuple[float, float]]) -> bool:
    return bool(_duplicate_point_pairs(points))


def _duplicate_point_pairs(points: list[tuple[float, float]]
                           ) -> list[tuple[int, int]]:
    return [
        (left_idx, right_idx)
        for left_idx in range(len(points))
        for right_idx in range(left_idx + 1, len(points))
        if _distance(points[left_idx], points[right_idx]) <= 1e-6
    ]


def _polygon_self_intersections(points: list[tuple[float, float]]
                                ) -> list[tuple[int, int]]:
    intersections = []
    count = len(points)
    if count < 4:
        return intersections

    for left_idx in range(count):
        left_next = (left_idx + 1) % count
        for right_idx in range(left_idx + 1, count):
            right_next = (right_idx + 1) % count
            if left_next == right_idx or right_next == left_idx:
                continue
            if left_idx == 0 and right_next == 0:
                continue
            if _segments_intersect(
                points[left_idx],
                points[left_next],
                points[right_idx],
                points[right_next],
            ):
                intersections.append((left_idx, right_idx))
    return intersections


def _quad_corner_order_valid(points: list[tuple[float, float]]) -> bool:
    if len(points) != 4:
        return False
    if _quad_self_intersects(points):
        return False
    return _quad_is_convex(points)


def _quad_self_intersects(points: list[tuple[float, float]]) -> bool:
    return (
        _segments_intersect(points[0], points[1], points[2], points[3])
        or _segments_intersect(points[1], points[2], points[3], points[0])
    )


def _segments_intersect(a: tuple[float, float],
                        b: tuple[float, float],
                        c: tuple[float, float],
                        d: tuple[float, float]) -> bool:
    eps = 1e-6
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)

    if abs(o1) <= eps and _on_segment(a, c, b):
        return True
    if abs(o2) <= eps and _on_segment(a, d, b):
        return True
    if abs(o3) <= eps and _on_segment(c, a, d):
        return True
    if abs(o4) <= eps and _on_segment(c, b, d):
        return True
    return (o1 > eps) != (o2 > eps) and (o3 > eps) != (o4 > eps)


def _quad_is_convex(points: list[tuple[float, float]]) -> bool:
    eps = 1e-6
    signs = []
    for idx in range(4):
        a = points[idx]
        b = points[(idx + 1) % 4]
        c = points[(idx + 2) % 4]
        cross = _orientation(a, b, c)
        if abs(cross) <= eps:
            return False
        signs.append(cross > 0)
    return all(sign == signs[0] for sign in signs)


def _orientation(a: tuple[float, float],
                 b: tuple[float, float],
                 c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: tuple[float, float],
                p: tuple[float, float],
                b: tuple[float, float]) -> bool:
    eps = 1e-6
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def _obb_side_lengths(points: list[tuple[float, float]]) -> tuple[float, float]:
    lengths = _edge_lengths(points)
    if len(lengths) != 4:
        return 0.0, 0.0
    return (lengths[0] + lengths[2]) / 2.0, (lengths[1] + lengths[3]) / 2.0


def _rect_area(rect: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = rect
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _rect_iou(left: tuple[float, float, float, float],
              right: tuple[float, float, float, float]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = _rect_area((ix1, iy1, ix2, iy2))
    if intersection <= 0:
        return 0.0
    union = _rect_area(left) + _rect_area(right) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _collect_person_boxes_and_points(
    shapes: list[dict],
) -> tuple[list[_BoxShape], list[_PointShape]]:
    boxes = []
    points = []
    for shape_idx, shape in enumerate(shapes):
        box_shape = _box_shape_from_json(shape_idx, shape)
        if box_shape is not None:
            boxes.append(box_shape)

        point_shape = _point_shape_from_json(shape_idx, shape)
        if point_shape is not None:
            points.append(point_shape)

    return boxes, points


def _shapes_key(data: dict) -> str | None:
    if isinstance(data.get('shapes'), list):
        return 'shapes'
    if isinstance(data.get('shapes '), list):
        return 'shapes '
    return None


def _boxes_by_group(boxes: list[_BoxShape]) -> dict[Any, list[_BoxShape]]:
    grouped: dict[Any, list[_BoxShape]] = {}
    for box in boxes:
        grouped.setdefault(box.group_id, []).append(box)
    return grouped


def _points_by_group(points: list[_PointShape]) -> dict[Any, list[_PointShape]]:
    grouped: dict[Any, list[_PointShape]] = {}
    for point in points:
        grouped.setdefault(point.group_id, []).append(point)
    return grouped


def _is_missing_group_id(group_id: Any) -> bool:
    return group_id is None or (isinstance(group_id, str) and not group_id.strip())


def _point_in_rect(point: tuple[float, float],
                   rect: tuple[float, float, float, float],
                   margin: float = 0.0) -> bool:
    x, y = point
    x1, y1, x2, y2 = rect
    return (
        x1 - margin <= x <= x2 + margin
        and y1 - margin <= y <= y2 + margin
    )


def _rect_margin(rect: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = rect
    diag = math.hypot(x2 - x1, y2 - y1)
    return max(
        _threshold('box_margin_min'),
        diag * _threshold('box_margin_ratio'),
    )


def _nearest_box(point: tuple[float, float],
                 boxes: list[_BoxShape]) -> _BoxShape:
    return min(boxes, key=lambda box: _distance_to_rect_center(point, box.rect))


def _distance_to_rect_center(point: tuple[float, float],
                             rect: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = rect
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    return _distance(point, center)


def _as_positive_int(value: Any) -> int | None:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _read_image_size(image_path: str | Path | None) -> tuple[int, int] | None:
    if not image_path:
        return None

    path = Path(image_path)
    if not path.is_file():
        return None

    try:
        from PIL import Image
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None


def _resolve_image_from_annotation(annotation_path: Path,
                                   data: dict) -> Path | None:
    image_name = data.get('imagePath')
    if not image_name:
        return None

    image_path = Path(str(image_name))
    if image_path.is_absolute() and image_path.is_file():
        return image_path

    candidates = [annotation_path.parent / image_path]
    parts = annotation_path.parts
    annotation_indexes = [
        idx for idx, part in enumerate(parts)
        if part == 'annotations' or part.startswith('annotations-')
    ]
    if annotation_indexes:
        ann_index = annotation_indexes[-1]
        root = Path(*parts[:ann_index])
        ann_root = Path(*parts[:ann_index + 1])
        rel_dir = annotation_path.parent.relative_to(ann_root)
        candidates.append(root / 'images' / rel_dir / image_path.name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _side_neighbors(label: str,
                    unique_points: dict[str, _PointShape]) -> list[tuple[float, float]]:
    neighbors = []
    for a, b in KPT_CONNECTION_LABELS:
        if a == label and b in unique_points:
            neighbors.append(unique_points[b].point)
        elif b == label and a in unique_points:
            neighbors.append(unique_points[a].point)
    return neighbors


def _neighbor_score(point: tuple[float, float],
                    neighbors: list[tuple[float, float]]) -> float:
    return sum(_distance(point, neighbor) for neighbor in neighbors)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _instance_diag(unique_points: dict[str, _PointShape]) -> float:
    xs = [point.point[0] for point in unique_points.values()]
    ys = [point.point[1] for point in unique_points.values()]
    if not xs or not ys:
        return 0.0
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def _rebuild_pose_indexes() -> None:
    KEYPOINT_SET.clear()
    KEYPOINT_SET.update(KEYPOINTS)
    KEYPOINT_INDEX.clear()
    KEYPOINT_INDEX.update({label: idx for idx, label in enumerate(KEYPOINTS)})
    TARGET_CLASS_SET.clear()
    TARGET_CLASS_SET.update(TARGET_CLASSES)
    KPT_CONNECTION_LABELS.clear()
    for a, b in KPT_CONNECTIONS:
        KPT_CONNECTION_LABELS.append((KEYPOINTS[a], KEYPOINTS[b]))


def _string_tuple(raw: Any, field: str) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise ValueError(f'{field} 必须是字符串数组')

    values = []
    for idx, value in enumerate(raw):
        if not isinstance(value, str):
            raise ValueError(f'{field}[{idx}] 必须是字符串')
        label = value.strip()
        if not label:
            raise ValueError(f'{field}[{idx}] 不能为空')
        values.append(label)
    return tuple(values)


def _simple_name(raw: Any, field: str) -> str:
    value = str(raw or '').strip()
    if not value:
        raise ValueError(f'{field} 不能为空')
    return value


def _annotation_dir_name(raw: Any) -> str:
    value = _simple_name(raw, 'annotation_dir')
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('annotation_dir 必须是数据根目录下的相对目录名')
    if any(part in ('', '.') for part in path.parts):
        raise ValueError('annotation_dir 不能包含空路径段')
    return value


def _connection_tuple(raw: Any,
                      keypoint_index: dict[str, int]
                      ) -> tuple[tuple[int, int], ...]:
    if not isinstance(raw, list):
        raise ValueError('kpt_connections 必须是二元数组列表')

    connections = []
    for idx, item in enumerate(raw):
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f'kpt_connections[{idx}] 必须包含两个端点')
        a = _connection_endpoint(item[0], keypoint_index, f'kpt_connections[{idx}][0]')
        b = _connection_endpoint(item[1], keypoint_index, f'kpt_connections[{idx}][1]')
        if a == b:
            raise ValueError(f'kpt_connections[{idx}] 两个端点不能相同')
        connections.append((a, b))
    return tuple(connections)


def _connection_endpoint(value: Any,
                         keypoint_index: dict[str, int],
                         field: str) -> int:
    if isinstance(value, str):
        label = value.strip()
        if label in keypoint_index:
            return keypoint_index[label]
        try:
            index = int(label)
        except ValueError as exc:
            raise ValueError(f'{field} 未知关键点: {value}') from exc
    elif isinstance(value, int) and not isinstance(value, bool):
        index = value
    else:
        raise ValueError(f'{field} 必须是关键点名称或索引')

    if index < 0 or index >= len(keypoint_index):
        raise ValueError(
            f'{field} 索引越界: {index}, keypoints 数量={len(keypoint_index)}'
        )
    return index


def _left_right_pair_tuple(raw: Any,
                           keypoint_index: dict[str, int]
                           ) -> tuple[tuple[str, str], ...]:
    if not isinstance(raw, list):
        raise ValueError('left_right_pairs 必须是二元字符串数组列表')

    pairs = []
    for idx, item in enumerate(raw):
        if not isinstance(item, list) and not isinstance(item, tuple):
            raise ValueError(f'left_right_pairs[{idx}] 必须包含两个关键点名称')
        if len(item) != 2:
            raise ValueError(f'left_right_pairs[{idx}] 必须包含两个关键点名称')
        left, right = str(item[0]).strip(), str(item[1]).strip()
        for label in (left, right):
            if label not in keypoint_index:
                raise ValueError(f'left_right_pairs[{idx}] 未知关键点: {label}')
        if left == right:
            raise ValueError(f'left_right_pairs[{idx}] 两个关键点不能相同')
        pairs.append((left, right))
    return tuple(pairs)


def _custom_rule_tuple(raw: Any,
                       keypoint_index: dict[str, int],
                       target_classes: set[str]) -> tuple[dict[str, Any], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError('custom_rules 必须是规则 object 数组')

    rules = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f'custom_rules[{idx}] 必须是 object')

        rule = _json_copy(item)
        rule_id = str(rule.get('id') or '').strip()
        rule_type = str(rule.get('type') or '').strip()
        if not rule_id:
            raise ValueError(f'custom_rules[{idx}].id 不能为空')
        if not rule_type:
            raise ValueError(f'custom_rules[{idx}].type 不能为空')

        severity = str(rule.get('severity', 'error')).strip()
        if severity not in {'error', 'warning'}:
            raise ValueError(f'custom_rules[{idx}].severity 必须是 error 或 warning')

        if rule_type in {
            'required_keypoints',
            'forbidden_keypoints',
            'relative_position',
            'distance_range',
            'paired_keypoints',
        }:
            _validate_custom_rule_labels(idx, rule, keypoint_index)
            _validate_custom_rule_target_classes(idx, rule, target_classes)
        rules.append(rule)
    return tuple(rules)


def _validate_custom_rule_labels(idx: int,
                                 rule: dict[str, Any],
                                 keypoint_index: dict[str, int]) -> None:
    rule_type = str(rule.get('type', '')).strip()
    labels = []
    if rule_type in {'required_keypoints', 'forbidden_keypoints'}:
        labels.extend(_custom_labels(rule, 'labels'))
    elif rule_type in {'relative_position', 'distance_range'}:
        for key in ('point_a', 'point_b'):
            label = str(_custom_value(rule, key, '')).strip()
            if label:
                labels.append(label)
    elif rule_type == 'paired_keypoints':
        for pair in _custom_pairs(rule):
            labels.extend(pair)

    for label in labels:
        if label not in keypoint_index:
            raise ValueError(f'custom_rules[{idx}] 未知关键点: {label}')


def _validate_custom_rule_target_classes(idx: int,
                                         rule: dict[str, Any],
                                         target_classes: set[str]) -> None:
    raw_classes = _custom_value(rule, 'target_classes', [])
    if raw_classes in (None, ''):
        return
    if isinstance(raw_classes, str):
        classes = [raw_classes.strip()]
    elif isinstance(raw_classes, list):
        classes = [str(value).strip() for value in raw_classes]
    else:
        raise ValueError(f'custom_rules[{idx}].target_classes 必须是字符串数组')

    for label in classes:
        if label and label not in target_classes:
            raise ValueError(f'custom_rules[{idx}] 未知人框类别: {label}')


def _copy_custom_rules(rules: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    return tuple(_json_copy(rule) for rule in rules)


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _default_rules_for_task(task_type: str) -> dict[str, bool]:
    rules = {rule: False for rule in SUPPORTED_REVIEW_RULES}
    task_rules = TASK_REVIEW_RULES.get(str(task_type or DEFAULT_TASK_TYPE))
    if task_rules is None:
        task_rules = ("image_size_missing", "image_size_mismatch")
    for rule in task_rules:
        rules[rule] = True
    return rules


def _rule_dict(raw: Any,
               base_rules: dict[str, bool] | None = None) -> dict[str, bool]:
    if isinstance(raw, list):
        rules = {rule: False for rule in SUPPORTED_REVIEW_RULES}
        for idx, item in enumerate(raw):
            rule = str(item).strip()
            rules[rule] = True
        return rules

    if not isinstance(raw, dict):
        raise ValueError('rules 必须是 object，或 enabled_rules 字符串数组')

    rules = dict(base_rules or DEFAULT_REVIEW_RULES)
    for rule, enabled in raw.items():
        rule_name = str(rule).strip()
        rules[rule_name] = bool(enabled)
    return rules


def _threshold_dict(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise ValueError('thresholds 必须是 object')

    thresholds = {}
    for name, value in raw.items():
        key = str(name).strip()
        if key not in DEFAULT_REVIEW_THRESHOLDS:
            raise ValueError(f'thresholds 中存在未知字段: {key}')
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'thresholds.{key} 必须是数字') from exc
        if parsed < 0:
            raise ValueError(f'thresholds.{key} 不能小于 0')
        thresholds[key] = parsed
    return thresholds


def _custom_rule_id(rule: dict[str, Any]) -> str:
    return str(
        rule.get('id')
        or rule.get('name')
        or rule.get('type')
        or 'custom_rule'
    ).strip()


def _custom_rule_name(rule: dict[str, Any]) -> str:
    return str(rule.get('name') or _custom_rule_id(rule)).strip()


def _custom_severity(rule: dict[str, Any]) -> str:
    severity = str(rule.get('severity', 'error')).strip()
    return severity if severity in {'error', 'warning'} else 'error'


def _custom_params(rule: dict[str, Any]) -> dict[str, Any]:
    params = rule.get('params', {})
    return params if isinstance(params, dict) else {}


def _custom_value(rule: dict[str, Any], key: str, default: Any = None) -> Any:
    params = _custom_params(rule)
    if key in params:
        return params.get(key)
    return rule.get(key, default)


def _custom_labels(rule: dict[str, Any], key: str) -> list[str]:
    raw = _custom_value(rule, key, [])
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, list):
        values = raw
    else:
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _custom_pairs(rule: dict[str, Any]) -> list[tuple[str, str]]:
    raw_pairs = _custom_value(rule, 'pairs', [])
    if not isinstance(raw_pairs, list):
        return []

    pairs = []
    for item in raw_pairs:
        if not isinstance(item, list) and not isinstance(item, tuple):
            continue
        if len(item) != 2:
            continue
        left, right = str(item[0]).strip(), str(item[1]).strip()
        if left and right and left != right:
            pairs.append((left, right))
    return pairs


def _custom_float(rule: dict[str, Any], key: str, default: float) -> float:
    value = _custom_value(rule, key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _custom_optional_float(rule: dict[str, Any], key: str) -> float | None:
    value = _custom_value(rule, key, None)
    if value in (None, ''):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _custom_rule_group_ids(context: ReviewContext,
                           rule: dict[str, Any]) -> list[Any]:
    target_classes = set(_custom_labels(rule, 'target_classes'))
    if not target_classes:
        return context.group_ids()

    group_ids = []
    for group_id in context.group_ids():
        boxes = context.boxes_in_group(group_id)
        if any(box.label in target_classes for box in boxes):
            group_ids.append(group_id)
    return group_ids


def _relative_position_ok(point_a: tuple[float, float],
                          point_b: tuple[float, float],
                          relation: str,
                          margin: float) -> bool:
    ax, ay = point_a
    bx, by = point_b
    if relation in {'above', '上方'}:
        return ay <= by - margin
    if relation in {'below', '下方'}:
        return ay >= by + margin
    if relation in {'left_of', 'left', '左侧'}:
        return ax <= bx - margin
    if relation in {'right_of', 'right', '右侧'}:
        return ax >= bx + margin
    return True


def _unavailable_custom_rule_issue(rule: dict[str, Any]) -> ReviewIssue:
    rule_type = str(rule.get('type', '')).strip() or '-'
    return ReviewIssue(
        rule='unavailable_rule',
        severity='warning',
        message=(
            f'规则未执行: {_custom_rule_name(rule)} 的 type={rule_type} '
            '当前没有对应执行器'
        ),
        group_id=None,
        label=_custom_rule_id(rule),
        shape_indices=[],
        point_indices=[],
    )


def _custom_rule_error_issue(rule: dict[str, Any], exc: Exception) -> ReviewIssue:
    return ReviewIssue(
        rule='custom_rule_error',
        severity='warning',
        message=f'自定义规则执行失败: {_custom_rule_name(rule)}; {exc}',
        group_id=None,
        label=_custom_rule_id(rule),
        shape_indices=[],
        point_indices=[],
    )


def _load_python_rule_callable(rule: dict[str, Any],
                               config_path: Path | None):
    path_value = _custom_value(rule, 'path', None)
    entry = str(_custom_value(rule, 'entry', '') or '').strip()
    function_name = str(_custom_value(rule, 'function', '') or '').strip()

    if entry and ':' in entry and not path_value:
        path_text, func_text = entry.split(':', 1)
        path_value = path_text.strip()
        function_name = function_name or func_text.strip()

    if path_value:
        function_name = function_name or 'check'
        plugin_path = _resolve_custom_plugin_path(str(path_value), config_path)
        if plugin_path is None:
            raise ValueError(f'插件文件不存在: {path_value}')

        module_name = f'_pose_review_plugin_{abs(hash(str(plugin_path)))}'
        spec = importlib.util.spec_from_file_location(module_name, plugin_path)
        if spec is None or spec.loader is None:
            raise ValueError(f'无法加载插件文件: {plugin_path}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        func = getattr(module, function_name, None)
    elif entry:
        module_name, sep, func_name = entry.rpartition('.')
        if not sep:
            raise ValueError('python 规则 entry 必须是 module.function 或 path.py:function')
        module = importlib.import_module(module_name)
        func = getattr(module, function_name or func_name, None)
    else:
        raise ValueError('python 规则需要 path/function 或 entry')

    if not callable(func):
        raise ValueError(f'插件函数不存在或不可调用: {function_name or entry}')
    return func


def _resolve_custom_plugin_path(path_text: str,
                                config_path: Path | None) -> Path | None:
    path = Path(path_text).expanduser()
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        if config_path is not None:
            candidates.append(config_path.parent / path)
        project_root = Path(__file__).resolve().parents[2]
        candidates.append(project_root / path)
        candidates.append(Path.cwd() / path)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _coerce_custom_issue(raw_issue: Any,
                         rule: dict[str, Any]) -> ReviewIssue:
    if isinstance(raw_issue, ReviewIssue):
        return raw_issue
    if not isinstance(raw_issue, dict):
        raise ValueError('插件返回的 issue 必须是 dict 或 ReviewIssue')

    return ReviewIssue(
        rule=str(raw_issue.get('rule') or _custom_rule_id(rule)),
        severity=str(raw_issue.get('severity') or _custom_severity(rule)),
        message=str(raw_issue.get('message') or _custom_rule_name(rule)),
        group_id=raw_issue.get('group_id'),
        label=str(raw_issue.get('label') or _custom_rule_id(rule)),
        shape_indices=[
            int(index)
            for index in raw_issue.get('shape_indices', [])
            if isinstance(index, int) and not isinstance(index, bool)
        ],
        point_indices=[
            (int(item[0]), int(item[1]))
            for item in raw_issue.get('point_indices', [])
            if isinstance(item, (list, tuple)) and len(item) == 2
        ],
    )


def _rule_enabled(rule: str) -> bool:
    return rule in ENABLED_REVIEW_RULES


def _threshold(name: str) -> float:
    return float(REVIEW_THRESHOLDS.get(
        name, DEFAULT_REVIEW_THRESHOLDS.get(name, 0.0)
    ))


def _format_group_id(group_id: Any) -> str:
    if group_id is None:
        return '未分组'
    return str(group_id)


def _group_sort_key(group_id: Any) -> str:
    if group_id is None:
        return ''
    return str(group_id)
