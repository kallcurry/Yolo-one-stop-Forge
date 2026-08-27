"""Infer annotation schema from dataset-owned files."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import yaml


@dataclass(frozen=True)
class AnnotationSchema:
    target_classes: tuple[str, ...] = ()
    keypoints: tuple[str, ...] = ()
    left_right_pairs: tuple[tuple[str, str], ...] = ()
    kpt_connections: tuple[tuple[int, int], ...] = ()


def infer_annotation_schema(
    annotation_paths: Iterable[str | Path],
    task_type: str = 'pose',
    dataset_yaml: str | Path | None = None,
) -> AnnotationSchema:
    """Infer class and keypoint structure without built-in label names."""
    task_type = str(task_type or 'pose').strip()
    yaml_classes, yaml_keypoints, yaml_pairs, yaml_connections = (
        _schema_from_dataset_yaml(dataset_yaml)
    )
    class_names = list(yaml_classes)
    class_seen = set(class_names)
    point_sequences: list[tuple[str, ...]] = []

    target_types = {
        'pose': {'rectangle'},
        'detection': {'rectangle'},
        'segmentation': {'polygon'},
        'obb': {'rotation'},
    }.get(task_type, set())

    for raw_path in annotation_paths:
        path = Path(raw_path)
        try:
            document = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        shapes = document.get('shapes', []) if isinstance(document, dict) else []
        if not isinstance(shapes, list):
            continue

        grouped_points: dict[str, list[str]] = defaultdict(list)
        for shape in shapes:
            if not isinstance(shape, dict):
                continue
            label = str(shape.get('label') or '').strip()
            shape_type = str(
                shape.get('shape_type', shape.get('shape_type ', '')) or ''
            ).strip()
            if not label:
                continue
            if shape_type in target_types and label not in class_seen:
                class_names.append(label)
                class_seen.add(label)
            if task_type == 'pose' and shape_type == 'point':
                group_key = _group_key(shape.get('group_id'))
                if label not in grouped_points[group_key]:
                    grouped_points[group_key].append(label)

        point_sequences.extend(
            tuple(sequence) for sequence in grouped_points.values() if sequence
        )

    keypoints = tuple(yaml_keypoints) or _consensus_order(point_sequences)
    left_right_pairs = tuple(yaml_pairs) or infer_left_right_pairs(keypoints)
    return AnnotationSchema(
        target_classes=tuple(class_names),
        keypoints=keypoints,
        left_right_pairs=left_right_pairs,
        kpt_connections=tuple(yaml_connections),
    )


def infer_left_right_pairs(
    keypoints: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    names = set(keypoints)
    pairs = []
    for name in keypoints:
        counterpart = ''
        if name == 'left':
            counterpart = 'right'
        elif name.startswith('left_'):
            counterpart = 'right_' + name[len('left_'):]
        elif name.endswith('_left'):
            counterpart = name[:-len('_left')] + '_right'
        if counterpart in names:
            pairs.append((name, counterpart))
    return tuple(pairs)


def _schema_from_dataset_yaml(
    raw_path: str | Path | None,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[tuple[str, str], ...],
    tuple[tuple[int, int], ...],
]:
    if raw_path is None:
        return (), (), (), ()
    path = Path(raw_path)
    if not path.is_file():
        return (), (), (), ()
    try:
        payload = yaml.safe_load(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, yaml.YAMLError):
        return (), (), (), ()
    if not isinstance(payload, dict):
        return (), (), (), ()

    names = payload.get('names')
    if isinstance(names, dict):
        indexed_names = []
        for raw_index, value in names.items():
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue
            name = str(value).strip()
            if index >= 0 and name:
                indexed_names.append((index, name))
        class_names = tuple(
            name for _index, name in sorted(indexed_names)
        )
    elif isinstance(names, (list, tuple)):
        class_names = tuple(str(value).strip() for value in names if str(value).strip())
    else:
        class_names = ()

    raw_keypoints = payload.get('keypoint_names')
    keypoints = (
        tuple(str(value).strip() for value in raw_keypoints if str(value).strip())
        if isinstance(raw_keypoints, (list, tuple)) else ()
    )
    pairs = []
    raw_flip = payload.get('flip_idx')
    if keypoints and isinstance(raw_flip, (list, tuple)):
        for index, raw_pair in enumerate(raw_flip):
            try:
                pair_index = int(raw_pair)
            except (TypeError, ValueError):
                continue
            if index < pair_index < len(keypoints):
                pairs.append((keypoints[index], keypoints[pair_index]))

    connections = []
    raw_connections = payload.get('kpt_connections', payload.get('skeleton'))
    if isinstance(raw_connections, (list, tuple)):
        for value in raw_connections:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                continue
            try:
                left, right = int(value[0]), int(value[1])
            except (TypeError, ValueError):
                continue
            if 0 <= left < len(keypoints) and 0 <= right < len(keypoints):
                connections.append((left, right))
    return class_names, keypoints, tuple(pairs), tuple(connections)


def _consensus_order(
    sequences: Sequence[Sequence[str]],
) -> tuple[str, ...]:
    """Build a stable order from partial keypoint sequences in annotations."""
    first_seen: dict[str, int] = {}
    votes: Counter[tuple[str, str]] = Counter()
    for sequence in sequences:
        unique = list(dict.fromkeys(str(value) for value in sequence if str(value)))
        for label in unique:
            first_seen.setdefault(label, len(first_seen))
        for index, left in enumerate(unique):
            for right in unique[index + 1:]:
                votes[(left, right)] += 1

    if not first_seen:
        return ()

    edges: dict[str, set[str]] = {name: set() for name in first_seen}
    indegree = {name: 0 for name in first_seen}
    for left in first_seen:
        for right in first_seen:
            if left == right:
                continue
            if votes[(left, right)] <= votes[(right, left)]:
                continue
            if right not in edges[left]:
                edges[left].add(right)
                indegree[right] += 1

    remaining = set(first_seen)
    ordered = []
    while remaining:
        available = sorted(
            (name for name in remaining if indegree[name] == 0),
            key=first_seen.get,
        )
        if not available:
            available = [min(remaining, key=first_seen.get)]
        name = available[0]
        remaining.remove(name)
        ordered.append(name)
        for neighbor in edges[name]:
            indegree[neighbor] = max(0, indegree[neighbor] - 1)
    return tuple(ordered)


def _group_key(group_id: object) -> str:
    try:
        return json.dumps(group_id, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return repr(group_id)
