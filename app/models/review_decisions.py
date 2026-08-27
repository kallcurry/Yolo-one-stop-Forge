"""Persistent manual decisions layered on top of automatic review results."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
REVIEW_ENGINE_VERSION = 1
DECISION_RELATIVE_PATH = Path('.review') / 'review_decisions.json'


@dataclass(frozen=True)
class ReviewDecisionResult:
    """Automatic issues split into unresolved and manually accepted groups."""

    raw_issues: tuple[Any, ...]
    active_issues: tuple[Any, ...]
    accepted_issues: tuple[Any, ...]
    stale: bool = False

    @property
    def manually_passed(self) -> bool:
        return bool(self.raw_issues) and not self.active_issues


def review_template_fingerprint(config: Any) -> str:
    """Hash only review semantics, excluding names, paths and UI descriptions."""
    payload = {
        'task_type': str(getattr(config, 'task_type', '') or ''),
        'annotation_dir': str(getattr(config, 'annotation_dir', '') or ''),
        'keypoints': list(getattr(config, 'keypoints', ()) or ()),
        'target_classes': list(getattr(config, 'target_classes', ()) or ()),
        'kpt_connections': [
            list(item) for item in getattr(config, 'kpt_connections', ()) or ()
        ],
        'left_right_pairs': [
            list(item) for item in getattr(config, 'left_right_pairs', ()) or ()
        ],
        'rules': dict(getattr(config, 'rules', {}) or {}),
        'thresholds': dict(getattr(config, 'thresholds', {}) or {}),
        'custom_rules': [
            dict(item) for item in getattr(config, 'custom_rules', ()) or ()
        ],
        'review_engine_version': REVIEW_ENGINE_VERSION,
    }
    return _sha256_json(payload)


def issue_fingerprint(issue: Any) -> str:
    """Return a stable identity for one issue on unchanged annotation data."""
    payload = {
        'rule': str(getattr(issue, 'rule', '') or ''),
        'severity': str(getattr(issue, 'severity', '') or ''),
        'group_id': getattr(issue, 'group_id', None),
        'label': str(getattr(issue, 'label', '') or ''),
        'shape_indices': sorted(
            int(value) for value in getattr(issue, 'shape_indices', []) or []
        ),
        'point_indices': sorted(
            [int(pair[0]), int(pair[1])]
            for pair in getattr(issue, 'point_indices', []) or []
        ),
    }
    return _sha256_json(payload)


class ReviewDecisionStore:
    """Read and atomically update manual review decisions for one dataset."""

    def __init__(self, dataset_root: str | Path):
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.path = self.dataset_root / DECISION_RELATIVE_PATH
        self._data = self._load()

    def evaluate(self, image_path: str | Path, annotation_path: str | Path,
                 config: Any, issues: Iterable[Any]) -> ReviewDecisionResult:
        raw = tuple(issues)
        context = self._context(image_path, annotation_path, config)
        record = self._data['decisions'].get(context['id'], {})
        accepted_fingerprints = set(
            record.get('accepted_issue_fingerprints', []) or []
        )
        accepted = tuple(
            issue for issue in raw
            if issue_fingerprint(issue) in accepted_fingerprints
        )
        active = tuple(
            issue for issue in raw
            if issue_fingerprint(issue) not in accepted_fingerprints
        )
        return ReviewDecisionResult(
            raw_issues=raw,
            active_issues=active,
            accepted_issues=accepted,
            stale=(
                not bool(accepted_fingerprints)
                and self._has_stale_record(context)
            ),
        )

    def accept(self, image_path: str | Path, annotation_path: str | Path,
               config: Any, issues: Iterable[Any], scope: str = 'file',
               reason: str = '算法误报', note: str = '') -> int:
        selected = tuple(issues)
        if not selected:
            return 0
        context = self._context(image_path, annotation_path, config)
        decisions = self._data['decisions']
        previous = dict(decisions.get(context['id'], {}) or {})
        fingerprints = set(
            previous.get('accepted_issue_fingerprints', []) or []
        )
        before = len(fingerprints)
        fingerprints.update(issue_fingerprint(issue) for issue in selected)
        snapshots = dict(previous.get('issue_snapshots', {}) or {})
        for issue in selected:
            fingerprint = issue_fingerprint(issue)
            snapshots[fingerprint] = _issue_snapshot(issue)

        decisions[context['id']] = {
            **context,
            'scope': 'issue' if scope == 'issue' else 'file',
            'decision': 'accepted',
            'accepted_issue_fingerprints': sorted(fingerprints),
            'issue_snapshots': snapshots,
            'reason': str(reason or '算法误报'),
            'note': str(note or ''),
            'reviewed_at': datetime.now(timezone.utc).isoformat(),
        }
        self._save()
        return len(fingerprints) - before

    def revoke(self, image_path: str | Path, annotation_path: str | Path,
               config: Any) -> bool:
        context = self._context(image_path, annotation_path, config)
        matching_ids = [
            record_id
            for record_id, record in self._data['decisions'].items()
            if (
                record.get('task_type') == context['task_type']
                and record.get('annotation_dir') == context['annotation_dir']
                and record.get('image') == context['image']
            )
        ]
        if not matching_ids:
            return False
        for record_id in matching_ids:
            self._data['decisions'].pop(record_id, None)
        self._save()
        return True

    def _context(self, image_path: str | Path, annotation_path: str | Path,
                 config: Any) -> dict[str, Any]:
        image_key = self._portable_path(image_path)
        annotation_key = self._portable_path(annotation_path)
        annotation_hash = _sha256_file(annotation_path)
        template_hash = review_template_fingerprint(config)
        task_type = str(getattr(config, 'task_type', '') or '')
        annotation_dir = str(getattr(config, 'annotation_dir', '') or '')
        identity = {
            'task_type': task_type,
            'annotation_dir': annotation_dir,
            'image': image_key,
            'annotation': annotation_key,
            'annotation_hash': annotation_hash,
            'template_hash': template_hash,
            'review_engine_version': REVIEW_ENGINE_VERSION,
        }
        return {'id': _sha256_json(identity), **identity}

    def _has_stale_record(self, context: dict[str, Any]) -> bool:
        for record_id, record in self._data['decisions'].items():
            if record_id == context['id']:
                continue
            if (
                record.get('task_type') == context['task_type']
                and record.get('annotation_dir') == context['annotation_dir']
                and record.get('image') == context['image']
                and record.get('accepted_issue_fingerprints')
            ):
                return True
        return False

    def _portable_path(self, value: str | Path) -> str:
        path = Path(value).expanduser().resolve()
        try:
            return path.relative_to(self.dataset_root).as_posix()
        except ValueError:
            return path.as_posix()

    def _load(self) -> dict[str, Any]:
        empty = {'schema_version': SCHEMA_VERSION, 'decisions': {}}
        if not self.path.is_file():
            return empty
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            return empty
        if not isinstance(data, dict) or not isinstance(data.get('decisions'), dict):
            return empty
        return {
            'schema_version': SCHEMA_VERSION,
            'decisions': dict(data['decisions']),
        }

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            self._data, ensure_ascii=False, indent=2, sort_keys=True,
        ) + '\n'
        temporary = self.path.with_name(f'.{self.path.name}.tmp')
        temporary.write_text(payload, encoding='utf-8')
        os.replace(temporary, self.path)


def _issue_snapshot(issue: Any) -> dict[str, Any]:
    return {
        'rule': str(getattr(issue, 'rule', '') or ''),
        'severity': str(getattr(issue, 'severity', '') or ''),
        'message': str(getattr(issue, 'message', '') or ''),
        'group_id': getattr(issue, 'group_id', None),
        'label': str(getattr(issue, 'label', '') or ''),
        'shape_indices': list(getattr(issue, 'shape_indices', []) or []),
        'point_indices': [
            list(pair) for pair in getattr(issue, 'point_indices', []) or []
        ],
    }


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
        default=str,
    ).encode('utf-8')
    return hashlib.sha256(serialized).hexdigest()
