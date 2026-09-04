"""Synchronize edited JSON annotations across derived dataset copies."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path


class AnnotationSyncError(ValueError):
    """Raised when an edited annotation cannot be synchronized safely."""


@dataclass(frozen=True)
class AnnotationSyncResult:
    """Result of synchronizing one edited annotation."""

    source: Path
    canonical: Path | None
    updated: tuple[Path, ...] = ()
    unchanged: tuple[Path, ...] = ()
    errors: tuple[str, ...] = ()
    ambiguous: bool = False


def annotation_file_fingerprint(path: str | Path) -> str:
    """Return a content fingerprint, or an empty string for a missing file."""
    candidate = Path(path)
    try:
        payload = candidate.read_bytes()
    except OSError:
        return ''
    return hashlib.sha256(payload).hexdigest()


def synchronize_annotation_replicas(
    changed_annotation: str | Path,
    dataset_root: str | Path | None,
    annotation_dir: str = 'annotations',
) -> AnnotationSyncResult:
    """Copy an edited JSON to safely identified source and derived replicas.

    Training manifests are the primary identity source. Filename fallback is
    used only when exactly one raw annotation has that name.
    """
    source = Path(changed_annotation).expanduser().resolve()
    if not source.is_file():
        raise AnnotationSyncError(f'修改后的标注文件不存在: {source}')
    if source.suffix.lower() != '.json':
        raise AnnotationSyncError(f'只支持同步 JSON 标注文件: {source.name}')

    payload = source.read_bytes()
    _validate_annotation_json(payload, source)
    root = _resolve_dataset_root(source, dataset_root, annotation_dir)
    if root is None:
        return AnnotationSyncResult(
            source=source,
            canonical=None,
            errors=('无法确定数据项目根目录，未同步其他文件',),
        )

    groups = _manifest_replica_groups(root, annotation_dir)
    matching_groups = [
        (canonical, replicas)
        for canonical, replicas in groups.items()
        if source == canonical or source in replicas
    ]
    if len(matching_groups) > 1:
        return AnnotationSyncResult(
            source=source,
            canonical=None,
            ambiguous=True,
            errors=('多个来源清单指向当前标注，已停止自动同步',),
        )

    raw_candidates = _raw_annotation_candidates(
        root, annotation_dir, source.name
    )
    unique_raw = raw_candidates[0] if len(raw_candidates) == 1 else None
    manifest_replicas: set[Path] = set()
    if matching_groups:
        canonical, manifest_replicas = matching_groups[0]
    elif _is_within(source, root / annotation_dir):
        canonical = source
    elif unique_raw is not None:
        canonical = unique_raw
    else:
        message = (
            '存在多个同名原始标注，无法确认当前文件来源，已停止自动同步'
            if len(raw_candidates) > 1 else
            '未找到可追溯的原始标注，未同步其他文件'
        )
        return AnnotationSyncResult(
            source=source,
            canonical=None,
            ambiguous=len(raw_candidates) > 1,
            errors=(message,),
        )

    targets = set(manifest_replicas)
    targets.add(canonical)
    if unique_raw is not None and canonical == unique_raw:
        targets.update(_legacy_replica_candidates(
            root, annotation_dir, source.name
        ))
    targets.discard(source)

    updated = []
    unchanged = []
    errors = []
    for target in sorted(targets, key=str):
        if not _is_within(target, root) or not target.is_file():
            continue
        try:
            if target.read_bytes() == payload:
                unchanged.append(target)
                continue
            _atomic_replace(target, payload, source)
            updated.append(target)
        except OSError as exc:
            errors.append(f'{target}: {exc}')

    return AnnotationSyncResult(
        source=source,
        canonical=canonical,
        updated=tuple(updated),
        unchanged=tuple(unchanged),
        errors=tuple(errors),
    )


def _validate_annotation_json(payload: bytes, path: Path):
    try:
        document = json.loads(payload.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnnotationSyncError(
            f'修改后的标注不是有效 JSON，未同步其他文件: {path.name}\n{exc}'
        ) from exc
    if not isinstance(document, dict):
        raise AnnotationSyncError(
            f'修改后的标注根节点不是 JSON 对象，未同步: {path.name}'
        )
    shapes = document.get('shapes')
    if shapes is not None and not isinstance(shapes, list):
        raise AnnotationSyncError(
            f'修改后的标注 shapes 字段不是列表，未同步: {path.name}'
        )


def _resolve_dataset_root(
    source: Path,
    configured_root: str | Path | None,
    annotation_dir: str,
) -> Path | None:
    if configured_root:
        candidate = Path(configured_root).expanduser().resolve()
        normalized = _project_root_for_scope(candidate)
        if normalized is not None and _is_within(source, normalized):
            return normalized

    # Prefer a complete project root over a nested training batch, which also
    # has its own images/ and annotations/ directories.
    for ancestor in source.parents:
        if not (ancestor / 'images').is_dir():
            continue
        if (
            (ancestor / 'training_data').is_dir()
            or (ancestor / 'test_data').is_dir()
        ):
            return ancestor
    for ancestor in source.parents:
        if not (ancestor / 'images').is_dir():
            continue
        if (
            (ancestor / annotation_dir).is_dir()
            or (ancestor / 'training_data').is_dir()
            or (ancestor / 'test_data').is_dir()
        ):
            return ancestor
    return None


def _project_root_for_scope(candidate: Path) -> Path | None:
    if candidate.name in ('training_data', 'test_data'):
        project = candidate.parent
    elif candidate.parent.name in ('training_data', 'test_data'):
        project = candidate.parent.parent
    else:
        project = candidate
    return project if (project / 'images').is_dir() else None


def _manifest_replica_groups(
    root: Path,
    annotation_dir: str,
) -> dict[Path, set[Path]]:
    groups: dict[Path, set[Path]] = {}
    for scope_name in ('training_data', 'test_data'):
        scope = root / scope_name
        if not scope.is_dir():
            continue
        for manifest in scope.rglob('preparation_manifest.json'):
            try:
                document = json.loads(manifest.read_text(encoding='utf-8'))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            request = document.get('request') if isinstance(document, dict) else None
            request = request if isinstance(request, dict) else {}
            manifest_annotation_dir = str(
                request.get('annotation_dir') or 'annotations'
            )
            if manifest_annotation_dir != annotation_dir:
                continue
            records = document.get('records') if isinstance(document, dict) else None
            if not isinstance(records, list):
                continue
            for record in records:
                if not isinstance(record, dict):
                    continue
                raw_path = str(record.get('annotation_path') or '').strip()
                if not raw_path:
                    continue
                canonical = Path(raw_path).expanduser().resolve()
                if not _is_within(canonical, root):
                    source_name = str(record.get('source_name') or '').strip()
                    if not source_name:
                        continue
                    canonical = (
                        root / annotation_dir / source_name /
                        Path(raw_path).name
                    ).resolve()
                    if not _is_within(canonical, root):
                        continue
                replica_name = canonical.name
                stem = str(record.get('stem') or '').strip()
                if stem:
                    replica_name = f'{stem}.json'
                replica = (
                    manifest.parent / annotation_dir / replica_name
                ).resolve()
                if not _is_within(replica, root):
                    continue
                groups.setdefault(canonical, set()).add(replica)
    return groups


def _raw_annotation_candidates(
    root: Path,
    annotation_dir: str,
    filename: str,
) -> list[Path]:
    raw_root = root / annotation_dir
    if not raw_root.is_dir():
        return []
    return sorted(
        (
            path.resolve() for path in raw_root.rglob('*.json')
            if path.is_file() and path.name == filename
        ),
        key=str,
    )


def _legacy_replica_candidates(
    root: Path,
    annotation_dir: str,
    filename: str,
) -> set[Path]:
    candidates = set()
    training_root = root / 'training_data'
    if training_root.is_dir():
        for batch in training_root.iterdir():
            path = batch / annotation_dir / filename
            if batch.is_dir() and path.is_file():
                candidates.add(path.resolve())
    test_root = root / 'test_data'
    direct = test_root / annotation_dir / filename
    if direct.is_file():
        candidates.add(direct.resolve())
    if test_root.is_dir():
        for batch in test_root.iterdir():
            path = batch / annotation_dir / filename
            if batch.is_dir() and path.is_file():
                candidates.add(path.resolve())
    return candidates


def _atomic_replace(target: Path, payload: bytes, source: Path):
    mode = stat.S_IMODE(target.stat().st_mode)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='wb', prefix=f'.{target.name}.sync-',
            dir=str(target.parent), delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, mode)
        os.replace(temporary_name, target)
        shutil.copystat(source, target)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def synchronize_annotation_folder(
    annotation_set_dir: str | Path,
    dataset_root: str | Path | None = None,
) -> tuple[int, int]:
    """Sync every annotation JSON in a folder to its replicas.

    Returns (synced_files, failed_files).
    """
    ann_dir = Path(annotation_set_dir)
    if not ann_dir.is_dir():
        return 0, 0
    synced = 0
    failed = 0
    import json as _json
    for annotation_path in sorted(ann_dir.glob('*.json')):
        if not annotation_path.is_file():
            continue
        try:
            payload = annotation_path.read_bytes()
            _validate_annotation_json(payload, annotation_path)
            result = synchronize_annotation_replicas(
                annotation_path, dataset_root,
                annotation_dir=str(ann_dir.name),
            )
            synced += 1 if result else 0
        except (OSError, ValueError, AnnotationSyncError) as exc:
            failed += 1
    return synced, failed


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
