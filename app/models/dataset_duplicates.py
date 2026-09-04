"""Duplicate detection for the raw data trees of a dataset project.

The data manager keeps raw sources in three sibling directories::

    images/<batch>/...
    annotations/<batch>/...
    labels/<batch>/...

This module intentionally does not walk ``training_data`` or ``test_data``.
Those are derived datasets and are already covered by their own tools.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


from app.models.app_defaults import image_extensions as _image_extensions

IMAGE_EXTENSIONS = _image_extensions()
RAW_DATA_DIRS = (
    ('images', 'image'),
    ('annotations', 'annotation'),
    ('labels', 'label'),
)
_DERIVED_DATA_DIRS = {'training_data', 'test_data'}
_HASH_CHUNK_SIZE = 1024 * 1024
_QUICK_FINGERPRINT_SIZE = 64 * 1024
_HASH_CACHE: dict[str, tuple[int, int, str]] = {}


@dataclass(frozen=True)
class DuplicateGroup:
    """A group of files with exactly the same byte content."""

    kind: str
    digest: str
    files: tuple[Path, ...]
    size: int

    @property
    def keeper(self) -> Path:
        """The deterministic file kept when this group is deleted."""
        return self.files[0]

    @property
    def duplicates(self) -> tuple[Path, ...]:
        return self.files[1:]

    @property
    def reclaimable_bytes(self) -> int:
        return self.size * len(self.duplicates)


@dataclass(frozen=True)
class OrphanRecord:
    """An annotation/label file with no same-stem image in its batch dir."""

    kind: str
    path: Path
    batch_dir: str = ''


@dataclass(frozen=True)
class NameConflictMember:
    """One location of a same-stem annotation/label file."""

    path: Path
    digest: str
    size: int


@dataclass(frozen=True)
class NameConflictGroup:
    """Same stem in several directories with **different** byte content.

    Name conflicts are reported per kind (annotation or label) and never when
    every location is byte-identical: identical copies already form duplicate
    groups and are handled by the duplicate workflows.
    """

    kind: str
    stem: str
    members: tuple[NameConflictMember, ...]


@dataclass(frozen=True)
class DuplicateScanResult:
    """Immutable result returned by :func:`scan_raw_duplicates`."""

    root: Path
    scanned_counts: dict[str, int]
    groups: dict[str, tuple[DuplicateGroup, ...]]
    errors: tuple[str, ...] = ()
    orphans: tuple[OrphanRecord, ...] = ()
    name_conflicts: tuple[NameConflictGroup, ...] = ()

    @property
    def all_groups(self) -> tuple[DuplicateGroup, ...]:
        return tuple(
            group
            for _name, _kind in RAW_DATA_DIRS
            for group in self.groups.get(_kind, ())
        )

    @property
    def duplicate_group_count(self) -> int:
        return len(self.all_groups)

    @property
    def duplicate_file_count(self) -> int:
        return sum(len(group.duplicates) for group in self.all_groups)

    @property
    def reclaimable_bytes(self) -> int:
        return sum(group.reclaimable_bytes for group in self.all_groups)

    @property
    def orphan_count(self) -> int:
        return len(self.orphans)

    @property
    def name_conflict_count(self) -> int:
        return len(self.name_conflicts)


@dataclass(frozen=True)
class DeleteResult:
    """Outcome of deleting duplicate members from one or more groups."""

    deleted: tuple[Path, ...]
    errors: tuple[str, ...]
    skipped: tuple[str, ...] = ()


@dataclass(frozen=True)
class MoveResult:
    """Outcome of moving files into a backup directory."""

    moved: tuple[Path, ...]
    errors: tuple[str, ...]


def scan_raw_duplicates(
    root: str | Path,
    progress: Callable[[int, int, str], None] | None = None,
) -> DuplicateScanResult:
    """Scan raw images, JSON annotations and TXT labels by exact content.

    Files are grouped by ``(kind, byte_size, SHA-256)``.  The byte-level
    comparison is deliberate: it is explainable to the user and cannot mark
    two visually similar but semantically different samples as duplicates.
    Missing ``annotations`` or ``labels`` directories are allowed because a
    project may currently contain only one task format.
    """

    dataset_root = resolve_raw_dataset_root(root)
    if not dataset_root.is_dir():
        raise ValueError(f'数据根目录不存在或不是目录: {dataset_root}')

    # First collect metadata only.  Most files have a unique size and never
    # need to be opened at all.
    size_buckets: dict[str, dict[int, list[tuple[Path, int]]]] = {
        kind: {} for _name, kind in RAW_DATA_DIRS
    }
    counts = {kind: 0 for _name, kind in RAW_DATA_DIRS}
    errors: list[str] = []
    discovered = 0
    all_paths_by_kind: dict[str, list[Path]] = {kind: [] for _n, kind in RAW_DATA_DIRS}
    image_stems_by_dir: dict[str, set[str]] = {}

    for directory_name, kind in RAW_DATA_DIRS:
        source_dir = dataset_root / directory_name
        if not source_dir.is_dir():
            continue
        for path in _iter_raw_files(source_dir, kind):
            try:
                stat = path.stat()
            except (OSError, PermissionError) as exc:
                errors.append(f'{path}: {exc}')
                continue

            counts[kind] += 1
            discovered += 1
            all_paths_by_kind[kind].append(path)
            size_buckets[kind].setdefault(stat.st_size, []).append(
                (path, stat.st_mtime_ns)
            )

    # Index image stems by batch directory: orphan checks need the same-batch
    # sibling.  ``dir_key`` is the relative path between the top-level raw
    # directory and the file name, e.g. ``batch-a/frames``.
    for path in all_paths_by_kind.get('image', ()):
        parts = path.relative_to(dataset_root).as_posix().split('/')
        dir_key = '/'.join(parts[1:-1]) if len(parts) >= 2 else ''
        image_stems_by_dir.setdefault(dir_key, set()).add(Path(parts[-1]).stem)

    if progress:
        progress(discovered, discovered, '已完成文件枚举，正在筛选相同大小文件')

    # Same-size files are still common for fixed-camera datasets.  A small
    # head/tail fingerprint avoids reading the complete image in most cases.
    quick_buckets: dict[str, dict[tuple[int, str], list[tuple[Path, int]]]] = {
        kind: {} for _name, kind in RAW_DATA_DIRS
    }
    quick_candidates = sum(
        len(files)
        for buckets in size_buckets.values()
        for files in buckets.values()
        if len(files) > 1
    )
    processed = 0
    for _directory_name, kind in RAW_DATA_DIRS:
        for size, files in size_buckets[kind].items():
            if len(files) < 2:
                continue
            for path, mtime_ns in files:
                try:
                    quick = _quick_fingerprint(path, size)
                except (OSError, PermissionError) as exc:
                    errors.append(f'{path}: {exc}')
                    continue
                quick_buckets[kind].setdefault((size, quick), []).append(
                    (path, mtime_ns)
                )
                processed += 1
                if progress:
                    progress(
                        processed,
                        max(quick_candidates, 1),
                        '正在快速比对候选文件',
                    )

    fingerprints: dict[str, dict[tuple[int, str], list[Path]]] = {
        kind: {} for _name, kind in RAW_DATA_DIRS
    }
    full_candidates = sum(
        len(files)
        for buckets in quick_buckets.values()
        for files in buckets.values()
        if len(files) > 1
    )
    hashed = 0
    for _directory_name, kind in RAW_DATA_DIRS:
        for (size, quick), files in quick_buckets[kind].items():
            if len(files) < 2:
                continue
            for path, mtime_ns in files:
                try:
                    digest = _cached_sha256(path, size, mtime_ns)
                except (OSError, PermissionError) as exc:
                    errors.append(f'{path}: {exc}')
                    continue
                fingerprints[kind].setdefault((size, digest), []).append(path)
                hashed += 1
                if progress:
                    progress(
                        hashed,
                        max(full_candidates, 1),
                        '正在计算候选文件完整指纹',
                    )

    groups: dict[str, tuple[DuplicateGroup, ...]] = {}
    for _directory_name, kind in RAW_DATA_DIRS:
        kind_groups = []
        for (size, digest), paths in fingerprints[kind].items():
            ordered = tuple(sorted(paths, key=lambda item: _relative_key(item, dataset_root)))
            if len(ordered) > 1:
                kind_groups.append(DuplicateGroup(kind, digest, ordered, size))
        groups[kind] = tuple(
            sorted(
                kind_groups,
                key=lambda group: _relative_key(group.keeper, dataset_root),
            )
        )

    orphans = _scan_orphans(all_paths_by_kind, dataset_root, image_stems_by_dir)
    name_conflicts = _scan_name_conflicts(all_paths_by_kind, dataset_root)

    return DuplicateScanResult(
        dataset_root, counts, groups, tuple(errors),
        orphans=orphans, name_conflicts=name_conflicts,
    )


def _scan_orphans(
    all_paths_by_kind: dict[str, list[Path]],
    dataset_root: Path,
    image_stems_by_dir: dict[str, set[str]],
) -> tuple[OrphanRecord, ...]:
    """Report annotation/label files without a same-batch image sibling."""

    records: list[OrphanRecord] = []
    for directory_name, kind in RAW_DATA_DIRS:
        if kind == 'image':
            continue
        for path in all_paths_by_kind.get(kind, ()):
            parts = path.relative_to(dataset_root).as_posix().split('/')
            dir_key = '/'.join(parts[1:-1]) if len(parts) >= 3 else ''
            if path.stem not in image_stems_by_dir.get(dir_key, ()):
                records.append(OrphanRecord(kind, path, dir_key))
    return tuple(sorted(records, key=lambda item: item.path.as_posix()))


def _scan_name_conflicts(
    all_paths_by_kind: dict[str, list[Path]],
    dataset_root: Path,
) -> tuple[NameConflictGroup, ...]:
    """Report same-stem files living in different directories with different
    content.  Byte-identical locations stay out: they belong to duplicate
    groups and are already handled by the duplicate workflows."""

    groups: list[NameConflictGroup] = []
    for directory_name, kind in RAW_DATA_DIRS:
        if kind == 'image':
            continue
        by_stem: dict[str, list[Path]] = {}
        for path in all_paths_by_kind.get(kind, ()):
            by_stem.setdefault(path.stem, []).append(path)
        for stem, paths in sorted(by_stem.items()):
            if len(paths) < 2:
                continue
            members: list[NameConflictMember] = []
            for path in sorted(
                paths, key=lambda item: _relative_key(item, dataset_root)
            ):
                try:
                    stat = path.stat()
                    digest = _cached_sha256(path, stat.st_size, stat.st_mtime_ns)
                except (OSError, PermissionError):
                    continue
                members.append(NameConflictMember(path, digest, stat.st_size))
            if len(members) < 2:
                continue
            if len({member.digest for member in members}) > 1:
                groups.append(NameConflictGroup(kind, stem, tuple(members)))
    return tuple(groups)


def resolve_raw_dataset_root(path: str | Path) -> Path:
    """Resolve a project root from its root, raw tree, or raw batch path.

    Users commonly select ``project/images/batch`` from a file dialog.  The
    audit still means the complete raw project, so normalize that path to the
    sibling-directory root before scanning.  Derived trees are rejected to
    avoid silently scanning generated train/test copies.
    """

    selected = Path(path).expanduser()
    if not str(selected).strip():
        raise ValueError('未选择数据目录')
    try:
        selected = selected.resolve()
    except OSError:
        selected = selected.absolute()
    if not selected.is_dir():
        raise ValueError(f'数据目录不存在或不是目录: {selected}')

    parts = set(selected.parts)
    if parts & _DERIVED_DATA_DIRS:
        raise ValueError(
            '原始数据重复审查不扫描 training_data 或 test_data，'
            '请返回并选择原始数据项目根目录。'
        )

    candidates = (selected, *selected.parents)
    for candidate in candidates:
        if candidate.name in {'images', 'annotations', 'labels'}:
            candidate = candidate.parent
        if _looks_like_raw_root(candidate):
            return candidate

    raise ValueError(
        f'未识别到原始数据项目结构: {selected}\n'
        '请选择包含 images、annotations 或 labels 的数据项目根目录，'
        '也可以选择其中的原始批次目录。'
    )


def _looks_like_raw_root(path: Path) -> bool:
    """Return whether path owns the separated raw-data directories."""
    if path.name in _DERIVED_DATA_DIRS:
        return False
    return any((path / directory_name).is_dir()
               for directory_name, _kind in RAW_DATA_DIRS)


def delete_duplicate_files(
    result: DuplicateScanResult,
    groups: Iterable[DuplicateGroup] | None = None,
    *,
    use_trash: bool = True,
    delete_companions: bool = False,
) -> DeleteResult:
    """Delete only duplicate members while preserving every group keeper.

    When ``delete_companions`` is enabled, deleting an image duplicate also
    removes its companion annotation/label duplicates — the file with the
    same relative path under ``annotations`` / ``labels``.  A companion is
    only deleted when the scan has confirmed it as a duplicate member of its
    own group; keepers and unconfirmed companions are reported in
    ``skipped`` and never removed.

    The result is checked against the scan result and every target must stay
    under the selected dataset root.  This prevents a stale UI selection or a
    malicious symlink from turning the action into an arbitrary file delete.
    """

    selected = tuple(groups) if groups is not None else result.all_groups
    group_by_path: dict[Path, DuplicateGroup] = {}
    keeper_paths: set[Path] = set()
    for group in result.all_groups:
        for path in group.duplicates:
            group_by_path[path.resolve()] = group
        keeper_paths.add(group.keeper.resolve())
    root = result.root.resolve()
    deleted: list[Path] = []
    errors: list[str] = []
    skipped: list[str] = []
    removed: set[Path] = set()

    def _remove_one(path: Path, group: DuplicateGroup, label: str = '') -> None:
        resolved = path.resolve()
        if resolved in removed:
            return
        try:
            # The UI may stay open while another process edits a file.
            # Re-check the fingerprint before making a destructive change.
            if (
                resolved.stat().st_size != group.size
                or _sha256(resolved) != group.digest
            ):
                skipped.append(f'{label}内容已变化，跳过删除: {resolved}')
                return
            _remove_file(resolved, use_trash=use_trash)
            removed.add(resolved)
            deleted.append(resolved)
        except (OSError, PermissionError) as exc:
            errors.append(f'删除失败 {label}{resolved}: {exc}')

    for group in selected:
        for raw_path in group.duplicates:
            path = Path(raw_path).expanduser()
            try:
                resolved = path.resolve()
                resolved.relative_to(root)
            except (OSError, ValueError):
                errors.append(f'拒绝删除（不在数据根目录内）: {path}')
                continue
            if resolved not in group_by_path:
                errors.append(f'拒绝删除（不是本次扫描确认的重复副本）: {path}')
                continue
            if not resolved.is_file() or resolved.is_symlink():
                errors.append(f'文件不存在或不是普通文件: {path}')
                continue
            _remove_one(path, group_by_path[resolved])

            if delete_companions and group.kind == 'image':
                for kind, companion in _companion_candidates(path, root):
                    if companion is None or not companion.exists():
                        continue
                    comp_resolved = companion.resolve()
                    try:
                        comp_resolved.relative_to(root)
                    except (OSError, ValueError):
                        skipped.append(f'{kind} 不在数据根目录内，未联动删除: {comp_resolved}')
                        continue
                    comp_group = group_by_path.get(comp_resolved)
                    if comp_group is None:
                        if comp_resolved in keeper_paths:
                            skipped.append(f'{kind} 是保留副本，未联动删除: {comp_resolved}')
                        else:
                            skipped.append(f'{kind} 未确认为重复副本，未联动删除: {comp_resolved}')
                        continue
                    if not comp_resolved.is_file() or comp_resolved.is_symlink():
                        skipped.append(f'{kind} 不是普通文件，未联动删除: {comp_resolved}')
                        continue
                    _remove_one(companion, comp_group, label='联动删除 ')

    return DeleteResult(tuple(deleted), tuple(errors), tuple(skipped))


def delete_orphan_files(
    paths: Iterable[Path],
    root: str | Path,
    *,
    use_trash: bool = True,
) -> DeleteResult:
    """Delete orphan annotation/label files with the same safety guards used
    by duplicate deletion: targets must stay under the dataset root, be
    regular files and not symlinks."""

    root_resolved = Path(root).resolve()
    deleted: list[Path] = []
    errors: list[str] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        try:
            resolved = path.resolve()
            resolved.relative_to(root_resolved)
        except (OSError, ValueError):
            errors.append(f'拒绝删除（不在数据根目录内）: {path}')
            continue
        if not resolved.is_file() or resolved.is_symlink():
            errors.append(f'文件不存在或不是普通文件: {path}')
            continue
        try:
            _remove_file(resolved, use_trash=use_trash)
            deleted.append(resolved)
        except (OSError, PermissionError) as exc:
            errors.append(f'删除失败 {resolved}: {exc}')
    return DeleteResult(tuple(deleted), tuple(errors))


def move_to_backup(
    paths: Iterable[Path],
    root: str | Path,
    backup_dir: str | Path,
) -> MoveResult:
    """Move files into ``backup_dir`` preserving their relative layout.

    Sources must stay under the dataset root and must be regular non-symlink
    files; an existing backup target is refused instead of overwritten.
    """

    root_resolved = Path(root).resolve()
    backup = Path(backup_dir).expanduser().resolve()
    moved: list[Path] = []
    errors: list[str] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        try:
            resolved = path.resolve()
            rel = resolved.relative_to(root_resolved)
        except (OSError, ValueError):
            errors.append(f'拒绝移出（不在数据根目录内）: {path}')
            continue
        if not resolved.is_file() or resolved.is_symlink():
            errors.append(f'文件不存在或不是普通文件: {path}')
            continue
        target = backup / rel
        try:
            if target.exists():
                errors.append(f'备份目标已存在，跳过: {target}')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(resolved), str(target))
            moved.append(resolved)
        except (OSError, PermissionError) as exc:
            errors.append(f'移出失败 {resolved}: {exc}')
    return MoveResult(tuple(moved), tuple(errors))


def _companion_candidates(image_path: Path, root: Path):
    """Return ``(kind, Path)`` companions of an image under the raw tree.

    An ``images/<batch>/<name>.jpg`` file maps to ``annotations/<batch>/<name>.json``
    and ``labels/<batch>/<name>.txt``.  Only the scanned raw directories are
    considered; task-specific JSON directories remain outside this scope.
    """

    try:
        rel = image_path.relative_to(root).as_posix()
    except ValueError:
        return ()
    parts = rel.split('/')
    if len(parts) < 2 or parts[0] != 'images':
        return ()
    sub = parts[1:]
    results = []
    for directory_name, kind in (('annotations', 'annotation'), ('labels', 'label')):
        companion = root / directory_name / Path(*sub).with_suffix(
            '.json' if kind == 'annotation' else '.txt'
        )
        results.append((kind, companion))
    return results


def _iter_raw_files(source_dir: Path, kind: str):
    allowed_extensions = (
        IMAGE_EXTENSIONS if kind == 'image'
        else {'.json'} if kind == 'annotation'
        else {'.txt'}
    )

    def on_error(error):
        # os.walk reports the error to its caller only through this callback;
        # the scanner records file-level errors where possible.
        return None

    for dirpath, dirnames, filenames in os.walk(
        str(source_dir), topdown=True, followlinks=False, onerror=on_error
    ):
        current = Path(dirpath)
        dirnames[:] = sorted(
            name for name in dirnames
            if not (current / name).is_symlink()
        )
        for filename in sorted(filenames):
            path = current / filename
            if path.is_symlink() or path.suffix.lower() not in allowed_extensions:
                continue
            if path.is_file():
                yield path.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            chunk = stream.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _quick_fingerprint(path: Path, size: int) -> str:
    """Read only the file edges as a cheap candidate filter."""
    with path.open('rb') as stream:
        head = stream.read(_QUICK_FINGERPRINT_SIZE)
        if size > _QUICK_FINGERPRINT_SIZE:
            stream.seek(max(size - _QUICK_FINGERPRINT_SIZE, 0))
            tail = stream.read(_QUICK_FINGERPRINT_SIZE)
        else:
            tail = b''
    digest = hashlib.blake2b(digest_size=16)
    digest.update(head)
    digest.update(tail)
    return digest.hexdigest()


def _cached_sha256(path: Path, size: int, mtime_ns: int) -> str:
    key = str(path)
    cached = _HASH_CACHE.get(key)
    if cached and cached[:2] == (size, mtime_ns):
        return cached[2]
    digest = _sha256(path)
    _HASH_CACHE[key] = (size, mtime_ns, digest)
    return digest


def _relative_key(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _remove_file(path: Path, *, use_trash: bool) -> None:
    if use_trash:
        try:
            import send2trash
        except ImportError:
            pass
        else:
            send2trash.send2trash(str(path))
            return
    path.unlink()
