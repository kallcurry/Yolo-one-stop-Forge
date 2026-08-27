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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'
}
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
class DuplicateScanResult:
    """Immutable result returned by :func:`scan_raw_duplicates`."""

    root: Path
    scanned_counts: dict[str, int]
    groups: dict[str, tuple[DuplicateGroup, ...]]
    errors: tuple[str, ...] = ()

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


@dataclass(frozen=True)
class DeleteResult:
    """Outcome of deleting duplicate members from one or more groups."""

    deleted: tuple[Path, ...]
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
            size_buckets[kind].setdefault(stat.st_size, []).append(
                (path, stat.st_mtime_ns)
            )

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

    return DuplicateScanResult(dataset_root, counts, groups, tuple(errors))


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
) -> DeleteResult:
    """Delete only duplicate members while preserving every group keeper.

    The result is checked against the scan result and every target must stay
    under the selected dataset root.  This prevents a stale UI selection or a
    malicious symlink from turning the action into an arbitrary file delete.
    """

    selected = tuple(groups) if groups is not None else result.all_groups
    allowed = {
        path.resolve()
        for group in result.all_groups
        for path in group.duplicates
    }
    root = result.root.resolve()
    deleted: list[Path] = []
    errors: list[str] = []

    for group in selected:
        for raw_path in group.duplicates:
            path = Path(raw_path).expanduser()
            try:
                resolved = path.resolve()
                resolved.relative_to(root)
            except (OSError, ValueError):
                errors.append(f'拒绝删除（不在数据根目录内）: {path}')
                continue
            if resolved not in allowed:
                errors.append(f'拒绝删除（不是本次扫描确认的重复副本）: {path}')
                continue
            if not resolved.is_file() or resolved.is_symlink():
                errors.append(f'文件不存在或不是普通文件: {path}')
                continue

            try:
                # The UI may stay open while another process edits a file.
                # Re-check the fingerprint before making a destructive change.
                if (
                    resolved.stat().st_size != group.size
                    or _sha256(resolved) != group.digest
                ):
                    errors.append(f'文件内容已变化，跳过删除: {resolved}')
                    continue
                _remove_file(resolved, use_trash=use_trash)
                deleted.append(resolved)
            except (OSError, PermissionError) as exc:
                errors.append(f'删除失败 {resolved}: {exc}')

    return DeleteResult(tuple(deleted), tuple(errors))


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
