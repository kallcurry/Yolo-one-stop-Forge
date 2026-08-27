"""File operations: move, copy, delete, rename images and annotations."""

import os
import shutil
from pathlib import Path

from app.models.file_system import (
    expected_annotation_path,
    find_annotation,
    IMAGE_EXTENSIONS,
)


def _sync_op(src: Path, dst: Path, do_sync: bool, op: str,
             annotation_dir: str = 'annotations') -> list[str]:
    """Helper: perform `op` (move/copy) on src, optionally syncing annotation."""
    errors = []
    try:
        if op == 'move':
            shutil.move(str(src), str(dst))
        elif op == 'copy':
            if src.is_dir():
                shutil.copytree(str(src), str(dst))
            else:
                shutil.copy2(str(src), str(dst))
    except (OSError, shutil.Error) as e:
        errors.append(f"操作 {src.name} 失败: {e}")
        return errors

    if do_sync:
        ann = find_annotation(src, annotation_dir=annotation_dir)
        if ann and ann.exists():
            ann_dst = (
                expected_annotation_path(dst, annotation_dir=annotation_dir)
                or dst.parent / (dst.stem + '.json')
            )
            try:
                ann_dst.parent.mkdir(parents=True, exist_ok=True)
                if op == 'move':
                    shutil.move(str(ann), str(ann_dst))
                else:
                    shutil.copy2(str(ann), str(ann_dst))
            except (OSError, shutil.Error) as e:
                errors.append(f"标注同步失败 {ann.name}: {e}")
    return errors


def move_images(sources: list[Path | str],
                dest_dir: Path | str,
                sync_annotation: bool = True,
                annotation_dir: str = 'annotations') -> list[str]:
    """Move image files to dest_dir. Optionally move paired annotations."""
    dest = Path(dest_dir).resolve()
    errors = []
    if not dest.is_dir():
        return [f"目标目录 '{dest}' 不存在"]

    for src in sources:
        src = Path(src)
        if not src.is_file():
            errors.append(f"'{src}' 不是文件")
            continue
        dst = dest / src.name
        if dst.exists():
            errors.append(f"'{src.name}' 在目标目录已存在，跳过")
            continue
        errors.extend(_sync_op(src, dst, sync_annotation, 'move', annotation_dir))
    return errors


def copy_images(sources: list[Path | str],
                dest_dir: Path | str,
                sync_annotation: bool = True,
                annotation_dir: str = 'annotations') -> list[str]:
    """Copy image files to dest_dir. Optionally copy paired annotations."""
    dest = Path(dest_dir).resolve()
    errors = []
    if not dest.is_dir():
        return [f"目标目录 '{dest}' 不存在"]

    for src in sources:
        src = Path(src)
        if not src.is_file():
            errors.append(f"'{src}' 不是文件")
            continue
        dst = dest / src.name
        if dst.exists():
            errors.append(f"'{src.name}' 在目标目录已存在，跳过")
            continue
        errors.extend(_sync_op(src, dst, sync_annotation, 'copy', annotation_dir))
    return errors


def delete_images(paths: list[Path | str],
                  delete_annotation: bool = True,
                  annotation_dir: str = 'annotations') -> list[str]:
    """Delete image files. Tries send2trash first, falls back to os.remove."""
    errors = []
    try:
        import send2trash
        _trash = lambda p: send2trash.send2trash(str(p))
    except ImportError:
        _trash = lambda p: os.remove(str(p))

    for p in paths:
        p = Path(p)
        if not p.is_file():
            errors.append(f"'{p}' 不是文件")
            continue
        try:
            _trash(p)
        except OSError as e:
            errors.append(f"删除 {p.name} 失败: {e}")
            continue

        if delete_annotation:
            ann = find_annotation(p, annotation_dir=annotation_dir)
            if ann and ann.is_file():
                try:
                    _trash(ann)
                except OSError:
                    pass  # annotation deletion is best-effort
    return errors


def rename_image(old_path: Path | str,
                 new_name: str,
                 rename_annotation: bool = True,
                 annotation_dir: str = 'annotations') -> str | None:
    """Rename an image file on disk. Returns error string or None."""
    old = Path(old_path)
    if not old.is_file():
        return f"'{old}' 不是文件"

    # Ensure new_name keeps the same extension
    new = old.parent / new_name
    if new.exists():
        return f"'{new_name}' 已存在"

    try:
        os.rename(str(old), str(new))
    except OSError as e:
        return str(e)

    if rename_annotation:
        ann = find_annotation(old, annotation_dir=annotation_dir)
        if ann and ann.is_file():
            ann_new = (
                expected_annotation_path(new, annotation_dir=annotation_dir)
                or new.parent / (new.stem + '.json')
            )
            try:
                ann_new.parent.mkdir(parents=True, exist_ok=True)
                os.rename(str(ann), str(ann_new))
            except OSError:
                pass  # best-effort
    return None


def batch_rename(paths: list[Path | str],
                 pattern: str = '{n:04d}',
                 start_num: int = 1) -> list[str]:
    """Batch rename files with a numbering pattern. Returns error list.

    pattern example: 'DSC_{n:04d}' produces DSC_0001, DSC_0002, ...
    Preserves original file extensions.
    """
    errors = []
    for i, p in enumerate(paths):
        p = Path(p)
        if not p.is_file():
            errors.append(f"'{p}' 不是文件")
            continue
        ext = p.suffix
        num = start_num + i
        new_name = pattern.format(n=num) + ext
        new_path = p.parent / new_name
        if new_path.exists() and new_path != p:
            errors.append(f"'{new_name}' 已存在，跳过")
            continue
        try:
            os.rename(str(p), str(new_path))
        except OSError as e:
            errors.append(f"重命名 {p.name} 失败: {e}")
    return errors
