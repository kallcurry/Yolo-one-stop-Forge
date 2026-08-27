"""File system scanner: format detection, image listing, annotation lookup."""

from enum import Enum
from pathlib import Path

from app.utils import log

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
DEFAULT_ANNOTATION_DIR = 'annotations'


class DirFormat(Enum):
    DJI_PAIR = 1     # has images/ + annotations/ as immediate children
    SEPARATED = 2    # parent has images/ + annotations/ top-level dirs
    FLAT = 3         # just image files, no sub-structure


def _has_direct_images(dir_path: Path) -> bool:
    """Check if a directory contains image files directly (not just subdirs)."""
    if not dir_path.is_dir():
        return False
    return any(
        f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        for f in dir_path.iterdir()
    )


def detect_format(path: str | Path,
                  annotation_dir: str = DEFAULT_ANNOTATION_DIR) -> DirFormat:
    """Detect which of the three directory formats `path` belongs to."""
    p = Path(path).resolve()
    if not p.is_dir():
        return DirFormat.FLAT

    images_dir = p / 'images'
    annotations_dir = p / _annotation_dir_name(annotation_dir)

    if images_dir.is_dir():
        # DJI_PAIR: images/ contains image files directly
        if _has_direct_images(images_dir):
            return DirFormat.DJI_PAIR

        # SEPARATED: images/ and annotations/ have matching subdirectories
        img_subdirs = {d.name for d in images_dir.iterdir() if d.is_dir()}
        ann_subdirs = (
            {d.name for d in annotations_dir.iterdir() if d.is_dir()}
            if annotations_dir.is_dir()
            else set()
        )
        if img_subdirs & ann_subdirs:
            return DirFormat.SEPARATED
        if img_subdirs:
            return DirFormat.SEPARATED

        # Fallback: has both dirs but no pattern match — treat as DJI_PAIR
        return DirFormat.DJI_PAIR

    # Walk up ancestors to determine format context
    for ancestor in p.parents:
        anc_images = ancestor / 'images'
        anc_annotations = ancestor / _annotation_dir_name(annotation_dir)
        if anc_images.is_dir():
            # DJI_PAIR ancestor: images/ contains image files directly
            if _has_direct_images(anc_images):
                return DirFormat.DJI_PAIR
            # SEPARATED ancestor: matching subdirectories
            img_subdirs = {d.name for d in anc_images.iterdir() if d.is_dir()}
            ann_subdirs = (
                {d.name for d in anc_annotations.iterdir() if d.is_dir()}
                if anc_annotations.is_dir()
                else set()
            )
            if img_subdirs & ann_subdirs:
                return DirFormat.SEPARATED
            if img_subdirs:
                return DirFormat.SEPARATED

    return DirFormat.FLAT


def _get_separated_root(
    path: Path,
    annotation_dir: str = DEFAULT_ANNOTATION_DIR,
) -> Path | None:
    """Find the root that contains peer images/ and annotation dirs."""
    annotation_dir = _annotation_dir_name(annotation_dir)
    for ancestor in [path, *path.parents]:
        if (ancestor / 'images').is_dir():
            return ancestor
    return None


def list_images(path: str | Path, fmt: DirFormat | None = None) -> list[Path]:
    """Return sorted absolute paths of all image files under `path`."""
    p = Path(path).resolve()
    if fmt is None:
        fmt = detect_format(p)
    log(f'📂 list_images: {p.name} fmt={fmt.name}')

    if fmt == DirFormat.DJI_PAIR:
        # If we're already inside the images/ subdirectory, use p directly
        if p.name == 'images' and _has_direct_images(p):
            search_dir = p
        else:
            search_dir = p / 'images'
    elif fmt == DirFormat.SEPARATED:
        search_dir = p
    else:
        search_dir = p

    if not search_dir.is_dir():
        return []

    images = sorted(
        f for f in search_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    return images


def expected_annotation_path(image_path: str | Path,
                             fmt: DirFormat | None = None,
                             annotation_dir: str = DEFAULT_ANNOTATION_DIR
                             ) -> Path | None:
    """Return the expected annotation JSON path, whether it exists or not."""
    img = Path(image_path).resolve()
    annotation_dir = _annotation_dir_name(annotation_dir)
    if fmt is None:
        fmt = detect_format(img.parent, annotation_dir)

    json_name = img.stem + '.json'

    if fmt == DirFormat.DJI_PAIR:
        # DJI_PAIR: images/ and annotations/ are siblings
        ann_dir = img.parent.parent / annotation_dir
        return ann_dir / json_name
    elif fmt == DirFormat.SEPARATED:
        root = _get_separated_root(img.parent, annotation_dir)
        if root is None:
            return None
        # figure out which subdirectory the image is in
        images_root = root / 'images'
        try:
            rel = img.parent.relative_to(images_root)
        except ValueError:
            return None
        return root / annotation_dir / rel / json_name

    return None


def annotation_set_dir_for_image(image_path: str | Path,
                                 fmt: DirFormat | None = None,
                                 annotation_dir: str = DEFAULT_ANNOTATION_DIR
                                 ) -> Path | None:
    """Return the expected annotation-set directory for an image."""
    expected = expected_annotation_path(image_path, fmt, annotation_dir)
    return expected.parent if expected is not None else None


def find_annotation(image_path: str | Path,
                    fmt: DirFormat | None = None,
                    annotation_dir: str = DEFAULT_ANNOTATION_DIR) -> Path | None:
    """Return the matching annotation JSON path, or None."""
    img = Path(image_path).resolve()
    annotation_dir = _annotation_dir_name(annotation_dir)
    if fmt is None:
        fmt = detect_format(img.parent, annotation_dir)
    log(f'🔍 find_annotation: {img.name} fmt={fmt.name} ann={annotation_dir}')

    candidate = expected_annotation_path(img, fmt, annotation_dir)
    if candidate is None:
        return None

    return candidate if candidate.is_file() else None


def _annotation_dir_name(annotation_dir: str | Path) -> str:
    value = str(annotation_dir or DEFAULT_ANNOTATION_DIR).strip()
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        return DEFAULT_ANNOTATION_DIR
    if any(part in ('', '.') for part in path.parts):
        return DEFAULT_ANNOTATION_DIR
    return value


def scan_tree(root: str | Path) -> dict:
    """Build a directory tree for the left panel.

    For SEPARATED roots, the tree flattens the images/ → subdirs hierarchy
    so the user sees image-containing directories directly.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        return {'path': str(root), 'format': DirFormat.FLAT, 'children': []}

    root_fmt = detect_format(root)

    if (root / 'images').is_dir() and (
        (root / 'training_data').is_dir()
        or (root / 'test_data').is_dir()
    ):
        return _build_project_dataset_tree(root, root_fmt)

    if root_fmt == DirFormat.SEPARATED:
        # Show subdirectories inside images/ as top-level children
        images_dir = root / 'images'
        children = []
        if images_dir.is_dir():
            for sub in sorted(images_dir.iterdir()):
                if sub.is_dir():
                    sub_fmt = detect_format(sub)
                    sub_children = _build_sub_children(sub, sub_fmt)
                    children.append({
                        'path': str(sub),
                        'format': sub_fmt,
                        'children': sub_children,
                    })
        return {'path': str(root), 'format': root_fmt, 'children': children}

    elif root_fmt == DirFormat.DJI_PAIR:
        img_dir = root / 'images'
        children = []
        if img_dir.is_dir():
            img_children = _build_sub_children(img_dir, DirFormat.FLAT)
            children.append({
                'path': str(img_dir),
                'format': DirFormat.FLAT,
                'children': img_children,
            })
        return {'path': str(root), 'format': root_fmt, 'children': children}

    else:  # FLAT
        children = []
        for entry in sorted(root.iterdir()):
            if entry.is_dir() and not entry.name.startswith('.'):
                fmt = detect_format(entry)
                children.append({
                    'path': str(entry),
                    'format': fmt,
                    'children': _build_sub_children(entry, fmt),
                })
        return {'path': str(root), 'format': root_fmt, 'children': children}


def _build_sub_children(path: Path, fmt: DirFormat) -> list[dict]:
    """Build child entries for a directory node."""
    children = []
    if fmt == DirFormat.DJI_PAIR:
        img_dir = path / 'images'
        if img_dir.is_dir():
            children.append({
                'path': str(img_dir),
                'format': DirFormat.FLAT,
                'children': [],
            })
    elif fmt == DirFormat.SEPARATED:
        images_dir = path / 'images'
        if images_dir.is_dir():
            for sub in sorted(images_dir.iterdir()):
                if sub.is_dir():
                    children.append({
                        'path': str(sub),
                        'format': detect_format(sub),
                        'children': [],
                    })
    # FLAT: no children
    return children


def _build_project_dataset_tree(root: Path, root_fmt: DirFormat) -> dict:
    """Build raw, training, and test scopes for a complete project dataset."""
    children = []
    images_dir = root / 'images'
    raw_children = []
    if images_dir.is_dir():
        for sub in sorted(images_dir.iterdir()):
            if not sub.is_dir():
                continue
            sub_fmt = detect_format(sub)
            raw_children.append({
                'path': str(sub),
                'format': sub_fmt,
                'children': _build_sub_children(sub, sub_fmt),
            })
    children.append({
        'path': str(images_dir),
        'display_name': '原始数据',
        'format': None,
        'selectable': False,
        'kind': 'scope',
        'children': raw_children,
    })

    training_root = root / 'training_data'
    if training_root.is_dir():
        children.append({
            'path': str(training_root),
            'display_name': '训练数据',
            'format': None,
            'selectable': False,
            'kind': 'scope',
            'children': _dataset_scope_batches(
                training_root, '默认训练集'
            ),
        })

    test_root = root / 'test_data'
    if test_root.is_dir():
        children.append({
            'path': str(test_root),
            'display_name': '验证 / 测试数据',
            'format': None,
            'selectable': False,
            'kind': 'scope',
            'children': _dataset_scope_batches(
                test_root, '默认测试集'
            ),
        })

    return {
        'path': str(root),
        'format': root_fmt,
        'selectable': False,
        'kind': 'project',
        'children': children,
    }


def _dataset_scope_batches(scope_root: Path,
                           default_name: str) -> list[dict]:
    """Return direct and nested image/annotation batch roots."""
    batches = []
    if (scope_root / 'images').is_dir():
        fmt = detect_format(scope_root)
        batches.append({
            'path': str(scope_root),
            'display_name': default_name,
            'format': fmt,
            'children': _batch_children(scope_root, fmt),
        })

    for child in sorted(scope_root.iterdir()):
        if not child.is_dir() or not (child / 'images').is_dir():
            continue
        fmt = detect_format(child)
        batches.append({
            'path': str(child),
            'format': fmt,
            'children': _batch_children(child, fmt),
        })
    return batches


def _batch_children(batch_root: Path, fmt: DirFormat) -> list[dict]:
    """Only expose nested image splits; direct image batches remain a leaf."""
    if fmt != DirFormat.SEPARATED:
        return []
    images_dir = batch_root / 'images'
    return [
        {
            'path': str(child),
            'format': detect_format(child),
            'children': [],
        }
        for child in sorted(images_dir.iterdir())
        if child.is_dir()
    ]


def create_folder(parent: str | Path, name: str) -> str | None:
    """Create a new folder under `parent`. Returns error message or None."""
    parent = Path(parent).resolve()
    if not parent.is_dir():
        return f"'{parent}' 不是有效目录"
    target = parent / name
    if target.exists():
        return f"'{name}' 已存在"
    try:
        target.mkdir(parents=True)
    except OSError as e:
        return str(e)
    return None
