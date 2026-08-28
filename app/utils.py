"""Simple logging utility — prints to stderr with timestamps."""

import sys
import time
from functools import wraps


def log(msg: str, *args):
    """Print a timestamped log message to stderr."""
    ts = time.strftime('%H:%M:%S')
    if args:
        msg = msg % args
    print(f'[{ts}] {msg}', file=sys.stderr, flush=True)


def trace(func):
    """Decorator: log function entry/exit and timing."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        name = func.__name__
        log(f'→ {name}')
        t0 = time.time()
        try:
            result = func(*args, **kwargs)
            dt = (time.time() - t0) * 1000
            log(f'← {name} ({dt:.0f}ms)')
            return result
        except Exception as e:
            log(f'✗ {name} ERROR: {e}')
            raise
    return wrapper


def discover_available_models(extra_repo=None):
    """Discover .pt weights from training runs, models/ and an optional repo.

    The training runner writes runs/<project>/<run>/weights/best.pt (two
    levels), so a plain ``*/weights`` glob misses them; this walks recursively
    and also picks up flat weights in ``models/``.
    """
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    roots = [project_root / 'training' / 'runs', project_root / 'models']
    if extra_repo:
        roots.append(Path(str(extra_repo)).expanduser())

    found = {}
    for root in roots:
        if not root.is_dir():
            continue
        for weights_dir in root.rglob('weights'):
            if not weights_dir.is_dir():
                continue
            for path in sorted(weights_dir.iterdir()):
                if path.is_file() and path.suffix == '.pt':
                    found.setdefault(str(path.resolve()), path)
        if root.name == 'models':
            for path in sorted(root.iterdir()):
                if path.is_file() and path.suffix == '.pt':
                    found.setdefault(str(path.resolve()), path)
    return [
        found[key]
        for key in sorted(found, key=lambda item: (found[item].name, item))
    ]
