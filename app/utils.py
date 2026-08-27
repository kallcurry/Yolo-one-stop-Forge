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
