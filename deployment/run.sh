#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -f "$SCRIPT_DIR/local.env" ]]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/local.env"
fi

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_ROOT/.runtime/matplotlib}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$PROJECT_ROOT/.runtime/ultralytics}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_ROOT/.runtime/cache}"

mkdir -p "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR" "$XDG_CACHE_HOME"
cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" main.py "$@"
