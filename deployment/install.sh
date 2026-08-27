#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCH_PROFILE="auto"
WITH_LABEL_TOOL=1

usage() {
  cat <<'EOF'
Usage: bash deployment/install.sh [options]

Options:
  --torch auto     Keep an existing PyTorch install, otherwise install from PyPI.
  --torch cpu      Install the CPU PyTorch wheels.
  --torch cu128    Install the CUDA 12.8 PyTorch wheels.
  --without-label-tool
                   Install the platform without X-AnyLabeling.
  -h, --help       Show this help.

The script always installs into the currently active Python environment. The
Conda or virtualenv name can be anything.
EOF
}

while (($#)); do
  case "$1" in
    --torch)
      [[ $# -ge 2 ]] || { echo "--torch requires a value" >&2; exit 2; }
      TORCH_PROFILE="$2"
      shift 2
      ;;
    --without-label-tool)
      WITH_LABEL_TOOL=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

"$PYTHON_BIN" - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] < (3, 12)):
    raise SystemExit(
        "Python 3.10 or 3.11 is required; Python 3.10 is the tested profile."
    )
print("Using Python:", sys.executable)
print("Python version:", sys.version.split()[0])
PY

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

case "$TORCH_PROFILE" in
  auto)
    if ! "$PYTHON_BIN" -c 'import torch, torchvision' >/dev/null 2>&1; then
      "$PYTHON_BIN" -m pip install torch torchvision
    fi
    ;;
  cpu)
    "$PYTHON_BIN" -m pip install torch torchvision \
      --index-url https://download.pytorch.org/whl/cpu
    ;;
  cu128)
    "$PYTHON_BIN" -m pip install torch torchvision \
      --index-url https://download.pytorch.org/whl/cu128
    ;;
  *)
    echo "Unsupported PyTorch profile: $TORCH_PROFILE" >&2
    exit 2
    ;;
esac

requirements=("-r" "$SCRIPT_DIR/requirements-core.txt")
doctor_args=()
if [[ "$WITH_LABEL_TOOL" == "1" ]]; then
  requirements+=("-r" "$SCRIPT_DIR/requirements-labeling.txt")
  doctor_args+=("--require-label-tool")
fi

"$PYTHON_BIN" -m pip install "${requirements[@]}"

mkdir -p \
  "$PROJECT_ROOT/.runtime" \
  "$PROJECT_ROOT/models" \
  "$PROJECT_ROOT/training/runs" \
  "$PROJECT_ROOT/training/tasks"

"$PYTHON_BIN" "$SCRIPT_DIR/doctor.py" "${doctor_args[@]}"

echo
echo "Installation completed. Start with: bash deployment/run.sh"
