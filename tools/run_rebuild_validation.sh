#!/usr/bin/env bash
set -euo pipefail

REBUILD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${REBUILD_PYTHON:-/home/dingyn/.conda/envs/rebuild-cuda128/bin/python}"
CUDA_ROOT="${REBUILD_CUDA_HOME:-/home/dingyn/.conda/envs/rebuild-cuda128}"

export CUDA_HOME="$CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:/usr/bin:/bin"
export PYTHONPATH="$REBUILD_ROOT:$REBUILD_ROOT/solver/Regularized-DROT/pytorch${PYTHONPATH:+:$PYTHONPATH}"

cd "$REBUILD_ROOT"
"$PYTHON_BIN" tools/validation_runtime.py
exec "$PYTHON_BIN" "$@"
