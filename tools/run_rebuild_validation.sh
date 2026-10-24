#!/usr/bin/env bash
set -euo pipefail

REBUILD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${REBUILD_PYTHON:-python3}"

cd "$REBUILD_ROOT"
"$PYTHON_BIN" tools/validation_runtime.py
exec "$PYTHON_BIN" "$@"
