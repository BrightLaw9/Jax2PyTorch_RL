#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
    echo 'This setup requires native Apple Silicon Python on macOS.' >&2
    exit 1
fi
MINIPORT_MAC_VENV="${MINIPORT_MAC_VENV:-.venv-mac}"
"${PYTHON_BIN:-python3}" -m venv "${MINIPORT_MAC_VENV}"
"${MINIPORT_MAC_VENV}/bin/python" -m pip install -r requirements-mac.txt
"${MINIPORT_MAC_VENV}/bin/python" -m pip install --no-deps -e .
"${MINIPORT_MAC_VENV}/bin/python" -c 'import mlx.core as mx; assert mx.metal.is_available(), "Apple GPU unavailable"; print("MLX Apple GPU ready")'
echo 'Build the CPU verifier image with Docker on this Mac before evaluation.'
