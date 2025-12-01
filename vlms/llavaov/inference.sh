#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/llavaov/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
	echo "Expected virtual environment python at $VENV_PYTHON not found" >&2
	exit 1
fi

"$VENV_PYTHON" "$SCRIPT_DIR/inference.py" "$SCRIPT_DIR/images.jpeg" "Describe this image in one detailed caption."