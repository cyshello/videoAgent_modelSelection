#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_PATH="${1:-images.jpeg}"
if [[ $# -ge 2 ]]; then
  QUERY="$2"
else
  QUERY="Describe the image in detail."
fi

if [[ ! -f "$IMAGE_PATH" ]]; then
  echo "Image not found: $IMAGE_PATH" >&2
  exit 1
fi

if [[ ! -d "$SCRIPT_DIR/internvl2" ]]; then
  echo "Virtual environment not found at $SCRIPT_DIR/internvl2" >&2
  exit 1
fi

source "$SCRIPT_DIR/internvl2/bin/activate"
python inference.py "$IMAGE_PATH" "$QUERY"
