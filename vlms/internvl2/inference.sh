#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/internvl2/bin/activate"
python inference.py /home/intern/youngseo/modelSelection/NExTQA/NExTQA/NExTVideo/2400084970.mp4 "Describe this image." 0
