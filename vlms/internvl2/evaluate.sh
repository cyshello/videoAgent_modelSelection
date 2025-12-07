#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/internvl2/bin/activate"
python evaluate_image.py /home/intern/youngseo/modelSelection/imageqa/ChartQA --output /home/intern/youngseo/modelSelection/videoAgent_modelSelection/vlms/internvl2/evaluation_results.json --question-key query