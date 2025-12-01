#!/usr/bin/env bash
set -euo pipefail

# Simple helper script to run Qwen2-VL inference on images.jpeg with a default query.
python3 inference.py images.jpeg "Describe this image." 
