#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEO_PATH="$PROJECT_DIR/testing_dataset/testing1.MP4"

if [[ ! -f "$VIDEO_PATH" ]]; then
  echo "Video tidak ditemukan: $VIDEO_PATH" >&2
  exit 1
fi

cd "$PROJECT_DIR"
CAMERAS="cam0=$VIDEO_PATH" exec python3 -m uvicorn backend.app:app \
  --host 127.0.0.1 --port 8000
