#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEO_PATH="${1:-$PROJECT_DIR/testing_dataset/testing1.MP4}"
if [[ "$VIDEO_PATH" != /* ]]; then
  VIDEO_PATH="$PROJECT_DIR/$VIDEO_PATH"
fi
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"

if [[ ! -f "$VIDEO_PATH" ]]; then
  echo "Video tidak ditemukan: $VIDEO_PATH" >&2
  echo "Contoh: ./run_dashboard.sh testing_dataset/video_lain.mp4" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi
if [[ ! -d "$PROJECT_DIR/frontend/node_modules" ]]; then
  echo "Dependensi frontend belum dipasang. Jalankan: cd frontend && npm install" >&2
  exit 1
fi

cd "$PROJECT_DIR"
CAMERAS="cam0=$VIDEO_PATH" "$PYTHON_BIN" -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!
(cd "$PROJECT_DIR/frontend" && npm run dev -- --host 127.0.0.1) &
FRONTEND_PID=$!

cleanup() {
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "PigWatch React: http://127.0.0.1:5173"
echo "FastAPI docs:   http://127.0.0.1:8000/docs"
wait "$FRONTEND_PID"
