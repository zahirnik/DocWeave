#!/usr/bin/env bash
# Restart FastAPI with verbose logging and reliable shutdown.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Load env if present
if [[ -f .env ]]; then
  set -a
  . ./.env
  set +a
fi

PORT="${PORT:-8000}"

# Stop any existing uvicorn
if [[ -f uvicorn.pid ]]; then
  OLD_PID="$(cat uvicorn.pid || true)"
  if [[ -n "${OLD_PID}" ]] && ps -p "${OLD_PID}" >/dev/null 2>&1; then
    echo "Stopping previous Uvicorn PID ${OLD_PID}..."
    kill "${OLD_PID}" || true
    sleep 1
  fi
  rm -f uvicorn.pid
fi
# Extra safety: kill any stray uvicorn on the same app/port
pkill -f "uvicorn .*apps\.api\.main:app" >/dev/null 2>&1 || true
sleep 1

# Ensure deps
python3 - <<'PY'
import sys, subprocess, importlib
for pkg in ("uvicorn","requests"):
    try:
        importlib.import_module(pkg)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
PY

# Start uvicorn with debug logs
echo "Starting Uvicorn (debug) on 0.0.0.0:${PORT}..."
nohup python3 -m uvicorn apps.api.main:app \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --workers 1 \
  --log-level debug \
  --access-log \
  > uvicorn.log 2>&1 &

PID=$!
echo "${PID}" > uvicorn.pid
echo "PID ${PID} written to uvicorn.pid"

# Wait for /health
echo "Waiting for /health..."
python3 - <<'PY'
import os, time, sys, requests
base = f"http://127.0.0.1:{os.environ.get('PORT','8000')}"
for i in range(60):
    try:
        r = requests.get(base + "/health", timeout=3)
        if r.status_code == 200:
            print("Health:", r.json())
            sys.exit(0)
    except Exception:
        pass
    time.sleep(1)
print("Health check failed after 60 tries", file=sys.stderr)
sys.exit(1)
PY

echo "API is healthy on http://127.0.0.1:${PORT}"
