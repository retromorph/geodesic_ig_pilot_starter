#!/usr/bin/env bash
# Idempotently start a loopback-only MLflow server (SQLite + proxied artifacts) in tmux session "mlflow".
# Never binds to 0.0.0.0, never touches firewall/SSH configuration.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
SESSION=mlflow
PORT="${MLFLOW_PORT:-5000}"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$PROJECT_ROOT/mlruns/artifacts" "$PROJECT_ROOT/mlruns/logs"

healthy() { curl --fail --silent --max-time 3 "http://127.0.0.1:${PORT}/health" >/dev/null; }

if tmux has-session -t "$SESSION" 2>/dev/null && healthy; then
  echo "MLFLOW_ALREADY_RUNNING http://127.0.0.1:${PORT}"
  exit 0
fi
tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"
LOG="$PROJECT_ROOT/mlruns/logs/mlflow-$(date -u +%Y%m%dT%H%M%SZ).log"
tmux new-session -d -s "$SESSION" -c "$PROJECT_ROOT" \
  "uv run --locked mlflow server \
     --backend-store-uri 'sqlite:///$PROJECT_ROOT/mlruns/mlflow.db' \
     --serve-artifacts --artifacts-destination '$PROJECT_ROOT/mlruns/artifacts' \
     --host 127.0.0.1 --port $PORT 2>&1 | tee '$LOG'"
for _ in $(seq 1 60); do
  if healthy; then
    echo "MLFLOW_OK http://127.0.0.1:${PORT} (log: $LOG)"
    if ss -ltn | grep -q ":${PORT} " && ss -ltn | grep ":${PORT} " | grep -vq "127.0.0.1:${PORT}"; then
      echo "ERROR: port ${PORT} is listening on a non-loopback address" >&2
      exit 3
    fi
    exit 0
  fi
  sleep 2
done
echo "MLFLOW_FAILED_TO_START (see $LOG)" >&2
exit 1
