#!/usr/bin/env bash
# Run one CLI stage inside a named tmux session with a timestamped log and an atomic exit-code file.
# Usage: scripts/run_stage.sh <session-name> <cli-subcommand> [extra args...]
# Survives SSH disconnects; inspect logs/<session>.exit and logs/<session>-*.log.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
SESSION="$1"; shift
mkdir -p "$PROJECT_ROOT/logs"
LOG="$PROJECT_ROOT/logs/${SESSION}-$(date -u +%Y%m%dT%H%M%SZ).log"
EXIT_FILE="$PROJECT_ROOT/logs/${SESSION}.exit"
rm -f "$EXIT_FILE"
CMD="export PATH=\$HOME/.local/bin:\$PATH; export MLFLOW_TRACKING_URI=http://127.0.0.1:5000; \
cd '$PROJECT_ROOT'; set -o pipefail; \
uv run --locked python -m geodesic_ig_pilot.cli $* --config configs/protocol.v1.yaml 2>&1 | tee '$LOG'; \
rc=\${PIPESTATUS[0]}; echo \"EXIT_CODE=\$rc\" | tee -a '$LOG'; echo \$rc > '$EXIT_FILE.tmp' && mv '$EXIT_FILE.tmp' '$EXIT_FILE'"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session $SESSION already exists; refusing to start a duplicate" >&2
  exit 2
fi
tmux new-session -d -s "$SESSION" "bash -c \"$CMD\""
echo "started tmux session $SESSION; log $LOG; exit code -> $EXIT_FILE"
