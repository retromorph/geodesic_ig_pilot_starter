#!/usr/bin/env bash
# Sequential E1 pipeline up to and including validation selection + freeze (stops on first failure).
# The blinded test evaluation is launched separately (scripts/run_stage.sh gig-e1-test evaluate-e1 --split test --resume).
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "$PROJECT_ROOT"
export PATH="$HOME/.local/bin:$PATH"
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
run() { echo "=== $(date -u +%FT%TZ) $*"; uv run --locked python -m geodesic_ig_pilot.cli "$@" --config configs/protocol.v1.yaml; }
run prepare-e1 --resume --device cpu
run train-e1 --resume
run fit-encoder --resume
run evaluate-e1 --split validation --resume
echo "=== $(date -u +%FT%TZ) E1 validation + freeze complete"
