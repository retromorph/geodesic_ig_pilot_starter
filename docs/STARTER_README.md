# Geodesic Integrated Gradients — Go/No-Go Pilot

**Status:** research specification and coding-agent handoff, not an implemented or executed experiment.

**Question:** Does a semantic-representation pullback geodesic improve *contrastive attribution fidelity* over Euclidean/pixel, latent-linear, and image-pullback paths **when model, input, baseline and score are held fixed**?

The study has three stages:

| Stage | Purpose | Execution |
|---|---|---|
| E0 | Analytic path/geodesic checks; geometry-induced misallocation; additive negative control | Required, CPU |
| E1 | Learned CNN on a small differentiable visual renderer; exact semantic-coalition evaluation; independent learned encoder | Required, primary |
| E2 | Part-based external evaluation, preferably FunnyBirds, using matched endpoints and a validated differentiable decoder | Implement all; full run conditional on E1 and preflight |

A **GO** requires all preregistered gates. Unconvincing or incomplete results must be reported as `NO_GO`, `AMBER` or `BLOCKED`; there is no obligation to find a positive result.

## Start here

1. For the agent, read [`AGENTS.md`](AGENTS.md) and paste [`docs/AGENT_PROMPT.md`](docs/AGENT_PROMPT.md) into the coding agent.
2. For scientific questions, read [`docs/RESEARCH_SPEC.md`](docs/RESEARCH_SPEC.md) and [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md).
3. For implementation, read [`docs/ENGINEERING_PLAN.md`](docs/ENGINEERING_PLAN.md).
4. For SSH/`tmux`/`uv`/MLflow, read [`docs/REMOTE_RUNBOOK.md`](docs/REMOTE_RUNBOOK.md).
5. For the optional/conditional real-image component, read [`docs/EXTERNAL_VALIDATION.md`](docs/EXTERNAL_VALIDATION.md).
6. For final outputs, use [`docs/REPORT_TEMPLATE.md`](docs/REPORT_TEMPLATE.md).
7. For method implementations and documentation, see [`docs/REFERENCES.md`](docs/REFERENCES.md).

Predeclared parameters and decision thresholds live in [`configs/protocol.v1.yaml`](configs/protocol.v1.yaml). Do **not** rewrite this file in place after seeing held-out results.

## Expected code layout (to be created by the agent)

```text
src/geodesic_ig_pilot/
  cli.py                  # doctor, exp0, prepare-e1, train-e1, etc.
  config.py               # validated immutable protocol
  tracking.py             # MLflow and provenance
  geometry.py             # metric, energy, path optimizer
  integration.py          # signed path attributions
  renderer.py             # 64x64 differentiable factor renderer
  scores.py               # analytic scores and discriminative margins
  datasets.py             # frozen split and paired-hybrid generation
  models.py               # small CNN and independent unlabeled encoder
  evaluations.py          # coalitions, errors, Shapley references
  statistics.py           # scene-grouped paired bootstrap
  external.py             # FunnyBirds adapter with validity checks
tests/
scripts/
reports/
mlruns/                    # ignored; lives on remote machine
```

## Commands to provide (expected interface, not yet implemented)

```bash
uv sync --locked
uv run python -m geodesic_ig_pilot.cli doctor --config configs/protocol.v1.yaml
uv run python -m geodesic_ig_pilot.cli smoke-mlflow --config configs/protocol.v1.yaml
uv run pytest -q
uv run python -m geodesic_ig_pilot.cli exp0 --config configs/protocol.v1.yaml
uv run python -m geodesic_ig_pilot.cli prepare-e1 --config configs/protocol.v1.yaml
uv run python -m geodesic_ig_pilot.cli train-e1 --config configs/protocol.v1.yaml
uv run python -m geodesic_ig_pilot.cli fit-encoder --config configs/protocol.v1.yaml
uv run python -m geodesic_ig_pilot.cli evaluate-e1 --config configs/protocol.v1.yaml
uv run python -m geodesic_ig_pilot.cli prepare-e2 --config configs/protocol.v1.yaml
uv run python -m geodesic_ig_pilot.cli evaluate-e2 --config configs/protocol.v1.yaml
uv run python -m geodesic_ig_pilot.cli aggregate --config configs/protocol.v1.yaml
uv run python -m geodesic_ig_pilot.cli report --config configs/protocol.v1.yaml
```

All commands must fail clearly if their prerequisites have not been met. Stage E2 is conditionally executed according to the protocol; do not claim it ran when it was skipped.

## Remote operations at a glance

- Remote Linux host with NVIDIA RTX A4000 (16 GB), SSH and tmux.
- `uv` project and committed lockfile; select CUDA-compatible PyTorch **after** checking `nvidia-smi`.
- MLflow SQL backend: SQLite file under project `mlruns/`; MLflow proxied artifacts under `mlruns/artifacts/`.
- Bind MLflow to `127.0.0.1:5000`, not public interfaces. On your laptop use `ssh -N -L 5000:127.0.0.1:5000 <ssh-alias>`, then visit `http://127.0.0.1:5000`.
- Do not enable `ufw` or open port 5000 for this workflow. See the full [runbook](docs/REMOTE_RUNBOOK.md).

## Deliverable expected from the coding agent

`reports/FINAL_REPORT.md`, `reports/decision.json`, sample-level numeric data, plots, test results, exact MLflow experiment/run IDs, `uv.lock`, and documented deviations or execution blockers.

**No experiments have been run in this starter bundle.** The coding agent must produce and verify actual measurements rather than fill report placeholders.
