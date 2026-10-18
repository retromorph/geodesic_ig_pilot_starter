# Remote runbook — SSH, tmux, uv, MLflow, A4000 16 GB

This guide assumes a remote Linux host reachable via an existing, user-authorized SSH key or `~/.ssh/config` alias. It intentionally does **not** assume hostname, username, SSH keys, sudo permissions, or the NVIDIA driver version. Commands marked **local** run on your laptop/workstation; commands marked **remote** run inside SSH.

## 0. Security choice: keep MLflow private

The supplied example used `--host 0.0.0.0`, permissive wildcard allowed-hosts/CORS, `ufw allow 5000` and `ufw enable`. **Do not use that configuration for this pilot.** With an unauthenticated local MLflow instance it can expose experiments and artifacts to the network, and enabling a firewall remotely can lock out SSH.

Instead:

- Bind MLflow to `127.0.0.1:5000` on the **remote** host.
- Access the web UI from the **local** workstation via SSH port forwarding.
- No inbound port 5000 firewall change is necessary; do not modify SSH/firewall settings.
- If the user later requires team-wide web access, that is a separate deployment requiring authentication, TLS, and host/CORS allowlists.

MLflow's modern built-in DNS rebinding and CORS checks are **not a substitute for authentication**. Keep default localhost security middleware active.

## 1. Discover connection (local)

```bash
# Substitute only an existing authorized alias or user@host.
export SSH_TARGET='YOUR_SSH_ALIAS'
ssh -o BatchMode=yes -o ConnectTimeout=10 "$SSH_TARGET" 'echo SSH_OK; hostname; id -un'
```

If this fails, do not invent keys/credentials or modify firewall configuration. Report a `BLOCKED_SSH` setup issue. The agent may implement code locally, but may not assert remote experiments executed.

## 2. Inspect machine before installing anything (remote)

```bash
uname -a
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
command -v uv || true
uv --version || true
command -v tmux || true
python3 --version
free -h
df -h .
```

Verify that the GPU is indeed **RTX A4000, 16 GB VRAM**, CUDA is usable, disk has room for the project, dataset and a few GB of checkpoints/artifacts, and no other process occupies all VRAM. `nvidia-smi` reports the highest CUDA runtime supported by the driver, **not necessarily the installed PyTorch wheel CUDA version**. Never use the GPU stage if `torch.cuda.is_available()` is false.

Do not install a system-wide CUDA toolkit just because `nvcc` is absent; compatible PyTorch wheels commonly bundle the runtime needed for normal PyTorch workloads. Do not make unsolicited system package or driver changes.

## 3. Project and uv (remote)

Use a directory under your own home account (example):

```bash
mkdir -p "$HOME/work/geodesic-ig-pilot"
cd "$HOME/work/geodesic-ig-pilot"
# Place repository files here by approved git clone or rsync.
uv python install 3.11  # only if missing; verify available uv version
```

The agent must create `pyproject.toml` targeting Python 3.11 and select a PyTorch wheel/index **after checking the driver**. An example of a `uv` CUDA source section (only when the installed driver supports cu126) is:

```toml
[tool.uv.sources]
torch = { index = "pytorch-cu126" }
torchvision = { index = "pytorch-cu126" }

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true
```

If driver compatibility requires an older index (e.g. cu118), change the index **before locking**, record the reason, and pin compatible torch/torchvision versions. Follow https://docs.astral.sh/uv/guides/integration/pytorch/ . The newer `--torch-backend=auto` option is currently documented for **`uv pip`**, not a replacement for reproducible `uv lock` / `uv sync` project configuration.

Then:

```bash
uv lock
uv sync --locked
uv run python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")'
uv run python -c 'import mlflow; print(mlflow.__version__)'
```

Commit `uv.lock`; future runs should use `uv sync --locked`. Store dependency versions and package hash in MLflow.

## 4. Start remote MLflow (remote, tmux)

`mlruns/` is a *directory* inside the repository root, ignored by git. The SQLite DB and artifacts should be stored under that directory. Use **absolute paths** to avoid dependence on the current working directory.

```bash
cd "$HOME/work/geodesic-ig-pilot"
export PROJECT_ROOT="$(pwd -P)"
mkdir -p "$PROJECT_ROOT/mlruns/artifacts" "$PROJECT_ROOT/mlruns/logs"

tmux new -s mlflow
```

Inside the newly attached tmux session, run:

```bash
cd "$HOME/work/geodesic-ig-pilot"
export PROJECT_ROOT="$(pwd -P)"
uv run mlflow server \
  --backend-store-uri "sqlite:///$PROJECT_ROOT/mlruns/mlflow.db" \
  --serve-artifacts \
  --artifacts-destination "$PROJECT_ROOT/mlruns/artifacts" \
  --host 127.0.0.1 \
  --port 5000
```

Detach with `Ctrl-b`, then `d`. To reattach later: `tmux attach -t mlflow`. To verify:

```bash
tmux has-session -t mlflow
curl --fail --silent http://127.0.0.1:5000/ > /dev/null && echo 'MLFLOW_OK'
ss -ltn | grep ':5000'   # verify listening only on 127.0.0.1
```

The agent should implement a reusable, checked `scripts/start_mlflow.sh` that starts an idempotent named tmux session and captures service logs. **If an experiment already exists with file-based artifacts from an earlier server configuration**, create a new experiment name for proxying rather than assuming its stored artifact URI changes automatically.

**Why not `--default-artifact-root file:./mlruns/artifacts`?** For this deployment, the server should **proxy** artifact upload/download. MLflow's documented recommendation is to use `--artifacts-destination` with artifact serving enabled and omit `--default-artifact-root`, so new experiments receive proxy artifact URIs. This also makes browser access through an SSH tunnel work without the browser knowing remote file paths.

## 5. MLflow access (remote client and local browser)

On the remote host, the Python code should use:

```bash
export MLFLOW_TRACKING_URI='http://127.0.0.1:5000'
uv run python -m geodesic_ig_pilot.cli smoke-mlflow --config configs/protocol.v1.yaml
```

Smoke must create a run, log metrics/params, upload an example file artifact, list it and download it back; record run ID and check bytes/checksum. Simple HTTP `curl` does not prove artifacts are configured correctly.

On your **local** workstation, keep an SSH tunnel running:

```bash
ssh -N -L 5000:127.0.0.1:5000 "$SSH_TARGET"
```

Visit `http://127.0.0.1:5000`. If laptop port 5000 is occupied, choose `ssh -N -L 15000:127.0.0.1:5000 "$SSH_TARGET"` and open `http://127.0.0.1:15000`; remote scripts still use port 5000.

If the UI complains about Host/CORS policy, first check whether the tunnel and local URL use `localhost`/`127.0.0.1` as intended. Do not solve it by setting `--allowed-hosts '*'` or disabling middleware.

## 6. Launch long GPU stages (remote, tmux)

Do not tie a long job's lifetime to a transient SSH connection. Start a named session, for example:

```bash
cd "$HOME/work/geodesic-ig-pilot"
mkdir -p logs
# Example: attach a session, then enter command within it.
tmux new -s gig-e1
```

Inside that tmux session:

```bash
cd "$HOME/work/geodesic-ig-pilot"
export MLFLOW_TRACKING_URI='http://127.0.0.1:5000'
set -o pipefail
uv run python -m geodesic_ig_pilot.cli evaluate-e1 \
  --config configs/protocol.v1.yaml --resume 2>&1 | tee "logs/e1-$(date -u +%Y%m%dT%H%M%SZ).log"
echo "EXIT_CODE=${PIPESTATUS[0]}"
```

Detach (`Ctrl-b`, `d`). Reattach: `tmux attach -t gig-e1`. Note that `PIPESTATUS` must be inspected immediately after the pipeline in bash; the agent's own `scripts/run_stage.sh` should propagate failures and write an atomic stage-status JSON file.

Inspect:

```bash
tmux ls
nvidia-smi
find reports -maxdepth 2 -type f | head
```

Do not start multiple GPU-intensive stages concurrently on a single 16 GB device. Never delete someone else's tmux session or kill unknown GPU processes.

## 7. Checkpoint and resume rules

- Save model checkpoints at defined epochs and path results every 10–20 pairs, with protocol/model/manifest hashes.
- `--resume` verifies those hashes before skipping completed units. Never mix epochs/models under the same run ID.
- Failures/NaNs/OOMs are logged with sample and stage IDs; do not use `--resume` to remove failed samples from the denominator.
- Use clean run names (`e1/seed11/m4/strong`, etc.) and MLflow tags; a fresh run after a fix should include `supersedes_run_id` rather than overwriting results.
- SQLite is fine for this small **single-server** pilot. Do not run multiple incompatible tracking server processes against the same DB.

## 8. Files and secrets

Ignore `.env`, `.venv/`, `mlruns/`, `logs/`, downloads, binary checkpoints, private keys, data caches and large reports except selected text/plots/metrics. The final report should contain run IDs and hashes, **not SSH usernames, addresses, authorization tokens or confidential paths**.

Use `rsync` or an approved git remote for transferring files. Avoid embedding credentials in shell history, Python scripts or MLflow tags. Respect model/dataset licenses; document source URLs, revisions/checksums and provenance. Do not change firewall/UFW or SSH daemon configuration as part of a machine learning experiment.

## 9. Resource and safety checkpoints

1. `doctor` prints: GPU name, VRAM, PyTorch CUDA availability, driver and wheel versions, disk space, MLflow URI, protocol hash, expected number of scenes, estimated GPU-hour remaining.
2. Profile 10 E1 validation pairs before full test. Keep **peak allocated memory below ~14.5 GB** if possible, leaving headroom for allocator/caches. A4000 16 GB is a hard cap.
3. Log actual wall/GPU hours. Stop before exceeding **40 GPU-hours**; spend at most 28 on E1, 10 on E2 and 2 on smoke/profiling.
4. If an external model needs >16 GB VRAM, prefer layerwise execution, CPU offload and microbatching. A new model that cannot fit under the budget is an explicit `RESOURCE_INFEASIBLE` issue, not permission to omit numerical fidelity requirements.
5. Do not mark E2 valid unless decoder endpoints and independent part fidelity have been audited.

## 10. Troubleshooting

| Symptom | Action |
|---|---|
| `torch.cuda.is_available()` is false | Verify driver and matching PyTorch CUDA wheels; do not run GPU stage on CPU silently |
| CUDA out of memory | Decrease batch size, clear leaked references, freeze models, reduce activation retention; don't relax primary accuracy/completeness gates |
| MLflow connection refused | Check `tmux ls`, `curl`, port 5000 binding, tracking URI |
| MLflow upload fails | Check artifact proxy configuration, directory permissions and newly created experiment root URI; run roundtrip smoke |
| Browser connection refused | Reestablish local SSH `-L` tunnel; avoid opening remote firewall |
| SQL database locked | Ensure only the intended tracking server owns DB; reduce simultaneous clients and log retries |
| SSH blocked | Stop remote work and report blocker; never risk firewall changes |
| Geodesic diverges | Inspect path energy/constraints, reduce optimizer step size on validation, do not cherry-pick successful test paths |
| External decoder distorts parts | Fail E2 preflight; no conclusions about real-image geodesic fidelity |

## 11. References for these instructions

- uv and PyTorch: https://docs.astral.sh/uv/guides/integration/pytorch/
- MLflow server: https://mlflow.org/docs/latest/tracking/server/
- MLflow proxied artifact configuration: https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/
- MLflow security: https://mlflow.org/docs/latest/ml/tracking/server/security/configuration/
