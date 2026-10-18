# AGENTS.md — Geodesic Integrated Gradients Go/No-Go Pilot

You are a coding/research agent implementing and **running** a preregistered research feasibility study. This file is an instruction contract, not a suggestion. Read `README.md`, `docs/PREREGISTRATION.md`, `docs/RESEARCH_SPEC.md`, `docs/ENGINEERING_PLAN.md`, and `docs/REMOTE_RUNBOOK.md` before writing experiment code.

## Mission

Determine whether a **learned semantic-representation geodesic** materially improves **contrastive path-integrated feature attribution** over strong **same-endpoint** baselines. Implement and, subject to the stop/budget rules, execute:

- E0: analytic geodesic sanity/negative controls (CPU).
- E1: controlled differentiable-renderer experiment with exact factor interventions, two classifier seeds, fixed validation/test splits, learned **unlabeled** representation, and matched endpoint comparisons (primary).
- E2: independent part-based visual benchmark (FunnyBirds preferred), with pretrained classifier, an explicitly validated differentiable decoder/path space, and matched baselines (external).

**Do not claim GO merely because a method creates smooth or attractive heatmaps.** Use the preregistered quantitative rules. A high-quality NO-GO report is a successful deliverable.

## Document authority and immutable protocol

1. `configs/protocol.v1.yaml` and `docs/PREREGISTRATION.md` define the fixed experimental questions, metrics, splits, thresholds, and budget.
2. `docs/RESEARCH_SPEC.md` defines formulas and scientific validity constraints.
3. `docs/ENGINEERING_PLAN.md` defines implementation contracts and testing requirements.
4. `docs/REMOTE_RUNBOOK.md` defines operational/security constraints.
5. `docs/REPORT_TEMPLATE.md` defines required final reporting.

If these documents conflict, **preserve the stricter scientific constraint**, record the conflict in `reports/deviations.md`, and do not quietly change the criteria. Record a SHA-256 hash of the protocol YAML in every run. Before unblinding the final test results, freeze model selection, hyperparameters, stopping choices, and the selected comparator. Protocol modifications after test-data exposure require a versioned amendment and all affected findings must be labeled *exploratory*; the original gate remains the primary decision.

## Mandatory scientific invariants

- All path methods compared for attribution quality **must use identical** `(x, baseline, frozen model score)` per sample. The path alone varies. Do not compare different baseline distributions as if that tested path geometry.
- The target for E1 is the **learned classifier's** two-class **logit margin**. The analytic nonlinear generating score is an oracle diagnostic, reported separately. No oracle factor information may enter the learned semantic encoder or its metric.
- For a path `x(t)=D(z(t))`, attribute in latent factors as `A_j = integral [partial_j (f o D)(z(t)) * z'_j(t)] dt`. This equals summed pixel-group attribution only because E1's renderer must have nonoverlapping, factor-specific pixel supports. Test this equality.
- For an arbitrary pixel path, integrate each **segment displacement**, never multiply the entire path-averaged gradient by `x - b`.
- Verify signed completeness against the **actual endpoints of the explained function**. If a VAE/AE reconstructs the images, explain reconstructed endpoints or explicitly include common correction segments. Never report reconstructed-path completeness for unreconstructed original inputs.
- Do not optimize the primary geodesic metric on the explained classifier's gradients or final-test coalition errors. The metric and representation encoder must be independent of the trained target classifier.
- E1 primary evaluation is **counterfactual coalition score-reconstruction error**, not saliency beauty, overlap alone, insertion/deletion alone, or performance relative only to black-baseline IG.
- The study must report negative controls, nonconvergence, reconstruction failures, selection/exclusion counts, numerical error, memory/time, and unfavorable comparisons.
- Shapley over the small, exact factor-coalition game is a **reference allocation**, not a weak baseline to beat and not proof of causal identifiability.
- A single-factor intervention has limited power to distinguish path allocations; E1 requires at least two changed predictive factors per pair. E2 must report how many genuinely multi-part comparisons are available.

## Operational constraints

- Host: remote Linux server via user-provided SSH access; GPU target NVIDIA RTX A4000 **16 GB**; use `nvidia-smi` to verify actual hardware and driver. **Never assume** a working CUDA stack.
- Packaging: `uv`, `pyproject.toml`, committed `uv.lock`; Python 3.11 preferred. Install a verified NVIDIA-driver-compatible CUDA PyTorch wheel using a pinned uv index. Never silently fall back to CPU for GPU stages.
- Experiments: local MLflow **on the remote host**, with SQLite backend and proxied local artifacts; code connects to `http://127.0.0.1:5000`. Run long-lived jobs in named `tmux` sessions. Access browser UI via `ssh -L` rather than publishing port 5000.
- **Do not run** `sudo ufw enable`, change firewall/SSH policies, expose MLflow on `0.0.0.0`, use wildcard `--allowed-hosts '*'` or `--cors-allowed-origins '*'`, or assume administrative rights. The sample broad exposure commands from the user are not the secure default.
- No secrets in source, artifacts, MLflow parameters, screenshots, or reports. Use an ignored `.env`/host environment. No destructive cleanup of remote files unrelated to this repository.
- Compute budget: **40 A4000 GPU-hours total**, per-stage ceilings in `configs/protocol.v1.yaml`. Keep machine-readable per-run GPU times and peak memory. CPU-only E0 has no GPU allocation.
- No expensive external repository/model downloads before licensing, disk capacity, provenance and checkpoint integrity checks. Reference implementation code is for comparison, not grounds for transplanting whole dependencies into the project.
- Do not announce runs as completed unless they actually executed and wrote verifiable artifacts. If SSH, CUDA, dataset, or checkpoint access is unavailable, implement the pipeline and report an explicit `BLOCKED` status; never synthesize metrics.

## Execution order and stopping

1. Inspect remote system, configure project, `uv lock` and `uv sync --locked`, MLflow healthcheck and smoke artifact upload/download.
2. Implement unit tests and E0. Fix correctness defects before any benchmark conclusions.
3. Generate E1 datasets/splits with a fixed manifest; train target classifiers (two seeds), then the independent unlabeled encoder; calibrate/validate on validation only.
4. Profile 10 E1 pairs, freeze reasonable resource-aware numerical settings, execute full E1 exactly once under the primary protocol. Reruns of failed infrastructure are permitted with traceable failure reasons; do not reroll hard examples.
5. Implement and preflight E2. Execute full E2 **if E1 meets Gate B and E2 validity preflight passes**. If E1 fails, execute only a bounded E2 smoke/preflight and record `NOT_RUN_BY_DESIGN`; do not spend the external-validation budget to rescue a failed primary hypothesis.
6. Aggregate all MLflow runs, calculate intervals from saved per-scene rows, write report and an unambiguous `GO`, `AMBER`, `NO_GO`, or `BLOCKED` verdict. `GO` requires Gate A + B + C. `AMBER` means unconfirmed, **not approved for full investment**.

No protocol goal can be rescued by testing extra seeds, metrics, image subsets, or alternative encoder weights after seeing test outcomes. Such analyses belong in a clearly separated exploratory appendix.

## Required repo interfaces

Build a `src/geodesic_ig_pilot/` package and `tests/` (see `docs/ENGINEERING_PLAN.md`). Provide CLI commands via `uv run python -m geodesic_ig_pilot.cli ...`:

- `doctor`, `smoke-mlflow`, `exp0`, `prepare-e1`, `train-e1`, `fit-encoder`, `evaluate-e1`, `prepare-e2`, `evaluate-e2`, `aggregate`, `report`.
- `--config configs/protocol.v1.yaml`, `--output-root`, `--resume`, `--dry-run` for applicable commands. `--resume` must skip completed immutable units using hashes, not silently overwrite them.
- Each stage writes a JSON status summary and machine-readable sample-level results. MLflow must link to these artifacts.

## Tests to implement first

- Finite difference vs autograd on analytic scores and renderer.
- Analytic E0 for `c in {0,1,4}` with absolute error `<1e-3`; additive path independence.
- Pixel-group versus latent-factor attribution equivalence on disjoint-support renderer.
- Completeness for straight and curved paths; signed attribution direction and nonzero endpoints.
- Metric symmetry and positive-definiteness for nondegenerate epsilon; latent endpoint immutability.
- Exact enumeration of all `2^m` factor coalitions, with hand-checkable small examples.
- Stable IDs, no split leakage, no test-based hyperparameter selection, deterministic seed hashes.
- Bootstrap resampling grouped by scene ID, pairing across methods and averaging seed outcomes within scene.
- MLflow roundtrip including logged file artifact and replayable run metadata.

## Required final outputs

- `reports/FINAL_REPORT.md` conforming to `docs/REPORT_TEMPLATE.md`, with numerical table, CIs, failure rates, figures, links/IDs, GPU-hours, verdict and limitations.
- `reports/decision.json` machine-readable gate statuses and reasons.
- `reports/metrics/*.parquet` per-pair/method/seed data; `reports/figures/` plots; `reports/deviations.md` (even if 'none').
- `reports/reproducibility.md` with OS/CUDA/GPU, `uv.lock` hash, git hash, dataset/checkpoint SHA-256, all invocation commands, MLflow run/experiment identifiers, and replay instructions.
- Unit/integration tests passing, README updated with actual run instructions, and a clean git worktree or a clear list of uncommitted changes.

## Communication standard

Work autonomously with available credentials and tools. Record concise checkpoints after each major stage. At completion, summarize actual completed/stopped steps, whether preregistered gates passed, and where the report is. If a requirement cannot be met, document why and what it prevents us from concluding. Treat a negative result as meaningful, not as a failure to be cosmetically repaired.
