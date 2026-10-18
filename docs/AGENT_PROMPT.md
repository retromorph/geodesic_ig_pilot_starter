# Copy-paste prompt for the coding agent

> Paste the text **below** into a coding agent operating in the project directory. The agent should treat the checked-in docs and YAML as authoritative. This prompt assumes the agent can access the repository files; it does not include SSH credentials or model-access tokens.

---

You are an autonomous senior ML research engineer and experimental scientist. Your job is to **implement, run, audit, and report** a preregistered go/no-go feasibility study for **Semantic/Manifold Geodesic Integrated Gradients**. This is an end-to-end research implementation assignment, NOT a request to draft another plan.

**Repository contract**

Immediately read, in order:

1. `AGENTS.md`
2. `README.md`
3. `docs/PREREGISTRATION.md`
4. `configs/protocol.v1.yaml`
5. `docs/RESEARCH_SPEC.md`
6. `docs/ENGINEERING_PLAN.md`
7. `docs/REMOTE_RUNBOOK.md`
8. `docs/EXTERNAL_VALIDATION.md`
9. `docs/REPORT_TEMPLATE.md`
10. `docs/REFERENCES.md`

Follow the documented scientific and security constraints. You may choose implementation details not fixed by the protocol, but **not** alter the acceptance criteria after inspecting test results. Before any held-out evaluation, snapshot the protocol and git/lockfile hashes into MLflow and a local immutable manifest. If there is a conflict between documents, apply the scientifically stricter interpretation and record the deviation, never quietly pick the easier condition.

## Research question

Does a geometric path minimizing a hybrid pullback metric induced by a learned, classifier-independent representation lead to meaningfully **better contrastive feature attribution** than same-endpoint pixel IG, decoded latent-linear IG, image-pullback geodesics, and random smooth paths?

The crucial comparison isolates **only the integration path**. Model checkpoint, target score, baseline and target image must be identical across methods. An algorithm is **not** successful merely because its attributions look better or because it chooses a closer baseline. You must measure how well factor/part attributions predict independent counterfactual score differences.

**The research may fail. An accurate NO-GO is a valid and desirable outcome. Never optimize for a positive headline.**

## Infrastructure to bring up

1. Inspect whether the authorized remote SSH host/alias is available in the execution environment, without exposing private keys. If it is unavailable, implement and test whatever is possible locally and report a hard `BLOCKED_SSH` status; do not fabricate remote results.
2. On the remote Linux host, inventory GPU, driver, VRAM, Python, disk, tmux, existing GPU processes, and installed uv. Target is one NVIDIA RTX A4000 with 16 GB VRAM. Do not assume any driver or CUDA wheel is usable until verified.
3. Create `pyproject.toml` for Python 3.11, install via **uv**, select an explicitly configured PyTorch CUDA index compatible with the actual driver, and commit/preserve `uv.lock`. `uv sync --locked` must reproduce the environment. Keep dependencies compact.
4. Set up remote MLflow on `127.0.0.1:5000`, with SQLite `mlruns/mlflow.db` and **proxied local artifacts** in `mlruns/artifacts/`. Use `tmux` session `mlflow`; run remote clients against `MLFLOW_TRACKING_URI=http://127.0.0.1:5000`. Provide `scripts/start_mlflow.sh` and verify a real artifact-upload-and-download roundtrip.
5. Use SSH port forwarding to view the remote MLflow UI from a laptop: `ssh -N -L 5000:127.0.0.1:5000 <authorized-ssh-alias>`.
6. **Do not** open port 5000 to the internet, bind MLflow to `0.0.0.0`, use wildcard host/CORS controls, automatically run `sudo ufw enable`, reconfigure SSH, install unrequested system drivers, or leak credentials. The user's example is not the required public exposure configuration.
7. Run all lengthy workloads in named `tmux` sessions. Use `--resume`, checkpoints, atomic result files, logs, exit codes and resource meters; a dropped SSH connection must not cause silent incomplete jobs.

## Build the following software

A documented Python package `src/geodesic_ig_pilot/` exposing CLI commands:

```text
doctor
smoke-mlflow
exp0
prepare-e1
train-e1
fit-encoder
evaluate-e1
prepare-e2
evaluate-e2
aggregate
report
```

All commands accept `--config configs/protocol.v1.yaml`; applicable long jobs accept `--resume` and `--dry-run`. Add tests/ and a small set of scripts to run stages in tmux. Use MLflow runs plus Parquet/JSON artifacts for every phase. Implement deterministic generation, hashed manifests and grouped bootstrap statistics. Produce readable source with docstrings/comments where necessary, without architecture bloat.

### Stage E0 — analytic & numerical sanity checks, CPU first

Implement a latent manifold `D_c(u,v)=(u,v,c*u^2)` and score `f=x1*x2`, endpoints `(0,0)->(1,1)`, with `c=0,1,4`. For image-pullback geodesics, expected `A_u` is approximately `0.500000`, `0.42637293`, `0.35209913`; `A_v=1-A_u`. Verify independently via analytic arc-length and your numerical waypoint optimizer. A latent-linear path gives `(0.5,0.5)`. A semantic representation ignoring third image coordinate recovers the linear solution. An additive `f=x1+x2` must give path-independent coordinate contributions. Include finite-difference gradient tests and signed completeness checks with tolerance `<1e-3`.

If E0 fails, debug before any scientific interpretation. `Gate A = BLOCKED` when an unresolved numerical/implementation bug persists; it is not an empirical NO-GO.

### Stage E1 — primary controlled experiment

Implement a simple differentiable 64x64 RGB renderer with disjoint pixels/regions controlled by `(a,b,c,n)`; the first three factors affect parts, the last affects a nuisance background. Nonlinear image changes are important. Build and test the non-overlap property of renderer Jacobian supports, so latent-factor path attributions equal sums of pixel-group attributions.

Use the *specified* polynomial and nonlinear teacher scores from `docs/RESEARCH_SPEC.md`. Calibrate the nonlinear centering constant `C` on the fixed train-only calibration draw and preserve it. Sample labels with `p(y=1|z)=sigmoid(3*g_NL(z))`; train a small CNN twice (seeds 11 and 29, 10k train images per seed) using only training/validation. Record AUROC, accuracy and calibration. Target explanation is **the CNN's frozen logit margin**. An oracle generative score is a separate diagnostic target, never a substitute for the primary result.

Train a **separate learned representation encoder** from unlabeled training images (no class labels, no factor vectors, no oracle support masks for representation pretraining, no classifier gradients, no held-out scenes). Freeze it before path experiments. If you also create an oracle factor representation for an upper bound, mark it `diagnostic`, ineligible for GO.

Generate a deterministic pair manifest with distinct scene IDs and independent splits: 80 validation pairs and test sets of 120 **strong-interaction** and 120 **weak-interaction** contrastive pairs. Select strata using the analytic teacher interaction index, not learned attributions; require 2+ predictive changed factors and opposite oracle teacher class. The exact seed, interaction cutoffs and candidate cap are in `configs/protocol.v1.yaml`. Match a 50/50 quota of unchanged versus changed nuisance within each interaction stratum as specified in the research protocol. Do not reroll inconvenient test pairs.

Compute five methods for **exactly the same** `(b,x, CNN seed, score)`:

- M1 pixel straight-line IG;
- M2 decoded latent straight-line IG;
- M3 image-pullback geodesic IG;
- M4 proposed hybrid learned-representation/image pullback geodesic IG;
- M5 fixed-seed random smooth latent path IG, averaged across defined candidates with actual length/energy recorded.

Use 16 path segments, 100 Adam iterations initially, three energy-only restarts, 64-point gradient quadrature initially, and adaptive numeric refinement as specified. Optimize paths by their **geometric energy**, not by held-out attribution scores or classifier gradients. Use small batches, frozen models, no full dense Jacobian/Hessian, and fp32 attributions; satisfy A4000 16 GB. Profile 10 validation pairs and freeze resource-appropriate numerical details before opening the test set.

For each held-out baseline/target pair, independently render all `2^m` hybrid scenes for changed factors, compute `Delta_S=f(D(z_S))-f(D(z_b))` using the frozen CNN, and evaluate the normalized full-coalition squared reconstruction error of each signed attribution vector. This is the **primary metric**. Compute exact small-game Shapley values as diagnostic reference. Record nuisance attributions, paired classifier-seed results, completeness, path failure rates, interaction index, path energy, runtimes and peak VRAM. Strong numerical checks and no train/val/test leakage are mandatory.

Select one primary comparator from M1/M2/M3/M5 based **only** on equal-stratum-weighted validation mean coalition error; select M4 metric hyperparameters using validation only. Freeze these before running the 240 scene test. Bootstrap 2000 times **by scene ID**, keeping methods paired and averaging the two model seeds within scene. Don't use unpaired confidence intervals.

**Gate B thresholds, all required:**

- `r_strong = 1 - mean(E_M4,strong)/mean(E_B,strong) >= 0.15`;
- lower bound of paired 95% bootstrap CI for `r_strong` > `0.05`;
- `r_all >= 0.10`, lower 95% bound > `0`;
- `r_weak >= -0.05`;
- positive strong-set improvement for each of seeds 11 and 29;
- a learned non-oracle encoder, classifier competence and all Gate A numerical tests pass.

If comparator mean error is near zero, flag the ratio as uninformative; report absolute error instead, and **do not** force a pass. If only pixel IG loses to latent paths, do not claim geodesic superiority. If results show no meaningful improvement, do not try numerous ad hoc encoder losses or path weights on the test set to manufacture it.

### Stage E2 — independent part-based evaluation

Implement an adapter around FunnyBirds (ICCV 2023) using the verified official evaluation framework and a pretrained ResNet-50 or another legitimately available checkpoint. Verify source revision, dataset license, checkpoint integrity and model correctness. E2 uses held-out part interventions and an explicit **part-level AP** computed from independent model-response-defined part importance. The framework's published metrics are not automatically AP; define the exact metric and labels and lock them using validation.

**Important:** the FunnyBirds/Blender renderer is not a differentiable PyTorch decoder. A differentiable image path representation is necessary for computing pullback geodesics. Use a compact verified AE/VAE decoder only if it can preserve both endpoint class decisions and parts. The initial external preflight includes 12 cases, and the validation set 20, before 120 held-out examples. Compare paths on **exactly the same reconstructed endpoints**; attribute in **pixel space**, grouped by semantic part masks. A general AE latent coordinate is not a semantic object part.

A small frozen DINOv3 backbone is optional for the representation metric if permitted by its license and within 16 GB memory; alternative pretrained learned representations are allowed and must be documented. Adapt contemporaneous MA-GIG as an extra baseline only when model/score/endpoints match fairly and cost permits. Do not claim superiority over unrun work.

**Conditional execution:** Implement all E2 code and run a smoke/preflight regardless. Execute **full** E2 only if E1 satisfies Gate B and reconstruction/data validity passes. If E1 fails, mark `NOT_RUN_BY_DESIGN` and keep the E2 smoke report; this respects the preregistered sequential stop and conserves compute. If reconstruction fails, mark `E2_INCONCLUSIVE`, not a fabricated negative AP.

**Gate C thresholds if full E2 is valid:** AP_M4 - AP_best_validation_selected_same_endpoint_baseline >= **0.03 absolute**, with paired 95% scene-bootstrap lower CI > **0**. At least 120 prespecified external valid scenes and realistic intermediate path diagnostics must be reported, otherwise flag the gate inconclusive. A positive E1 with inconclusive E2 is **AMBER**, not a top-tier readiness claim.

### Budget

Hard planning budget: **40 A4000 GPU-hours**, of which up to 28 for E1, 10 for E2, 2 for smoke/profiling. E0 CPU. Record actual device-time, wall-clock, peak memory and OOM events. Avoid training a diffusion classifier or large generator; that is not required by this pilot. Do not quietly relax numerical accuracy, switch benchmarks, exclude failures, or substitute a GPU larger than 16 GB without flagging a protocol violation. If infeasible under this budget, report it honestly and stop.

## Reporting requirements (non-negotiable)

Log real MLflow runs and artifacts and save:

1. `reports/FINAL_REPORT.md`, following `docs/REPORT_TEMPLATE.md`, with **measured** results, methods, failure cases, selected comparator, confidence intervals, GPU-hours and clear **GO/AMBER/NO_GO/BLOCKED** verdict.
2. `reports/decision.json` calculated from preregistered thresholds; a human-readable explanation for each gate.
3. `reports/metrics/` per-scene/per-seed/per-method machine-readable results, including missing and invalid samples.
4. `reports/figures/`: scene-paired error-difference plot, error vs interaction strength, geodesic-energy/numeric residual diagnostics, side-by-side positive and negative attribution examples, E2 part metrics if valid.
5. `reports/deviations.md` logging all code/protocol issues, amendments, rerun reasons, noncomparable baseline/implementation constraints, and unblinding timeline.
6. `reports/reproducibility.md` with `uv.lock` hash, git commit, Python/PyTorch/CUDA/driver, dataset/checkpoint hashes, MLflow run IDs, exact commands, paths and successful tests. Do not include SSH secrets or public MLflow links.
7. Tests: `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .`, plus a CPU end-to-end smoke and a GPU mini-run.
8. `README.md` updated from a starter spec to actual commands and evidence links, without overwriting/erasing the original preregistration.

**Final communication:** Tell me exactly what ran on the remote GPU and what did not; give file paths and MLflow experiment/run IDs; show the preregistered Gate A/B/C verdicts and strong/weak/all errors and confidence intervals. Summarize unfavorable findings, runtime and limits. If SSH/model/data access was unavailable, state the specific blocker, leave the implementation as complete as possible, and do not imply runs were performed.

## Quality/ethics requirements

- Don't use unknown third-party code without checking licenses and commit provenance.
- No secrets, open MLflow HTTP service, root-level firewall changes or unauthorized model access.
- No hidden exclusion of hard pairs, validation-on-test, model score changes across methods, post-hoc metric selection, or selective plotting.
- Never claim mathematical completeness for a path whose reconstructed endpoints do not equal the stated original endpoints.
- Distinguish explanatory fidelity to a specific trained model from human semantic plausibility, and distinguish both from causal feature necessity.
- Do not spend compute on diffusion classification until a subsequent, explicitly authorized research phase.

**Start execution now:** Inspect files/environment, record a concise task checklist, then implement and work through the stages autonomously. Make scientifically defensible decisions where instructions leave discretion, document those choices *before* evaluating the held-out set, and complete the report even if the final verdict is NO-GO or BLOCKED.
