# Engineering plan and implementation contracts

Build a small, testable PyTorch research package. Avoid unnecessary libraries and custom CUDA extensions; the RTX A4000 has 16 GB VRAM and the experiment depends more on correctness than throughput.

## 1. Environment and dependency choices

- Linux, Python **3.11**; `uv` for `pyproject.toml`, `uv.lock`, synchronization and `uv run` entry points.
- PyTorch + torchvision CUDA build compatible with the **actual driver** (verify `nvidia-smi` before selecting `cu118`, `cu126`, etc.). Use an **explicit PyTorch uv index**, as described in the official uv/PyTorch guide: `[[tool.uv.index]] name='pytorch-cu126', url='https://download.pytorch.org/whl/cu126', explicit=true` plus `[tool.uv.sources]` for `torch`/`torchvision`. The `cu126` tag is illustrative, **not permission to use it if incompatible**. Commit `uv.lock` and record SHA-256. Do not mix `pip install` into a uv-managed reproducible environment.
- Suggested small stack: numpy, scipy, PyYAML, pydantic, pandas, pyarrow, scikit-learn, matplotlib, pillow, tqdm, mlflow, torch, torchvision, pytest, ruff. Add optional packages only when justified.
- Use fp32 for E0 and attribution integrals; AMP/bfloat16 may be used for classifier/encoder training after testing accuracy and gradient reliability. Do not differentiate a geodesic energy through an underflowing fp16 pipeline.
- Set deterministic generators, log CUDA/cuDNN deterministic configuration and known nondeterministic operations; log source code, hardware and package versions.

## 2. Modules and responsibilities

Suggested package:

| Module | Contract |
|---|---|
| `config.py` | Load and validate immutable protocol; hash files; protect final-test settings |
| `geometry.py` | Exact E0 metric and numerical geometric energy; finite-difference hybrid pullback; endpoint-fixed Adam/L-BFGS optimizer and path checks |
| `integration.py` | Signed midpoint path integration in pixel space and via chain rule in disjoint-factor latent space; completeness diagnostics |
| `renderer.py` | Differentiable 4-factor renderer; fixed semantic masks; factor-separated Jacobians |
| `scores.py` | E0 additive/interaction scores; E1 polynomial/nonlinear analytic teachers; CNN logit margin |
| `models.py` | Small CNN classifier and separately trained unlabeled encoder; model checkpoint hashes |
| `datasets.py` | Train/val/test manifests, counterfactual baselines, oracle strata; exact hybrid coalitions and leakage tests |
| `evaluations.py` | Coalition score tensors, error, Shapley reference, endpoint/nuisance validity, AP for external study |
| `statistics.py` | Within-scene two-seed averaging, matched bootstrap, stratified confidence intervals, gated decisions |
| `tracking.py` | MLflow experiment/run creation and provenance, machine-readable artifact/logging schema |
| `external.py` | FunnyBirds model/data adapter, reconstruction preflight, independent part-level evaluator |
| `cli.py` | Explicit stage commands; resumable idempotent runs and helpful precondition errors |

`python -m geodesic_ig_pilot.cli --help` and subcommand `--help` must work. Use standard exit codes (`0` success, nonzero failure), atomic local writes, and fully qualified run IDs for resumability.

## 3. Minimal experiment algorithms

### E0: numerical oracle

Implement `D_c`, `f=x_1 x_2`, the exact geodesic via invertible arc-length transform `s(u)`, and pixel/latent attributions. Independently implement a waypoint energy optimizer, not merely evaluation of the analytic path, so E0 tests the optimizer. Compare to analytically computed `(A_u,A_v)` values in `RESEARCH_SPEC.md`. Add additive score tests and random path completeness.

### E1: sample generation

Use train-only calibration to determine `C` once. Deterministic `numpy.random.Generator` streams must be disjoint for:

- training dataset and stochastic labels;
- unlabeled encoder training;
- classifier validation/test;
- attribution validation pair selection;
- attribution held-out pair selection;
- M5 random paths;
- bootstrap statistics.

All E1 image-generation inputs must be logged as structured factor vectors. Generate enough candidates with oracle contrast and 2+ predictive changes; stratify before any target-model attributions. Save files as Parquet/JSON with `scene_id`, `pair_id`, `baseline_z`, `target_z`, `changed_factors`, `eta`, `stratum`, `split`, `input_label_oracle`, `baseline_label_oracle`, `nuisance_changed`. Enforce the predeclared 50/50 nuisance-changed quota per interaction stratum and save a content hash; reject scene-ID leakage automatically.

Train the same CNN architecture with two independent seeds. Accuracy and AUROC floors are in `PREREGISTRATION.md`. Use a small network such as 3-4 convolution blocks with group/batch normalization and a compact MLP head. Fix training budget/hyperparameters on training + classifier validation; no test-driven search. Save checkpoints and training curves. Do not manufacture classifier confidence by post-hoc scoring on attribution test pairs.

Train an independent learned unlabeled encoder with a clearly specified objective (compact masked-reconstruction autoencoder or standard self-supervised contrastive learning). Train only on unlabeled training scenes; no factor information or labels and no classifier-derived gradients. Freeze before tuning the geodesic metric. Record feature dimensionality, feature normalization and any collapse/constant-feature diagnostics. An oracle factor encoder is diagnostic only and must have different `method_id`.

### E1: paths

All sample paths must have `z[0]==z_b`, `z[-1]==z_x` exactly. For M1 create interpolation in input space; for M2 decoded latent-linear; for M3/M4 minimize discrete energy with `K=16` segments by default. Suggested normalization of hybrid terms:

1. On validation only, estimate median squared feature changes and median squared pixel changes over small random latent displacements.
2. Normalize individual terms to comparable scale, and tune `alpha` from a small locked grid (e.g. `0.1, 1.0, 10.0` relative to normalization; exact values and choice go in frozen manifest).
3. Fix `epsilon` for metric numerical regularity before test; report sensitivity as **secondary**.
4. Use three initial paths: linear plus two seeded smooth perturbations, selecting only by minimal geometric energy. Record initial and final energy, feasibility, solver status and runtime.

M5 should use fixed-seed cubic/smooth endpoint-preserving paths with nontrivial deformation, matched to an admissible **energy band** as closely as possible. An exact length match to the unique shortest geodesic is in general impossible; never assert one. Record achieved energy ratios and the number of rejected random candidates.

For attribution, use *separate* path quadrature (64 initial segments) and gradient evaluations for the **same frozen classifier**. Compute latent-gradient factor attribution for M2–M5 and pixel-group attribution for M1. Check completeness. In E1's disjoint-support renderer, independently cross-check latent and pixel integrals on several examples to test the chain rule implementation. If an actual model changes nuisance even when it should not, the score difference belongs to the model and should be attributed, not hidden.

### E1: exact hybrid evaluation

For each scene and each changed-factor subset, render hybrid `z_S` and cache the **learned classifier margin** for each of the two seeds. Because `m<=4`, compute the entire powerset (at most 16 evaluations) without approximation. Evaluation is independent of chosen path; do not expose these test coalition scores to M4 optimization or hyperparameter tuning.

Compute normalized error for every method. Save **per-pair per-seed** records, plus all attributed feature values and numerators/denominators. Compute exact Shapley reference from the same coalition values for interpretative comparison. Use sign-specific diagnostics, rather than dropping negative attributions or taking absolute values prematurely.

### E2: supported independent benchmark

Start with `docs/EXTERNAL_VALIDATION.md`. The key implementation challenge is a differentiable decoder with semantically valid endpoints. The original FunnyBirds rendering framework is not itself a PyTorch-differentiable decoder. Candidate solutions include a small pretrained or lightly adapted convolutional autoencoder/VAE, but reconstruction **must** pass independent preflight checks. For external images, report pixel-space attribution grouped by image part masks, unless you can mathematically justify the factor-to-pixel equivalence.

E2 should default to a **pretrained FunnyBirds ResNet-50** and compare fixed endpoint/path variants. DINOv3 is an optional frozen representation if it fits 16 GB; record checkpoint provenance and use stagewise feature caching/activation checkpointing/offload if needed. The independent external classifier must not be used to train the semantic encoder or choose test paths by coalition error.

## 4. Logging contract

Create separate MLflow experiments `gig-pilot/e0`, `gig-pilot/e1-models`, `gig-pilot/e1-paths`, `gig-pilot/e2`, `gig-pilot/decisions` (or a documented equivalent). Every run must log:

- tags: `protocol_sha256`, `git_commit`, `uv_lock_sha256`, `host`, `gpu_name`, `cuda_driver`, `pytorch_cuda`, `dataset_manifest_sha256`, `model_checkpoint_sha256`, `representation_checkpoint_sha256`, `method_id`, `split`, `stage`;
- parameters: fixed numeric settings, selected baseline comparator, target score, selected metric family, path restarts, seed;
- metrics: errors, numerator/denominator, completeness, constraint validity, runtime, peak GPU memory, GPU-hours, failures, classifier metrics;
- artifacts: config snapshot, sample manifest, training/solver curves, per-example Parquet, figures, failed-case examples, dependency summary, checklists and aggregate decision JSON.

Do not log secrets; do not hide NaNs by replacing them with zero. Store samples as **file artifacts**, not only scalar metrics. Capture MLflow run IDs for every result in the report.

## 5. Required tests and static checks

Before any test-set run, implement tests for:

1. `grad(f)` analytic vs `autograd` (including `f_NL`, renderer Jacobian and E0).
2. `D` factor masking and zero cross-support derivatives.
3. E0 analytic geodesic numerical integration and optimizer energy result.
4. Additive score path independence.
5. Endpoint invariance, metric symmetry, epsilon positive-definiteness, finite solver outputs.
6. Completeness for latent-linear, random curved, geodesic and pixel interpolation.
7. Pixel-group attribution vs latent-factor attribution on a factor-separable renderer.
8. Exact Shapley hand checks for `f=a+b` and `f=a*b` with endpoints `(0,0)->(1,1)`.
9. Coalition enumeration, stable ordering, normalization and consistency of `Delta_empty=0`.
10. No scene leakage; same baseline/input/model checkpoint across M1–M5.
11. Bootstrap unit: paired methods and classifier seeds are resampled by scene.
12. MLflow: artifact roundtrip and safe URI handling.

Use `uv run pytest -q`, `uv run ruff check .`, and `uv run ruff format --check .` before final report. At a minimum, test a complete 4-example dry run on CPU and a 2-example CUDA smoke run prior to full E1.

## 6. Resource-aware implementation

- Optimize paths in small batches, potentially one path at a time; CUDA peak memory **<14.5 GB** is preferable to leave headroom below physical 16 GB.
- Freeze target-model and representation weights while differentiating their outputs with respect to input; `requires_grad=False` on parameters must **not** disable gradients with respect to input.
- Do not wrap attribution code in `torch.no_grad()`; use `autograd.grad` targeted to inputs.
- During path energy optimization, use first-order gradients through `D` and `psi`. Avoid explicit Hessians and dense Jacobians.
- Save results/checkpoints after each scene chunk (10-20 pairs). `--resume` must verify hashes and not recompute completed valid chunks.
- Time the full pipeline, not just neural inference. Report amortized preprocessing (encoder training) separately from per-explanation computation.
- GPU profiling must precede selection of the numeric test configuration. If memory is exceeded, reduce batch size/activation memory first, not quadrature accuracy.

## 7. Status codes and invariant violation behavior

Implement a machine-readable stage outcome:

- `COMPLETE`, `COMPLETE_WITH_FAILURES`, `NOT_RUN_BY_DESIGN`, `INVALID_PROTOCOL`, `BLOCKED`, `RESOURCE_INFEASIBLE`.
- Per-sample reasons: `NUMERIC_INCOMPLETE`, `SOLVER_DIVERGED`, `PATH_OUT_OF_DOMAIN`, `MODEL_FLAT`, `RECONSTRUCTION_FAILED`, `NO_VALID_PAIR`, `OOM`, `DATA_ERROR`.

Do not silently exclude any of these. If E1 Gate B is negative, E2 should still have a smoke/preflight status, but not a full external result. All gate decisions must be mechanically derivable from saved numbers, not authored manually in prose.

## 8. Expected stable CLI contract

```text
uv run python -m geodesic_ig_pilot.cli doctor --config configs/protocol.v1.yaml
uv run python -m geodesic_ig_pilot.cli smoke-mlflow --config configs/protocol.v1.yaml
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

Long-running commands must accept `--resume`; analysis/report generation must be deterministic given saved data, without rerunning expensive experiments. A `--dry-run` command should print tasks, expected memory/samples and budget impact, rather than silently launching them.
