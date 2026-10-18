# Geodesic Integrated Gradients — Go/No-Go Pilot

**Status (2026-10-18): executed. Verdict `NO_GO`** — Gate A PASS, Gate B FAIL, Gate C NOT_RUN_BY_DESIGN.
See [`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md) and [`reports/decision.json`](reports/decision.json).

**Question:** does a learned semantic-representation pullback geodesic (M4) improve *contrastive attribution
fidelity*, measured as counterfactual coalition score-reconstruction error, over pixel IG (M1), decoded latent-linear
IG (M2), image-pullback geodesics (M3) and random smooth paths (M5)? Model, input, baseline and score are held fixed;
only the path varies.

**Answer from this pilot:** no. On 240 frozen test scenes, M4 had higher error than the validation-selected comparator
(M5): r_strong = −0.043 [−0.075, −0.015], r_all = −0.053 [−0.082, −0.028], r_weak = −0.108 [−0.170, −0.050]
(paired scene bootstrap, 2,000 resamples). M4 was indistinguishable from M2/M3 and beat only pixel IG on
strong-interaction pairs. All numerical checks passed. Total measured GPU time was about 0.1 A4000 GPU-hours.

The preregistration is unchanged: [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md),
[`configs/protocol.v1.yaml`](configs/protocol.v1.yaml) (SHA-256 `521ce0a9…0b9017`). The original starter README is
preserved verbatim in [`docs/STARTER_README.md`](docs/STARTER_README.md).

## Evidence

| What | Where |
|---|---|
| Final report, verdict, gates | [`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md), [`reports/decision.json`](reports/decision.json) |
| Deviations + unblinding timeline (pre-unblinding part committed before test) | [`reports/deviations.md`](reports/deviations.md) |
| Hashes, versions, run IDs, replay commands | [`reports/reproducibility.md`](reports/reproducibility.md) |
| Frozen selection manifest (read-only) and unblinding stamp | `reports/manifest/frozen_manifest.json`, `reports/manifest/unblinding.json` |
| Per-scene / per-seed / per-method rows (incl. invalid flags) | `reports/metrics/e1_test_rows.parquet`, `e1_test_coalitions.parquet`, `e1_test_paths.parquet`, `e1_test_oracle.parquet`, `e1_test_scene_paired.parquet` |
| Figures | `reports/figures/` |
| Stage status JSON (authoritative stage outcomes) | `reports/status/*.json` |

`reports/metrics/` and `reports/figures/` are git-ignored by the starter `.gitignore`. They are present in this
working copy, as MLflow artifacts on the remote server and under `~/work/geodesic-ig-pilot/reports/` on the GPU host.

## Layout

```text
src/geodesic_ig_pilot/
  cli.py          doctor, smoke-mlflow, exp0, prepare-e1, train-e1, fit-encoder, evaluate-e1,
                  prepare-e2, evaluate-e2, aggregate, report
  config.py       immutable protocol loading, threshold guard, hashes, named RNG streams
  tracking.py     MLflow (loopback-only), provenance tags, atomic writes, stage status, GPU ledger
  renderer.py     64x64 differentiable 4-factor renderer with disjoint factor supports
  scores.py       E0 scores, E1 teachers (g_poly, g_NL, g_short), CNN logit margin
  geometry.py     E0 analytics, finite-difference pullback energies, endpoint-fixed projected-Adam path optimizer
  integration.py  segment-displacement midpoint IG (latent and pixel), completeness refinement
  datasets.py     calibration of C, stochastic labels, deterministic stratified pair manifest, leakage checks
  models.py       small CNN classifier, unlabeled masked autoencoder
  evaluations.py  coalitions, normalised coalition error, exact Shapley, interaction index
  statistics.py   seed-averaged paired scene bootstrap, conservative invalid handling, Gate B
  e0.py / e1.py   stage implementations (E1 test refuses to run unless the frozen manifest hashes match)
  external.py     FunnyBirds adapter: HTTP-range member download, AE path space, preflight, part AP
  report.py, report_md.py   mechanical aggregation, figures, decision.json, FINAL_REPORT.md
scripts/ start_mlflow.sh, run_stage.sh (tmux + log + atomic exit code), run_e1_pipeline.sh
tests/   28 tests (gradients, renderer support, E0, completeness, pixel/latent equivalence, Shapley,
         coalitions, determinism/leakage, bootstrap pairing, gates, MLflow roundtrip, loopback guard)
```

## Commands (as actually run)

Remote host (Linux, RTX A4000 16 GB), inside `~/work/geodesic-ig-pilot`:

```bash
uv sync --locked                                   # torch 2.14.1+cu126 via explicit index (Linux only)
scripts/start_mlflow.sh                            # tmux "mlflow", 127.0.0.1:5000, SQLite + proxied artifacts
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
uv run python -m geodesic_ig_pilot.cli smoke-mlflow --config configs/protocol.v1.yaml
uv run python -m geodesic_ig_pilot.cli doctor --config configs/protocol.v1.yaml
uv run pytest -q
scripts/run_stage.sh gig-e0 exp0 --device cpu      # E0 (CPU)
scripts/run_e1_pipeline.sh                         # prepare-e1, train-e1, fit-encoder, evaluate-e1 --split validation (+freeze)
scripts/run_stage.sh gig-e1-test evaluate-e1 --split test --resume   # single blinded test pass
uv run python -m geodesic_ig_pilot.cli aggregate --config configs/protocol.v1.yaml
scripts/run_stage.sh gig-e2prep prepare-e2         # FunnyBirds provenance + selective download
scripts/run_stage.sh gig-e2pre evaluate-e2 --phase preflight
uv run python -m geodesic_ig_pilot.cli aggregate --config configs/protocol.v1.yaml
uv run python -m geodesic_ig_pilot.cli report --config configs/protocol.v1.yaml
```

Smoke runs: `--smoke` uses tiny sizes in `outputs/smoke-<device>/` (never used for gates), e.g.
`GIG_DISABLE_MLFLOW=1 uv run python -m geodesic_ig_pilot.cli prepare-e1 --config configs/protocol.v1.yaml --smoke --device cpu`.
Every long command accepts `--resume` (hash-verified skipping) and `--dry-run`.

MLflow UI from a laptop: `ssh -N -L 5000:127.0.0.1:5000 <authorized-ssh-alias>`, then open http://127.0.0.1:5000.
MLflow is never bound to a public interface, and no firewall or SSH configuration was changed.
