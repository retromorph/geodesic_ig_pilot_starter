# Reproducibility record

No SSH secrets, host addresses or public MLflow links appear here. The GPU host is identified by the hashed
`host` tag that is stored on every MLflow run.

## Environment

| Item | Value |
|---|---|
| GPU host OS / kernel | Ubuntu 26.04 LTS / 7.0.0-15-generic, x86_64, 8 vCPU, 31 GiB RAM |
| GPU / driver | NVIDIA RTX A4000, 16376 MiB (15.6 GiB usable to PyTorch) / driver 595.71.05 (CUDA 13.2 capable) |
| Python | CPython 3.11.13 (uv-managed) |
| PyTorch / CUDA wheel | 2.14.1+cu126 / 12.6 (explicit index `https://download.pytorch.org/whl/cu126`, Linux-marker scoped) |
| MLflow | 3.17.1, `127.0.0.1:5000`, SQLite `mlruns/mlflow.db`, proxied artifacts `mlruns/artifacts/` |
| uv | 0.7.12 (laptop and GPU host) |
| `uv.lock` SHA-256 | `44a2e38747550f63fcb378b0ae08800bc92c6c768946e0333d28dfd04ad580ae` |
| `configs/protocol.v1.yaml` SHA-256 | `521ce0a919ea58bac42c2c4f2c369bd6842e039bce047378d5c7d7afcd0b9017` |
| `docs/PREREGISTRATION.md` SHA-256 | `73e0665bc1d831984935921780c2223c1dd79b423be01dab5ff31be8ff73927b` |
| Starter bundle | all 13 entries of `SHA256SUMS.txt` verified OK at session start. `README.md` was later rewritten on purpose; the original is `docs/STARTER_README.md` (`52aaa8e1…`) |
| Git commit used for all E1 runs | `76474c5f7fef54a66f22270ca9210486098b1786` (branch `pilot-run`). MLflow `git_dirty` = False for prepare/train/encoder/validation; **True for the test run** (non-E1 files `external.py`, `run_stage.sh`, `deviations.md` had been updated on the host). The E1 source hash was identical on all runs; those files were committed locally as `1e85d35` |
| E1 frozen-module source SHA-256 | `0a390006a0bc088d921377532fd3a9a2991adb6abd361118f1c91fbc223aee76` (re-verified locally after all later edits: MATCH) |
| Frozen manifest SHA-256 | `3a7fa2d2fc26bbaa01744f62e79b5155b0fe979c087d54ec51951cde09e1f085` (frozen 18:42:29Z; unblinded 18:44:41Z) |

The modules that changed after the E1 runs (`cli.py`, `external.py`, `report.py`, `report_md.py`) are outside the
frozen E1 module set. They only affect smoke exit codes, E2 MLflow logging and reporting.

## Data and checkpoints (SHA-256)

| Artifact | SHA-256 |
|---|---|
| `data/e1/pairs_test.parquet` | `8f869982e6f67ee74e3b39695c9a6314001d203c966dbb92d5863c54ba6c54a9` |
| `data/e1/pairs_validation.parquet` | `77ff1908005c5ca367a0ffc80a1dcde9165ba85a3fa5715637d0f04ef44449a1` |
| `data/e1/scenes_train_seed11.parquet` | `2a852c4ee320ec9685212e7c7252b7c79c087c007b5b79a2939d7038f2a07491` |
| `data/e1/scenes_train_seed29.parquet` | `669a4703d3813fe975b3df02263def84129d5b28d747cfb57d2c1b762dee8f38` |
| `data/e1/scenes_encoder_train.parquet` | `c67c740c69c939d0509501bc0eb908be9d702bc0342ef7ac3b1059241cf4879d` |
| `data/e1/scenes_encoder_val.parquet` | `aad18849cc44430c87bb0881289c91de78f3c05e591caf58f67e6b9febf5649d` |
| `data/e1/scenes_classifier_val.parquet` | `bd041ed3850dc5e914f7f4c6466c67ccb2e4bec7e5856910c3a477c635cea075` |
| `data/e1/scenes_classifier_test.parquet` | `8126bff15946ffa95e848ed177ffdaaf0fc67dece463be0a831b20e30cc79151` |
| CNN seed 11 (`checkpoints/cnn_seed11.pt`) | `61f6c8e19b79579854d20f06712c9df71772a03e650ce90a09a28cbd3c37e723` |
| CNN seed 29 (`checkpoints/cnn_seed29.pt`) | `a4613068d900838890896142893315d7a968ab77cbc0072e82ba0f74b4721cfc` |
| Unlabeled masked AE (`checkpoints/encoder_ae.pt`) | `ececefa1fe292b009acb0668688378dd6035415a2431975b063892fc552a85c3` |
| FunnyBirds ResNet-50 (`resnet50_final_0_checkpoint_best.pth.tar`, ETag `"64ccc879-b47b87d"`) | `88f0fcf517ab5aa318325db7ffef46d65b52f0b8e18c325f5280d97f631ad139` |
| torchvision ResNet-18 IMAGENET1K_V1 (`resnet18-f37072fd.pth`) | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |
| E2 selection (`data/e2/e2_selection.json`) | `775800cfa9dbbef12a716913e5e16ead76c6c3a9ae087f55b8e018c9963e4d1c` |

The calibrated teacher constant is C = 2.2394745583776543 (median of 20,000 train-only draws). The checkpoint
binaries live on the GPU host (git-ignored). Their JSON metadata is in `checkpoints/*.json`.

## MLflow (experiment id → runs); view via `ssh -N -L 5000:127.0.0.1:5000 <alias>`

| Experiment | Run name → run ID |
|---|---|
| 1 `gig-pilot/smoke` | mlflow-roundtrip → `8e8a2e73f5d644149ffcf560b7899cd6` (upload/download SHA-256 match, proxied `mlflow-artifacts:` URI) |
| 2 `gig-pilot/e0` | exp0 → `70f5955fcf734b518ef3c52845e1f293` |
| 3 `gig-pilot/e1-data` | prepare-e1 → `0a6a35ebb159465a9f1e84c2d5e86d2a` |
| 4 `gig-pilot/e1-models` | cnn-seed11 → `842562d201de4281814f85beb98a4ce8`; cnn-seed29 → `eed8ae7514b94578bc370e5168fb8c6e`; encoder-masked-ae → `f5aa46169a7c4d358cfd135b70d7980e` |
| 5 `gig-pilot/e1-paths` | validation → `229a9bdd026842909931270105572f0a`; test → `4957c544f65c4858bb89f502c7c54f87` |
| 6 `gig-pilot/decisions` | freeze → `6929718b3bfd484eb1026be809c24184`; aggregate runs (each re-aggregation is a new run; the latest is authoritative) |
| 7–10 `*-smoke` | GPU mini-run smoke runs (tiny sizes; includes two aborted smoke attempts caused by a smoke-only output-root/exit-code issue) |
| `gig-pilot/e2` | E2 preflight record (see `reports/status/evaluate-e2.json`) |

## Exact commands (GPU host, `~/work/geodesic-ig-pilot`)

```bash
curl -LsSf https://astral.sh/uv/0.7.12/install.sh | sh          # user-level uv
uv python install 3.11 && uv sync --locked
scripts/start_mlflow.sh
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
uv run --locked python -m geodesic_ig_pilot.cli smoke-mlflow --config configs/protocol.v1.yaml
uv run --locked python -m geodesic_ig_pilot.cli doctor --config configs/protocol.v1.yaml
uv run --locked pytest -q                                          # 28 passed on GPU host (29 locally after adding the part-AP test)
scripts/run_stage.sh gig-e0 exp0 --device cpu                     # 36/36 E0 checks PASS
scripts/run_e1_pipeline.sh                                          # prepare-e1 → train-e1 → fit-encoder → evaluate-e1 --split validation (freeze)
scripts/run_stage.sh gig-e1-test evaluate-e1 --split test --resume # single blinded test pass
uv run --locked python -m geodesic_ig_pilot.cli aggregate --config configs/protocol.v1.yaml
scripts/run_stage.sh gig-e2prep prepare-e2
scripts/run_stage.sh gig-e2pre evaluate-e2 --phase preflight
uv run --locked python -m geodesic_ig_pilot.cli aggregate --config configs/protocol.v1.yaml
uv run --locked python -m geodesic_ig_pilot.cli report --config configs/protocol.v1.yaml
# GPU mini-run smoke (tiny sizes, separate output root outputs/smoke-cuda):
for c in prepare-e1 train-e1 fit-encoder "evaluate-e1 --split validation" "evaluate-e1 --split test"; do
  uv run --locked python -m geodesic_ig_pilot.cli $c --smoke --config configs/protocol.v1.yaml; done
```

Laptop CPU end-to-end smoke (macOS, PyPI CPU wheels from the same lock):
`GIG_DISABLE_MLFLOW=1 uv run python -m geodesic_ig_pilot.cli <stage> --config configs/protocol.v1.yaml --smoke --device cpu`
for prepare-e1, train-e1, fit-encoder, evaluate-e1 (validation, then test), aggregate and report. All completed.

Replay from saved data, with no recomputation: `aggregate` then `report`. Both are deterministic given
`reports/metrics/*.parquet`, the frozen manifest and the bootstrap seed 9001. Re-running `evaluate-e1 --split test
--resume` verifies every chunk hash and skips completed chunks.

## Checks run before the report

- `uv run pytest -q` → 29 passed (laptop); 28 passed on the GPU host before the part-AP test was added.
- `uv run ruff check .` → all checks passed; `uv run ruff format --check .` → all files formatted.
- CPU end-to-end smoke (laptop) and GPU mini-run smoke (`GPU_SMOKE_DONE`, exit 0) both completed.
