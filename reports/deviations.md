# Deviations, implementation decisions, reruns and unblinding timeline

Part A was written **before the E1 test set was unblinded** (frozen manifest written 2026-10-18T18:42:29Z by the
validation stage; this file committed before `evaluate-e1 --split test` was launched). Later parts are appended
after unblinding and are labelled as such. Nothing in Part A was changed after unblinding.

## Part A — decisions fixed before test unblinding

### A1. Infrastructure and environment
1. **Original host unreachable (`BLOCKED_SSH` for the first host).** The first SSH host accepted TCP via the local
   TUN proxy but reset every connection before the SSH banner (18:07–18:33 UTC; neighbouring hosts answered, so
   the reset was upstream). No firewall/SSH changes were attempted. At ~18:33 UTC the user supplied a replacement
   host; all remote work ran there. Authentication used the user-provided password once via a transient
   `SSH_ASKPASS` helper reading an environment variable, then an SSH ControlMaster socket. No keys were installed,
   no sshd/firewall configuration was touched, and no credentials are stored in the repository, MLflow or reports.
2. The provided account is `root` (the runbook assumes a normal user). No `sudo`, no system packages, no drivers
   were installed. `uv` 0.7.12 (same version as the lock author) was installed user-level in `~/.local/bin`;
   CPython 3.11.13 via `uv python install`.
3. PyTorch index: driver 595.71.05 (CUDA 13.2 capable) supports the CUDA 12.6 wheels, so the explicit `pytorch-cu126`
   index (Linux only, marker-scoped) was kept; macOS resolves CPU wheels from PyPI for local CPU tests from the same
   lock. Verified: `torch 2.14.1+cu126`, `torch.cuda.is_available() == True`, RTX A4000 15.6 GiB usable.
4. MLflow 3.17.1 server bound to `127.0.0.1:5000` (SQLite `mlruns/mlflow.db`, proxied artifacts
   `mlruns/artifacts/`) in tmux session `mlflow`. Port 5000 verified loopback-only (`ss -ltn`).
5. `scripts/run_stage.sh` initially wrote an empty exit-code file (tmux's `sh -c` expanded `${PIPESTATUS[0]}`).
   Fixed at 18:42 UTC by writing a per-stage runner script. Affected only the `.exit` files of `exp0` and
   `prepare-e2`; their stage status JSON files (authoritative) were unaffected.

### A2. E1 data and models (fixed before training)
6. Each classifier seed gets its **own independent 10k training draw** (streams `train_seed11`, `train_seed29`) —
   a stricter reading of "10,000 training images per seed". Classifier validation 2,000 and held-out classifier test
   5,000 scenes (shared by both seeds). CNN: 5 conv blocks (GroupNorm, SiLU), flattened 4x4 head; AdamW lr 2e-3,
   OneCycle, 40 epochs, batch 128; checkpoint chosen by classifier-validation NLL. No augmentation.
7. Labels are stochastic, `y ~ Bernoulli(sigmoid(3 g_NL))`; accuracy/AUROC are computed against these held-out
   labels (the teacher's own sign accuracy, 0.921, is logged as a Bayes reference — estimated, not asserted).
8. All RNG streams are `SeedSequence(20261018, spawn_key=(k,))` children with fixed named keys (`config.STREAM_KEYS`).
   The 200,000 candidate cap is split 50,000 (validation stream) / 150,000 (test stream).
9. Pair rule (stricter than the minimum): k ∈ {2,3} predictive factors redrawn uniformly; nuisance redrawn with
   probability 1/2; **every** changed factor must move by ≥ 0.15; opposite oracle teacher sign; eta strata
   `>= 0.25` strong / `<= 0.05` weak; first-come quota filling in stream order (50/50 nuisance quota per stratum).
   Result: all quotas filled (validation 1,096 candidates drawn, test 3,489).
10. **Structural confound (noted before unblinding):** `g_NL` has no a–c interaction term, so weak pairs are
    dominated by {a, c} changes and strong pairs involve b. Strong-vs-weak contrasts therefore also differ in which
    factors change.
11. Representation: masked-reconstruction AE (25% random 8x8 patches, MSE), 32-d code, 20,000 unlabeled train-split
    scenes, 40 epochs; no labels, factors, masks or classifier. ψ is standardised with validation-split feature
    statistics (W = I). λ_ψ, λ_D = median squared change per unit latent displacement (δ = 0.01, 512 validation
    scenes). No oracle factor encoder was trained: for ψ(D(z)) = z the pullback is the identity, whose geodesic is
    exactly the latent-linear M2 path (so the "oracle representation" upper bound coincides with M2 + alpha·image).

### A3. Paths, numerics and selection (fixed before validation; profiling used validation pairs only)
12. ε = 1e-3 (normalised units), shared by M3 and M4, fixed a priori; **no ε sensitivity analysis was run**.
13. Optimiser: projected Adam (lr 0.01) on K−1 = 15 internal waypoints, box [0,1]^4, restarts = latent-linear +
    two seeded sin-mode perturbations (scale 0.1); the selected restart is the lowest final energy (geometry only).
    Iteration rule (coded before profiling): smallest of {100, 200, 400} with ≥ 90% of M3/M4 paths changing energy by
    < 1e-3 relative over the last 20% of iterations → **100** (100% converged on the 10 profiling pairs).
14. Quadrature: segment-displacement midpoint rule on the piecewise-linear path, N = 64; uniform doubling to
    128/256/512 for any path with |residual|/(R+1e-8) > 0.005 (same rule for every method); a sample is
    `NUMERIC_INCOMPLETE` if > 0.02 at N = 512.
15. M5: 8 accepted fixed-seed sin-mode (3 modes) perturbations of the latent-linear path, applied to the **changed
    factors only**, inside [0,1]^4, with image-metric energy ratio vs M2 in [1.05, 3.0]; up to 64 draws; attributions
    averaged; accepted/rejected counts and energy ratios recorded.
16. Invalid handling: `MODEL_FLAT` (R ≤ 1e-3) scene-seeds are excluded for all methods alike and counted; invalid M4
    rows are raised to at least the comparator's error; non-finite comparator errors become ties. Validation
    selection replaces non-finite errors by the worst error of that scene-seed (none occurred).
17. Gate A "systematic optimizer divergence": selected final energy non-finite or > (1 + 1e-3) × latent-linear energy;
    the gate fails if > 2% of M3/M4 paths diverge.
18. Bootstrap: 2,000 resamples (seed 9001), scenes resampled with replacement **within** each stratum so the
    equal-weight union stays exactly equal-weight; M4 and B always paired; each scene value = mean over seeds.
19. Verdict mapping when Gate B fails (stricter reading of PREREGISTRATION §7 and `no_go_if_e1_mean_gain_below`):
    `NO_GO` if r_all < 0.05 or r_strong < 0.05 or M4 has no strong-stratum advantage over M2; otherwise `AMBER`.
20. The frozen manifest binds the SHA-256 of the E1 computational modules (`config.E1_CORE_MODULES`); reporting,
    CLI and E2 modules may change after the freeze without affecting E1 numbers.

### A4. Validation outcome (validation data only)
21. Selected comparator **B = M5** (equal-weight validation error 0.007603; M3 0.007790, M2 0.007800, M1 0.008548).
    Selected **M4 alpha = 10** (0.007809; alpha 1: 0.007994; alpha 0.1: 0.008298). Validation already shows all
    methods within a few percent of each other; this does not alter the preregistered plan.

### A5. E2 plan (fixed before any E2 model evaluation)
22. FunnyBirds test-split images are selected model-blind with stream `e2`: removed set R = random 2- or 3-subset of
    parts with ≥ 20 visible part-map pixels; baseline = pre-rendered intervention with R removed; input = full image.
    Disjoint 12 preflight / 20 validation / 120 test images. Only needed zip members are fetched by HTTP range.
23. Target score: smooth contrastive margin `logit_c − logsumexp_{j≠c} logit_j` for the true class c of x.
24. Differentiable path space: compact spatial AE (256²×3 → 32²×8) trained on 5,000 unlabeled FunnyBirds **train**
    images (first 100 per class). All methods explain the identical reconstructed endpoints; completeness is checked
    against reconstructed endpoint scores.
25. Preflight pass rule (a priori, not tuned): ≥ 90% of the 24 preflight endpoints keep the classifier's top-1, and
    ≥ 90% pass a threshold-free part-presence check (part-region SSE to the true render ≤ 0.5 × SSE to the
    toggled-part render, for every visible part).
26. Representation for E2-M4: torchvision ResNet-18 ImageNet weights (layer3), not DINOv3 (gated access; not
    requested). It shares ImageNet pretraining data with the FunnyBirds classifier's initialisation — recorded as a
    limitation of independence.
27. Part-level AP: rank the 5 parts by signed grouped attribution; label relevant iff the part is in R and restoring
    it alone raises the margin by ≥ τ (τ to be locked on validation; 0.5 used only for the exploratory preflight
    smoke); no positives → `UNDEFINED_LABEL`. Full E2 runs only if Gate B passes (sequential stop).
28. MA-GIG was not adapted (`NOT_COMPARABLE_WITHIN_PILOT`): no fair same-endpoint/same-score adaptation was attempted
    within the time budget. No superiority over MA-GIG is claimed.
