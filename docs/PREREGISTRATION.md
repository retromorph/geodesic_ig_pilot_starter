# Preregistration: outcomes, statistical decisions, and stopping

**Version:** v1.0, prepared before experiments. `configs/protocol.v1.yaml` is the companion machine-readable source of thresholds. Hash both this file and the YAML before any held-out evaluations. This is a feasibility pilot, not a confirmatory proof that a new method is universally better.

## 1. Primary claim to test

> Under fixed input, counterfactual baseline and learned classifier logit margin, an independently learned semantic-representation geodesic (M4) produces **lower counterfactual coalition score-reconstruction error** than the strongest validation-selected same-endpoint path comparator.

H1: the advantage is larger in high-interaction scenes; H2: it is not explained solely by being on the generator manifold; H3: the effect persists with a non-oracle representation and replicates on independently evaluated part-based images.

Null/alternative comparisons must be predeclared. Do not treat a large number of ablations as independent confirmation.

## 2. Target populations and splits

- E0: deterministic analytic controls; `c in {0,1,4}` and additive score.
- E1: train a CNN with seeds **11 and 29** using 10,000 renderer-generated training images per seed, plus separate held-out classification validation/test sets. Training scenes and labeled data are disjoint from all attribution validation/test pair scene IDs.
- E1: **80 validation pairs**, **120 strong-interaction test pairs**, **120 weak-interaction test pairs**. A factor-pair manifest contains immutable baseline/input latent vectors, background-change flag, teacher scores, interaction index, and scene ID. The same 240 test pairs are evaluated on both CNN seeds.
- Generate candidate pairs via a single fixed RNG seed `20261018`, maximum **200,000** candidate pairs. Require different teacher-side binary classes, at least two changed predictive factors, and a nontrivial factor change (`abs(delta)>=0.15` in >=2 predictive factors). Classifier output must **not** be used to select or reroll pairs.
- Strong if the *oracle* interaction index `eta>=0.25`; weak if `eta<=0.05`; middle pairs are omitted from the preregistered primary contrast. `eta` and factor-construction details in `RESEARCH_SPEC.md`. Record selection and rejection counts.
- E2: target **120 held-out valid external comparisons**, separate **20 tuning/validation** candidates and **12 decoder/path preflight** candidates, with disjoint IDs wherever dataset permits. External data sampling is locked before method evaluation. If fewer valid pairs exist, mark external gate inconclusive rather than repeatedly searching for favorable examples.

## 3. Trained-model competence / numerical gate

Before E1 attributions, require for each CNN seed:

1. Binary held-out **AUROC >= 0.90**, accuracy **>= 0.83**, and a logged class-balance diagnostic (these are feasibility floors, not scientific endpoints).
2. At least 98% of preregistered test pairs must have `R > 1e-3` for the learned logit-margin factor-coalition game. Report any flat-scored pairs as a count and missingness reason; do not replace them.
3. No train, encoder-pretraining, validation or test overlap by scene ID. Check programmatically.

The expectation for the teacher with label probability `sigmoid(3*g_NL)` is Bayes accuracy around 0.92 under roughly uniform factors, but actual rates depend on the implemented, calibrated generator. Never assert Bayes optimality without estimating it from the fixed teacher distribution.

**Numeric (Gate A):** E0 analytic errors below `1e-3`; additive-score path-invariance error below `1e-3`; median normalized completeness error below `0.005` and 95th percentile below `0.02` for E1 methods, with normalization `R+1e-8`; at least 98% paths stay within permitted latent factor bounds; and no unresolved systematic optimizer divergence. Log raw absolute errors and all denominator values as well.

Use 64 integration steps initially; refine uniformly by doubling to 128/256/512 as needed to resolve numeric failures. Predefine refinement based solely on completeness error, not explanation quality. On invalid samples, preserve an `invalid_reason` row; report invalid rates by method. Do **not** delete failures before calculating the main metric. For the gate, invalid methods are penalized as failures unless the error is confirmed infrastructure corruption and a fully documented replay is possible.

## 4. Main metric and comparator selection

Compute factor-coalition error for every valid test pair and seed:

\[
 E(A)=\frac1{2^m}\sum_{S\subseteq[m]}
 \left(\frac{f(D(z_S))-f(D(z_b))-\sum_{j\in S}A_j}{R+10^{-8}}\right)^2.
\]

`m` is the set of factors changed between the input/baseline, including nuisance if changed. `R = max_S f(D(z_S)) - min_S f(D(z_S))`. Do not evaluate on a subset of coalitions selected after seeing scores.

**Primary comparator B:** choose exactly one of **M1 pixel IG, M2 latent-linear IG, M3 image-pullback geodesic, M5 random smooth latent paths** using the minimum equal-stratum-weighted mean validation error. Freeze its name/configuration. M4 semantic geodesic is tuned on validation; M4 is not allowed to inspect test errors. Report results against **all** M1–M5 as secondary, but the primary gate uses the frozen B selection.

For test scene `s`, average `E` for each method across the two classifier seeds. Then compute:

\[
 r_U=1-\frac{\operatorname{mean}_{s\in U} E_{M4,s}}
                   {\operatorname{mean}_{s\in U} E_{B,s}},
\]

where `U` is `strong`, `weak`, or the equal-weight union. Positive is better. Avoid per-example ratios, which explode when comparator error is near zero. If the comparator mean error is `<1e-6`, declare the primary comparison **uninformative** and report both absolute errors; do not manufacture a large relative effect.

Compute **2,000 paired bootstrap resamples of scene IDs** (same resample for M4/B; keep classifier seeds paired within scene), with a fixed statistical seed `9001`. Bootstrap percentiles define the 95% confidence interval. If `R` or method validity differs across methods, report the affected paired count and use conservative failure handling. The primary interpretation requires all methods on comparable scene populations.

## 5. Gate B: pilot result from E1

**All conditions required for a positive Gate B:**

| Criterion | Threshold |
|---|---|
| Strong-interaction mean error reduction | `r_strong >= 0.15` |
| Evidence beyond noise | Lower 95% paired-bootstrap bound for `r_strong` **> 0.05** |
| Full 240-scene test reduction | `r_all >= 0.10` |
| Full test statistical evidence | Lower 95% paired-bootstrap bound for `r_all` **> 0** |
| Weak-interaction harm | `r_weak >= -0.05` |
| Seed robustness | M4 better than B on strong scenes for **each** of seeds 11 and 29 |
| Independent representation | Gains must use a learned **unlabeled** encoder, not factor/oracle labels |
| Competence and validity | Both CNNs and Gate A pass; no unexplained selection/failure imbalance |

These are **go/no-go engineering thresholds**, not established scientific effect sizes. They intentionally require a substantial, reproducible advantage to justify pursuing a costly method.

Report both per-seed and scene-averaged statistics. Also evaluate exact analytic oracle targets, including polynomial control, as secondary results. If M4 loses on the exact oracle but wins on CNN, mark a research concern; do not replace CNN as the preregistered primary target.

## 6. Gate C: external validity

Full E2 runs only when Gate B is positive and the E2 preflight passes. The agent must implement E2 support in any case and run at least the preflight/smoke path. Otherwise log `NOT_RUN_BY_DESIGN`.

Use only **same reconstructed endpoints** for geodesic versus latent-linear comparisons. Need a differentiable decoder/path space verified by reconstruction, endpoint-score and part-preservation diagnostics. A nondifferentiable Blender renderer does not itself support Riemannian-gradient path optimization.

- **E2 preflight target:** >=90% same classifier top-1 label between each original/reconstructed endpoint, no missing critical object parts according to a fixed part-detection/segmentation check, and >=90% of preflight endpoint images pass the threshold selected using the 12 preflight examples and locked before E2 test. If the part detector is unavailable, use a fixed image fidelity proxy and explicitly declare the limitation; no discretionary visual-only selection.
- **E2 result target:** M4 improves part-level AP by at least **0.03 absolute** (three percentage points) relative to the strongest validation-selected **matched-endpoint** competitor; lower bound of a paired 95% scene-bootstrap interval for the AP *difference* > 0. The primary AP definition must be explicitly coded, checked against the benchmark's available part importance ground truth, and locked using validation only.
- **If a valid external benchmark with >=2 independently editable parts cannot be obtained:** report `E2_INCONCLUSIVE` and at most `AMBER`, even if E1 succeeds. Do not report GO based only on a custom renderer.

No new generator or diffusion model should be trained at high cost solely to make E2 appear to work under a hard 16 GB / 40 GPU-hour pilot cap.

## 7. Overall decision table

| Status | Interpretation | Action |
|---|---|---|
| **GO** | Gate A + B + C all pass, with documented compute/failure rates | Worth full research investment |
| **AMBER** | E1 has a positive but below-threshold or uncertain effect, or Gate B passes but independent E2 cannot substantiate it | Do not commit; one separately preregistered replication could be justified |
| **NO_GO** | Under valid evaluation, error reduction `<0.05`, consistent harm or no advantage over latent-linear, gains only oracle/procedural leakage, or E1 hard gate B failure without a compelling quantified ambiguity | Stop geodesic-method branch |
| **BLOCKED** | Missing SSH/CUDA/data/permissions, invalid implementation or benchmark that cannot be repaired inside budget | No scientific conclusion; state blocker and next remediation |

If Gate A numerical correctness fails, return `BLOCKED`, not scientific `NO_GO`. If E1 passes but E2 fails to improve under valid conditions, report `NO_GO` for **full-project investment** and characterize the synthetic positive result; if E2 cannot be validly run, report `AMBER`, not a false external negative.

## 8. Compute and stopping rules

Total ceiling **40 GPU-hours** (time on one A4000 or summed device-time if hardware differs). Stage planning allocation: E1 up to **28 GPU-hours**, E2 up to **10**, engineering smoke/profiling up to **2**. E0 is CPU. No more than 16 GB GPU memory. Log wall-time, device count, peak memory and GPU-hour accounting for all runs.

At the 10-sample validation-only profiling checkpoint, choose a feasible fixed numeric configuration (waypoints, optimizer iterations, batch size, quadrature refinements) and freeze it for final testing. Reducing batch size to avoid OOM is permissible because it does not change the mathematical experiment; reducing integration accuracy, changing metric weights, dropping difficult scenes, or silently switching models is not.

When the planned E2 study would exceed 10 GPU-hours, prioritize the predefined 12-example preflight and label the full E2 **resource-infeasible** rather than running a biased small subset. A resource-infeasible result is not evidence of scientific superiority.

## 9. Safeguards against post-hoc adaptation

- Create an immutable `manifest.json` recording protocol SHA256, source commit, generator version, all sample IDs and hashes, target/encoder model checkpoints and hashes, comparator selection and validation scores.
- Test metrics stay blinded until the chosen settings are written to the manifest. A readable script that enforces this is preferable to relying on operator discipline.
- Log all methods including failures, not just the best restart. Optimizer restart choice can use **path energy only**, not test coalition fidelity.
- Any correction after test unblinding is a protocol deviation. If a genuine code bug is found, document it, preserve old results, rerun the complete relevant test population, and keep both versions in MLflow.
- Do not claim statistical significance based on one tuned test split; state that this is a go/no-go pilot and multiple decisions were made during development.

## 10. Required evidence for later review

`reports/FINAL_REPORT.md` must contain: protocol hash; exact Git/uv.lock hashes; GPU and driver; full E0 table; model competence; frozen baselines; per-stage sample counts; E1 error table with means, confidence intervals, seed breakdown and interaction strata; E2 feasibility/results; failure/exclusion counts; runtime; decision JSON; MLflow experiment/run IDs; at least one strong negative case; reproducibility commands; and an explicit statement of what **cannot** be concluded.
