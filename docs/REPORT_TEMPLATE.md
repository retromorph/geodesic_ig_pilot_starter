# FINAL_REPORT.md — mandatory structure for the coding agent

> **Template only.** Do not fill bracketed fields with estimates or illustrative numbers and present them as measured results. The completed file must be written as `reports/FINAL_REPORT.md` after actual runs. Record `NOT_RUN`, `BLOCKED`, or `UNAVAILABLE` explicitly.

## Executive decision

- **Final investment verdict:** `[GO | AMBER | NO_GO | BLOCKED]`
- **Reason in 3–5 sentences:** `[actual finding, strongest comparator, scope, constraints]`
- **Gate A:** `[PASS | FAIL | BLOCKED]`
- **Gate B:** `[PASS | FAIL | INCONCLUSIVE | NOT_RUN]`
- **Gate C:** `[PASS | FAIL | INCONCLUSIVE | NOT_RUN_BY_DESIGN]`
- **Compute used:** `[actual measured GPU-hours] / 40 A4000 GPU-hours`
- **Strictly preregistered hypothesis supported?** `[yes/no/inconclusive]`

## 1. Immutable provenance

| Item | Value | Verification |
|---|---|---|
| Protocol SHA-256 | `[hash]` | `sha256sum configs/protocol.v1.yaml` |
| Protocol docs SHA-256 | `[hash]` | `sha256sum docs/PREREGISTRATION.md` |
| Git commit / dirty status | `[id/status]` | `git status --short; git rev-parse HEAD` |
| uv lockfile SHA-256 | `[hash]` | `sha256sum uv.lock` |
| Python / torch / CUDA / driver | `[versions]` | captured doctor log |
| GPU model / peak VRAM | `[value]` | doctor and runtime log |
| Dataset/checkpoint SHA-256 | `[hashes]` | manifest |
| MLflow experiment IDs | `[ids]` | tracking server |
| Holdout dataset manifest SHA-256 | `[hash]` | manifest |
| Unblinding timestamp | `[UTC ISO8601]` | signed manifest |

## 2. Research question and methods

Restate the path-isolation claim, exact model score and identical endpoint constraint. Identify M1–M5; explain how the semantic encoder was trained without labels/oracle information. Report how the selected baseline B was frozen on validation. Include method pseudocode or a concise link to its implementation and all deviations.

## 3. E0 — analytic results and negative controls

| Test | Expected | Observed | Absolute error | PASS? |
|---|---:|---:|---:|---|
| c=0, A_u | 0.500000 | `[value]` | `[value]` | `[ ]` |
| c=1, A_u | 0.426373 | `[value]` | `[value]` | `[ ]` |
| c=4, A_u | 0.352099 | `[value]` | `[value]` | `[ ]` |
| Additive path invariance | <=0.001 | `[value]` | `[value]` | `[ ]` |
| Complete signed attribution | equals score difference | `[value]` | `[value]` | `[ ]` |

Add geodesic solver energy optimality diagnostics and explain whether irrelevant metric curvature redistributed importance.

## 4. E1 — population and model competence

| Attribute | Count / value |
|---|---:|
| Synthetic images for training | `[ ]` |
| Train/validation/test unique scene IDs | `[ ]` |
| Candidate pairs sampled | `[ ]` |
| Validation pairs | `[ ]` |
| Strong test pairs | `[ ]` |
| Weak test pairs | `[ ]` |
| Rejected pairs and reasons | `[ ]` |
| Classifier seed 11 AUROC / accuracy | `[ ]` |
| Classifier seed 29 AUROC / accuracy | `[ ]` |
| Learned encoder training objective / checkpoint hash | `[ ]` |

Present data-generation formula, calibrated `C`, fixed seeds, and evidence that class labels and learned representation did not leak from evaluation scenes.

## 5. E1 — primary result

Fill the table from saved per-scene test records, with the **same scene IDs across methods**.

| Method | E_coal strong ↓ | E_coal weak ↓ | E_coal all ↓ | Completeness p95 | Invalid % | Median sec / pair |
|---|---:|---:|---:|---:|---:|---:|
| M1 pixel IG | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| M2 latent-linear IG | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| M3 image geodesic | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| M4 semantic geodesic | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| M5 random paths | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |

**Frozen comparator B:** `[method and validation rationale]`

| Gate B quantity | Measured | 95% paired scene-bootstrap CI | Requirement | Pass? |
|---|---:|---|---|---|
| Strong relative reduction | `[ ]` | `[ ]` | >=15%, lower bound >5% | `[ ]` |
| Full relative reduction | `[ ]` | `[ ]` | >=10%, lower bound >0 | `[ ]` |
| Weak relative reduction | `[ ]` | `[ ]` | >=-5% | `[ ]` |
| Seed 11 strong reduction | `[ ]` | n/a | >0 | `[ ]` |
| Seed 29 strong reduction | `[ ]` | n/a | >0 | `[ ]` |
| Learned non-oracle representation | `[yes/no]` | n/a | yes | `[ ]` |

State whether the absolute error differences are scientifically meaningful even when relative ratios are large. Plot (1) scene-paired differences M4–B, (2) interaction-index vs difference, (3) completeness and invalidity rates, (4) representative favorable and unfavorable examples selected by a **predefined** ordering rule.

## 6. E1 — controls and mechanism diagnosis

Report results using analytic polynomial and nonlinear log-odds teachers (diagnostic only), exact Shapley agreement, nuisance-attribution stress test, oracle representation upper bound, metric weights and alternative random path energy ratios. Explicitly answer:

- Does M4 beat M2, or only M1?
- Is the gain unique to geodesic optimization, or matched by random curved paths?
- Do larger model interactions align with larger improvements?
- Do model shortcut features receive appropriate attribution when they change?
- How dependent are conclusions on the representation encoder and optimizer stability?

## 7. E2 — independent validation

- Execution status: `[COMPLETE | E2_INCONCLUSIVE | NOT_RUN_BY_DESIGN | BLOCKED]`.
- Dataset revision/license/checkpoint; exact number of preflight, validation and held-out examples.
- Decoder method and reconstruction qualification; % endpoints retaining top-1, part preservation score, effect of decoder on classifier margin.
- AP definition and explicit part-label construction; same-endpoint matching audit.
- Result table: M2/M3/M4/(optional MA-GIG) part-AP and paired difference with bootstrap CI, invalids and runtime.
- Gate C assessment: `>=0.03 absolute AP` and 95% lower bound >0, or honest reason it cannot be assessed.
- If not run: explain exact stop condition, cost or validity blocker. Do not show guessed AP values.

## 8. Error analysis and validity threats

Present **all** failures, not just successful paths. Include at least:

- optimizer non-convergence and geodesic energy suboptimality;
- numerical completeness and quadrature error;
- learned encoder collapse / sensitivity to nuisance;
- reconstruction distortions / out-of-support intermediates;
- selection bias, test unblinding/deviations, failed samples;
- difference between functional coalition fidelity and causal semantics;
- dependence on number of changed factors and score-range normalization.

## 9. Resource accounting and reproducibility

| Stage | GPU-hours | Max VRAM | Wall-clock | Status |
|---|---:|---:|---:|---|
| Smoke/profile | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| E0 | `0` | n/a | `[ ]` | `[ ]` |
| E1 training + encoder | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| E1 paths/evaluation | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| E2 preflight + full | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| **Total** | `[ ]` | `[ ]` | `[ ]` | `[ ]` |

List commands to replay all stages, `uv sync --locked`, hashes, MLflow experiment/run URLs reachable over SSH tunnel, and paths to per-scene Parquet and figures. Do not expose sensitive server credentials or network addresses.

## 10. Protocol deviations and exploratory observations

Link `reports/deviations.md`. Distinguish a fully preregistered result from exploratory hyperparameter changes. List every rerun, reason, superseded run ID and population effect.

## 11. Verdict and next steps

Produce a mechanically calculated gate JSON and human-readable verdict. State precisely what empirical result would overturn the conclusion in an independent replication. If `GO`, propose the next *limited* research milestone (e.g., two natural-image datasets with stronger contemporaneous baselines and exact likelihood-model tests). If `NO_GO`, explain whether baseline/manifold membership or geodesic metric inadequacy was the likely bottleneck. If `AMBER`/`BLOCKED`, state which unmeasured condition prevents investment, without making a positive claim.

### Attached machine-readable output

`reports/decision.json` should include:

```json
{
  "protocol_version": "1.0",
  "protocol_sha256": "TO_BE_FILLED_FROM_FILE",
  "gate_a": {"status": "NOT_EVALUATED", "reasons": []},
  "gate_b": {"status": "NOT_EVALUATED", "reasons": []},
  "gate_c": {"status": "NOT_EVALUATED", "reasons": []},
  "verdict": "NOT_EVALUATED",
  "total_gpu_hours": null,
  "mlflow_experiment_ids": [],
  "selected_comparator": null,
  "report_generated_utc": null
}
```

Do not leave `NOT_EVALUATED` in a report that claims completed experiments.
