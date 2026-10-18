# FINAL REPORT — Geodesic Integrated Gradients go/no-go pilot

_Generated mechanically by `geodesic_ig_pilot.cli report` at 2026-10-18T19:16:00Z from saved results (`reports/metrics/aggregate.json`). Hand-written interpretation is confined to the clearly marked section taken from `reports/interpretation.md`._

## Executive decision

- **Final investment verdict:** `NO_GO`
- **Mechanical reasons:** mean relative error reduction below 0.05 (protocol no_go_if_e1_mean_gain_below); no strong-stratum advantage over decoded latent-linear IG (M2); Gate B failed
- **Gate A:** `PASS`
- **Gate B:** `FAIL`
- **Gate C:** `NOT_RUN_BY_DESIGN` — Gate B did not pass: full E2 not run by design (sequential stop); E2 preflight/smoke status E2_INCONCLUSIVE: E2 preflight failed (decoder does not preserve endpoints/parts); E2_INCONCLUSIVE
- **Compute used:** 0.471 measured GPU-hours (wall-clock of GPU stages on one device) / 40 A4000 GPU-hours; peak allocated VRAM 11.52 GB; OOM events 0
- **Strictly preregistered hypothesis supported?** no

### Interpretation (hand-written after aggregation; does not alter any gate)

**Bottom line: NO_GO for the semantic-geodesic IG branch.** On the 240 frozen E1 test scenes, M4 (learned-representation
hybrid pullback geodesic) had *higher* counterfactual coalition error than the validation-selected comparator
B = M5 in every stratum, for both classifier seeds, with paired scene-bootstrap 95% intervals that exclude zero on the
harmful side (r_strong −0.043 [−0.075, −0.015]; r_weak −0.108 [−0.170, −0.050]; r_all −0.053 [−0.082, −0.028]).
Every numerical check passed (Gate A), both classifiers were competent (AUROC 0.975/0.976, accuracy 0.919/0.915
against a teacher Bayes reference of 0.921), and there were no invalid, flat or out-of-domain samples, so this is a
valid negative result, not an implementation artefact.

Answers to the template's mechanism questions (secondary analyses, no gate):

- **Does M4 beat M2, or only M1?** Only pixel IG on strong pairs (r_strong vs M1 +0.081 [0.047, 0.114]; vs M1 on weak
  pairs it is worse, −0.105). Against decoded latent-linear IG the strong-stratum difference is indistinguishable from
  zero (−0.012 [−0.035, 0.009]); M4 wins 57.5% of strong scenes against M2 but loses on average because of a heavy
  tail. Per the preregistration, a gain over pixel IG alone is not evidence for geodesic superiority.
- **Is any gain unique to geodesic optimisation?** No. The best method overall was the *random* smooth-path average
  (M5), which has ~1.9x the image-metric energy of the geodesics. Its advantage is plausibly from averaging several
  paths (it also has the smallest Shapley discrepancy, 0.061), not from geometry.
- **M4 versus M3.** With the validation-selected alpha = 10 the hybrid metric is dominated by the image term, and
  M4 is practically identical to the image-pullback geodesic (r_all vs M3 −0.003 [−0.0047, −0.0005]; median latent
  length 0.859 for both). On validation the representation-dominated setting (alpha = 0.1) was the *worst* M4
  variant. The M4 geodesic lowers the learned-representation energy by only ~1% relative to the latent-linear path
  (median 0.348 vs 0.351), and the image geodesic M3 lowers it almost as much (0.349). A reconstruction-trained
  representation largely inherits pixel geometry.
- **Interaction strength.** The paired difference E_M4 − E_B is unrelated to the oracle interaction index (Spearman
  −0.02 overall, −0.10 within strong; both non-significant). Larger interactions did not produce larger M4 gains.
- **Shortcut / nuisance.** When the nuisance changed, all methods gave it a similar small share (|A_n|/R ≈ 0.019–0.023).
  When it did not change, M4's path wandered slightly in n and assigned it 2.5e-4 R, which is negligible but nonzero.
  The diagnostic shortcut teacher `g_short` showed the same method ordering as the other targets.
- **Oracle targets (diagnostic).** On the exact analytic log-odds, polynomial and shortcut teachers, latent paths are
  again close: within ~2% of each other on strong pairs, and up to ~9% on the much smaller weak-pair errors. M5 or M2
  is best on strong pairs, and M3 on weak oracle-log-odds pairs. M4 does not win on the oracle either, so there is no "M4 loses on the CNN
  but wins on the oracle" concern.
- **Why so little separation? (EXPLORATORY, post-unblinding.)** The coalition-error metric is dominated by
  non-additivity that no complete additive attribution can remove. The best complete least-squares attribution (which
  here coincides with exact Shapley, E = 0.0120 strong) removes only 13.5% of M5's strong-stratum error and 16% of
  M2's. A ≥15% reduction against M5 was therefore unattainable for *any* complete path method on this population.
  This is a design lesson for the metric and threshold, not a rescue: M4 did not merely fall short of the bar; it was
  worse than the comparator. In this renderer the image metric is diagonal and separable, so geodesics are per-factor
  monotone reparametrisations of the linear path; they reshuffle interaction credit only slightly.

**What cannot be concluded.** This is a single synthetic renderer, one CNN architecture (two seeds), one learned
representation family (a masked AE) and one metric family. The result says nothing about natural images, diffusion or
likelihood-based scores, DINO-type representations, or E2 (not run by design). Coalition fidelity measures
faithfulness to this *trained model's* responses to exact factor edits. It is neither human semantic plausibility nor
causal necessity, and the Shapley reference is an allocation convention, not ground truth.

**What would overturn this NO_GO in an independent replication:** a preregistered study where a representation whose
pullback differs materially from the decoder's image metric yields M4 coalition error at least 15% below the best of
M2 and M5 on strong-interaction pairs (95% lower bound > 0.05), on a population where such a reduction is attainable
(verified by the complete-additive lower bound) and replicated on an external part-based benchmark.

## 1. Immutable provenance

| Item | Value | Verification |
|---|---|---|
| Protocol SHA-256 | `521ce0a919ea58bac42c2c4f2c369bd6842e039bce047378d5c7d7afcd0b9017` | `sha256sum configs/protocol.v1.yaml` |
| Preregistration SHA-256 | `73e0665bc1d831984935921780c2223c1dd79b423be01dab5ff31be8ff73927b` | `sha256sum docs/PREREGISTRATION.md` |
| Git commit / dirty (tracked files) | `76474c5f7fef54a66f22270ca9210486098b1786` / True | `git rev-parse HEAD` |
| uv.lock SHA-256 | `44a2e38747550f63fcb378b0ae08800bc92c6c768946e0333d28dfd04ad580ae` | `sha256sum uv.lock` |
| E1 source SHA-256 (frozen modules) | `0a390006a0bc088d921377532fd3a9a2991adb6abd361118f1c91fbc223aee76` | `config.source_tree_sha256()` |
| Python / torch / CUDA (wheel) / driver | 3.11.13 / 2.14.1+cu126 / 12.6 / 595.71.05 | doctor report |
| GPU / peak VRAM | NVIDIA RTX A4000 / 11.52 GB | gpu_ledger.jsonl |
| Checkpoints SHA-256 | seed 11: `61f6c8e19b795798…`, seed 29: `a4613068d9008388…`; encoder `ececefa1fe292b00…` | `checkpoints/*.json` |
| Pair manifests (content SHA-256) | {'test': '4f178e4c0ab96069e48d8e8d2f3c0b86afa189ec4aaa141d56311edff8a5ab90', 'validation': 'de65c9931a54ad29b2cfbc60cc3ff9479f766350b1acf230d02849e9daeca3c2'} | prepare_manifest.json |
| Frozen manifest SHA-256 / frozen at | `3a7fa2d2fc26bbaa01744f62e79b5155b0fe979c087d54ec51951cde09e1f085` / 2026-10-18T18:42:29Z | `reports/manifest/frozen_manifest.json` (read-only) |
| Unblinding timestamp | 2026-10-18T18:44:41Z | `reports/manifest/unblinding.json` |
| MLflow experiments | {'gig-pilot/decisions': '6', 'gig-pilot/decisions-smoke': '10', 'gig-pilot/e0': '2', 'gig-pilot/e1-data': '3', 'gig-pilot/e1-data-smoke': '7', 'gig-pilot/e1-models': '4', 'gig-pilot/e1-models-smoke': '8', 'gig-pilot/e1-paths': '5', 'gig-pilot/e1-paths-smoke': '9', 'gig-pilot/e2': '11', 'gig-pilot/smoke': '1'} | tracking server via SSH tunnel |

## 2. Research question and methods

**Path-isolation claim.** For every held-out pair the five methods explain the *same* frozen CNN score
`f_CNN(x) = logit_1(x) − logit_0(x)` between the *same* rendered baseline `D(z_b)` and input `D(z_x)`; only the
integration path differs. Both classifier seeds (11, 29) are evaluated on the same 240 test pairs.

**Renderer.** `D: [0,1]^4 → [0,1]^{3×64×64}` partitions pixels into four hard regions: part tiles A (disk radius
3+8a), B (soft square, nonlinear colour curve in b), C (striped disk, frequency 1+2c+2c²) and the background
(nonlinear colour in n). Each region depends on exactly one factor, so `J_D^T J_D` is diagonal (tested exactly) and
latent-factor attributions equal pixel-group attributions (tested exactly with jvp displacements; to ≤3e-3 at
magnitude 12 with finite pixel displacements at N=512). Every factor's image metric is non-constant (tested).

**Methods** (implementation: `src/geodesic_ig_pilot/e1.py::process_chunk`).
- **M1 pixel IG:** straight line in pixel space between the two rendered endpoints; per-pixel segment-displacement
  midpoint rule; sums within the four region masks.
- **M2 latent-linear IG:** `x(t) = D(z_b + t(z_x − z_b))`; attributions `A_j = Σ_i ∂_j(f∘D)(z(m_i)) Δz_j,i`.
- **M3 image-pullback geodesic:** waypoints minimising `(K/2)Σ_k ||D(z_{k+1}) − D(z_k)||²/λ_D + ε||Δz||²`.
- **M4 hybrid semantic geodesic (proposed):** minimises `(K/2)Σ_k [||ψ̂(z_{k+1}) − ψ̂(z_k)||²/λ_ψ + α||D(z_{k+1}) −
  D(z_k)||²/λ_D + ε||Δz||²]`, ψ̂ = standardised code of a masked-reconstruction AE trained on unlabeled train-split
  images only (no labels, factors, masks or classifier gradients); α chosen on validation from {0.1, 1, 10}.
- **M5 random smooth paths:** mean attribution over 8 fixed-seed, endpoint-preserving sin-mode perturbations of the
  latent-linear path (changed factors only), constrained to [0,1]^4 and to an image-energy ratio band [1.05, 3.0].

Geodesics use K = 16 segments, projected Adam (lr 0.01) on internal waypoints, three restarts (linear + 2 smooth
perturbations) selected by final **energy only**; endpoints are copied exactly and verified bit-for-bit. Paths never
see classifier gradients or coalition scores.

**Primary metric.** For the changed-factor set M (|M| ∈ {2,3,4}, nuisance included when changed), every coalition
`z_S` is rendered independently and scored by the frozen CNN; `E = 2^{−|M|} Σ_S ((Δ_S − Σ_{j∈S} A_j)/(R+1e-8))²`.
Exact Shapley values of the same game are the reference allocation (not a competitor).

**Selection and blinding.** Comparator B (one of M1/M2/M3/M5) and M4's α were selected by the minimum
equal-stratum-weighted validation error over 80 validation pairs, then written with all hashes (protocol,
preregistration, uv.lock, E1 source, data files, checkpoints, test manifest) into a read-only
`reports/manifest/frozen_manifest.json` (also logged to MLflow). The test stage refuses to start if any hash differs
and writes an immutable unblinding timestamp before computing any test attribution.

**Frozen comparator B:** `M5`; **frozen M4 alpha:** `10.0` (rule: min equal-stratum-weighted validation mean coalition error, validation run `229a9bdd026842909931270105572f0a`).

Validation equal-stratum-weighted mean coalition errors (selection inputs):

| Method | strong | weak | equal-weight |
|---|---:|---:|---:|
| M5 | 0.01282 | 0.002383 | 0.007603 |
| M3 | 0.01302 | 0.002559 | 0.00779 |
| M2 | 0.01314 | 0.002461 | 0.0078 |
| M4_a10 | 0.01305 | 0.002569 | 0.007809 |
| M4_a1 | 0.01329 | 0.0027 | 0.007994 |
| M4_a0.1 | 0.01367 | 0.002922 | 0.008298 |
| M1 | 0.01452 | 0.002573 | 0.008548 |

Frozen numerics: K=16 segments, 100 Adam iterations (lr 0.01), 3 energy-only restarts, quadrature [64, 128, 256, 512] with uniform doubling while |residual|/(R+1e-8) > 0.005, invalid if > 0.02 at 512, epsilon 0.001, M5 8 candidates in energy band [1.05, 3.0].
Profiling trials (validation only): [{'fraction_converged': 1.0, 'iterations': 100, 'seconds': {'0.1': 2.397331476211548, '1.0': 2.1543986797332764, '10.0': 2.1533913612365723, 'None': 1.7192225456237793}}]; pixel/latent cross-check: {'curved': {'max_abs_attr': 11.947652816772461, 'max_abs_latent_vs_pixel_finite512': 0.002994537353515625, 'max_abs_latent_vs_pixel_jvp': 1.621246337890625e-05}, 'linear': {'max_abs_attr': 12.211226463317871, 'max_abs_latent_vs_pixel_finite512': 0.001354217529296875, 'max_abs_latent_vs_pixel_jvp': 2.86102294921875e-06}}

## 3. E0 — analytic results and negative controls

| c | expected A_u | analytic A_u | optimizer A_u | abs err | optimizer A_v | latent-linear A_u | semantic-psi A_u | additive max err | completeness err | discrete E / analytic geodesic E / linear E |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.500000 | 0.500000 | 0.500000 | 0.00e+00 | 0.500000 | 0.500000 | 0.500000 | 0.0e+00 | 0.0e+00 | 1.00000 / 1.00000 / 1.00000 |
| 1 | 0.426373 | 0.426373 | 0.426387 | 1.38e-05 | 0.573613 | 0.500000 | 0.500000 | 0.0e+00 | 1.1e-16 | 1.59365 / 1.59364 / 1.66667 |
| 4 | 0.352099 | 0.352099 | 0.352039 | 5.98e-05 | 0.647961 | 0.500000 | 0.500000 | 1.1e-16 | 0.0e+00 | 9.34351 / 9.33958 / 11.66667 |

Hybrid metric psi+alpha*image (curvature transfer illustration), A_u: c=0: alpha_0.1=0.5000, alpha_1.0=0.5000, alpha_10.0=0.5000; c=1: alpha_0.1=0.4866, alpha_1.0=0.4499, alpha_10.0=0.4297; c=4: alpha_0.1=0.4131, alpha_1.0=0.3636, alpha_10.0=0.3532

E0 checks passed: 36/36. Settings: {'iterations': 4000, 'lr': 0.002, 'quadrature': 1024, 'segments': 128}. Random smooth endpoint-preserving paths keep completeness but redistribute interaction credit, e.g. c=1 A_u over 5 random paths: [0.5756, 0.6748, 0.4331, 0.3552, 0.4063].

## 4. E1 — population and model competence

| Attribute | Count / value |
|---|---:|
| Scene pools | {'classifier_test': 5000, 'classifier_val': 2000, 'encoder_train': 20000, 'encoder_val': 2000, 'train_seed11': 10000, 'train_seed29': 10000} |
| Calibrated C (median raw g_NL, 20k train-only draw) | 2.239475 |
| Candidates drawn (validation) | 1096 (cap 50000) |
| Filled quotas (validation) | {'strong|nuisance_changed=False': 20, 'strong|nuisance_changed=True': 20, 'weak|nuisance_changed=False': 20, 'weak|nuisance_changed=True': 20} |
| Rejections (validation) | {'middle_eta': 106, 'quota_full': 34, 'same_teacher_class': 195, 'small_delta': 681} |
| Candidates drawn (test) | 3489 (cap 150000) |
| Filled quotas (test) | {'strong|nuisance_changed=False': 60, 'strong|nuisance_changed=True': 60, 'weak|nuisance_changed=False': 60, 'weak|nuisance_changed=True': 60} |
| Rejections (test) | {'middle_eta': 296, 'quota_full': 150, 'same_teacher_class': 638, 'small_delta': 2165} |
| Leakage check (scene IDs and exact factor vectors, all pools) | no_leakage=True |
| Classifier seed 11: AUROC / accuracy / ECE / NLL (5k held-out scenes) | 0.9751 / 0.9188 / 0.0086 / 0.2001 (competent=True) |
| Seed 11 class balance: label pos rate / predicted pos rate | 0.502 / 0.498 |
| Seed 11 teacher reference: sign(g_NL) accuracy / AUROC; MAE p vs teacher; corr(margin, 3g) | 0.9210 / 0.9769; 0.0205; 0.9893 |
| Classifier seed 29: AUROC / accuracy / ECE / NLL (5k held-out scenes) | 0.9758 / 0.9152 / 0.0110 / 0.1977 (competent=True) |
| Seed 29 class balance: label pos rate / predicted pos rate | 0.502 / 0.495 |
| Seed 29 teacher reference: sign(g_NL) accuracy / AUROC; MAE p vs teacher; corr(margin, 3g) | 0.9210 / 0.9769; 0.0223; 0.9914 |
| Learned encoder objective / oracle? | masked reconstruction (25% random 8x8 patches, MSE), unlabeled train-split images / False |
| Encoder diagnostics | {'effective_rank_participation': 5.390288127101134, 'feature_dim': 32, 'n_near_constant': 0}; lambda_psi=198.3, lambda_D=920.7 |

## 5. E1 — primary result

Scene-averaged (two classifier seeds) mean normalised coalition error on the 240 frozen test scenes; all methods share identical (baseline, input, checkpoint, score) per row.

| Method | E_coal strong ↓ | E_coal weak ↓ | E_coal all (equal-weight) ↓ | abs. err strong | Completeness p95 | Invalid % | Median sec / pair (path + 2 seeds) | Shapley discrepancy | sign acc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 | 0.01578 | 0.002416 | 0.009097 | 3.762 | 8.037e-05 | 0 | 0.007706 | 0.07396 | 0.9417 |
| M2 | 0.01432 | 0.002755 | 0.008539 | 3.274 | 0.0001587 | 0 | 0.007205 | 0.06745 | 0.9431 |
| M3 | 0.01446 | 0.002654 | 0.008558 | 3.362 | 0.000253 | 0 | 0.05262 | 0.06622 | 0.9417 |
| M4 | 0.01449 | 0.002669 | 0.008581 | 3.369 | 0.0002397 | 0 | 0.2045 | 0.06646 | 0.9417 |
| M5 | 0.0139 | 0.002408 | 0.008153 | 3.165 | 0.0004266 | 0 | 0.06549 | 0.06116 | 0.9477 |

**Frozen comparator B:** `M5` (validation-selected); M4 alpha `10.0`.

| Gate B quantity | Measured | 95% paired scene-bootstrap CI | Requirement | Pass? |
|---|---:|---|---|---|
| Strong relative reduction r_strong | -0.04288 | [-0.07488, -0.01517] | >=0.15, lower bound >0.05 | no / no |
| Full relative reduction r_all | -0.05256 | [-0.08211, -0.02776] | >=0.10, lower bound >0 | no / no |
| Weak relative reduction r_weak | -0.1085 | [-0.1696, -0.05] | >=-0.05 | no |
| Seed 11 strong reduction | -0.04633 | n/a | >0 | no |
| Seed 29 strong reduction | -0.03657 | n/a | >0 | no |
| learned_non_oracle_encoder | yes | n/a | true | yes |
| classifiers_competent | yes | n/a | true | yes |
| nonflat_fraction>=0.98 | yes | n/a | true | yes |
| no_scene_leakage | yes | n/a | true | yes |
| gate_a_pass | yes | n/a | true | yes |
| test_population_complete | yes | n/a | true | yes |

Absolute means: strong M4 0.01449 vs B 0.0139; weak M4 0.002669 vs B 0.002408; all M4 0.008581 vs B 0.008153. Comparator mean error negligible (<1e-6)? no. Pair counts: {'b_invalid': 0, 'm4_invalid': 0, 'model_flat': 0, 'scene_seeds': 480}; model-flat fraction 0.

Figures: `reports/figures/e1_paired_differences.png`, `reports/figures/e1_difference_vs_eta.png`, `reports/figures/e1_numeric_diagnostics.png`, `reports/figures/e1_examples_predefined.png` (examples chosen by a fixed rule: strong scene with the largest and the smallest seed-averaged E_B - E_M4).

## 6. E1 — controls and mechanism diagnosis (secondary; no gate)

M4 versus every other method (same paired scene bootstrap; secondary, not the preregistered gate):

| vs | r_strong [95% CI] | r_weak [95% CI] | r_all [95% CI] | r_strong seed 11 | r_strong seed 29 |
|---|---|---|---|---:|---:|
| M1 | 0.08142 [0.04746, 0.1137] | -0.1049 [-0.2426, 0.0359] | 0.05668 [0.02367, 0.08975] | 0.07378 | 0.0952 |
| M2 | -0.01197 [-0.03502, 0.009443] | 0.03129 [0.003715, 0.06046] | -0.004988 [-0.02435, 0.0137] | -0.01913 | 0.0009996 |
| M3 | -0.002167 [-0.004147, 0.0001607] | -0.005633 [-0.01273, 0.001133] | -0.002705 [-0.004693, -0.0004705] | -0.002109 | -0.002275 |
| M5 | -0.04288 [-0.07488, -0.01517] | -0.1085 [-0.1696, -0.05] | -0.05256 [-0.08211, -0.02776] | -0.04633 | -0.03657 |

Oracle analytic targets along the same latent paths (diagnostic only; M1 undefined in latent space):

| target | method | strong | weak |
|---|---|---:|---:|
| oracle_nl | M2 | 0.01376 | 0.0002513 |
| oracle_nl | M3 | 0.01389 | 0.0002303 |
| oracle_nl | M4 | 0.01389 | 0.0002342 |
| oracle_nl | M5 | 0.01361 | 0.0002368 |
| oracle_poly | M2 | 0.03655 | 0.003395 |
| oracle_poly | M3 | 0.03692 | 0.003429 |
| oracle_poly | M4 | 0.03691 | 0.003437 |
| oracle_poly | M5 | 0.03694 | 0.003475 |
| oracle_short | M2 | 0.01482 | 0.000573 |
| oracle_short | M3 | 0.01496 | 0.0005586 |
| oracle_short | M4 | 0.01496 | 0.0005604 |
| oracle_short | M5 | 0.01475 | 0.0005671 |

Mean |A_n|/R (nuisance attribution) by whether the nuisance changed:

| method | nuisance unchanged | nuisance changed |
|---|---:|---:|
| M1 | 0 | 0.02336 |
| M2 | 0 | 0.02053 |
| M3 | 9.655e-07 | 0.02039 |
| M4 | 0.0002455 | 0.02038 |
| M5 | 0 | 0.01872 |

Path diagnostics: {'energy_by_method': {'E_img': {'M2': 0.36390088498592377, 'M3': 0.35392889380455017, 'M4': 0.35401827096939087, 'M5': 0.6866868436336517}, 'E_sem': {'M2': 0.3507540822029114, 'M3': 0.3492164760828018, 'M4': 0.34769099950790405, 'M5': 0.6498373746871948}, 'latent_length': {'M2': 0.8497908711433411, 'M3': 0.8590087294578552, 'M4': 0.8588259816169739, 'M5': 1.0574721693992615}}, 'm3_discrete_over_exact': {'25%': 0.9914226769257787, '50%': 0.9946724715658497, '75%': 0.996429447042544, 'count': 240.0, 'max': 0.9997731822283015, 'mean': 0.9939549547914373, 'min': 0.9814714758688896, 'std': 0.0036453006235133735}, 'm5_accepted': {'25%': 8.0, '50%': 8.0, '75%': 8.0, 'count': 240.0, 'max': 8.0, 'mean': 7.95, 'min': 4.0, 'std': 0.3955822994631275}, 'opt_rel_tail_change': {'25%': {'M3': 1.3229532530844254e-05, 'M4': 1.6224956612024704e-05}, '50%': {'M3': 2.1083363811544883e-05, 'M4': 2.6502112699915628e-05}, '75%': {'M3': 3.6977242881896306e-05, 'M4': 4.433787626543779e-05}, 'count': {'M3': 240.0, 'M4': 240.0}, 'max': {'M3': 0.0002875655693026001, 'M4': 0.0005000588955794661}, 'mean': {'M3': 3.163781498347703e-05, 'M4': 3.9404874762366897e-05}, 'min': {'M3': 3.7107327171592937e-06, 'M4': 4.65608958316358e-06}, 'std': {'M3': 3.4667028816076734e-05, 'M4': 4.832543578938361e-05}}}


**EXPLORATORY (post-unblinding; not a gate input).** Lower bounds on the coalition error of any additive attribution for the same CNN games (scene-seed means):

| quantity | strong | weak |
|---|---:|---:|
| best additive A (LS_complete) | 0.01203 | 0.001312 |
| best additive A (LS_unconstrained) | 0.007392 | 0.0008241 |
| exact Shapley | 0.01203 | 0.001312 |
| M1 | 0.01578 | 0.002416 |
| M2 | 0.01432 | 0.002755 |
| M3 | 0.01446 | 0.002654 |
| M4 | 0.01449 | 0.002669 |
| M5 | 0.0139 | 0.002408 |

Fraction of each method's error that is reducible at all by a complete additive attribution (1 − bound/method): {'M1': {'strong': 0.2378246287335059, 'weak': 0.4568793153626105}, 'M2': {'strong': 0.1603392330482899, 'weak': 0.5238261076721319}, 'M3': {'strong': 0.16846968144989702, 'weak': 0.505676172583451}, 'M4': {'strong': 0.17026793386833672, 'weak': 0.508445103173619}, 'M5': {'strong': 0.13468832437222344, 'weak': 0.45513191624854304}}

## 7. E2 — independent validation

- **Gate C:** `NOT_RUN_BY_DESIGN` — Gate B did not pass: full E2 not run by design (sequential stop); E2 preflight/smoke status E2_INCONCLUSIVE: E2 preflight failed (decoder does not preserve endpoints/parts); E2_INCONCLUSIVE
- **Dataset / code:** FunnyBirds (visinf/funnybirds-framework @ `1350bbe812ff`, visinf/funnybirds @ `91b4b4628ffa`); code license Apache-2.0 (both repositories); data license: none published; repository Apache-2.0 (inference, recorded limitation). Zip members read selectively by HTTP range (126887 entries indexed, 969 MB fetched).
- **Classifier:** official FunnyBirds ResNet-50 (`resnet50_final_0_checkpoint_best.pth.tar`, SHA-256 `88f0fcf517ab5aa3…`, ETag "64ccc879-b47b87d"); top-1 on the 152 selected original test images = 1.000. Target: logit_c − logsumexp_(j≠c) logit_j.
- **Frozen representation for E2-M4:** torchvision ResNet-18 IMAGENET1K_V1 layer3 (SHA-256 `f37072fd47e89c5e…`). DINOv3 was not used (gated access).
- **Locked selection (model-blind, stream `e2`):** 12 preflight / 20 validation / 120 held-out test images, disjoint IDs; every pair removes 2–3 visible parts (120/120 held-out pairs are multi-part); all 2^|R| coalition renders are pre-rendered interventions.
- **Decoder:** compact spatial AE 256²×3 → 32²×8, 5000 unlabeled train-split images, 30 epochs (MSE + 0.5 L1 reconstruction, unlabeled train subset; SHA-256 `00d6d895a7772c87…`).

| Preflight criterion (24 endpoints of 12 pairs) | Measured | Requirement | Pass? |
|---|---:|---|---|
| Top-1 preserved after reconstruction | 0.750 | ≥ 0.90 | no |
| Part-presence fidelity (all visible parts) | 0.500 | ≥ 0.90 | no |
| Pairs with both endpoints part-faithful | 0.417 | reported | |
| Median endpoint reconstruction MSE | 5.03e-04 | reported | |
| Median / max abs. target-margin distortion (logits) | 0.972 / 7.290 | reported | |
| Contrast sign preserved (f(x)−f(b)) / median abs. contrast change | 0.917 / 2.526 | reported | |

Part-fidelity rule (prespecified): reconstruction part-region SSE <= 0.5 x SSE to the toggled-part render (prespecified, not tuned). **Preflight pass: no** → E2 status `E2_INCONCLUSIVE`; preflight GPU-hours 0.374, peak 11.5 GB.

Exploratory path smoke on 4 preflight pairs (identical reconstructed endpoints; **not** a result, the decoder failed preflight; AP uses tau = 0.5 only for this smoke; quadrature capped at N = 256):

| scene | removed | method | AP | completeness rel. | N | sec |
|---|---|---|---:|---:|---:|---:|
| fb-000441 | eye,foot,tail | M1 | 0.75 | 0.0096 | 256 | 1.5 |
| fb-000441 | eye,foot,tail | M2 | 1 | 0.0066 | 256 | 1.5 |
| fb-000441 | eye,foot,tail | M3 | 1 | 0.0016 | 64 | 20.9 |
| fb-000441 | eye,foot,tail | M4 | 1 | 0.0006 | 64 | 48.2 |
| fb-000134 | beak,eye,foot | M1 | n/a | 0.0008 | 64 | 0.2 |
| fb-000134 | beak,eye,foot | M2 | n/a | 0.0111 | 256 | 1.5 |
| fb-000134 | beak,eye,foot | M3 | n/a | 0.0878 | 256 | 22.2 |
| fb-000134 | beak,eye,foot | M4 | n/a | 0.0355 | 256 | 49.5 |
| fb-000082 | foot,tail | M1 | 1 | 0.0011 | 256 | 1.5 |
| fb-000082 | foot,tail | M2 | 1 | 0.0027 | 256 | 1.5 |
| fb-000082 | foot,tail | M3 | 1 | 0.0042 | 256 | 22.2 |
| fb-000082 | foot,tail | M4 | 1 | 0.0018 | 256 | 49.5 |
| fb-000191 | eye,foot | M1 | 0.75 | 0.0019 | 64 | 0.2 |
| fb-000191 | eye,foot | M2 | 0.75 | 0.0012 | 128 | 0.7 |
| fb-000191 | eye,foot | M3 | 0.75 | 0.0019 | 256 | 22.2 |
| fb-000191 | eye,foot | M4 | 0.75 | 0.0016 | 64 | 48.2 |

AP definition (coded in `external.part_ap`, to be locked on validation for a full run): rank the 5 parts by signed grouped pixel attribution; a part is relevant iff it was removed in the baseline and restoring it alone raises the target margin by ≥ tau on the *rendered* intervention; no relevant part → `UNDEFINED_LABEL`. MA-GIG: `NOT_COMPARABLE_WITHIN_PILOT` (not adapted). No AP values are claimed as E2 results.

## 8. Error analysis and validity threats

- **Failures and exclusions:** none. Zero invalid rows, zero `MODEL_FLAT` scene-seeds, zero out-of-domain paths,
  zero solver divergences, all 240 test scenes × 2 seeds × 5 methods present (2,400 rows). No pair was rerolled or
  dropped; quotas were filled first-come from the fixed streams.
- **Optimizer convergence / energy suboptimality:** M3/M4 relative energy change over the last 20 iterations has a
  median of ~2–3e-5 (max 5e-4). For M3, where the exact geodesic is known (separable metric), the discrete optimum
  reaches 0.981–1.000 (median 0.995) of the exact continuous energy. The ratio is slightly below 1 because chord
  energies underestimate arc energies at K = 16. The optimizer is not the bottleneck.
- **Numerical completeness:** relative residual p95 ≤ 4.3e-4 for every method (limits: median 0.005, p95 0.02); no
  path needed refinement beyond N = 64. Latent-vs-pixel chain-rule agreement is 1.6e-5 at attribution magnitude ~12
  (float32).
- **Encoder collapse / nuisance sensitivity:** 0 near-constant features out of 32, participation-ratio rank 5.4
  (≈ the 4 generative factors). The AE encodes the background nuisance like any other factor; nothing stops an
  unlabeled reconstruction objective from doing so.
- **Out-of-support intermediates:** latent paths stay inside [0,1]^4 by construction (projection), so every
  intermediate image is an exact renderer output. M1's pixel path leaves the renderer manifold, as intended.
- **Selection bias / unblinding:** comparator and alpha were selected on validation only and frozen with hashes
  before the single test pass (frozen 18:42:29Z, unblinded 18:44:41Z). Some secondary analyses were computed after
  unblinding: the exploratory headroom bound, the eta correlation and win rates. They are labelled exploratory and do
  not affect any gate.
- **Design confounds:** weak pairs are mostly {a, c} changes (`g_NL` has no a–c term), so strata differ in which
  factors change as well as in interaction strength. Coalition error is normalised by each game's range R, so scenes
  with small R weigh relatively more.
- **Functional vs causal:** coalition fidelity measures how well additive attributions reproduce *this trained CNN's*
  responses to exact factor edits. It does not establish semantic plausibility or causal necessity; with
  interactions, no additive attribution is unique.
- **Single run:** one renderer, one encoder family, ε fixed at 1e-3 (no sensitivity analysis run), one test pass.
  This is a go/no-go pilot, not a significance claim.

## 9. Resource accounting and reproducibility

| Stage | GPU-hours | Max VRAM (GB) | Wall-clock (s) | Exit |
|---|---:|---:|---:|---|
| exp0 (cpu) | 0.0000 | 0.00 | 38.3 | ok |
| train-e1-seed11 (cuda) | 0.0106 | 1.12 | 38.0 | ok |
| train-e1-seed29 (cuda) | 0.0098 | 1.12 | 35.1 | ok |
| fit-encoder (cuda) | 0.0296 | 0.37 | 106.6 | ok |
| profile-e1 (cuda) | 0.0048 | 5.64 | 17.4 | ok |
| evaluate-e1-validation (cuda) | 0.0173 | 5.65 | 62.1 | ok |
| evaluate-e1-test (cuda) | 0.0239 | 5.65 | 86.2 | ok |
| evaluate-e2-preflight (cuda) | 0.3743 | 11.52 | 1347.3 | ok |
| e2-preflight-margins (cuda) | 0.0006 | 0.15 | 2.3 | ok |
| **Total** | **0.4709** | 11.52 | | |

Replay commands, hashes and run IDs: `reports/reproducibility.md`.

## 10. Protocol deviations and exploratory observations

See `reports/deviations.md` (all deviations, reruns and the unblinding timeline).

## 11. Verdict and next steps

Mechanical verdict: **NO_GO** — mean relative error reduction below 0.05 (protocol no_go_if_e1_mean_gain_below); no strong-stratum advantage over decoded latent-linear IG (M2); Gate B failed. Machine-readable: `reports/decision.json`.

**Recommendation: stop the semantic-geodesic IG branch (NO_GO).** The likely bottleneck is metric inadequacy rather
than baseline choice or manifold membership. All latent methods (M2–M5) stay on the exact renderer manifold and share
endpoints, yet the learned-representation metric produced paths almost identical to the image-pullback geodesic.
Neither geodesic beat a cheap average of random smooth paths. If the question is revisited, it should be under a
**new, separately preregistered** protocol that (i) checks before any test data that the coalition-error lower bound
leaves room for the target effect, (ii) uses a representation whose pullback measurably differs from the image metric
(validated on validation data), and (iii) includes multi-path averaging (M5-type) as a mandatory strong baseline.
Under the current protocol, E2 was implemented and smoke-tested only; its budget was not spent, by design.

