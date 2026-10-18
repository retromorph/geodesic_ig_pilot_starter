# Research specification — geometry, attribution, and experiments

The objective is a **go/no-go study of the path choice**, not a production-quality XAI library or a claim that every semantic path is faithful. This document fixes the mathematical contract; numerical thresholds are in `PREREGISTRATION.md` and `configs/protocol.v1.yaml`.

## A. Core mathematical definitions

For a fixed differentiable contrastive score `f: X -> R`, input `x`, baseline `b`, and absolutely continuous path `x(t)` from `b` to `x`, define signed coordinate attribution

\[
 A_i[x(\cdot)] = \int_0^1 \partial_i f(x(t))\, \dot x_i(t)\,dt.
\]

For nonoverlapping pixel regions `S_j`, define `A_{S_j}=sum_{i in S_j} A_i`. Completeness is

\[
\sum_i A_i=f(x)-f(b)
\]

if the path connects the actual endpoints, the function is appropriately regular, and integration is exact. Numerical integration must report an error against the exact endpoint-score difference.

If `x(t)=D(z(t))` where a differentiable renderer/decoder `D:Z -> X` uses disjoint, factor-specific supports, group attribution can be computed efficiently as

\[
 A_j=\int_0^1 \partial_{z_j}(f\circ D)(z(t))\,\dot z_j(t)\,dt.
\]

It equals the pixel-group attribution only when `z_j` affects **exactly** the region attributed to factor `j`, no other factor affects that region, and no untracked global postprocessing couples pixels. Write tests for this. If a future decoder does not have that property, report pixel-space grouped attributions instead.

**Path is the controlled variable.** For comparisons M1–M5, the same `b`, `x`, exact model checkpoint and score must be used. Changing endpoints or the explained score invalidates the path-isolation claim.

For a given latent decoder `D` and frozen representation `psi`, define the proposed hybrid pullback metric

\[
 G_{\rm sem}(z) = J_{\psi\circ D}(z)^T W J_{\psi\circ D}(z)
 + \alpha J_D(z)^T J_D(z)+\varepsilon I.
\]

Take `W=I` initially after validation-set normalization of feature scales. `alpha>=0` and `epsilon>0` are chosen on validation only. Minimize

\[
 \mathcal E_G[z] =\tfrac12\int_0^1 \dot z(t)^T G(z(t))\dot z(t)\,dt
\]

at fixed endpoints. For discretized waypoints, the first two pullback energies may be approximated using finite differences in `psi(D(z))` and `D(z)`; do not build dense Jacobian matrices. Optimize only `K-1` internal latent waypoints. Preserve endpoints exactly. Require at least a latent-linear initialization and independent smooth restarts. Log energy, optimized length, constraint violations and solver failure.

This is a geometric length/energy criterion, **not a theorem of attribution optimality**. A DINO or SSL pullback can be singular without regularization; a representation may be invariant to features a particular classifier uses. This is why shortcut and nuisance controls are required.

### Interaction dependence

If `f(x)=sum_i h_i(x_i)` is additive, attribution to each coordinate is path-independent (at fixed endpoints). For smooth interactions, differences between attributions from two paths can be related to mixed derivatives of `f` integrated over a surface joining the paths. Therefore, if geodesics help, improvement plausibly depends on *both* model interactions and path geometry. Test this rather than assuming it.

## B. E0: exactly solvable geometry test

Latent `z=(u,v)` with `0<=u,v<=1` and renderer

\[
 D_c(u,v)=(u,v,cu^2),\quad c\in\{0,1,4\}.
\]

Use baseline `D_c(0,0)`, input `D_c(1,1)`, and score `f(x)=x_1 x_2`. For image-pullback geodesics,

\[
 G_c(u,v)={\rm diag}(1+4c^2u^2,1).
\]

Let `s_c(u)=integral_0^u sqrt(1+4c² r²) dr`, `L_c=s_c(1)`. A minimizing geodesic in transformed coordinates has `s_c(u(t))=tL_c`, `v(t)=t`. This gives

\[
 A_u=\frac1{L_c}\int_0^1 s_c(u)du,\qquad A_v=1-A_u.
\]

Expected analytic values (rounded; numerical tolerance is separate):

| c | `A_u` | `A_v` |
|---|---:|---:|
| 0 | 0.500000 | 0.500000 |
| 1 | 0.426373 | 0.573627 |
| 4 | 0.352099 | 0.647901 |

For the latent-linear path `u=v=t`, use `(0.5,0.5)` for all `c`. For a representation `psi(x)=(x_1,x_2)` with the semantic pullback only (or a negligible stated isotropic regularizer), the minimum-energy path is also linear. Quantify how changing **irrelevant** curvature transfers importance from one interacting factor to another.

Add controls:

1. `f_add(x)=x_1+x_2` must give `A=(1,1,0)` for **all** paths; if using factor attributions, `(1,1)`.
2. Random smooth endpoint-preserving paths must satisfy completeness, but may redistribute interaction credit.
3. Analytical gradient comparisons (autograd vs finite differences).
4. Confirm energy of the analytic geodesic is no higher than latent-linear under the relevant metric (numerical tolerance).

E0 is a correctness and failure-mode test. It cannot establish that semantic geodesics are superior on real classifiers.

## C. E1: controlled visual data

### C1. Renderer

Implement a deterministic differentiable `D:[0,1]^4 -> [0,1]^(3x64x64)` where `z=(a,b,c,n)` controls four **disjoint fixed image regions** (soft masks with no overlap):

- `a`: smooth morphological/color attribute in part A;
- `b`: different smooth attribute in part B;
- `c`: different smooth attribute in part C;
- `n`: background appearance nuisance, strictly disjoint from the three part regions.

Each factor should induce a nonlinear, nonconstant image-space metric. Use antialiased smooth shapes, deterministic color/shape parametrizations, and no postprocessing (such as global contrast normalization) that mixes factor supports. Save masks and factor vectors. Test `dD_i/dz_j=0` outside factor `j`'s support for several random points. If the chosen implementation cannot guarantee this, alter the renderer before collecting data rather than claiming equivalent factor/pixel attribution incorrectly.

### C2. Analytical teacher

Use two oracle target functions:

Polynomial control:
\[
 g_{\rm poly}(a,b,c)=2(a-.5)+2(b-.5)+1.5(c-.5)
  +4(a-.5)(b-.5)+3(b-.5)(c-.5).
\]

Primary nonlinear target:
\[
 g_{\rm NL}(a,b,c)=1.5a+1.2b+0.8c
   +2\tanh(3(ab-.25))+1.5\sin(\pi bc)-C.
\]

Choose `C` as the **median** of the raw nonlinear expression on a fixed 20,000-vector, separately seeded, **train-only** calibration draw, then save `C` in the manifest. Do not change `C` to improve attribution. The nuisance factor `n` appears in neither target.

For the nonlinear task, use `P(Y=1|z)=sigmoid(3*g_NL(z))` to generate stochastic binary labels, or sample deterministic labels from an independently generated Bernoulli draw using this probability. This makes `3*g_NL` the exact posterior log-odds and, modulo a class-prior constant, the log-likelihood ratio. Also store the deterministic sign of `g_NL` for sampling paired class-contrast inputs. Do not confuse this analytic teacher with the learned CNN logit margin.

**Classifier:** train a lightweight CNN on rendered samples, two independent seeds (`11`, `29`), each with a fixed 10k train set and held-out validation/test scenes. E1 primary score is `f_CNN(x)=logit_1(x)-logit_0(x)`. Do not select test examples by the model's attribution performance. Report classifier accuracy, AUROC, class balance, and error against teacher probabilities. If classifier fails the preregistered minimum competence gate, do not interpret attribution results as a valid test of explaining the intended target.

**Representation:** train a separate small, unlabeled image encoder on train images only (e.g. self-supervised or reconstruction pretraining). Use no labels, no oracle factor vectors, no test scenes, no classifier gradients, and no factor-specific augmentations exploiting known masks. Freeze before path optimization. Optionally train an **oracle factor encoder** as an explicitly marked upper-bound diagnostic, which cannot satisfy the learned-representation GO gate.

### C3. Paired inputs and stratification

Pairs are generated from distinct seed-scene IDs; a pair consists of two factor vectors with class-contrast according to `sign(g_NL)` and **at least two changed predictive factors**, not a random unrelated baseline. Target 50% of each pair stratum with unchanged `n` and 50% with a nuisance change `abs(Delta n)>=0.15`, using quotas fixed before scoring (report any unattainable quota). Save a fixed manifest of `z_b,z_x,scene_id,teacher_scores,changed_factors` *before* any test attribution.

For the polynomial control and the main nonlinear task, independently construct exact hybrid scenes `z_S` for all `2^m` subsets of the changed factors. Define a fixed interaction-stratification index on the **oracle nonlinear function**, for each pair:

\[
 I_{jk}=g(z_{jk})-g(z_j)-g(z_k)+g(z_\emptyset),\quad
 \eta=\frac{\sum_{j<k}|I_{jk}|}{\max_S g(z_S)-\min_S g(z_S)+10^{-8}}.
\]

Here `z_j` changes only factor `j` relative to `z_b`, `z_jk` changes two, and the sum is over changed **predictive** factors (exclude nuisance). Preassign `strong` if `eta>=0.25` and `weak` if `eta<=0.05`; intermediate cases are not used in the primary strong/weak test. Log counts and rejection causes. Do not adjust thresholds after looking at attribution performance.

Target 80 validation pairs, 120 strong test pairs, 120 weak test pairs; distinct scene IDs across splits. Match pairs between classifier seeds. A fixed seed sequence governs candidate generation; allocate a preregistered maximum candidate search count and report if strata cannot be filled.

### C4. Attribution methods (M1–M5)

- **M1:** pixel-linear path, image-space IG on `x(t)=b+t(x-b)`.
- **M2:** latent-linear path, `x(t)=D(z_b+t(z_x-z_b))`.
- **M3:** image-pullback geodesic with `G=J_D^T J_D+epsilon I`.
- **M4:** proposed learned-semantic pullback geodesic with hybrid metric defined above.
- **M5:** fixed-seed random smooth endpoint-preserving latent paths, adjusted to a preregistered plausible energy/length range where possible; use averaged attributions and report the actual energy/length distribution.

All methods must yield **signed** factor attributions for the same `f_CNN` and paired endpoints. E1 results use 16 waypoints initially, three starts for optimized paths, and 64 midpoint/trapezoidal integration intervals. Validate quadrature/refine adaptively **equally across methods** if completeness fails, without tuning for relative attribution quality. Keep the final computation budget and rules frozen after validation profiling.

The polynomial analytical teacher, exact Shapley attributions, oracle semantic metric, and different classification score variants are **diagnostics**, not opportunities to substitute an easier endpoint or metric in the primary test.

### C5. Evaluation

For each fixed pair and set of changed factors `M` (including nuisance when changed), generate every hybrid `z_S`. For the **learned model**, define

\[
 \Delta_S=f_{CNN}(D(z_S))-f_{CNN}(D(z_b)),\quad
 R=\max_S f_{CNN}(D(z_S))-\min_S f_{CNN}(D(z_S)).
\]

For attribution vector `A` evaluated along an entire baseline-to-input path, predict `\hat\Delta_S=sum_{j in S} A_j` and calculate the **primary** error

\[
 E_{coal}(A)=2^{-|M|}\sum_{S\subseteq M}
 \left(\frac{\Delta_S-\sum_{j\in S}A_j}{R+10^{-8}}\right)^2.
\]

Lower is better. The coalition score evaluations must be computed **independently** of path optimization. Use the same `f_CNN` and fixed hybrid scenes for every method. Report `R`, absolute and relative errors, and the distribution of errors. If `R` is effectively zero, follow the prespecified invalid-sample rule; never secretly drop difficult examples.

Secondary outcomes: per-factor signed attribution, Shapley discrepancy, sign accuracy against independent single-factor effects, nuisance factor attribution on nuisance-varying pairs, an additional analytic shortcut stress target `g_short(z)=g_NL(a,b,c)+3(n-.5)(b-.5)` (diagnostic only), interaction-strength response, path length/energy, expected numerical completeness residual, classification competence, solver failures, GPU-hours, runtime and peak memory. Plot the entire distribution including failures, not only positive examples.

### C6. Causal interpretation caveat

The hybrid intervention `z_S` is a precise operation in the **synthetic structural renderer**. It measures how this particular *trained model* responds to controlled edits and is superior to arbitrary zero masking as a diagnostic. It does not imply unique attribution of non-additive interactions and does not automatically generalize to natural-image causal semantics.

## D. E2: independent part-based validation

Use the FunnyBirds benchmark only if a valid **differentiable path representation** between matched endpoints can be built. Begin from the official evaluation framework and pretrained checkpoint, without retraining a large model. Follow `EXTERNAL_VALIDATION.md` for data, matched pairing, reconstruction preflight, numerical budgets, and fallback/status rules.

E2 should compare M2/M3/M4 against each other at identical reconstructed endpoints, and MA-GIG if it can be adapted fairly within budget. Score by the benchmark's part-based protocol; target a 3 percentage-point absolute part-average-precision gain over the strongest validation-selected matched-endpoint baseline. For E2, the published framework has several different part metrics: explicitly implement/define AP and not silently rename a built-in metric.

A valid external score cannot be claimed when endpoints reconstruct poorly, the method changes the baseline, or a non-differentiable rendering system is incorrectly treated as a differentiable manifold.

## E. Interpretation boundaries

- **Good evidence:** nontrivial, statistically replicated gain for M4 over strong *same-endpoint* baselines on held-out coalition score reconstruction, positive external replication, valid numerical checks and manageable compute.
- **Weak evidence:** better-looking heatmaps, score gain only from a better baseline, only oracle representation, or a gain against pixel IG but not decoded latent-linear IG.
- **Important negative finding:** M2 and M4 have indistinguishable error even in high-interaction scenes, or changing irrelevant metric curvature improves visual saliency while harming coalition fidelity.

Do not frame the pilot as discovering a theorem that geodesics optimize attribution fidelity. This experiment tests a hypothesis about specific models, metrics and representations.
