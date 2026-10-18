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
