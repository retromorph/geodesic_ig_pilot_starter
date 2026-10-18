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
