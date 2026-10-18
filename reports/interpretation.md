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
  variant. The masked-reconstruction AE's pullback energy differs from the image pullback by about 1% along these
  paths: a reconstruction-trained representation largely inherits pixel geometry.
- **Interaction strength.** The paired difference E_M4 − E_B is unrelated to the oracle interaction index (Spearman
  −0.02 overall, −0.10 within strong; both non-significant). Larger interactions did not produce larger M4 gains.
- **Shortcut / nuisance.** When the nuisance changed, all methods gave it a similar small share (|A_n|/R ≈ 0.019–0.023).
  When it did not change, M4's path wandered slightly in n and assigned it 2.5e-4 R, which is negligible but nonzero.
  The diagnostic shortcut teacher `g_short` showed the same method ordering as the other targets.
- **Oracle targets (diagnostic).** On the exact analytic log-odds and polynomial teachers, latent paths are again within
  about 1% of each other, with M5 or M2 best. M4 does not win on the oracle either, so there is no "M4 loses on the CNN
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
