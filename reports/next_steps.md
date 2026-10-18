**Recommendation: stop the semantic-geodesic IG branch (NO_GO).** The likely bottleneck is metric inadequacy rather
than baseline choice or manifold membership. All latent methods (M2–M5) stay on the exact renderer manifold and share
endpoints, yet the learned-representation metric produced paths almost identical to the image-pullback geodesic.
Neither geodesic beat a cheap average of random smooth paths. If the question is revisited, it should be under a
**new, separately preregistered** protocol that (i) checks before any test data that the coalition-error lower bound
leaves room for the target effect, (ii) uses a representation whose pullback measurably differs from the image metric
(validated on validation data), and (iii) includes multi-path averaging (M5-type) as a mandatory strong baseline.
Under the current protocol, E2 was implemented and smoke-tested only; its budget was not spent, by design.
