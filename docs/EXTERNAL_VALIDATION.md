# E2 external validation: feasibility, adapters, and strict caveats

This is the **hardest** stage on a 16 GB A4000. It is included to prevent a strong synthetic-only result from being mistaken for broadly applicable vision XAI. It is *not* permission to train a large diffusion model, replace the counterfactual baseline without disclosure, or report an attractive saliency-map figure as quantitative replication.

## Candidate dataset: FunnyBirds

- Paper: https://openaccess.thecvf.com/content/ICCV2023/html/Hesse_FunnyBirds_A_Synthetic_Vision_Dataset_for_a_Part-Based_Analysis_of_ICCV_2023_paper.html
- Full code and renderer: https://github.com/visinf/funnybirds
- Simpler evaluation framework: https://github.com/visinf/funnybirds-framework

The official framework reports semantically meaningful object-part interventions and includes pretrained **ResNet-50, VGG16 and ViT** model checkpoints. Its README lists a ~1.6 GB dataset and an explainer interface with `explain()`, `get_part_importance()` and `get_important_parts()`. Its default image interface uses 256×256 RGB inputs with part maps. The agent should inspect current source and pin a commit, rather than assume every published API example is unchanged.

Before downloading large files: inspect repo licenses, data license, download terms, disk space, SHA/checkpoint provenance, and the ability to run the selected pretrained checkpoint under the uv environment. Do not combine incompatible legacy dependencies into the primary uv lockfile without testing; if required, write a slim adapter around the minimal model/dataset loader.

## E2 question

Among paths that share the same two counterfactual endpoints and frozen classifier margin, does an independently defined learned-perceptual/semantic-pullback geodesic rank **causally influential object parts** more accurately than decoded latent-linear and image-pullback geodesic paths?

The external evaluation should use **model-response-defined part importance**, such as the change in a fixed margin under each externally rendered part intervention. Attribution regions are defined by original, fixed part masks. A high AP means part rankings agree with this external intervention relevance; it is not proof of a unique feature decomposition.

## 1. Feasibility preflight (12 examples)

### 1.1 Model and data

1. Download only the data/checkpoints necessary for the chosen small subset. Verify classifier forward pass and accuracy plausibility with frozen weights.
2. Identify available part interventions and whether the renderer can create **matched, multi-part** counterfactual pairs. Save IDs and before/after part sets.
3. Pin image normalization and target score definition (e.g. fixed `logit_y1-logit_y0` for an explicitly selected class pair). Do not choose the class target after seeing attribution maps.

### 1.2 Differentiable path representation

The official Blender-based part renderer is **not differentiable with PyTorch autograd**. For IG/geodesic computation, construct a separately validated differentiable decoder `D` (e.g., a compact pretrained or lightly trained autoencoder/VAE) and encode both original and counterfactual endpoints. Reconstruct endpoints `D(E(b))` and `D(E(x))` before taking paths. The primary comparison is on **identical reconstructions**; if output images differ across compared methods, it is not a path-controlled test.

A lightweight AE can be trained using the *permitted training split* and paired augmentations, but must fit within E2's GPU budget. If the AE/VAEs do not preserve parts and margins, **fail preflight**. Non-differentiable part renders are acceptable only to create and independently score the endpoints/hybrid interventions, not for differentiating the geodesic solver.

Do **not** silently replace the decoder with `D(z)=b+z(x-b)` and claim to be following a high-density natural-image manifold. Such a construction is useful as a negative control but does not establish realistic manifold geometry. A partwise alpha-blending generator is likewise a diagnostic, not external manifold validation.

### 1.3 Quantitative preflight

Predeclare a differentiable decoder quality check before computing test explanations. On the 12 preflight examples:

- At least **90%** of endpoint reconstructions preserve the frozen classifier's original top-1 prediction.
- A fixed automated part-preservation measure (segmentation/part detector if available; otherwise a prespecified generic image fidelity proxy and a recorded limitation) must pass for **90%** of endpoints at its frozen validation threshold.
- The median endpoint margin distortion and reconstruction error must be reported even if thresholds pass.
- Both baseline and input reconstructions must be part-plausible for inclusion in the validated external comparison.
- Test images must not be visually selected by looking at method saliency maps.

The agent may select a numeric reconstruction-quality cutoff **only using preflight/validation** and must persist it before evaluating held-out E2 results. If no independent automatic part-preservation check can be built, call the preflight `E2_INCONCLUSIVE`; visual inspection alone is insufficient to claim Gate C.

## 2. Paths and fair comparators

Compute:

- E2-M2: decoded linear path in latent `z`.
- E2-M3: pullback geodesic using only decoder image metric.
- E2-M4: hybrid semantic-pullback geodesic using a frozen learned representation (DINOv3 small backbone if available and licensed, or a documented frozen alternative).
- E2-M1: pixel-linear IG as context/negative control, not the sole primary comparator.

**MA-GIG** code: https://github.com/leekwoon/ma-gig . Treat as an optional contemporaneous method: adapt if model/score/endpoints can be kept matched and within budget; if not, document `NOT_COMPARABLE_WITHIN_PILOT` and do not make an outperforming-MA-GIG claim. `Manifold-Integrated-Gradients` (https://github.com/eszaher/Manifold-Integrated-Gradients) is informative but its repository notes a 32 GB V100 configuration; do not assume the implementation runs unchanged on A4000 16 GB.

DINOv3 source: https://github.com/facebookresearch/dinov3 . Its model weights may require explicit license acceptance and access approval; do not bypass gated access or embed weight URLs/secrets. Prefer the smallest validated backbone; all metric derivatives must remain differentiable wrt input despite frozen model parameters.

Group **pixel-coordinate** attributions by the part maps for E2. A general AE latent dimension is not aligned with object part labels; do not aggregate latent-coordinate IG and call it pixel-part attribution. For arbitrary decoders, calculate image input gradients along decoded paths and then sum signed pixel attribution within each part mask, with a separate background bucket. Preserve both positive and negative channels.

## 3. External evaluation

Use 20 **validation** examples to lock target classes, representation layer, metric normalization, `alpha`, solver iterations, AP convention (treatment of zero/negative part relevance), and baseline comparator. Reserve 120 **held-out** valid pairs under fixed sampling rules, requiring at least two independently manipulated parts when doing the multi-part contrastive comparison. If only single-part deletions can be produced, report built-in FunnyBirds scores as **secondary** and declare the primary multi-part path benefit unverified.

Define part relevance using independently measured score changes for each rendered part intervention, preferably a fixed logit-margin difference rather than predicted-label correctness. Use a **prespecified absolute magnitude threshold** (selected solely on validation) to define relevant parts for AP, or a continuous-rank metric with an explicitly documented label generation rule. Compute per-scene AP, with undefined AP (no positive parts) counted and reported as `UNDEFINED_LABEL`, not silently omitted.

Compute paired AP difference `AP_M4 - AP_B` where `B` is the strongest **same-endpoint** validation-selected comparator. A Gate C positive result requires **>=0.03 absolute AP** and a paired scene-bootstrap 95% confidence lower bound **>0**. Capture preflight and evaluation failure rates separately.

Also report endpoint model-score differences before/after AE reconstruction, reconstruction and part-fidelity distribution, attribution completeness against **reconstructed** endpoint-score differences, CUDA memory and runtime, and whether a meaningful subset had realistic intermediate paths.

## 4. Stop rules and interpretation

- If E1 fails Gate B: implement E2 and perform a small smoke/preflight as feasible, but do **not** launch a full E2 study. Record `NOT_RUN_BY_DESIGN` and final primary NO-GO/AMBER from E1.
- If no valid differentiable decoder can reconstruct the endpoints within **10 E2 GPU-hours**, stop and report `E2_INCONCLUSIVE` / resource infeasibility. It is not a negative scientific estimate of geodesic efficacy.
- If full E2 is valid and its gain is below 3pp or CI includes zero: no external confirmation; no full-project GO.
- If E1 and valid full E2 both pass: proceed with the **GO** decision only after numerical, leakage, failure and run-provenance audits.

## 5. Honest fallback if E2 cannot be executed

Document a feasibility failure using tables of attempted decoder, number of examples, reconstruction/prediction mismatch, peak memory and elapsed GPU hours. If useful, still supply part-level IG saliency diagnostics under identical baselines, labeled *exploratory*. Never convert a weak, unpaired or proxy-only comparison into a positive external validation claim.
