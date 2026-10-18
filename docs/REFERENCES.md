# Research and implementation references

The agent may consult these resources but should not import an entire research repository or claim numerical parity without adapting models, endpoints, and scoring functions. Check license and current revision before using code or weights. These links are starting references, not a guarantee of external API stability.

## Attribution theory and competing methods

1. Sundararajan, Taly & Yan (2017), **Axiomatic Attribution for Deep Networks** (Integrated Gradients). Paper: https://proceedings.mlr.press/v70/sundararajan17a.html
2. Zaher et al. (ICML 2024), **Manifold Integrated Gradients: Riemannian Geometry for Feature Attribution**. Paper: https://proceedings.mlr.press/v235/zaher24a.html ; code: https://github.com/eszaher/Manifold-Integrated-Gradients . Its published code environment describes a 32 GB V100, so do not assume it fits on an A4000 unchanged.
3. Salek & Enguehard (2025 preprint), **Using the Path of Least Resistance to Explain Deep Networks**. https://arxiv.org/abs/2502.12108 . Inspect theorem assumptions critically; do not assume minimizing a model-induced length implies no signed-feature cancellation.
4. Kim et al. (ICML 2026), **Manifold-Aligned Guided Integrated Gradients for Reliable Feature Attribution**. Paper: https://proceedings.mlr.press/v306/kim26aw.html ; code: https://github.com/leekwoon/ma-gig . Relevant same-family comparator, but adaptation of endpoints, models and GPUs is required.
5. IG² (TPAMI 2024), **Integrated Gradient on Iterative Gradient Path for Feature Attribution**. Code: https://github.com/JoeZhuo-ZY/IG2 . This paper already addresses joint path/baseline adaptation; don't claim joint optimization is categorically new.
6. Lundberg & Lee (2017), **A Unified Approach to Interpreting Model Predictions** (SHAP). https://papers.neurips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions . In E1, exact 4-factor Shapley is easy to enumerate and should be reported as a reference.
7. Janizek, Sturmfels & Lee (2021), **Explaining Explanations: Axiomatic Feature Interactions for Deep Networks** (Integrated Hessians). https://jmlr.org/papers/v22/20-1223.html . Useful for discussing path dependence and interactions.

## Datasets and representations

8. Hesse, Schaub-Meyer & Roth (ICCV 2023), **FunnyBirds: A Synthetic Vision Dataset for a Part-Based Analysis of Explainable AI Methods**. Paper: https://openaccess.thecvf.com/content/ICCV2023/html/Hesse_FunnyBirds_A_Synthetic_Vision_Dataset_for_a_Part-Based_Analysis_of_ICCV_2023_paper.html ; full repo: https://github.com/visinf/funnybirds ; lean evaluation framework: https://github.com/visinf/funnybirds-framework . The framework provides pretrained classifier checkpoints and structured part evaluations.
9. Siméoni et al. (2025), **DINOv3**. Repo: https://github.com/facebookresearch/dinov3 ; license: https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md . Model access can be gated by license acceptance. Use compact backbones on 16 GB GPU and document provenance.

## Engineering and runtime

10. Astral uv, **Using uv with PyTorch**. https://docs.astral.sh/uv/guides/integration/pytorch/ . Explicit indexes for reproducible project lockfiles; auto backend selection currently documented for uv's `pip` interface.
11. MLflow, **Tracking Server**. https://mlflow.org/docs/latest/tracking/server/
12. MLflow, **Self-hosted Tracking and artifact proxy**. https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/
13. MLflow, **Security Configuration**. https://mlflow.org/docs/latest/ml/tracking/server/security/configuration/ . Prefer loopback binding + SSH tunnel rather than exposing unauthenticated UI.

## Optional higher-cost direction (out of pilot scope)

14. Li et al. (ICCV 2023), **Your Diffusion Model is Secretly a Zero-Shot Classifier**. https://openaccess.thecvf.com/content/ICCV2023/html/Li_Your_Diffusion_Model_is_Secretly_a_Zero-Shot_Classifier_ICCV_2023_paper.html . A possible *later* generative contrastive scoring test; **not** a requirement for E0–E2, and a denoising-based class score is not automatically the exact log-likelihood ratio.
