"""E1 pipeline: data preparation, classifier and encoder training, path attribution evaluation.

Blinding is enforced in code: `evaluate_e1(split="test")` refuses to run unless the frozen
manifest (written after validation-only selection) exists and every hash still matches.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import Paths, Protocol, sha256_file, source_tree_sha256, torch_seed_for
from .datasets import (
    FACTOR_NAMES,
    calibrate_C,
    check_no_leakage,
    generate_pairs,
    make_scenes,
    manifest_sha256,
    pair_arrays,
)
from .evaluations import (
    DENOM_EPS,
    coalition_error,
    hybrid_latents,
    shapley_values,
    single_factor_sign_accuracy,
)
from .geometry import (
    MetricTerm,
    PathOptConfig,
    discrete_energy,
    linear_waypoints,
    optimize_paths,
    path_latent_length,
    segment_sqnorms,
    smooth_perturbation,
)
from .integration import (
    eval_batched,
    latent_path_attribution,
    latent_path_pixel_attribution,
    pixel_path_attribution,
    refine_until_complete,
)
from .models import (
    MaskedAE,
    SmallCNN,
    encoder_diagnostics,
    evaluate_classifier,
    freeze,
    train_autoencoder,
    train_classifier,
)
from .renderer import image_metric_diag, region_masks, render
from .scores import g_nl, g_poly, g_short, logit_margin
from .tracking import (
    ResourceMeter,
    Tracker,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_parquet,
    utcnow,
    write_immutable_json,
    write_status,
)

# Numerical defaults fixed *before* any validation/test evaluation (see reports/deviations.md).
EPSILON = 1e-3  # isotropic metric regulariser in normalised units, shared by M3 and M4
M5_CANDIDATES = 8
M5_MAX_DRAWS = 64
M5_ENERGY_BAND = (1.05, 3.0)  # image-metric energy ratio vs latent-linear path
M5_SCALES = (0.05, 0.1, 0.2)
REFINE_TARGET_REL = 0.005
INVALID_REL = 0.02
ITERATION_CANDIDATES = (100, 200, 400)
CONVERGENCE_REL_TAIL = 1e-3
CONVERGENCE_FRACTION = 0.9
CHUNK_PAIRS = 20
CNN_EPOCHS = 40
AE_EPOCHS = 40
AE_TRAIN_SCENES = 20000
CLF_VAL_SCENES = 2000
CLF_TEST_SCENES = 5000
ENC_VAL_SCENES = 2000

SMOKE = {
    "train": 512,
    "encoder_train": 512,
    "clf_val": 256,
    "clf_test": 512,
    "enc_val": 256,
    "cnn_epochs": 2,
    "ae_epochs": 2,
    "val_quotas": {("strong", True): 1, ("strong", False): 1, ("weak", True): 1, ("weak", False): 1},
    "test_quotas": {("strong", True): 1, ("strong", False): 1, ("weak", True): 1, ("weak", False): 1},
    "profile_pairs": 2,
    "iteration_candidates": (20, 40),
    "chunk_pairs": 2,
}


@dataclass
class Ctx:
    protocol: Protocol
    paths: Paths
    device: str
    tracker: Tracker
    smoke: bool

    def exp(self, name: str) -> str:
        return f"gig-pilot/{name}" + ("-smoke" if self.smoke else "")

    @property
    def gen_seed(self) -> int:
        return int(self.protocol.get("e1", "generator_seed"))


def _require(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"prerequisite missing: {what} ({path}). Run the earlier stage first.")


def _save_torch(obj: Any, path: Path) -> str:
    import io

    buf = io.BytesIO()
    torch.save(obj, buf)
    atomic_write_bytes(path, buf.getvalue())
    return sha256_file(path)


# =========================================================================== prepare-e1
def prepare_e1(ctx: Ctx, resume: bool = False, dry_run: bool = False) -> dict:
    e1 = ctx.protocol.get("e1")
    sizes = {
        "train_seed11": SMOKE["train"] if ctx.smoke else e1["train_samples_per_seed"],
        "train_seed29": SMOKE["train"] if ctx.smoke else e1["train_samples_per_seed"],
        "encoder_train": SMOKE["encoder_train"] if ctx.smoke else AE_TRAIN_SCENES,
        "classifier_val": SMOKE["clf_val"] if ctx.smoke else CLF_VAL_SCENES,
        "classifier_test": SMOKE["clf_test"] if ctx.smoke else CLF_TEST_SCENES,
        "encoder_val": SMOKE["enc_val"] if ctx.smoke else ENC_VAL_SCENES,
    }
    if dry_run:
        print(json.dumps({"stage": "prepare-e1", "scene_pools": sizes, "gpu_hours": 0.0}, indent=2))
        return {}
    out = ctx.paths.data_e1
    manifest_path = out / "prepare_manifest.json"
    if manifest_path.exists():
        if not resume:
            raise RuntimeError(f"{manifest_path} exists; use --resume to verify and reuse it")
        man = json.loads(manifest_path.read_text())
        for fname, h in man["files"].items():
            if sha256_file(out / fname) != h:
                raise RuntimeError(f"hash mismatch for {fname}; data changed after preparation")
        print("prepare-e1: existing manifest verified; skipping")
        return man
    cal = calibrate_C(ctx.protocol)
    C = cal["C"]
    prefixes = {
        "train_seed11": "train11",
        "train_seed29": "train29",
        "encoder_train": "enctrain",
        "classifier_val": "clfval",
        "classifier_test": "clftest",
        "encoder_val": "encval",
    }
    frames = {k: make_scenes(ctx.protocol, k, n, C, prefixes[k]) for k, n in sizes.items()}
    # The unlabeled encoder pool keeps no labels on disk (it never sees them).
    frames["encoder_train"] = frames["encoder_train"].drop(columns=["y", "p_teacher", "g_nl"])
    frames["encoder_val"] = frames["encoder_val"].drop(columns=["y", "p_teacher", "g_nl"])
    cap = e1["max_candidate_pairs"]
    caps = {"validation": cap // 4, "test": cap - cap // 4}
    pairs, pair_info = {}, {}
    for split in ("validation", "test"):
        q = (SMOKE["val_quotas"] if split == "validation" else SMOKE["test_quotas"]) if ctx.smoke else None
        pairs[split], pair_info[split] = generate_pairs(ctx.protocol, split, C, caps[split], quotas=q)
    leakage = check_no_leakage(frames, {f"pairs_{k}": v for k, v in pairs.items()})
    if not leakage["no_leakage"]:
        raise RuntimeError(f"scene leakage detected: {leakage}")
    files = {}
    for name, df in frames.items():
        atomic_write_parquet(out / f"scenes_{name}.parquet", df)
        files[f"scenes_{name}.parquet"] = sha256_file(out / f"scenes_{name}.parquet")
    for split, df in pairs.items():
        atomic_write_parquet(out / f"pairs_{split}.parquet", df)
        files[f"pairs_{split}.parquet"] = sha256_file(out / f"pairs_{split}.parquet")
    man = {
        "created_utc": utcnow(),
        "protocol_sha256": ctx.protocol.sha256,
        "generator_seed": ctx.gen_seed,
        "generator_version": "renderer-v1",
        "calibration": cal,
        "scene_pool_sizes": sizes,
        "pair_generation": pair_info,
        "pair_manifest_content_sha256": {k: manifest_sha256(v) for k, v in pairs.items()},
        "pair_counts": {
            k: v.groupby(["stratum", "nuisance_changed"]).size().to_dict() for k, v in pairs.items()
        },
        "leakage": leakage,
        "files": files,
        "smoke": ctx.smoke,
    }
    man["pair_counts"] = {
        k: {f"{a}|{b}": int(n) for (a, b), n in v.items()} for k, v in man["pair_counts"].items()
    }
    write_immutable_json(manifest_path, man)
    with ctx.tracker.run(
        ctx.exp("e1-data"), "prepare-e1", tags={"stage": "prepare-e1", "split": "all"}
    ) as run:
        ctx.tracker.log_metrics(
            {"C": C, **{f"candidates_{k}": v["candidates_drawn"] for k, v in pair_info.items()}}
        )
        ctx.tracker.log_artifact(manifest_path, "manifest")
        for split in pairs:
            ctx.tracker.log_artifact(out / f"pairs_{split}.parquet", "pairs")
        man_run = run.info.run_id
    all_filled = all(v["all_quotas_filled"] for v in pair_info.values())
    write_status(
        ctx.paths,
        "prepare-e1",
        "COMPLETE" if all_filled else "COMPLETE_WITH_FAILURES",
        mlflow_run_id=man_run,
        manifest=str(manifest_path),
        C=C,
        pair_generation=pair_info,
        smoke=ctx.smoke,
    )
    return man


def _load_prepared(ctx: Ctx) -> dict:
    p = ctx.paths.data_e1 / "prepare_manifest.json"
    _require(p, "prepare-e1 manifest")
    return json.loads(p.read_text())


def _scenes(ctx: Ctx, name: str) -> pd.DataFrame:
    return pd.read_parquet(ctx.paths.data_e1 / f"scenes_{name}.parquet")


def _z(df: pd.DataFrame) -> np.ndarray:
    return df[list(FACTOR_NAMES)].to_numpy(np.float64)


def _require_cuda(ctx: Ctx) -> None:
    if ctx.smoke:
        return
    if not (ctx.device.startswith("cuda") and torch.cuda.is_available()):
        raise RuntimeError(
            "GPU stage requires CUDA; refusing silent CPU fallback (use --smoke for CPU smoke)"
        )


# =========================================================================== train-e1
def train_e1(ctx: Ctx, resume: bool = False, dry_run: bool = False) -> dict:
    seeds = ctx.protocol.get("e1", "classifier_seeds")
    epochs = SMOKE["cnn_epochs"] if ctx.smoke else CNN_EPOCHS
    if dry_run:
        print(
            json.dumps(
                {"stage": "train-e1", "seeds": seeds, "epochs": epochs, "est_gpu_hours": 0.1}, indent=2
            )
        )
        return {}
    _require_cuda(ctx)
    _load_prepared(ctx)
    comp = ctx.protocol.get("e1", "competence")
    val, test = _scenes(ctx, "classifier_val"), _scenes(ctx, "classifier_test")
    results = {}
    for seed in seeds:
        ck = ctx.paths.checkpoints / f"cnn_seed{seed}.pt"
        meta_p = ctx.paths.checkpoints / f"cnn_seed{seed}.json"
        if resume and ck.exists() and meta_p.exists():
            meta = json.loads(meta_p.read_text())
            if sha256_file(ck) == meta["checkpoint_sha256"]:
                print(f"train-e1: seed {seed} checkpoint verified; skipping")
                results[seed] = meta
                continue
        train = _scenes(ctx, f"train_seed{seed}")
        with (
            ctx.tracker.run(
                ctx.exp("e1-models"),
                f"cnn-seed{seed}",
                tags={"stage": "train-e1", "method_id": "cnn", "split": "train", "seed": seed},
                params={"epochs": epochs, "batch_size": 128, "lr": 2e-3, "n_train": len(train)},
            ) as run,
            ResourceMeter(ctx.paths, f"train-e1-seed{seed}", ctx.device) as meter,
        ):
            model, info = train_classifier(
                _z(train),
                train["y"].to_numpy(),
                _z(val),
                val["y"].to_numpy(),
                seed,
                ctx.device,
                epochs,
                log=lambda r: ctx.tracker.log_metrics(r, step=r["epoch"]),
            )
            freeze(model)
            m_test = evaluate_classifier(model, _z(test), test["y"].to_numpy(), ctx.device)
            # Teacher-agreement diagnostics (oracle quantities are diagnostic only).
            from sklearn.metrics import roc_auc_score

            from .models import classifier_logits

            logits = classifier_logits(model, _z(test), ctx.device)
            margin = logits[:, 1] - logits[:, 0]
            teacher_lo = 3 * test["g_nl"].to_numpy()
            p_cnn = 1 / (1 + np.exp(-margin))
            bayes = {
                "teacher_sign_accuracy": float(np.mean((test["g_nl"].to_numpy() > 0) == (test["y"] == 1))),
                "teacher_auroc": float(roc_auc_score(test["y"], teacher_lo)),
                "mae_p_vs_teacher": float(np.mean(np.abs(p_cnn - test["p_teacher"].to_numpy()))),
                "corr_margin_vs_teacher_logodds": float(np.corrcoef(margin, teacher_lo)[0, 1]),
            }
            sha = _save_torch(model.state_dict(), ck)
            competent = m_test["auroc"] >= comp["min_auroc"] and m_test["accuracy"] >= comp["min_accuracy"]
            meta = {
                "seed": seed,
                "checkpoint_sha256": sha,
                "test_metrics": m_test,
                "teacher_reference": bayes,
                "competent": bool(competent),
                "train_info": {k: v for k, v in info.items() if k != "curves"},
                "mlflow_run_id": run.info.run_id,
                "resources": meter.summary(),
            }
            atomic_write_json(meta_p, meta)
            atomic_write_json(ctx.paths.metrics / f"cnn_seed{seed}_curves.json", info["curves"])
            ctx.tracker.log_metrics({f"test_{k}": v for k, v in m_test.items()} | bayes)
            ctx.tracker.set_tags({"model_checkpoint_sha256": sha, "competent": competent})
            ctx.tracker.log_artifact(meta_p, "model")
            ctx.tracker.log_artifact(ctx.paths.metrics / f"cnn_seed{seed}_curves.json", "curves")
        results[seed] = meta
    ok = all(r["competent"] for r in results.values())
    write_status(
        ctx.paths,
        "train-e1",
        "COMPLETE" if ok else "COMPLETE_WITH_FAILURES",
        competence={str(k): v["competent"] for k, v in results.items()},
        test_metrics={str(k): v["test_metrics"] for k, v in results.items()},
        smoke=ctx.smoke,
    )
    return results


# =========================================================================== fit-encoder
def fit_encoder(ctx: Ctx, resume: bool = False, dry_run: bool = False) -> dict:
    epochs = SMOKE["ae_epochs"] if ctx.smoke else AE_EPOCHS
    if dry_run:
        print(
            json.dumps(
                {"stage": "fit-encoder", "epochs": epochs, "objective": "masked_reconstruction"}, indent=2
            )
        )
        return {}
    _require_cuda(ctx)
    _load_prepared(ctx)
    ck = ctx.paths.checkpoints / "encoder_ae.pt"
    meta_p = ctx.paths.checkpoints / "encoder_ae.json"
    if resume and ck.exists() and meta_p.exists():
        meta = json.loads(meta_p.read_text())
        if sha256_file(ck) == meta["checkpoint_sha256"]:
            print("fit-encoder: checkpoint verified; skipping")
            return meta
    tr, va = _scenes(ctx, "encoder_train"), _scenes(ctx, "encoder_val")
    seed = torch_seed_for("encoder_train", ctx.gen_seed)
    with (
        ctx.tracker.run(
            ctx.exp("e1-models"),
            "encoder-masked-ae",
            tags={
                "stage": "fit-encoder",
                "method_id": "learned_masked_ae",
                "split": "train",
                "oracle": False,
            },
            params={"epochs": epochs, "latent": 32, "mask_frac": 0.25, "patch": 8, "n_train": len(tr)},
        ) as run,
        ResourceMeter(ctx.paths, "fit-encoder", ctx.device) as meter,
    ):
        ae, info = train_autoencoder(
            _z(tr),
            _z(va),
            seed,
            ctx.device,
            epochs,
            log=lambda r: ctx.tracker.log_metrics(r, step=r["epoch"]),
        )
        freeze(ae)
        diag = encoder_diagnostics(ae, _z(va), ctx.device)
        mu = torch.tensor(diag["feature_mean"], dtype=torch.float32, device=ctx.device)
        sd = torch.tensor(diag["feature_std"], dtype=torch.float32, device=ctx.device).clamp_min(1e-6)
        # Metric normalisation on validation scenes only: median squared change per unit
        # latent displacement for psi and for pixels.
        rng = np.random.default_rng(torch_seed_for("metric_normalization", ctx.gen_seed))
        idx = rng.choice(len(va), size=min(512, len(va)), replace=False)
        z0 = torch.as_tensor(np.clip(_z(va)[idx], 0.02, 0.98), dtype=torch.float32, device=ctx.device)
        dirs = torch.as_tensor(rng.normal(size=(len(idx), 4)), dtype=torch.float32, device=ctx.device)
        delta = 0.01 * dirs / dirs.norm(dim=1, keepdim=True)
        with torch.no_grad():
            x0, x1 = render(z0), render(z0 + delta)
            p0, p1 = (ae.encode(x0) - mu) / sd, (ae.encode(x1) - mu) / sd
            dn2 = (delta**2).sum(1)
            lam_psi = float(torch.median(((p1 - p0) ** 2).sum(1) / dn2))
            lam_D = float(torch.median(((x1 - x0) ** 2).flatten(1).sum(1) / dn2))
        sha = _save_torch(ae.state_dict(), ck)
        meta = {
            "checkpoint_sha256": sha,
            "objective": "masked reconstruction (25% random 8x8 patches, MSE), unlabeled train-split images",
            "uses_labels": False,
            "uses_factor_vectors_as_targets": False,
            "uses_classifier": False,
            "oracle": False,
            "diagnostics": diag,
            "feature_norm_mean": diag["feature_mean"],
            "feature_norm_std": sd.cpu().tolist(),
            "lambda_psi": lam_psi,
            "lambda_D": lam_D,
            "train_info": {k: v for k, v in info.items() if k != "curves"},
            "mlflow_run_id": run.info.run_id,
            "resources": meter.summary(),
        }
        atomic_write_json(meta_p, meta)
        atomic_write_json(ctx.paths.metrics / "encoder_curves.json", info["curves"])
        ctx.tracker.log_metrics(
            {
                "best_val_mse": info["best_val_mse"],
                "lambda_psi": lam_psi,
                "lambda_D": lam_D,
                "n_near_constant": diag["n_near_constant"],
                "effective_rank": diag["effective_rank_participation"],
            }
        )
        ctx.tracker.set_tags({"representation_checkpoint_sha256": sha})
        ctx.tracker.log_artifact(meta_p, "encoder")
    write_status(
        ctx.paths, "fit-encoder", "COMPLETE", checkpoint_sha256=sha, diagnostics=diag, smoke=ctx.smoke
    )
    return meta


# =========================================================================== evaluation machinery
class E1Models:
    """Frozen classifiers, frozen representation, metric terms and score closures."""

    def __init__(self, ctx: Ctx) -> None:
        dev = ctx.device
        self.device = dev
        self.seeds = list(ctx.protocol.get("e1", "classifier_seeds"))
        self.cnns, self.cnn_sha = {}, {}
        for s in self.seeds:
            ck = ctx.paths.checkpoints / f"cnn_seed{s}.pt"
            _require(ck, f"classifier seed {s}")
            m = SmallCNN().to(dev)
            m.load_state_dict(torch.load(ck, map_location=dev, weights_only=True))
            self.cnns[s] = freeze(m)
            self.cnn_sha[s] = sha256_file(ck)
        ck = ctx.paths.checkpoints / "encoder_ae.pt"
        _require(ck, "encoder")
        self.enc_meta = json.loads((ctx.paths.checkpoints / "encoder_ae.json").read_text())
        ae = MaskedAE().to(dev)
        ae.load_state_dict(torch.load(ck, map_location=dev, weights_only=True))
        self.ae = freeze(ae)
        self.enc_sha = sha256_file(ck)
        self.mu = torch.tensor(self.enc_meta["feature_norm_mean"], dtype=torch.float32, device=dev)
        self.sd = torch.tensor(self.enc_meta["feature_norm_std"], dtype=torch.float32, device=dev)
        self.lam_psi = self.enc_meta["lambda_psi"]
        self.lam_D = self.enc_meta["lambda_D"]
        self.masks = region_masks(dev)

    def psi(self, z: torch.Tensor) -> torch.Tensor:
        return (self.ae.encode(render(z)) - self.mu) / self.sd

    def img_feat(self, z: torch.Tensor) -> torch.Tensor:
        return render(z).flatten(1) / math.sqrt(self.lam_D)

    def sem_feat(self, z: torch.Tensor) -> torch.Tensor:
        return self.psi(z) / math.sqrt(self.lam_psi)

    def terms(self, alpha: float | None) -> list[MetricTerm]:
        if alpha is None:  # M3: image pullback only
            return [MetricTerm("image", self.img_feat, 1.0)]
        return [MetricTerm("psi", self.sem_feat, 1.0), MetricTerm("image", self.img_feat, float(alpha))]

    def energy_fn(self, alpha: float | None):
        terms = self.terms(alpha)
        return lambda Z: discrete_energy(Z, terms, EPSILON)

    @torch.no_grad()
    def energy_parts(self, Z: torch.Tensor) -> dict[str, np.ndarray]:
        K = Z.shape[1] - 1
        img = 0.5 * K * segment_sqnorms(Z, self.img_feat).sum(-1)
        sem = 0.5 * K * segment_sqnorms(Z, self.sem_feat).sum(-1)
        eps = 0.5 * K * EPSILON * ((Z[:, 1:] - Z[:, :-1]) ** 2).sum(-1).sum(-1)
        return {"E_img": img.cpu().numpy(), "E_sem": sem.cpu().numpy(), "E_eps": eps.cpu().numpy()}

    def latent_score(self, seed: int):
        cnn = self.cnns[seed]
        return lambda z: logit_margin(cnn, render(z))

    def pixel_score(self, seed: int):
        cnn = self.cnns[seed]
        return lambda x: logit_margin(cnn, x)

    def exact_m3_energy(self, zb: np.ndarray, zx: np.ndarray) -> np.ndarray:
        """Minimum continuous M3 energy: the metric is diagonal and separable (disjoint supports)."""
        if not hasattr(self, "_s_tables"):
            grid = torch.linspace(0, 1, 2049, dtype=torch.float32, device=self.device)
            z = torch.full((len(grid), 4), 0.5, device=self.device)
            tables = []
            for j in range(4):
                zj = z.clone()
                zj[:, j] = grid
                g = image_metric_diag(zj)[:, j] / self.lam_D + EPSILON
                s = torch.cat(
                    [torch.zeros(1, device=self.device), torch.cumulative_trapezoid(torch.sqrt(g), grid)]
                )
                tables.append(s.double().cpu().numpy())
            self._s_tables, self._grid = tables, grid.double().cpu().numpy()
        ds = np.stack(
            [
                np.interp(zx[:, j], self._grid, self._s_tables[j])
                - np.interp(zb[:, j], self._grid, self._s_tables[j])
                for j in range(4)
            ],
            1,
        )
        return 0.5 * (ds**2).sum(1)


def _m5_paths(models: E1Models, zb: torch.Tensor, zx: torch.Tensor, changed: list[int], seed: int, K: int):
    """Fixed-seed random smooth endpoint-preserving latent paths within an energy band."""
    gen = torch.Generator().manual_seed(seed)
    lin = linear_waypoints(zb[None], zx[None], K)
    pert = torch.cat(
        [
            smooth_perturbation(1, K, 4, M5_SCALES[i % len(M5_SCALES)], gen, zb.device, zb.dtype, modes=3)
            for i in range(M5_MAX_DRAWS)
        ]
    )
    unchanged = [j for j in range(4) if j not in changed]
    pert[..., unchanged] = 0.0
    cand = lin + pert
    cand[:, 0], cand[:, -1] = zb, zx  # perturbations vanish at the ends; copy endpoints exactly
    in_box = ((cand >= 0) & (cand <= 1)).flatten(1).all(1)
    with torch.no_grad():
        e_img = discrete_energy(cand, models.terms(None), EPSILON)
        e_lin = discrete_energy(lin, models.terms(None), EPSILON)
    ratio = (e_img / e_lin).cpu()
    in_band = (ratio >= M5_ENERGY_BAND[0]) & (ratio <= M5_ENERGY_BAND[1])
    ok = (in_box.cpu() & in_band).nonzero().flatten()[:M5_CANDIDATES]
    info = {
        "m5_accepted": int(len(ok)),
        "m5_rejected_out_of_box": int((~in_box.cpu()).sum()),
        "m5_rejected_energy_band": int((in_box.cpu() & ~in_band).sum()),
        "m5_energy_ratios": ratio[ok].tolist(),
    }
    return cand[ok.to(cand.device)], info


def _attribute_latent(F, Zs: torch.Tensor, denom: torch.Tensor, steps) -> tuple[dict, np.ndarray]:
    def compute(N, idx):
        return latent_path_attribution(F, Zs[idx.to(Zs.device)], N)

    res, n_used = refine_until_complete(compute, denom, steps, REFINE_TARGET_REL)
    return {k: v.detach().cpu().numpy() for k, v in res.items()}, n_used.numpy()


def _attribute_pixel(f, X: torch.Tensor, masks, denom: torch.Tensor, steps) -> tuple[dict, np.ndarray]:
    def compute(N, idx):
        return pixel_path_attribution(f, X[idx.to(X.device)], masks, N)

    res, n_used = refine_until_complete(compute, denom, steps, REFINE_TARGET_REL)
    return {k: v.detach().cpu().numpy() for k, v in res.items()}, n_used.numpy()


def _methods(split: str, alphas: list[float], selected_alpha: float | None) -> list[tuple[str, float | None]]:
    base = [("M1", None), ("M2", None), ("M3", None), ("M5", None)]
    if split == "test":
        return base + [("M4", selected_alpha)]
    return base + [(f"M4_a{a:g}", a) for a in alphas]


def process_chunk(
    ctx: Ctx, models: E1Models, chunk: pd.DataFrame, methods, num: dict, C: float, split: str
) -> dict[str, Any]:
    """Compute paths, coalition values and attributions for one chunk of pairs."""
    dev = ctx.device
    K, steps = num["segments"], tuple(num["quadrature_steps"])
    zb_np, zx_np, changed = pair_arrays(chunk)
    P = len(chunk)
    zb = torch.as_tensor(np.array(zb_np), dtype=torch.float32, device=dev)
    zx = torch.as_tensor(np.array(zx_np), dtype=torch.float32, device=dev)
    seeds_restart = [torch_seed_for("path_restarts", ctx.gen_seed, int(i)) for i in chunk["candidate_index"]]
    split_off = 0 if split == "validation" else 1_000_000
    cfg = PathOptConfig(
        segments=K,
        iterations=num["iterations"],
        lr=num["lr"],
        restarts=num["restarts"],
        perturb_scale=num["perturb_scale"],
    )
    timings: dict[str, float] = {}
    paths: dict[str, torch.Tensor] = {}
    opt_info: dict[str, dict] = {}
    # ---- paths (independent of classifier seed)
    paths["M2"] = linear_waypoints(zb, zx, K)
    for name, alpha in methods:
        if name.startswith("M3") or name.startswith("M4"):
            t0 = time.time()
            res = optimize_paths(
                zb, zx, models.energy_fn(alpha), cfg, seeds=[s + split_off for s in seeds_restart]
            )
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            timings[f"path_{name}"] = (time.time() - t0) / P
            paths[name] = res["waypoints"]
            opt_info[name] = res
    m5_sets, m5_info = [], []
    t0 = time.time()
    for p in range(P):
        s = torch_seed_for("m5_paths", ctx.gen_seed, int(chunk["candidate_index"].iloc[p]) + split_off)
        cand, info = _m5_paths(models, zb[p], zx[p], changed[p], s, K)
        m5_sets.append(cand)
        m5_info.append(info)
    timings["path_M5"] = (time.time() - t0) / P
    # ---- path diagnostics
    path_rows = []
    exact_m3 = models.exact_m3_energy(zb_np, zx_np)
    for name, _ in methods:
        if name == "M1":
            continue
        if name == "M5":
            continue
        Z = paths[name]
        parts = models.energy_parts(Z)
        lengths = path_latent_length(Z).cpu().numpy()
        in_dom = ((Z >= -1e-6) & (Z <= 1 + 1e-6)).flatten(1).all(1).cpu().numpy()
        for p in range(P):
            r = {
                "scene_id": chunk["scene_id"].iloc[p],
                "method": name,
                "latent_length": float(lengths[p]),
                "in_domain": bool(in_dom[p]),
                "E_img": float(parts["E_img"][p]),
                "E_sem": float(parts["E_sem"][p]),
                "E_eps": float(parts["E_eps"][p]),
                "E_m3_exact_continuous": float(exact_m3[p]),
                "endpoint_exact": bool(torch.equal(Z[p, 0], zb[p]) and torch.equal(Z[p, -1], zx[p])),
            }
            if name in opt_info:
                oi = opt_info[name]
                e_fin = float(oi["energy_final"][p].min())
                e_lin = float(oi["energy_linear"][p])
                r.update(
                    {
                        "opt_energy_linear": e_lin,
                        "opt_energy_final": e_fin,
                        "opt_best_restart": int(oi["best_restart"][p]),
                        "opt_rel_tail_change": float(oi["rel_tail_change"][p]),
                        "opt_restart_energies": json.dumps([float(v) for v in oi["energy_final"][p]]),
                        "solver_diverged": bool(not math.isfinite(e_fin) or e_fin > e_lin * (1 + 1e-3)),
                    }
                )
            path_rows.append(r)
    for p in range(P):
        Zc = m5_sets[p]
        if len(Zc):
            parts = models.energy_parts(Zc)
            in_dom = bool(((Zc >= -1e-6) & (Zc <= 1 + 1e-6)).all())
            path_rows.append(
                {
                    "scene_id": chunk["scene_id"].iloc[p],
                    "method": "M5",
                    "latent_length": float(path_latent_length(Zc).mean()),
                    "in_domain": in_dom,
                    "E_img": float(parts["E_img"].mean()),
                    "E_sem": float(parts["E_sem"].mean()),
                    "E_eps": float(parts["E_eps"].mean()),
                    "E_m3_exact_continuous": float(exact_m3[p]),
                    "endpoint_exact": True,
                    **{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in m5_info[p].items()},
                }
            )
        else:
            path_rows.append(
                {
                    "scene_id": chunk["scene_id"].iloc[p],
                    "method": "M5",
                    "in_domain": True,
                    **{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in m5_info[p].items()},
                }
            )
    path_df = pd.DataFrame(path_rows)
    # ---- coalition values per seed (independent of all paths)
    hyb = [hybrid_latents(zb_np[p], zx_np[p], changed[p]) for p in range(P)]
    offsets = np.cumsum([0] + [len(h) for h in hyb])
    hyb_all = torch.as_tensor(np.concatenate(hyb), dtype=torch.float32, device=dev)
    coal_rows, values = [], {}
    for seed in models.seeds:
        v = eval_batched(models.latent_score(seed), hyb_all, 1024).double().cpu().numpy()
        values[seed] = [v[offsets[p] : offsets[p + 1]] for p in range(P)]
    oracle_vals = {
        "oracle_nl": [3 * g_nl(torch.as_tensor(h), C).numpy() for h in hyb],
        "oracle_poly": [g_poly(torch.as_tensor(h)).numpy() for h in hyb],
        "oracle_short": [g_short(torch.as_tensor(h), C).numpy() for h in hyb],
    }
    for p in range(P):
        for s_idx in range(len(hyb[p])):
            coal_rows.append(
                {
                    "scene_id": chunk["scene_id"].iloc[p],
                    "subset_mask": s_idx,
                    **{f"f_seed{seed}": float(values[seed][p][s_idx]) for seed in models.seeds},
                    **{k: float(v[p][s_idx]) for k, v in oracle_vals.items()},
                }
            )
    # ---- attributions
    rows, oracle_rows = [], []
    flat_floor = ctx.protocol.get("e1", "competence", "model_flat_range_floor")
    pathinfo = path_df.set_index(["scene_id", "method"])
    for seed in models.seeds:
        R = np.array([values[seed][p].max() - values[seed][p].min() for p in range(P)])
        denom = torch.as_tensor(R + DENOM_EPS, dtype=torch.float32)
        for name, _alpha in methods:
            t0 = time.time()
            if name == "M1":
                X = torch.stack([render(zb), render(zx)], 1)
                res, n_used = _attribute_pixel(models.pixel_score(seed), X, models.masks, denom, steps)
                A, resid = res["A"], res["residual"]
            elif name == "M5":
                counts = [len(c) for c in m5_sets]
                A = np.full((P, 4), np.nan)
                resid = np.full(P, np.nan)
                n_used = np.zeros(P, dtype=int)
                if sum(counts):
                    Zc = torch.cat([c for c in m5_sets if len(c)])
                    owner = np.repeat(np.arange(P), counts)
                    res, nu = _attribute_latent(
                        models.latent_score(seed),
                        Zc,
                        torch.as_tensor(R[owner] + DENOM_EPS, dtype=torch.float32),
                        steps,
                    )
                    for p in range(P):
                        sel = owner == p
                        if sel.any():
                            A[p] = res["A"][sel].mean(0)
                            resid[p] = res["residual"][sel].mean()
                            n_used[p] = nu[sel].max()
            else:
                res, n_used = _attribute_latent(models.latent_score(seed), paths[name], denom, steps)
                A, resid = res["A"], res["residual"]
            dt_attr = (time.time() - t0) / P
            for p in range(P):
                ch = changed[p]
                vals = values[seed][p]
                sid = chunk["scene_id"].iloc[p]
                finite = bool(np.all(np.isfinite(A[p])))
                ce = (
                    coalition_error(vals, A[p][ch])
                    if finite
                    else {"E_coal": np.nan, "E_coal_abs": np.nan, "R": float(R[p])}
                )
                phi = shapley_values(vals)
                rel = abs(resid[p]) / (R[p] + DENOM_EPS) if finite else np.nan
                reason = ""
                if name == "M5" and m5_info[p]["m5_accepted"] == 0:
                    reason = "NO_VALID_PAIR"
                elif not finite:
                    reason = "NUMERIC_INCOMPLETE"
                elif name not in ("M1",) and not bool(pathinfo.loc[(sid, name), "in_domain"]):
                    reason = "PATH_OUT_OF_DOMAIN"
                elif name in opt_info and bool(pathinfo.loc[(sid, name), "solver_diverged"]):
                    reason = "SOLVER_DIVERGED"
                elif rel > INVALID_REL:
                    reason = "NUMERIC_INCOMPLETE"
                unchanged = [j for j in range(4) if j not in ch]
                row = {
                    "scene_id": sid,
                    "split": split,
                    "stratum": chunk["stratum"].iloc[p],
                    "nuisance_changed": bool(chunk["nuisance_changed"].iloc[p]),
                    "eta": float(chunk["eta"].iloc[p]),
                    "m": len(ch),
                    "changed_factors": chunk["changed_factors"].iloc[p],
                    "seed": seed,
                    "method": name,
                    "alpha": _alpha if _alpha is not None else np.nan,
                    **{f"A_{f}": float(A[p][j]) for j, f in enumerate(FACTOR_NAMES)},
                    "f_b": float(vals[0]),
                    "f_x": float(vals[-1]),
                    "delta_full": float(vals[-1] - vals[0]),
                    "R": float(R[p]),
                    "model_flat": bool(R[p] <= flat_floor),
                    "E_coal": ce["E_coal"],
                    "E_coal_abs": ce["E_coal_abs"],
                    "completeness_abs": float(abs(resid[p])) if finite else np.nan,
                    "completeness_rel": float(rel),
                    "N_used": int(n_used[p]),
                    "shapley": json.dumps([float(v) for v in phi]),
                    "E_coal_shapley": coalition_error(vals, phi)["E_coal"],
                    "shapley_discrepancy": float(np.linalg.norm(A[p][ch] - phi) / (R[p] + DENOM_EPS))
                    if finite
                    else np.nan,
                    "sign_accuracy_single": single_factor_sign_accuracy(vals, A[p][ch]) if finite else np.nan,
                    "unchanged_attr_mass": float(np.abs(A[p][unchanged]).sum() / (R[p] + DENOM_EPS))
                    if unchanged and finite
                    else 0.0,
                    "valid": reason == "",
                    "invalid_reason": reason,
                    "sec_attr_per_pair": dt_attr,
                    "sec_path_per_pair": timings.get(f"path_{name}", 0.0),
                }
                rows.append(row)
    # ---- oracle diagnostic targets along latent paths (seed-independent)
    for target, fn in (
        ("oracle_nl", lambda z: 3 * g_nl(z, C)),
        ("oracle_poly", g_poly),
        ("oracle_short", lambda z: g_short(z, C)),
    ):
        Ro = np.array([oracle_vals[target][p].max() - oracle_vals[target][p].min() for p in range(P)])
        for name, _ in methods:
            if name == "M1":
                continue
            if name == "M5":
                counts = [len(c) for c in m5_sets]
                if not sum(counts):
                    continue
                Zc = torch.cat([c for c in m5_sets if len(c)])
                owner = np.repeat(np.arange(P), counts)
                res, _ = _attribute_latent(
                    fn, Zc, torch.as_tensor(Ro[owner] + DENOM_EPS, dtype=torch.float32), steps
                )
                A = np.stack(
                    [
                        res["A"][owner == p].mean(0) if (owner == p).any() else np.full(4, np.nan)
                        for p in range(P)
                    ]
                )
            else:
                res, _ = _attribute_latent(
                    fn, paths[name], torch.as_tensor(Ro + DENOM_EPS, dtype=torch.float32), steps
                )
                A = res["A"]
            for p in range(P):
                if not np.all(np.isfinite(A[p])):
                    continue
                ce = coalition_error(oracle_vals[target][p], A[p][changed[p]])
                oracle_rows.append(
                    {
                        "scene_id": chunk["scene_id"].iloc[p],
                        "stratum": chunk["stratum"].iloc[p],
                        "method": name,
                        "target": target,
                        "E_coal": ce["E_coal"],
                        "R": ce["R"],
                    }
                )
    path_store = {name: paths[name].cpu().numpy() for name in paths}
    for p in range(P):
        path_store[f"M5_{p}"] = m5_sets[p].cpu().numpy()
    return {
        "rows": pd.DataFrame(rows),
        "coalitions": pd.DataFrame(coal_rows),
        "paths": path_df,
        "oracle": pd.DataFrame(oracle_rows),
        "path_arrays": path_store,
    }


def _chunk_key(parts: dict) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def _run_chunks(
    ctx: Ctx,
    models: E1Models,
    pairs: pd.DataFrame,
    methods,
    num: dict,
    C: float,
    split: str,
    tag: str,
    resume: bool,
) -> dict[str, pd.DataFrame]:
    out_dir = ctx.paths.metrics / f"e1_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_key = {
        "pairs_sha": manifest_sha256(pairs),
        "methods": methods,
        "num": num,
        "cnn_sha": models.cnn_sha,
        "enc_sha": models.enc_sha,
        "source_sha": source_tree_sha256(),
        "protocol_sha": ctx.protocol.sha256,
    }
    size = num["chunk_pairs"]
    collected: dict[str, list] = {"rows": [], "coalitions": [], "paths": [], "oracle": []}
    for ci, s in enumerate(range(0, len(pairs), size)):
        chunk = pairs.iloc[s : s + size].reset_index(drop=True)
        key = _chunk_key(base_key | {"chunk": ci, "scenes": list(chunk["scene_id"])})
        meta_p = out_dir / f"chunk_{ci:03d}.json"
        files = {k: out_dir / f"chunk_{ci:03d}_{k}.parquet" for k in collected}
        if resume and meta_p.exists():
            meta = json.loads(meta_p.read_text())
            if meta["key"] == key and all(sha256_file(files[k]) == meta["files"][k] for k in files):
                for k in collected:
                    collected[k].append(pd.read_parquet(files[k]))
                print(f"[{tag}] chunk {ci} verified; skipping")
                continue
        t0 = time.time()
        res = process_chunk(ctx, models, chunk, methods, num, C, split)
        for k in collected:
            atomic_write_parquet(files[k], res[k])
            collected[k].append(res[k])
        import io

        buf = io.BytesIO()
        np.savez_compressed(buf, **res["path_arrays"])
        atomic_write_bytes(out_dir / f"chunk_{ci:03d}_paths.npz", buf.getvalue())
        atomic_write_json(
            meta_p,
            {
                "key": key,
                "files": {k: sha256_file(files[k]) for k in files},
                "seconds": time.time() - t0,
                "written_utc": utcnow(),
            },
        )
        print(f"[{tag}] chunk {ci} done in {time.time() - t0:.1f}s", flush=True)
    return {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for k, v in collected.items()}


def _num_config_path(ctx: Ctx) -> Path:
    return ctx.paths.manifest_dir / "numeric_config.json"


def _frozen_path(ctx: Ctx) -> Path:
    return ctx.paths.manifest_dir / "frozen_manifest.json"


def selection_score(rows: pd.DataFrame) -> pd.DataFrame:
    """Equal-stratum-weighted mean coalition error per method (scene-averaged over seeds).

    Model-flat scene-seeds are excluded for all methods alike; non-finite errors are replaced
    by the worst finite error observed for that scene (a penalty, never an exclusion).
    """
    r = rows[~rows["model_flat"]].copy()
    worst = r.groupby(["scene_id", "seed"])["E_coal"].transform("max")
    r["E_sel"] = r["E_coal"].where(np.isfinite(r["E_coal"]), worst)
    scene = r.groupby(["method", "scene_id", "stratum"], as_index=False)["E_sel"].mean()
    strata = scene.groupby(["method", "stratum"])["E_sel"].mean().unstack()
    strata["equal_weight"] = 0.5 * strata["strong"] + 0.5 * strata["weak"]
    return strata.reset_index()


def profile_numerics(ctx: Ctx, models: E1Models, pairs: pd.DataFrame, C: float) -> dict:
    """Validation-only profiling on the first `profiling.validation_pair_count` pairs.

    Rule (fixed in code before profiling): use the smallest iteration count among the candidates
    for which >= 90% of optimised M3/M4 paths have relative energy change < 1e-3 over the last
    20% of iterations; otherwise the largest candidate. Quadrature refinement, epsilon, segments
    and restarts are not tuned.
    """
    n_prof = (
        SMOKE["profile_pairs"] if ctx.smoke else ctx.protocol.get("e1", "profiling", "validation_pair_count")
    )
    prof = pairs.iloc[:n_prof].reset_index(drop=True)
    path_cfg = ctx.protocol.get("e1", "path")
    alphas = path_cfg["alpha_validation_grid"]
    cands = SMOKE["iteration_candidates"] if ctx.smoke else ITERATION_CANDIDATES
    zb_np, zx_np, _ = pair_arrays(prof)
    zb = torch.as_tensor(zb_np, dtype=torch.float32, device=ctx.device)
    zx = torch.as_tensor(zx_np, dtype=torch.float32, device=ctx.device)
    seeds = [torch_seed_for("path_restarts", ctx.gen_seed, int(i)) for i in prof["candidate_index"]]
    trials, chosen = [], cands[-1]
    for iters in cands:
        cfg = PathOptConfig(
            segments=path_cfg["initial_segments"],
            iterations=iters,
            lr=0.01,
            restarts=path_cfg["restart_count"],
        )
        tails, secs = [], {}
        for alpha in [None, *alphas]:
            t0 = time.time()
            res = optimize_paths(zb, zx, models.energy_fn(alpha), cfg, seeds)
            secs[str(alpha)] = time.time() - t0
            tails += res["rel_tail_change"].tolist()
        frac = float(np.mean(np.array(tails) < CONVERGENCE_REL_TAIL))
        trials.append({"iterations": iters, "fraction_converged": frac, "seconds": secs})
        if frac >= CONVERGENCE_FRACTION:
            chosen = iters
            break
    num = {
        "segments": path_cfg["initial_segments"],
        "iterations": int(chosen),
        "lr": 0.01,
        "restarts": path_cfg["restart_count"],
        "perturb_scale": 0.1,
        "quadrature_steps": list(path_cfg["allowed_quadrature_steps"]),
        "refine_target_rel": REFINE_TARGET_REL,
        "invalid_rel": INVALID_REL,
        "epsilon": EPSILON,
        "chunk_pairs": SMOKE["chunk_pairs"] if ctx.smoke else CHUNK_PAIRS,
        "m5": {
            "candidates": M5_CANDIDATES,
            "max_draws": M5_MAX_DRAWS,
            "energy_band": M5_ENERGY_BAND,
            "scales": M5_SCALES,
            "perturbs_changed_factors_only": True,
        },
        "profiling_trials": trials,
        "profiling_pairs": list(prof["scene_id"]),
    }
    num["pixel_latent_crosscheck"] = pixel_latent_crosscheck(models, zb[:3], zx[:3], ctx.gen_seed)
    # Full-method dry pass on the profiling pairs for runtime/VRAM/completeness evidence.
    if ctx.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    res = process_chunk(ctx, models, prof, _methods("validation", alphas, None), num, C, "validation")
    num["profile_seconds_full_methods"] = time.time() - t0
    num["profile_peak_gpu_gb"] = (
        (torch.cuda.max_memory_allocated() / 2**30) if ctx.device.startswith("cuda") else 0
    )
    rows = res["rows"]
    num["profile_completeness_rel"] = rows.groupby("method")["completeness_rel"].describe().to_dict()
    atomic_write_parquet(ctx.paths.metrics / "e1_profile_rows.parquet", rows)
    return num


def pixel_latent_crosscheck(models: E1Models, zb: torch.Tensor, zx: torch.Tensor, gen_seed: int) -> dict:
    """Latent-factor vs pixel-group attribution on linear and curved decoded paths (seed 11 model).

    With jvp displacements the two must agree to float precision (disjoint supports); with
    finite pixel displacements they agree up to quadrature error.
    """
    K = 16
    lin = linear_waypoints(zb, zx, K)
    gen = torch.Generator().manual_seed(gen_seed % 2**31)
    curved = (lin + smooth_perturbation(len(zb), K, 4, 0.1, gen, zb.device, zb.dtype)).clamp(0, 1)
    curved[:, 0], curved[:, -1] = zb, zx
    seed = models.seeds[0]
    out = {}
    for name, Z in (("linear", lin), ("curved", curved)):
        lat = latent_path_attribution(models.latent_score(seed), Z, 64)["A"]
        pj = latent_path_pixel_attribution(
            models.pixel_score(seed), render, Z, models.masks, 64, use_jvp=True
        )
        pf = latent_path_pixel_attribution(
            models.pixel_score(seed), render, Z, models.masks, 512, use_jvp=False
        )
        out[name] = {
            "max_abs_latent_vs_pixel_jvp": float((lat - pj).abs().max()),
            "max_abs_latent_vs_pixel_finite512": float((lat - pf).abs().max()),
            "max_abs_attr": float(lat.abs().max()),
        }
    return out


def evaluate_e1(ctx: Ctx, split: str, resume: bool = False, dry_run: bool = False) -> dict:
    path_cfg = ctx.protocol.get("e1", "path")
    alphas = path_cfg["alpha_validation_grid"]
    if dry_run:
        n = 80 if split == "validation" else 240
        print(
            json.dumps(
                {
                    "stage": f"evaluate-e1:{split}",
                    "pairs": n,
                    "seeds": 2,
                    "methods": [m for m, _ in _methods(split, alphas, 1.0)],
                    "est_gpu_hours": 0.5,
                    "peak_vram_target_gb": 14.5,
                },
                indent=2,
            )
        )
        return {}
    _require_cuda(ctx)
    man = _load_prepared(ctx)
    C = man["calibration"]["C"]
    pairs = pd.read_parquet(ctx.paths.data_e1 / f"pairs_{split}.parquet")
    for p in ("cnn_seed11.json", "cnn_seed29.json", "encoder_ae.json"):
        _require(ctx.paths.checkpoints / p, p)
    models = E1Models(ctx)
    if split == "validation":
        num_p = _num_config_path(ctx)
        if not num_p.exists():
            with ResourceMeter(ctx.paths, "profile-e1", ctx.device):
                num = profile_numerics(ctx, models, pairs, C)
            write_immutable_json(num_p, num)
        num = json.loads(num_p.read_text())
        methods = _methods("validation", alphas, None)
        with (
            ctx.tracker.run(
                ctx.exp("e1-paths"),
                "validation",
                tags={
                    "stage": "evaluate-e1",
                    "split": "validation",
                    "model_checkpoint_sha256": json.dumps(models.cnn_sha),
                    "representation_checkpoint_sha256": models.enc_sha,
                    "dataset_manifest_sha256": manifest_sha256(pairs),
                },
                params={"iterations": num["iterations"], "epsilon": EPSILON, "alphas": alphas},
            ) as run,
            ResourceMeter(ctx.paths, "evaluate-e1-validation", ctx.device) as meter,
        ):
            res = _run_chunks(ctx, models, pairs, methods, num, C, "validation", "validation", resume)
            sel = selection_score(res["rows"])
            comp_rows = sel[sel["method"].isin(["M1", "M2", "M3", "M5"])].sort_values("equal_weight")
            m4_rows = sel[sel["method"].str.startswith("M4")].sort_values("equal_weight")
            comparator = str(comp_rows.iloc[0]["method"])
            m4_best = str(m4_rows.iloc[0]["method"])
            alpha = float(m4_best.split("_a")[1])
            for k in ("rows", "coalitions", "paths", "oracle"):
                atomic_write_parquet(ctx.paths.metrics / f"e1_validation_{k}.parquet", res[k])
            atomic_write_parquet(ctx.paths.metrics / "e1_validation_selection.parquet", sel)
            for k in ("rows", "paths"):
                ctx.tracker.log_artifact(ctx.paths.metrics / f"e1_validation_{k}.parquet", "validation")
            ctx.tracker.log_artifact(ctx.paths.metrics / "e1_validation_selection.parquet", "validation")
            for _, r in sel.iterrows():
                ctx.tracker.log_metrics(
                    {
                        f"val_{r['method']}_equal_weight": r["equal_weight"],
                        f"val_{r['method']}_strong": r["strong"],
                        f"val_{r['method']}_weak": r["weak"],
                    }
                )
            val_run = run.info.run_id
        selection = {
            "comparator": comparator,
            "m4_alpha": alpha,
            "m4_validation_method": m4_best,
            "validation_scores": sel.to_dict(orient="records"),
            "validation_mlflow_run_id": val_run,
            "rule": "min equal-stratum-weighted validation mean coalition error",
        }
        frozen = freeze_manifest(ctx, models, num, selection, pairs_val=pairs)
        write_status(
            ctx.paths,
            "evaluate-e1-validation",
            "COMPLETE",
            selection=selection,
            frozen_manifest_sha256=frozen,
            resources=meter.summary(),
            smoke=ctx.smoke,
        )
        return selection
    # ------------------------------------------------------------------ test (blinded until frozen)
    frozen_p = _frozen_path(ctx)
    _require(frozen_p, "frozen manifest (run validation first)")
    frozen = json.loads(frozen_p.read_text())
    verify_frozen(ctx, frozen, models, pairs)
    unblind_p = ctx.paths.manifest_dir / "unblinding.json"
    if not unblind_p.exists():
        write_immutable_json(
            unblind_p,
            {"test_evaluation_started_utc": utcnow(), "frozen_manifest_sha256": sha256_file(frozen_p)},
        )
    num = frozen["numeric_config"]
    sel = frozen["selection"]
    methods = _methods("test", alphas, sel["m4_alpha"])
    with (
        ctx.tracker.run(
            ctx.exp("e1-paths"),
            "test",
            tags={
                "stage": "evaluate-e1",
                "split": "test",
                "selected_comparator": sel["comparator"],
                "m4_alpha": sel["m4_alpha"],
                "frozen_manifest_sha256": sha256_file(frozen_p),
                "model_checkpoint_sha256": json.dumps(models.cnn_sha),
                "representation_checkpoint_sha256": models.enc_sha,
                "dataset_manifest_sha256": manifest_sha256(pairs),
            },
            params={
                "iterations": num["iterations"],
                "comparator": sel["comparator"],
                "alpha": sel["m4_alpha"],
            },
        ) as run,
        ResourceMeter(ctx.paths, "evaluate-e1-test", ctx.device) as meter,
    ):
        ctx.tracker.log_artifact(frozen_p, "frozen")
        res = _run_chunks(ctx, models, pairs, methods, num, C, "test", "test", resume)
        for k in ("rows", "coalitions", "paths", "oracle"):
            atomic_write_parquet(ctx.paths.metrics / f"e1_test_{k}.parquet", res[k])
            ctx.tracker.log_artifact(ctx.paths.metrics / f"e1_test_{k}.parquet", "test")
        test_run = run.info.run_id
    n_invalid = int((~res["rows"]["valid"]).sum())
    write_status(
        ctx.paths,
        "evaluate-e1-test",
        "COMPLETE" if n_invalid == 0 else "COMPLETE_WITH_FAILURES",
        mlflow_run_id=test_run,
        n_rows=len(res["rows"]),
        n_invalid=n_invalid,
        resources=meter.summary(),
        smoke=ctx.smoke,
    )
    return {"mlflow_run_id": test_run}


def _frozen_inputs(ctx: Ctx, models: E1Models) -> dict:
    lock = ctx.paths.root / "uv.lock"
    from .tracking import provenance

    prov = provenance(ctx.protocol)
    return {
        "protocol_sha256": ctx.protocol.sha256,
        "preregistration_sha256": prov["preregistration_sha256"],
        "source_sha256": source_tree_sha256(),
        "git_commit": prov["git_commit"],
        "uv_lock_sha256": sha256_file(lock) if lock.exists() else "MISSING",
        "classifier_checkpoints_sha256": {str(k): v for k, v in models.cnn_sha.items()},
        "encoder_checkpoint_sha256": models.enc_sha,
        "data_files_sha256": json.loads((ctx.paths.data_e1 / "prepare_manifest.json").read_text())["files"],
    }


def freeze_manifest(ctx: Ctx, models: E1Models, num: dict, selection: dict, pairs_val: pd.DataFrame) -> str:
    test_pairs = pd.read_parquet(ctx.paths.data_e1 / "pairs_test.parquet")
    obj = {
        "frozen_utc": utcnow(),
        **_frozen_inputs(ctx, models),
        "numeric_config": num,
        "selection": selection,
        "validation_pairs_content_sha256": manifest_sha256(pairs_val),
        "test_pairs_content_sha256": manifest_sha256(test_pairs),
        "test_scene_ids": list(test_pairs["scene_id"]),
        "encoder": {"lambda_psi": models.lam_psi, "lambda_D": models.lam_D, "epsilon": EPSILON},
        "note": "Written after validation-only selection and before any test attribution.",
    }
    p = _frozen_path(ctx)
    h = write_immutable_json(p, obj)
    with ctx.tracker.run(
        ctx.exp("decisions"), "freeze", tags={"stage": "freeze", "frozen_manifest_sha256": h}
    ):
        ctx.tracker.log_artifact(p, "frozen")
    return h


def verify_frozen(ctx: Ctx, frozen: dict, models: E1Models, test_pairs: pd.DataFrame) -> None:
    now = _frozen_inputs(ctx, models)
    bad = [
        k
        for k in (
            "protocol_sha256",
            "preregistration_sha256",
            "source_sha256",
            "uv_lock_sha256",
            "classifier_checkpoints_sha256",
            "encoder_checkpoint_sha256",
            "data_files_sha256",
        )
        if now[k] != frozen[k]
    ]
    if manifest_sha256(test_pairs) != frozen["test_pairs_content_sha256"]:
        bad.append("test_pairs_content_sha256")
    if bad:
        raise RuntimeError(f"frozen manifest mismatch for {bad}; refusing to unblind test set")
