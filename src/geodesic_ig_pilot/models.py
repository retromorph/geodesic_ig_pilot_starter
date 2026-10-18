"""Small CNN classifier (explained model) and an independent unlabeled masked autoencoder.

Both use smooth SiLU activations and GroupNorm (per-sample, so eval-mode outputs never
depend on batch composition). Images are rendered on the fly from stored factor vectors.
"""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .renderer import render


def _block(cin: int, cout: int, stride: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1), nn.GroupNorm(8, cout), nn.SiLU())


class SmallCNN(nn.Module):
    """4 stride-2 conv blocks (64->4 px), flattened spatial head -> 2 logits."""

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        w = width
        self.features = nn.Sequential(
            _block(3, w, 1),
            _block(w, w, 2),
            _block(w, 2 * w, 2),
            _block(2 * w, 3 * w, 2),
            _block(3 * w, 4 * w, 2),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(4 * w * 16, 128), nn.SiLU(), nn.Linear(128, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x - 0.5))


class MaskedAE(nn.Module):
    """Compact convolutional autoencoder; `encode` gives the frozen representation psi."""

    def __init__(self, width: int = 32, latent: int = 32) -> None:
        super().__init__()
        w = width
        self.enc = nn.Sequential(
            _block(3, w, 2),
            _block(w, 2 * w, 2),
            _block(2 * w, 4 * w, 2),
            _block(4 * w, 4 * w, 2),
            nn.Flatten(),
        )
        self.to_z = nn.Linear(4 * w * 16, latent)
        self.from_z = nn.Linear(latent, 4 * w * 16)
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2),
            _block(4 * w, 4 * w, 1),
            nn.Upsample(scale_factor=2),
            _block(4 * w, 2 * w, 1),
            nn.Upsample(scale_factor=2),
            _block(2 * w, w, 1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(w, 3, 3, padding=1),
        )
        self.w = w

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_z(self.enc(x - 0.5))

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.dec(self.from_z(h).view(-1, 4 * self.w, 4, 4)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def freeze(model: nn.Module) -> nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def random_patch_mask(x: torch.Tensor, frac: float, patch: int, gen: torch.Generator) -> torch.Tensor:
    """Generic (mask-agnostic) random patch masking with gray fill for masked reconstruction."""
    B, _, H, W = x.shape
    keep = (torch.rand(B, 1, H // patch, W // patch, generator=gen) >= frac).to(x.device, x.dtype)
    keep = F.interpolate(keep, scale_factor=patch, mode="nearest")
    return x * keep + 0.5 * (1 - keep)


def _render_batches(z: np.ndarray, device: str, bs: int):
    zt = torch.as_tensor(z, dtype=torch.float32)
    for s in range(0, len(zt), bs):
        yield s, render(zt[s : s + bs].to(device))


def train_classifier(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    device: str,
    epochs: int,
    batch_size: int = 128,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    log=None,
) -> tuple[SmallCNN, dict[str, Any]]:
    """Train on train split only; checkpoint selection by classifier-validation NLL."""
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    model = SmallCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = epochs * int(np.ceil(len(z_train) / batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    zt = torch.as_tensor(np.array(z_train), dtype=torch.float32)
    yt = torch.as_tensor(y_train, dtype=torch.long)
    best, best_state, curves = float("inf"), None, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = np_rng.permutation(len(zt))
        tot, nb = 0.0, 0
        for s in range(0, len(perm), batch_size):
            idx = perm[s : s + batch_size]
            x = render(zt[idx].to(device))
            loss = F.cross_entropy(model(x), yt[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach())
            nb += 1
        val = evaluate_classifier(model, z_val, y_val, device)
        curves.append({"epoch": ep, "train_nll": tot / nb, **{f"val_{k}": v for k, v in val.items()}})
        if log:
            log(curves[-1])
        if val["nll"] < best:
            best, best_state = val["nll"], copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, {"curves": curves, "best_val_nll": best, "train_seconds": time.time() - t0}


@torch.no_grad()
def classifier_logits(model: nn.Module, z: np.ndarray, device: str, bs: int = 1024) -> np.ndarray:
    model.eval()
    out = [model(x).float().cpu() for _, x in _render_batches(z, device, bs)]
    return torch.cat(out).numpy()


def evaluate_classifier(model: nn.Module, z: np.ndarray, y: np.ndarray, device: str) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    logits = classifier_logits(model, z, device)
    margin = logits[:, 1] - logits[:, 0]
    p = 1 / (1 + np.exp(-margin))
    eps = 1e-7
    nll = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
    # 10-bin expected calibration error.
    bins = np.clip((p * 10).astype(int), 0, 9)
    ece = float(
        sum(
            abs(p[bins == k].mean() - y[bins == k].mean()) * (bins == k).mean()
            for k in range(10)
            if (bins == k).any()
        )
    )
    return {
        "auroc": float(roc_auc_score(y, margin)),
        "accuracy": float(np.mean((margin > 0) == (y == 1))),
        "nll": nll,
        "ece": ece,
        "pred_pos_rate": float(np.mean(margin > 0)),
        "label_pos_rate": float(np.mean(y)),
    }


def train_autoencoder(
    z_train: np.ndarray,
    z_val: np.ndarray,
    seed: int,
    device: str,
    epochs: int,
    batch_size: int = 128,
    lr: float = 2e-3,
    mask_frac: float = 0.25,
    log=None,
) -> tuple[MaskedAE, dict[str, Any]]:
    """Masked-reconstruction AE on unlabeled train-split images (no labels/factors/masks used)."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    model = MaskedAE().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    steps = epochs * int(np.ceil(len(z_train) / batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    zt = torch.as_tensor(np.array(z_train), dtype=torch.float32)
    best, best_state, curves = float("inf"), None, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = np_rng.permutation(len(zt))
        tot, nb = 0.0, 0
        for s in range(0, len(perm), batch_size):
            x = render(zt[perm[s : s + batch_size]].to(device))
            xm = random_patch_mask(x, mask_frac, 8, gen)
            loss = F.mse_loss(model(xm), x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach())
            nb += 1
        model.eval()
        with torch.no_grad():
            val = float(
                np.mean([F.mse_loss(model(x), x).item() for _, x in _render_batches(z_val, device, 512)])
            )
        curves.append({"epoch": ep, "train_masked_mse": tot / nb, "val_mse": val})
        if log:
            log(curves[-1])
        if val < best:
            best, best_state = val, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, {"curves": curves, "best_val_mse": best, "train_seconds": time.time() - t0}


@torch.no_grad()
def encoder_diagnostics(ae: MaskedAE, z_val: np.ndarray, device: str) -> dict[str, Any]:
    feats = torch.cat([ae.encode(x).cpu() for _, x in _render_batches(z_val, device, 512)]).double().numpy()
    mu, sd = feats.mean(0), feats.std(0)
    cov = np.cov(((feats - mu) / np.maximum(sd, 1e-12)).T)
    ev = np.clip(np.linalg.eigvalsh(cov), 0, None)
    return {
        "feature_dim": int(feats.shape[1]),
        "feature_mean": mu.tolist(),
        "feature_std": sd.tolist(),
        "n_near_constant": int(np.sum(sd < 1e-3 * sd.max())),
        "effective_rank_participation": float(ev.sum() ** 2 / (ev**2).sum()),
    }


def state_dict_sha256(path) -> str:
    from .config import sha256_file

    return sha256_file(path)
