"""Pullback metrics, discrete path energies and an endpoint-fixed waypoint optimizer.

A path is stored as waypoints `Z` of shape (P, K+1, d) with Z[:,0]=z_b and Z[:,K]=z_x held
fixed exactly. For a feature map phi (decoder, representation, or identity) the pullback
energy 1/2 * int zdot^T J_phi^T J_phi zdot dt is approximated by finite differences

    E = (K/2) * sum_k ||phi(z_{k+1}) - phi(z_k)||^2,

so no dense Jacobian is ever formed. A hybrid metric is a weighted sum of such terms plus
an isotropic epsilon term.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from scipy import integrate, optimize

FeatureFn = Callable[[torch.Tensor], torch.Tensor]


# --------------------------------------------------------------------------- E0 analytic pieces
def e0_decoder(z: torch.Tensor, c: float) -> torch.Tensor:
    """D_c(u, v) = (u, v, c u^2)."""
    u, v = z[..., 0], z[..., 1]
    return torch.stack([u, v, c * u**2], dim=-1)


def e0_arclength(u: float | np.ndarray, c: float) -> float | np.ndarray:
    """s_c(u) = int_0^u sqrt(1 + 4 c^2 r^2) dr (closed form)."""
    u = np.asarray(u, dtype=np.float64)
    if c == 0:
        return u
    k = 2.0 * c
    return 0.5 * u * np.sqrt(1 + (k * u) ** 2) + np.arcsinh(k * u) / (2 * k)


def e0_analytic_attribution(c: float) -> tuple[float, float]:
    """(A_u, A_v) of f=x1*x2 along the image-pullback geodesic (0,0)->(1,1)."""
    L = float(e0_arclength(1.0, c))
    integral, _ = integrate.quad(lambda u: float(e0_arclength(u, c)), 0.0, 1.0, epsabs=1e-13, epsrel=1e-13)
    a_u = integral / L
    return a_u, 1.0 - a_u


def e0_analytic_geodesic(c: float, n: int) -> np.ndarray:
    """Waypoints (n+1, 2) of the exact geodesic: s_c(u(t)) = t L_c, v(t) = t."""
    L = float(e0_arclength(1.0, c))
    ts = np.linspace(0.0, 1.0, n + 1)
    us = [0.0]
    for t in ts[1:-1]:
        us.append(optimize.brentq(lambda u, t=t: float(e0_arclength(u, c)) - t * L, 0.0, 1.0, xtol=1e-15))
    us.append(1.0)
    return np.stack([np.array(us), ts], axis=1)


def e0_analytic_energies(c: float) -> tuple[float, float]:
    """Continuous energies (geodesic, latent-linear) under G_c = diag(1+4c^2u^2, 1)."""
    L = float(e0_arclength(1.0, c))
    return 0.5 * (L**2 + 1.0), 0.5 * (1.0 + 4.0 * c**2 / 3.0 + 1.0)


# --------------------------------------------------------------------------- discrete energies
@dataclass(frozen=True)
class MetricTerm:
    name: str
    fn: FeatureFn  # maps (N, d) latent points to features (N, ...)
    weight: float


def segment_sqnorms(Z: torch.Tensor, fn: FeatureFn) -> torch.Tensor:
    """||phi(z_{k+1}) - phi(z_k)||^2 for every segment: (P, K)."""
    P, K1, d = Z.shape
    feats = fn(Z.reshape(P * K1, d))
    feats = feats.reshape(P, K1, -1)
    diff = feats[:, 1:] - feats[:, :-1]
    return (diff**2).sum(-1)


def discrete_energy(Z: torch.Tensor, terms: Sequence[MetricTerm], eps: float) -> torch.Tensor:
    """Hybrid discrete energy per path: (P,)."""
    K = Z.shape[1] - 1
    seg = eps * ((Z[:, 1:] - Z[:, :-1]) ** 2).sum(-1)
    for t in terms:
        if t.weight != 0:
            seg = seg + t.weight * segment_sqnorms(Z, t.fn)
    return 0.5 * K * seg.sum(-1)


def path_latent_length(Z: torch.Tensor) -> torch.Tensor:
    return (Z[:, 1:] - Z[:, :-1]).norm(dim=-1).sum(-1)


def linear_waypoints(zb: torch.Tensor, zx: torch.Tensor, K: int) -> torch.Tensor:
    """Latent-linear waypoints with endpoints copied exactly (no floating-point drift)."""
    t = torch.linspace(0, 1, K + 1, device=zb.device, dtype=zb.dtype).view(1, K + 1, 1)
    Z = zb.unsqueeze(1) + t * (zx - zb).unsqueeze(1)
    Z[:, 0] = zb
    Z[:, -1] = zx
    return Z


def smooth_perturbation(
    P: int, K: int, d: int, scale: float, gen: torch.Generator, device, dtype, modes: int = 2
) -> torch.Tensor:
    """sum_m c_m sin(m pi t), c_m ~ N(0, scale^2): zero at both endpoints, (P, K+1, d)."""
    t = torch.linspace(0, 1, K + 1, dtype=torch.float64)
    coefs = torch.randn(P, modes, d, generator=gen, dtype=torch.float64) * scale
    basis = torch.stack([torch.sin((m + 1) * math.pi * t) for m in range(modes)], dim=0)  # (modes, K+1)
    pert = torch.einsum("pmd,mk->pkd", coefs, basis)
    return pert.to(device=device, dtype=dtype)


@dataclass(frozen=True)
class PathOptConfig:
    segments: int = 16
    iterations: int = 100
    lr: float = 0.01
    restarts: int = 3
    perturb_scale: float = 0.1
    box_lo: float = 0.0
    box_hi: float = 1.0


def optimize_paths(
    zb: torch.Tensor,
    zx: torch.Tensor,
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    cfg: PathOptConfig,
    seeds: Sequence[int],
) -> dict[str, torch.Tensor]:
    """Minimise the discrete energy over internal waypoints with projected Adam.

    Restart 0 is latent-linear; restarts 1..R-1 are seeded smooth perturbations of it.
    The returned path per pair is the restart with the lowest *final energy* (geometry only).
    Endpoints are never optimised and are copied exactly from (zb, zx).
    """
    P, d = zb.shape
    K, R = cfg.segments, cfg.restarts
    lin = linear_waypoints(zb, zx, K)  # (P, K+1, d)
    inits = [lin]
    for r in range(1, R):
        perts = []
        for s in seeds:
            g = torch.Generator().manual_seed(int(s) * 7919 + r)
            perts.append(smooth_perturbation(1, K, d, cfg.perturb_scale, g, zb.device, zb.dtype))
        inits.append((lin + torch.cat(perts, 0)).clamp(cfg.box_lo, cfg.box_hi))
    Z0 = torch.stack(inits, dim=1).reshape(P * R, K + 1, d)  # restart-major within pair
    start = Z0[:, :1].detach()
    end = Z0[:, -1:].detach()
    inner = Z0[:, 1:-1].detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([inner], lr=cfg.lr)

    def full(inner_: torch.Tensor) -> torch.Tensor:
        return torch.cat([start, inner_, end], dim=1)

    with torch.no_grad():
        e_lin = energy_fn(lin)
        e_init = energy_fn(Z0)
    curve = torch.empty(cfg.iterations + 1, P * R, dtype=torch.float64)
    curve[0] = e_init.detach().double().cpu()
    for it in range(cfg.iterations):
        opt.zero_grad(set_to_none=True)
        e = energy_fn(full(inner))
        e.sum().backward()
        opt.step()
        with torch.no_grad():
            inner.clamp_(cfg.box_lo, cfg.box_hi)
            curve[it + 1] = energy_fn(full(inner)).double().cpu()
    with torch.no_grad():
        Zf = full(inner).detach()
        e_final = curve[-1].view(P, R)
        finite = torch.isfinite(e_final)
        masked = torch.where(finite, e_final, torch.full_like(e_final, float("inf")))
        best = masked.argmin(dim=1)
        Zsel = Zf.view(P, R, K + 1, d)[torch.arange(P), best.to(Zf.device)]
        # Exact endpoint immutability (guards against any in-place drift).
        Zsel[:, 0] = zb
        Zsel[:, -1] = zx
        tail = max(1, cfg.iterations // 5)
        c = curve.view(cfg.iterations + 1, P, R)[:, torch.arange(P), best]  # (iters+1, P)
        rel_tail = (c[-tail - 1] - c[-1]).abs() / c[-1].abs().clamp_min(1e-12)
    return {
        "waypoints": Zsel,
        "energy_linear": e_lin.double().cpu(),
        "energy_init": e_init.view(P, R).double().cpu(),
        "energy_final": e_final,
        "best_restart": best,
        "curve_selected": c,
        "rel_tail_change": rel_tail,
    }
