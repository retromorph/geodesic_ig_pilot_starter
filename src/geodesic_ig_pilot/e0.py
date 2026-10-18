"""E0: analytic geodesic sanity checks and negative controls (CPU, float64)."""

from __future__ import annotations

from typing import Any

import torch

from .geometry import (
    MetricTerm,
    PathOptConfig,
    discrete_energy,
    e0_analytic_attribution,
    e0_analytic_energies,
    e0_analytic_geodesic,
    e0_decoder,
    linear_waypoints,
    optimize_paths,
    smooth_perturbation,
)
from .integration import latent_path_attribution, resample_path
from .scores import e0_add, e0_mul

DT = torch.float64
E0_SEGMENTS = 128
E0_ITERS = 4000
E0_LR = 0.002
E0_QUAD = 1024


def image_coordinate_attribution(f, c: float, Z: torch.Tensor, N: int) -> torch.Tensor:
    """A_i in *image* coordinates along x(t)=D_c(z(t)) with finite segment displacements: (P, 3)."""
    pts = resample_path(Z, N)
    mids = 0.5 * (pts[:, 1:] + pts[:, :-1])
    xm = e0_decoder(mids, c).detach().requires_grad_(True)
    (g,) = torch.autograd.grad(f(xm).sum(), xm)
    disp = e0_decoder(pts[:, 1:], c) - e0_decoder(pts[:, :-1], c)
    return (g * disp).sum(1)


def fd_gradient_check(c: float, n_points: int = 8, h: float = 1e-6) -> float:
    """Max |autograd - central finite difference| of (f o D_c) at random latent points."""
    gen = torch.Generator().manual_seed(123)
    z = torch.rand(n_points, 2, generator=gen, dtype=DT).requires_grad_(True)
    F = e0_mul(e0_decoder(z, c)) + 0.3 * e0_decoder(z, c)[..., 2] ** 2
    (g,) = torch.autograd.grad(F.sum(), z)
    fd = torch.zeros_like(z)
    with torch.no_grad():
        for j in range(2):
            e = torch.zeros(2, dtype=DT)
            e[j] = h

            def fn(zz):
                x = e0_decoder(zz, c)
                return e0_mul(x) + 0.3 * x[..., 2] ** 2

            fd[:, j] = (fn(z + e) - fn(z - e)) / (2 * h)
    return float((g - fd).abs().max())


def run_e0(protocol) -> dict[str, Any]:
    e0 = protocol.get("e0")
    tol = e0["absolute_tolerance"]
    zb = torch.zeros(1, 2, dtype=DT)
    zx = torch.ones(1, 2, dtype=DT)
    rows, checks = [], []
    for c, exp_u, exp_v in zip(e0["curvature_c"], e0["expected_u_attribution"], e0["expected_v_attribution"]):
        au, av = e0_analytic_attribution(c)
        # (1) analytic geodesic waypoints integrated by our quadrature
        Zg = torch.as_tensor(e0_analytic_geodesic(c, 2048), dtype=DT).unsqueeze(0)
        A_geo_path = image_coordinate_attribution(e0_mul, c, Zg, 2048)[0]
        # (2) numerical waypoint optimizer on the image-pullback energy
        img_term = [MetricTerm("image", lambda z, c=c: e0_decoder(z, c), 1.0)]
        cfg = PathOptConfig(
            segments=E0_SEGMENTS, iterations=E0_ITERS, lr=E0_LR, restarts=3, perturb_scale=0.1
        )
        opt = optimize_paths(zb, zx, lambda Z, t=img_term: discrete_energy(Z, t, 0.0), cfg, seeds=[11])
        Zopt = opt["waypoints"]
        A_opt = image_coordinate_attribution(e0_mul, c, Zopt, E0_QUAD)[0]
        A_opt_latent = latent_path_attribution(lambda z, c=c: e0_mul(e0_decoder(z, c)), Zopt, E0_QUAD)
        e_geo_cont, e_lin_cont = e0_analytic_energies(c)
        # (3) latent-linear path
        Zl = linear_waypoints(zb, zx, E0_SEGMENTS)
        A_lin = image_coordinate_attribution(e0_mul, c, Zl, E0_QUAD)[0]
        # (4) semantic representation psi(x)=(x1,x2): pullback is identity -> linear path
        sem_term = [MetricTerm("psi", lambda z, c=c: e0_decoder(z, c)[..., :2], 1.0)]
        opt_sem = optimize_paths(zb, zx, lambda Z, t=sem_term: discrete_energy(Z, t, 0.0), cfg, seeds=[12])
        A_sem = image_coordinate_attribution(e0_mul, c, opt_sem["waypoints"], E0_QUAD)[0]
        # (5) hybrid psi + alpha * image metric (illustrates curvature transfer)
        hybrid = {}
        for alpha in protocol.get("e1", "path", "alpha_validation_grid"):
            terms = [sem_term[0], MetricTerm("image", img_term[0].fn, alpha)]
            cfg_h = PathOptConfig(segments=64, iterations=2000, lr=0.003, restarts=1)
            oh = optimize_paths(zb, zx, lambda Z, t=terms: discrete_energy(Z, t, 0.0), cfg_h, seeds=[13])
            hybrid[f"alpha_{alpha}"] = float(
                image_coordinate_attribution(e0_mul, c, oh["waypoints"], 512)[0, 0]
            )
        # (6) additive control on geodesic, linear and random smooth paths
        gen = torch.Generator().manual_seed(int(1000 + c))
        Zr = (Zl + smooth_perturbation(5, E0_SEGMENTS, 2, 0.15, gen, "cpu", DT)).clamp(0, 1)
        Zr[:, 0], Zr[:, -1] = zb, zx
        add_err = 0.0
        for Z in (Zg, Zopt, Zl, Zr):
            A_add = image_coordinate_attribution(e0_add, c, Z, E0_QUAD if Z.shape[1] - 1 != 2048 else 2048)
            add_err = max(add_err, float((A_add - torch.tensor([1.0, 1.0, 0.0], dtype=DT)).abs().max()))
        A_rand = image_coordinate_attribution(e0_mul, c, Zr, E0_QUAD)
        compl_rand = float((A_rand.sum(-1) - 1.0).abs().max())
        fd_err = fd_gradient_check(c)
        row = {
            "c": c,
            "expected_A_u": exp_u,
            "expected_A_v": exp_v,
            "analytic_A_u": au,
            "analytic_A_v": av,
            "analytic_vs_protocol_err": abs(au - exp_u),
            "geodesic_path_quadrature_A_u": float(A_geo_path[0]),
            "optimizer_A_u": float(A_opt[0]),
            "optimizer_A_v": float(A_opt[1]),
            "optimizer_A_x3": float(A_opt[2]),
            "optimizer_A_u_latent_chainrule": float(A_opt_latent["A"][0, 0]),
            "optimizer_abs_err_u": abs(float(A_opt[0]) - exp_u),
            "optimizer_abs_err_v": abs(float(A_opt[1]) - exp_v),
            "optimizer_completeness_err": abs(float(A_opt.sum()) - 1.0),
            "optimizer_energy_discrete": float(opt["energy_final"].min()),
            "analytic_geodesic_energy": e_geo_cont,
            "linear_energy": e_lin_cont,
            "optimizer_best_restart": int(opt["best_restart"][0]),
            "optimizer_rel_tail_change": float(opt["rel_tail_change"][0]),
            "linear_A_u": float(A_lin[0]),
            "linear_abs_err": abs(float(A_lin[0]) - 0.5),
            "semantic_psi_A_u": float(A_sem[0]),
            "semantic_psi_abs_err": abs(float(A_sem[0]) - 0.5),
            "additive_max_abs_err": add_err,
            "random_paths_A_u": [float(v) for v in A_rand[:, 0]],
            "random_paths_completeness_max_err": compl_rand,
            "fd_grad_max_err": fd_err,
            **{f"hybrid_A_u_{k}": v for k, v in hybrid.items()},
        }
        rows.append(row)
        checks += [
            (f"c={c} analytic==protocol", row["analytic_vs_protocol_err"] < tol),
            (f"c={c} geodesic quadrature", abs(row["geodesic_path_quadrature_A_u"] - exp_u) < tol),
            (f"c={c} optimizer A_u", row["optimizer_abs_err_u"] < tol),
            (f"c={c} optimizer A_v", row["optimizer_abs_err_v"] < tol),
            (f"c={c} optimizer completeness", row["optimizer_completeness_err"] < tol),
            (f"c={c} optimizer energy <= linear", row["optimizer_energy_discrete"] <= e_lin_cont + 1e-9),
            (
                f"c={c} optimizer energy ~ analytic geodesic",
                abs(row["optimizer_energy_discrete"] - e_geo_cont) < 1e-3 * e_geo_cont,
            ),
            (f"c={c} latent-linear (0.5,0.5)", row["linear_abs_err"] < tol),
            (f"c={c} semantic psi recovers linear", row["semantic_psi_abs_err"] < tol),
            (f"c={c} additive path invariance", add_err < e0["additive_invariance_tolerance"]),
            (f"c={c} random path completeness", compl_rand < tol),
            (f"c={c} finite-difference gradient", fd_err < 1e-6),
        ]
    passed = all(ok for _, ok in checks)
    return {
        "rows": rows,
        "checks": [{"check": n, "pass": bool(ok)} for n, ok in checks],
        "all_pass": passed,
        "settings": {"segments": E0_SEGMENTS, "iterations": E0_ITERS, "lr": E0_LR, "quadrature": E0_QUAD},
    }
