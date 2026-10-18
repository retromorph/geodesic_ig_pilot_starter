"""Gradients, renderer support, E0 geometry, completeness and attribution equivalences."""

import math

import numpy as np
import pytest
import torch

from geodesic_ig_pilot.geometry import (
    MetricTerm,
    PathOptConfig,
    discrete_energy,
    e0_analytic_attribution,
    e0_arclength,
    e0_decoder,
    linear_waypoints,
    optimize_paths,
    smooth_perturbation,
)
from geodesic_ig_pilot.integration import (
    latent_path_attribution,
    latent_path_pixel_attribution,
    pixel_path_attribution,
    refine_until_complete,
    resample_path,
)
from geodesic_ig_pilot.models import SmallCNN, freeze
from geodesic_ig_pilot.renderer import factor_jacobian_columns, image_metric_diag, region_masks, render
from geodesic_ig_pilot.scores import e0_add, e0_mul, g_nl, g_poly, logit_margin

DT = torch.float64


def fd_grad(fn, z, h=1e-6):
    g = torch.zeros_like(z)
    for j in range(z.shape[-1]):
        e = torch.zeros_like(z)
        e[..., j] = h
        g[..., j] = (fn(z + e) - fn(z - e)) / (2 * h)
    return g


@pytest.mark.parametrize("fn", [lambda z: g_nl(z, 2.2), g_poly])
def test_teacher_gradients_match_finite_differences(fn):
    z = torch.rand(16, 4, dtype=DT, generator=torch.Generator().manual_seed(0)).requires_grad_(True)
    (g,) = torch.autograd.grad(fn(z).sum(), z)
    assert torch.allclose(g, fd_grad(fn, z.detach()), atol=1e-7)


def test_renderer_jacobian_matches_finite_differences():
    z = torch.rand(3, 4, dtype=DT, generator=torch.Generator().manual_seed(1)) * 0.8 + 0.1
    cols = factor_jacobian_columns(z)
    for j in range(4):
        e = torch.zeros_like(z)
        e[:, j] = 1e-6
        fd = (render(z + e) - render(z - e)) / 2e-6
        assert torch.allclose(cols[:, j], fd, atol=1e-6)


def test_renderer_regions_partition_and_disjoint_support():
    masks = region_masks()
    assert torch.equal(masks.sum(0), torch.ones(64, 64, dtype=torch.long))
    z = torch.rand(8, 4, dtype=DT, generator=torch.Generator().manual_seed(2))
    cols = factor_jacobian_columns(z)  # (B, 4, 3, 64, 64)
    for j in range(4):
        outside = ~masks[j]
        assert cols[:, j][:, :, outside].abs().max() == 0.0
        assert cols[:, j][:, :, masks[j]].abs().max() > 0.0
    # off-diagonal pullback metric entries vanish exactly
    flat = cols.flatten(2)
    gram = flat @ flat.transpose(1, 2)
    off = gram - torch.diag_embed(torch.diagonal(gram, dim1=1, dim2=2))
    assert off.abs().max() == 0.0
    assert torch.allclose(torch.diagonal(gram, dim1=1, dim2=2), image_metric_diag(z))


def test_renderer_metric_is_nonconstant_per_factor():
    grid = torch.linspace(0.05, 0.95, 7, dtype=DT)
    for j in range(4):
        z = torch.full((7, 4), 0.5, dtype=DT)
        z[:, j] = grid
        g = image_metric_diag(z)[:, j]
        assert (g.max() / g.min()) > 1.3, f"factor {j} metric nearly constant"


@pytest.mark.parametrize("c,expected", [(0.0, 0.5), (1.0, 0.426372930526833), (4.0, 0.352099127183979)])
def test_e0_analytic_values(c, expected):
    au, av = e0_analytic_attribution(c)
    assert abs(au - expected) < 1e-9 and abs(au + av - 1) < 1e-12
    # closed-form arclength equals numerical quadrature
    from scipy.integrate import quad

    num, _ = quad(lambda r: math.sqrt(1 + 4 * c * c * r * r), 0, 0.7)
    assert abs(float(e0_arclength(0.7, c)) - num) < 1e-10


def test_e0_waypoint_optimizer_recovers_geodesic_attribution():
    c = 1.0
    zb, zx = torch.zeros(1, 2, dtype=DT), torch.ones(1, 2, dtype=DT)
    terms = [MetricTerm("img", lambda z: e0_decoder(z, c), 1.0)]
    cfg = PathOptConfig(segments=64, iterations=2500, lr=0.003, restarts=2)
    res = optimize_paths(zb, zx, lambda Z: discrete_energy(Z, terms, 0.0), cfg, seeds=[5])
    Z = res["waypoints"]
    assert torch.equal(Z[:, 0], zb) and torch.equal(Z[:, -1], zx)
    A = latent_path_attribution(lambda z: e0_mul(e0_decoder(z, c)), Z, 512)["A"][0]
    assert abs(float(A[0]) - 0.426372930526833) < 1e-3
    assert abs(float(A.sum()) - 1.0) < 1e-6


def test_additive_score_path_independent():
    zb, zx = torch.zeros(1, 2, dtype=DT), torch.ones(1, 2, dtype=DT)
    lin = linear_waypoints(zb, zx, 32)
    gen = torch.Generator().manual_seed(3)
    curved = (lin + smooth_perturbation(4, 32, 2, 0.2, gen, "cpu", DT)).clamp(0, 1)
    for Z in (lin, curved):
        A = latent_path_attribution(lambda z: e0_add(e0_decoder(z, 4.0)), Z, 256)["A"]
        assert torch.allclose(A, torch.ones_like(A), atol=1e-12)


def test_completeness_linear_curved_and_pixel_paths():
    torch.manual_seed(0)
    cnn = freeze(SmallCNN().double())
    zb = torch.rand(2, 4, dtype=DT)
    zx = torch.rand(2, 4, dtype=DT)
    F = lambda z: logit_margin(cnn, render(z))  # noqa: E731
    lin = linear_waypoints(zb, zx, 16)
    curved = (lin + smooth_perturbation(2, 16, 4, 0.1, torch.Generator().manual_seed(4), "cpu", DT)).clamp(
        0, 1
    )
    curved[:, 0], curved[:, -1] = zb, zx
    for Z in (lin, curved):
        r = latent_path_attribution(F, Z, 256)
        assert r["residual"].abs().max() < 1e-4 * (1 + (r["f_x"] - r["f_b"]).abs().max())
    X = torch.stack([render(zb), render(zx)], 1)
    rp = pixel_path_attribution(lambda x: logit_margin(cnn, x), X, region_masks(), 256)
    assert rp["residual"].abs().max() < 1e-4 * (1 + (rp["f_x"] - rp["f_b"]).abs().max())
    # unchanged pixels receive exactly zero attribution in straight-line pixel IG
    zx2 = zb.clone()
    zx2[:, 0] = 0.9
    rp2 = pixel_path_attribution(
        lambda x: logit_margin(cnn, x), torch.stack([render(zb), render(zx2)], 1), region_masks(), 64
    )
    assert rp2["A"][:, 1:].abs().max() == 0.0


def test_latent_factor_equals_pixel_group_attribution():
    torch.manual_seed(1)
    cnn = freeze(SmallCNN().double())
    zb, zx = torch.rand(2, 4, dtype=DT), torch.rand(2, 4, dtype=DT)
    lin = linear_waypoints(zb, zx, 16)
    Z = (lin + smooth_perturbation(2, 16, 4, 0.1, torch.Generator().manual_seed(6), "cpu", DT)).clamp(0, 1)
    Z[:, 0], Z[:, -1] = zb, zx
    lat = latent_path_attribution(lambda z: logit_margin(cnn, render(z)), Z, 64)["A"]
    pix_jvp = latent_path_pixel_attribution(
        lambda x: logit_margin(cnn, x), render, Z, region_masks(), 64, True
    )
    assert torch.allclose(lat, pix_jvp, atol=1e-10)
    pix_fd = latent_path_pixel_attribution(
        lambda x: logit_margin(cnn, x), render, Z, region_masks(), 512, False
    )
    assert torch.allclose(lat, pix_fd, atol=2e-3 * (1 + lat.abs().max()))


def test_metric_symmetry_positive_definiteness_and_endpoint_immutability():
    zb = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=DT)
    zx = torch.tensor([[0.8, 0.7, 0.6, 0.5]], dtype=DT)
    terms = [MetricTerm("img", lambda z: render(z).flatten(1), 1.0)]
    Z = linear_waypoints(zb, zx, 8)
    assert torch.equal(Z[:, 0], zb) and torch.equal(Z[:, -1], zx)
    e_fwd = discrete_energy(Z, terms, 1e-3)
    e_rev = discrete_energy(torch.flip(Z, dims=[1]), terms, 1e-3)
    assert torch.allclose(e_fwd, e_rev)
    # epsilon > 0 makes the energy strictly positive even for metric-null displacements
    Zc = torch.zeros(1, 3, 4, dtype=DT)
    Zc[0, 1] = 0.1
    null_terms = [MetricTerm("zero", lambda z: torch.zeros(z.shape[0], 1, dtype=DT), 1.0)]
    assert discrete_energy(Zc, null_terms, 1e-3) > 0
    res = optimize_paths(
        zb,
        zx,
        lambda W: discrete_energy(W, terms, 1e-3),
        PathOptConfig(segments=8, iterations=5, restarts=2),
        seeds=[1],
    )
    W = res["waypoints"]
    assert torch.equal(W[:, 0], zb) and torch.equal(W[:, -1], zx)
    assert torch.isfinite(res["energy_final"]).all()
    assert (W >= 0).all() and (W <= 1).all()


def test_refinement_rule_only_uses_completeness():
    calls = []

    def compute(N, idx):
        calls.append((N, idx.tolist()))
        resid = torch.tensor([1e-6 if i == 0 else 1.0 / N for i in idx.tolist()])
        return {"A": torch.zeros(len(idx), 2), "residual": resid}

    res, n_used = refine_until_complete(compute, torch.ones(2), (64, 128, 256, 512), 0.005)
    assert n_used.tolist() == [64, 256]
    assert calls[0] == (64, [0, 1]) and calls[1] == (128, [1])


def test_resample_path_hits_waypoints():
    Z = torch.rand(2, 5, 3, dtype=DT)
    pts = resample_path(Z, 20)
    assert torch.allclose(pts[:, ::5], Z)
    with pytest.raises(ValueError):
        resample_path(Z, 7)


def test_energy_of_geodesic_not_higher_than_linear_e0():
    from geodesic_ig_pilot.geometry import e0_analytic_energies

    for c in (0.0, 1.0, 4.0):
        geo, lin = e0_analytic_energies(c)
        assert geo <= lin + 1e-12
    assert np.isclose(e0_analytic_energies(0.0)[0], 1.0)
