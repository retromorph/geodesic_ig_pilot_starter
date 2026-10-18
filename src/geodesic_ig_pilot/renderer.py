"""Deterministic differentiable 4-factor renderer D: [0,1]^4 -> [0,1]^(3x64x64).

Factors z = (a, b, c, n). The image is partitioned into four *hard*, fixed regions:
three 28x28 part tiles (A, B, C) and the background (everything else). Every pixel of
tile A depends only on `a`, tile B only on `b`, tile C only on `c`, and background pixels
only on `n`. There is no global post-processing, so the renderer Jacobian has disjoint
factor supports and latent-factor path attributions equal pixel-group attributions.

Each factor induces a nonlinear image change with a non-constant pullback metric:
  a: radius of an antialiased red disk (r = 3 + 8a)                -> morphology
  b: colour of a soft square along a nonlinear RGB curve            -> colour
  c: spatial frequency of stripes in a disk, f = 1 + 2c + 2c^2      -> texture
  n: background colour along a nonlinear curve with fixed shading   -> nuisance
"""

from __future__ import annotations

import math

import torch

H = W = 64
TILE = 28
TILE_ORIGINS = {"a": (4, 4), "b": (4, 34), "c": (34, 18)}  # (row0, col0)
FACTORS = ("a", "b", "c", "n")
BACKDROP = (0.5, 0.5, 0.5)
EDGE_TAU = 0.5  # antialiasing softness in pixels


def region_masks(device: torch.device | str = "cpu") -> torch.Tensor:
    """Boolean (4, 64, 64) partition masks in factor order (a, b, c, n)."""
    m = torch.zeros(4, H, W, dtype=torch.bool, device=device)
    for i, k in enumerate(("a", "b", "c")):
        r0, c0 = TILE_ORIGINS[k]
        m[i, r0 : r0 + TILE, c0 : c0 + TILE] = True
    m[3] = ~(m[0] | m[1] | m[2])
    return m


def _tile_grid(device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    ys = torch.arange(TILE, device=device, dtype=dtype) + 0.5
    yy, xx = torch.meshgrid(ys, ys, indexing="ij")
    return yy - TILE / 2, xx - TILE / 2


def color_b(b: torch.Tensor) -> torch.Tensor:
    """Nonlinear colour curve for part B; b shape (...,) -> (..., 3)."""
    return torch.stack([0.15 + 0.7 * b**2, 0.75 - 0.55 * b, 0.25 + 0.6 * torch.sin(math.pi * b) ** 2], dim=-1)


def color_n(n: torch.Tensor) -> torch.Tensor:
    return torch.stack([0.15 + 0.6 * n, 0.35 + 0.3 * n**2, 0.65 - 0.45 * n], dim=-1)


def stripe_freq(c: torch.Tensor) -> torch.Tensor:
    return 1.0 + 2.0 * c + 2.0 * c**2


def render(z: torch.Tensor) -> torch.Tensor:
    """Render a batch of factor vectors z (B, 4) into images (B, 3, 64, 64).

    Differentiable in z. Values outside [0,1] are allowed mathematically (smooth
    extension), but all experiment paths are constrained to the unit box.
    """
    if z.ndim != 2 or z.shape[-1] != 4:
        raise ValueError(f"expected z of shape (B,4), got {tuple(z.shape)}")
    B, dev, dt = z.shape[0], z.device, z.dtype
    a, b, c, n = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
    yy, xx = _tile_grid(dev, dt)
    dist = torch.sqrt(yy**2 + xx**2)
    backdrop = torch.tensor(BACKDROP, device=dev, dtype=dt).view(1, 3, 1, 1)

    # Part A: disk with radius 3 + 8a.
    r = (3.0 + 8.0 * a).view(B, 1, 1)
    cov_a = torch.sigmoid((r - dist) / EDGE_TAU).unsqueeze(1)
    red = torch.tensor((0.9, 0.2, 0.2), device=dev, dtype=dt).view(1, 3, 1, 1)
    tile_a = backdrop + cov_a * (red - backdrop)

    # Part B: soft square, colour along nonlinear curve.
    half = 9.0
    cov_sq = (torch.sigmoid((half - yy.abs()) / EDGE_TAU) * torch.sigmoid((half - xx.abs()) / EDGE_TAU)).view(
        1, 1, TILE, TILE
    )
    col_b = color_b(b).view(B, 3, 1, 1)
    tile_b = backdrop + cov_sq * (col_b - backdrop)

    # Part C: disk with vertical stripes of frequency f(c) cycles per disk diameter.
    radius_c = 11.0
    cov_c = torch.sigmoid((radius_c - dist) / EDGE_TAU).view(1, 1, TILE, TILE)
    f = stripe_freq(c).view(B, 1, 1)
    s = 0.5 + 0.5 * torch.sin(2 * math.pi * f * xx.view(1, TILE, TILE) / (2 * radius_c))
    yellow = torch.tensor((0.95, 0.85, 0.2), device=dev, dtype=dt).view(1, 3, 1, 1)
    blue = torch.tensor((0.1, 0.2, 0.6), device=dev, dtype=dt).view(1, 3, 1, 1)
    stripes = s.unsqueeze(1) * yellow + (1 - s.unsqueeze(1)) * blue
    tile_c = backdrop + cov_c * (stripes - backdrop)

    # Background: colour(n) times fixed vertical shading.
    shade = 0.8 + 0.2 * torch.arange(H, device=dev, dtype=dt) / (H - 1)
    bg = color_n(n).view(B, 3, 1, 1) * shade.view(1, 1, H, 1)
    img = bg.expand(B, 3, H, W).clone()
    for key, tile in (("a", tile_a), ("b", tile_b), ("c", tile_c)):
        r0, c0 = TILE_ORIGINS[key]
        img[:, :, r0 : r0 + TILE, c0 : c0 + TILE] = tile.expand(B, 3, TILE, TILE)
    return img


def factor_jacobian_columns(z: torch.Tensor) -> torch.Tensor:
    """dD/dz_j for each factor via forward-mode AD: (B, 4, 3, 64, 64)."""
    cols = []
    for j in range(4):
        v = torch.zeros_like(z)
        v[:, j] = 1.0
        _, jv = torch.func.jvp(render, (z,), (v,))
        cols.append(jv)
    return torch.stack(cols, dim=1)


def image_metric_diag(z: torch.Tensor) -> torch.Tensor:
    """Diagonal of J_D^T J_D (exact: off-diagonals vanish by disjoint support), shape (B, 4)."""
    cols = factor_jacobian_columns(z)
    return (cols**2).flatten(2).sum(-1)
