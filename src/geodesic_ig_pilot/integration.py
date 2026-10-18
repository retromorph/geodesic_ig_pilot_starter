"""Signed path-integrated attributions with segment-displacement midpoint quadrature.

For a piecewise-linear latent path with waypoints Z (P, K+1, d) and N quadrature intervals
(N a multiple of K so every interval lies inside one segment):

    A_j = sum_i  d_j F(z(m_i)) * (z_j(t_{i+1}) - z_j(t_i)),   m_i = (t_i + t_{i+1}) / 2.

For pixel paths the same rule is applied per pixel (displacement of each interval, never
the whole-path average gradient times x - b), then summed within region masks.
Completeness is always checked against F at the *actual* path endpoints.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

ScoreFn = Callable[[torch.Tensor], torch.Tensor]


def resample_path(Z: torch.Tensor, N: int) -> torch.Tensor:
    """Points z(t_i), i=0..N, of the piecewise-linear path through waypoints Z: (P, N+1, d)."""
    P, K1, d = Z.shape
    K = K1 - 1
    if N % K != 0:
        raise ValueError(f"quadrature N={N} must be a multiple of segments K={K}")
    r = N // K
    frac = torch.arange(r, device=Z.device, dtype=Z.dtype) / r  # (r,)
    seg_start = Z[:, :-1].unsqueeze(2)  # (P,K,1,d)
    seg_delta = (Z[:, 1:] - Z[:, :-1]).unsqueeze(2)
    pts = (seg_start + frac.view(1, 1, r, 1) * seg_delta).reshape(P, K * r, d)
    return torch.cat([pts, Z[:, -1:]], dim=1)


def _grad_batched(fn: ScoreFn, pts: torch.Tensor, max_batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Values and input-gradients of a scalar-per-sample score over a flat batch of points."""
    vals, grads = [], []
    for s in range(0, pts.shape[0], max_batch):
        x = pts[s : s + max_batch].detach().clone().requires_grad_(True)
        with torch.enable_grad():
            y = fn(x)
            (g,) = torch.autograd.grad(y.sum(), x)
        vals.append(y.detach())
        grads.append(g.detach())
    return torch.cat(vals), torch.cat(grads)


def eval_batched(fn: ScoreFn, pts: torch.Tensor, max_batch: int) -> torch.Tensor:
    out = []
    with torch.no_grad():
        for s in range(0, pts.shape[0], max_batch):
            out.append(fn(pts[s : s + max_batch]))
    return torch.cat(out)


def latent_path_attribution(F: ScoreFn, Z: torch.Tensor, N: int, max_batch: int = 2048) -> dict:
    """Factor attributions of F along latent paths Z (P, K+1, d)."""
    P, _, d = Z.shape
    pts = resample_path(Z, N)  # (P, N+1, d)
    mids = 0.5 * (pts[:, 1:] + pts[:, :-1])
    disp = pts[:, 1:] - pts[:, :-1]
    _, g = _grad_batched(F, mids.reshape(P * N, d), max_batch)
    A = (g.view(P, N, d) * disp).sum(1)
    ends = eval_batched(F, torch.stack([Z[:, 0], Z[:, -1]], 1).reshape(2 * P, d), max_batch).view(P, 2)
    target = ends[:, 1] - ends[:, 0]
    return {"A": A, "f_b": ends[:, 0], "f_x": ends[:, 1], "residual": A.sum(-1) - target}


def pixel_path_attribution(
    f: ScoreFn,
    X: torch.Tensor,
    masks: torch.Tensor,
    N: int,
    max_batch: int = 1024,
) -> dict:
    """Region-grouped attributions along piecewise-linear *pixel* paths.

    X: (P, K+1, C, H, W) pixel waypoints (K=1 gives straight-line pixel IG).
    masks: (G, H, W) boolean partition of pixels into attribution groups.
    """
    P, K1 = X.shape[:2]
    img_shape = X.shape[2:]
    flat = X.reshape(P, K1, -1)
    pts = resample_path(flat, N)  # (P, N+1, C*H*W)
    mids = 0.5 * (pts[:, 1:] + pts[:, :-1])
    disp = pts[:, 1:] - pts[:, :-1]
    groups = masks.reshape(masks.shape[0], -1).to(X.dtype)  # (G, HW)
    A = torch.zeros(P, masks.shape[0], device=X.device, dtype=X.dtype)
    for p in range(P):
        _, g = _grad_batched(lambda x: f(x.view(-1, *img_shape)), mids[p], max_batch)
        contrib = (g * disp[p]).sum(0).view(img_shape)  # per-pixel attribution summed over intervals
        A[p] = contrib.sum(0).reshape(-1) @ groups.T  # sum channels, then pixels within each group
    ends = eval_batched(
        lambda x: f(x.view(-1, *img_shape)),
        torch.stack([X[:, 0], X[:, -1]], 1).reshape(2 * P, *img_shape),
        max_batch,
    ).view(P, 2)
    target = ends[:, 1] - ends[:, 0]
    return {"A": A, "f_b": ends[:, 0], "f_x": ends[:, 1], "residual": A.sum(-1) - target}


def latent_path_pixel_attribution(
    f: ScoreFn,
    D: Callable[[torch.Tensor], torch.Tensor],
    Z: torch.Tensor,
    masks: torch.Tensor,
    N: int,
    use_jvp: bool,
    max_batch: int = 1024,
) -> torch.Tensor:
    """Pixel-space attribution of the decoded path x(t)=D(z(t)), grouped by masks: (P, G).

    use_jvp=True uses displacement J_D(m_i) dz_i (equals latent attribution exactly for a
    disjoint-support decoder); use_jvp=False uses finite displacement D(z_{i+1}) - D(z_i)
    (converges to the same integral as N grows).
    """
    P, _, d = Z.shape
    pts = resample_path(Z, N)
    mids = 0.5 * (pts[:, 1:] + pts[:, :-1])
    dz = pts[:, 1:] - pts[:, :-1]
    groups = masks.reshape(masks.shape[0], -1).to(Z.dtype)
    out = torch.zeros(P, masks.shape[0], device=Z.device, dtype=Z.dtype)
    for p in range(P):
        xm = D(mids[p])
        if use_jvp:
            _, disp = torch.func.jvp(D, (mids[p],), (dz[p],))
        else:
            disp = D(pts[p, 1:]) - D(pts[p, :-1])
        _, g = _grad_batched(f, xm, max_batch)
        contrib = (g * disp).sum(0).sum(0)  # sum over intervals and channels -> (H, W)
        out[p] = contrib.reshape(-1) @ groups.T
    return out


def refine_until_complete(
    compute: Callable[[int, torch.Tensor], dict],
    denom: torch.Tensor,
    steps: tuple[int, ...],
    target_rel: float,
) -> tuple[dict, torch.Tensor]:
    """Uniform doubling refinement applied identically to every method.

    `compute(N, idx)` returns a dict with 'A' (len(idx), d) and 'residual' (len(idx),) for the
    paths `idx`. Every path starts at steps[0]; paths whose |residual|/denom exceeds
    `target_rel` are recomputed at the next N. The rule depends only on completeness error,
    never on explanation quality. Returns merged results and the N used per path.
    """
    P = denom.shape[0]
    idx = torch.arange(P)
    res: dict | None = None
    n_used = torch.full((P,), steps[0], dtype=torch.long)
    for N in steps:
        r = compute(N, idx)
        if res is None:
            res = {k: v.clone() for k, v in r.items()}
        else:
            for k in res:
                res[k][idx.to(res[k].device)] = r[k]
        n_used[idx] = N
        rel = r["residual"].abs().cpu() / denom[idx].cpu()
        idx = idx[rel > target_rel]
        if idx.numel() == 0:
            break
    return res, n_used
