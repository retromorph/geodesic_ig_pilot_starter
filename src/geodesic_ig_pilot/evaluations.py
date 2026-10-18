"""Exact factor coalitions, coalition reconstruction error, Shapley reference, interaction index."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence

import numpy as np

DENOM_EPS = 1e-8


def coalition_masks(m: int) -> np.ndarray:
    """All 2^m subsets as a boolean (2^m, m) array in bitmask order (row 0 = empty set)."""
    return np.array([[(s >> j) & 1 for j in range(m)] for s in range(2**m)], dtype=bool)


def hybrid_latents(zb: np.ndarray, zx: np.ndarray, changed: Sequence[int]) -> np.ndarray:
    """z_S for every subset S of `changed` (bitmask order): (2^m, d). z_S takes x-values on S."""
    masks = coalition_masks(len(changed))
    out = np.repeat(zb[None, :].astype(np.float64), len(masks), axis=0)
    for row, mask in enumerate(masks):
        for j, on in zip(changed, mask):
            if on:
                out[row, j] = zx[j]
    return out


def coalition_error(values: np.ndarray, A_changed: np.ndarray) -> dict[str, float]:
    """Normalised full-coalition squared reconstruction error.

    values: f(D(z_S)) for all subsets in bitmask order (2^m,).
    A_changed: attributions of the m changed factors in the same factor order.
    """
    m = len(A_changed)
    masks = coalition_masks(m)
    delta = values - values[0]
    R = float(values.max() - values.min())
    pred = masks.astype(np.float64) @ A_changed
    resid = delta - pred
    err = float(np.mean((resid / (R + DENOM_EPS)) ** 2))
    return {"E_coal": err, "E_coal_abs": float(np.mean(resid**2)), "R": R}


def shapley_values(values: np.ndarray) -> np.ndarray:
    """Exact Shapley values of the game v(S) = values[S] - values[empty]."""
    m = int(round(math.log2(len(values))))
    v = values - values[0]
    phi = np.zeros(m)
    for j in range(m):
        for s in range(2**m):
            if (s >> j) & 1:
                continue
            size = bin(s).count("1")
            w = math.factorial(size) * math.factorial(m - size - 1) / math.factorial(m)
            phi[j] += w * (v[s | (1 << j)] - v[s])
    return phi


def interaction_index(
    g: Callable[[np.ndarray], np.ndarray], zb: np.ndarray, zx: np.ndarray, changed_predictive: Sequence[int]
) -> float:
    """eta = sum_{j<k} |I_jk| / (max_S g - min_S g + 1e-8) over changed predictive factors."""
    hyb = hybrid_latents(zb, zx, changed_predictive)
    vals = g(hyb)
    m = len(changed_predictive)
    num = 0.0
    for j, k in itertools.combinations(range(m), 2):
        sj, sk = 1 << j, 1 << k
        num += abs(vals[sj | sk] - vals[sj] - vals[sk] + vals[0])
    return float(num / (vals.max() - vals.min() + DENOM_EPS))


def single_factor_sign_accuracy(values: np.ndarray, A_changed: np.ndarray) -> float:
    """Fraction of changed factors whose attribution sign matches its single-factor effect."""
    m = len(A_changed)
    single = np.array([values[1 << j] - values[0] for j in range(m)])
    return float(np.mean(np.sign(single) == np.sign(A_changed)))
