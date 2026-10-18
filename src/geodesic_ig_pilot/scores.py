"""Analytic scores (E0, E1 teachers) and the learned-classifier logit margin."""

from __future__ import annotations

import math

import torch


# --------------------------------------------------------------------------- E0
def e0_mul(x: torch.Tensor) -> torch.Tensor:
    """f(x) = x1 * x2 on image coordinates (..., 3)."""
    return x[..., 0] * x[..., 1]


def e0_add(x: torch.Tensor) -> torch.Tensor:
    """Additive negative control f(x) = x1 + x2."""
    return x[..., 0] + x[..., 1]


# --------------------------------------------------------------------------- E1 teachers
def g_poly(z: torch.Tensor) -> torch.Tensor:
    a, b, c = z[..., 0] - 0.5, z[..., 1] - 0.5, z[..., 2] - 0.5
    return 2 * a + 2 * b + 1.5 * c + 4 * a * b + 3 * b * c


def g_nl_raw(z: torch.Tensor) -> torch.Tensor:
    a, b, c = z[..., 0], z[..., 1], z[..., 2]
    return 1.5 * a + 1.2 * b + 0.8 * c + 2 * torch.tanh(3 * (a * b - 0.25)) + 1.5 * torch.sin(math.pi * b * c)


def g_nl(z: torch.Tensor, C: float) -> torch.Tensor:
    """Primary nonlinear teacher; 3*g_nl is the exact posterior log-odds of the labels."""
    return g_nl_raw(z) - C


def oracle_log_odds(z: torch.Tensor, C: float) -> torch.Tensor:
    return 3.0 * g_nl(z, C)


def g_short(z: torch.Tensor, C: float) -> torch.Tensor:
    """Diagnostic shortcut stress target (nuisance-coupled); never a primary target."""
    return g_nl(z, C) + 3 * (z[..., 3] - 0.5) * (z[..., 1] - 0.5)


# --------------------------------------------------------------------------- learned model
def logit_margin(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Primary E1 score f_CNN(x) = logit_1(x) - logit_0(x)."""
    logits = model(x)
    return logits[:, 1] - logits[:, 0]
