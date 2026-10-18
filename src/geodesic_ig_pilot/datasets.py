"""E1 scene generation, teacher calibration, stochastic labels and the frozen pair manifest."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import Protocol, rng_for
from .evaluations import interaction_index
from .scores import g_nl, g_nl_raw

FACTOR_NAMES = ("a", "b", "c", "n")
PREDICTIVE = (0, 1, 2)
NUISANCE = 3


def _np_g_nl(C: float):
    def g(z: np.ndarray) -> np.ndarray:
        return g_nl(torch.as_tensor(z, dtype=torch.float64), C).numpy()

    return g


def calibrate_C(protocol: Protocol) -> dict[str, Any]:
    """C = median of the raw nonlinear expression on the fixed train-only calibration draw."""
    n = protocol.get("e1", "calibration_draw_size")
    rng = rng_for("calibration", protocol.get("e1", "generator_seed"))
    z = rng.uniform(0.0, 1.0, size=(n, 4))
    raw = g_nl_raw(torch.as_tensor(z, dtype=torch.float64)).numpy()
    C = float(np.median(raw))
    return {
        "C": C,
        "calibration_draw_size": n,
        "calibration_stream": "calibration",
        "calibration_z_sha256": hashlib.sha256(z.tobytes()).hexdigest(),
    }


def make_scenes(protocol: Protocol, stream: str, n: int, C: float, prefix: str) -> pd.DataFrame:
    """Uniform factor scenes with stochastic labels y ~ Bernoulli(sigmoid(3 g_NL(z)))."""
    rng = rng_for(stream, protocol.get("e1", "generator_seed"))
    z = rng.uniform(0.0, 1.0, size=(n, 4))
    g = _np_g_nl(C)(z)
    p = 1.0 / (1.0 + np.exp(-3.0 * g))
    y = (rng.uniform(size=n) < p).astype(np.int64)
    df = pd.DataFrame(z, columns=list(FACTOR_NAMES))
    df.insert(0, "scene_id", [f"{prefix}-{i:06d}" for i in range(n)])
    df["g_nl"] = g
    df["p_teacher"] = p
    df["y"] = y
    df["stream"] = stream
    return df


def z_hashes(z: np.ndarray) -> set[str]:
    return {hashlib.sha256(np.ascontiguousarray(row, dtype=np.float64).tobytes()).hexdigest() for row in z}


def _pair_quotas(protocol: Protocol, split: str) -> dict[tuple[str, bool], int]:
    if split == "validation":
        total = protocol.get("e1", "validation_pairs")
        per = {"strong": total // 2, "weak": total - total // 2}
    else:
        per = {
            "strong": protocol.get("e1", "test_pairs_strong"),
            "weak": protocol.get("e1", "test_pairs_weak"),
        }
    frac = protocol.get("e1", "nuisance_changed_fraction_target")
    quotas = {}
    for stratum, k in per.items():
        n_changed = int(round(k * frac))
        quotas[(stratum, True)] = n_changed
        quotas[(stratum, False)] = k - n_changed
    return quotas


def generate_pairs(
    protocol: Protocol,
    split: str,
    C: float,
    max_candidates: int,
    quotas: dict[tuple[str, bool], int] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Deterministic, quota-filling contrastive pair generation for one split.

    Candidate rule (fixed before any attribution): z_b ~ U[0,1]^4; choose k in {2,3} predictive
    factors uniformly, redraw them ~U[0,1]; with prob. 1/2 also redraw the nuisance. Reject if any
    changed factor moves by < min_factor_delta, or if the oracle teacher sign does not flip.
    Stratify by the oracle interaction index eta; fill (stratum, nuisance_changed) quotas
    first-come in stream order. Middle-eta candidates are discarded (counted). Classifier
    outputs are never consulted.
    """
    e1 = protocol.get("e1")
    stream = "pairs_validation" if split == "validation" else "pairs_test"
    rng = rng_for(stream, e1["generator_seed"])
    g = _np_g_nl(C)
    quotas = quotas or _pair_quotas(protocol, split)
    filled = {k: 0 for k in quotas}
    rejects = {"small_delta": 0, "same_teacher_class": 0, "middle_eta": 0, "quota_full": 0}
    strong_min, weak_max = e1["interaction_index"]["strong_min"], e1["interaction_index"]["weak_max"]
    min_delta = e1["min_factor_delta"]
    rows = []
    n_drawn = 0
    prefix = "val" if split == "validation" else "test"
    while n_drawn < max_candidates and any(filled[k] < quotas[k] for k in quotas):
        i = n_drawn
        n_drawn += 1
        zb = rng.uniform(0.0, 1.0, size=4)
        k = int(rng.integers(2, 4))
        which = np.sort(rng.choice(3, size=k, replace=False))
        new_vals = rng.uniform(0.0, 1.0, size=k)
        change_n = bool(rng.uniform() < 0.5)
        new_n = rng.uniform()
        zx = zb.copy()
        zx[which] = new_vals
        if change_n:
            zx[NUISANCE] = new_n
        changed = [int(j) for j in which] + ([NUISANCE] if change_n else [])
        if np.any(np.abs(zx[changed] - zb[changed]) < min_delta):
            rejects["small_delta"] += 1
            continue
        gb, gx = float(g(zb[None])[0]), float(g(zx[None])[0])
        if (gb > 0) == (gx > 0):
            rejects["same_teacher_class"] += 1
            continue
        eta = interaction_index(g, zb, zx, [int(j) for j in which])
        if eta >= strong_min:
            stratum = "strong"
        elif eta <= weak_max:
            stratum = "weak"
        else:
            rejects["middle_eta"] += 1
            continue
        key = (stratum, change_n)
        if filled[key] >= quotas[key]:
            rejects["quota_full"] += 1
            continue
        filled[key] += 1
        sid = f"{prefix}-{i:06d}"
        rows.append(
            {
                "scene_id": sid,
                "pair_id": sid,
                "baseline_scene_id": f"{sid}-b",
                "target_scene_id": f"{sid}-x",
                "split": split,
                "candidate_index": i,
                "stratum": stratum,
                "nuisance_changed": change_n,
                "changed_factors": ",".join(FACTOR_NAMES[j] for j in changed),
                "changed_idx": ",".join(str(j) for j in changed),
                "m": len(changed),
                "eta": eta,
                **{f"zb_{f}": float(zb[j]) for j, f in enumerate(FACTOR_NAMES)},
                **{f"zx_{f}": float(zx[j]) for j, f in enumerate(FACTOR_NAMES)},
                "g_nl_b": gb,
                "g_nl_x": gx,
                "baseline_label_oracle": int(gb > 0),
                "input_label_oracle": int(gx > 0),
            }
        )
    df = pd.DataFrame(rows)
    info = {
        "split": split,
        "stream": stream,
        "candidates_drawn": n_drawn,
        "max_candidates": max_candidates,
        "quotas": {f"{k[0]}|nuisance_changed={k[1]}": v for k, v in quotas.items()},
        "filled": {f"{k[0]}|nuisance_changed={k[1]}": v for k, v in filled.items()},
        "all_quotas_filled": all(filled[k] >= quotas[k] for k in quotas),
        "rejections": rejects,
    }
    return df, info


def pair_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    zb = df[[f"zb_{f}" for f in FACTOR_NAMES]].to_numpy(np.float64)
    zx = df[[f"zx_{f}" for f in FACTOR_NAMES]].to_numpy(np.float64)
    changed = [[int(c) for c in s.split(",")] for s in df["changed_idx"]]
    return zb, zx, changed


def manifest_sha256(df: pd.DataFrame) -> str:
    """Content hash of a manifest independent of parquet encoding."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()).hexdigest()


def check_no_leakage(frames: dict[str, pd.DataFrame], pair_frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Scene-ID and exact factor-vector disjointness across train/encoder/val/test pools."""
    ids: dict[str, set[str]] = {}
    hashes: dict[str, set[str]] = {}
    for name, df in frames.items():
        ids[name] = set(df["scene_id"])
        hashes[name] = z_hashes(df[list(FACTOR_NAMES)].to_numpy())
    for name, df in pair_frames.items():
        ids[name] = set(df["baseline_scene_id"]) | set(df["target_scene_id"]) | set(df["scene_id"])
        zb, zx, _ = pair_arrays(df)
        hashes[name] = z_hashes(np.concatenate([zb, zx]))
    overlaps = {}
    names = sorted(ids)
    ok = True
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            n_id = len(ids[a] & ids[b])
            n_z = len(hashes[a] & hashes[b])
            overlaps[f"{a}|{b}"] = {"scene_id_overlap": n_id, "z_vector_overlap": n_z}
            ok &= n_id == 0 and n_z == 0
    return {"no_leakage": bool(ok), "overlaps": overlaps}
