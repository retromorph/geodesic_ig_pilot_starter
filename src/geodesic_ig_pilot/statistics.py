"""Scene-grouped paired bootstrap and mechanical gate decisions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def conservative_pair(e_m4: float, ok_m4: bool, e_b: float, ok_b: bool) -> tuple[float, float]:
    """Penalise invalid samples so they can never create an advantage for M4.

    Invalid M4: its error is raised to at least the comparator's. Invalid comparator with a
    non-finite error: tie (no advantage for M4). Finite invalid comparator errors are kept.
    """
    if not ok_m4 or not np.isfinite(e_m4):
        e_m4 = max(e_m4 if np.isfinite(e_m4) else -np.inf, e_b if np.isfinite(e_b) else 0.0)
        if not np.isfinite(e_m4):
            e_m4 = 0.0
    if not np.isfinite(e_b):
        e_b = e_m4
    return float(e_m4), float(e_b)


def paired_scene_table(
    rows: pd.DataFrame, m4: str, comparator: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Per-scene seed-averaged errors for (M4, B) with conservative invalid handling.

    rows: one row per (scene_id, seed, method) with columns E_coal, valid, model_flat, stratum.
    Model-flat scene-seeds are excluded for *both* methods and counted.
    """
    counts = {"scene_seeds": 0, "model_flat": 0, "m4_invalid": 0, "b_invalid": 0}
    recs = []
    sub = rows[rows["method"].isin([m4, comparator])]
    for (scene, seed), grp in sub.groupby(["scene_id", "seed"], sort=True):
        counts["scene_seeds"] += 1
        g = grp.set_index("method")
        if bool(g["model_flat"].iloc[0]):
            counts["model_flat"] += 1
            continue
        if m4 not in g.index or comparator not in g.index:
            raise ValueError(f"missing method rows for scene {scene} seed {seed}")
        ok4, okb = bool(g.loc[m4, "valid"]), bool(g.loc[comparator, "valid"])
        counts["m4_invalid"] += int(not ok4)
        counts["b_invalid"] += int(not okb)
        e4, eb = conservative_pair(float(g.loc[m4, "E_coal"]), ok4, float(g.loc[comparator, "E_coal"]), okb)
        recs.append({"scene_id": scene, "seed": seed, "stratum": g["stratum"].iloc[0], "E_m4": e4, "E_b": eb})
    per_seed = pd.DataFrame(recs)
    table = per_seed.groupby(["scene_id", "stratum"], as_index=False)[["E_m4", "E_b"]].mean()
    return table, per_seed, counts


def relative_reduction(e_m4: np.ndarray, e_b: np.ndarray) -> float:
    return float(1.0 - np.mean(e_m4) / np.mean(e_b))


def equal_weight_r(strong: pd.DataFrame, weak: pd.DataFrame) -> float:
    num = 0.5 * strong["E_m4"].mean() + 0.5 * weak["E_m4"].mean()
    den = 0.5 * strong["E_b"].mean() + 0.5 * weak["E_b"].mean()
    return float(1.0 - num / den)


def bootstrap_reductions(table: pd.DataFrame, n_boot: int, seed: int, level: float = 0.95) -> dict[str, Any]:
    """Paired, scene-resampled bootstrap of r_strong, r_weak and equal-weight r_all.

    Scenes are resampled with replacement *within* each stratum (keeping the equal-weight
    union exact); M4 and B errors of a scene always travel together (paired), and each
    scene's value is already the mean over the two classifier seeds.
    """
    rng = np.random.default_rng(seed)
    s = table[table["stratum"] == "strong"].reset_index(drop=True)
    w = table[table["stratum"] == "weak"].reset_index(drop=True)
    s4, sb, w4, wb = (s["E_m4"].to_numpy(), s["E_b"].to_numpy(), w["E_m4"].to_numpy(), w["E_b"].to_numpy())
    rs, rw, ra = np.empty(n_boot), np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        si = rng.integers(0, len(s), len(s))
        wi = rng.integers(0, len(w), len(w))
        rs[i] = 1 - s4[si].mean() / sb[si].mean()
        rw[i] = 1 - w4[wi].mean() / wb[wi].mean()
        ra[i] = 1 - (0.5 * s4[si].mean() + 0.5 * w4[wi].mean()) / (0.5 * sb[si].mean() + 0.5 * wb[wi].mean())
    lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2

    def ci(x):
        return [float(np.percentile(x, lo)), float(np.percentile(x, hi))]

    return {
        "r_strong": relative_reduction(s4, sb),
        "r_weak": relative_reduction(w4, wb),
        "r_all": equal_weight_r(s, w),
        "ci_strong": ci(rs),
        "ci_weak": ci(rw),
        "ci_all": ci(ra),
        "n_strong": len(s),
        "n_weak": len(w),
        "mean_E_m4_strong": float(s4.mean()),
        "mean_E_b_strong": float(sb.mean()),
        "mean_E_m4_weak": float(w4.mean()),
        "mean_E_b_weak": float(wb.mean()),
        "mean_E_m4_all": float(0.5 * s4.mean() + 0.5 * w4.mean()),
        "mean_E_b_all": float(0.5 * sb.mean() + 0.5 * wb.mean()),
        "bootstrap_seed": seed,
        "bootstrap_samples": n_boot,
    }


def per_seed_strong_reduction(per_seed: pd.DataFrame) -> dict[int, float]:
    out = {}
    strong = per_seed[per_seed["stratum"] == "strong"]
    for seed, g in strong.groupby("seed"):
        out[int(seed)] = relative_reduction(g["E_m4"].to_numpy(), g["E_b"].to_numpy())
    return out


def gate_b_decision(
    boot: dict[str, Any],
    per_seed: dict[int, float],
    thresholds: dict[str, Any],
    negligible: float,
    prerequisites: dict[str, bool],
) -> dict[str, Any]:
    """Mechanical Gate B from saved numbers. Returns status and per-criterion reasons."""
    criteria = []

    def add(name, value, rule, ok):
        criteria.append({"criterion": name, "value": value, "rule": rule, "pass": bool(ok)})

    uninformative = boot["mean_E_b_strong"] < negligible or boot["mean_E_b_all"] < negligible
    add(
        "r_strong",
        boot["r_strong"],
        f">= {thresholds['strong_mean_relative_reduction_min']}",
        boot["r_strong"] >= thresholds["strong_mean_relative_reduction_min"],
    )
    add(
        "r_strong_ci_lower",
        boot["ci_strong"][0],
        f"> {thresholds['strong_bootstrap_lower_bound_gt']}",
        boot["ci_strong"][0] > thresholds["strong_bootstrap_lower_bound_gt"],
    )
    add(
        "r_all",
        boot["r_all"],
        f">= {thresholds['all_mean_relative_reduction_min']}",
        boot["r_all"] >= thresholds["all_mean_relative_reduction_min"],
    )
    add(
        "r_all_ci_lower",
        boot["ci_all"][0],
        f"> {thresholds['all_bootstrap_lower_bound_gt']}",
        boot["ci_all"][0] > thresholds["all_bootstrap_lower_bound_gt"],
    )
    add(
        "r_weak",
        boot["r_weak"],
        f">= {thresholds['weak_relative_reduction_min']}",
        boot["r_weak"] >= thresholds["weak_relative_reduction_min"],
    )
    for seed, r in sorted(per_seed.items()):
        add(f"seed_{seed}_strong_reduction", r, "> 0", r > 0)
    for name, ok in prerequisites.items():
        add(name, ok, "== True", ok)
    if uninformative:
        status = "INCONCLUSIVE"
    elif all(c["pass"] for c in criteria):
        status = "PASS"
    else:
        status = "FAIL"
    return {"status": status, "comparator_mean_error_negligible": uninformative, "criteria": criteria}
