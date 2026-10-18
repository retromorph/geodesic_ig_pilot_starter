"""Aggregation (gates, bootstrap, tables, figures) and deterministic report generation.

Every gate status is computed mechanically from saved files. The only hand-written text
in FINAL_REPORT.md is `reports/interpretation.md`, included verbatim and clearly labelled.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import sha256_file
from .statistics import (
    bootstrap_reductions,
    gate_b_decision,
    paired_scene_table,
    per_seed_strong_reduction,
)
from .tracking import atomic_write_json, atomic_write_parquet, provenance, read_status, utcnow

LATENT_METHODS = ("M2", "M3", "M4", "M5")


def _load_json(p: Path) -> Any:
    return json.loads(p.read_text()) if p.exists() else None


def gate_a(ctx, rows: pd.DataFrame | None, paths_df: pd.DataFrame | None) -> dict:
    ga = ctx.protocol.get("e1", "gate_a")
    reasons, details = [], {}
    e0 = _load_json(ctx.paths.metrics / "e0_results.json")
    if e0 is None:
        return {"status": "BLOCKED", "reasons": ["E0 results missing"], "details": {}}
    failed = [c["check"] for c in e0["checks"] if not c["pass"]]
    details["e0_failed_checks"] = failed
    details["e0_max_optimizer_abs_err"] = max(
        max(r["optimizer_abs_err_u"], r["optimizer_abs_err_v"]) for r in e0["rows"]
    )
    details["e0_max_additive_err"] = max(r["additive_max_abs_err"] for r in e0["rows"])
    if failed:
        reasons.append(f"E0 checks failed: {failed}")
    if rows is None or rows.empty:
        return {
            "status": "BLOCKED" if failed else "INCOMPLETE",
            "reasons": reasons + ["E1 test rows missing"],
            "details": details,
        }
    use = rows[~rows["model_flat"]]
    comp = {}
    for m, g in use.groupby("method"):
        v = g["completeness_rel"].to_numpy(dtype=float)
        v = np.where(np.isfinite(v), v, np.inf)
        comp[m] = {
            "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)),
            "max": float(np.max(v)),
            "median_abs": float(np.nanmedian(g["completeness_abs"])),
            "n": len(v),
        }
        if comp[m]["median"] >= ga["completeness_median_relative_max"]:
            reasons.append(
                f"{m}: median completeness {comp[m]['median']:.4g} >= {ga['completeness_median_relative_max']}"
            )
        if comp[m]["p95"] >= ga["completeness_p95_relative_max"]:
            reasons.append(
                f"{m}: p95 completeness {comp[m]['p95']:.4g} >= {ga['completeness_p95_relative_max']}"
            )
    details["completeness_rel"] = comp
    if paths_df is not None and not paths_df.empty:
        lat = paths_df[paths_df["method"].isin(LATENT_METHODS)]
        frac = float(lat["in_domain"].astype(bool).mean())
        details["latent_domain_valid_fraction"] = frac
        if frac < ga["latent_domain_valid_fraction_min"]:
            reasons.append(f"latent domain validity {frac:.3f} < {ga['latent_domain_valid_fraction_min']}")
        opt = paths_df[paths_df["method"].isin(["M3", "M4"])]
        if "solver_diverged" in opt:
            div = float(opt["solver_diverged"].astype(bool).mean())
            details["solver_divergence_fraction"] = div
            if div > 0.02:
                reasons.append(f"systematic optimizer divergence: {div:.3f} of M3/M4 paths")
        details["endpoint_exact_fraction"] = float(paths_df["endpoint_exact"].astype(bool).mean())
        if details["endpoint_exact_fraction"] < 1.0:
            reasons.append("some latent paths do not reproduce endpoints exactly")
    return {"status": "PASS" if not reasons else "FAIL", "reasons": reasons, "details": details}


def method_table(rows: pd.DataFrame, paths_df: pd.DataFrame | None) -> pd.DataFrame:
    """Scene-averaged mean error per method/stratum, completeness, invalid rate and runtime."""
    use = rows[~rows["model_flat"]]
    scene = use.groupby(["method", "scene_id", "stratum"], as_index=False)["E_coal"].mean()
    st = scene.groupby(["method", "stratum"])["E_coal"].mean().unstack()
    st["all_equal_weight"] = 0.5 * st["strong"] + 0.5 * st["weak"]
    abs_st = (
        use.groupby(["method", "scene_id", "stratum"], as_index=False)["E_coal_abs"]
        .mean()
        .groupby(["method", "stratum"])["E_coal_abs"]
        .mean()
        .unstack()
        .add_prefix("abs_")
    )
    extra = rows.groupby("method").agg(
        completeness_p95=("completeness_rel", lambda v: float(np.nanpercentile(v, 95))),
        invalid_pct=("valid", lambda v: 100.0 * float((~v.astype(bool)).mean())),
        sec_attr_per_pair_seed=("sec_attr_per_pair", "median"),
        sec_path_per_pair=("sec_path_per_pair", "median"),
        shapley_discrepancy_mean=("shapley_discrepancy", "mean"),
        sign_accuracy_single=("sign_accuracy_single", "mean"),
        unchanged_attr_mass=("unchanged_attr_mass", "mean"),
    )
    out = st.join(abs_st).join(extra).reset_index()
    out["median_sec_per_pair_total"] = out["sec_path_per_pair"] + 2 * out["sec_attr_per_pair_seed"]
    return out


def secondary_comparisons(rows: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    recs = []
    for other in sorted(set(rows["method"]) - {"M4"}):
        table, per_seed, _ = paired_scene_table(rows, "M4", other)
        b = bootstrap_reductions(table, n_boot, seed)
        ps = per_seed_strong_reduction(per_seed)
        recs.append(
            {
                "comparator": other,
                **{k: b[k] for k in ("r_strong", "r_weak", "r_all")},
                "ci_strong": b["ci_strong"],
                "ci_weak": b["ci_weak"],
                "ci_all": b["ci_all"],
                **{f"r_strong_seed{s}": v for s, v in ps.items()},
            }
        )
    return pd.DataFrame(recs)


def nuisance_table(rows: pd.DataFrame) -> pd.DataFrame:
    r = rows[~rows["model_flat"]].copy()
    r["abs_An_over_R"] = r["A_n"].abs() / (r["R"] + 1e-8)
    t = r.groupby(["method", "nuisance_changed"])["abs_An_over_R"].mean().unstack()
    t.columns = ["nuisance_changed" if bool(c) else "nuisance_unchanged" for c in t.columns]
    return t.reset_index()


def make_figures(ctx, rows, table, comparator, paths_df, coal) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .renderer import render

    figs = []
    fdir = ctx.paths.figures
    fdir.mkdir(parents=True, exist_ok=True)
    colors = {"strong": "#c0392b", "weak": "#2471a3"}
    t = table.copy()
    t["diff"] = t["E_m4"] - t["E_b"]
    eta = rows.drop_duplicates("scene_id").set_index("scene_id")["eta"]
    t["eta"] = t["scene_id"].map(eta)
    # 1. paired scene differences (all scenes, sorted, both strata)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, st in enumerate(("strong", "weak")):
        d = t[t["stratum"] == st].sort_values("diff")["diff"].to_numpy()
        ax[i].bar(np.arange(len(d)), d, color=colors[st], width=1.0)
        ax[i].axhline(0, color="k", lw=0.8)
        ax[i].set_title(f"{st}: E_M4 - E_{comparator} per scene (n={len(d)}, seed-avg)")
        ax[i].set_xlabel("scene (sorted)")
        ax[i].set_ylabel("paired error difference (negative favours M4)")
    fig.tight_layout()
    fig.savefig(fdir / "e1_paired_differences.png", dpi=130)
    plt.close(fig)
    figs.append("e1_paired_differences.png")
    # 2. difference vs interaction strength
    fig, ax = plt.subplots(figsize=(6, 4))
    for st in ("strong", "weak"):
        s = t[t["stratum"] == st]
        ax.scatter(s["eta"], s["diff"], s=12, alpha=0.7, color=colors[st], label=st)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("oracle interaction index eta")
    ax.set_ylabel(f"E_M4 - E_{comparator} (scene, seed-avg)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fdir / "e1_difference_vs_eta.png", dpi=130)
    plt.close(fig)
    figs.append("e1_difference_vs_eta.png")
    # 3. numeric diagnostics: completeness by method, M3 energy optimality, path energies
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    meths = sorted(rows["method"].unique())
    data = [
        np.clip(rows.loc[rows["method"] == m, "completeness_rel"].to_numpy(float), 1e-9, None) for m in meths
    ]
    ax[0].boxplot(data, tick_labels=meths)
    ax[0].set_yscale("log")
    ax[0].axhline(0.005, color="orange", ls="--", lw=0.8, label="median limit")
    ax[0].axhline(0.02, color="red", ls="--", lw=0.8, label="p95 limit")
    ax[0].set_title("relative completeness residual (all rows)")
    ax[0].legend(fontsize=7)
    if paths_df is not None and not paths_df.empty:
        m3 = paths_df[paths_df["method"] == "M3"]
        ratio = (m3["E_img"] + m3["E_eps"]) / m3["E_m3_exact_continuous"]
        ax[1].hist(ratio.dropna(), bins=30, color="gray")
        ax[1].set_title("M3 discrete energy / exact separable geodesic energy")
        for m, c in (("M2", "k"), ("M3", "g"), ("M4", "r"), ("M5", "b")):
            s = paths_df[paths_df["method"] == m]
            if len(s):
                ax[2].scatter(s["E_img"], s["E_sem"], s=8, alpha=0.6, color=c, label=m)
        ax[2].set_xlabel("image-metric energy (normalised)")
        ax[2].set_ylabel("representation-metric energy (normalised)")
        ax[2].set_title("path energies by method")
        ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fdir / "e1_numeric_diagnostics.png", dpi=130)
    plt.close(fig)
    figs.append("e1_numeric_diagnostics.png")
    # 4. predefined examples: strong-stratum scene with the largest M4 advantage and the
    #    largest M4 disadvantage (by seed-averaged paired difference); seed 11 attributions.
    strong = t[t["stratum"] == "strong"].sort_values("diff")
    if len(strong):
        picks = [
            ("most favourable to M4", strong.iloc[0]["scene_id"]),
            ("most unfavourable to M4", strong.iloc[-1]["scene_id"]),
        ]
        pairs = pd.read_parquet(ctx.paths.data_e1 / "pairs_test.parquet").set_index("scene_id")
        fig, ax = plt.subplots(2, 3, figsize=(13, 7))
        import torch

        for r, (label, sid) in enumerate(picks):
            pr = pairs.loc[sid]
            zb = torch.tensor([[pr[f"zb_{f}"] for f in "abcn"]], dtype=torch.float32)
            zx = torch.tensor([[pr[f"zx_{f}"] for f in "abcn"]], dtype=torch.float32)
            ax[r, 0].imshow(render(zb)[0].permute(1, 2, 0).numpy())
            ax[r, 0].set_title(f"{sid} baseline ({label})", fontsize=8)
            ax[r, 1].imshow(render(zx)[0].permute(1, 2, 0).numpy())
            ax[r, 1].set_title(f"target; changed={pr['changed_factors']} eta={pr['eta']:.2f}", fontsize=8)
            sub = rows[(rows["scene_id"] == sid) & (rows["seed"] == rows["seed"].min())].set_index("method")
            ms = [m for m in ("M1", "M2", "M3", "M4", "M5") if m in sub.index]
            w = 0.15
            for k, m in enumerate(ms):
                ax[r, 2].bar(
                    np.arange(4) + k * w,
                    [sub.loc[m, f"A_{f}"] for f in "abcn"],
                    width=w,
                    label=f"{m} E={sub.loc[m, 'E_coal']:.3g}",
                )
            phi = json.loads(sub.iloc[0]["shapley"])
            ch = pr["changed_idx"].split(",")
            sh = np.zeros(4)
            for j, c in enumerate(ch):
                sh[int(c)] = phi[j]
            ax[r, 2].scatter(np.arange(4) + 2 * w, sh, color="k", marker="x", label="Shapley ref", zorder=5)
            ax[r, 2].set_xticks(np.arange(4) + 2 * w, ["a", "b", "c", "n"])
            ax[r, 2].axhline(0, color="k", lw=0.6)
            ax[r, 2].legend(fontsize=6)
            ax[r, 2].set_title("signed factor attributions (classifier seed 11)", fontsize=8)
            for c in (0, 1):
                ax[r, c].axis("off")
        fig.tight_layout()
        fig.savefig(fdir / "e1_examples_predefined.png", dpi=130)
        plt.close(fig)
        figs.append("e1_examples_predefined.png")
    return figs


def _verdict(ga: str, gb: str, gc: str, boot: dict | None, vs_m2: dict | None) -> tuple[str, list[str]]:
    why = []
    if ga != "PASS":
        return "BLOCKED", [f"Gate A is {ga}: numerical/implementation validity not established"]
    if gb in ("NOT_RUN", "BLOCKED", "INCOMPLETE"):
        return "BLOCKED", [f"Gate B is {gb}"]
    if gb == "PASS":
        if gc == "PASS":
            return "GO", ["Gates A, B and C pass"]
        if gc == "FAIL":
            return "NO_GO", ["E1 passes but valid E2 fails to replicate (no full-project investment)"]
        return "AMBER", [f"Gate B passes but Gate C is {gc}; external replication unverified"]
    if gb == "INCONCLUSIVE":
        return "AMBER", ["Comparator error negligible; relative effect uninformative"]
    # Gate B FAIL: NO_GO unless a positive-but-below-threshold effect (stricter reading).
    small = boot is None or boot["r_all"] < 0.05 or boot["r_strong"] < 0.05
    no_m2_gain = vs_m2 is not None and vs_m2["r_strong"] <= 0
    if small:
        why.append("mean relative error reduction below 0.05 (protocol no_go_if_e1_mean_gain_below)")
    if no_m2_gain:
        why.append("no strong-stratum advantage over decoded latent-linear IG (M2)")
    if small or no_m2_gain:
        return "NO_GO", why + ["Gate B failed"]
    return "AMBER", ["Gate B failed but M4 shows a positive, below-threshold effect (>=0.05)"]


def aggregate(ctx) -> int:
    prot = ctx.protocol
    st = prot.get("e1", "statistics")
    m = ctx.paths.metrics
    rows = pd.read_parquet(m / "e1_test_rows.parquet") if (m / "e1_test_rows.parquet").exists() else None
    paths_df = (
        pd.read_parquet(m / "e1_test_paths.parquet") if (m / "e1_test_paths.parquet").exists() else None
    )
    frozen = _load_json(ctx.paths.manifest_dir / "frozen_manifest.json")
    out: dict[str, Any] = {"generated_utc": utcnow(), "provenance": provenance(prot)}
    ga = gate_a(ctx, rows, paths_df)
    out["gate_a"] = ga
    cnn = {
        s: _load_json(ctx.paths.checkpoints / f"cnn_seed{s}.json") for s in prot.get("e1", "classifier_seeds")
    }
    enc = _load_json(ctx.paths.checkpoints / "encoder_ae.json")
    prep = _load_json(ctx.paths.data_e1 / "prepare_manifest.json")
    out["classifiers"] = {
        str(k): (
            v
            and {
                kk: v[kk]
                for kk in (
                    "test_metrics",
                    "teacher_reference",
                    "competent",
                    "checkpoint_sha256",
                    "mlflow_run_id",
                )
            }
        )
        for k, v in cnn.items()
    }
    out["encoder"] = enc and {
        k: enc[k]
        for k in ("checkpoint_sha256", "objective", "oracle", "lambda_psi", "lambda_D", "mlflow_run_id")
    } | {
        "diagnostics": {
            k: enc["diagnostics"][k]
            for k in ("feature_dim", "n_near_constant", "effective_rank_participation")
        }
    }
    out["data"] = prep and {
        k: prep[k]
        for k in (
            "calibration",
            "pair_generation",
            "pair_counts",
            "leakage",
            "pair_manifest_content_sha256",
            "scene_pool_sizes",
        )
    }
    out["frozen"] = frozen and {
        k: frozen[k]
        for k in ("frozen_utc", "selection", "numeric_config", "encoder", "source_sha256", "git_commit")
    }
    out["unblinding"] = _load_json(ctx.paths.manifest_dir / "unblinding.json")
    gb: dict[str, Any] = {"status": "NOT_RUN", "reasons": ["E1 test rows missing"]}
    boot = vs_m2 = None
    if rows is not None and frozen is not None:
        B = frozen["selection"]["comparator"]
        table, per_seed, counts = paired_scene_table(rows, "M4", B)
        boot = bootstrap_reductions(
            table, st["bootstrap_samples"], st["bootstrap_seed"], st["confidence_level"]
        )
        ps = per_seed_strong_reduction(per_seed)
        flat_frac = float(rows.drop_duplicates(["scene_id", "seed"])["model_flat"].mean())
        prereq = {
            "learned_non_oracle_encoder": bool(enc and not enc["oracle"] and not enc["uses_labels"]),
            "classifiers_competent": all(v and v["competent"] for v in cnn.values()),
            "nonflat_fraction>=0.98": (1 - flat_frac)
            >= prot.get("e1", "competence", "min_fraction_nonflat_pair_scores"),
            "no_scene_leakage": bool(prep and prep["leakage"]["no_leakage"]),
            "gate_a_pass": ga["status"] == "PASS",
            "test_population_complete": len(table)
            == prot.get("e1", "test_pairs_strong") + prot.get("e1", "test_pairs_weak"),
        }
        gb = gate_b_decision(
            boot,
            ps,
            prot.get("e1", "gate_b"),
            prot.get("e1", "metrics", "negligible_baseline_mean_error"),
            prereq,
        )
        gb.update(
            {
                "comparator": B,
                "m4_alpha": frozen["selection"]["m4_alpha"],
                "bootstrap": boot,
                "per_seed_strong_reduction": ps,
                "pair_counts": counts,
                "model_flat_fraction": flat_frac,
            }
        )
        atomic_write_parquet(m / "e1_test_scene_paired.parquet", table)
        atomic_write_parquet(m / "e1_test_seed_paired.parquet", per_seed)
        mt = method_table(rows, paths_df)
        atomic_write_parquet(m / "e1_test_method_table.parquet", mt)
        out["method_table"] = mt.to_dict(orient="records")
        sec = secondary_comparisons(rows, st["bootstrap_samples"], st["bootstrap_seed"])
        atomic_write_parquet(
            m / "e1_test_secondary_comparisons.parquet",
            sec.assign(**{c: sec[c].map(json.dumps) for c in ("ci_strong", "ci_weak", "ci_all")}),
        )
        out["secondary"] = sec.to_dict(orient="records")
        vs_m2 = next((r for r in out["secondary"] if r["comparator"] == "M2"), None)
        out["nuisance"] = nuisance_table(rows).to_dict(orient="records")
        orc = m / "e1_test_oracle.parquet"
        if orc.exists():
            o = pd.read_parquet(orc)
            out["oracle"] = (
                o.groupby(["target", "method", "stratum"])["E_coal"]
                .mean()
                .unstack()
                .reset_index()
                .to_dict(orient="records")
            )
        if paths_df is not None:
            pdf = paths_df.copy()
            pdf["E_total"] = pdf["E_img"] + pdf["E_eps"]
            out["paths"] = {
                "m3_discrete_over_exact": pdf.loc[pdf.method == "M3", "E_total"]
                .div(pdf.loc[pdf.method == "M3", "E_m3_exact_continuous"])
                .describe()
                .to_dict(),
                "energy_by_method": pdf.groupby("method")[["E_img", "E_sem", "latent_length"]]
                .median()
                .to_dict(),
                "m5_accepted": pdf.loc[pdf.method == "M5", "m5_accepted"].describe().to_dict()
                if "m5_accepted" in pdf
                else None,
                "opt_rel_tail_change": pdf.loc[pdf.method.isin(["M3", "M4"])]
                .groupby("method")["opt_rel_tail_change"]
                .describe()
                .to_dict(),
            }
        out["figures"] = make_figures(ctx, rows, table, B, paths_df, None)
    out["gate_b"] = gb
    e2 = read_status(ctx.paths, "evaluate-e2")
    gc_status = "NOT_RUN_BY_DESIGN" if gb["status"] != "PASS" else "INCONCLUSIVE"
    gc_reasons = (
        ["Gate B did not pass: full E2 not run by design (sequential stop)"] if gb["status"] != "PASS" else []
    )
    if e2 and e2.get("gate_c"):
        gc_status, gc_reasons = e2["gate_c"]["status"], e2["gate_c"]["reasons"]
    out["gate_c"] = {"status": gc_status, "reasons": gc_reasons, "e2_status": e2}
    verdict, why = _verdict(ga["status"], gb["status"], gc_status, boot, vs_m2)
    ledger = m / "gpu_ledger.jsonl"
    led = [json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
    out["resources"] = {
        "ledger": led,
        "total_gpu_hours": float(sum(r["gpu_hours"] for r in led)),
        "peak_gpu_mem_gb": float(max([r["peak_gpu_mem_gb"] for r in led] or [0.0])),
        "oom_events": int(sum(r.get("oom_events", 0) for r in led)),
    }
    out["verdict"] = verdict
    out["verdict_reasons"] = why
    atomic_write_json(m / "aggregate.json", out)
    decision = {
        "protocol_version": prot.get("protocol", "version"),
        "protocol_sha256": prot.sha256,
        "gate_a": {"status": ga["status"], "reasons": ga["reasons"]},
        "gate_b": {
            "status": gb["status"],
            "reasons": [
                f"{c['criterion']}: {c['value']} (rule {c['rule']}) -> {'pass' if c['pass'] else 'FAIL'}"
                for c in gb.get("criteria", [])
            ]
            or gb.get("reasons", []),
        },
        "gate_c": {"status": gc_status, "reasons": gc_reasons},
        "verdict": verdict,
        "verdict_reasons": why,
        "total_gpu_hours": out["resources"]["total_gpu_hours"],
        "mlflow_experiment_ids": _experiment_ids(ctx),
        "selected_comparator": gb.get("comparator"),
        "selected_m4_alpha": gb.get("m4_alpha"),
        "e1_test_errors": boot
        and {
            k: boot[k]
            for k in (
                "mean_E_m4_strong",
                "mean_E_b_strong",
                "mean_E_m4_weak",
                "mean_E_b_weak",
                "mean_E_m4_all",
                "mean_E_b_all",
                "r_strong",
                "ci_strong",
                "r_weak",
                "ci_weak",
                "r_all",
                "ci_all",
            )
        },
        "report_generated_utc": utcnow(),
    }
    atomic_write_json(ctx.paths.reports / "decision.json", decision)
    with ctx.tracker.run(ctx.exp("decisions"), "aggregate", tags={"stage": "aggregate", "verdict": verdict}):
        ctx.tracker.log_artifact(ctx.paths.reports / "decision.json", "decision")
        ctx.tracker.log_artifact(m / "aggregate.json", "decision")
        for f in out.get("figures", []):
            ctx.tracker.log_artifact(ctx.paths.figures / f, "figures")
    print(json.dumps(decision, indent=2, default=str))
    return 0


def _experiment_ids(ctx) -> dict[str, str]:
    if ctx.tracker.disabled:
        return {}
    import mlflow

    ids = {}
    for e in mlflow.search_experiments():
        if e.name.startswith("gig-pilot/"):
            ids[e.name] = e.experiment_id
    return ids


# =========================================================================== report
def _fmt(x, nd=4) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(_fmt(v, nd) for v in x) + "]"
    if isinstance(x, float):
        if not np.isfinite(x):
            return str(x)
        return f"{x:.{nd}g}"
    return str(x)


def write_report(ctx) -> int:
    agg = _load_json(ctx.paths.metrics / "aggregate.json")
    if agg is None:
        raise FileNotFoundError("run `aggregate` first")
    from .report_md import render_report

    md = render_report(ctx, agg)
    (ctx.paths.reports / "FINAL_REPORT.md").write_text(md)
    print(
        f"wrote {ctx.paths.reports / 'FINAL_REPORT.md'} ({len(md)} chars); sha256 "
        f"{sha256_file(ctx.paths.reports / 'FINAL_REPORT.md')[:16]}"
    )
    return 0
