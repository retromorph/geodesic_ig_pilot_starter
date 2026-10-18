"""Render reports/FINAL_REPORT.md (structure of docs/REPORT_TEMPLATE.md) from aggregate.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import REPO_ROOT, sha256_file


def _f(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(_f(v, nd) for v in x) + "]"
    if isinstance(x, (float, np.floating)):
        return f"{x:.{nd}g}" if np.isfinite(x) else str(x)
    return str(x)


def _status(ctx, stage):
    p = ctx.paths.status / f"{stage}.json"
    return json.loads(p.read_text()) if p.exists() else None


def render_report(ctx, agg: dict) -> str:
    prov = agg["provenance"]
    ga, gb, gc = agg["gate_a"], agg["gate_b"], agg["gate_c"]
    boot = gb.get("bootstrap")
    L: list[str] = []
    w = L.append
    w("# FINAL REPORT — Geodesic Integrated Gradients go/no-go pilot\n")
    w(
        f"_Generated mechanically by `geodesic_ig_pilot.cli report` at {agg['generated_utc']} from saved results "
        "(`reports/metrics/aggregate.json`). Hand-written interpretation is confined to the clearly marked section "
        "taken from `reports/interpretation.md`._\n"
    )
    w("## Executive decision\n")
    w(f"- **Final investment verdict:** `{agg['verdict']}`")
    w(f"- **Mechanical reasons:** {'; '.join(agg['verdict_reasons'])}")
    w(f"- **Gate A:** `{ga['status']}`" + (f" — {'; '.join(ga['reasons'])}" if ga["reasons"] else ""))
    w(f"- **Gate B:** `{gb['status']}`")
    w(f"- **Gate C:** `{gc['status']}` — {'; '.join(gc['reasons'])}")
    res = agg["resources"]
    w(
        f"- **Compute used:** {res['total_gpu_hours']:.3f} measured GPU-hours (wall-clock of GPU stages on one device) "
        f"/ 40 A4000 GPU-hours; peak allocated VRAM {res['peak_gpu_mem_gb']:.2f} GB; OOM events {res['oom_events']}"
    )
    supported = (
        "yes"
        if agg["verdict"] == "GO"
        else ("inconclusive" if agg["verdict"] in ("AMBER", "BLOCKED") else "no")
    )
    w(f"- **Strictly preregistered hypothesis supported?** {supported}\n")
    interp = ctx.paths.reports / "interpretation.md"
    if interp.exists():
        w("### Interpretation (hand-written after aggregation; does not alter any gate)\n")
        w(interp.read_text().strip() + "\n")

    w("## 1. Immutable provenance\n")
    w("| Item | Value | Verification |\n|---|---|---|")
    w(f"| Protocol SHA-256 | `{prov['protocol_sha256']}` | `sha256sum configs/protocol.v1.yaml` |")
    w(
        f"| Preregistration SHA-256 | `{prov['preregistration_sha256']}` | `sha256sum docs/PREREGISTRATION.md` |"
    )
    w(
        f"| Git commit / dirty (tracked files) | `{prov['git_commit']}` / {prov['git_dirty']} | `git rev-parse HEAD` |"
    )
    w(f"| uv.lock SHA-256 | `{prov['uv_lock_sha256']}` | `sha256sum uv.lock` |")
    w(f"| E1 source SHA-256 (frozen modules) | `{prov['source_sha256']}` | `config.source_tree_sha256()` |")
    w(
        f"| Python / torch / CUDA (wheel) / driver | {prov['python']} / {prov['torch']} / {prov['pytorch_cuda']} / "
        f"{prov['cuda_driver']} | doctor report |"
    )
    w(f"| GPU / peak VRAM | {prov['gpu_name']} / {res['peak_gpu_mem_gb']:.2f} GB | gpu_ledger.jsonl |")
    cl = agg.get("classifiers") or {}
    enc = agg.get("encoder") or {}
    ck = ", ".join(f"seed {k}: `{(v or {}).get('checkpoint_sha256', 'n/a')[:16]}…`" for k, v in cl.items())
    w(
        f"| Checkpoints SHA-256 | {ck}; encoder `{enc.get('checkpoint_sha256', 'n/a')[:16]}…` | `checkpoints/*.json` |"
    )
    data = agg.get("data") or {}
    w(
        f"| Pair manifests (content SHA-256) | {_f(data.get('pair_manifest_content_sha256'))} | prepare_manifest.json |"
    )
    fr = agg.get("frozen") or {}
    fm = ctx.paths.manifest_dir / "frozen_manifest.json"
    w(
        f"| Frozen manifest SHA-256 / frozen at | `{sha256_file(fm) if fm.exists() else 'n/a'}` / "
        f"{fr.get('frozen_utc', 'n/a')} | `reports/manifest/frozen_manifest.json` (read-only) |"
    )
    ub = agg.get("unblinding") or {}
    w(
        f"| Unblinding timestamp | {ub.get('test_evaluation_started_utc', 'NOT UNBLINDED')} | "
        "`reports/manifest/unblinding.json` |"
    )
    w(
        f"| MLflow experiments | {_f(_load_decision(ctx).get('mlflow_experiment_ids'))} | tracking server via SSH tunnel |\n"
    )

    w("## 2. Research question and methods\n")
    w(
        (REPO_ROOT / "reports" / "methods_summary.md").read_text().strip() + "\n"
        if (REPO_ROOT / "reports" / "methods_summary.md").exists()
        else "_methods summary missing_\n"
    )
    if fr:
        sel = fr["selection"]
        w(
            f"**Frozen comparator B:** `{sel['comparator']}`; **frozen M4 alpha:** `{sel['m4_alpha']}` "
            f"(rule: {sel['rule']}, validation run `{sel['validation_mlflow_run_id']}`).\n"
        )
        w("Validation equal-stratum-weighted mean coalition errors (selection inputs):\n")
        w("| Method | strong | weak | equal-weight |\n|---|---:|---:|---:|")
        for r in sorted(sel["validation_scores"], key=lambda r: r["equal_weight"]):
            w(f"| {r['method']} | {_f(r['strong'])} | {_f(r['weak'])} | {_f(r['equal_weight'])} |")
        nc = fr["numeric_config"]
        w(
            f"\nFrozen numerics: K={nc['segments']} segments, {nc['iterations']} Adam iterations (lr {nc['lr']}), "
            f"{nc['restarts']} energy-only restarts, quadrature {nc['quadrature_steps']} with uniform doubling while "
            f"|residual|/(R+1e-8) > {nc['refine_target_rel']}, invalid if > {nc['invalid_rel']} at 512, epsilon "
            f"{nc['epsilon']}, M5 {nc['m5']['candidates']} candidates in energy band {nc['m5']['energy_band']}."
        )
        w(
            f"Profiling trials (validation only): {_f(nc['profiling_trials'])}; pixel/latent cross-check: "
            f"{_f(nc.get('pixel_latent_crosscheck'))}\n"
        )

    w("## 3. E0 — analytic results and negative controls\n")
    e0p = ctx.paths.metrics / "e0_results.json"
    if e0p.exists():
        e0 = json.loads(e0p.read_text())
        w(
            "| c | expected A_u | analytic A_u | optimizer A_u | abs err | optimizer A_v | latent-linear A_u | "
            "semantic-psi A_u | additive max err | completeness err | discrete E / analytic geodesic E / linear E |"
        )
        w("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for r in e0["rows"]:
            w(
                f"| {r['c']:g} | {r['expected_A_u']:.6f} | {r['analytic_A_u']:.6f} | {r['optimizer_A_u']:.6f} | "
                f"{r['optimizer_abs_err_u']:.2e} | {r['optimizer_A_v']:.6f} | {r['linear_A_u']:.6f} | "
                f"{r['semantic_psi_A_u']:.6f} | {r['additive_max_abs_err']:.1e} | {r['optimizer_completeness_err']:.1e} | "
                f"{r['optimizer_energy_discrete']:.5f} / {r['analytic_geodesic_energy']:.5f} / {r['linear_energy']:.5f} |"
            )
        w(
            "\nHybrid metric psi+alpha*image (curvature transfer illustration), A_u: "
            + "; ".join(
                f"c={r['c']:g}: "
                + ", ".join(
                    f"{k.replace('hybrid_A_u_', '')}={v:.4f}"
                    for k, v in r.items()
                    if k.startswith("hybrid_A_u")
                )
                for r in e0["rows"]
            )
        )
        n_pass = sum(c["pass"] for c in e0["checks"])
        w(
            f"\nE0 checks passed: {n_pass}/{len(e0['checks'])}. Settings: {_f(e0['settings'])}. Random smooth "
            "endpoint-preserving paths keep completeness but redistribute interaction credit, e.g. c=1 A_u over 5 "
            f"random paths: {_f(e0['rows'][1]['random_paths_A_u'])}.\n"
        )
    else:
        w("E0 not run.\n")

    w("## 4. E1 — population and model competence\n")
    if data:
        pg = data["pair_generation"]
        w("| Attribute | Count / value |\n|---|---:|")
        w(f"| Scene pools | {_f(data['scene_pool_sizes'])} |")
        w(f"| Calibrated C (median raw g_NL, 20k train-only draw) | {data['calibration']['C']:.6f} |")
        for s in ("validation", "test"):
            w(f"| Candidates drawn ({s}) | {pg[s]['candidates_drawn']} (cap {pg[s]['max_candidates']}) |")
            w(f"| Filled quotas ({s}) | {_f(pg[s]['filled'])} |")
            w(f"| Rejections ({s}) | {_f(pg[s]['rejections'])} |")
        w(
            f"| Leakage check (scene IDs and exact factor vectors, all pools) | no_leakage={data['leakage']['no_leakage']} |"
        )
        for k, v in cl.items():
            if v:
                t, tr = v["test_metrics"], v["teacher_reference"]
                w(
                    f"| Classifier seed {k}: AUROC / accuracy / ECE / NLL (5k held-out scenes) | {t['auroc']:.4f} / "
                    f"{t['accuracy']:.4f} / {t['ece']:.4f} / {t['nll']:.4f} (competent={v['competent']}) |"
                )
                w(
                    f"| Seed {k} class balance: label pos rate / predicted pos rate | {t['label_pos_rate']:.3f} / "
                    f"{t['pred_pos_rate']:.3f} |"
                )
                w(
                    f"| Seed {k} teacher reference: sign(g_NL) accuracy / AUROC; MAE p vs teacher; corr(margin, 3g) | "
                    f"{tr['teacher_sign_accuracy']:.4f} / {tr['teacher_auroc']:.4f}; {tr['mae_p_vs_teacher']:.4f}; "
                    f"{tr['corr_margin_vs_teacher_logodds']:.4f} |"
                )
        if enc:
            w(f"| Learned encoder objective / oracle? | {enc['objective']} / {enc['oracle']} |")
            w(
                f"| Encoder diagnostics | {_f(enc['diagnostics'])}; lambda_psi={_f(enc['lambda_psi'])}, "
                f"lambda_D={_f(enc['lambda_D'])} |"
            )
    w("")

    w("## 5. E1 — primary result\n")
    if boot:
        w(
            "Scene-averaged (two classifier seeds) mean normalised coalition error on the 240 frozen test scenes; "
            "all methods share identical (baseline, input, checkpoint, score) per row.\n"
        )
        w(
            "| Method | E_coal strong ↓ | E_coal weak ↓ | E_coal all (equal-weight) ↓ | abs. err strong | "
            "Completeness p95 | Invalid % | Median sec / pair (path + 2 seeds) | Shapley discrepancy | sign acc. |"
        )
        w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in agg["method_table"]:
            w(
                f"| {r['method']} | {_f(r['strong'])} | {_f(r['weak'])} | {_f(r['all_equal_weight'])} | "
                f"{_f(r.get('abs_strong'))} | {_f(r['completeness_p95'])} | {_f(r['invalid_pct'])} | "
                f"{_f(r['median_sec_per_pair_total'])} | {_f(r['shapley_discrepancy_mean'])} | "
                f"{_f(r['sign_accuracy_single'])} |"
            )
        w(
            f"\n**Frozen comparator B:** `{gb['comparator']}` (validation-selected); M4 alpha `{gb['m4_alpha']}`.\n"
        )
        w(
            "| Gate B quantity | Measured | 95% paired scene-bootstrap CI | Requirement | Pass? |\n|---|---:|---|---|---|"
        )
        crit = {c["criterion"]: c for c in gb["criteria"]}
        w(
            f"| Strong relative reduction r_strong | {_f(boot['r_strong'])} | {_f(boot['ci_strong'])} | >=0.15, lower "
            f"bound >0.05 | {_f(crit['r_strong']['pass'])} / {_f(crit['r_strong_ci_lower']['pass'])} |"
        )
        w(
            f"| Full relative reduction r_all | {_f(boot['r_all'])} | {_f(boot['ci_all'])} | >=0.10, lower bound >0 | "
            f"{_f(crit['r_all']['pass'])} / {_f(crit['r_all_ci_lower']['pass'])} |"
        )
        w(
            f"| Weak relative reduction r_weak | {_f(boot['r_weak'])} | {_f(boot['ci_weak'])} | >=-0.05 | "
            f"{_f(crit['r_weak']['pass'])} |"
        )
        for s, v in gb["per_seed_strong_reduction"].items():
            w(
                f"| Seed {s} strong reduction | {_f(v)} | n/a | >0 | {_f(crit[f'seed_{s}_strong_reduction']['pass'])} |"
            )
        for name in (
            "learned_non_oracle_encoder",
            "classifiers_competent",
            "nonflat_fraction>=0.98",
            "no_scene_leakage",
            "gate_a_pass",
            "test_population_complete",
        ):
            if name in crit:
                w(f"| {name} | {_f(crit[name]['value'])} | n/a | true | {_f(crit[name]['pass'])} |")
        w(
            f"\nAbsolute means: strong M4 {_f(boot['mean_E_m4_strong'])} vs B {_f(boot['mean_E_b_strong'])}; weak M4 "
            f"{_f(boot['mean_E_m4_weak'])} vs B {_f(boot['mean_E_b_weak'])}; all M4 {_f(boot['mean_E_m4_all'])} vs B "
            f"{_f(boot['mean_E_b_all'])}. Comparator mean error negligible (<1e-6)? "
            f"{_f(gb['comparator_mean_error_negligible'])}. Pair counts: "
            f"{_f({k: v for k, v in gb['pair_counts'].items()})}; model-flat fraction {_f(gb['model_flat_fraction'])}.\n"
        )
        w(
            "Figures: `reports/figures/e1_paired_differences.png`, `reports/figures/e1_difference_vs_eta.png`, "
            "`reports/figures/e1_numeric_diagnostics.png`, `reports/figures/e1_examples_predefined.png` (examples chosen "
            "by a fixed rule: strong scene with the largest and the smallest seed-averaged E_B - E_M4).\n"
        )
    else:
        w(f"Not run: {gb.get('reasons')}\n")

    w("## 6. E1 — controls and mechanism diagnosis (secondary; no gate)\n")
    if agg.get("secondary"):
        w(
            "M4 versus every other method (same paired scene bootstrap; secondary, not the preregistered gate):\n"
        )
        w(
            "| vs | r_strong [95% CI] | r_weak [95% CI] | r_all [95% CI] | r_strong seed 11 | r_strong seed 29 |"
        )
        w("|---|---|---|---|---:|---:|")
        for r in agg["secondary"]:
            w(
                f"| {r['comparator']} | {_f(r['r_strong'])} {_f(r['ci_strong'])} | {_f(r['r_weak'])} {_f(r['ci_weak'])} | "
                f"{_f(r['r_all'])} {_f(r['ci_all'])} | {_f(r.get('r_strong_seed11'))} | {_f(r.get('r_strong_seed29'))} |"
            )
    if agg.get("oracle"):
        w(
            "\nOracle analytic targets along the same latent paths (diagnostic only; M1 undefined in latent space):\n"
        )
        w("| target | method | strong | weak |\n|---|---|---:|---:|")
        for r in agg["oracle"]:
            w(f"| {r['target']} | {r['method']} | {_f(r.get('strong'))} | {_f(r.get('weak'))} |")
    if agg.get("nuisance"):
        w("\nMean |A_n|/R (nuisance attribution) by whether the nuisance changed:\n")
        w("| method | nuisance unchanged | nuisance changed |\n|---|---:|---:|")
        for r in agg["nuisance"]:
            w(f"| {r['method']} | {_f(r.get('nuisance_unchanged'))} | {_f(r.get('nuisance_changed'))} |")
    if agg.get("paths"):
        w(f"\nPath diagnostics: {_f(agg['paths'])}\n")
    hr = agg.get("exploratory_headroom")
    if hr:
        w(
            "\n**EXPLORATORY (post-unblinding; not a gate input).** Lower bounds on the coalition error of any additive "
            "attribution for the same CNN games (scene-seed means):\n"
        )
        w("| quantity | strong | weak |\n|---|---:|---:|")
        for r in hr["bounds"]:
            w(f"| best additive A ({r['bound']}) | {_f(r.get('strong'))} | {_f(r.get('weak'))} |")
        w(f"| exact Shapley | {_f(hr['shapley'].get('strong'))} | {_f(hr['shapley'].get('weak'))} |")
        for r in hr["methods"]:
            w(f"| {r['method']} | {_f(r.get('strong'))} | {_f(r.get('weak'))} |")
        w(
            "\nFraction of each method's error that is reducible at all by a complete additive attribution "
            f"(1 − bound/method): {_f(hr['headroom_fraction_vs_method'])}\n"
        )

    w("## 7. E2 — independent validation\n")
    e2 = gc.get("e2_status") or {}
    w(f"- **Gate C:** `{gc['status']}` — {'; '.join(gc['reasons'])}")
    prov_p = ctx.paths.data_e2 / "e2_provenance.json"
    pre_p = ctx.paths.metrics / "e2_preflight.json"
    mar_p = ctx.paths.metrics / "e2_preflight_margins.json"
    if prov_p.exists():
        pv = json.loads(prov_p.read_text())
        w(
            f"- **Dataset / code:** FunnyBirds (visinf/funnybirds-framework @ `{pv['framework_commit'][:12]}`, "
            f"visinf/funnybirds @ `{pv['funnybirds_commit'][:12]}`); code license {pv['code_license']}; data license: "
            f"{pv['data_license']}. Zip members read selectively by HTTP range ({pv['zip_members']} entries indexed, "
            f"{pv['bytes_downloaded'] / 1e6:.0f} MB fetched)."
        )
        w(
            f"- **Classifier:** official FunnyBirds ResNet-50 (`resnet50_final_0_checkpoint_best.pth.tar`, SHA-256 "
            f"`{pv['resnet50_sha256'][:16]}…`, ETag {pv['ckpt_head']['ETag']}); top-1 on the 152 selected original "
            f"test images = {pv['classifier_top1_on_selected_originals']:.3f}. Target: logit_c − logsumexp_(j≠c) logit_j."
        )
        w(
            f"- **Frozen representation for E2-M4:** torchvision ResNet-18 IMAGENET1K_V1 layer3 (SHA-256 "
            f"`{pv['resnet18_imagenet_sha256'][:16]}…`). DINOv3 was not used (gated access)."
        )
    sel_p = ctx.paths.data_e2 / "e2_selection.json"
    if sel_p.exists():
        sl = json.loads(sel_p.read_text())
        multi = sum(len(i["removed"]) >= 2 for i in sl["test"])
        w(
            f"- **Locked selection (model-blind, stream `e2`):** {len(sl['preflight'])} preflight / {len(sl['validation'])} "
            f"validation / {len(sl['test'])} held-out test images, disjoint IDs; every pair removes 2–3 visible parts "
            f"({multi}/{len(sl['test'])} held-out pairs are multi-part); all 2^|R| coalition renders are pre-rendered "
            "interventions."
        )
    if pre_p.exists():
        pr = json.loads(pre_p.read_text())
        w(
            f"- **Decoder:** compact spatial AE 256²×3 → 32²×8, {pr['ae']['n_train']} unlabeled train-split images, "
            f"{pr['ae']['epochs']} epochs ({pr['ae']['objective']}; SHA-256 `{pr['ae']['sha256'][:16]}…`)."
        )
        w(
            "\n| Preflight criterion (24 endpoints of 12 pairs) | Measured | Requirement | Pass? |\n|---|---:|---|---|"
        )
        w(
            f"| Top-1 preserved after reconstruction | {pr['top1_preserved_fraction']:.3f} | ≥ 0.90 | "
            f"{_f(pr['top1_preserved_fraction'] >= 0.9)} |"
        )
        w(
            f"| Part-presence fidelity (all visible parts) | {pr['part_fidelity_fraction']:.3f} | ≥ 0.90 | "
            f"{_f(pr['part_fidelity_fraction'] >= 0.9)} |"
        )
        w(f"| Pairs with both endpoints part-faithful | {pr['pairs_both_endpoints_ok']:.3f} | reported | |")
        w(f"| Median endpoint reconstruction MSE | {pr['median_endpoint_mse']:.2e} | reported | |")
        if mar_p.exists():
            mg = json.loads(mar_p.read_text())
            w(
                f"| Median / max abs. target-margin distortion (logits) | {mg['median_abs_margin_distortion']:.3f} / "
                f"{mg['max_abs_margin_distortion']:.3f} | reported | |"
            )
            w(
                f"| Contrast sign preserved (f(x)−f(b)) / median abs. contrast change | "
                f"{mg['contrast_sign_preserved_fraction']:.3f} / {mg['median_abs_contrast_change']:.3f} | reported | |"
            )
        w(
            f"\nPart-fidelity rule (prespecified): {pr['part_fidelity_rule']}. **Preflight pass: "
            f"{_f(pr['preflight_pass'])}** → E2 status `{e2.get('status', 'n/a')}`; preflight GPU-hours "
            f"{pr['resources']['gpu_hours']:.3f}, peak {pr['resources']['peak_gpu_mem_gb']:.1f} GB.\n"
        )
        if pr.get("path_smoke"):
            w(
                "Exploratory path smoke on 4 preflight pairs (identical reconstructed endpoints; **not** a result, the decoder "
                "failed preflight; AP uses tau = 0.5 only for this smoke; quadrature capped at N = 256):\n"
            )
            w(
                "| scene | removed | method | AP | completeness rel. | N | sec |\n|---|---|---|---:|---:|---:|---:|"
            )
            for r in pr["path_smoke"]:
                w(
                    f"| {r['scene_id']} | {r['removed']} | {r['method']} | {_f(r['AP'])} | {r['completeness_rel']:.4f} | "
                    f"{r['N_used']} | {r['sec']:.1f} |"
                )
            w("")
    w(
        "AP definition (coded in `external.part_ap`, to be locked on validation for a full run): rank the 5 parts by "
        "signed grouped pixel attribution; a part is relevant iff it was removed in the baseline and restoring it alone "
        "raises the target margin by ≥ tau on the *rendered* intervention; no relevant part → `UNDEFINED_LABEL`. "
        "MA-GIG: `NOT_COMPARABLE_WITHIN_PILOT` (not adapted). No AP values are claimed as E2 results.\n"
    )

    w("## 8. Error analysis and validity threats\n")
    w(
        (ctx.paths.reports / "validity_threats.md").read_text().strip() + "\n"
        if (ctx.paths.reports / "validity_threats.md").exists()
        else "_see interpretation section_\n"
    )

    w("## 9. Resource accounting and reproducibility\n")
    w("| Stage | GPU-hours | Max VRAM (GB) | Wall-clock (s) | Exit |\n|---|---:|---:|---:|---|")
    for r in res["ledger"]:
        w(
            f"| {r['stage']} ({r['device']}) | {r['gpu_hours']:.4f} | {r['peak_gpu_mem_gb']:.2f} | {r['wall_seconds']:.1f} | "
            f"{r.get('exit')} |"
        )
    w(f"| **Total** | **{res['total_gpu_hours']:.4f}** | {res['peak_gpu_mem_gb']:.2f} | | |\n")
    w("Replay commands, hashes and run IDs: `reports/reproducibility.md`.\n")

    w("## 10. Protocol deviations and exploratory observations\n")
    w("See `reports/deviations.md` (all deviations, reruns and the unblinding timeline).\n")

    w("## 11. Verdict and next steps\n")
    w(
        f"Mechanical verdict: **{agg['verdict']}** — {'; '.join(agg['verdict_reasons'])}. Machine-readable: "
        "`reports/decision.json`.\n"
    )
    nxt = ctx.paths.reports / "next_steps.md"
    if nxt.exists():
        w(nxt.read_text().strip() + "\n")
    return "\n".join(L) + "\n"


def _load_decision(ctx) -> dict:
    p = ctx.paths.reports / "decision.json"
    return json.loads(p.read_text()) if p.exists() else {}


__all__ = ["render_report", "Path"]
