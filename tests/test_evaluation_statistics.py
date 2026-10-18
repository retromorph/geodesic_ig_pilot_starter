"""Coalitions, Shapley, data generation/leakage, bootstrap pairing, gates and MLflow roundtrip."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from geodesic_ig_pilot.config import REPO_ROOT, load_protocol, rng_for
from geodesic_ig_pilot.datasets import (
    calibrate_C,
    check_no_leakage,
    generate_pairs,
    make_scenes,
    manifest_sha256,
)
from geodesic_ig_pilot.evaluations import (
    coalition_error,
    coalition_masks,
    hybrid_latents,
    interaction_index,
    shapley_values,
)
from geodesic_ig_pilot.statistics import (
    bootstrap_reductions,
    conservative_pair,
    gate_b_decision,
    paired_scene_table,
    per_seed_strong_reduction,
)

PROTO = REPO_ROOT / "configs" / "protocol.v1.yaml"


def test_coalition_enumeration_order_and_empty_delta():
    m = coalition_masks(3)
    assert m.shape == (8, 3) and not m[0].any() and m[-1].all()
    assert m[1].tolist() == [True, False, False] and m[2].tolist() == [False, True, False]
    zb, zx = np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.9, 0.2, 0.7, 0.4])
    h = hybrid_latents(zb, zx, [0, 2])
    assert np.allclose(h[0], zb) and np.allclose(h[-1], zx) and np.allclose(h[1], [0.9, 0.2, 0.3, 0.4])


def test_shapley_hand_checks():
    # f = a + b, endpoints (0,0)->(1,1): v = [0, 1, 1, 2]
    assert np.allclose(shapley_values(np.array([0.0, 1.0, 1.0, 2.0])), [1.0, 1.0])
    # f = a * b: v = [0, 0, 0, 1] -> (0.5, 0.5)
    assert np.allclose(shapley_values(np.array([0.0, 0.0, 0.0, 1.0])), [0.5, 0.5])
    # efficiency for a random 3-player game
    v = np.random.default_rng(0).normal(size=8)
    assert np.isclose(shapley_values(v).sum(), v[-1] - v[0])


def test_coalition_error_normalisation():
    vals = np.array([0.0, 1.0, 1.0, 2.0])
    assert coalition_error(vals, np.array([1.0, 1.0]))["E_coal"] == 0.0
    e = coalition_error(vals, np.array([2.0, 0.0]))
    # residuals: 0, -1, +1, 0 over R=2 -> mean((r/2)^2) = (0.25+0.25)/4
    assert np.isclose(e["E_coal"], 0.125, atol=1e-9) and e["R"] == 2.0
    # error invariant to a constant shift in scores
    assert np.isclose(coalition_error(vals + 7.0, np.array([2.0, 0.0]))["E_coal"], e["E_coal"])


def test_interaction_index_additive_is_zero():
    g = lambda z: z[:, 0] + 2 * z[:, 1]  # noqa: E731
    assert interaction_index(g, np.zeros(4), np.ones(4), [0, 1]) < 1e-12
    gm = lambda z: z[:, 0] * z[:, 1]  # noqa: E731
    assert np.isclose(interaction_index(gm, np.zeros(4), np.ones(4), [0, 1]), 1.0, atol=1e-6)


def test_deterministic_streams_and_pairs():
    p = load_protocol(PROTO)
    a = rng_for("pairs_test", 20261018).uniform(size=5)
    b = rng_for("pairs_test", 20261018).uniform(size=5)
    c = rng_for("pairs_validation", 20261018).uniform(size=5)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    C = calibrate_C(p)["C"]
    q = {("strong", True): 2, ("strong", False): 2, ("weak", True): 2, ("weak", False): 2}
    d1, info = generate_pairs(p, "validation", C, 20000, quotas=q)
    d2, _ = generate_pairs(p, "validation", C, 20000, quotas=q)
    assert manifest_sha256(d1) == manifest_sha256(d2) and info["all_quotas_filled"]
    assert (d1["m"] >= 2).all()
    assert (d1["baseline_label_oracle"] != d1["input_label_oracle"]).all()
    assert d1.groupby(["stratum", "nuisance_changed"]).size().min() == 2
    assert (d1.loc[d1.stratum == "strong", "eta"] >= 0.25).all() and (
        d1.loc[d1.stratum == "weak", "eta"] <= 0.05
    ).all()


def test_leakage_detection():
    p = load_protocol(PROTO)
    tr = make_scenes(p, "train_seed11", 50, 0.0, "train11")
    va = make_scenes(p, "classifier_val", 50, 0.0, "clfval")
    assert check_no_leakage({"tr": tr, "va": va}, {})["no_leakage"]
    va2 = va.copy()
    va2.loc[0, ["a", "b", "c", "n"]] = tr.loc[3, ["a", "b", "c", "n"]].to_numpy()
    assert not check_no_leakage({"tr": tr, "va": va2}, {})["no_leakage"]


def test_protocol_threshold_guard(tmp_path):
    text = PROTO.read_text().replace(
        "strong_mean_relative_reduction_min: 0.15", "strong_mean_relative_reduction_min: 0.05"
    )
    bad = tmp_path / "p.yaml"
    bad.write_text(text)
    with pytest.raises(ValueError):
        load_protocol(bad)


def _rows(e4_strong, eb_strong, e4_weak, eb_weak, seeds=(11, 29)):
    rows = []
    for stratum, e4s, ebs in (("strong", e4_strong, eb_strong), ("weak", e4_weak, eb_weak)):
        for i, (e4, eb) in enumerate(zip(e4s, ebs)):
            for k, s in enumerate(seeds):
                for meth, e in (("M4", e4 * (1 + 0.1 * k)), ("B", eb)):
                    rows.append(
                        {
                            "scene_id": f"{stratum}{i}",
                            "seed": s,
                            "method": meth,
                            "E_coal": e,
                            "valid": True,
                            "model_flat": False,
                            "stratum": stratum,
                        }
                    )
    return pd.DataFrame(rows)


def test_paired_scene_bootstrap_and_seed_averaging():
    rng = np.random.default_rng(0)
    eb = rng.uniform(0.5, 1.0, 40)
    df = _rows(0.5 * eb[:20], eb[:20], eb[20:], eb[20:])
    table, per_seed, counts = paired_scene_table(df, "M4", "B")
    assert len(table) == 40 and counts["scene_seeds"] == 80
    # seed averaging: M4 seed29 error is 1.1x seed11 -> scene mean is 1.05x
    s0 = table[table.scene_id == "strong0"].iloc[0]
    assert np.isclose(s0["E_m4"], 0.5 * eb[0] * 1.05)
    boot = bootstrap_reductions(table, 500, 9001)
    boot2 = bootstrap_reductions(table, 500, 9001)
    assert boot == boot2
    assert np.isclose(boot["r_strong"], 1 - 0.525 * eb[:20].mean() / eb[:20].mean())
    assert boot["ci_strong"][0] <= boot["r_strong"] <= boot["ci_strong"][1]
    # paired resampling: identical errors give a degenerate (zero-width) interval at r = 0
    same = _rows(eb[:20], eb[:20], eb[20:], eb[20:], seeds=(11,))
    t2, _, _ = paired_scene_table(same, "M4", "B")
    b2 = bootstrap_reductions(t2, 200, 1)
    assert b2["ci_strong"] == [0.0, 0.0] and b2["ci_all"] == [0.0, 0.0]
    seeds = per_seed_strong_reduction(per_seed)
    assert set(seeds) == {11, 29} and seeds[11] > seeds[29]


def test_conservative_invalid_handling_and_flat_exclusion():
    assert conservative_pair(0.1, False, 0.5, True) == (0.5, 0.5)
    assert conservative_pair(0.9, False, 0.5, True) == (0.9, 0.5)
    assert conservative_pair(0.1, True, float("nan"), False) == (0.1, 0.1)
    df = _rows([0.1], [0.2], [0.1], [0.2])
    df.loc[(df.scene_id == "strong0") & (df.seed == 29), "model_flat"] = True
    table, per_seed, counts = paired_scene_table(df, "M4", "B")
    assert counts["model_flat"] == 1 and len(per_seed) == 3


def test_gate_b_mechanics():
    th = {
        "strong_mean_relative_reduction_min": 0.15,
        "strong_bootstrap_lower_bound_gt": 0.05,
        "all_mean_relative_reduction_min": 0.10,
        "all_bootstrap_lower_bound_gt": 0.0,
        "weak_relative_reduction_min": -0.05,
    }
    boot = {
        "r_strong": 0.2,
        "ci_strong": [0.1, 0.3],
        "r_all": 0.12,
        "ci_all": [0.01, 0.2],
        "r_weak": 0.0,
        "mean_E_b_strong": 0.1,
        "mean_E_b_all": 0.1,
    }
    ok = gate_b_decision(boot, {11: 0.1, 29: 0.2}, th, 1e-6, {"learned_encoder": True})
    assert ok["status"] == "PASS"
    bad = gate_b_decision(boot, {11: 0.1, 29: -0.01}, th, 1e-6, {"learned_encoder": True})
    assert bad["status"] == "FAIL"
    tiny = gate_b_decision(boot | {"mean_E_b_strong": 1e-8}, {11: 0.1, 29: 0.2}, th, 1e-6, {"x": True})
    assert tiny["status"] == "INCONCLUSIVE"


def test_mlflow_artifact_roundtrip(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'm.db'}")
    exp = mlflow.create_experiment("t", artifact_location=(tmp_path / "art").as_uri())
    payload = b"geodesic" * 100
    src = tmp_path / "x.bin"
    src.write_bytes(payload)
    with mlflow.start_run(experiment_id=exp) as run:
        mlflow.log_param("p", 1)
        mlflow.log_artifact(str(src), "a")
    got = mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id, artifact_path="a/x.bin", dst_path=str(tmp_path / "d")
    )
    assert hashlib.sha256(open(got, "rb").read()).hexdigest() == hashlib.sha256(payload).hexdigest()
    assert mlflow.get_run(run.info.run_id).data.params["p"] == "1"


def test_tracker_refuses_non_loopback(monkeypatch):
    from geodesic_ig_pilot.tracking import Tracker

    monkeypatch.setenv("GIG_DISABLE_MLFLOW", "0")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://0.0.0.0:5000")
    with pytest.raises(RuntimeError):
        Tracker(load_protocol(PROTO))


def test_part_ap_definition_hand_check():
    from geodesic_ig_pilot.external import part_ap, part_masks

    A = {"beak": 3.0, "eye": 0.1, "foot": 2.0, "tail": -1.0, "wing": 0.0}
    # removed parts beak, tail; only beak's single-part restoration effect exceeds tau
    assert part_ap(A, {"beak": 1.0, "tail": 0.2}, tau=0.5) == 1.0
    # tail relevant but ranked last (5th of 5) -> AP = 1/5
    assert part_ap(A, {"beak": 0.1, "tail": 0.9}, tau=0.5) == 0.2
    assert part_ap(A, {"beak": 0.1}, tau=0.5) is None  # UNDEFINED_LABEL
    # part masks form a partition with the framework's colour code
    pm = np.zeros((16, 16, 3), dtype=np.int32)
    pm[2:4, 2:4] = (255, 255, 0)  # beak
    pm[8:12, 8:12] = (170, 170, 170)  # body
    m = part_masks(pm)
    total = sum(v.astype(int) for v in m.values())
    assert (total == 1).all() and m["beak"].sum() >= 4 and m["body"].any() and m["background"].any()
