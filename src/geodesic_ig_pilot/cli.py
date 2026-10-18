"""Command-line entry point: `uv run python -m geodesic_ig_pilot.cli <command> --config ...`."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from .config import REPO_ROOT, Paths, load_protocol
from .tracking import (
    ResourceMeter,
    Tracker,
    atomic_write_json,
    atomic_write_parquet,
    gpu_hours_used,
    provenance,
    write_status,
)


def _ctx(args):
    from .e1 import Ctx

    protocol = load_protocol(args.config)
    root = (
        Path(args.output_root)
        if args.output_root
        else (REPO_ROOT / "outputs" / f"smoke-{args.device.split(':')[0]}" if args.smoke else REPO_ROOT)
    )
    paths = Paths(root.resolve()).ensure()
    return Ctx(
        protocol=protocol, paths=paths, device=args.device, tracker=Tracker(protocol), smoke=args.smoke
    )


# --------------------------------------------------------------------------- doctor
def cmd_doctor(args) -> int:
    ctx = _ctx(args)
    import torch

    info = {"provenance": provenance(ctx.protocol), "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_total_vram_gb"] = props.total_memory / 2**30
        x = torch.randn(1024, 1024, device="cuda")
        info["cuda_matmul_ok"] = bool(torch.isfinite(x @ x).all())
        info["cudnn_version"] = torch.backends.cudnn.version()
    du = shutil.disk_usage(ctx.paths.root)
    info["disk_free_gb"] = du.free / 2**30
    info["mlflow_uri"] = ctx.tracker.uri
    info["mlflow_disabled"] = ctx.tracker.disabled
    if not ctx.tracker.disabled:
        import urllib.request

        try:
            with urllib.request.urlopen(ctx.tracker.uri + "/health", timeout=5) as r:
                info["mlflow_health"] = r.read().decode()[:20]
        except Exception as e:  # noqa: BLE001
            info["mlflow_health"] = f"UNREACHABLE: {type(e).__name__}"
    e1 = ctx.protocol.get("e1")
    info["expected"] = {
        "validation_pairs": e1["validation_pairs"],
        "test_pairs": e1["test_pairs_strong"] + e1["test_pairs_weak"],
        "classifier_seeds": e1["classifier_seeds"],
    }
    used = gpu_hours_used(ctx.paths)
    info["gpu_hours_used"] = used
    info["gpu_hours_remaining"] = ctx.protocol.get("budget", "total_gpu_hours") - used
    print(json.dumps(info, indent=2, default=str))
    atomic_write_json(ctx.paths.status / "doctor_report.json", info)
    write_status(ctx.paths, "doctor", "COMPLETE", report="doctor_report.json")
    return 0


# --------------------------------------------------------------------------- smoke-mlflow
def cmd_smoke_mlflow(args) -> int:
    ctx = _ctx(args)
    if ctx.tracker.disabled:
        print(
            "MLflow is disabled (GIG_DISABLE_MLFLOW=1); smoke-mlflow requires a live server", file=sys.stderr
        )
        return 2
    import mlflow

    payload = os.urandom(4096)
    digest = hashlib.sha256(payload).hexdigest()
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "roundtrip.bin"
        src.write_bytes(payload)
        with ctx.tracker.run(
            "gig-pilot/smoke",
            "mlflow-roundtrip",
            tags={"stage": "smoke-mlflow"},
            params={"payload_sha256": digest},
        ) as run:
            ctx.tracker.log_metrics({"smoke_metric": 1.0})
            ctx.tracker.log_artifact(src, "roundtrip")
            run_id, art_uri = run.info.run_id, run.info.artifact_uri
        client = mlflow.tracking.MlflowClient()
        listed = [f.path for f in client.list_artifacts(run_id, "roundtrip")]
        dst = Path(td) / "dl"
        dst.mkdir()
        local = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="roundtrip/roundtrip.bin", dst_path=str(dst)
        )
        back = hashlib.sha256(Path(local).read_bytes()).hexdigest()
        replay = client.get_run(run_id)
    ok = (
        back == digest
        and art_uri.startswith("mlflow-artifacts:")
        and replay.data.params["payload_sha256"] == digest
    )
    res = {
        "run_id": run_id,
        "experiment_id": replay.info.experiment_id,
        "artifact_uri": art_uri,
        "listed": listed,
        "uploaded_sha256": digest,
        "downloaded_sha256": back,
        "proxied_artifacts": art_uri.startswith("mlflow-artifacts:"),
        "roundtrip_ok": ok,
        "replayed_tags": {
            k: replay.data.tags[k] for k in ("protocol_sha256", "git_commit") if k in replay.data.tags
        },
    }
    print(json.dumps(res, indent=2))
    write_status(ctx.paths, "smoke-mlflow", "COMPLETE" if ok else "BLOCKED", **res)
    return 0 if ok else 1


# --------------------------------------------------------------------------- exp0
def cmd_exp0(args) -> int:
    ctx = _ctx(args)
    if args.dry_run:
        print(json.dumps({"stage": "exp0", "device": "cpu", "gpu_hours": 0}))
        return 0
    import pandas as pd

    from .e0 import run_e0

    with (
        ctx.tracker.run(
            "gig-pilot/e0" + ("-smoke" if ctx.smoke else ""),
            "exp0",
            tags={"stage": "exp0", "split": "analytic"},
        ) as run,
        ResourceMeter(ctx.paths, "exp0", "cpu") as meter,
    ):
        res = run_e0(ctx.protocol)
        out = ctx.paths.metrics / "e0_results.json"
        atomic_write_json(out, res)
        flat = pd.DataFrame(
            [{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()} for r in res["rows"]]
        )
        atomic_write_parquet(ctx.paths.metrics / "e0_results.parquet", flat)
        for r in res["rows"]:
            ctx.tracker.log_metrics(
                {
                    f"c{r['c']:g}_optimizer_abs_err_u": r["optimizer_abs_err_u"],
                    f"c{r['c']:g}_additive_err": r["additive_max_abs_err"],
                }
            )
        ctx.tracker.log_artifact(out, "e0")
        run_id = run.info.run_id
    for c in res["checks"]:
        print(("PASS " if c["pass"] else "FAIL ") + c["check"])
    write_status(
        ctx.paths,
        "exp0",
        "COMPLETE" if res["all_pass"] else "COMPLETE_WITH_FAILURES",
        all_pass=res["all_pass"],
        mlflow_run_id=run_id,
        resources=meter.summary(),
    )
    return 0 if res["all_pass"] else 1


# --------------------------------------------------------------------------- E1 stages
def cmd_prepare_e1(args) -> int:
    from .e1 import prepare_e1

    prepare_e1(_ctx(args), resume=args.resume, dry_run=args.dry_run)
    return 0


def cmd_train_e1(args) -> int:
    from .e1 import train_e1

    res = train_e1(_ctx(args), resume=args.resume, dry_run=args.dry_run)
    return 0 if args.dry_run or all(r["competent"] for r in res.values()) else 3


def cmd_fit_encoder(args) -> int:
    from .e1 import fit_encoder

    fit_encoder(_ctx(args), resume=args.resume, dry_run=args.dry_run)
    return 0


def cmd_evaluate_e1(args) -> int:
    from .e1 import evaluate_e1

    evaluate_e1(_ctx(args), split=args.split, resume=args.resume, dry_run=args.dry_run)
    return 0


# --------------------------------------------------------------------------- E2 + reporting
def cmd_prepare_e2(args) -> int:
    from .external import prepare_e2

    return prepare_e2(_ctx(args), resume=args.resume, dry_run=args.dry_run)


def cmd_evaluate_e2(args) -> int:
    from .external import evaluate_e2

    return evaluate_e2(_ctx(args), phase=args.phase, resume=args.resume, dry_run=args.dry_run)


def cmd_aggregate(args) -> int:
    from .report import aggregate

    return aggregate(_ctx(args))


def cmd_report(args) -> int:
    from .report import write_report

    return write_report(_ctx(args))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="geodesic_ig_pilot.cli", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    commands = {
        "doctor": (cmd_doctor, "inspect GPU/CUDA/disk/MLflow/protocol and budget"),
        "smoke-mlflow": (cmd_smoke_mlflow, "MLflow run + artifact upload/download roundtrip"),
        "exp0": (cmd_exp0, "E0 analytic geodesic checks (CPU)"),
        "prepare-e1": (cmd_prepare_e1, "generate E1 scenes, calibration and frozen pair manifest"),
        "train-e1": (cmd_train_e1, "train the two CNN classifiers"),
        "fit-encoder": (cmd_fit_encoder, "train the unlabeled masked-AE representation"),
        "evaluate-e1": (cmd_evaluate_e1, "path attributions on validation (select+freeze) or test"),
        "prepare-e2": (cmd_prepare_e2, "FunnyBirds provenance checks and subset download"),
        "evaluate-e2": (cmd_evaluate_e2, "E2 preflight / full run (conditional on Gate B)"),
        "aggregate": (cmd_aggregate, "gates, bootstrap, metrics tables and figures"),
        "report": (cmd_report, "write reports/FINAL_REPORT.md from aggregated results"),
    }
    for name, (fn, help_) in commands.items():
        p = sub.add_parser(name, help=help_)
        p.add_argument("--config", required=True)
        p.add_argument("--output-root", default=None)
        p.add_argument("--device", default="cuda" if name not in ("exp0",) else "cpu")
        p.add_argument("--resume", action="store_true")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument(
            "--smoke", action="store_true", help="tiny CPU/GPU smoke sizes in a separate output root"
        )
        if name == "evaluate-e1":
            p.add_argument("--split", choices=["validation", "test"], required=True)
        if name == "evaluate-e2":
            p.add_argument("--phase", choices=["preflight", "full"], default="preflight")
        p.set_defaults(func=fn)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
