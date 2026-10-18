"""MLflow tracking, provenance tags, atomic file writes, stage status and GPU accounting."""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import os
import platform
import socket
import subprocess
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from .config import REPO_ROOT, Paths, Protocol, sha256_file, source_tree_sha256

STATUS_CODES = {
    "COMPLETE",
    "COMPLETE_WITH_FAILURES",
    "NOT_RUN_BY_DESIGN",
    "INVALID_PROTOCOL",
    "BLOCKED",
    "RESOURCE_INFEASIBLE",
    "E2_INCONCLUSIVE",
}


def utcnow() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- atomic IO
def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def _json_default(o: Any) -> Any:
    import numpy as np

    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def atomic_write_json(path: str | Path, obj: Any) -> None:
    atomic_write_bytes(path, json.dumps(obj, indent=2, sort_keys=True, default=_json_default).encode())


def atomic_write_parquet(path: str | Path, df: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    df = df.copy()
    for col in [c for c in df.columns if df[c].dtype == object]:
        df[col] = df[col].where(df[col].notna(), None)  # NaN in text columns -> null
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def write_immutable_json(path: str | Path, obj: Any) -> str:
    """Write a JSON file once. Rewriting with different content is an error."""
    path = Path(path)
    payload = json.dumps(obj, indent=2, sort_keys=True, default=_json_default).encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite immutable file {path}")
        return hashlib.sha256(payload).hexdigest()
    atomic_write_bytes(path, payload)
    os.chmod(path, 0o444)
    return hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------- provenance
def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=20
        ).stdout.strip()
    except Exception:
        return "UNAVAILABLE"


def host_tag() -> str:
    """Hostname hash: identifies the machine across runs without publishing its name/address."""
    return "host-" + hashlib.sha256(socket.gethostname().encode()).hexdigest()[:12]


def gpu_info() -> dict[str, str]:
    info = {"gpu_name": "NONE", "cuda_driver": "NONE", "pytorch_cuda": "NONE", "torch": "NONE"}
    try:
        import torch

        info["torch"] = torch.__version__
        info["pytorch_cuda"] = str(torch.version.cuda)
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if out.returncode == 0:
            info["cuda_driver"] = out.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return info


def provenance(protocol: Protocol) -> dict[str, str]:
    lock = REPO_ROOT / "uv.lock"
    prereg = REPO_ROOT / "docs" / "PREREGISTRATION.md"
    status = _git("status", "--porcelain", "--untracked-files=no")
    prov = {
        "protocol_sha256": protocol.sha256,
        "preregistration_sha256": sha256_file(prereg) if prereg.exists() else "MISSING",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": "UNAVAILABLE" if status == "UNAVAILABLE" else str(bool(status)),
        "uv_lock_sha256": sha256_file(lock) if lock.exists() else "MISSING",
        "source_sha256": source_tree_sha256(),
        "host": host_tag(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    prov.update(gpu_info())
    return prov


# --------------------------------------------------------------------------- MLflow
class _NullRun:
    """Stand-in used only when MLflow is explicitly disabled (local CPU smoke tests)."""

    def __init__(self) -> None:
        self.info = type("Info", (), {"run_id": "MLFLOW_DISABLED", "experiment_id": "MLFLOW_DISABLED"})()


class Tracker:
    """Thin MLflow wrapper. Disabled only with GIG_DISABLE_MLFLOW=1 (recorded in status files)."""

    def __init__(self, protocol: Protocol) -> None:
        self.protocol = protocol
        self.disabled = os.environ.get("GIG_DISABLE_MLFLOW", "0") == "1"
        self.uri = os.environ.get("MLFLOW_TRACKING_URI", protocol.get("environment", "mlflow_tracking_uri"))
        if not self.disabled:
            import mlflow

            if not self.uri.startswith(("http://127.0.0.1", "http://localhost")):
                raise RuntimeError(f"refusing non-loopback MLflow URI {self.uri!r}")
            mlflow.set_tracking_uri(self.uri)

    @contextlib.contextmanager
    def run(
        self,
        experiment: str,
        run_name: str,
        tags: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Iterator[Any]:
        base_tags = provenance(self.protocol)
        base_tags.update({k: str(v) for k, v in (tags or {}).items()})
        if self.disabled:
            yield _NullRun()
            return
        import mlflow

        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags(base_tags)
            if params:
                mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
            mlflow.log_artifact(str(self.protocol.path), artifact_path="protocol")
            yield run

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self.disabled:
            return
        import math

        import mlflow

        clean = {k: float(v) for k, v in metrics.items() if v is not None and math.isfinite(float(v))}
        mlflow.log_metrics(clean, step=step)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        if self.disabled:
            return
        import mlflow

        mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def set_tags(self, tags: dict[str, Any]) -> None:
        if self.disabled:
            return
        import mlflow

        mlflow.set_tags({k: str(v) for k, v in tags.items()})


# --------------------------------------------------------------------------- status + resources
def write_status(paths: Paths, stage: str, status: str, **info: Any) -> Path:
    if status not in STATUS_CODES:
        raise ValueError(f"unknown status {status}")
    obj = {"stage": stage, "status": status, "written_utc": utcnow(), **info}
    out = paths.status / f"{stage}.json"
    atomic_write_json(out, obj)
    return out


def read_status(paths: Paths, stage: str) -> dict | None:
    p = paths.status / f"{stage}.json"
    return json.loads(p.read_text()) if p.exists() else None


class ResourceMeter:
    """Wall-clock and CUDA peak memory for one stage; appends a row to the GPU ledger."""

    def __init__(self, paths: Paths, stage: str, device: str) -> None:
        self.paths, self.stage, self.device = paths, stage, device
        self.oom_events = 0

    def __enter__(self) -> ResourceMeter:
        self.t0 = time.time()
        self.start_utc = utcnow()
        if self.device.startswith("cuda"):
            import torch

            torch.cuda.reset_peak_memory_stats()
        return self

    def summary(self) -> dict[str, Any]:
        wall = time.time() - self.t0
        peak = 0.0
        if self.device.startswith("cuda"):
            import torch

            peak = torch.cuda.max_memory_allocated() / 2**30
        gpu_hours = wall / 3600.0 if self.device.startswith("cuda") else 0.0
        return {
            "stage": self.stage,
            "device": self.device,
            "start_utc": self.start_utc,
            "end_utc": utcnow(),
            "wall_seconds": wall,
            "gpu_hours": gpu_hours,
            "peak_gpu_mem_gb": peak,
            "oom_events": self.oom_events,
        }

    def __exit__(self, exc_type, exc, tb) -> None:
        row = self.summary()
        row["exit"] = "ok" if exc_type is None else f"error:{exc_type.__name__}"
        ledger = self.paths.metrics / "gpu_ledger.jsonl"
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with open(ledger, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        self.last = row


def gpu_hours_used(paths: Paths) -> float:
    ledger = paths.metrics / "gpu_ledger.jsonl"
    if not ledger.exists():
        return 0.0
    return sum(json.loads(line)["gpu_hours"] for line in ledger.read_text().splitlines() if line.strip())
