"""Immutable protocol loading, hashing and output-path conventions.

The YAML protocol is the machine-readable preregistration. We never write to it; we
hash it, check that the decision thresholds still equal the preregistered values,
and expose read-only accessors.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# Preregistered decision thresholds (copied from protocol v1.0). If the YAML no longer
# matches these, the run refuses to start: thresholds may not drift silently.
_EXPECTED = {
    ("protocol", "version"): "1.0",
    ("e1", "gate_b", "strong_mean_relative_reduction_min"): 0.15,
    ("e1", "gate_b", "strong_bootstrap_lower_bound_gt"): 0.05,
    ("e1", "gate_b", "all_mean_relative_reduction_min"): 0.10,
    ("e1", "gate_b", "all_bootstrap_lower_bound_gt"): 0.0,
    ("e1", "gate_b", "weak_relative_reduction_min"): -0.05,
    ("e1", "gate_a", "completeness_median_relative_max"): 0.005,
    ("e1", "gate_a", "completeness_p95_relative_max"): 0.02,
    ("e1", "gate_a", "latent_domain_valid_fraction_min"): 0.98,
    ("e1", "competence", "min_auroc"): 0.90,
    ("e1", "competence", "min_accuracy"): 0.83,
    ("e1", "statistics", "bootstrap_seed"): 9001,
    ("e1", "statistics", "bootstrap_samples"): 2000,
    ("e1", "generator_seed"): 20261018,
    ("e1", "validation_pairs"): 80,
    ("e1", "test_pairs_strong"): 120,
    ("e1", "test_pairs_weak"): 120,
    ("e2", "gate_c", "absolute_part_ap_improvement_min"): 0.03,
    ("budget", "total_gpu_hours"): 40.0,
}

# Named, disjoint RNG streams derived from the single preregistered generator seed.
STREAM_KEYS = {
    "calibration": 1,
    "train_seed11": 2,
    "train_seed29": 3,
    "encoder_train": 4,
    "classifier_val": 5,
    "classifier_test": 6,
    "encoder_val": 7,
    "pairs_validation": 8,
    "pairs_test": 9,
    "m5_paths": 10,
    "path_restarts": 11,
    "metric_normalization": 12,
    "e2": 13,
}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# Modules whose code determines E1 numbers; the frozen manifest binds their hash.
E1_CORE_MODULES = (
    "config.py",
    "datasets.py",
    "e1.py",
    "evaluations.py",
    "geometry.py",
    "integration.py",
    "models.py",
    "renderer.py",
    "scores.py",
)


def source_tree_sha256(modules: tuple[str, ...] = E1_CORE_MODULES) -> str:
    """Order-stable hash of the E1 computational source files. Binds frozen settings to code."""
    h = hashlib.sha256()
    pkg = REPO_ROOT / "src" / "geodesic_ig_pilot"
    for p in sorted(pkg / m for m in modules):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def rng_for(stream: str, generator_seed: int) -> np.random.Generator:
    """Deterministic generator for a named stream (SeedSequence child of the protocol seed)."""
    ss = np.random.SeedSequence(entropy=generator_seed, spawn_key=(STREAM_KEYS[stream],))
    return np.random.default_rng(ss)


def torch_seed_for(stream: str, generator_seed: int, extra: int = 0) -> int:
    ss = np.random.SeedSequence(entropy=generator_seed, spawn_key=(STREAM_KEYS[stream], extra))
    return int(ss.generate_state(1, dtype=np.uint32)[0])


@dataclass(frozen=True)
class Protocol:
    path: Path
    sha256: str
    _data: dict = field(repr=False)

    def get(self, *keys: str) -> Any:
        node: Any = self._data
        for k in keys:
            node = node[k]
        return copy.deepcopy(node)

    @property
    def data(self) -> dict:
        return copy.deepcopy(self._data)


def load_protocol(path: str | Path) -> Protocol:
    path = Path(path)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"protocol file not found: {path}")
    data = yaml.safe_load(path.read_text())
    for keys, expected in _EXPECTED.items():
        node: Any = data
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                raise ValueError(f"protocol missing required key {'.'.join(keys)}")
            node = node[k]
        if node != expected:
            raise ValueError(
                f"protocol value {'.'.join(keys)}={node!r} differs from preregistered {expected!r}; "
                "thresholds may only change through a versioned amendment"
            )
    return Protocol(path=path, sha256=sha256_file(path), _data=data)


@dataclass(frozen=True)
class Paths:
    """Output layout below an output root (default: repository root)."""

    root: Path

    @property
    def data_e1(self) -> Path:
        return self.root / "data" / "e1"

    @property
    def data_e2(self) -> Path:
        return self.root / "data" / "e2"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def metrics(self) -> Path:
        return self.reports / "metrics"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    @property
    def status(self) -> Path:
        return self.reports / "status"

    @property
    def manifest_dir(self) -> Path:
        return self.reports / "manifest"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def ensure(self) -> Paths:
        for p in (
            self.data_e1,
            self.data_e2,
            self.checkpoints,
            self.metrics,
            self.figures,
            self.status,
            self.manifest_dir,
            self.logs,
        ):
            p.mkdir(parents=True, exist_ok=True)
        return self
