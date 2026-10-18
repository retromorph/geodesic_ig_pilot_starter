"""E2: FunnyBirds adapter — provenance, selective download, reconstruction preflight, part-level AP.

Data/code provenance (checked 2026-10-18; see reports/reproducibility.md):
  * visinf/funnybirds-framework @ 1350bbe812ff18df25b2708df861245b9a63b9c9 (Apache-2.0)
  * visinf/funnybirds @ 91b4b4628ffa962148144ee6bb5af5f022cac2f8 (Apache-2.0)
  * dataset zip and ResNet-50 checkpoint from download.visinf.tu-darmstadt.de; no explicit
    dataset license is published (recorded as a limitation). No checksums are published, so we
    record ETag/Last-Modified and our own SHA-256.
We do not import the framework (its loader imports `base64.decodestring`, removed in Python 3.9);
the minimal loader below follows its documented file layout.

The Blender renderer is not differentiable. Paths live in the latent space of a compact
autoencoder trained on a FunnyBirds *train* subset; all methods explain the *reconstructed*
endpoints D(E(b)), D(E(x)) and completeness is checked against those. Pre-rendered part
interventions define the independent model-response part importance.
"""

from __future__ import annotations

import io
import json
import math
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import rng_for, sha256_file, torch_seed_for
from .geometry import MetricTerm, PathOptConfig, discrete_energy, linear_waypoints, optimize_paths
from .integration import pixel_path_attribution, refine_until_complete
from .tracking import ResourceMeter, atomic_write_bytes, atomic_write_json, utcnow, write_status

BASE = "https://download.visinf.tu-darmstadt.de/data/funnybirds"
ZIP_URL = f"{BASE}/FunnyBirds.zip"
CKPT_URL = f"{BASE}/models/resnet50_final_0_checkpoint_best.pth.tar"
RESNET18_URL = (
    "https://download.pytorch.org/models/resnet18-f37072fd.pth"  # torchvision IMAGENET1K_V1 (BSD-3)
)
FRAMEWORK_COMMIT = "1350bbe812ff18df25b2708df861245b9a63b9c9"
FUNNYBIRDS_COMMIT = "91b4b4628ffa962148144ee6bb5af5f022cac2f8"
PARTS = ("beak", "eye", "foot", "tail", "wing")
PART_COLORS = {
    "eye": [(255, 255, 253), (255, 255, 254)],
    "beak": [(255, 255, 0)],
    "foot": [(255, 0, 1), (255, 0, 2)],
    "wing": [(0, 255, 1), (0, 255, 2)],
    "tail": [(0, 0, 255)],
}
BODY_COLOR = (170, 170, 170)
GROUPS = (*PARTS, "body", "background")
TRAIN_PER_CLASS = 100
N_PREFLIGHT, N_VAL, N_TEST = 12, 20, 120
PART_FIDELITY_RATIO = 0.5  # prespecified: reconstruction must be >=2x closer to the true part state
MIN_PART_PIXELS = 20
AE_EPOCHS = 30


# --------------------------------------------------------------------------- selective zip download
class _RangeReader(io.RawIOBase):
    def __init__(self, url: str, size: int) -> None:
        self.url, self.size, self.pos = url, size, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b) -> int:
        if self.pos >= self.size:
            return 0
        end = min(self.size, self.pos + len(b)) - 1
        data = _get_range(self.url, self.pos, end)
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def _get_range(url: str, start: int, end: int, retries: int = 4) -> bytes:
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception:
            if k == retries - 1:
                raise
            time.sleep(2 * (k + 1))
    raise RuntimeError("unreachable")


def _head(url: str) -> dict[str, str]:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return {k: r.headers.get(k) for k in ("Content-Length", "ETag", "Last-Modified", "Accept-Ranges")}


def _download(url: str, dst: Path) -> str:
    if not dst.exists():
        tmp = dst.with_suffix(dst.suffix + ".part")
        with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as fh:
            while chunk := r.read(1 << 20):
                fh.write(chunk)
        tmp.rename(dst)
    return sha256_file(dst)


class RemoteZip:
    """Reads the central directory over HTTP ranges and fetches selected members in block requests."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.size = int(_head(url)["Content-Length"])
        raw = io.BufferedReader(_RangeReader(url, self.size), buffer_size=1 << 20)
        self.zf = zipfile.ZipFile(raw)
        self.infos = {i.filename: i for i in self.zf.infolist()}

    def fetch(self, names: list[str], out_dir: Path, max_gap: int = 1 << 20) -> int:
        """Extract members to out_dir (path below the zip root), grouping nearby members per request."""
        todo = sorted(
            (self.infos[n] for n in names if not (out_dir / n).exists()), key=lambda i: i.header_offset
        )
        nbytes, i = 0, 0
        while i < len(todo):
            j = i
            start = todo[i].header_offset
            end = start + 30 + len(todo[i].filename) + 256 + todo[i].compress_size
            while (
                j + 1 < len(todo) and todo[j + 1].header_offset - end < max_gap and end - start < (32 << 20)
            ):
                j += 1
                end = todo[j].header_offset + 30 + len(todo[j].filename) + 256 + todo[j].compress_size
            blob = _get_range(self.url, start, min(end, self.size) - 1)
            nbytes += len(blob)
            for info in todo[i : j + 1]:
                off = info.header_offset - start
                fn_len = int.from_bytes(blob[off + 26 : off + 28], "little")
                ex_len = int.from_bytes(blob[off + 28 : off + 30], "little")
                data_start = off + 30 + fn_len + ex_len
                comp = blob[data_start : data_start + info.compress_size]
                if info.compress_type == zipfile.ZIP_STORED:
                    data = comp
                elif info.compress_type == zipfile.ZIP_DEFLATED:
                    import zlib

                    data = zlib.decompress(comp, -15)
                else:
                    raise RuntimeError(f"unsupported compression {info.compress_type}")
                if zipfile.crc32(data) & 0xFFFFFFFF != info.CRC:
                    raise RuntimeError(f"CRC mismatch for {info.filename}")
                atomic_write_bytes(out_dir / info.filename, data)
            i = j + 1
        return nbytes


# --------------------------------------------------------------------------- data helpers
def _img(path: Path) -> torch.Tensor:
    from PIL import Image

    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _partmap(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.int32)


def part_masks(pm: np.ndarray) -> dict[str, np.ndarray]:
    """Dilated (5x5) part masks as in the framework, made a partition: parts > body > background."""
    out: dict[str, np.ndarray] = {}
    taken = np.zeros(pm.shape[:2], dtype=bool)
    t = torch.tensor
    for p in PARTS:
        m = np.zeros(pm.shape[:2], dtype=bool)
        for c in PART_COLORS[p]:
            m |= np.all(pm == np.array(c), axis=-1)
        dil = F.max_pool2d(t(m, dtype=torch.float32)[None, None], 5, 1, 2)[0, 0].numpy() > 0
        out[p] = dil & ~taken
        taken |= out[p]
    body = np.all(pm == np.array(BODY_COLOR), axis=-1)
    body = (F.max_pool2d(t(body, dtype=torch.float32)[None, None], 5, 1, 2)[0, 0].numpy() > 0) & ~taken
    out["body"] = body
    out["background"] = ~(taken | body)
    return out


def _paths_for(root: Path, cls: int, idx: int) -> dict[str, Path]:
    return {
        "image": root / "FunnyBirds" / "test" / str(cls) / f"{idx:06d}.png",
        "part_map": root / "FunnyBirds" / "test_part_map" / str(cls) / f"{idx:06d}.png",
        "interventions": root / "FunnyBirds" / "test_interventions" / str(cls) / f"{idx:06d}",
    }


def intervention_name(kept: list[str] | tuple[str, ...]) -> str:
    return "body_" + "_".join(sorted(kept)) + ".png"


# --------------------------------------------------------------------------- models
def load_classifier(ckpt: Path, device: str) -> nn.Module:
    from torchvision.models import resnet50

    m = resnet50(num_classes=50)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"]
    m.load_state_dict(sd)
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def class_margin(model: nn.Module, x: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
    """Smooth contrastive target: logit_c - logsumexp_{j != c} logit_j (target class = true class of x)."""
    logits = model(x)
    lc = logits.gather(1, cls.view(-1, 1)).squeeze(1)
    other = logits.scatter(1, cls.view(-1, 1), float("-inf"))
    return lc - torch.logsumexp(other, dim=1)


def _cblock(cin, cout, stride):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1), nn.GroupNorm(8, cout), nn.SiLU())


class SpatialAE(nn.Module):
    """Compact conv AE, 256x256x3 -> 32x32x8 latent (stride 8) -> 256x256x3."""

    def __init__(self, w: int = 64, zc: int = 8) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            _cblock(3, w, 1),
            _cblock(w, w, 2),
            _cblock(w, 2 * w, 2),
            _cblock(2 * w, 4 * w, 2),
            _cblock(4 * w, 4 * w, 1),
            nn.Conv2d(4 * w, zc, 1),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(zc, 4 * w, 3, 1, 1),
            nn.SiLU(),
            _cblock(4 * w, 4 * w, 1),
            nn.Upsample(scale_factor=2),
            _cblock(4 * w, 2 * w, 1),
            nn.Upsample(scale_factor=2),
            _cblock(2 * w, w, 1),
            nn.Upsample(scale_factor=2),
            _cblock(w, w, 1),
            nn.Conv2d(w, 3, 3, 1, 1),
        )

    def encode(self, x):
        return self.enc(x - 0.5)

    def decode(self, z):
        return torch.sigmoid(self.dec(z))

    def forward(self, x):
        return self.decode(self.encode(x))


def load_resnet18_features(path: Path, device: str) -> nn.Module:
    from torchvision.models import resnet18

    m = resnet18()
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    feat = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3).eval().to(device)
    for p in feat.parameters():
        p.requires_grad_(False)
    return feat


# --------------------------------------------------------------------------- prepare-e2
def _select(ctx, meta_test: list[dict], root: Path, rz: RemoteZip | None) -> dict[str, list[dict]]:
    """Deterministic, model-blind selection of preflight/validation/test pairs from the test split.

    For each selected test image: removed set R = random 2- or 3-subset of parts with >= 20 visible
    pixels in its part map; baseline = pre-rendered intervention with R removed; input = full image.
    """
    rng = rng_for("e2", ctx.gen_seed)
    order = rng.permutation(len(meta_test))
    need = N_PREFLIGHT + N_VAL + N_TEST
    picked: list[dict] = []
    skipped = 0
    for gi in order:
        if len(picked) >= need:
            break
        cls = int(meta_test[gi]["class_idx"])
        p = _paths_for(root, cls, int(gi))
        if rz is not None:
            rz.fetch([str(p["part_map"].relative_to(root))], root)
        masks = part_masks(_partmap(p["part_map"]))
        visible = [q for q in PARTS if masks[q].sum() >= MIN_PART_PIXELS]
        if len(visible) < 2:
            skipped += 1
            continue
        k = int(rng.integers(2, 4)) if len(visible) >= 3 else 2
        removed = sorted(rng.choice(visible, size=k, replace=False).tolist())
        picked.append(
            {
                "image_idx": int(gi),
                "class_idx": cls,
                "removed": removed,
                "visible": visible,
                "scene_id": f"fb-{int(gi):06d}",
            }
        )
    return {
        "preflight": picked[:N_PREFLIGHT],
        "validation": picked[N_PREFLIGHT : N_PREFLIGHT + N_VAL],
        "test": picked[N_PREFLIGHT + N_VAL :],
        "skipped_lt2_visible_parts": skipped,
    }


def _needed_members(root: Path, items: list[dict]) -> list[str]:
    names = []
    for it in items:
        p = _paths_for(root, it["class_idx"], it["image_idx"])
        names += [str(p["image"].relative_to(root)), str(p["part_map"].relative_to(root))]
        keep_base = [q for q in PARTS if q not in it["removed"]]
        R = it["removed"]
        for s in range(2 ** len(R)):
            kept = keep_base + [R[j] for j in range(len(R)) if (s >> j) & 1]
            names.append(str((p["interventions"] / intervention_name(kept)).relative_to(root)))
            # single-part toggles used by the part-fidelity check
        for q in PARTS:
            for kept in (
                [x for x in PARTS if x != q],
                [x for x in keep_base if x != q] + ([q] if q in R else []),
            ):
                names.append(str((p["interventions"] / intervention_name(kept)).relative_to(root)))
    return sorted(set(names))


def prepare_e2(ctx, resume: bool = False, dry_run: bool = False) -> int:
    root = ctx.paths.data_e2
    if dry_run:
        print(
            json.dumps(
                {
                    "stage": "prepare-e2",
                    "downloads": [ZIP_URL + " (selected members only)", CKPT_URL, RESNET18_URL],
                    "est_bytes": "~0.4-0.6 GB",
                }
            )
        )
        return 0
    import shutil

    free_gb = shutil.disk_usage(root).free / 2**30
    if free_gb < 5:
        write_status(ctx.paths, "prepare-e2", "BLOCKED", reason=f"insufficient disk {free_gb:.1f} GB")
        return 1
    t0 = time.time()
    prov: dict[str, Any] = {
        "framework_commit": FRAMEWORK_COMMIT,
        "funnybirds_commit": FUNNYBIRDS_COMMIT,
        "code_license": "Apache-2.0 (both repositories)",
        "data_license": "none published; repository Apache-2.0 (inference, recorded limitation)",
        "zip_head": _head(ZIP_URL),
        "ckpt_head": _head(CKPT_URL),
    }
    rz = RemoteZip(ZIP_URL)
    prov["zip_members"] = len(rz.infos)
    base = ["FunnyBirds/classes.json", "FunnyBirds/parts.json", "FunnyBirds/dataset_test.json"]
    nbytes = rz.fetch(base, root)
    meta_test = json.loads((root / "FunnyBirds/dataset_test.json").read_text())
    sel = _select(ctx, meta_test, root, rz)
    allitems = sel["preflight"] + sel["validation"] + sel["test"]
    nbytes += rz.fetch(_needed_members(root, allitems), root)
    # Unlabeled AE training subset: first TRAIN_PER_CLASS train images of every class (train split only).
    train_names = [
        f"FunnyBirds/train/{c}/{c * 1000 + i:06d}.png" for c in range(50) for i in range(TRAIN_PER_CLASS)
    ]
    train_names = [n for n in train_names if n in rz.infos]
    nbytes += rz.fetch(train_names, root)
    (root / "models").mkdir(exist_ok=True)
    prov["resnet50_sha256"] = _download(
        CKPT_URL, root / "models" / "resnet50_final_0_checkpoint_best.pth.tar"
    )
    prov["resnet18_imagenet_sha256"] = _download(RESNET18_URL, root / "models" / "resnet18-f37072fd.pth")
    prov["resnet18_hash_prefix_matches_filename"] = prov["resnet18_imagenet_sha256"].startswith("f37072fd")
    # Model correctness: top-1 on the selected original test images.
    dev = ctx.device
    clf = load_classifier(root / "models" / "resnet50_final_0_checkpoint_best.pth.tar", dev)
    correct = []
    with torch.no_grad():
        for it in allitems:
            x = _img(_paths_for(root, it["class_idx"], it["image_idx"])["image"])[None].to(dev)
            correct.append(int(clf(x).argmax(1).item() == it["class_idx"]))
    prov["classifier_top1_on_selected_originals"] = float(np.mean(correct))
    sel_meta = {**sel, "rule": _select.__doc__, "selection_stream": "e2", "locked_utc": utcnow()}
    atomic_write_json(root / "e2_selection.json", sel_meta)
    prov.update(
        {
            "bytes_downloaded": nbytes,
            "n_train_images": len(train_names),
            "seconds": time.time() - t0,
            "selection_sha256": sha256_file(root / "e2_selection.json"),
        }
    )
    atomic_write_json(root / "e2_provenance.json", prov)
    ok = prov["classifier_top1_on_selected_originals"] >= 0.9
    write_status(
        ctx.paths,
        "prepare-e2",
        "COMPLETE" if ok else "BLOCKED",
        **{k: v for k, v in prov.items() if k != "zip_head"},
    )
    return 0 if ok else 1


# --------------------------------------------------------------------------- evaluate-e2
def _train_ae(ctx, root: Path, dev: str) -> tuple[SpatialAE, dict]:
    ck = root / "models" / "e2_spatial_ae.pt"
    files = sorted((root / "FunnyBirds" / "train").rglob("*.png"))
    ae = SpatialAE().to(dev)
    if ck.exists():
        ae.load_state_dict(torch.load(ck, map_location=dev, weights_only=True))
        meta = json.loads((root / "models" / "e2_spatial_ae.json").read_text())
    else:
        torch.manual_seed(torch_seed_for("e2", ctx.gen_seed, 1))
        X = torch.stack([_img(f) for f in files])  # (N,3,256,256) ~ 4 GB for 5000 images in fp32 (CPU RAM)
        opt = torch.optim.AdamW(ae.parameters(), lr=1e-3)
        steps = AE_EPOCHS * math.ceil(len(X) / 16)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
        g = torch.Generator().manual_seed(7)
        curve = []
        t0 = time.time()
        for ep in range(AE_EPOCHS):
            perm = torch.randperm(len(X), generator=g)
            tot = 0.0
            for s in range(0, len(X), 16):
                x = X[perm[s : s + 16]].to(dev, non_blocking=True)
                r = ae(x)
                loss = F.mse_loss(r, x) + 0.5 * F.l1_loss(r, x)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                tot += float(loss.detach()) * len(x)
            curve.append(tot / len(X))
            print(f"[e2-ae] epoch {ep} loss {curve[-1]:.5f} ({time.time() - t0:.0f}s)", flush=True)
        buf = io.BytesIO()
        torch.save(ae.state_dict(), buf)
        atomic_write_bytes(ck, buf.getvalue())
        meta = {
            "n_train": len(X),
            "epochs": AE_EPOCHS,
            "curve": curve,
            "seconds": time.time() - t0,
            "sha256": sha256_file(ck),
            "objective": "MSE + 0.5 L1 reconstruction, unlabeled train subset",
        }
        atomic_write_json(root / "models" / "e2_spatial_ae.json", meta)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae, meta


def _endpoint_checks(ae, clf, root: Path, it: dict, dev: str) -> dict:
    """Top-1 preservation and threshold-free part fidelity for both endpoints of one pair."""
    p = _paths_for(root, it["class_idx"], it["image_idx"])
    masks = part_masks(_partmap(p["part_map"]))
    keep_base = [q for q in PARTS if q not in it["removed"]]
    x = _img(p["image"])
    b = _img(p["interventions"] / intervention_name(keep_base))
    out = {}
    with torch.no_grad():
        for name, img, kept in (("x", x, list(PARTS)), ("b", b, keep_base)):
            rec = ae(img[None].to(dev))[0].cpu()
            pred_o = int(clf(img[None].to(dev)).argmax(1))
            pred_r = int(clf(rec[None].to(dev)).argmax(1))
            part_ok = {}
            for q in PARTS:
                m = torch.from_numpy(masks[q])
                if m.sum() < MIN_PART_PIXELS:
                    continue
                toggled = [z for z in kept if z != q] if q in kept else kept + [q]
                alt = _img(p["interventions"] / intervention_name(toggled))
                d_true = float(((rec - img)[:, m] ** 2).sum())
                d_alt = float(((rec - alt)[:, m] ** 2).sum())
                part_ok[q] = bool(d_true <= PART_FIDELITY_RATIO * d_alt) if d_alt > 0 else True
            out[name] = {
                "top1_original": pred_o,
                "top1_reconstruction": pred_r,
                "top1_preserved": pred_o == pred_r,
                "mse": float(((rec - img) ** 2).mean()),
                "parts_ok": part_ok,
                "all_parts_ok": all(part_ok.values()),
            }
    return out


def _pair_paths_and_ap(ae, clf, feat, root, it, dev, num: dict, alpha: float, tau: float) -> list[dict]:
    """M1-M4 on identical reconstructed endpoints; pixel attributions grouped by part masks; part AP."""
    p = _paths_for(root, it["class_idx"], it["image_idx"])
    masks = part_masks(_partmap(p["part_map"]))
    gmask = torch.from_numpy(np.stack([masks[g] for g in GROUPS])).to(dev)
    keep_base = [q for q in PARTS if q not in it["removed"]]
    x = _img(p["image"])[None].to(dev)
    b = _img(p["interventions"] / intervention_name(keep_base))[None].to(dev)
    cls = torch.tensor([it["class_idx"]], device=dev)
    f_img = lambda im: class_margin(clf, im, cls.expand(im.shape[0]))  # noqa: E731
    # Independent part importance: classifier on the *rendered* single-part restorations of b.
    with torch.no_grad():
        v0 = float(f_img(b))
        eff = {
            q: float(f_img(_img(p["interventions"] / intervention_name(keep_base + [q]))[None].to(dev))) - v0
            for q in it["removed"]
        }
    with torch.no_grad():
        zb, zx = ae.encode(b), ae.encode(x)
    shape = zb.shape[1:]
    flat_b, flat_x = zb.flatten(1), zx.flatten(1)
    dec = lambda z: ae.decode(z.view(-1, *shape))  # noqa: E731
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    with torch.no_grad():
        lam_D = float(((dec(flat_x) - dec(flat_b)) ** 2).sum() / ((flat_x - flat_b) ** 2).sum())
        fx, fb = feat((dec(flat_x) - mean) / std), feat((dec(flat_b) - mean) / std)
        lam_F = float(((fx - fb) ** 2).sum() / ((flat_x - flat_b) ** 2).sum())
    img_t = MetricTerm("image", lambda z: dec(z).flatten(1) / math.sqrt(lam_D), 1.0)
    sem_t = MetricTerm("feat", lambda z: feat((dec(z) - mean) / std).flatten(1) / math.sqrt(lam_F), 1.0)
    cfg = PathOptConfig(
        segments=num["segments"],
        iterations=num["iterations"],
        lr=num["lr"],
        restarts=num["restarts"],
        perturb_scale=num["perturb_scale"],
        box_lo=-1e4,
        box_hi=1e4,
    )
    seeds = [torch_seed_for("e2", 0, it["image_idx"])]
    paths = {"M2": linear_waypoints(flat_b, flat_x, num["segments"])}
    timing = {}
    for name, terms in (("M3", [img_t]), ("M4", [sem_t, MetricTerm("image", img_t.fn, alpha)])):
        t0 = time.time()
        res = optimize_paths(flat_b, flat_x, lambda Z, t=terms: discrete_energy(Z, t, 1e-3), cfg, seeds)
        paths[name] = res["waypoints"]
        timing[name] = time.time() - t0
    rows = []
    with torch.no_grad():
        rb, rx = dec(flat_b), dec(flat_x)
        f_rb, f_rx = float(f_img(rb)), float(f_img(rx))
    denom = torch.tensor([abs(f_rx - f_rb) + 1e-8])
    for name in ("M1", "M2", "M3", "M4"):
        t0 = time.time()
        if name == "M1":
            X = torch.stack([rb[0], rx[0]])[None]
        else:
            with torch.no_grad():
                X = torch.cat([dec(w[None]) for w in paths[name][0]])[None]  # decoded waypoints (1,K+1,3,H,W)

        def compute(N, idx, X=X):
            return pixel_path_attribution(f_img, X[idx.to(X.device)], gmask, N, max_batch=16)

        res, n_used = refine_until_complete(compute, denom, (64, 128, 256), 0.005)
        A = res["A"][0].cpu().numpy()
        A_parts = {g: float(A[i]) for i, g in enumerate(GROUPS)}
        ap = part_ap(A_parts, eff, tau)
        rows.append(
            {
                "scene_id": it["scene_id"],
                "method": name,
                "removed": ",".join(it["removed"]),
                **{f"A_{g}": A_parts[g] for g in GROUPS},
                **{f"eff_{q}": eff.get(q, np.nan) for q in PARTS},
                "f_rb": f_rb,
                "f_rx": f_rx,
                "completeness_rel": float(abs(res["residual"][0]) / denom[0]),
                "N_used": int(n_used[0]),
                "AP": ap,
                "ap_undefined": ap is None,
                "sec": time.time() - t0 + timing.get(name, 0.0),
            }
        )
    return rows


def part_ap(A: dict[str, float], eff: dict[str, float], tau: float) -> float | None:
    """AP of ranking the 5 parts by signed attribution (descending) against labels
    relevant(p) = p was removed in b and restoring it alone raises the target margin by >= tau.
    Returns None (UNDEFINED_LABEL) when no part is relevant."""
    labels = {q: int(q in eff and eff[q] >= tau) for q in PARTS}
    if sum(labels.values()) == 0:
        return None
    order = sorted(PARTS, key=lambda q: (-A[q], q))
    hits, precs = 0, []
    for k, q in enumerate(order, 1):
        if labels[q]:
            hits += 1
            precs.append(hits / k)
    return float(np.mean(precs))


def evaluate_e2(ctx, phase: str = "preflight", resume: bool = False, dry_run: bool = False) -> int:
    root = ctx.paths.data_e2
    if dry_run:
        print(
            json.dumps(
                {
                    "stage": f"evaluate-e2:{phase}",
                    "preflight_pairs": N_PREFLIGHT,
                    "val": N_VAL,
                    "test": N_TEST,
                    "budget_gpu_hours": 10,
                }
            )
        )
        return 0
    if not (root / "e2_selection.json").exists():
        raise FileNotFoundError("run prepare-e2 first")
    sel = json.loads((root / "e2_selection.json").read_text())
    dev = ctx.device
    clf = load_classifier(root / "models" / "resnet50_final_0_checkpoint_best.pth.tar", dev)
    feat = load_resnet18_features(root / "models" / "resnet18-f37072fd.pth", dev)
    dec_p = ctx.paths.reports / "decision.json"
    gate_b = json.loads(dec_p.read_text())["gate_b"] if dec_p.exists() else None
    with ResourceMeter(ctx.paths, f"evaluate-e2-{phase}", dev) as meter:
        ae, ae_meta = _train_ae(ctx, root, dev)
        checks = [
            {"scene_id": it["scene_id"], **_endpoint_checks(ae, clf, root, it, dev)}
            for it in sel["preflight"]
        ]
        ends = [c[k] for c in checks for k in ("b", "x")]
        top1 = float(np.mean([e["top1_preserved"] for e in ends]))
        parts = float(np.mean([e["all_parts_ok"] for e in ends]))
        pair_ok = float(np.mean([c["b"]["all_parts_ok"] and c["x"]["all_parts_ok"] for c in checks]))
        preflight_pass = top1 >= 0.9 and parts >= 0.9
        pre = {
            "top1_preserved_fraction": top1,
            "part_fidelity_fraction": parts,
            "pairs_both_endpoints_ok": pair_ok,
            "median_endpoint_mse": float(np.median([e["mse"] for e in ends])),
            "preflight_pass": preflight_pass,
            "part_fidelity_rule": f"reconstruction part-region SSE <= {PART_FIDELITY_RATIO} x SSE to the "
            "toggled-part render (prespecified, not tuned)",
            "checks": checks,
            "ae": {k: v for k, v in ae_meta.items() if k != "curve"},
        }
        # Exploratory path smoke on the 12 preflight pairs (never a gate input).
        num = {"segments": 16, "iterations": 100, "lr": 0.05, "restarts": 3, "perturb_scale": 0.1}
        smoke_rows = []
        for it in sel["preflight"][:4]:
            smoke_rows += _pair_paths_and_ap(ae, clf, feat, root, it, dev, num, alpha=1.0, tau=0.5)
        pre["path_smoke"] = smoke_rows
        pre["resources"] = meter.summary()
    atomic_write_json(ctx.paths.metrics / "e2_preflight.json", pre)
    gb_pass = bool(gate_b and gate_b.get("status") == "PASS")
    if not preflight_pass:
        gc = {
            "status": "INCONCLUSIVE",
            "reasons": ["E2 preflight failed (decoder does not preserve endpoints/parts)", "E2_INCONCLUSIVE"],
        }
        status = "E2_INCONCLUSIVE"
    elif not gb_pass:
        gc = {
            "status": "NOT_RUN_BY_DESIGN",
            "reasons": ["Gate B did not pass; full E2 not run (sequential stop)", "preflight passed"],
        }
        status = "NOT_RUN_BY_DESIGN"
    else:
        gc = {
            "status": "INCONCLUSIVE",
            "reasons": ["full E2 run required: execute `evaluate-e2 --phase full`"],
        }
        status = "COMPLETE"
    if phase == "full" and preflight_pass and gb_pass:
        raise NotImplementedError("full E2 is only reached when Gate B passes; see reports/deviations.md")
    write_status(
        ctx.paths,
        "evaluate-e2",
        status,
        phase=phase,
        gate_c=gc,
        preflight={k: v for k, v in pre.items() if k not in ("checks", "path_smoke")},
        path_smoke_summary=_smoke_summary(smoke_rows),
    )
    return 0


def endpoint_margin_distortion(ctx) -> dict:
    """Target-margin change caused by AE reconstruction on the 24 preflight endpoints (diagnostic)."""
    root = ctx.paths.data_e2
    sel = json.loads((root / "e2_selection.json").read_text())
    dev = ctx.device
    clf = load_classifier(root / "models" / "resnet50_final_0_checkpoint_best.pth.tar", dev)
    ae, _ = _train_ae(ctx, root, dev)  # loads the saved checkpoint; never retrains
    rows = []
    with torch.no_grad():
        for it in sel["preflight"]:
            p = _paths_for(root, it["class_idx"], it["image_idx"])
            keep_base = [q for q in PARTS if q not in it["removed"]]
            cls = torch.tensor([it["class_idx"]], device=dev)
            for name, path in (("x", p["image"]), ("b", p["interventions"] / intervention_name(keep_base))):
                img = _img(path)[None].to(dev)
                f_o, f_r = float(class_margin(clf, img, cls)), float(class_margin(clf, ae(img), cls))
                rows.append(
                    {"scene_id": it["scene_id"], "endpoint": name, "f_original": f_o, "f_reconstructed": f_r}
                )
            rows[-2]["contrast_original"] = rows[-2]["f_original"] - rows[-1]["f_original"]
            rows[-2]["contrast_reconstructed"] = rows[-2]["f_reconstructed"] - rows[-1]["f_reconstructed"]
    d = np.array([abs(r["f_reconstructed"] - r["f_original"]) for r in rows])
    co = np.array([r["contrast_original"] for r in rows if "contrast_original" in r])
    cr = np.array([r["contrast_reconstructed"] for r in rows if "contrast_reconstructed" in r])
    out = {
        "rows": rows,
        "median_abs_margin_distortion": float(np.median(d)),
        "max_abs_margin_distortion": float(d.max()),
        "contrast_sign_preserved_fraction": float(np.mean(np.sign(co) == np.sign(cr))),
        "median_abs_contrast_change": float(np.median(np.abs(cr - co))),
    }
    atomic_write_json(ctx.paths.metrics / "e2_preflight_margins.json", out)
    return out


def log_e2_to_mlflow(ctx) -> str:
    """Record E2 provenance, selection, preflight results and statuses as one MLflow run (gig-pilot/e2)."""
    root = ctx.paths.data_e2
    status = json.loads((ctx.paths.status / "evaluate-e2.json").read_text())
    with ctx.tracker.run(
        ctx.exp("e2"),
        "e2-preflight",
        tags={"stage": "evaluate-e2", "split": "preflight", "e2_status": status["status"]},
    ) as run:
        pre = status.get("preflight", {})
        ctx.tracker.log_metrics({k: float(v) for k, v in pre.items() if isinstance(v, (int, float))})
        for f in (
            root / "e2_provenance.json",
            root / "e2_selection.json",
            ctx.paths.metrics / "e2_preflight.json",
            ctx.paths.status / "prepare-e2.json",
            ctx.paths.status / "evaluate-e2.json",
            root / "models" / "e2_spatial_ae.json",
        ):
            if f.exists():
                ctx.tracker.log_artifact(f, "e2")
        return run.info.run_id


def _smoke_summary(rows: list[dict]) -> dict:
    if not rows:
        return {}
    import pandas as pd

    df = pd.DataFrame(rows)
    return {
        "n_pairs": int(df["scene_id"].nunique()),
        "completeness_rel_max": float(df["completeness_rel"].max()),
        "mean_AP_by_method": df.groupby("method")["AP"].mean().to_dict(),
        "ap_undefined": int(df["ap_undefined"].sum()),
        "sec_by_method": df.groupby("method")["sec"].mean().to_dict(),
        "note": "exploratory smoke on 4 preflight pairs; not a gate input",
    }
