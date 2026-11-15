"""Shared loaders, the independent CSS evaluator and the sphere projection.

The evaluator is a direct numpy transcription of analyzePRF's model
(makegaussian2d / 2*pi*sigma^2, posrect, .^n, conv2run with the HRF, per-run
polynomial projection, calccod) in float64. It shares no code with ffs_pyrf,
so scoring both tools' parameters with it is a common yardstick: whichever
parameter set gives the higher R2 here found the better optimum of the SAME
objective.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.io as sio

SUB = "sub-wlsubj001"
SES = "ses-nyu3t01"
HEMIS = ("lh", "rh")


# ----------------------------------------------------------------------------- loading


@dataclass
class Bucket:
    maps: dict[str, np.ndarray]  # name -> (n_vertices_total,), lh then rh

    def __getitem__(self, key: str) -> np.ndarray:
        return self.maps[key]


def load_bucket(prefix: str | Path) -> Bucket:
    """An ffs surface bucket (.lh/.rh split) as one lh-then-rh array per brick."""
    maps: dict[str, list[np.ndarray]] = {}
    for hemi in HEMIS:
        img = nib.load(f"{prefix}.{hemi}.func.gii")
        for i, arr in enumerate(img.darrays):
            name = arr.meta.get("Name") or f"col{i}"
            maps.setdefault(name, []).append(np.asarray(arr.data, np.float64))
    return Bucket({k: np.concatenate(v) for k, v in maps.items()})


def load_map(prefix: str | Path) -> np.ndarray:
    """A single-brick ffs surface map, lh then rh."""
    return np.concatenate(
        [np.asarray(nib.load(f"{prefix}.{h}.func.gii").darrays[0].data, np.float64) for h in HEMIS]
    )


def load_mgz_pair(pattern: str) -> np.ndarray:
    """``pattern`` with {hemi} -> lh-then-rh vector."""
    return np.concatenate(
        [
            np.asarray(nib.load(pattern.format(hemi=h)).dataobj, np.float64).reshape(-1)
            for h in HEMIS
        ]
    )


def load_runs(work: Path, index: np.ndarray | None = None) -> list[np.ndarray]:
    """Per-run (vertices, time) float64 series, lh then rh, optionally subset."""
    meta = json.loads((work / "meta.json").read_text())
    out = []
    for run in meta["runs"]:
        parts = [
            np.asarray(
                nib.load(work / "runs" / f"run-{run:02d}.{h}.func.gii").agg_data(), np.float32
            )
            for h in HEMIS
        ]
        # agg_data stacks darrays as (V, T)
        d = np.concatenate(parts)
        out.append((d if index is None else d[index]).astype(np.float64))
    return out


def ffs_raw_params(bucket: Bucket, resmx: int = 100) -> np.ndarray:
    """(row, col, sigma, n) in analyzePRF's one-based square frame, from x/y."""
    center = (1 + resmx) / 2
    return np.stack(
        [center - bucket["y"], bucket["x"] + center, bucket["sigma"], bucket["exponent"]], axis=1
    )


@dataclass
class APR:
    index: np.ndarray  # zero-based lh-then-rh vertex index of each fitted row
    params: np.ndarray  # (V, 5) row, col, sigma, gain, n
    r2: np.ndarray  # percent
    ang: np.ndarray
    ecc: np.ndarray
    rfsize: np.ndarray
    expt: np.ndarray
    hrf: np.ndarray
    elapsed: float
    nworkers: int


def load_apr(inp: Path, out: Path, n_lh: int) -> APR:
    i = sio.loadmat(inp, variable_names=["lh_index", "rh_index"])
    o = sio.loadmat(out)
    index = np.concatenate([i["lh_index"].ravel() - 1, i["rh_index"].ravel() - 1 + n_lh])
    # analyzePRF returns params as (1, 5, V) for vertices x time input.
    params = o["params"].astype(np.float64).reshape(5, -1).T
    n = params.shape[0]
    return APR(
        index=index[:n],
        params=params,
        r2=o["R2"].ravel().astype(np.float64),
        ang=o["ang"].ravel().astype(np.float64),
        ecc=o["ecc"].ravel().astype(np.float64),
        rfsize=o["rfsize"].ravel().astype(np.float64),
        expt=o["expt"].ravel().astype(np.float64),
        hrf=o["hrf"].ravel().astype(np.float64),
        elapsed=float(o["elapsed"].squeeze()),
        nworkers=int(o["nworkers"].squeeze()),
    )


# ----------------------------------------------------------------------------- evaluator


def legendre_basis(n: int, degree: int) -> np.ndarray:
    """Orthonormal basis of polynomials 0..degree over n samples."""
    t = np.linspace(-1, 1, n)
    q, _ = np.linalg.qr(np.polynomial.legendre.legvander(t, degree))
    return q


def project_runs(
    series: np.ndarray, run_len: int, degree: int, extra: list[np.ndarray] | None = None
):
    """Project per-run polynomials (and optional per-run extra regressors) out of
    (V, T_total) series. Returns the same shape."""
    out = np.empty_like(series)
    n_runs = series.shape[1] // run_len
    for r in range(n_runs):
        s = slice(r * run_len, (r + 1) * run_len)
        basis = legendre_basis(run_len, degree)
        if extra is not None and extra[r] is not None and extra[r].shape[1]:
            basis, _ = np.linalg.qr(np.concatenate([basis, extra[r]], axis=1))
        block = series[:, s]
        out[:, s] = block - (block @ basis) @ basis.T
    return out


def css_predict(
    params4: np.ndarray,
    aperture: np.ndarray,
    hrf: np.ndarray | list,
    n_runs: int,
    chunk: int = 4096,
) -> np.ndarray:
    """Unit-gain CSS prediction, (V, n_runs*T). ``hrf`` is one kernel or one per vertex."""
    rows, cols, frames = aperture.shape
    stim = aperture.reshape(rows * cols, frames).T.astype(np.float64)  # (T, P), row-major pixels
    rr, cc = np.meshgrid(np.arange(1, rows + 1), np.arange(1, cols + 1), indexing="ij")
    rr = rr.reshape(-1)
    cc = cc.reshape(-1)
    out = np.empty((params4.shape[0], n_runs * frames))
    for s in range(0, params4.shape[0], chunk):
        p = params4[s : s + chunk]
        sig2 = p[:, 2:3] ** 2
        g = np.exp(-((rr[None] - p[:, 0:1]) ** 2 + (cc[None] - p[:, 1:2]) ** 2) / (2 * sig2))
        g /= 2 * np.pi * sig2
        drive = np.clip(stim @ g.T, 0, None) ** np.maximum(p[:, 3], 1e-3)[None]  # (T, v)
        if isinstance(hrf, np.ndarray) and hrf.ndim == 1:
            conv = _conv_same(drive, hrf)
        else:
            conv = np.stack(
                [_conv_same(drive[:, j : j + 1], hrf[s + j])[:, 0] for j in range(p.shape[0])], 1
            )
        out[s : s + chunk] = np.tile(conv.T, (1, n_runs))  # identical aperture every run
    return out


def _conv_same(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """conv2run for one run: causal convolution truncated to the run length."""
    n = x.shape[0]
    return np.apply_along_axis(lambda c: np.convolve(c, h)[:n], 0, x)


def r2_percent(data_proj: np.ndarray, pred_proj: np.ndarray, gain: np.ndarray | None = None):
    """calccod: 100*(1 - SSE/SS), with the given gain or the least-squares one."""
    if gain is None:
        # Normalise first: a far-off-screen receptive field with a tiny sigma predicts
        # values around 1e-20, whose squares fall below any fixed floor; analyzePRF
        # fits those with a huge gain, and an unnormalised floor scored them near zero.
        scale = np.abs(pred_proj).max(1)
        unit = pred_proj / np.where(scale > 0, scale, 1.0)[:, None]
        unit_gain = np.sum(unit * data_proj, 1) / np.maximum(np.sum(unit**2, 1), 1e-300)
        gain = np.where(scale > 0, unit_gain / np.where(scale > 0, scale, 1.0), 0.0)
    resid = data_proj - gain[:, None] * pred_proj
    centred = data_proj - data_proj.mean(1, keepdims=True)
    return 100 * (1 - np.sum(resid**2, 1) / np.sum(centred**2, 1)), gain


# ----------------------------------------------------------------------------- statistics


def circ_corr(a_deg: np.ndarray, b_deg: np.ndarray) -> float:
    """Jammalamadaka-SenGupta circular correlation."""
    a = np.deg2rad(a_deg)
    b = np.deg2rad(b_deg)
    ma = np.angle(np.mean(np.exp(1j * a)))
    mb = np.angle(np.mean(np.exp(1j * b)))
    sa = np.sin(a - ma)
    sb = np.sin(b - mb)
    return float(np.sum(sa * sb) / np.sqrt(np.sum(sa**2) * np.sum(sb**2)))


def ang_diff(a_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    """Signed smallest angular difference a - b in degrees, in (-180, 180]."""
    return (a_deg - b_deg + 180) % 360 - 180


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


# ----------------------------------------------------------------------------- sphere maps


@dataclass
class FlatView:
    xy: np.ndarray  # (V_hemi, 2) projected coordinates
    faces: np.ndarray  # faces inside the view
    keep: np.ndarray  # bool per vertex, inside the view


def occipital_view(sphere_path: Path, roi: np.ndarray, radius_deg: float = 42.0) -> FlatView:
    """Lambert azimuthal equal-area projection of the sphere about the V1-V3 centroid.

    Rotated so the calcarine axis (V1's long axis) runs left-right and hV4
    (ventral) sits below it, so every hemisphere is drawn the same way up.
    """
    verts, faces = nib.freesurfer.read_geometry(str(sphere_path))
    u = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    c = u[(roi >= 1) & (roi <= 3)].mean(0)
    c /= np.linalg.norm(c)
    # Basis with c as the pole.
    a = np.array([0, 0, 1.0]) if abs(c[2]) < 0.9 else np.array([1.0, 0, 0])
    e1 = np.cross(c, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(c, e1)
    cosd = u @ c
    theta = np.arccos(np.clip(cosd, -1, 1))
    phi = np.arctan2(u @ e2, u @ e1)
    rho = 2 * np.sin(theta / 2)
    xy = np.stack([rho * np.cos(phi), rho * np.sin(phi)], 1)
    # Align V1's principal axis with x, ventral (hV4) down.
    v1 = xy[roi == 1]
    w, vecs = np.linalg.eigh(np.cov((v1 - v1.mean(0)).T))
    axis = vecs[:, np.argmax(w)]
    rot = np.array([[axis[0], axis[1]], [-axis[1], axis[0]]])
    xy = xy @ rot.T
    if np.any(roi == 4) and xy[roi == 4, 1].mean() > 0:
        xy[:, 1] *= -1
    keep = theta < np.deg2rad(radius_deg)
    fkeep = keep[faces].all(1)
    return FlatView(xy=xy, faces=faces[fkeep], keep=keep)
