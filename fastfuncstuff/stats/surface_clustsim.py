"""Cluster-area thresholds on a surface (SurfClustSim): mixed-ACF noise, max cluster area.

The surface twin of :mod:`stats.clustsim`, with SUMA's ``SurfClustSim`` (the C port of
``slow_surf_clustsim.py``, AFNI ``SUMA_SurfClustSim_core.c``) as the design reference:

* **Noise with a mixed ACF** ``a exp(-r^2/2b^2) + (1-a) exp(-r/c)`` is a weighted sum of
  *independent* basis fields, white noise smoothed to a ladder of widths. Independence
  is what makes the mixture's ACF the weighted sum of the bases' ACFs. Each component
  is scaled by its own realisation's standard deviation, because a broad field keeps
  few spatial modes and its variance swings from draw to draw (SUMA's lesson).
* Unlike SUMA, the bases are our heat smoother, whose kernel is known in closed form,
  so their ACFs come from a Hankel transform and the weights from one non-negative
  least squares fit, with no calibration simulation. The noise is **area-scaled**
  (variance proportional to 1 / vertex area), which is continuum white noise, so
  the generated ACF belongs to the cortex rather than to the mesh's vertex density.
  :func:`check_generated_acf` measures what is actually generated and fits it again,
  SUMA's acceptance test.
* Clusters are connected components of suprathreshold vertices over mesh edges,
  sized by **area** (mm^2 on the midthickness). Many realisations go through one
  ``connected_components`` call as a block-diagonal graph.
* The null depends only on (geometry, mask, ACF, thresholds), never on data, so a
  run is **cached** on that key: a template mesh's table is computed once.

Thresholds are z (the fields are unit variance), so one table serves every sub-brick.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from fastfuncstuff.stats.clustsim import (
    ACF,
    DEFAULT_CS_ATHR,
    DEFAULT_CS_PTHR,
    acf_rfunc,
    gumbel_extent_table,
    zthresholds,
)
from fastfuncstuff.surface.mesh import vertex_areas
from fastfuncstuff.surface.smooth import HeatSmoother, _step_fwhm_ratio, mesh_edges

__all__ = [
    "MixedNoise",
    "SurfClustSimResult",
    "check_generated_acf",
    "max_cluster_areas",
    "surface_clustsim",
]

_LN2 = float(np.log(2.0))
SIDEDS = ("1-sided", "2-sided", "bi-sided")
_CACHE_VERSION = 1


@lru_cache(maxsize=256)
def _basis_acf(kernel_fwhm: float, n_steps: int, r_key: tuple[float, ...]) -> np.ndarray:
    """ACF at radii ``r`` of continuum white noise through the n-step heat kernel."""
    from scipy.special import j0

    r = np.asarray(r_key)
    ratio = _step_fwhm_ratio(n_steps)
    t = (kernel_fwhm / ratio) ** 2 / (16.0 * _LN2)
    dt = t / n_steps
    kmax = 60.0 / np.sqrt(dt * n_steps)
    k = np.linspace(0.0, kmax, 40001)
    power = (1.0 + dt * k * k) ** (-2 * n_steps) * k  # |S(k)|^2, radial measure
    acf = np.trapezoid(power[None, :] * j0(np.outer(r, k)), k, axis=1)
    return acf / acf[0]


class MixedNoise:
    """Unit-variance random fields on a mesh with a prescribed mixed ACF."""

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        acf: ACF,
        mask: np.ndarray | None = None,
        n_basis: int = 10,
        n_steps: int = 16,
        device=None,
    ):
        self.vertices = np.asarray(vertices, np.float64)
        self.faces = np.asarray(faces, np.int64)
        n = len(self.vertices)
        self.mask = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
        self.acf = acf
        self.device = device
        self.area = vertex_areas(self.vertices, self.faces)
        # The ladder spans the model's own two scales, as SUMA's does: the Gaussian part's
        # ACF FWHM is 2.355 b and the exponential's 2 c ln2; a kernel of width h gives an
        # ACF about sqrt(2) h wide. SUMA stops at 1.5x the exponential's scale; our bases
        # are near-Gaussian and building an exponential TAIL out of them needs wider
        # members: reaching 3x took the analytic fit to (0.5, 2.5, 6) from 0.018 to 0.007
        # rms (5x gains nothing).
        lo = 0.5 * (2.355 * acf.b) / np.sqrt(2.0)
        hi = 3.0 * (2.0 * acf.c * _LN2) / np.sqrt(2.0) if acf.a < 1.0 else 2.0 * lo
        hi = max(hi, 4.0 * lo)
        self.kernels = lo * (hi / lo) ** (np.arange(n_basis) / max(n_basis - 1, 1))
        reach = max(4.0 * acf.b, 6.0 * acf.c if acf.a < 1.0 else 0.0)  # where the target dies
        r = np.linspace(0.0, reach, 200)
        basis = np.stack([_basis_acf(float(h), n_steps, tuple(r)) for h in self.kernels], axis=1)
        target = acf_rfunc(r, acf)
        from scipy.optimize import nnls

        w, _ = nnls(basis, target)
        self.weights = w / w.sum() if w.sum() > 0 else np.full(n_basis, 1.0 / n_basis)
        self.fit_rms = float(np.sqrt(np.mean((basis @ self.weights - target) ** 2)))
        self._smoothers = [
            HeatSmoother(self.vertices, self.faces, float(h), mask=self.mask,
                         n_steps=n_steps, device=device)
            if wk > 1e-4 else None
            for h, wk in zip(self.kernels, self.weights, strict=True)
        ]  # fmt: skip

    def sample(self, n: int, rng: np.random.Generator):
        """``(V, n)`` float32 fields, zero outside the mask, unit SD inside each.

        The white noise always comes from ``rng`` on the host, so a seed gives the same
        fields on any device. With a torch ``device`` the fields come back as a tensor
        on it (ClustSim keeps them there for the cluster pass).
        """
        m = self.mask
        scale = (1.0 / np.sqrt(np.maximum(self.area, 1e-12))).astype(np.float32)
        if self.device is None:
            out = np.zeros((len(self.vertices), n), np.float32)
        else:
            import torch

            out = torch.zeros((len(self.vertices), n), dtype=torch.float32, device=self.device)
            mt = torch.as_tensor(m, device=self.device)
        for wk, hs in zip(self.weights, self._smoothers, strict=True):
            if hs is None:
                continue
            white = rng.standard_normal((len(self.vertices), n), dtype=np.float32)
            white *= scale[:, None]
            white[~m] = 0.0
            if self.device is None:
                comp = np.asarray(hs(white))
                comp[~m] = 0.0
                sd = comp[m].std(axis=0)
                out += (np.sqrt(wk) / np.maximum(sd, 1e-12))[None, :] * comp
            else:
                comp = hs(torch.as_tensor(white, device=self.device))
                comp[~mt] = 0.0
                sd = comp[mt].std(dim=0)
                out += (float(np.sqrt(wk)) / sd.clamp_min(1e-12))[None, :] * comp
        if self.device is None:
            out /= np.maximum(out[m].std(axis=0), 1e-12)[None, :]
        else:
            out /= out[mt].std(dim=0).clamp_min(1e-12)[None, :]
        return out


def check_generated_acf(noise: MixedNoise, n: int = 64, seed: int = 1) -> tuple[float, ...]:
    """Fit the mixed model to fields the generator actually makes: (a, b, c, FWHM)."""
    from fastfuncstuff.surface.acf import surface_acf

    fields = noise.sample(n, np.random.default_rng(seed))
    if not isinstance(fields, np.ndarray):
        fields = fields.cpu().numpy()
    # surface_acf correlates over its second axis; here that axis is realisations,
    # each a sample of the same stationary field.
    a = surface_acf(fields, noise.vertices, noise.faces, noise.mask, n_centres=800)
    return a.a, a.b, a.c, a.fwhm


def max_cluster_areas(
    fields: np.ndarray,
    edges: np.ndarray,
    area: np.ndarray,
    thresholds: np.ndarray,
    sided: str,
) -> np.ndarray:
    """``(n, n_thr)`` largest cluster area per field and threshold.

    ``1-sided``: z > thr. ``2-sided``: |z| > thr, and positive and negative vertices may
    join one cluster. ``bi-sided``: |z| > thr, positive and negative clustered apart
    (the largest of either). As 3dClustSim, the thresholds are already the per-side z.
    """
    import torch

    on_torch = isinstance(fields, torch.Tensor)
    largest = _largest_torch if on_torch else _largest
    absf = fields.abs() if on_torch else np.abs(fields)
    n = fields.shape[1]
    out = np.zeros((n, len(thresholds)))
    for j, thr in enumerate(thresholds):
        if sided == "1-sided":
            parts = [fields > thr]
        elif sided == "2-sided":
            parts = [absf > thr]
        else:
            parts = [fields > thr, fields < -thr]
        for act in parts:
            out[:, j] = np.maximum(out[:, j], largest(act, edges, area))
    return out


def _largest_torch(active, edges, area):
    """:func:`_largest` on a torch device: min-label propagation with pointer jumping.

    Each sweep pulls every active edge's endpoints to the smaller label, then labels
    jump to their label's label until stable; sweeps repeat until nothing changes.
    Typically a dozen or two sweeps, each a couple of scatter/gathers over all
    realisations at once, against scipy's single-threaded pass on the host.
    """
    import torch

    v, n = active.shape
    dev = active.device
    node = torch.arange(v * n, device=dev).view(n, v)  # node id = col * v + vertex
    act = active.T  # (n, v)
    e0 = torch.as_tensor(edges[:, 0], device=dev)
    e1 = torch.as_tensor(edges[:, 1], device=dev)
    both = act[:, e0] & act[:, e1]  # (n, E)
    a = node[:, e0][both]
    b = node[:, e1][both]
    labels = node.reshape(-1).clone()
    while True:
        la, lb = labels[a], labels[b]
        low = torch.minimum(la, lb)
        new = labels.clone()
        new.scatter_reduce_(0, a, low, reduce="amin")
        new.scatter_reduce_(0, b, low, reduce="amin")
        while True:
            jumped = new[new]
            if torch.equal(jumped, new):
                break
            new = jumped
        if torch.equal(new, labels):
            break
        labels = new
    weight = (act * torch.as_tensor(area, device=dev, dtype=torch.float64)[None, :]).reshape(-1)
    comp_area = torch.zeros(v * n, dtype=torch.float64, device=dev).index_add_(0, labels, weight)
    return comp_area.view(n, v).amax(dim=1).cpu().numpy()


def _largest(active: np.ndarray, edges: np.ndarray, area: np.ndarray) -> np.ndarray:
    """Largest connected active area in each column, all columns in one graph."""
    v, n = active.shape
    both = active[edges[:, 0]] & active[edges[:, 1]]  # (E, n)
    e_idx, col = np.nonzero(both)
    a = edges[e_idx, 0] + col * v
    b = edges[e_idx, 1] + col * v
    g = sparse.csr_matrix((np.ones(a.size, np.int8), (a, b)), shape=(v * n, v * n))
    _, labels = connected_components(g, directed=False)
    weight = (active * area[:, None]).T.reshape(-1)  # node order col * v + vertex
    comp_area = np.bincount(labels, weights=weight)
    comp_col = np.zeros(comp_area.size, np.int64)
    comp_col[labels] = np.repeat(np.arange(n), v)
    best = np.zeros(n)
    np.maximum.at(best, comp_col, comp_area)
    return best


@dataclass
class SurfClustSimResult:
    """Per-iteration max cluster areas (mm^2), and the tables built from them."""

    pthr: tuple[float, ...]
    max_areas: dict[str, np.ndarray]  # sided -> (niter, npthr)
    niter: int
    acf: ACF
    weights: list[float] = field(default_factory=list)
    fit_rms: float = float("nan")
    cached: bool = False

    def table(self, sided: str, athr=DEFAULT_CS_ATHR, unit: float = 1.0) -> np.ndarray:
        """``[npthr, nathr]`` cluster-area thresholds in mm^2 (AFNI's Gumbel interpolation,
        on areas counted in ``unit`` mm^2 steps)."""
        sizes = np.ceil(self.max_areas[sided] / unit)
        return gumbel_extent_table(sizes, tuple(athr), self.niter) * unit


def _cache_key(vertices, faces, mask, acf, niter, pthr, sideds, seed, n_basis) -> str:
    h = hashlib.sha1()
    for arr in (np.asarray(vertices, np.float32), np.asarray(faces, np.int32),
                np.asarray(mask, bool)):  # fmt: skip
        h.update(np.ascontiguousarray(arr).tobytes())
    h.update(json.dumps([round(acf.a, 4), round(acf.b, 4), round(acf.c, 4), niter,
                         list(pthr), list(sideds), seed, n_basis, _CACHE_VERSION]).encode())  # fmt: skip
    return h.hexdigest()[:20]


def default_cache_dir() -> Path:
    return (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "fastfuncstuff"
        / "surf_clustsim"
    )


def surface_clustsim(
    vertices: np.ndarray,
    faces: np.ndarray,
    acf: ACF,
    mask: np.ndarray | None = None,
    niter: int = 1000,
    pthr: tuple[float, ...] = DEFAULT_CS_PTHR,
    sideds: tuple[str, ...] = SIDEDS,
    batch: int = 100,
    seed: int = 0,
    device=None,
    cache_dir: str | os.PathLike | None | bool = None,
    n_basis: int = 10,
    verb: int = 1,
) -> SurfClustSimResult:
    """Simulate ``niter`` null fields and record the largest cluster area per threshold.

    ``cache_dir``: None uses the default cache, False disables it. The result is a pure
    function of (geometry, mask, ACF, niter, pthr, sideds, seed), so a hit returns the
    stored maxima without simulating.
    """
    from tqdm import tqdm

    v = np.asarray(vertices, np.float64)
    f = np.asarray(faces, np.int64)
    m = np.ones(len(v), bool) if mask is None else np.asarray(mask, bool)
    path = None
    if cache_dir is not False:
        base = default_cache_dir() if cache_dir is None else Path(cache_dir)
        key = _cache_key(v, f, m, acf, niter, pthr, sideds, seed, n_basis)
        path = base / f"{key}.npz"
        if path.exists():
            z = np.load(path)
            return SurfClustSimResult(
                tuple(pthr), {s: z[s] for s in sideds}, niter, acf,
                list(z["weights"]), float(z["fit_rms"]), cached=True,
            )  # fmt: skip
    noise = MixedNoise(v, f, acf, m, n_basis=n_basis, device=device)
    edges = mesh_edges(f)
    edges = edges[m[edges[:, 0]] & m[edges[:, 1]]]
    area = np.where(m, vertex_areas(v, f), 0.0)
    zthr = zthresholds(tuple(pthr), tuple(sideds))
    out = {s: np.zeros((niter, len(pthr))) for s in sideds}
    rng = np.random.default_rng(seed)
    starts = range(0, niter, batch)
    for start in tqdm(starts, desc="SurfClustSim", leave=True, disable=verb == 0 or niter <= batch):
        n = min(batch, niter - start)
        fields = noise.sample(n, rng)
        for s in sideds:
            out[s][start : start + n] = max_cluster_areas(fields, edges, area, zthr[s], s)
    res = SurfClustSimResult(tuple(pthr), out, niter, acf, list(noise.weights), noise.fit_rms)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, weights=noise.weights, fit_rms=noise.fit_rms, **out)
    return res


def table_text(res: SurfClustSimResult, sided: str, athr=DEFAULT_CS_ATHR) -> str:
    """The table as 3dClustSim lays it out, in mm^2 of cortex."""
    from fastfuncstuff.stats.clustsim import _prob6, _prob9

    t = res.table(sided, athr)
    lines = [
        f"# {sided} thresholding, surface",
        f"# ACF a={res.acf.a:.4f} b={res.acf.b:.3f} c={res.acf.c:.3f}; {res.niter} iterations",
        "# CLUSTER AREA THRESHOLD(pthr,alpha) in mm^2",
        "#  pthr  |" + "".join(f" {_prob6(a)}" for a in athr),
        "# ------ |" + " ------" * len(athr),
    ]
    for i, p in enumerate(res.pthr):
        lines.append(
            f"{_prob9(p)} " + "".join(f"{x:7.1f}" if x <= 9999.9 else f"{x:7.0f}" for x in t[i])
        )
    return "\n".join(lines) + "\n"
