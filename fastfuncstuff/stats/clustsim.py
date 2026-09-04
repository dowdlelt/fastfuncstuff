"""
3dClustSim-style Monte-Carlo cluster-size thresholds.

Simulate noise-only volumes with a prescribed spatial autocorrelation,
threshold them at a range of per-voxel ``pthr``, and record the largest
null cluster.  The distribution of that per-iteration maximum gives the
cluster-size threshold at each family-wise ``athr``.

The null field
--------------
AFNI's ``-acf`` method (``mri_radial_random_field.c``), which is the
recommended one — a real fMRI residual's ACF is *not* Gaussian, it has a
long tail that a pure Gaussian blur badly underestimates::

    ACF(r) = a·exp(-r²/2b²) + (1-a)·exp(-r/c)

The field is built in Fourier space: the transform of white noise is
white noise, so multiplying an i.i.d. complex Gaussian spectrum by
``sqrt(FFT(ACF))`` and transforming back gives a field with the target
ACF.  ``sqrt`` because the ACF is effectively squared when it is
re-estimated off the result.

Two things fall out of the complex formulation, and we keep both:

* the real and imaginary parts of one inverse transform are two
  **independent** fields, so each FFT yields two iterations;
* the simulation grid is padded past the mask (by the ACF's own radius,
  rounded up to an FFT-friendly size) and cropped back, so the wrap-around
  in the periodic transform never reaches brain voxels.

The threshold is a **z**, not a t: fields are renormalised to unit
standard deviation before masking, exactly as ``generate_image()`` does,
so one table applies to every sub-brick regardless of its dof.

Reproducing AFNI bit-for-bit is not possible (it draws from a ziggurat
generator we don't share), so parity here means statistical agreement of
the tables, not identical numbers.
"""

from __future__ import annotations

import math
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import norm
from tqdm.auto import tqdm

from fastfuncstuff.stats.cluster import (
    DEFAULT_NN,
    DEFAULT_SIDED,
    ClusterNull,
    _null_worker_chunk,
    _null_worker_init,
)

# 3dClustSim's own defaults (pthr_init / athr_init in 3dClustSim.c).
DEFAULT_CS_PTHR = (0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001)
DEFAULT_CS_ATHR = (0.10, 0.05, 0.02, 0.01)

# '-LOTS'
LOTS_PTHR = (
    0.10,
    0.09,
    0.08,
    0.07,
    0.06,
    0.05,
    0.04,
    0.03,
    0.02,
    0.015,
    0.01,
    0.007,
    0.005,
    0.003,
    0.002,
    0.0015,
    0.001,
    0.0007,
    0.0005,
    0.0003,
    0.0002,
    0.00015,
    0.0001,
    7e-5,
    5e-5,
    3e-5,
    2e-5,
    1.5e-5,
    1e-5,
)
LOTS_ATHR = (0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01)

_S2F = 2.3548200450309493  # sqrt(8 ln 2): Gaussian sigma -> FWHM


# ---------------------------------------------------------------------------
# The ACF model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ACF:
    """Mixed-model spatial autocorrelation parameters (3dFWHMx ``-acf``)."""

    a: float
    b: float
    c: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.a <= 1.0):
            raise ValueError(f"ACF 'a' must be in [0, 1], got {self.a}")
        if self.b <= 0.0:
            raise ValueError(f"ACF 'b' must be positive, got {self.b}")
        if self.c <= 0.0:
            raise ValueError(f"ACF 'c' must be positive, got {self.c}")

    @classmethod
    def from_fwhm(cls, fwhm_mm: float) -> ACF:
        """The pure-Gaussian ACF of a field blurred to ``fwhm_mm``.

        Smoothing white noise with a Gaussian of FWHM *f* leaves an ACF that
        is Gaussian with FWHM ``f·√2`` — hence the ``√2``.  This is how
        ``-fwhm`` is served here; AFNI instead runs its FIR blur directly, so
        the two agree in the limit but not voxel-for-voxel.
        """
        return cls(a=1.0, b=fwhm_mm * math.sqrt(2.0) / _S2F, c=1.0)


def acf_rfunc(r: np.ndarray | float, acf: ACF) -> np.ndarray | float:
    """``a·exp(-r²/2b²) + (1-a)·exp(-r/c)``."""
    return acf.a * np.exp(-0.5 * r * r / (acf.b * acf.b)) + (1.0 - acf.a) * np.exp(-r / acf.c)


def acf_rfunc_inv(val: float, acf: ACF) -> float:
    """Radius at which the ACF falls to ``val``.

    ``rfunc`` is monotone decreasing, so a bisection is both simpler and
    tighter than AFNI's regula falsi (``rfunc_inv``); they agree to well
    inside the ``ceil()`` that consumes this.
    """
    if val >= 1.0:
        return 0.0
    rtop = 3.0 * acf.b + 6.0 * acf.c
    if val <= 0.0001:
        return rtop
    lo, hi = 0.0, rtop
    if acf_rfunc(hi, acf) > val:  # never decays that far
        return rtop
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if acf_rfunc(mid, acf) > val:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def acf_fwhm(acf: ACF) -> float:
    """Effective FWHM (mm) of the ACF — ``2·rfunc_inv(0.5)``."""
    return 2.0 * acf_rfunc_inv(0.5, acf)


# ---------------------------------------------------------------------------
# Simulation grid
# ---------------------------------------------------------------------------


def next_fft_size(n: int) -> int:
    """Smallest FFT-friendly size ``>= n``: 2^p · 3^q · 5^r with q, r ≤ 1.

    Matches AFNI's ``csfft_nextup_one35`` so the padded grid — and hence
    the amount of wrap-around guard around the mask — is the same size
    ours as theirs.  cuFFT likes these too.
    """
    if n <= 1:
        return 1
    best = None
    for q in (1, 3):
        for r in (1, 5):
            base = q * r
            p = base
            while p < n:
                p *= 2
            if best is None or p < best:
                best = p
    assert best is not None
    return best


def random_field_grid(
    shape: tuple[int, int, int],
    voxmm: tuple[float, float, float],
    acf: ACF,
) -> tuple[int, int, int]:
    """Padded simulation grid (``get_random_field_size``).

    Expand by the radius at which the ACF has decayed to 0.02 — the point
    past which wrap-around contamination is negligible — with a 16-voxel
    floor, then round each axis up to an FFT-friendly size.
    """
    r = acf_rfunc_inv(0.02, acf)
    out = []
    for n, d in zip(shape, voxmm, strict=True):
        v = n + 2 * int(math.ceil(r / d))
        out.append(next_fft_size(max(v, 16)))
    return (out[0], out[1], out[2])


def make_radial_weight(
    grid: tuple[int, int, int],
    voxmm: tuple[float, float, float],
    acf: ACF,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Fourier-space amplitude weight, ``sqrt(Re(FFT(ACF)))``.

    AFNI (``make_radial_weight``) builds one octant and reflects it eight
    ways while filling the noise; the full array here is the same thing
    written out, since the transform of a real symmetric function is itself
    symmetric under ``i -> n-i``.

    Nyquist planes are zeroed in real space and the DC bin is zeroed in the
    weight, which is what makes the generated field zero-mean.
    """
    nx, ny, nz = grid
    dx, dy, dz = voxmm
    # Wrapped coordinate: distance to the nearest periodic image of the origin.
    ax = [
        torch.where(
            torch.arange(n, device=device) < n // 2,
            torch.arange(n, device=device),
            n - torch.arange(n, device=device),
        ).to(torch.float64)
        * d
        for n, d in ((nx, dx), (ny, dy), (nz, dz))
    ]
    rr = torch.sqrt(
        ax[0].view(-1, 1, 1) ** 2 + ax[1].view(1, -1, 1) ** 2 + ax[2].view(1, 1, -1) ** 2
    )
    w = acf.a * torch.exp(-0.5 * rr * rr / (acf.b**2)) + (1.0 - acf.a) * torch.exp(-rr / acf.c)
    # Zero the Nyquist planes (they have no symmetric partner).
    w[nx // 2, :, :] = 0.0
    w[:, ny // 2, :] = 0.0
    w[:, :, nz // 2] = 0.0

    spec = torch.fft.fftn(w.to(torch.complex128)).real
    ftop = 1e-5 * float(spec.reshape(-1)[0].abs())
    spec = torch.where(spec < ftop, torch.zeros_like(spec), spec)
    weight = torch.sqrt(spec)
    weight[0, 0, 0] = 0.0  # zero mean
    return weight.to(dtype)


# ---------------------------------------------------------------------------
# Field generation
# ---------------------------------------------------------------------------


class NullFieldSimulator:
    """Batched generator of masked noise volumes with a prescribed ACF.

    ``generate(n)`` returns ``[n, V_in_mask]`` float32, each row a null
    volume renormalised to unit standard deviation *over the whole cropped
    grid* (in-mask and out, matching ``generate_image()``) and then
    restricted to the mask.
    """

    def __init__(
        self,
        mask: np.ndarray,
        voxmm: tuple[float, float, float],
        acf: ACF,
        *,
        device: torch.device | None = None,
        seed: int | None = None,
    ) -> None:
        self.mask = np.ascontiguousarray(mask.astype(bool))
        self.shape = (int(mask.shape[0]), int(mask.shape[1]), int(mask.shape[2]))
        self.voxmm = voxmm
        self.acf = acf
        self.device = device if device is not None else torch.device("cpu")
        self.grid = random_field_grid(self.shape, voxmm, acf)
        self.weight = make_radial_weight(self.grid, voxmm, acf, device=self.device)
        # AFNI crops the centre of the padded grid: ex_pad = (nxx - nx)/2.
        self.pad = tuple((g - s) // 2 for g, s in zip(self.grid, self.shape, strict=True))
        self.n_vox = int(np.prod(self.shape))
        self.mask_idx = torch.from_numpy(np.flatnonzero(self.mask.ravel())).to(
            self.device, torch.int64
        )
        self.gen = torch.Generator(device=self.device)
        if seed is not None:
            self.gen.manual_seed(int(seed))

    @property
    def fwhm(self) -> float:
        return acf_fwhm(self.acf)

    def bytes_per_field(self) -> int:
        """Peak device bytes per *generated* field (a complex pair is two)."""
        gx, gy, gz = self.grid
        # complex64 spectrum + its transform, amortised over the 2 fields.
        return int(gx * gy * gz * 8 * 2 / 2)

    def generate(self, n: int) -> torch.Tensor:
        """``[n, V_in_mask]`` float32 null fields on ``self.device``."""
        gx, gy, gz = self.grid
        px, py, pz = self.pad
        sx, sy, sz = self.shape
        n_pair = (n + 1) // 2
        noise = torch.randn(
            (n_pair, gx, gy, gz, 2), generator=self.gen, device=self.device, dtype=torch.float32
        )
        spec = torch.view_as_complex(noise) * self.weight
        del noise
        vol = torch.fft.ifftn(spec, dim=(1, 2, 3))
        del spec
        # Real and imaginary parts are two independent fields with this ACF.
        pair = torch.stack((vol.real, vol.imag), dim=1).reshape(2 * n_pair, gx, gy, gz)
        del vol
        crop = pair[:, px : px + sx, py : py + sy, pz : pz + sz].reshape(2 * n_pair, -1)
        del pair
        crop = crop[:n].contiguous()
        # Unit stdev over the whole cropped volume, before masking.
        scale = torch.sqrt(self.n_vox / crop.pow(2).sum(dim=1).clamp_min(1e-30))
        crop *= scale.unsqueeze(1)
        return crop[:, self.mask_idx]


# ---------------------------------------------------------------------------
# Alpha table
# ---------------------------------------------------------------------------


def gumbel_extent_table(
    max_sizes: np.ndarray,
    athr: tuple[float, ...],
    n_iter: int,
) -> np.ndarray:
    """``[npthr, nathr]`` cluster-size thresholds from per-iteration maxima.

    Port of ``get_one_clust_thresh``.  AFNI does *not* take a plain
    quantile of the null maxima: it builds the survival function of the max
    cluster size and inverse-interpolates in **Gumbel** coordinates,
    ``log(-log(1-α))``, between the two bracketing integer sizes.  That is
    what produces the fractional thresholds (``10935.33``) in a real table,
    and a linear quantile lands a voxel or two off it in the tail.

    ``max_sizes`` is ``[n_iter, npthr]`` of per-iteration maximum cluster
    sizes; the returned value for a cell can be fractional.
    """
    n_pthr = max_sizes.shape[1]
    out = np.zeros((n_pthr, len(athr)), dtype=np.float64)
    for ip in range(n_pthr):
        sizes = max_sizes[:, ip]
        top = int(sizes.max())
        if top < 1:
            # Never a single suprathreshold voxel: any cluster is significant.
            out[ip, :] = 1.0
            continue
        # alpha[s] = P(max cluster == s), then accumulated to P(max >= s).
        alpha = np.zeros(top + 2, dtype=np.float64)
        counts = np.bincount(sizes.astype(np.int64), minlength=top + 2)
        alpha[1 : top + 1] = counts[1 : top + 1] / float(n_iter)
        itop = int(np.flatnonzero(alpha > 0.0).max()) if np.any(alpha > 0.0) else 1
        alpha[1:] = np.cumsum(alpha[1:][::-1])[::-1]
        for j, aval in enumerate(athr):
            if aval > alpha[1]:
                # Not bracketed: even a 1-voxel cluster is rarer than aval.
                out[ip, j] = 1.0
                continue
            ii = itop
            for s in range(1, itop):
                if alpha[s] >= aval and alpha[s + 1] <= aval:
                    ii = s
                    break
            alo = alpha[ii]
            ahi = alpha[ii + 1] if ii + 1 < alpha.size else 0.0
            if alo >= 1.0:
                alo = 1.0 - 0.1 / n_iter
            if ahi <= 0.0:
                ahi = 0.1 / n_iter
            if ahi >= alo:
                ahi = 0.1 * alo
            g = lambda a: math.log(-math.log(1.0 - a))  # noqa: E731
            jj = ii + (g(aval) - g(alo)) / (g(ahi) - g(alo))
            out[ip, j] = max(jj, 1.0)
    return out


def zthresholds(
    pthr: tuple[float, ...],
    sideds: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Per-sidedness N(0,1) thresholds (``zthr_1sid`` / ``zthr_2sid``).

    1-sided tests the upper tail at ``p``; 2-sided and bi-sided both split
    the mass and threshold at ``p/2`` — they differ in whether opposite-sign
    voxels may join one cluster, not in where the cut is.
    """
    out: dict[str, np.ndarray] = {}
    for s in sideds:
        p = np.asarray(pthr, dtype=np.float64)
        out[s] = norm.isf(p if s == "1-sided" else p / 2.0)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def simulate_cluster_null(
    mask: np.ndarray,
    voxmm: tuple[float, float, float],
    acf: ACF,
    *,
    n_iter: int = 10000,
    pthr: tuple[float, ...] = DEFAULT_CS_PTHR,
    athr: tuple[float, ...] = DEFAULT_CS_ATHR,
    nns: tuple[int, ...] = DEFAULT_NN,
    sideds: tuple[str, ...] = DEFAULT_SIDED,
    device: torch.device | None = None,
    n_jobs: int | None = None,
    batch: int | None = None,
    seed: int | None = None,
    verbose: bool = True,
) -> ClusterNull:
    """Run the Monte-Carlo simulation and return the accumulated null.

    Fields are generated in batches on ``device`` and handed to a CPU worker
    pool for the connected-components pass, with generation of the next
    batch overlapping the clustering of the current one.  Only the batch is
    ever resident, so memory is independent of ``n_iter``.
    """
    sim = NullFieldSimulator(mask, voxmm, acf, device=device, seed=seed)
    tcrits = zthresholds(pthr, sideds)

    null = ClusterNull(pthr=pthr, athr=athr, nns=nns, sideds=sideds)
    null.init_storage(n_iter)

    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)
    n_jobs = min(n_jobs, n_iter)
    if batch is None:
        batch = _plan_batch(sim, n_iter)

    bar = tqdm(total=n_iter, desc="clustsim", leave=True, disable=not verbose)

    def _store(start: int, me: dict) -> None:
        pc = next(iter(me.values())).shape[0]
        for k, v in me.items():
            null.max_extent[k][start : start + pc] = v
        bar.update(pc)

    if n_jobs <= 1:
        _null_worker_init(sim.mask, 1, pthr, nns, sideds, True, tcrits)
        try:
            for start in range(0, n_iter, batch):
                nb = min(batch, n_iter - start)
                fields = sim.generate(nb).cpu().numpy()
                _, me, _ = _null_worker_chunk((start, fields))
                _store(start, me)
        finally:
            bar.close()
        return null

    import multiprocessing as mp

    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
    pool = ProcessPoolExecutor(
        max_workers=n_jobs,
        mp_context=ctx,
        initializer=_null_worker_init,
        initargs=(sim.mask, 1, pthr, nns, sideds, True, tcrits),
    )
    # Chunk small enough that every worker gets several, so the tail of a
    # batch doesn't leave cores idle.
    chunk = max(1, batch // (n_jobs * 4))
    inflight: deque = deque()
    max_inflight = n_jobs * 8
    try:
        for start in range(0, n_iter, batch):
            nb = min(batch, n_iter - start)
            fields = sim.generate(nb).cpu().numpy()
            for s in range(0, nb, chunk):
                e = min(s + chunk, nb)
                inflight.append(
                    pool.submit(_null_worker_chunk, (start + s, np.ascontiguousarray(fields[s:e])))
                )
            del fields
            # Drain far enough to bound memory, but leave the pool fed so the
            # next batch's generation overlaps the clustering still running.
            while len(inflight) > max_inflight:
                pstart, me, _ = inflight.popleft().result()
                _store(pstart, me)
        while inflight:
            pstart, me, _ = inflight.popleft().result()
            _store(pstart, me)
    finally:
        bar.close()
        pool.shutdown(wait=True)

    return null


def _plan_batch(sim: NullFieldSimulator, n_iter: int) -> int:
    """Fields per generation batch, from free memory on the target device."""
    from fastfuncstuff.memory import get_available_memory

    avail = get_available_memory(sim.device)
    # 0.5 safety factor: PyTorch's caching allocator holds more than the
    # live tensors, and the host-side copy of the batch lands alongside.
    budget = int(avail * 0.5)
    n = max(2, budget // max(sim.bytes_per_field(), 1))
    return int(min(n, n_iter, 4096))
