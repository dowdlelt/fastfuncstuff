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

import html
import math
import os
import re
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
    *,
    nodec: bool = False,
) -> np.ndarray:
    """``[npthr, nathr]`` cluster-size thresholds from per-iteration maxima.

    Port of ``get_one_clust_thresh``.  AFNI does *not* take a plain
    quantile of the null maxima: it builds the survival function of the max
    cluster size and inverse-interpolates in **Gumbel** coordinates,
    ``log(-log(1-α))``, between the two bracketing integer sizes.  That is
    what produces the fractional thresholds (``10935.33``) in a real table,
    and a linear quantile lands a voxel or two off it in the tail.

    ``max_sizes`` is ``[n_iter, npthr]`` of per-iteration maximum cluster
    sizes; the returned value for a cell can be fractional unless ``nodec``.

    The finished table is forced monotone along both axes — a looser pthr
    and a stricter athr can only ever demand a larger cluster.  Monte-Carlo
    noise does occasionally invert two adjacent cells, and AFNI edits those
    out ("shouldn\'t be needed") before writing either output format.
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

    if nodec:
        # AFNI's (int)(x + 0.951): round up, but tolerate a hair under.
        out = np.floor(out + 0.951)
    # Each column increases as pthr loosens; each row as athr tightens.
    out = np.maximum.accumulate(out[::-1], axis=0)[::-1]
    out = np.maximum.accumulate(out, axis=1)
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
    on_device: bool | None = None,
    verbose: bool = True,
) -> ClusterNull:
    """Run the Monte-Carlo simulation and return the accumulated null.

    On CUDA the whole loop stays on the card — fields are generated and
    labelled there, and only a ``[batch, npthr]`` table of cluster maxima
    ever crosses PCIe.  Otherwise fields are generated in batches and handed
    to a CPU worker pool, with the next batch's generation overlapping the
    current one's clustering.  Either way only one batch is resident, so
    memory is independent of ``n_iter``.

    ``on_device`` forces the choice; the default follows the device.

    ``seed`` reproduces a run only together with ``batch``: the automatic
    batch size is read from *free* memory, so an otherwise identical rerun
    on a busier card draws a different set of fields.  The tables agree to
    Monte-Carlo error either way; pin ``batch`` when you need the same
    numbers twice.
    """
    sim = NullFieldSimulator(mask, voxmm, acf, device=device, seed=seed)
    tcrits = zthresholds(pthr, sideds)

    null = ClusterNull(pthr=pthr, athr=athr, nns=nns, sideds=sideds)
    null.init_storage(n_iter)

    if on_device is None:
        on_device = sim.device.type == "cuda"
    if on_device:
        return _simulate_on_device(sim, null, tcrits, n_iter, pthr, nns, sideds, batch, verbose)

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


def _simulate_on_device(
    sim: NullFieldSimulator,
    null: ClusterNull,
    tcrits: dict[str, np.ndarray],
    n_iter: int,
    pthr: tuple[float, ...],
    nns: tuple[int, ...],
    sideds: tuple[str, ...],
    batch: int | None,
    verbose: bool,
) -> ClusterNull:
    """Generate and label entirely on the accelerator."""
    from fastfuncstuff.stats.cluster_gpu import build_neighbor_table, cluster_extent_batched

    dev = sim.device
    n_vox = int(sim.mask.sum())
    if batch is None:
        batch = _plan_batch(sim, n_iter)

    # Both paths want thresholds tightest-first; remember how to put the
    # columns back in the caller's pthr order.
    tc_dev: dict[str, torch.Tensor] = {}
    order: dict[str, np.ndarray] = {}
    for s in sideds:
        arr = np.asarray(tcrits[s], dtype=np.float64)
        idx = np.argsort(-arr)
        order[s] = idx
        tc_dev[s] = torch.from_numpy(arr[idx].copy()).to(dev, torch.float32)

    nbr = {nn: build_neighbor_table(sim.mask, nn, dev) for nn in nns}
    scratch = torch.full((batch * n_vox,), -1, device=dev, dtype=torch.int32)

    bar = tqdm(total=n_iter, desc="clustsim", leave=True, disable=not verbose)
    try:
        for start in range(0, n_iter, batch):
            nb = min(batch, n_iter - start)
            fields = sim.generate(nb)
            view = scratch[: nb * n_vox] if nb != batch else scratch
            res = cluster_extent_batched(fields, nbr, nns, sideds, tc_dev, view)
            del fields
            for (sided, nn), tab in res.items():
                null.max_extent[(sided, nn)][start : start + nb, order[sided]] = tab.cpu().numpy()
            bar.update(nb)
    finally:
        bar.close()
    return null


def _plan_batch(sim: NullFieldSimulator, n_iter: int) -> int:
    """Fields per generation batch, from free memory on the target device."""
    from fastfuncstuff.memory import get_available_memory

    avail = get_available_memory(sim.device)
    # 0.5 safety factor: PyTorch's caching allocator holds more than the
    # live tensors, and the host-side copy of the batch lands alongside.
    budget = int(avail * 0.5)
    per_field = sim.bytes_per_field()
    if sim.device.type == "cuda":
        # Labelling on-device costs far more than generation: the volume->
        # compact-id scratch, the [K, T] adjacency and the labels all scale
        # with the batch.  Measured ~48 bytes per (volume x mask voxel).
        per_field = max(per_field, 48 * int(sim.mask.sum()))
    n = max(2, budget // max(per_field, 1))
    return int(min(n, n_iter, 4096))


# ---------------------------------------------------------------------------
# Tables, and attaching them to a stats dataset
# ---------------------------------------------------------------------------


def _prob6(p: float) -> str:
    """AFNI ``prob6``: a p-value in exactly 6 characters (column headers)."""
    if p >= 0.00010:
        return f"{p:7.5f}"[1:]
    dec = int(0.9999 - math.log10(p))
    return f"{p * 10.0**dec:4.1f}e-{dec:1d}"[1:]


def _prob9(p: float) -> str:
    """AFNI ``prob9``: a p-value in exactly 9 characters (row labels)."""
    if p >= 0.00010:
        return f"{p:9.6f}"
    dec = int(0.9999 - math.log10(p))
    return f"{p * 10.0**dec:6.3f}e-{dec:1d}"


def write_1d_table(
    path,
    table: np.ndarray,
    *,
    nn: int,
    sidedness: str,
    pthr: tuple[float, ...],
    athr: tuple[float, ...],
    shape: tuple[int, int, int],
    voxmm: tuple[float, float, float],
    mask_count: int,
    commandline: str,
    nodec: bool,
) -> None:
    """Write one ``ppp.NN{n}_{sided}.1D``.

    Byte-for-byte 3dClustSim's layout, down to ``prob9``/``prob6`` and the
    ``%7.1f`` cells, because these files are read by eye and by afni_proc.
    """
    from pathlib import Path

    n_total = int(np.prod(shape))
    in_mask = " in mask" if mask_count < n_total else ""
    lines = [
        f"# {commandline}",
        f"# {sidedness} thresholding",
        "# Grid: {}x{}x{} {:.2f}x{:.2f}x{:.2f} mm^3 ({} voxels{})".format(
            *shape, *voxmm, mask_count, in_mask
        ),
        "#",
        "# CLUSTER SIZE THRESHOLD(pthr,alpha) in Voxels",
        f"# -NN {nn}  | alpha = Prob(Cluster >= given size)",
        "#  pthr  |" + "".join(f" {_prob6(a)}" for a in athr),
        "# ------ |" + " ------" * len(athr),
    ]
    for i, p in enumerate(pthr):
        cells = ""
        for v in table[i]:
            if nodec:
                cells += f"{int(v):7d}"  # already rounded by gumbel_extent_table
            elif v <= 9999.9:
                cells += f"{v:7.1f}"
            else:
                cells += f"{v:7.0f}"
        lines.append(f"{_prob9(p)} {cells}")
    Path(path).write_text("\n".join(lines) + "\n")


def print_table_summary(null, nns, sideds, pthr, athr, niter, stream=None) -> None:
    """Echo the first NN/sidedness table — the one people read off the terminal."""
    import sys

    stream = stream or sys.stderr
    sided, nn = sideds[0], nns[0]
    table = gumbel_extent_table(null.max_extent[(sided, nn)], athr, niter)
    print(f"\n# NN{nn} {sided} — cluster size threshold (voxels)", file=stream)
    print("#  pthr  | " + " ".join(f"{a:.5f}"[1:].rjust(6) for a in athr), file=stream)
    print("# ------ | " + " ".join("------" for _ in athr), file=stream)
    for i, p in enumerate(pthr):
        print(f" {p:.6f} " + " ".join(f"{v:6.1f}" for v in table[i]), file=stream)
    print("", file=stream)


#: NIML `thresholding` keeps the hyphen; filenames and 3drefit attributes don't.
SIDED_ATTR = {"1-sided": "1sided", "2-sided": "2sided", "bi-sided": "bisided"}


def attach_clustsim_tables(
    mask: np.ndarray,
    voxmm: tuple[float, float, float],
    acf: ACF,
    *,
    prefix,
    refit=None,
    n_iter: int = 10000,
    pthr: tuple[float, ...] = DEFAULT_CS_PTHR,
    athr: tuple[float, ...] = DEFAULT_CS_ATHR,
    nns: tuple[int, ...] = (1, 2, 3),
    sideds: tuple[str, ...] = ("1-sided", "2-sided", "bi-sided"),
    device=None,
    n_jobs: int | None = None,
    batch: int | None = None,
    seed: int | None = None,
    on_device: bool | None = None,
    nodec: bool = False,
    commandline: str = "",
    mask_name: str = "<inline>",
    mask_idcode: str | None = None,
    summary: bool = True,
    verbose: bool = True,
) -> dict:
    """Simulate the null, write the ``.1D``/``.niml``/``.mask`` set, and refit.

    The whole of ``ffs_clustsim`` downstream of "what is the ACF": both the CLI
    and ``ffs_reml -clustsim`` enter here, so a table attached during a GLM fit
    and one attached afterwards are produced by the same code, not by two
    spellings of it.

    ``mask`` is a boolean volume, ``prefix`` the output stem. Returns the
    ``{(NN, sided_attr): niml path}`` map that was injected into ``refit``.
    """
    from pathlib import Path

    from fastfuncstuff.stats.niml import run_refit, write_clustsim_niml, write_mask_b64

    prefix = Path(prefix)
    out_dir = prefix.parent
    base = prefix.name
    out_dir.mkdir(parents=True, exist_ok=True)
    shape = tuple(int(s) for s in mask.shape)

    null = simulate_cluster_null(
        mask,
        voxmm,
        acf,
        n_iter=n_iter,
        pthr=pthr,
        athr=athr,
        nns=nns,
        sideds=sideds,
        device=device,
        n_jobs=n_jobs,
        batch=batch,
        seed=seed,
        on_device=on_device,
        verbose=verbose,
    )

    mask_b64 = out_dir / f"{base}.mask"
    mask_count = write_mask_b64(mask_b64, mask)

    niml_files: dict[tuple[int, str], Path] = {}
    for sided in sideds:
        for nn in nns:
            table = gumbel_extent_table(null.max_extent[(sided, nn)], athr, n_iter, nodec=nodec)
            tag = SIDED_ATTR[sided]
            write_1d_table(
                out_dir / f"{base}.NN{nn}_{tag}.1D",
                table,
                nn=nn,
                sidedness=sided,
                pthr=pthr,
                athr=athr,
                shape=shape,
                voxmm=voxmm,
                mask_count=mask_count,
                commandline=commandline,
                nodec=nodec,
            )
            niml_path = out_dir / f"{base}.NN{nn}_{tag}.niml"
            write_clustsim_niml(
                niml_path,
                table,
                nn=nn,
                sidedness=sided,
                commandline=commandline,
                nxyz=shape,
                dxyz=voxmm,
                pthr=pthr,
                athr=athr,
                n_perms=n_iter,
                mask_count=mask_count,
                mask_idcode=mask_idcode,
                mask_name=mask_name,
            )
            niml_files[(nn, tag)] = niml_path

    if summary and verbose:
        print_table_summary(null, nns, sideds, pthr, athr, n_iter)

    if refit is not None:
        run_refit(
            stat_path=Path(refit),
            niml_files=niml_files,
            mask_b64_path=mask_b64,
            write_script_path=out_dir / f"{base}.3drefit.cmd",
            verbose=verbose,
        )
    return niml_files


# ---------------------------------------------------------------------------
# Reading tables back out of a dataset that carries them
#
# 3dClustSim's output is only useful if something reads it. AFNI's Clusterize
# panel reads the attributes 3drefit wrote; so does this, against the same
# format write_clustsim_niml produces -- which is the point of matching AFNI's
# spelling there rather than inventing one.
# ---------------------------------------------------------------------------

_CS_HEAD_RE = re.compile(
    r"<3dClustSim_NN[123]\b(?P<attrs>.*?)>(?P<body>.*?)</3dClustSim_NN[123]>", re.S
)
_CS_KV_RE = re.compile(r'(\w+)\s*=\s*"([^"]*)"')

SIDED_FROM_ATTR = {v: k for k, v in SIDED_ATTR.items()}


@dataclass(frozen=True)
class ClustSimTable:
    """One ``3dClustSim`` table: cluster sizes by per-voxel p and corrected alpha."""

    nn: int
    sidedness: str
    pthr: tuple[float, ...]
    athr: tuple[float, ...]
    #: ``(len(pthr), len(athr))`` minimum cluster size, in voxels.
    sizes: np.ndarray
    n_iter: int = 0

    def _row(self, pthr: float) -> np.ndarray:
        """The size-vs-alpha curve at the nearest tabulated per-voxel p.

        Nearest rather than interpolated between rows: the rows are decades
        apart (0.01, 0.005, 0.002, …) and a number interpolated across that gap
        would carry a precision the simulation never had. The caller is told
        which p was actually used.
        """
        return self.sizes[int(np.argmin(np.abs(np.asarray(self.pthr) - float(pthr))))]

    def nearest_pthr(self, pthr: float) -> float:
        return float(self.pthr[int(np.argmin(np.abs(np.asarray(self.pthr) - float(pthr))))])

    def size_for(self, pthr: float, alpha: float) -> float:
        """Cluster size that survives at ``alpha``, for a per-voxel ``pthr``."""
        row = self._row(pthr)
        athr = np.asarray(self.athr, dtype=float)
        # The table runs from loose alpha to strict; interp needs ascending x.
        order = np.argsort(athr)
        return float(np.interp(float(alpha), athr[order], row[order]))

    def alpha_for(self, pthr: float, size: int) -> float:
        """Corrected alpha for a cluster of ``size`` voxels.

        Clamped to the table at both ends rather than extrapolated. Past the
        largest tabulated size the truth is "more significant than the smallest
        alpha simulated", and past the smallest it is "less significant than
        the largest" -- both are bounds, and a curve fitted past the last row
        would turn either into a number that looks like a measurement.
        :attr:`alpha_range` is what tells a caller which end it landed on.
        """
        row = np.asarray(self._row(pthr), dtype=float)
        athr = np.asarray(self.athr, dtype=float)
        order = np.argsort(row)  # size ascending <=> alpha descending
        return float(np.interp(float(size), row[order], athr[order]))

    @property
    def alpha_range(self) -> tuple[float, float]:
        """``(strictest, loosest)`` alpha the simulation actually covers."""
        return (float(min(self.athr)), float(max(self.athr)))


def parse_clustsim_niml(text: str) -> ClustSimTable | None:
    """Parse one ``<3dClustSim_NNn …>`` element. Inverse of the writer above."""
    m = _CS_HEAD_RE.search(html.unescape(text))
    if not m:
        return None
    attrs = dict(_CS_KV_RE.findall(m.group("attrs")))
    try:
        pthr = tuple(float(v) for v in attrs["pthr"].split(","))
        athr = tuple(float(v) for v in attrs["athr"].split(","))
    except (KeyError, ValueError):
        return None
    rows = [
        [float(v) for v in line.split()]
        for line in m.group("body").strip().splitlines()
        if line.strip()
    ]
    sizes = np.asarray(rows, dtype=np.float64)
    if sizes.shape != (len(pthr), len(athr)):
        return None
    nn_match = re.search(r"<3dClustSim_NN([123])", m.group(0))
    return ClustSimTable(
        nn=int(nn_match.group(1)) if nn_match else 1,
        sidedness=attrs.get("thresholding", "bi-sided"),
        pthr=pthr,
        athr=athr,
        sizes=sizes,
        n_iter=int(float(attrs.get("iter", 0) or 0)),
    )


def read_clustsim_tables(img) -> dict[tuple[int, str], ClustSimTable]:
    """Every ClustSim table attached to a dataset, keyed ``(nn, sidedness)``.

    Empty when the dataset carries none, which is the common case and is worth
    saying out loud rather than filling in with a default simulation: a cluster
    threshold from somebody else's smoothness is worse than no threshold.
    """
    from fastfuncstuff.io.headers import _afni_ext_text

    text = _afni_ext_text(img)
    out: dict[tuple[int, str], ClustSimTable] = {}
    for m in re.finditer(
        r'atr_name\s*=\s*"AFNI_CLUSTSIM_NN([123])_(1sided|2sided|bisided)"[^>]*>(.*?)</AFNI_atr>',
        text,
        re.S,
    ):
        table = parse_clustsim_niml(m.group(3))
        if table is not None:
            out[(int(m.group(1)), SIDED_FROM_ATTR[m.group(2)])] = table
    return out
