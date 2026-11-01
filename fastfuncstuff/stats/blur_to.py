"""Blurring data TO a smoothness: the master, its time-point sample, and the kernel search.

Geometry-free: the surface (heat kernel, mesh FWHM) and the volume (Gaussian, 3dFWHMx
1-difference) each supply "a smoother of width k" and "the FWHM of this master", and
:func:`bisect_kernel` does the rest. The smoother is linear, so the search runs on a
small master alone and the chosen kernel is then applied to the data once; AFNI's
3dBlurToFWHM / SurfSmooth instead re-estimate after every incremental pass.

**The master** is the residual of the model's own OLS fit (SurfSmooth's advice for
``-blurmaster``): detrended data still carry the task, which is spatially smooth and
inflates the measured FWHM, so the data come out under-blurred. With one design shared
by every voxel, the spatial smoother ``S`` and the residual-forming matrix ``R`` act on
different axes and commute, ``R (Y S) = (R Y) S``: residuals of the unblurred data
measure exactly what the blurred data's residuals will be, with no refit. Per-voxel
regressors (``-dsort``) break that, and are left out of the master.

**The sample**: a FWHM estimate averages over every edge or voxel pair, so a few dozen
time points pin it (3dBlurToFWHM uses at most 32). They are spread over the runs in
proportion to each run's uncensored length, because smoothness differs between runs
(motion, distortion), and censored points never enter.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "MaskedGaussian",
    "bisect_kernel",
    "residual_master",
    "sample_columns",
    "volume_blur_to_fwhm",
    "volume_fwhm",
]


def sample_columns(
    run_lengths: list[int],
    n_columns: int,
    good: np.ndarray | list[int] | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Sorted time indices (into the concatenation), ``n_columns`` of them spread over
    the runs in proportion to each run's kept length, never a censored one."""
    total = int(sum(run_lengths))
    keep = np.ones(total, bool)
    if good is not None:
        keep[:] = False
        keep[np.asarray(good, np.int64)] = True
    starts = np.r_[0, np.cumsum(run_lengths)[:-1]].astype(np.int64)
    pools = [np.flatnonzero(keep[s : s + n]) + s for s, n in zip(starts, run_lengths, strict=True)]
    n_kept = np.array([len(p) for p in pools])
    want = min(int(n_columns), int(n_kept.sum()))
    if want == int(n_kept.sum()):
        return np.sort(np.concatenate(pools))
    # Largest-remainder apportionment, then a run never asks for more than it has.
    share = want * n_kept / n_kept.sum()
    take = np.floor(share).astype(int)
    for i in np.argsort(-(share - take))[: want - take.sum()]:
        take[i] += 1
    take = np.minimum(take, n_kept)
    rng = np.random.default_rng(seed)
    picks = [rng.choice(p, size=k, replace=False) for p, k in zip(pools, take, strict=True) if k]
    return np.sort(np.concatenate(picks))


def residual_master(
    data: np.ndarray,
    design: np.ndarray,
    columns: np.ndarray,
    good: np.ndarray | list[int] | None = None,
    chunk: int = 16384,
) -> np.ndarray:
    """``(V, k)`` OLS residuals of ``data`` (``(V, T)``) against ``design`` (``(T, p)``),
    at the time points ``columns`` only. Censored rows (outside ``good``) take no part
    in the fit. Never copies ``data``: one ``(k, T)`` residual-forming block is built in
    float64 and applied a voxel chunk at a time."""
    x = np.asarray(design, np.float64)
    t = x.shape[0]
    keep = np.arange(t) if good is None else np.asarray(good, np.int64)
    pinv = np.linalg.pinv(x[keep])  # (p, G): rank-deficient designs are fine
    cols = np.asarray(columns, np.int64)
    rows = np.zeros((len(cols), t))
    rows[:, keep] = -(x[cols] @ pinv)
    rows[np.arange(len(cols)), cols] += 1.0
    out = np.empty((data.shape[0], len(cols)), np.float32)
    for v0 in range(0, data.shape[0], chunk):
        block = np.asarray(data[v0 : v0 + chunk], np.float64)
        out[v0 : v0 + chunk] = block @ rows.T
    return out


def bisect_kernel(
    measure: Callable[[float], tuple[float, Any]],
    target: float,
    tol: float = 0.02,
) -> tuple[Any, float, float]:
    """The smallest kernel that brings a master to ``target`` FWHM.

    ``measure(k)`` returns ``(fwhm of the master smoothed by k, the smoother)``, with
    ``k = 0`` meaning unsmoothed (its smoother is ignored). Returns ``(smoother,
    kernel, achieved)``; ``smoother`` is None when the master already measures at
    least ``target`` -- nothing to add, as 3dBlurToFWHM reports.
    """
    start, _ = measure(0.0)
    if not np.isfinite(start) or start >= target:
        return None, 0.0, float(start)
    lo, hi = 0.0, float(target)
    fhi, sm_hi = measure(hi)
    while fhi < target and hi < 8 * target:
        lo, hi = hi, hi * 1.5
        fhi, sm_hi = measure(hi)
    best = (hi, fhi, sm_hi)
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        fmid, sm_mid = measure(mid)
        if fmid >= target:
            hi, best = mid, (mid, fmid, sm_mid)
        else:
            lo = mid
        if abs(best[1] - target) <= tol * target:
            break
    kernel, achieved, smoother = best
    return smoother, float(kernel), float(achieved)


class MaskedGaussian:
    """A Gaussian of ``fwhm`` mm applied *within* a mask, as 3dBlurInMask does it:
    ``blur(x m) / blur(m)``. Nothing outside the brain bleeds into the rim (which
    would read as smoothness to the estimate, and as dropout to everything else), and
    voxels outside the mask pass through untouched. Linear, so it commutes with the
    OLS residual like the surface heat kernel.

    Called on ``(V, T)`` rows over the whole ``volume_shape`` grid (C order), numpy in,
    numpy out; time points go through the device in memory-sized chunks.
    """

    def __init__(self, mask: np.ndarray, voxdims, fwhm: float, device=None):
        import torch

        from fastfuncstuff.stats.smooth3d import fwhm_mm_to_sigma_vox, gaussian3d_batched
        from fastfuncstuff.utils import get_device

        self.device = device if device is not None else get_device()
        self.fwhm = float(fwhm)
        self.shape = tuple(int(n) for n in mask.shape)
        self._sigma = fwhm_mm_to_sigma_vox(self.fwhm, tuple(abs(float(v)) for v in voxdims))
        self._idx = torch.as_tensor(np.flatnonzero(mask), device=self.device)
        self._m = torch.as_tensor(mask.astype(np.float32), device=self.device)[None]
        self._denom = gaussian3d_batched(self._m, self._sigma).clamp_min(1e-6)
        self._blur = gaussian3d_batched

    def __call__(self, data: np.ndarray) -> np.ndarray:
        import torch

        from fastfuncstuff.memory import get_available_memory

        x = np.asarray(data)
        out = np.array(x, dtype=np.float32, copy=True)
        if self.fwhm <= 0:
            return out
        n_t = x.shape[1]
        # The block, it masked, the padded working copy and the result.
        per_t = 5 * int(np.prod(self.shape)) * 4
        chunk = max(1, min(n_t, int(get_available_memory(self.device) // per_t)))
        idx = self._idx.cpu().numpy()
        n_vox, n_in = x.shape[0], idx.size
        # Upload each (V, k) column block as it lies and transpose on the device, and
        # bring back only the in-mask rows already in (V, k) order: a host ``.T`` of
        # every block and the transposed write-back were ~12 of the 12.7 s of a
        # 4.2M-voxel x 246 blur.  On CUDA both directions go through page-locked
        # buffers reused across blocks; a download into fresh pageable memory ran
        # at 1.5 GB/s.
        pinned = self.device.type == "cuda"
        stage_in = torch.empty(n_vox * chunk, dtype=torch.float32, pin_memory=pinned)
        stage_out = torch.empty(n_in * chunk, dtype=torch.float32, pin_memory=pinned)
        for t0 in range(0, n_t, chunk):
            t1 = min(n_t, t0 + chunk)
            k = t1 - t0
            rows = stage_in[: n_vox * k].view(n_vox, k)
            np.copyto(rows.numpy(), x[:, t0:t1], casting="same_kind")
            block = rows.to(self.device).T.reshape(k, *self.shape)
            sm = self._blur(block * self._m, self._sigma) / self._denom
            back = stage_out[: n_in * k].view(n_in, k)
            back.copy_(sm.reshape(k, -1)[:, self._idx].T)
            out[idx, t0:t1] = back.numpy()
            del block, sm
        return out


def volume_fwhm(x: np.ndarray, mask: np.ndarray, voxdims, device=None) -> float:
    """Combined FWHM (mm) of ``(V, k)`` time series on ``mask``'s grid, from the mean
    temporal correlation of in-mask neighbours along each axis.

    This is 3dFWHMx's classic ``-unif`` estimate (Forman's ``1 - dvar / 2 var`` *is* the
    neighbour correlation once every voxel has unit variance, which is what ``-unif``
    is for, and what ffs_reml's post-fit ``-save_acf`` uses), computed so that no
    per-voxel scale has to be estimated: a correlation is scale-free. Normalising each
    voxel by a MAD from a few dozen points instead leaves ~10% spatially white scale
    error, which reads as roughness -- a synthetic search then overshot by 14%. The
    surface estimate (``surface_fwhm``) is per-edge correlation for the same reason.
    """
    import torch

    from fastfuncstuff.stats.fwhmx import _geom_combine

    device = device if device is not None else torch.device("cpu")
    m = torch.as_tensor(np.asarray(mask, bool), device=device)
    z, ok = _unit_rows(torch.as_tensor(np.asarray(x, np.float32), device=device), m)
    out = []
    for ax, width in enumerate(abs(float(v)) for v in voxdims):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[ax], hi[ax] = slice(0, -1), slice(1, None)
        pair = ok[tuple(lo)] & ok[tuple(hi)]
        if int(pair.sum()) < 6:
            out.append(0.0)
            continue
        corr = (z[(slice(None), *lo)] * z[(slice(None), *hi)]).sum(0)[pair]
        rho = float(corr.cpu().double().mean())
        out.append(2.35482 * width * np.sqrt(-1.0 / (4.0 * np.log(rho))) if 0 < rho < 1 else 0.0)
    return _geom_combine(*out)


def volume_blur_to_fwhm(
    master: np.ndarray,
    mask: np.ndarray,
    voxdims,
    target: float,
    device=None,
    tol: float = 0.02,
) -> tuple[MaskedGaussian | None, float, float]:
    """The in-mask Gaussian that brings ``master`` (``(V, k)`` on ``mask``'s grid) to
    ``target`` mm by :func:`volume_fwhm`."""
    from fastfuncstuff.utils import get_device

    device = device if device is not None else get_device()
    mask = np.asarray(mask, bool)

    def measure(kernel: float) -> tuple[float, MaskedGaussian | None]:
        if kernel <= 0:
            return volume_fwhm(master, mask, voxdims, device), None
        g = MaskedGaussian(mask, voxdims, kernel, device=device)
        return volume_fwhm(g(master), mask, voxdims, device), g

    return bisect_kernel(measure, target, tol)


# ---------------------------------------------------------------------------
# Spatially varying: diffuse until each place reaches the target (3dBlurToFWHM)
# ---------------------------------------------------------------------------


def _unit_rows(x, m):
    """``(V, k)`` -> ``(k, *m.shape)`` torch, each voxel centred to unit norm over time."""
    import torch

    z = x - x.mean(1, keepdim=True)
    sd = z.norm(dim=1, keepdim=True)
    z = torch.where(sd > 0, z / sd.clamp_min(1e-30), torch.zeros_like(z))
    return z.T.reshape(z.shape[1], *m.shape), m & (sd.reshape(m.shape) > 0)


def local_volume_fwhm(x, mask, voxdims, radius_mm: float, device=None):
    """Per-voxel FWHM (mm, ``mask``-shaped torch, NaN outside): :func:`volume_fwhm`'s
    neighbour correlation, pooled per axis over a Gaussian neighbourhood of
    ``radius_mm`` (its FWHM) inside the mask, then combined. 3dLocalstat -stat FWHM's
    role, scale-free like the global estimate."""
    import torch

    from fastfuncstuff.stats.smooth3d import fwhm_mm_to_sigma_vox, gaussian3d_batched

    device = device if device is not None else torch.device("cpu")
    m = torch.as_tensor(np.asarray(mask, bool), device=device)
    xt = torch.as_tensor(np.asarray(x, np.float32), device=device) if not torch.is_tensor(x) else x
    z, ok = _unit_rows(xt, m)
    widths = [abs(float(v)) for v in voxdims]
    sig = fwhm_mm_to_sigma_vox(radius_mm, tuple(widths))
    logs = torch.zeros(m.shape, device=device)
    n_ax = torch.zeros(m.shape, device=device)
    for ax, width in enumerate(widths):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[ax], hi[ax] = slice(0, -1), slice(1, None)
        pair = (ok[tuple(lo)] & ok[tuple(hi)]).float()
        c = (z[(slice(None), *lo)] * z[(slice(None), *hi)]).sum(0) * pair
        # Each pair counts at both of its voxels.
        num = torch.zeros(m.shape, device=device)
        den = torch.zeros(m.shape, device=device)
        num[tuple(lo)] += c
        num[tuple(hi)] += c
        den[tuple(lo)] += pair
        den[tuple(hi)] += pair
        stack = torch.stack([num, den])
        num, den = gaussian3d_batched(stack, sig)
        rho = (num / den.clamp_min(1e-12)).clamp(1e-6, 1 - 1e-6)
        f = 2.35482 * width * torch.sqrt(-1.0 / (4.0 * torch.log(rho)))
        good = den > 1e-3
        logs += torch.where(good, torch.log(f), torch.zeros_like(f))
        n_ax += good.float()
    out = torch.exp(logs / n_ax.clamp_min(1))
    return torch.where(m & (n_ax > 0), out, torch.full_like(out, float("nan")))


class AdaptiveDiffusion:
    """A recorded sequence of explicit diffusion steps with a per-voxel diffusivity,
    inside a mask (no flux across its edge -- 3dBlurToFWHM's Neumann wall). Replaying
    it on any ``(V, T)`` rows applies exactly the operator the search measured."""

    def __init__(self, mask: np.ndarray, voxdims, device=None):
        import torch

        from fastfuncstuff.utils import get_device

        self.device = device if device is not None else get_device()
        self.shape = tuple(int(n) for n in mask.shape)
        self.m = torch.as_tensor(np.asarray(mask, bool), device=self.device)
        self.h2 = [abs(float(v)) ** 2 for v in voxdims]
        # Explicit Euler is stable for dt * sum(2 / h^2) <= 1 at unit diffusivity.
        self.dt_max = 0.9 / sum(2.0 / h for h in self.h2)
        self.steps: list[tuple[float, object]] = []  # (dt, map of who diffuses, how fast)
        self._idx = torch.as_tensor(np.flatnonzero(np.asarray(mask, bool)), device=self.device)

    def step(self, u, dt: float, d):
        """One step on ``(k, *shape)`` torch ``u``; ``d`` (bool, or a 0..1 diffusivity)
        marks the voxels still diffusing."""
        import torch

        du = torch.zeros_like(u)
        dm = d.float() * self.m
        for ax, h2 in enumerate(self.h2):
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[ax], hi[ax] = slice(0, -1), slice(1, None)
            # Face diffusivity: the smaller side's, so a frozen voxel neither takes
            # nor gives -- and zero across the mask edge.
            face = torch.minimum(dm[tuple(lo)], dm[tuple(hi)])
            flux = face * (u[(slice(None), *hi)] - u[(slice(None), *lo)]) / h2
            du[(slice(None), *lo)] += flux
            du[(slice(None), *hi)] -= flux
        return u + dt * du

    def _volumes(self, x):
        import torch

        k = x.shape[1]
        vol = torch.zeros((k, *self.shape), device=self.device)
        # The in-mask rows as they lie, (V_in, k), transposed on the device.
        rows = np.ascontiguousarray(np.asarray(x)[self._idx.cpu().numpy()], np.float32)
        vol.reshape(k, -1)[:, self._idx] = torch.as_tensor(rows).to(self.device).T
        return vol

    def __call__(self, data: np.ndarray) -> np.ndarray:
        from fastfuncstuff.memory import get_available_memory

        x = np.asarray(data)
        out = np.array(x, dtype=np.float32, copy=True)
        if not self.steps:
            return out
        idx = self._idx.cpu().numpy()
        per_t = 6 * int(np.prod(self.shape)) * 4
        chunk = max(1, min(x.shape[1], int(get_available_memory(self.device) // per_t)))
        for t0 in range(0, x.shape[1], chunk):
            u = self._volumes(x[:, t0 : t0 + chunk])
            for dt, d in self.steps:
                u = self.step(u, dt, d)
            # Back in (V_in, k) order on the device, so the host write is row copies.
            out[idx, t0 : t0 + chunk] = (
                u.reshape(u.shape[0], -1)[:, self._idx].T.contiguous().cpu().numpy()
            )
        return out


def volume_blur_to_fwhm_local(
    master: np.ndarray,
    mask: np.ndarray,
    voxdims,
    target: float,
    radius_mm: float = 8.0,
    device=None,
    max_steps: int = 400,
    tol: float = 0.02,
) -> tuple[AdaptiveDiffusion | None, float, float, Tensor]:
    """Spatially varying blur-to (3dBlurToFWHM's method): diffuse the master in small
    steps, re-measure each voxel's local FWHM (:func:`local_volume_fwhm`, pooled over
    ``radius_mm``) after each, and stop diffusing wherever it has reached ``target``.
    Done when the global FWHM reaches it -- the last step trimmed so it lands -- or
    nothing is left to diffuse. Rough places get more blur than smooth ones, so the
    result is closer to stationary, which is what a single cluster table assumes.

    Returns ``(diffusion, start_fwhm, achieved_fwhm, local_fwhm_map)``; ``diffusion``
    is None when the master is already that smooth. Local estimates need more time
    points than the global one: ~160 keeps the 90th-percentile error near 3%.
    """
    import torch

    from fastfuncstuff.utils import get_device

    device = device if device is not None else get_device()
    mask = np.asarray(mask, bool)
    ad = AdaptiveDiffusion(mask, voxdims, device=device)
    u = ad._volumes(master)

    def glob(u_):
        return volume_fwhm(u_.reshape(u_.shape[0], -1).T.cpu().numpy(), mask, voxdims, device)

    start = glob(u)
    if not np.isfinite(start) or start >= target:
        return (
            None,
            float(start),
            float(start),
            local_volume_fwhm(u.reshape(u.shape[0], -1).T, mask, voxdims, radius_mm, device),
        )
    # Small steps: about a tenth of the remaining FWHM^2 per step, never past stability.
    dt_goal = ((target**2 - start**2) / (8 * np.log(2)) / 2.0) / 10.0
    dt = min(ad.dt_max, max(dt_goal, 1e-4))
    achieved = start
    for _ in range(max_steps):
        local = local_volume_fwhm(u.reshape(u.shape[0], -1).T, mask, voxdims, radius_mm, device)
        d = (local < target) & torch.isfinite(local)
        if not bool(d.any()):
            break
        nxt = ad.step(u, dt, d)
        f = glob(nxt)
        if f >= target * (1 + tol):
            # Trim the last step: the global FWHM is monotone in dt.
            lo, hi = 0.0, dt
            for _ in range(12):
                mid = 0.5 * (lo + hi)
                if glob(ad.step(u, mid, d)) >= target:
                    hi = mid
                else:
                    lo = mid
            ad.steps.append((hi, d))
            u = ad.step(u, hi, d)
            achieved = glob(u)
            break
        ad.steps.append((dt, d))
        u, achieved = nxt, f
        if achieved >= target:
            break
    local = local_volume_fwhm(u.reshape(u.shape[0], -1).T, mask, voxdims, radius_mm, device)
    return ad, float(start), float(achieved), local


# ---------------------------------------------------------------------------
# Blur to a residual tSNR: as much smoothing as each place needs, up to a cap
# ---------------------------------------------------------------------------


def local_volume_tsnr(u, mean, mask, voxdims, radius_mm: float):
    """Per-voxel residual tSNR from ``(k, *shape)`` residual volumes ``u`` and the
    ``mask``-shaped temporal ``mean``: ``|mean| / sd``, the residual variance pooled over
    a Gaussian neighbourhood of ``radius_mm`` inside the mask (an SD from k points
    alone is ~1/sqrt(2k) noisy per voxel)."""
    import torch

    from fastfuncstuff.stats.smooth3d import fwhm_mm_to_sigma_vox, gaussian3d_batched

    m = mask.float()
    sig = fwhm_mm_to_sigma_vox(radius_mm, tuple(abs(float(v)) for v in voxdims))
    var = u.var(dim=0) * m
    pooled = gaussian3d_batched(torch.stack([var, m]), sig)
    sd = torch.sqrt(pooled[0] / pooled[1].clamp_min(1e-12))
    tsnr = mean.abs() / sd.clamp_min(1e-12)
    return torch.where(mask, tsnr, torch.full_like(tsnr, float("nan")))


def volume_blur_to_tsnr(
    master: np.ndarray,
    mean: np.ndarray,
    mask: np.ndarray,
    voxdims,
    target: float,
    max_fwhm: float,
    radius_mm: float = 8.0,
    device=None,
    growth: float = 0.25,
    min_gain: float = 0.03,
    max_steps: int = 60,
):
    """Diffuse each voxel only until its residual tSNR reaches ``target`` -- or its
    local FWHM reaches ``max_fwhm``, where smoothing has stopped paying (structured,
    physiological noise is spatially smooth and does not average down). Noisy places
    get more blur, clean ones little or none.

    Steps are geometric: each voxel's diffusivity is its own local FWHM^2, so every
    step grows every active voxel's FWHM^2 by about ``growth`` (~12% in FWHM) whatever
    its starting smoothness. A fixed step overshoots on white-ish noise (tSNR 25 -> 71
    against a target of 60, in one step) and crawls on noise that is already smooth.

    A voxel also stops for good once a step raised its tSNR by less than ``min_gain``:
    white noise in 3-D gains ~18% per such step, noise that is spatially structured
    almost nothing, and the FWHM cap alone misses it where a pooled local FWHM reads
    low (a narrow smooth slab whose true 19 mm read as 9 kept creeping).

    ``master`` is ``(V, k)`` residuals and ``mean`` ``(V,)`` the temporal mean, both on
    ``mask``'s grid; the mean goes through the same operator. Returns ``(diffusion |
    None, tsnr_before, tsnr_after, local_fwhm_after)`` (``mask``-shaped tensors).
    """
    import math

    import torch

    from fastfuncstuff.utils import get_device

    device = device if device is not None else get_device()
    mask = np.asarray(mask, bool)
    ad = AdaptiveDiffusion(mask, voxdims, device=device)
    u = ad._volumes(master)
    mu = ad._volumes(np.asarray(mean, np.float32).reshape(-1, 1))
    m_t = ad.m

    def fwhm_map(u_):
        return local_volume_fwhm(u_.reshape(u_.shape[0], -1).T, mask, voxdims, radius_mm, device)

    tsnr0 = local_volume_tsnr(u, mu[0], m_t, voxdims, radius_mm)
    tsnr, local = tsnr0, fwhm_map(u)
    stalled = torch.zeros_like(m_t)
    for _ in range(max_steps):
        active = (tsnr < target) & (local < max_fwhm) & torch.isfinite(local) & ~stalled
        if not bool(active.any()):
            break
        lmax = float(local[active].max())
        # Heat time t adds 16 ln2 t to FWHM^2 at full diffusivity.
        t_step = growth * lmax**2 / (16 * math.log(2))
        d = torch.where(active, (local / lmax) ** 2, torch.zeros_like(local))
        n_sub = max(1, math.ceil(t_step / ad.dt_max))
        for _ in range(n_sub):
            ad.steps.append((t_step / n_sub, d))
            u = ad.step(u, t_step / n_sub, d)
            mu = ad.step(mu, t_step / n_sub, d)
        prev = tsnr
        tsnr = local_volume_tsnr(u, mu[0], m_t, voxdims, radius_mm)
        local = fwhm_map(u)
        stalled |= active & (tsnr < prev * (1 + min_gain))
    return (ad if ad.steps else None), tsnr0, tsnr, local
