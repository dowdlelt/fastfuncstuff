"""Timepoint censoring: motion traces, outlier fractions, and censor files.

Ports the two censoring sources afni_proc.py wires into 3dDeconvolve/3dREMLfit,
so a censor file made here agrees TR-for-TR with one AFNI would make:

* **Motion** -- ``1d_tool.py -censor_motion LIMIT -censor_prev_TR`` on the
  3dvolreg ``-1Dfile`` (``roll pitch yaw dS dL dP``, degrees and mm): backward
  difference *within each run* (first TR of a run is 0), Euclidean norm across
  the six columns ("enorm"), keep a TR while ``enorm <= LIMIT``. The previous TR
  is censored too, because a backward difference spans two volumes. Framewise
  displacement (Power 2012) is offered beside it as a second one-line trace.
* **Outliers** -- ``3dToutcount -automask -fraction -polort P -legendre`` per
  run: each voxel is detrended by an **L1** polynomial fit, and a sample is an
  outlier when ``|resid| > qginv(0.001/nt) * sqrt(pi/2) * MAD``. The per-TR
  fraction of automask voxels that are outliers is censored when it is strictly
  greater than the limit (afni_proc's ``1deval -expr "1-step(a-LIMIT)"``).

Censor files are AFNI's keep masks (1 = keep, 0 = censor), one row per TR. Two
masks combine by product (afni_proc's ``1deval -expr a*b``). :func:`censor_to_spikes`
turns a mask into one-hot nuisance columns for tools that take ``-ortvec``
instead of ``-censor``: identical to dropping the rows for OLS, but NOT for an
ARMA/REML noise model, which should drop the rows (ffs_reml ``-censor``).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from tqdm import tqdm

# afni_proc.py's documented values for a bare -censor_motion / -censor_outliers
# are task-dependent (0.2-0.3, 0.05-0.1); these are the ffs defaults, chosen to
# catch real events without eating a run.
DEFAULT_MOTION_LIMIT = 0.5
DEFAULT_OUTLIER_LIMIT = 0.1
# Power et al. 2012: rotations become arc length on a 50 mm sphere.
FD_HEAD_RADIUS_MM = 50.0
# 3dToutcount -qthr default.
DEFAULT_QTHR = 0.001
# afni_proc warns "possible pre-steady state TRs" above this TR-0 outlier fraction.
PRE_STEADY_STATE_LIMIT = 0.4

MOTION_METRICS = ("enorm", "fd")


def _run_bounds(n_timepoints: int, run_lengths: list[int] | None) -> list[tuple[int, int]]:
    if not run_lengths:
        return [(0, n_timepoints)]
    if sum(run_lengths) != n_timepoints:
        raise ValueError(f"run lengths {run_lengths} sum to {sum(run_lengths)}, not {n_timepoints}")
    bounds, start = [], 0
    for n in run_lengths:
        bounds.append((start, start + n))
        start += n
    return bounds


def backward_diff_per_run(x: np.ndarray, run_lengths: list[int] | None = None) -> np.ndarray:
    """``1d_tool.py -derivative``: ``d[t] = x[t] - x[t-1]``, 0 at each run's first TR."""
    a = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(a)
    for s, e in _run_bounds(a.shape[0], run_lengths):
        out[s + 1 : e] = a[s + 1 : e] - a[s : e - 1]
    return out


def motion_enorm(afni_params: np.ndarray, run_lengths: list[int] | None = None) -> np.ndarray:
    """(T,) Euclidean norm of the per-run backward difference of the 6 motion params.

    ``afni_params`` is the 3dvolreg/ffs_moco ``-1Dfile`` layout (degrees, mm).
    Degrees and mm are summed as if commensurate, exactly as AFNI does.
    """
    d = backward_diff_per_run(_as_six_columns(afni_params), run_lengths)
    return np.sqrt((d**2).sum(axis=1))


def framewise_displacement(
    afni_params: np.ndarray,
    run_lengths: list[int] | None = None,
    radius_mm: float = FD_HEAD_RADIUS_MM,
) -> np.ndarray:
    """(T,) Power 2012 FD: ``sum |d translation| + radius * sum |d rotation (rad)|``."""
    d = backward_diff_per_run(_as_six_columns(afni_params), run_lengths)
    rot = np.deg2rad(d[:, :3]) * radius_mm
    return np.abs(rot).sum(axis=1) + np.abs(d[:, 3:]).sum(axis=1)


def motion_trace(
    afni_params: np.ndarray, metric: str, run_lengths: list[int] | None = None
) -> np.ndarray:
    if metric == "enorm":
        return motion_enorm(afni_params, run_lengths)
    if metric == "fd":
        return framewise_displacement(afni_params, run_lengths)
    raise ValueError(f"unknown motion metric {metric!r} (known: {MOTION_METRICS})")


def _as_six_columns(p: np.ndarray) -> np.ndarray:
    a = np.atleast_2d(np.asarray(p, dtype=np.float64))
    if a.shape[1] != 6:
        raise ValueError(f"expected 6 motion columns (roll pitch yaw dS dL dP), got {a.shape}")
    return a


def censor_from_trace(
    trace: np.ndarray,
    limit: float,
    run_lengths: list[int] | None = None,
    censor_prev: bool = True,
    censor_next: bool = False,
    first_trs: int = 0,
) -> np.ndarray:
    """Keep mask (uint8) from a one-line motion trace, ``1d_tool.py -censor_motion``.

    Keep while ``trace <= limit`` (``-moderate_mask`` is inclusive), then extend
    to the next and previous TR, then censor each run's first ``first_trs`` --
    the same order 1d_tool applies them, so first-TR censoring never spreads.
    Extensions stay inside a run; 1d_tool's do not, but a motion derivative is 0
    at every run's first TR, so the two only differ on inputs AFNI never sees.
    """
    t = np.asarray(trace, dtype=np.float64).ravel()
    flagged = ~(np.abs(t) <= limit)  # NaN counts as censored
    out = flagged.copy()
    for s, e in _run_bounds(t.size, run_lengths):
        f = flagged[s:e]
        if censor_next:
            out[s + 1 : e] |= f[:-1]
        if censor_prev:
            out[s : e - 1] |= f[1:]
        if first_trs:
            if first_trs > e - s:
                raise ValueError(f"first_trs={first_trs} exceeds a run of {e - s} TRs")
            out[s : s + first_trs] = True
    return (~out).astype(np.uint8)


def censor_from_outliers(
    fraction: np.ndarray,
    limit: float,
    run_lengths: list[int] | None = None,
    skip_first: int = 0,
) -> np.ndarray:
    """Keep mask (uint8): censor TRs whose outlier fraction is strictly > ``limit``.

    ``skip_first`` exempts each run's first TRs (afni_proc
    ``-regress_skip_first_outliers``), for data whose leading volumes are
    known pre-steady-state and are dropped some other way.
    """
    f = np.asarray(fraction, dtype=np.float64).ravel()
    censored = f > limit
    for s, _ in _run_bounds(f.size, run_lengths):
        censored[s : s + skip_first] = False
    return (~censored).astype(np.uint8)


def combine_censor(*masks: np.ndarray) -> np.ndarray:
    """Product of keep masks: a TR survives only if every source keeps it."""
    out = np.ones_like(np.asarray(masks[0]), dtype=np.uint8)
    for m in masks:
        out &= (np.asarray(m) != 0).astype(np.uint8)
    return out


def censor_to_spikes(keep: np.ndarray) -> np.ndarray:
    """(T, n_censored) one-hot columns, one per censored TR (``1d_tool.py -write_censor_spikes``).

    Zero columns when nothing is censored -- callers must treat that run as
    contributing no block, never as an empty column (fatal to a direct OLS).
    """
    k = np.asarray(keep).ravel()
    idx = np.flatnonzero(k == 0)
    spikes = np.zeros((k.size, idx.size), dtype=np.float32)
    spikes[idx, np.arange(idx.size)] = 1.0
    return spikes


def default_outlier_polort(tr: float, n_timepoints: int) -> int:
    """afni_proc's 3dToutcount polort: 3dDeconvolve's ``1 + floor(TR*nt/150)``."""
    return int(1 + math.floor(tr * n_timepoints / 150.0))


def outlier_alpha(n_timepoints: int, qthr: float = DEFAULT_QTHR) -> float:
    """Outlier cut in MADs: ``qginv(qthr/nt) * sqrt(pi/2)`` (3dToutcount)."""
    from scipy import stats

    return float(stats.norm.isf(qthr / n_timepoints) * math.sqrt(0.5 * math.pi))


def afni_median(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Median with AFNI's even-length convention (mean of the two middle values).

    ``torch.median`` returns the lower middle value, which shifts every MAD on
    even-length runs.
    """
    n = x.shape[dim]
    hi = torch.kthvalue(x, n // 2 + 1, dim=dim).values
    if n % 2:
        return hi
    lo = torch.kthvalue(x, n // 2, dim=dim).values
    return 0.5 * (lo + hi)


def _ipm_step_bound(v: torch.Tensor, dv: torch.Tensor) -> torch.Tensor:
    """Largest step keeping ``v + f*dv >= 0``, per column (lp.fnm's ``bound``)."""
    f = torch.where(dv < 0, -v / dv, torch.full_like(v, 1e20))
    return f.amin(dim=0)


def l1_detrend(
    y: torch.Tensor,
    basis: torch.Tensor,
    max_iter: int = 30,
    rtol: float = 1e-10,
) -> torch.Tensor:
    """Residuals of a per-column L1 (least absolute deviation) fit, batched.

    ``y`` is (T, V), ``basis`` (T, p). 3dToutcount solves this exactly with a
    simplex (``cl1_solve``). IRLS is the obvious batched substitute and the
    wrong one: it crawls (still ~1 voxel/TR off AFNI's counts after 2000
    iterations), and snapping it onto a vertex stalls on a suboptimal one. This
    is the Frisch-Newton interior point (Portnoy & Koenker 1997; quantreg's
    ``rq.fit.fnb``) on the dual LP at tau = 0.5, every voxel at once: each Newton
    step is one p x p solve per voxel, and it reaches the optimum in ~20 steps.
    """
    T, p = basis.shape
    X = basis.to(dtype=y.dtype)
    XX = (X[:, :, None] * X[:, None, :]).reshape(T, p * p)
    beta = 0.99995
    n = float(T)
    b_rhs = 0.5 * X.sum(dim=0)  # (p,)

    def solve(Q: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return torch.linalg.solve((Q.T @ XX).reshape(-1, p, p), rhs.T @ X).T  # (p, V)

    c = -y
    x = torch.full_like(y, 0.5)  # dual variable a, feasible: X'a = 0.5 X'1
    s = 1.0 - x
    yd = torch.linalg.lstsq(X, c).solution  # (p, V)
    r = c - X @ yd
    z = r.clamp_min(0.0)
    w = z - r
    tol = rtol * y.abs().sum(dim=0).clamp_min(torch.finfo(y.dtype).tiny)
    coef = torch.empty_like(yd)
    # Columns leave the working set as they converge, so late iterations only
    # touch the stragglers (the loop is memory-bound, not flop-bound).
    cols = torch.arange(y.shape[1], device=y.device)
    for _ in range(max_iter):
        gap = (c * x).sum(0) - (yd * b_rhs[:, None]).sum(0) + w.sum(0)
        done = gap <= tol
        if bool(done.any()):
            coef[:, cols[done]] = yd[:, done]
            keep = ~done
            if not bool(keep.any()):
                return y - X @ (-coef)
            cols = cols[keep]
            c, x, s, yd, z, w, tol = (
                c[:, keep],
                x[:, keep],
                s[:, keep],
                yd[:, keep],
                z[:, keep],
                w[:, keep],
                tol[keep],
            )
        q = 1.0 / (z / x + w / s)
        r = z - w
        rhs = q * r
        dy = solve(q, rhs)
        dx = q * (X @ dy - r)
        ds = -dx
        dz = -z * (dx / x + 1.0)
        dw = -w * (ds / s + 1.0)
        fp = (beta * torch.minimum(_ipm_step_bound(x, dx), _ipm_step_bound(s, ds))).clamp_max(1.0)
        fd = (beta * torch.minimum(_ipm_step_bound(w, dw), _ipm_step_bound(z, dz))).clamp_max(1.0)
        # Mehrotra corrector wherever the affine step was blocked. Computed for
        # every column and selected, since columns are independent.
        mu = (z * x).sum(0) + (w * s).sum(0)
        g = ((z + fd * dz) * (x + fp * dx)).sum(0) + ((w + fd * dw) * (s + fp * ds)).sum(0)
        mu = mu * (g / mu) ** 3 / (2.0 * n)
        dxdz = dx * dz
        dsdw = ds * dw
        xinv = 1.0 / x
        sinv = 1.0 / s
        xi = mu * (xinv - sinv)
        dy_c = solve(q, rhs + q * (dxdz - dsdw - xi))
        dx_c = q * (X @ dy_c + xi - r - dxdz + dsdw)
        ds_c = -dx_c
        dz_c = mu * xinv - z - xinv * z * dx_c - dxdz
        dw_c = mu * sinv - w - sinv * w * ds_c - dsdw
        fp_c = (beta * torch.minimum(_ipm_step_bound(x, dx_c), _ipm_step_bound(s, ds_c))).clamp_max(
            1.0
        )
        fd_c = (beta * torch.minimum(_ipm_step_bound(w, dw_c), _ipm_step_bound(z, dz_c))).clamp_max(
            1.0
        )
        corr = torch.minimum(fp, fd) < 1.0
        dx = torch.where(corr, dx_c, dx)
        ds = torch.where(corr, ds_c, ds)
        dy = torch.where(corr, dy_c, dy)
        dz = torch.where(corr, dz_c, dz)
        dw = torch.where(corr, dw_c, dw)
        fp = torch.where(corr, fp_c, fp)
        fd = torch.where(corr, fd_c, fd)
        x = x + fp * dx
        s = s + fp * ds
        yd = yd + fd * dy
        w = w + fd * dw
        z = z + fd * dz
    coef[:, cols] = yd
    return y - X @ (-coef)


def outlier_counts(
    series: torch.Tensor,
    polort: int,
    qthr: float = DEFAULT_QTHR,
    chunk_size: int | None = None,
    progress: bool = False,
) -> torch.Tensor:
    """(T,) count of outlier voxels per TR in ``series`` (T, V), 3dToutcount's rule.

    Voxels with MAD = 0 contribute no outliers but still sit in V, the
    denominator of the fraction, as in AFNI.
    """
    from fastfuncstuff.glm.core import construct_polynomial_matrix
    from fastfuncstuff.memory import estimate_chunk_size
    from fastfuncstuff.utils import linalg_device

    T, V = series.shape
    # float64 everywhere: in float32 the interior point loses the optimum
    # (measured up to 569 voxels/TR off AFNI, vs <= 1 in float64). On CUDA it
    # is still ~10x faster than 3dToutcount; Metal has no float64 at all.
    device = linalg_device(series.device)
    dtype = torch.float64
    alpha = outlier_alpha(T, qthr)
    basis = (
        construct_polynomial_matrix(T, polort, device=device, dtype=dtype) if polort > 0 else None
    )
    if chunk_size is None:
        # The CPU loop is memory-bound: chunks that stay cache-resident ran 6x
        # faster than one 31k-voxel block (1024 measured best of 256..32768).
        chunk_size = estimate_chunk_size(
            V,
            T,
            polort + 1,
            device,
            operation="l1_detrend",
            max_chunk_size=1024 if device.type == "cpu" else None,
        )
    counts = torch.zeros(T, dtype=torch.int64, device=device)
    starts = range(0, V, chunk_size)
    for s in tqdm(starts, desc="outliers", leave=True, disable=not progress or len(starts) < 2):
        y = series[:, s : s + chunk_size].to(device=device, dtype=dtype)
        if basis is None:
            r = y - afni_median(y, dim=0)
        else:
            r = l1_detrend(y, basis)
        mad = afni_median(r.abs(), dim=0)
        top = alpha * mad
        out = (r.abs() > top) & (mad > 0)
        counts += out.sum(dim=1)
    return counts


def outlier_fraction_4d(
    data: torch.Tensor,
    run_lengths: list[int] | None = None,
    tr: float | None = None,
    polort: int | None = None,
    mask: torch.Tensor | None = None,
    qthr: float = DEFAULT_QTHR,
    progress: bool = False,
) -> tuple[np.ndarray, list[int]]:
    """Per-TR outlier fraction for a (T, nz, ny, nx) series, run by run.

    Each run is its own 3dToutcount call, as afni_proc does it: its own automask
    (from that run's mean |value|, ``THD_automask``) unless ``mask`` is given,
    and its own polort (``1 + floor(TR*nt/150)`` unless ``polort`` is given).
    Returns the concatenated fractions and the voxel count each run used.
    """
    from fastfuncstuff.processing.mask import afni_automask

    T = data.shape[0]
    fractions, nvox = [], []
    for s, e in _run_bounds(T, run_lengths):
        run = data[s:e]
        nt = e - s
        if nt < 5:
            raise ValueError(f"outlier counting needs >= 5 TRs per run, got {nt}")
        if mask is None:
            m = afni_automask(run.abs().mean(dim=0))
        else:
            m = mask.to(device=run.device, dtype=torch.bool)
        p = polort
        if p is None:
            if tr is None:
                raise ValueError("outlier polort needs either polort= or tr=")
            p = default_outlier_polort(tr, nt)
        flat = run.reshape(nt, -1)[:, m.reshape(-1).to(run.device)]
        counts = outlier_counts(flat, p, qthr=qthr, progress=progress)
        nvox.append(int(flat.shape[1]))
        fractions.append(counts.cpu().numpy() / max(flat.shape[1], 1))
    return np.concatenate(fractions), nvox


def read_1d_column(path: str) -> np.ndarray:
    """One number per row (AFNI ``.1D``), comments (``#``) skipped."""
    return np.loadtxt(path, comments="#", ndmin=2)[:, 0]


def write_1d(path: str, values: np.ndarray, fmt: str = "%g") -> None:
    np.savetxt(path, np.asarray(values).reshape(len(values), -1), fmt=fmt)
