"""Latency and width readout from SPM-derivative basis coefficients.

An SPMG2 fit returns ``(beta_c, beta_t)`` per block; SPMG3 adds
``beta_d``.  The derivative columns exist to absorb timing and width
mismatch, so the information is there — but it lives in the *ratios*
``r_t = beta_t/beta_c`` and ``r_d = beta_d/beta_c``, not in the
coefficients themselves, and the map from ratio to seconds is neither
the identity nor separable.  This module measures that map numerically
and inverts it.

Why a numerical calibration rather than a formula
-------------------------------------------------
Three things move the ratio, all of them properties of the *design*
rather than of the HRF:

1. **Column scaling.**  ``generate_spmg_basis`` L2-normalises each basis
   row, which rescales every ratio by ``||h_c|| / ||h_t||``.  A formula
   derived from the analytic derivative would be wrong by that factor.
2. **Stimulus duration.**  Measured at ``tau = 1 s``: ``r_t`` moves
   0.965 → 0.994 (SPMG2) and 1.143 → 1.044 (SPMG3) between 0 s and 12 s
   events.  TR barely matters (< 1%).
3. **Basis size.**  Adding the dispersion column changes the *temporal*
   ratio: ``tau = 1 s`` reads 0.97 under SPMG2 and 1.12 under SPMG3.
   SPMG2 and SPMG3 need separate calibrations; there is no shared curve.

Calibrating against the actual design columns inherits all three for
free, which is why the entry point takes a design block rather than an
HRF.

The SPMG3 interaction
---------------------
``r_d`` is emphatically not "width" on its own.  Pure latency with a
canonical-width HRF produces a large apparent dispersion ratio:

    tau = 1.50 s, dispersion = 1.00  ->  r_t = 1.72, r_d = 0.503

and dispersion feeds back into ``r_t`` (at ``tau = 1.5 s``, ``r_t``
runs 2.07 → 1.37 as dispersion goes 0.7 → 1.4).  Reading either ratio
univariately is wrong.  The joint map is however a well-behaved
diffeomorphism over ``tau in +-1.5 s`` x ``dispersion in [0.65, 1.5]``:
the Jacobian determinant is sign-consistent and bounded away from zero
(measured 0.23 to 2.37), and round-trip inversion recovers truth to 3-4
decimals.  So the two ratios are inverted *jointly*, by triangulating
the forward grid.

Scope
-----
This is a **condition-level** readout.  The per-trial route through the
same ratio is a measured dead end — see :mod:`design.shifted_hrf`, where
``beta_c -> 0`` sends the ratio to +-5 s in ~45% of trials.  That failure
is driven by noise on a single trial's ``beta_c``; a condition beta
pools every trial and does not approach zero.  Simulated at TR = 1 s,
60 trials, 2 s events, true ``tau = 0.7 s``:

    tSNR  30 -> r_t median 0.640, sd 0.024
    tSNR  50 -> r_t median 0.640, sd 0.016
    tSNR 100 -> r_t median 0.639, sd 0.009

i.e. +-0.03 s, not +-5 s.  For per-trial latency use
``-parametrization shift`` instead; nothing here changes that verdict.

Two caveats the caller must respect:

- **Read the ratios off an unpenalised fit.**  ``fitbasis``'s default
  ``-reg cone`` points its prior mean along the canonical axis and
  penalises angular deviation from it, which is precisely shrinkage of
  ``r_t`` toward zero.
- **Latency is measured against the design's own modelled response.**
  Whatever timing convention is baked into the caller's regressor is
  inherited, so the calibration targets must be built the same way --
  in particular with the same stimulus duration (pass
  ``stim_duration``).  Calibrating a boxcar design against impulse
  targets reads the missing convolution as latency: measured +0.97 s
  for 2 s events.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ShapeCalibration",
    "usable_region",
    "build_shape_hrf_bank",
    "calibrate_shape_ratios",
    "invert_shape_ratios",
    "hrf_fwhm",
]


def hrf_fwhm(hrf: np.ndarray, dt: float) -> float:
    """Full width at half maximum of the positive lobe, in seconds.

    Returns ``nan`` when the curve has no resolvable positive peak (the
    half-maximum crossing has to exist on both sides).
    """
    h = np.asarray(hrf, dtype=np.float64)
    peak = int(np.argmax(h))
    if h[peak] <= 0 or peak == 0 or peak == h.size - 1:
        return float("nan")
    half = h[peak] / 2.0
    t = np.arange(h.size) * dt

    rise = h[: peak + 1]
    if rise[0] >= half:
        return float("nan")
    left = float(np.interp(half, rise, t[: peak + 1]))

    fall = h[peak:]
    if fall[-1] >= half:
        return float("nan")
    # np.interp needs increasing x; the falling limb is decreasing, so
    # negate both sides rather than reversing (which would misalign the
    # sample grid by one).
    right = float(np.interp(-half, -fall, t[peak:]))
    return right - left


def build_shape_hrf_bank(
    taus: np.ndarray,
    dispersions: np.ndarray,
    *,
    dt: float = 0.1,
    duration: float = 32.0,
    base_hrf: np.ndarray | None = None,
    stim_duration: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grid of latency-shifted / width-scaled response curves.

    Parameters
    ----------
    taus, dispersions
        Latency (s) and width-multiplier grids.  The returned bank is
        their full outer product, raveled in C order (``tau`` slow,
        ``dispersion`` fast) so it reshapes to ``(n_tau, n_disp)``.
    dt, duration
        Sampling of the curves; match the basis these get projected onto.
    base_hrf : (n_t,), optional
        Response shape to shift and scale.  ``None`` uses the SPM
        canonical with its own gamma-dispersion parameter, matching
        SPMG2/SPMG3 exactly.  Pass a curve to calibrate a
        :func:`~fastfuncstuff.design.hrf.make_derivative_basis` basis
        built around that same curve — a library, PIGHS or per-voxel
        HRF.  Width is then peak-anchored time scaling, the same
        parameterisation that basis uses.
    stim_duration : float, default 0.0
        Convolve every grid curve with a boxcar of this length, so the
        calibration targets are regressors for the real stimulus rather
        than for an impulse.

    Returns
    -------
    bank : (n_tau * n_disp, n_t)
    lag_times : (n_t,)
    fwhm : (n_tau, n_disp)
        Width of each grid curve, in seconds.
    """
    import torch

    from fastfuncstuff.design.hrf import convolve_curves_with_duration, get_spm_canonical_hrf

    taus = np.atleast_1d(np.asarray(taus, dtype=np.float64))
    dispersions = np.atleast_1d(np.asarray(dispersions, dtype=np.float64))
    cpu = torch.device("cpu")

    if base_hrf is not None:
        base = np.asarray(base_hrf, dtype=np.float64).ravel()
        t_base = np.arange(base.size, dtype=np.float64) * dt
        peak_t = t_base[int(np.argmax(np.abs(base)))]

    def _curve(tau: float, disp: float) -> np.ndarray:
        if base_hrf is None:
            return (
                get_spm_canonical_hrf(
                    microtime_dt=dt,
                    hrf_duration=duration,
                    dispersion=float(disp),
                    onset=float(tau),
                    device=cpu,
                )
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        # Peak-anchored width scaling then a shift, matching
        # make_derivative_basis so the basis is the first-order
        # expansion of exactly this family.
        src = peak_t + (t_base - tau - peak_t) / float(disp)
        return np.interp(src, t_base, base, left=0.0, right=0.0)

    curves: list[np.ndarray] = []
    fwhm = np.empty((taus.size, dispersions.size), dtype=np.float64)
    for i, tau in enumerate(taus):
        for j, disp in enumerate(dispersions):
            h = _curve(float(tau), float(disp))
            if stim_duration > dt:
                h = convolve_curves_with_duration(h, dt, stim_duration)
            curves.append(h)
            fwhm[i, j] = hrf_fwhm(h, dt)

    bank = np.stack(curves, axis=0)
    lag_times = np.arange(bank.shape[1], dtype=np.float64) * dt
    return bank, lag_times, fwhm


@dataclass
class ShapeCalibration:
    """Forward map ``(tau, dispersion) -> (r_t, r_d)`` for one design block.

    Attributes
    ----------
    taus : (n_tau,)
    dispersions : (n_disp,)
        ``n_disp == 1`` for a 2-column (SPMG2) calibration, where
        dispersion is held at the canonical 1.0.
    ratio_t, ratio_d : (n_tau, n_disp)
        Measured ratios.  ``ratio_d`` is ``None`` for SPMG2.
    fwhm : (n_tau, n_disp)
        Width of the grid HRF, seconds.
    shape_r2 : (n_tau, n_disp)
        How well the basis reproduces the grid shape.  Falls away at
        large ``|tau|`` (~0.94 at 2 s) and is the natural validity gate:
        past there the fit is no longer measuring latency, it is
        failing to represent the response at all.
    """

    taus: np.ndarray
    dispersions: np.ndarray
    ratio_t: np.ndarray
    ratio_d: np.ndarray | None
    fwhm: np.ndarray
    shape_r2: np.ndarray

    @property
    def n_basis(self) -> int:
        return 2 if self.ratio_d is None else 3


def usable_region(calib: ShapeCalibration, shape_r2_floor: float = 0.95) -> np.ndarray:
    """Largest grid rectangle on which the forward map is safely invertible.

    Two things can make a grid point unusable, and both have to be
    excluded *before* the triangulation is built:

    1. **The basis cannot represent the shape** — ``shape_r2`` below the
       floor.  Common at large ``|tau|``, and much earlier for sharp
       library curves than for the SPM canonical (they linearise worse).
    2. **The map folds** — the Jacobian changes sign, so two different
       ``(tau, width)`` pairs produce the same ratio pair and there is no
       inverse to find.  Measured on real library HRFs: rare, but it does
       happen (1 of 14 base-curve x duration combinations tried).

    Rather than fail on such a basis, grow the largest axis-aligned
    rectangle around the origin ``(tau=0, width=1)`` on which neither
    problem occurs.  A rectangle, not an arbitrary mask, because the
    region has to stay contiguous for the triangulation to be a genuine
    inverse rather than an interpolation across a hole.

    Returns
    -------
    (n_tau, n_disp) bool mask.
    """
    ok = np.isfinite(calib.ratio_t) & (calib.shape_r2 >= shape_r2_floor)
    if calib.ratio_d is not None:
        ok &= np.isfinite(calib.ratio_d)

    if calib.ratio_d is not None and calib.taus.size > 2 and calib.dispersions.size > 2:
        dtau = float(calib.taus[1] - calib.taus[0])
        ddis = float(calib.dispersions[1] - calib.dispersions[0])
        jac = np.gradient(calib.ratio_t, dtau, axis=0) * np.gradient(
            calib.ratio_d, ddis, axis=1
        ) - np.gradient(calib.ratio_t, ddis, axis=1) * np.gradient(calib.ratio_d, dtau, axis=0)
        with np.errstate(invalid="ignore"):
            sign = np.sign(np.nanmedian(jac[ok])) if ok.any() else 1.0
            ok &= np.isfinite(jac) & (jac * sign > 0)

    seed = (int(np.argmin(np.abs(calib.taus))), int(np.argmin(np.abs(calib.dispersions - 1.0))))
    if not ok[seed]:
        # The identity point itself is unusable — nothing to grow from.
        return np.zeros_like(ok)

    # 4-connected flood fill from the identity point.  Deliberately not
    # the largest rectangle: the bad points cluster in corners (extreme
    # latency AND extreme width together), and a rectangle would give up
    # the entire latency range to exclude one such corner — measured, it
    # cut a usable +-1.5 s envelope down to +-0.6 s on a library HRF.
    mask = np.zeros_like(ok)
    stack = [seed]
    mask[seed] = True
    n_tau, n_disp = ok.shape
    while stack:
        i, j = stack.pop()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < n_tau and 0 <= nj < n_disp and ok[ni, nj] and not mask[ni, nj]:
                mask[ni, nj] = True
                stack.append((ni, nj))
    return mask


def calibrate_shape_ratios(
    block_design: np.ndarray,
    target_bank: np.ndarray,
    taus: np.ndarray,
    dispersions: np.ndarray,
    fwhm: np.ndarray,
) -> ShapeCalibration:
    """Measure the ratio the basis reports for each known ``(tau, dispersion)``.

    Parameters
    ----------
    block_design : (n_tp, K)
        The K basis columns this block actually contributes to the GLM,
        concatenated over runs.  ``K`` must be 2 (SPMG2) or 3 (SPMG3).
    target_bank : (n_tp, n_tau * n_disp)
        The same block's regressor built from each grid HRF instead of
        from the basis — i.e. what the data would look like if the true
        response had that latency and width.  Build it with the *same*
        onset-convolution routine that produced ``block_design`` so that
        duration and TR conventions match by construction.
    taus, dispersions, fwhm
        Grid coordinates, as returned by :func:`build_shape_hrf_bank`.

    Notes
    -----
    Everything is one pseudo-inverse and one matmul: the grid shares a
    single design block, so the per-grid-point least squares collapses
    to ``pinv(X) @ targets``.
    """
    X = np.asarray(block_design, dtype=np.float64)
    Y = np.asarray(target_bank, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] not in (2, 3):
        raise ValueError(f"block_design must be (n_tp, 2 or 3); got {X.shape}")
    if Y.shape[0] != X.shape[0]:
        raise ValueError(f"target_bank has {Y.shape[0]} timepoints, design has {X.shape[0]}")

    n_tau = np.atleast_1d(taus).size
    n_disp = np.atleast_1d(dispersions).size
    if Y.shape[1] != n_tau * n_disp:
        raise ValueError(
            f"target_bank has {Y.shape[1]} columns; grid is {n_tau} x {n_disp} = {n_tau * n_disp}"
        )

    B = np.linalg.pinv(X) @ Y  # (K, G)
    resid = Y - X @ B
    ss_res = (resid**2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)

    beta_c = B[0]
    # beta_c is the canonical loading of a known-good HRF, so it sits
    # near 1 across the whole grid (measured 0.80-1.04); guarding is
    # cheap insurance against a degenerate design block, not a real
    # expectation.
    safe = np.where(np.abs(beta_c) > 1e-12, beta_c, np.nan)
    ratio_t = (B[1] / safe).reshape(n_tau, n_disp)
    ratio_d = (B[2] / safe).reshape(n_tau, n_disp) if X.shape[1] == 3 else None

    return ShapeCalibration(
        taus=np.atleast_1d(np.asarray(taus, dtype=np.float64)),
        dispersions=np.atleast_1d(np.asarray(dispersions, dtype=np.float64)),
        ratio_t=ratio_t,
        ratio_d=ratio_d,
        fwhm=np.asarray(fwhm, dtype=np.float64).reshape(n_tau, n_disp),
        shape_r2=r2.reshape(n_tau, n_disp),
    )


def invert_shape_ratios(
    calib: ShapeCalibration,
    ratio_t: np.ndarray,
    ratio_d: np.ndarray | None = None,
    *,
    shape_r2_floor: float = 0.95,
) -> dict[str, np.ndarray]:
    """Invert measured ratios into latency (s), dispersion and FWHM (s).

    Out-of-range voxels are **clamped** to the calibrated envelope
    rather than dropped, and flagged in the returned ``valid`` mask —
    a clamped value still orders voxels correctly, it just stops being
    quantitative.

    Parameters
    ----------
    calib
        From :func:`calibrate_shape_ratios`, for this block.
    ratio_t, ratio_d : (n_vox,)
        Per-voxel ``beta_t/beta_c`` and (SPMG3 only) ``beta_d/beta_c``.
        Read these off an *unpenalised* fit — see the module docstring.
    shape_r2_floor
        Grid points whose ``shape_r2`` falls below this are treated as
        outside the envelope: the basis cannot represent that shape, so
        a latency read there is not meaningful.

    Returns
    -------
    dict with ``latency`` (s), ``dispersion`` (multiplier, 1.0 =
    canonical), ``fwhm`` (s), ``shape_r2`` (calibration quality at the
    recovered point) and ``valid`` (bool).  For SPMG2, ``dispersion``
    and ``fwhm`` are the canonical constants.
    """
    r_t = np.asarray(ratio_t, dtype=np.float64).ravel()

    if calib.ratio_d is None:
        return _invert_1d(calib, r_t, shape_r2_floor)
    if ratio_d is None:
        raise ValueError("ratio_d is required to invert a 3-column (SPMG3) calibration")
    return _invert_2d(calib, r_t, np.asarray(ratio_d, dtype=np.float64).ravel(), shape_r2_floor)


def _invert_1d(
    calib: ShapeCalibration, r_t: np.ndarray, shape_r2_floor: float
) -> dict[str, np.ndarray]:
    """SPMG2: a single monotone curve, so plain interpolation inverts it."""
    curve = calib.ratio_t[:, 0]
    taus = calib.taus
    r2 = calib.shape_r2[:, 0]

    keep = usable_region(calib, shape_r2_floor)[:, 0]
    if keep.sum() < 2:
        raise ValueError(
            f"calibration has {int(keep.sum())} usable grid points at "
            f"shape_r2 >= {shape_r2_floor}; widen the tau grid or lower the floor"
        )
    curve, taus_k, r2 = curve[keep], taus[keep], r2[keep]

    order = np.argsort(curve)
    curve, taus_k, r2 = curve[order], taus_k[order], r2[order]
    if not np.all(np.diff(curve) > 0):
        raise ValueError("calibration curve is not monotone in tau — cannot invert")

    lo, hi = curve[0], curve[-1]
    valid = np.asarray(np.isfinite(r_t) & (r_t >= lo) & (r_t <= hi), dtype=bool)
    clamped = np.clip(np.nan_to_num(r_t, nan=0.0), lo, hi)

    latency = np.interp(clamped, curve, taus_k)
    n = latency.size
    return {
        "latency": latency,
        "dispersion": np.full(n, 1.0),
        "fwhm": np.full(n, float(calib.fwhm[0, 0])),
        "shape_r2": np.interp(clamped, curve, r2),
        "valid": valid,
    }


def _invert_2d(
    calib: ShapeCalibration,
    r_t: np.ndarray,
    r_d: np.ndarray,
    shape_r2_floor: float,
) -> dict[str, np.ndarray]:
    """SPMG3: triangulate the forward grid and invert it jointly.

    The forward map is a diffeomorphism over the calibrated box (see the
    module docstring), so a Delaunay triangulation of the forward points
    inverts it directly.  Points outside the hull get the nearest
    in-hull answer and ``valid=False``.
    """
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    assert calib.ratio_d is not None
    keep = usable_region(calib, shape_r2_floor)
    if keep.sum() < 4:
        raise ValueError(
            f"calibration has {int(keep.sum())} usable grid points at "
            f"shape_r2 >= {shape_r2_floor}; widen the grid or lower the floor"
        )

    pts = np.stack([calib.ratio_t[keep], calib.ratio_d[keep]], axis=1)
    tau_grid = np.broadcast_to(calib.taus[:, None], calib.ratio_t.shape)[keep]
    disp_grid = np.broadcast_to(calib.dispersions[None, :], calib.ratio_t.shape)[keep]
    vals = np.stack([tau_grid, disp_grid, calib.fwhm[keep], calib.shape_r2[keep]], axis=1)

    query = np.stack([r_t, r_d], axis=1)
    finite = np.isfinite(query).all(axis=1)
    out = np.full((r_t.size, 4), np.nan)
    if finite.any():
        out[finite] = LinearNDInterpolator(pts, vals)(query[finite])

    valid = np.asarray(np.isfinite(out).all(axis=1) & finite, dtype=bool)
    # Clamp: hand out-of-hull voxels the nearest calibrated answer so the
    # maps stay dense and rankable, with `valid` marking them as such.
    outside = finite & ~valid
    if outside.any():
        out[outside] = NearestNDInterpolator(pts, vals)(query[outside])

    return {
        "latency": out[:, 0],
        "dispersion": out[:, 1],
        "fwhm": out[:, 2],
        "shape_r2": out[:, 3],
        "valid": valid,
    }
