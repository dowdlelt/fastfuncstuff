"""Per-trial amplitude + latency by exact HRF shifting.

The basis-set route to per-trial latency (SPMG2: canonical + temporal
derivative, latency read off as ``β_d/β_c``) fails for two compounding
reasons, both measured:

1. The linearisation ``h(t − τ) ≈ h(t) − τ·h'(t)`` is only good to about
   ±0.5 s.  Representing a shifted HRF that way costs 7.9 % error at
   τ = 1 s for the SPM canonical and 19.4 % for a typical library HRF
   (library curves are sharper, so they linearise worse).  At τ = 2 s it
   is 29 % / 56 %.
2. Latency lives in a *ratio*, so a Gaussian penalty on ``β_d`` acts
   absolutely while the meaningful quantity is scale-relative.  Left
   free, ``β_d/β_c`` runs to ±5 s of nonsense in ~45 % of trials
   (it explodes whenever ``β_c → 0``); tightened enough to stay inside
   the linearisation's valid window, it collapses to ±0.15 s and carries
   no information.  There is no good operating point between the two.

This module takes the other road.  Each block contributes **one** column,
the voxel's own HRF shifted *exactly*:

.. math::

    y(t) = \\sum_b A_b \\, h(t - \\mathrm{onset}_b - \\tau_b) + Z\\gamma

with ``A_b`` a free linear coefficient and ``τ_b`` a box-bounded
nonlinear parameter reported directly in seconds.  Amplitude and latency
can no longer trade against each other — one is a coefficient, the other
a bounded coordinate — so nonsensical fits are impossible by
construction rather than penalised after the fact.  Exact shifting also
has no validity ceiling on ``τ`` and works for an arbitrary HRF shape,
so a per-voxel curve from a library / PIGHS / ``ffs_hrfopt`` drops in
with no derivative machinery at all.

Measured against SPMG2 + ridge on synthetic data with true
``τ ~ N(0, 1 s)``: latency recovery ``r = 0.63`` vs ``0.08``, and
amplitude median 1.11 vs 0.61 against a true 1.0.

Solver
------
Separable nonlinear least squares (variable projection).  Given every
``τ``, the amplitudes are an exact linear solve, so only the ``τ`` live
in the outer search — one bounded scalar per block.  The outer search is
coordinate descent over blocks with a grid line-search per block.

The trick that makes it fast is that nothing in the inner loop touches
the time axis.  Pre-project the nuisance out of both the design bank and
the data, then precompute

* ``bank``  — ``C[b, g, t]``, the column for block ``b`` at grid shift ``g``
* ``gram``  — ``G[b, g, b', g'] = ⟨C[b,g], C[b',g']⟩``
* ``proj``  — ``Y[b, g, v] = ⟨C[b,g], y_v⟩``

after which every candidate evaluation, and the joint amplitude solve,
is a *gather* from ``gram`` / ``proj`` plus small batched linear algebra.
Cost per coordinate step is O(n_blocks) rather than O(n_timepoints).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm

from fastfuncstuff.utils import get_device, parabolic_peak_offset

AMP_RIDGE_DEFAULT = 1e-3
"""Default ridge on the amplitude solve (relative to ``mean(diag(XtX))``).

See :func:`_solve_amplitudes` for the measurements behind this value.
"""


@dataclass
class ShiftedHRFResult:
    """Output of :func:`fit_shifted_hrf`.

    Attributes
    ----------
    amplitudes : np.ndarray, shape (n_voxels, n_blocks)
        ``A_b`` per voxel — the coefficient on the shifted HRF.  In the
        same units as the input data (no peak-picking involved, so this
        is directly comparable across voxels and trials).
    delays : np.ndarray, shape (n_voxels, n_blocks)
        ``τ_b`` per voxel, **in seconds**.  Positive = response later
        than the nominal onset.  Bounded by the search grid.
    r2 : np.ndarray, shape (n_voxels,)
        In-sample R² **of the task, relative to non-drift variance**:
        ``1 - SSE / ||y_projected||²``, where the nuisance has been removed
        from both terms.  This is the number to look at when asking which
        voxels responded to the task.

        It is deliberately NOT ``1 - SSE / ||y_raw - mean||²``.  With
        per-run polynomial drift (``polort 4`` over 10 runs is 50 nuisance
        columns) the drift absorbs a large share of the raw variance, and
        putting that in the denominator credits it to the model: the same
        fit reported 0.60 on real data by that definition and far less by
        this one.  ``r2_total`` keeps the old convention for reference.
    r2_total : np.ndarray, shape (n_voxels,)
        In-sample R² against RAW total variance, i.e. task + drift together.
        Inflated by the nuisance model and near-useless for identifying
        task-responsive voxels; retained because it is what most tools
        print, so it is what a cross-tool comparison needs.
    r2_fixed : np.ndarray, shape (n_voxels,)
        R² of the same model with every ``τ`` pinned at 0 — the
        no-latency baseline.

        ``r2 − r2_fixed`` is **not** evidence that latency is real.  Both
        are in-sample, and n_blocks free latency parameters always buy
        in-sample fit: on data simulated with *zero* true latency, a
        142-trial fit still gained +0.14.  The gap is a measure of how
        much freedom the latency parameters have, not of whether they
        found anything.  Deciding whether a delay map is believable needs
        held-out prediction (LORO), a split-half correlation of the
        per-trial delays, or a permutation null.
    fstat : np.ndarray, shape (n_voxels,)
        Task F against the nuisance-only model, from ``r2`` (so also relative
        to non-drift variance).  Unlike R² this is charged for the parameters
        spent: ``n_blocks`` amplitudes plus ``n_blocks`` delays.

        Read it as a ranking statistic, not a calibrated p-value.  The delays
        are **grid-searched, not linearly fit**, and selecting the best of
        ``n_grid`` candidates costs more than the one df charged here -- there
        is no clean closed form for how much.  The sensitivity is real: on a
        192-trial fit the median F is 2.83 charging amplitudes only, 1.15
        charging one df per delay (what this reports), and 0.13 charging the
        full log2(n_grid) search cost, i.e. "99.7 % of voxels respond" to
        "0.3 %" on an assumption nobody can pin down.  For a calibrated answer
        use the held-out map from :func:`xval_shifted_hrf`, which sidesteps
        the degrees-of-freedom question entirely by scoring unseen data.
    amp_lambda : np.ndarray, shape (n_voxels,)
        The ridge actually applied to each voxel's amplitude solve, in
        absolute units (added to ``diag(XtX)``).  Saved because a per-voxel
        hyperparameter should ship its own map ([[Per-voxel optimization]]):
        a lambda map that is flat means the empirical-Bayes step found nothing
        to adapt to, and one that tracks the R² map means it is doing its job.
    fstat_df : (int, int)
        ``(numerator, denominator)`` df actually charged, so the caller can
        re-derive a p-value or recompute under a different assumption.
    n_sweeps : int
        Coordinate-descent sweeps actually run.
    tau_grid : np.ndarray
        The candidate shifts searched, in seconds.
    """

    amplitudes: np.ndarray
    delays: np.ndarray
    r2: np.ndarray
    r2_fixed: np.ndarray
    r2_total: np.ndarray
    fstat: np.ndarray
    fstat_df: tuple[int, int]
    amp_lambda: np.ndarray
    n_sweeps: int
    tau_grid: np.ndarray


def _memory_budget(device: torch.device, fraction: float = 0.6) -> int:
    """Free-memory budget in bytes, tolerant of a broken NVML.

    ``get_available_memory`` defaults to calling ``empty_cache()``, which
    pokes the CUDA caching allocator.  On hosts where NVML fails to
    initialise that raises

        NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_() INTERNAL ASSERT FAILED

    from inside the allocator.  This path is hit once per ``fit_shifted_hrf``
    call, and cross-validation calls it once per fold per shape group, so a
    flaky NVML turned into a mid-run crash after the first fold.  Fall back
    through progressively dumber queries rather than taking the process down
    over a memory *hint*.
    """
    total_fallback = 4 * 1024**3
    try:
        from fastfuncstuff.memory import get_available_memory

        return int(get_available_memory(device, empty_cache=False) * fraction)
    except Exception:
        pass
    if device.type == "cuda":
        try:
            free, _total = torch.cuda.mem_get_info(device)
            return int(free * fraction)
        except Exception:
            try:
                props = torch.cuda.get_device_properties(device)
                return int(props.total_memory * 0.5 * fraction)
            except Exception:
                return total_fallback
    return total_fallback


def build_shifted_design_bank(
    block_onsets: list[np.ndarray],
    hrf: np.ndarray,
    hrf_dt: float,
    tau_grid: np.ndarray,
    tr: float,
    n_timepoints: int,
    *,
    durations: list[float] | None = None,
    run_bounds: list[tuple[int, int]] | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """Build ``C[b, g, t]`` — every block's column at every candidate shift.

    The column for block ``b`` at shift ``τ_g`` is
    ``Σ_e h(t_TR − onset_e − τ_g)`` summed over that block's events, with
    ``h`` sampled by linear interpolation on its own ``hrf_dt`` grid and
    taken as zero outside its support.  This is an *exact* shift, not a
    derivative-based approximation, so it stays valid across the whole
    grid rather than only for small ``τ``.

    Parameters
    ----------
    block_onsets : list of arrays
        One entry per block; each an array of event onset times (s).  In
        single-trial mode each block holds exactly one onset.
    hrf : (n_h,)
        The HRF shape, sampled at ``hrf_dt``.  Arbitrary — canonical,
        library, PIGHS, or a per-voxel curve.  Not required to be
        normalised; whatever scale it has is absorbed into ``A``.
    tau_grid : (n_grid,)
        Candidate shifts in seconds.  Its span *is* the box bound on τ.
    durations : list of float, optional
        Per-block stimulus duration (s).  When given and > 0, the HRF is
        boxcar-convolved by summing shifted copies at ``hrf_dt`` spacing.
    run_bounds : list of (start, stop) sample indices, optional
        Run boundaries in the concatenated timeline.  Each block's response
        is confined to the run its onset falls in.  Without this, an event
        in the last ~32 s of a run spills its tail into the NEXT run's
        early samples — physically wrong, since runs are separate
        acquisitions, and it induces spurious coupling between the last
        trials of one run and the first of the next.  Supply it whenever
        the data is concatenated across runs.

    Returns
    -------
    Tensor, shape (n_blocks, n_grid, n_timepoints), float64.
    """
    if device is None:
        device = get_device()
    n_blocks = len(block_onsets)
    n_grid = int(tau_grid.size)
    h = torch.as_tensor(np.asarray(hrf, dtype=np.float64), device=device)
    n_h = h.numel()
    t_tr = torch.arange(n_timepoints, dtype=torch.float64, device=device) * tr
    tau_t = torch.as_tensor(np.asarray(tau_grid, dtype=np.float64), device=device)

    bank = torch.zeros((n_blocks, n_grid, n_timepoints), dtype=torch.float64, device=device)
    for b, onsets in enumerate(block_onsets):
        ons = np.atleast_1d(np.asarray(onsets, dtype=np.float64))
        if ons.size == 0:
            continue
        # sub-events implementing a boxcar of the requested duration
        dur = 0.0 if durations is None else float(durations[b])
        n_sub = max(1, int(round(dur / hrf_dt))) if dur > 0 else 1
        sub = torch.arange(n_sub, dtype=torch.float64, device=device) * hrf_dt
        ons_t = torch.as_tensor(ons, device=device)
        # lag[e, s, g, t] -> collapse e and s, they only ever get summed
        lag = (
            t_tr.view(1, 1, 1, -1)
            - ons_t.view(-1, 1, 1, 1)
            - sub.view(1, -1, 1, 1)
            - tau_t.view(1, 1, -1, 1)
        )
        pos = lag / hrf_dt
        i0 = torch.floor(pos)
        frac = pos - i0
        i0 = i0.to(torch.long)
        valid = (i0 >= 0) & (i0 < n_h - 1)
        i0c = i0.clamp(0, n_h - 2)
        vals = torch.where(valid, h[i0c] * (1.0 - frac) + h[i0c + 1] * frac, torch.zeros_like(frac))
        col = vals.sum(dim=(0, 1))
        if run_bounds is not None:
            # confine to the run containing this block's first onset
            t0 = float(ons.min())
            keep = None
            for r0, r1 in run_bounds:
                if r0 * tr <= t0 < r1 * tr:
                    keep = (r0, r1)
                    break
            if keep is None:
                keep = run_bounds[-1]
            m = torch.zeros(n_timepoints, dtype=col.dtype, device=device)
            m[keep[0] : keep[1]] = 1.0
            col = col * m
            del m
        bank[b] = col
        del lag, pos, i0, frac, valid, i0c, vals, col
    return bank


def _solve_amplitudes(
    gi: Tensor,
    banded: Tensor,
    proj: Tensor,
    nbr_idx: Tensor,
    nbr_w: Tensor,
    ar_b: Tensor,
    ar_w: Tensor,
    ar_v: Tensor,
    n_blocks: int,
    amp_batch: int,
    amp_ridge: float | Tensor = AMP_RIDGE_DEFAULT,
) -> Tensor:
    """Exact joint amplitude solve given every block's chosen grid index.

    Given ``τ``, the model is linear in ``A``, so this is the projection
    half of variable projection: ``A = (XᵀX)⁻¹Xᵀy``, with both sides read
    out of the precomputed tables instead of recomputed against the time
    axis.

    ``XᵀX`` is banded — only overlapping blocks interact — so it is built
    by scattering the band into a dense ``(nb_v, NB, NB)`` for the batched
    Cholesky, rather than gathering ``NB²`` entries of which almost all are
    structurally zero.

    Parameters
    ----------
    gi : (nv, n_blocks) long
        Grid index chosen per (voxel, block).
    banded : (NB, G, W, G)
        ``banded[b, g, w, g'] = <C[b,g], C[nbr[b,w], g']>``.
    proj : (NB, G, nv)
    nbr_idx : (NB, W) long, nbr_w : (NB, W) float
        Neighbour indices and a 0/1 weight masking the padding.
    ar_b, ar_w, ar_v : aranges over blocks, band width, and voxels.
    amp_ridge : float
        Ridge on the amplitude solve, relative to ``mean(diag(XtX))``.  This
        is a **statistical** prior, not the numerical jitter it replaced.

        The delay search chooses each block's shift to maximise fit, and it
        is free to slide two overlapping trials into near-coincidence --
        which improves in-sample fit, because the resulting (+huge, -huge)
        amplitude pair absorbs noise.  Nothing in the profiled per-block
        objective sees the joint conditioning, so nothing prevents it.
        Measured on a 192-trial 2.05 s-ISI design: ``cond(XtX)`` is 61.7 at
        tau=0 but has a median of 1.1e5 at the fitted delays, with 82 % of
        voxels above 1e4 -- and those voxels carry amplitudes an order of
        magnitude too large.  The old 1e-8 was sized to keep the batched
        Cholesky well-posed and is powerless against that.

        Recovery of known per-trial amplitudes on that design, r vs truth:
        0.575 at 1e-8, **0.744 at 1e-3**, 0.745 at 1e-2, 0.684 at 3e-2.  For
        reference, not modelling latency at all scores 0.688 -- so as
        shipped, freeing the delays cost more amplitude accuracy than the
        latency was worth, and 1e-3 reverses that.  Past ~1e-2 amplitude
        shrinks visibly (mean 0.98 -> 0.83 -> 0.61) and delay recovery
        collapses with it.  Delay recovery also improves at 1e-3 (0.567 ->
        0.589), because a stabilised ``A`` feeds the next sweep's search.

    amp_batch : int
        Voxels per sub-batch.  Sub-batching here is what keeps the
        ``(nb_v, NB, NB)`` allocation from dictating the caller's chunk
        size, which previously forced 113-voxel chunks at 420 blocks and
        left the coordinate loop launch-bound.

    Returns
    -------
    (nv, n_blocks) amplitudes.
    """
    nv = gi.shape[0]
    band_w = nbr_idx.shape[1]
    out = torch.empty((nv, n_blocks), dtype=banded.dtype, device=banded.device)
    eye = torch.eye(n_blocks, dtype=banded.dtype, device=banded.device)
    step = max(16, int(amp_batch))
    for s in range(0, nv, step):
        e = min(s + step, nv)
        gi_s = gi[s:e]
        nbv = gi_s.shape[0]
        g_b = gi_s.T  # (NB, nb_v)
        # gi[v, nbr[b, w]] -> (NB, nb_v, W)
        gi_nb = gi_s[:, nbr_idx].permute(1, 0, 2)
        band_vals = banded[
            ar_b.view(-1, 1, 1),  # b
            g_b.unsqueeze(-1),  # gi[v, b]
            ar_w.view(1, 1, -1),  # w
            gi_nb,  # gi[v, nbr[b, w]]
        ]  # (NB, nb_v, W)
        XtX = torch.zeros((nbv, n_blocks, n_blocks), dtype=banded.dtype, device=banded.device)
        # scatter_ADD, not scatter_: the padding slots point at block b itself
        # with weight 0, and a plain scatter lets the last write win, so the
        # padding would zero the real diagonal entry.  Adding is safe because
        # the genuine neighbour indices in a row are unique, so the only
        # duplicates are zero-weight padding.
        XtX.scatter_add_(
            2,
            nbr_idx.view(n_blocks, 1, band_w).expand(n_blocks, nbv, band_w).permute(1, 0, 2),
            (band_vals * nbr_w.view(n_blocks, 1, band_w)).permute(1, 0, 2),
        )
        Xty = proj[ar_b[:, None], g_b, ar_v[s:e][None, :]].T  # (nb_v, NB)
        # Two forms.  A float is a factor RELATIVE to the mean diagonal; a
        # tensor is an ABSOLUTE per-voxel lambda (the empirical-Bayes path,
        # sigma^2_v / tau^2_v, which is not expressible as one global factor
        # because sigma^2 varies by orders of magnitude across a brain).
        _scale = torch.diagonal(XtX, dim1=1, dim2=2).mean(dim=1).clamp_min(1e-30)
        if isinstance(amp_ridge, Tensor):
            ridge = amp_ridge[s:e].to(XtX.dtype).clamp_min(0.0)
        else:
            ridge = max(float(amp_ridge), 0.0) * _scale
        # Floor at the old 1e-8 relative so the batched Cholesky stays
        # well-posed even where the prior asks for (almost) nothing.
        ridge = torch.maximum(ridge, 1e-8 * _scale)
        XtX = XtX + ridge[:, None, None] * eye
        out[s:e] = torch.linalg.solve(XtX, Xty.unsqueeze(-1)).squeeze(-1)
        del XtX, Xty, ridge, band_vals, gi_nb
    return out


def _project_out(mat: Tensor, Z: Tensor | None) -> Tensor:
    """Remove the column space of ``Z`` from the last axis of ``mat``."""
    if Z is None:
        return mat
    # mat: (..., T); Z: (T, nz)
    shp = mat.shape
    flat = mat.reshape(-1, shp[-1])
    coef = torch.linalg.lstsq(Z, flat.T).solution  # (nz, N)
    return (flat - (Z @ coef).T).reshape(shp)


def fit_shifted_hrf(
    data: np.ndarray | Tensor,
    block_onsets: list[np.ndarray],
    hrf: np.ndarray,
    hrf_dt: float,
    tr: float,
    *,
    nuisance: np.ndarray | Tensor | None = None,
    tau_max: float = 2.0,
    tau_step: float = 0.25,
    durations: list[float] | None = None,
    run_bounds: list[tuple[int, int]] | None = None,
    n_sweeps: int = 4,
    delay_prior_sd: float | None = 0.75,
    amp_ridge: float | str = AMP_RIDGE_DEFAULT,
    refine_delays: bool = True,
    device: torch.device | None = None,
    chunk_size: int | None = None,
    verbose: bool = False,
) -> ShiftedHRFResult:
    """Fit per-block amplitude and latency by exact HRF shifting.

    See the module docstring for the model and why it beats the
    derivative-basis route.

    Parameters
    ----------
    data : (n_voxels, n_timepoints)
    block_onsets : list of arrays
        One block per condition, or per trial for single-trial fits.
    hrf, hrf_dt : (n_h,), float
        The response shape and its sample spacing.  Arbitrary shape.
    nuisance : (n_timepoints, n_nuisance), optional
        Drift / motion / etc.  Projected out of both the design bank and
        the data up front, so it never enters the inner loop.
    tau_max, tau_step : float
        Search grid is ``arange(-tau_max, tau_max + step, step)``.
        ``tau_max`` is a hard box bound: no latency outside it is
        representable, which is what makes nonsense impossible.

        Set it wide enough to contain the real latency spread.  Truth
        *outside* the bound does not degrade gracefully: the HRF cannot be
        placed correctly, so the joint amplitude solve compensates using
        overlapping neighbouring trials and produces alternating signed
        amplitudes (see ``test_truth_outside_the_bound_oscillates_amplitudes``).
        That is the nonsensical-output failure mode re-entering through the
        amplitude solve rather than the latency parameter, and it is the
        main reason to leave ``delay_prior_sd`` on.
    n_sweeps : int
        Coordinate-descent sweeps over blocks.  The amplitudes are
        re-solved jointly after each sweep.
    delay_prior_sd : float or None, default 1.0
        Standard deviation (s) of a Gaussian prior pulling each block's
        ``τ`` toward the voxel's own mean ``τ``.  This is not decoration:
        per-trial latency is chosen by *maximising fit*, so at low SNR
        the winning ``τ`` is partly fitting noise and drags amplitude up
        with it (measured: amplitude median inflating to 1.58 against a
        true 1.0 at low SNR with a free ±3 s search).  Shrinking ``τ``
        toward the voxel's central latency removes most of that.  Pass
        ``None`` to disable and get the raw per-trial optimum.
    amp_ridge : float
        Ridge on the amplitude solve, relative to ``mean(diag(XtX))``.  The
        delay search can slide overlapping trials into near-coincidence and
        blow the amplitudes up; this is what stops it.  See
        :func:`_solve_amplitudes` for the measurements.  Set to 0 for the
        old (numerical-only) behaviour.

    Returns
    -------
    ShiftedHRFResult
    """
    if device is None:
        device = get_device()
    y_src = torch.as_tensor(data) if not isinstance(data, torch.Tensor) else data
    if y_src.ndim != 2:
        raise ValueError(f"data must be 2-D (n_voxels, n_t); got {y_src.shape}")
    n_voxels, n_t = y_src.shape
    n_blocks = len(block_onsets)
    if n_blocks == 0:
        raise ValueError("block_onsets is empty — nothing to fit.")
    if tau_step <= 0 or tau_max < 0:
        raise ValueError(f"need tau_step > 0 and tau_max >= 0; got {tau_step}, {tau_max}")

    tau_grid = np.arange(-tau_max, tau_max + 0.5 * tau_step, tau_step, dtype=np.float64)
    n_grid = tau_grid.size
    zero_idx = int(np.argmin(np.abs(tau_grid)))

    Z = None
    if nuisance is not None:
        Z = (
            torch.as_tensor(nuisance, dtype=torch.float64, device=device)
            if not isinstance(nuisance, torch.Tensor)
            else nuisance.to(device=device, dtype=torch.float64)
        )
        if Z.ndim != 2 or Z.shape[0] != n_t:
            raise ValueError(f"nuisance must be (n_t, n_nuisance) with n_t={n_t}; got {Z.shape}")

    bank = build_shifted_design_bank(
        block_onsets,
        hrf,
        hrf_dt,
        tau_grid,
        tr,
        n_t,
        durations=durations,
        run_bounds=run_bounds,
        device=device,
    )
    bank_raw = bank
    bank = _project_out(bank, Z)  # (NB, G, T)
    flat = bank.reshape(n_blocks * n_grid, n_t)
    self_norm = torch.einsum("bgt,bgt->bg", bank, bank).clamp_min(1e-30)  # (NB, G)

    # ---- banded gram ------------------------------------------------
    # The dense gram is (NB, G, NB, G) -- quadratic in BOTH block count and
    # grid size (500 trials x a 17-point grid = 578 MB; 2000 trials = 9.2 GB)
    # and it makes every coordinate step gather over all NB blocks.
    #
    # Most of it is structurally zero, but working out WHICH part needs care.
    # Two trials couple if either
    #   (a) their raw responses overlap in time, or
    #   (b) they share nuisance support -- projecting the drift out of the
    #       bank couples every column through the nuisance subspace, so the
    #       PROJECTED gram is dense even when the raw one is banded.  This
    #       was a real bug: banding on time overlap alone silently dropped
    #       the nuisance-induced terms and wrecked the fit.
    # With per-run polynomial drift, (b) means "same run", so the band comes
    # out as within-run and cross-run pairs are exactly zero.  Both patterns
    # are MEASURED here rather than assumed, so a globally-supported nuisance
    # regressor simply widens the band instead of corrupting the result.
    raw_support = (bank_raw.abs().amax(dim=1) > 0).to(torch.float64)  # (NB, T)
    overlap = (raw_support @ raw_support.T) > 0  # (NB, NB)
    if Z is not None:
        Q, _ = torch.linalg.qr(Z)  # orthonormal basis of the nuisance space
        u = (bank_raw.reshape(n_blocks * n_grid, n_t) @ Q).reshape(n_blocks, n_grid, -1)
        umag = u.abs().amax(dim=1)  # (NB, nz)
        cpl = umag @ umag.T
        # RELATIVE threshold, not `> 0`.  QR leaves ~1e-17 fuzz outside each
        # run's rows, so an exact-zero test marks every cross-run pair as
        # coupled and the band collapses to nearly dense (measured 378/420
        # instead of 42/420).
        overlap = overlap | (cpl > 1e-12 * float(cpl.max()))
        del Q, u, umag, cpl
    del raw_support

    nbr_lists = [np.flatnonzero(overlap[b].cpu().numpy()) for b in range(n_blocks)]
    band_w = max(1, max(len(x) for x in nbr_lists))
    nbr_idx = np.zeros((n_blocks, band_w), dtype=np.int64)
    nbr_msk = np.zeros((n_blocks, band_w), dtype=bool)
    self_pos = np.zeros(n_blocks, dtype=np.int64)
    for b, lst in enumerate(nbr_lists):
        if b not in lst:
            lst = np.union1d(lst, [b])
        nbr_idx[b, : len(lst)] = lst
        nbr_msk[b, : len(lst)] = True
        nbr_idx[b, len(lst) :] = b  # padding points at self; zero-weighted below
        self_pos[b] = int(np.flatnonzero(lst == b)[0])
    nbr_idx_t = torch.from_numpy(nbr_idx).to(device)
    nbr_w_t = torch.from_numpy(nbr_msk.astype(np.float64)).to(device)
    self_pos_t = torch.from_numpy(self_pos).to(device)
    del overlap

    # banded[b, g, w, g'] = <C[b,g], C[nbr[b,w], g']> on the PROJECTED bank
    banded = torch.empty((n_blocks, n_grid, band_w, n_grid), dtype=torch.float64, device=device)
    for b in range(n_blocks):
        nb = bank[nbr_idx_t[b]].reshape(band_w * n_grid, n_t)  # (W*G, T)
        banded[b] = (bank[b] @ nb.T).reshape(n_grid, band_w, n_grid)
        del nb
    if verbose:
        dense_gb = (n_blocks * n_grid) ** 2 * 8 / 1024**3
        band_gb = banded.numel() * 8 / 1024**3
        print(
            f"  Banded gram: width {band_w}/{n_blocks} blocks "
            f"({band_gb:.3f} GB vs {dense_gb:.2f} GB dense, "
            f"{n_blocks / band_w:.1f}x less work per coordinate step)"
        )
    del bank_raw

    # At tau=0 every voxel shares one design, so the OLS variance factor
    # [(XtX)^-1]_bb is a single vector rather than a per-voxel quantity.  That
    # is what makes the empirical-Bayes lambda below essentially free.
    _auto_ridge = isinstance(amp_ridge, str) and amp_ridge == "auto"
    dinv0 = None
    if _auto_ridge:
        X0 = bank[:, zero_idx, :]  # (NB, T)
        G0 = X0 @ X0.T
        G0 = G0 + 1e-10 * torch.diagonal(G0).mean().clamp_min(1e-30) * torch.eye(
            n_blocks, dtype=G0.dtype, device=device
        )
        dinv0 = torch.diagonal(torch.linalg.inv(G0)).clamp_min(0.0)
        del X0, G0

    tau_t = torch.as_tensor(tau_grid, device=device)
    # Gaussian prior on tau expressed in the grid's own units.  Scaled by
    # sigma^2 below so it competes with SSE on equal footing.

    # Real free memory drives both the chunk size and the amplitude-solve
    # sub-batch; [[Memory module]] is the single source of truth rather than
    # hardcoded element counts.
    bytes_budget = _memory_budget(device, 0.6)
    # Amplitude solve peak: the (nv, NB, NB) gather plus the batched
    # Cholesky's workspace, float64.  Give it a third of the budget.
    amp_batch = max(16, int((bytes_budget / 3) // max(n_blocks * n_blocks * 8 * 2, 1)))

    if chunk_size is None:
        from fastfuncstuff.memory import estimate_chunk_size

        chunk_size = estimate_chunk_size(
            n_voxels=n_voxels,
            n_timepoints=n_t,
            n_regressors=n_blocks,
            device=device,
            operation="glm",
            use_double=True,
        )
        # The chunk size is bounded by the coordinate step's (G, NB, nv)
        # gather ONLY.  The amplitude solve's (nv, NB, NB) gather used to
        # cap it too, and that was a serious performance bug: at 420
        # blocks it forced 113 voxels per chunk, so a 381k-voxel brain ran
        # 3374 chunks × 420 blocks × n_sweeps tiny kernels — millions of
        # launches, GPU sitting at 30-50 % util and launch-bound.  The
        # amplitude solve is now sub-batched internally (see
        # ``amp_batch``), which decouples the two constraints: the
        # expensive sequential coordinate loop gets a big chunk, and the
        # memory-hungry solve gets a small one.
        cap_grid = int(bytes_budget // max(n_blocks * n_grid * 8 * 4, 1))
        chunk_size = max(64, min(chunk_size, max(1, cap_grid)))

    amps = np.zeros((n_voxels, n_blocks), dtype=np.float32)
    delays = np.zeros((n_voxels, n_blocks), dtype=np.float32)
    lam_out = np.zeros(n_voxels, dtype=np.float32)
    r2_out = np.zeros(n_voxels, dtype=np.float32)
    r2_fixed_out = np.zeros(n_voxels, dtype=np.float32)
    r2_total_out = np.zeros(n_voxels, dtype=np.float32)
    nz = 0 if Z is None else Z.shape[1]
    # Parameters actually spent: one amplitude per block, plus one delay per
    # block when the search has anywhere to go (n_grid == 1 means pinned).
    p_model = n_blocks * (2 if n_grid > 1 else 1)
    df_den = max(1, n_t - nz - p_model)

    n_chunks = (n_voxels + chunk_size - 1) // chunk_size
    ar_b = torch.arange(n_blocks, device=device)
    ar_w = torch.arange(band_w, device=device)
    for start in tqdm(
        range(0, n_voxels, chunk_size),
        total=n_chunks,
        desc="  Shifted-HRF fit",
        unit="chunk",
        leave=True,
        disable=n_chunks <= 1,
    ):
        end = min(start + chunk_size, n_voxels)
        nv = end - start
        y_raw = y_src[start:end].to(device=device, dtype=torch.float64, non_blocking=True)
        y = _project_out(y_raw, Z)
        ss_tot = ((y_raw - y_raw.mean(dim=1, keepdim=True)) ** 2).sum(dim=1)
        # proj[b, g, v] = <C[b,g], y_v>
        proj = (flat @ y.T).reshape(n_blocks, n_grid, nv)
        ss_y = (y**2).sum(dim=1)  # residual energy after nuisance removal

        gi = torch.full((nv, n_blocks), zero_idx, dtype=torch.long, device=device)

        ar_v = torch.arange(nv, device=device)
        # The auto path needs sigma^2 to set lambda, and sigma^2 comes from a
        # fit -- so bootstrap this first solve with the fixed default, then
        # derive lambda and re-solve before the sweeps begin.
        _ridge_now = AMP_RIDGE_DEFAULT if _auto_ridge else amp_ridge
        A = _solve_amplitudes(
            gi, banded, proj, nbr_idx_t, nbr_w_t, ar_b, ar_w, ar_v, n_blocks, amp_batch, _ridge_now
        )
        # σ² for the τ prior, taken from the τ=0 fit.  With A solved
        # exactly, SSE = ‖y‖² − Aᵀ(Xᵀy), so no prediction is needed.
        sse0 = (
            ss_y - torch.einsum("vb,vb->v", A, proj[ar_b[:, None], gi.T, ar_v[None, :]].T)
        ).clamp_min(1e-30)
        # Task-relative: denominator is the NON-DRIFT variance (ss_y), not the
        # raw total (ss_tot).  ss_y is already mean-free because polort >= 0
        # includes the constant term, so it is exactly "variance left after
        # the nuisance model".
        ss_ref = ss_y.clamp_min(1e-12)
        r2_fixed_out[start:end] = (1.0 - sse0 / ss_ref).clamp(-10, 1).float().cpu().numpy()
        dof = max(1, n_t - n_blocks - nz)
        sigma2 = (sse0 / dof).clamp_min(1e-30)
        lam_tau = (
            None if delay_prior_sd is None else sigma2 / float(delay_prior_sd) ** 2
        )  # (nv,) weight on (tau - tau_bar)^2

        if _auto_ridge:
            # Empirical-Bayes ridge: lambda_v = sigma^2_v / tau^2_v, the exact
            # posterior weight for a Gaussian prior A ~ N(0, tau^2) under noise
            # sigma^2.  A fixed RELATIVE factor cannot express this: sigma^2
            # spans orders of magnitude across a brain, so one factor is too
            # strong where SNR is high and too weak where it is low -- the
            # [[Per-voxel optimization]] complaint exactly.
            #
            # tau^2 by method of moments off the tau=0 fit.  The observed
            # second moment is E[A^2] = tau^2 + sigma^2 * [(XtX)^-1]_bb, so
            # subtract the sampling part instead of treating all the spread as
            # signal.  Floored at 5 % of the observed moment because a voxel
            # whose amplitudes are pure noise would otherwise demand infinite
            # ridge and have its amplitudes crushed to zero.
            assert dinv0 is not None
            a2 = (A**2).mean(dim=1)
            tau2 = (a2 - sigma2 * dinv0.mean()).clamp_min(0.05 * a2.clamp_min(1e-30))
            amp_lam = (sigma2 / tau2).clamp_min(0.0)
            A = _solve_amplitudes(
                gi, banded, proj, nbr_idx_t, nbr_w_t, ar_b, ar_w, ar_v, n_blocks, amp_batch, amp_lam
            )
        else:
            amp_lam = amp_ridge
        # Record what was actually applied.  The fixed path stores its factor
        # as-is (relative to mean(diag(XtX))); the auto path stores absolute
        # lambda.  The two are not on one scale -- the metadata records which
        # mode ran, and a map is only comparable within a mode.
        lam_out[start:end] = (
            amp_lam.float().cpu().numpy() if isinstance(amp_lam, Tensor) else float(amp_lam)
        )

        for _sweep in range(max(1, n_sweeps)):
            # Mean-field centre for the delay prior: the voxel's OWN mean
            # delay across blocks, recomputed once per sweep and held
            # fixed within it (updating mid-sweep feeds back on itself).
            #
            # Centring on the voxel mean rather than on zero is the whole
            # point.  A tau^2 penalty shrinks toward zero, which curbs
            # winner's curse but also biases every genuine delay toward
            # zero -- measured 3x compression, a true 1.2 s coming back as
            # 0.40 s.  Centring on the voxel mean shrinks trial-to-trial
            # JITTER while leaving the voxel's central delay free, so
            # absolute delays stay interpretable.
            #
            # The first sweep runs UNSHRUNK.  Starting from gi = 0 would
            # make tau_bar = 0 on sweep 1, reintroducing exactly the
            # shrink-to-zero bias the centring exists to avoid; it only
            # decays as sweeps re-estimate the centre (measured 0.39 /
            # 0.86 / 1.11 / 1.18 s against a true 1.2 s at 1 / 3 / 6 / 12
            # sweeps).  One unshrunk sweep gives an unbiased centre
            # immediately.  Winner's curse in that sweep is harmless here
            # because only the per-voxel MEAN is taken from it, which
            # averages the noise out over blocks.
            lam_sweep = None if _sweep == 0 else lam_tau
            tau_bar = tau_t[gi].mean(dim=1) if lam_sweep is not None else None
            for b in range(n_blocks):
                # Residual excluding block b, via the precomputed tables:
                #   <C[b,g], r_other> = proj[b,g,v]
                #                       - sum_{b'!=b} A_b' gram[b,g,b',g_b']
                # The sum runs only over b's OVERLAPPING NEIGHBOURS, since
                # every other term is identically zero.  That turns the
                # gather from (G, NB, nv) into (G, W, nv) -- at 420 blocks
                # with a band of ~16 that is ~26x less memory traffic per
                # step, and traffic is what this loop is bound by.
                nb_i = nbr_idx_t[b]  # (W,)
                gsel = gi[:, nb_i].T  # (W, nv)
                cr = banded[b][:, ar_w[:, None], gsel]  # (G, W, nv)
                A_nb = A[:, nb_i] * nbr_w_t[b].unsqueeze(0)  # (nv, W), padding zeroed
                contrib = torch.einsum("gwv,vw->gv", cr, A_nb)
                own = A[:, b]  # remove block b's own term
                self_cr = banded[b][:, self_pos_t[b], :][:, gi[:, b]]  # (G, nv)
                num = proj[b] - (contrib - own.unsqueeze(0) * self_cr)
                # profiled SSE reduction for each candidate g
                gain = num**2 / self_norm[b].unsqueeze(-1)
                if lam_sweep is not None:
                    # tau_bar is set above whenever lam_sweep is not None
                    # (same condition), just outside this per-block loop.
                    assert tau_bar is not None
                    # (tau_g - tau_bar_v)^2 -> (G, nv)
                    dev_tau = tau_t.unsqueeze(-1) - tau_bar.unsqueeze(0)
                    gain = gain - lam_sweep.unsqueeze(0) * dev_tau**2
                gi[:, b] = torch.argmax(gain, dim=0)
            A = _solve_amplitudes(
                gi,
                banded,
                proj,
                nbr_idx_t,
                nbr_w_t,
                ar_b,
                ar_w,
                ar_v,
                n_blocks,
                amp_batch,
                amp_lam,
            )

        # ---- sub-grid delay refinement --------------------------------
        # The grid search reports delays quantised to tau_step, which shows
        # up as visible banding in a delay map.  The profiled objective is
        # smooth in tau (it is built from inner products of a smoothly
        # shifted HRF), so a parabola through the samples around each
        # argmax recovers the peak to a fraction of a step -- the standard
        # cross-correlation sub-sample trick.
        #
        # This is a reporting refinement, deliberately run AFTER the sweeps
        # and without touching gi: the amplitudes were solved on the grid,
        # and re-solving them at off-grid delays would mean building a
        # per-voxel design, which is exactly what this solver's speed
        # depends on avoiding.  The residual mismatch is at most half a
        # step of HRF misplacement, and the HRF is smooth on that scale.
        tau_fine = tau_t[gi]
        if refine_delays and n_grid >= 3:
            for b in range(n_blocks):
                nb_i = nbr_idx_t[b]
                gsel = gi[:, nb_i].T
                cr = banded[b][:, ar_w[:, None], gsel]
                A_nb = A[:, nb_i] * nbr_w_t[b].unsqueeze(0)
                contrib = torch.einsum("gwv,vw->gv", cr, A_nb)
                own = A[:, b]
                self_cr = banded[b][:, self_pos_t[b], :][:, gi[:, b]]
                num = proj[b] - (contrib - own.unsqueeze(0) * self_cr)
                gain = num**2 / self_norm[b].unsqueeze(-1)
                if lam_tau is not None:
                    dev_tau = tau_t.unsqueeze(-1) - tau_t[gi].mean(dim=1).unsqueeze(0)
                    gain = gain - lam_tau.unsqueeze(0) * dev_tau**2
                off = parabolic_peak_offset(gain, gi[:, b])
                tau_fine[:, b] = tau_fine[:, b] + off * tau_step
                del cr, A_nb, contrib, self_cr, num, gain, off
            tau_fine = tau_fine.clamp(-tau_max, tau_max)

        sse = (
            ss_y - torch.einsum("vb,vb->v", A, proj[ar_b[:, None], gi.T, ar_v[None, :]].T)
        ).clamp_min(0.0)
        r2_out[start:end] = (1.0 - sse / ss_ref).clamp(-10, 1).float().cpu().numpy()
        r2_total_out[start:end] = (
            (1.0 - sse / ss_tot.clamp_min(1e-12)).clamp(-10, 1).float().cpu().numpy()
        )
        amps[start:end] = A.float().cpu().numpy()
        delays[start:end] = tau_fine.float().cpu().numpy()
        del y_raw, y, proj, A, gi

    if verbose:
        gained = float(np.median(r2_out - r2_fixed_out))
        print(
            f"  Shifted-HRF: {n_blocks} blocks, grid ±{tau_max}s/{tau_step}s, "
            f"{n_sweeps} sweep(s).  median task R² {float(np.median(r2_out)):.3f} "
            f"(in-sample, vs non-drift variance; τ=0 baseline "
            f"{float(np.median(r2_fixed_out)):.3f}; incl. drift "
            f"{float(np.median(r2_total_out)):.3f}, "
            f"Δ={gained:+.4f} — free parameters, NOT evidence of real latency)"
        )
    r2c = np.clip(r2_out.astype(np.float64), 0.0, 1.0 - 1e-9)
    fstat = ((r2c / p_model) / ((1.0 - r2c) / df_den)).astype(np.float32)
    return ShiftedHRFResult(
        amplitudes=amps,
        delays=delays,
        r2=r2_out,
        r2_fixed=r2_fixed_out,
        r2_total=r2_total_out,
        fstat=fstat,
        fstat_df=(int(p_model), int(df_den)),
        amp_lambda=lam_out,
        n_sweeps=int(max(1, n_sweeps)),
        tau_grid=tau_grid,
    )


def build_blockdiag_polys(
    n_tp_per_run: list[int], polort: int, device: torch.device
) -> Tensor | None:
    """Per-run Legendre drift in zero-padded block-diagonal columns.

    Follows [[Block-diagonal nuisance]]: each run gets its own polynomial
    basis so drift is never shared across run boundaries.
    """
    from fastfuncstuff.glm.core import construct_polynomial_matrix

    if polort < 0:
        return None
    blocks = [
        construct_polynomial_matrix(n, polort, device=device, dtype=torch.float64)
        for n in n_tp_per_run
    ]
    total = sum(n_tp_per_run)
    ncol = sum(b.shape[1] for b in blocks)
    Z = torch.zeros((total, ncol), dtype=torch.float64, device=device)
    r0 = c0 = 0
    for b in blocks:
        Z[r0 : r0 + b.shape[0], c0 : c0 + b.shape[1]] = b
        r0 += b.shape[0]
        c0 += b.shape[1]
    return Z


def append_blockdiag_extras(
    Z: Tensor | None,
    extra_per_run: list[Tensor],
    n_tp_per_run: list[int],
    device: torch.device,
) -> Tensor:
    """Append external per-run nuisance to a block-diagonal drift matrix.

    Same convention as the polynomials and as
    ``builder.pack_for_shared_task_glm``: run *r*'s columns are zero
    outside run *r*, so a component estimated on one run can never explain
    another ([[Block-diagonal nuisance]]).
    """
    n_runs = len(n_tp_per_run)
    if len(extra_per_run) != n_runs:
        raise ValueError(
            f"extra_per_run has {len(extra_per_run)} runs but n_tp_per_run has {n_runs}."
        )
    n_extra = int(extra_per_run[0].shape[1])
    dtype = Z.dtype if Z is not None else torch.float64
    E = torch.zeros((sum(n_tp_per_run), n_runs * n_extra), dtype=dtype, device=device)
    r0 = 0
    for r in range(n_runs):
        x = extra_per_run[r].to(device=device, dtype=dtype)
        if x.shape[0] != n_tp_per_run[r]:
            raise ValueError(
                f"extra_per_run[{r}] has {x.shape[0]} timepoints; run has {n_tp_per_run[r]}."
            )
        E[r0 : r0 + n_tp_per_run[r], r * n_extra : (r + 1) * n_extra] = x
        r0 += n_tp_per_run[r]
    return E if Z is None else torch.cat([Z, E], dim=1)


def xval_shifted_hrf(
    per_run_data: list[Tensor],
    per_run_condition_onsets: list[list[np.ndarray]],
    hrf: np.ndarray,
    hrf_dt: float,
    tr: float,
    polort: int,
    *,
    single_trials: bool,
    shapes: np.ndarray | None = None,
    shape_index: np.ndarray | None = None,
    extra_regs_per_run: list[Tensor] | None = None,
    tau_max: float = 2.0,
    tau_step: float = 0.25,
    delay_prior_sd: float | None = 0.75,
    amp_ridge: float | str = AMP_RIDGE_DEFAULT,
    n_sweeps: int = 4,
    leave_n_out: int = 1,
    device: torch.device | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-run-out held-out R² for the shifted-HRF model.

    Why this shape of validator.  Per-trial parameters cannot predict a
    held-out run — that run's trials were never fit.  So the question this
    answers is the one that *is* answerable and is also the one worth
    asking: **does the latency structure generalise across runs?**  Fit on
    N−1 runs, collapse each condition's trials to a per-voxel condition-
    level ``(A_c, τ_c)``, then predict the held-out run with it.  The same
    fold is scored a second time with ``τ_c`` forced to 0.

    Unlike the in-sample ``r2 − r2_fixed``, the difference between the two
    returned maps is real evidence: the held-out run cannot be fitted by
    spending latency parameters on noise, so a positive gap means the
    estimated delays carry structure that repeats across runs.  A gap at
    or below zero means they do not, however good the in-sample fit looked.

    Aggregation is done on the *grid index*, not on ``τ`` in seconds, so
    the condition-level delay stays exactly on the search grid and the
    held-out design is a plain gather from the test-run bank.

    Both the training fit and the held-out scoring project drift
    fold-locally ([[LORO cross-validation]] — projecting polynomials from
    the full dataset before splitting is the classic way to leak).

    Parameters
    ----------
    per_run_data : list of (n_voxels, n_tp_r)
    per_run_condition_onsets : list over runs of list over conditions
        Onset times **within** each run, in seconds.
    single_trials : bool
        Fit one block per trial (then aggregate per condition to score),
        or one block per condition directly.
    leave_n_out : int
        Runs held out per fold; 1 = LORO.
    shapes : (n_shapes, n_h), optional
        Candidate response shapes.  When given, each fold selects a
        per-voxel shape **from its own training runs** and both scored
        models use it.  This is not optional bookkeeping: shape choice is
        a free parameter that buys in-sample fit exactly as delay does, so
        selecting it once on all the data and then "validating" would
        leak, and scoring a single shared shape while the main fit used
        per-voxel shapes would validate a different model than was fitted.
        ``hrf`` is ignored when this is supplied.
    shape_index : (n_voxels,), optional
        Fixed per-voxel shape assignment (an imported ``ffs_hrfopt`` map).
        Selection is then skipped in every fold, because the shape is an
        input to this model rather than something it fits.  The absolute
        held-out R² is optimistic to the extent that map was chosen on
        these same runs; the ``shift − τ=0`` gap is not, since both scored
        models carry the same fixed shape and only the delays are refit.
    extra_regs_per_run : list of (n_tp_r, n_extra), optional
        External nuisance (motion, denoising components) per run.  Applied
        fold-locally like the polynomials — never projected from the full
        dataset before splitting ([[LORO cross-validation]]).

    Returns
    -------
    (r2_shift, r2_tau0) : each (n_voxels,)
        Held-out R², with estimated delays and with delays pinned to 0.
    """
    if device is None:
        device = get_device()
    n_runs = len(per_run_data)
    if n_runs < 2:
        raise ValueError(f"cross-validation needs >= 2 runs; got {n_runs}")
    n_cond = len(per_run_condition_onsets[0])
    n_voxels = per_run_data[0].shape[0]
    tau_grid = np.arange(-tau_max, tau_max + 0.5 * tau_step, tau_step, dtype=np.float64)
    zero_idx = int(np.argmin(np.abs(tau_grid)))

    folds = [list(range(i, min(i + leave_n_out, n_runs))) for i in range(0, n_runs, leave_n_out)]
    folds = [f for f in folds if f and len(f) < n_runs]

    num_s = np.zeros(n_voxels)
    num_0 = np.zeros(n_voxels)
    den = np.zeros(n_voxels)

    for test_runs in tqdm(
        folds, total=len(folds), desc="  Shifted xval folds", unit="fold", leave=True
    ):
        train_runs = [r for r in range(n_runs) if r not in test_runs]
        n_tp_train = [int(per_run_data[r].shape[1]) for r in train_runs]
        offsets = np.cumsum([0] + n_tp_train[:-1]) * tr

        # blocks in CONCATENATED train time, plus which condition each belongs to
        block_onsets: list[np.ndarray] = []
        block_cond: list[int] = []
        for c in range(n_cond):
            if single_trials:
                for k, r in enumerate(train_runs):
                    for t in np.atleast_1d(per_run_condition_onsets[r][c]):
                        block_onsets.append(np.array([float(t) + offsets[k]]))
                        block_cond.append(c)
            else:
                merged = [
                    np.atleast_1d(per_run_condition_onsets[r][c]) + offsets[k]
                    for k, r in enumerate(train_runs)
                ]
                block_onsets.append(np.concatenate(merged) if merged else np.array([]))
                block_cond.append(c)
        if not block_onsets:
            continue

        y_train = torch.cat([per_run_data[r] for r in train_runs], dim=1)
        Z_train = build_blockdiag_polys(n_tp_train, polort, device)
        if extra_regs_per_run is not None:
            Z_train = append_blockdiag_extras(
                Z_train, [extra_regs_per_run[r] for r in train_runs], n_tp_train, device
            )
        _tb = np.cumsum([0] + n_tp_train)
        train_bounds = [(int(_tb[k]), int(_tb[k + 1])) for k in range(len(train_runs))]
        if shapes is not None:
            # condition-level blocks in the same concatenated train time
            sel_blocks = []
            for c in range(n_cond):
                merged = [
                    np.atleast_1d(per_run_condition_onsets[r][c]) + offsets[k]
                    for k, r in enumerate(train_runs)
                ]
                sel_blocks.append(np.concatenate(merged) if merged else np.array([]))
            fit, fold_shape_idx = fit_shifted_hrf_per_voxel_shape(
                data=y_train,
                block_onsets=block_onsets,
                selection_block_onsets=sel_blocks,
                shapes=shapes,
                shape_index=shape_index,
                hrf_dt=hrf_dt,
                run_bounds=train_bounds,
                tr=tr,
                nuisance=Z_train,
                tau_max=tau_max,
                tau_step=tau_step,
                n_sweeps=n_sweeps,
                delay_prior_sd=delay_prior_sd,
                amp_ridge=amp_ridge,
                device=device,
                verbose=False,
            )
        else:
            fold_shape_idx = None
            fit = fit_shifted_hrf(
                data=y_train,
                block_onsets=block_onsets,
                hrf=hrf,
                hrf_dt=hrf_dt,
                run_bounds=train_bounds,
                tr=tr,
                nuisance=Z_train,
                tau_max=tau_max,
                tau_step=tau_step,
                n_sweeps=n_sweeps,
                delay_prior_sd=delay_prior_sd,
                amp_ridge=amp_ridge,
                device=device,
                verbose=False,
            )
        del y_train, Z_train

        # collapse to condition level per voxel: mean amplitude, median grid index
        bc = np.asarray(block_cond)
        # delays are exactly on-grid, so the index recovers by rounding
        gidx = np.rint((fit.delays.astype(np.float64) + tau_max) / tau_step).astype(np.int64)
        gidx = np.clip(gidx, 0, tau_grid.size - 1)
        A_c = np.zeros((n_voxels, n_cond), dtype=np.float64)
        g_c = np.full((n_voxels, n_cond), zero_idx, dtype=np.int64)
        for c in range(n_cond):
            sel = bc == c
            if not sel.any():
                continue
            A_c[:, c] = fit.amplitudes[:, sel].mean(axis=1)
            g_c[:, c] = np.rint(np.median(gidx[:, sel], axis=1)).astype(np.int64)
        del fit

        A_t = torch.from_numpy(A_c).to(device)
        g_t = torch.from_numpy(g_c).to(device)
        g_zero = torch.full_like(g_t, zero_idx)

        for r in test_runs:
            n_tp_r = int(per_run_data[r].shape[1])
            test_onsets = [np.atleast_1d(per_run_condition_onsets[r][c]) for c in range(n_cond)]
            Z_test = build_blockdiag_polys([n_tp_r], polort, device)
            if extra_regs_per_run is not None:
                Z_test = append_blockdiag_extras(Z_test, [extra_regs_per_run[r]], [n_tp_r], device)
            y_test = per_run_data[r].to(device=device, dtype=torch.float64)
            y_p = _project_out(y_test, Z_test)
            ss_tot = ((y_p - y_p.mean(dim=1, keepdim=True)) ** 2).sum(dim=1)
            ar_c = torch.arange(n_cond, device=device)
            den[:] += ss_tot.cpu().numpy()
            # Predict within shape groups: the test-run bank depends on the
            # shape, so it is built once per occupied group and shared by
            # that group's voxels -- the same structure the fit uses.
            if fold_shape_idx is None:
                groups = [(hrf, np.arange(n_voxels))]
            else:
                groups = [
                    (shapes[si], np.flatnonzero(fold_shape_idx == si))
                    for si in np.unique(fold_shape_idx)
                ]
            for curve, vsel in groups:
                if vsel.size == 0:
                    continue
                bank_test = build_shifted_design_bank(
                    test_onsets, curve, hrf_dt, tau_grid, tr, n_tp_r, device=device
                )  # (n_cond, G, T)
                vt = torch.from_numpy(vsel).to(device)
                for gsel, acc in ((g_t, num_s), (g_zero, num_0)):
                    cols = bank_test[ar_c.unsqueeze(0), gsel[vt]]  # (nvs, n_cond, T)
                    pred = torch.einsum("vc,vct->vt", A_t[vt], cols)
                    pred = _project_out(pred, Z_test)
                    acc[vsel] += ((y_p[vt] - pred) ** 2).sum(dim=1).cpu().numpy()
                    del cols, pred
                del bank_test, vt
            del y_test, y_p, ss_tot

    den_safe = np.maximum(den, 1e-12)
    r2_shift = np.clip(1.0 - num_s / den_safe, -10, 1).astype(np.float32)
    r2_tau0 = np.clip(1.0 - num_0 / den_safe, -10, 1).astype(np.float32)
    if verbose:
        gap = float(np.median(r2_shift - r2_tau0))
        verdict = "delays generalise" if gap > 0 else "delays do NOT generalise"
        print(
            f"  Held-out R² over {len(folds)} fold(s): "
            f"shift={float(np.median(r2_shift)):+.4f}  "
            f"τ=0={float(np.median(r2_tau0)):+.4f}  "
            f"Δ={gap:+.4f}  → {verdict}"
        )
    return r2_shift, r2_tau0


def build_shape_library(
    source: str,
    dt: float,
    duration: float,
    *,
    n_hrfs: int = 20,
    n_flobs_basis: int = 3,
    drop_empty: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Candidate response shapes for per-voxel shape selection.

    Every curve is peak-normalised so the fitted amplitude stays in data
    units and is comparable across voxels that ended up on different
    shapes — without this, "amplitude" would silently mean a different
    thing in each shape group.

    Sources
    -------
    ``library``  the 20-HRF double-gamma library (same file ffs_hrfopt uses)
    ``pighs``    ``n_hrfs`` half-cosine curves, stratified over peak time
    ``flobs``    curves reconstructed from the FLOBS eigenbasis by sampling
                 the empirical MVN(m, C) on its coefficients, so every
                 candidate is a shape the FLOBS prior considers sensible
    ``canonical`` a single curve (degenerate; equivalent to no selection)
    *a path*     a custom HRF library TSV in getcanonicalhrflibrary.tsv
                 format — ``(n_timepoints, n_hrfs)`` at 0.1 s, columns are
                 HRFs — e.g. written by ``ffs_librarian``

    ``drop_empty=False`` keeps every row of the source library even if a
    curve is all zeros, so row *i* of the result is row *i* of the source.
    Required when the caller carries externally computed indices (an
    ``ffs_hrfopt`` index map): silently dropping a curve would renumber
    every shape after it.

    Returns
    -------
    (n_shapes, n_t) curves, and a label per curve.
    """
    from fastfuncstuff.design.hrf import (
        create_pighs_library,
        get_hrf_library,
        get_spm_hrf_with_derivatives,
    )

    cpu = torch.device("cpu")
    key = source.strip().lower()
    if key == "canonical":
        curves = (
            get_spm_hrf_with_derivatives(
                microtime_dt=dt, hrf_duration=duration, n_basis=1, device=cpu
            )
            .cpu()
            .numpy()
        )
        labels = ["canonical"]
    elif key == "library" or Path(source).expanduser().is_file():
        # A path is treated as a custom library TSV (ffs_librarian output).
        # Row order is the contract with any external index map, so it is
        # preserved exactly as loaded.
        lib_path = None if key == "library" else str(Path(source).expanduser())
        curves = (
            get_hrf_library(
                mode="library",
                microtime_dt=dt,
                hrf_duration=duration,
                device=cpu,
                library_path=lib_path,
            )
            .cpu()
            .numpy()
        )
        curves = np.atleast_2d(curves)
        labels = [f"lib{i:02d}" for i in range(curves.shape[0])]
    elif key == "pighs":
        lib, _ = create_pighs_library(n_hrfs=n_hrfs, duration=duration, microtime_dt=dt, device=cpu)
        curves = lib.cpu().numpy()
        labels = [f"pighs{i:02d}" for i in range(curves.shape[0])]
    elif key == "flobs":
        from fastfuncstuff.design.flobs import generate_flobs_basis

        basis = generate_flobs_basis(
            n_basis=n_flobs_basis, n_samples=1000, duration=duration, dt=dt
        )
        # Sample the empirical coefficient prior rather than perturbing
        # coefficients arbitrarily: every candidate is then a shape the
        # FLOBS prior itself regards as plausible, which is the whole
        # point of having (m, C).
        rng = np.random.default_rng(0)
        coefs = rng.multivariate_normal(basis.m, basis.C, size=max(1, n_hrfs))
        coefs[0] = basis.m  # keep the mean shape as a candidate
        curves = coefs @ basis.basis_functions
        labels = [f"flobs{i:02d}" for i in range(curves.shape[0])]
    else:
        raise ValueError(
            f"unknown shape source {source!r}; expected one of canonical, library, pighs, flobs"
        )
    curves = np.atleast_2d(np.asarray(curves, dtype=np.float64))
    # Orient each curve positive, then peak-normalise.  A FLOBS draw can
    # come out sign-flipped; leaving it would put a negative "HRF" in the
    # library and let a voxel fit a positive response with a negative
    # amplitude, which is indistinguishable from real negative BOLD.
    for i in range(curves.shape[0]):
        c = curves[i]
        if abs(c.min()) > abs(c.max()):
            c = -c
        pk = float(np.max(np.abs(c)))
        curves[i] = c / pk if pk > 0 else c
    keep = np.array([np.any(np.abs(c) > 0) for c in curves])
    if not drop_empty:
        if not keep.all():
            raise ValueError(
                f"shape source {source!r} contains "
                f"{int((~keep).sum())} all-zero curve(s) at row(s) "
                f"{np.flatnonzero(~keep).tolist()}.  Row order must be "
                "preserved here (external indices point at it), so they "
                "cannot be dropped — fix the library instead."
            )
        return curves, labels
    return curves[keep], [lbl for lbl, k in zip(labels, keep, strict=True) if k]


def select_shape_per_voxel(
    data: np.ndarray | Tensor,
    block_onsets: list[np.ndarray],
    shapes: np.ndarray,
    hrf_dt: float,
    tr: float,
    *,
    nuisance: Tensor | None = None,
    durations: list[float] | None = None,
    run_bounds: list[tuple[int, int]] | None = None,
    device: torch.device | None = None,
    chunk_size: int | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick each voxel's best-fitting response shape at zero delay.

    Selection is done with all delays pinned to 0 and amplitudes solved
    exactly, so shape and delay are chosen in separate stages rather than
    competing inside one search.  That ordering matters: with both free at
    once, a wrong shape can be partly absorbed by a delay (and vice
    versa), and neither parameter ends up meaning what it says.

    Shape and delay are partly CONFOUNDED, and badly so
    -----------------------------------------------------
    Any library that varies peak time competes directly with the delay
    parameter, because both move the response in time.  Measured on data
    with a true run-stable ±1.2 s delay and a FLOBS library: voxels with
    −1.2 s all selected a 4.1 s-peak curve, voxels with +1.2 s selected
    6.4–6.5 s-peak curves (``corr(shape_index, true delay) = +0.992``),
    the recovered *delay* collapsed to ±0.15 s, and the held-out delay
    gain fell to +0.000 — the shape absorbed the whole thing.

    So with a shape library active:

    - the per-voxel **absolute** delay is not separable from the selected
      shape's peak time.  Read the delay map as *residual* timing beyond
      what the shape already accounted for, and treat ``shape_index`` as
      the primary carrier of voxel-level timing.
    - the per-**trial** delay is still meaningful, because the shape is
      fixed within a voxel across trials, so trial-to-trial deviations
      cannot be absorbed by it.

    If you want one clean absolute delay map, use a single shared shape so
    that all timing is forced into the delay parameter.

    Returns
    -------
    (shape_index, r2_by_shape) : (n_voxels,) long, (n_voxels, n_shapes)
        ``r2_by_shape`` is IN-SAMPLE.  Shape choice is another free
        parameter, so it buys in-sample fit exactly as delay does — judge
        it with :func:`xval_shifted_hrf`, which selects fold-locally.
    """
    if device is None:
        device = get_device()
    y_src = torch.as_tensor(data) if not isinstance(data, torch.Tensor) else data
    n_voxels, n_t = y_src.shape
    n_shapes = shapes.shape[0]
    zero = np.array([0.0])
    if chunk_size is None:
        budget = _memory_budget(device, 0.5)
        chunk_size = max(256, min(n_voxels, int(budget // max(n_t * 8 * 6, 1))))

    ss_res = np.zeros((n_voxels, n_shapes), dtype=np.float32)
    ss_tot = np.zeros(n_voxels, dtype=np.float32)
    n_blocks = len(block_onsets)

    for si in tqdm(
        range(n_shapes),
        total=n_shapes,
        desc="  Shape selection",
        unit="shape",
        leave=False,
        disable=n_shapes <= 1,
    ):
        bank = build_shifted_design_bank(
            block_onsets,
            shapes[si],
            hrf_dt,
            zero,
            tr,
            n_t,
            durations=durations,
            run_bounds=run_bounds,
            device=device,
        )[:, 0, :]  # (NB, T)
        X = _project_out(bank, nuisance).T.contiguous()  # (T, NB)
        XtX = X.T @ X
        XtX = XtX + 1e-8 * float(torch.diagonal(XtX).mean()) * torch.eye(
            n_blocks, dtype=XtX.dtype, device=device
        )
        L = torch.linalg.cholesky(XtX)
        for s in range(0, n_voxels, chunk_size):
            e = min(s + chunk_size, n_voxels)
            y = _project_out(
                y_src[s:e].to(device=device, dtype=torch.float64, non_blocking=True), nuisance
            )
            beta = torch.cholesky_solve(X.T @ y.T, L)  # (NB, nv)
            # SSE = ||y||^2 - beta' X'y  (exact for the OLS solution)
            sse = (y**2).sum(dim=1) - (beta * (X.T @ y.T)).sum(dim=0)
            ss_res[s:e, si] = sse.clamp_min(0).float().cpu().numpy()
            if si == 0:
                ss_tot[s:e] = (
                    ((y - y.mean(dim=1, keepdim=True)) ** 2)
                    .sum(dim=1)
                    .clamp_min(1e-12)
                    .float()
                    .cpu()
                    .numpy()
                )
            del y, beta, sse
        del bank, X, XtX, L

    r2 = 1.0 - ss_res / ss_tot[:, None]
    idx = np.argmax(r2, axis=1).astype(np.int64)
    if verbose:
        counts = np.bincount(idx, minlength=n_shapes)
        top = np.argsort(-counts)[: min(5, n_shapes)]
        print(
            f"  Shape selection over {n_shapes} candidates: "
            + ", ".join(f"#{int(t)}={int(counts[t])}" for t in top)
            + f" (median in-sample R² {float(np.median(r2.max(axis=1))):.3f})"
        )
    return idx, r2.astype(np.float32)


def fit_shifted_hrf_per_voxel_shape(
    data: np.ndarray | Tensor,
    block_onsets: list[np.ndarray],
    shapes: np.ndarray,
    hrf_dt: float,
    tr: float,
    *,
    selection_block_onsets: list[np.ndarray] | None = None,
    shape_index: np.ndarray | None = None,
    nuisance: Tensor | None = None,
    durations: list[float] | None = None,
    run_bounds: list[tuple[int, int]] | None = None,
    tau_max: float = 2.0,
    tau_step: float = 0.25,
    n_sweeps: int = 4,
    delay_prior_sd: float | None = 0.75,
    amp_ridge: float | str = AMP_RIDGE_DEFAULT,
    device: torch.device | None = None,
    verbose: bool = False,
) -> tuple[ShiftedHRFResult, np.ndarray]:
    """Per-voxel HRF shape, then per-trial amplitude + delay on that shape.

    Two stages, deliberately not one.  First each voxel picks its shape at
    zero delay (:func:`select_shape_per_voxel`); then the delay search runs
    within each shape group.  Letting shape and delay compete in a single
    search lets each absorb the other's error — a mis-specified shape can
    masquerade as a delay and vice versa — so the estimates stop meaning
    what their names say.

    This is cheap because the design bank is a function of the *shape*,
    not of the voxel: it is built once per group and shared by every voxel
    assigned to that group.  No per-voxel design is ever constructed.

    Returns
    -------
    (result, shape_index)
        ``result`` fields are in the caller's original voxel order.
    """
    if device is None:
        device = get_device()
    y_src = torch.as_tensor(data) if not isinstance(data, torch.Tensor) else data
    n_voxels = y_src.shape[0]
    n_blocks = len(block_onsets)

    if shape_index is None:
        # Select the shape from CONDITION-level blocks by default: pooling a
        # condition's trials into one regressor spends 1 amplitude DOF
        # instead of n_trials, so the shape comparison is far better
        # determined.  The delay fit that follows still uses the per-trial
        # blocks.
        sel_blocks = selection_block_onsets if selection_block_onsets else block_onsets
        shape_index, _ = select_shape_per_voxel(
            y_src,
            sel_blocks,
            shapes,
            hrf_dt,
            tr,
            nuisance=nuisance,
            durations=None if selection_block_onsets else durations,
            run_bounds=run_bounds,
            device=device,
            verbose=verbose,
        )
    shape_index = np.asarray(shape_index, dtype=np.int64)
    if shape_index.shape != (n_voxels,):
        raise ValueError(f"shape_index must have shape ({n_voxels},); got {shape_index.shape}")

    amps = np.zeros((n_voxels, n_blocks), dtype=np.float32)
    delays = np.zeros((n_voxels, n_blocks), dtype=np.float32)
    r2 = np.zeros(n_voxels, dtype=np.float32)
    r2_fixed = np.zeros(n_voxels, dtype=np.float32)
    r2_total = np.zeros(n_voxels, dtype=np.float32)
    fstat = np.zeros(n_voxels, dtype=np.float32)
    lam = np.zeros(n_voxels, dtype=np.float32)
    fstat_df = (n_blocks * 2, max(1, int(y_src.shape[1]) - n_blocks * 2))
    tau_grid = np.arange(-tau_max, tau_max + 0.5 * tau_step, tau_step, dtype=np.float64)
    sweeps = 1

    used = np.unique(shape_index)
    for si in tqdm(
        used,
        total=used.size,
        desc="  Shift fit per shape",
        unit="shape",
        leave=True,
        disable=used.size <= 1,
    ):
        sel = np.flatnonzero(shape_index == si)
        if sel.size == 0:
            continue
        sub = fit_shifted_hrf(
            data=y_src[torch.from_numpy(sel)],
            block_onsets=block_onsets,
            hrf=shapes[si],
            hrf_dt=hrf_dt,
            tr=tr,
            nuisance=nuisance,
            tau_max=tau_max,
            tau_step=tau_step,
            durations=durations,
            run_bounds=run_bounds,
            n_sweeps=n_sweeps,
            delay_prior_sd=delay_prior_sd,
            amp_ridge=amp_ridge,
            device=device,
            verbose=False,
        )
        amps[sel] = sub.amplitudes
        delays[sel] = sub.delays
        r2[sel] = sub.r2
        r2_fixed[sel] = sub.r2_fixed
        r2_total[sel] = sub.r2_total
        fstat[sel] = sub.fstat
        lam[sel] = sub.amp_lambda
        fstat_df = sub.fstat_df  # identical across groups: same blocks, same nuisance
        sweeps = max(sweeps, sub.n_sweeps)

    if verbose:
        print(
            f"  Shift fit across {used.size} occupied shape group(s): "
            f"median task R² {float(np.median(r2)):.3f} "
            f"(vs non-drift variance; τ=0 baseline "
            f"{float(np.median(r2_fixed)):.3f})"
        )
    return (
        ShiftedHRFResult(
            amplitudes=amps,
            delays=delays,
            r2=r2,
            r2_fixed=r2_fixed,
            r2_total=r2_total,
            fstat=fstat,
            fstat_df=fstat_df,
            amp_lambda=lam,
            n_sweeps=sweeps,
            tau_grid=tau_grid,
        ),
        shape_index,
    )


def shape_time_to_peak(shapes: np.ndarray, hrf_dt: float) -> np.ndarray:
    """Time to peak (s) of each candidate curve.

    This is what makes the shape/delay confound harmless rather than
    destructive.  A voxel responding 1 s later than its neighbour will be
    assigned a later-peaking curve rather than a positive delay, so the
    *delay* map understates it — but the information is not lost, it moved
    here.  Report ``shape_time_to_peak(shapes, dt)[shape_index]`` as the
    voxel's mean response timing, and the per-trial delays as deviations
    around it.
    """
    return (np.argmax(np.asarray(shapes, dtype=np.float64), axis=1) * float(hrf_dt)).astype(
        np.float32
    )
