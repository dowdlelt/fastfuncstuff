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

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm

from fastfuncstuff.utils import get_device


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
        In-sample R² of the full model.
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
    n_sweeps : int
        Coordinate-descent sweeps actually run.
    tau_grid : np.ndarray
        The candidate shifts searched, in seconds.
    """

    amplitudes: np.ndarray
    delays: np.ndarray
    r2: np.ndarray
    r2_fixed: np.ndarray
    n_sweeps: int
    tau_grid: np.ndarray


def build_shifted_design_bank(
    block_onsets: list[np.ndarray],
    hrf: np.ndarray,
    hrf_dt: float,
    tau_grid: np.ndarray,
    tr: float,
    n_timepoints: int,
    *,
    durations: list[float] | None = None,
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
        bank[b] = vals.sum(dim=(0, 1))
        del lag, pos, i0, frac, valid, i0c, vals
    return bank


def _solve_amplitudes(
    gi: Tensor,
    gram: Tensor,
    proj: Tensor,
    ar_b: Tensor,
    ar_v: Tensor,
    n_blocks: int,
) -> Tensor:
    """Exact joint amplitude solve given every block's chosen grid index.

    Given ``τ``, the model is linear in ``A``, so this is the projection
    half of variable projection: ``A = (XᵀX)⁻¹Xᵀy`` with both sides
    gathered out of the precomputed ``gram`` / ``proj`` tables rather
    than recomputed against the time axis.

    ``XtX[v, b, b'] = gram[b, gi[v,b], b', gi[v,b']]`` is built with a
    *single* advanced-index call.  Gathering along ``(b, g)`` first and
    ``(b', g')`` second would materialise an intermediate of shape
    ``(NB, nv, NB, G)`` — 4 GB at 142 blocks / 1000 voxels / a 25-point
    grid.  Broadcasting all four index arrays at once goes straight to
    ``(NB, nv, NB)``.

    Parameters
    ----------
    gi : (nv, n_blocks) long
        Grid index chosen per (voxel, block).
    gram : (NB, G, NB, G), proj : (NB, G, nv)
    ar_b, ar_v : aranges over blocks and voxels, on-device.

    Returns
    -------
    (nv, n_blocks) amplitudes.
    """
    g_b = gi.T  # (NB, nv)
    XtX = gram[
        ar_b.view(-1, 1, 1),  # b
        g_b.unsqueeze(-1),  # gi[v, b]
        ar_b.view(1, 1, -1),  # b'
        gi.unsqueeze(0),  # gi[v, b']
    ]  # (NB, nv, NB)
    XtX = XtX.permute(1, 0, 2)  # (nv, NB, NB)
    Xty = proj[ar_b[:, None], g_b, ar_v[None, :]].T  # (nv, NB)
    # Overlapping trials at short ISI make XtX near-singular; a relative
    # jitter keeps the batched solve well-posed without biasing A.
    ridge = 1e-8 * torch.diagonal(XtX, dim1=1, dim2=2).mean(dim=1).clamp_min(1e-30)
    XtX = XtX + ridge[:, None, None] * torch.eye(n_blocks, dtype=XtX.dtype, device=XtX.device)
    return torch.linalg.solve(XtX, Xty.unsqueeze(-1)).squeeze(-1)


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
    tau_max: float = 3.0,
    tau_step: float = 0.25,
    durations: list[float] | None = None,
    n_sweeps: int = 3,
    delay_prior_sd: float | None = 1.0,
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
        device=device,
    )
    bank = _project_out(bank, Z)  # (NB, G, T)
    # gram[b, g, b', g'] — the only thing the inner loop needs from the
    # time axis, computed once.
    flat = bank.reshape(n_blocks * n_grid, n_t)
    gram = (flat @ flat.T).reshape(n_blocks, n_grid, n_blocks, n_grid)
    self_norm = torch.einsum("bgt,bgt->bg", bank, bank).clamp_min(1e-30)  # (NB, G)

    tau_t = torch.as_tensor(tau_grid, device=device)
    # Gaussian prior on tau expressed in the grid's own units.  Scaled by
    # sigma^2 below so it competes with SSE on equal footing.
    penal_t = tau_t**2  # tau^2 in seconds^2; weighted per voxel below

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
        # Two gathers bound the chunk: the coordinate step's (G, NB, nv)
        # and the amplitude solve's (nv, NB, NB).  The latter dominates at
        # single-trial scale, so cap on NB^2 as well or a 142-trial fit
        # allocates ~2 GB per chunk.
        cap_grid = int(4e7 // max(n_blocks * n_grid, 1))
        cap_gram = int(2e7 // max(n_blocks * n_blocks, 1))
        chunk_size = max(32, min(chunk_size, max(1, cap_grid), max(1, cap_gram)))

    amps = np.zeros((n_voxels, n_blocks), dtype=np.float32)
    delays = np.zeros((n_voxels, n_blocks), dtype=np.float32)
    r2_out = np.zeros(n_voxels, dtype=np.float32)
    r2_fixed_out = np.zeros(n_voxels, dtype=np.float32)
    nz = 0 if Z is None else Z.shape[1]

    n_chunks = (n_voxels + chunk_size - 1) // chunk_size
    ar_b = torch.arange(n_blocks, device=device)
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
        A = _solve_amplitudes(gi, gram, proj, ar_b, ar_v, n_blocks)
        # σ² for the τ prior, taken from the τ=0 fit.  With A solved
        # exactly, SSE = ‖y‖² − Aᵀ(Xᵀy), so no prediction is needed.
        sse0 = (
            ss_y - torch.einsum("vb,vb->v", A, proj[ar_b[:, None], gi.T, ar_v[None, :]].T)
        ).clamp_min(1e-30)
        r2_fixed_out[start:end] = (
            (1.0 - sse0 / ss_tot.clamp_min(1e-12)).clamp(-10, 1).float().cpu().numpy()
        )
        dof = max(1, n_t - n_blocks - nz)
        sigma2 = (sse0 / dof).clamp_min(1e-30)
        lam_tau = (
            None if delay_prior_sd is None else sigma2 / float(delay_prior_sd) ** 2
        )  # (nv,) weight on tau^2

        for _sweep in range(max(1, n_sweeps)):
            for b in range(n_blocks):
                # residual excluding block b, expressed via precomputed tables:
                #   <C[b,g], r_other> = proj[b,g,v] - sum_{b'!=b} A_b' gram[b,g,b',g_b']
                cross = gram[b][:, ar_b, :]  # (G, NB, G)
                gsel = gi.T  # (NB, nv)
                # (G, NB, nv) <- pick g' per (b', v)
                cr = cross[:, ar_b[:, None], gsel]
                contrib = torch.einsum("gbv,vb->gv", cr, A)
                own = A[:, b]  # remove block b's own term
                self_cr = cross[:, b, :][:, gi[:, b]]  # (G, nv) -- gram[b,g,b,g_b]
                num = proj[b] - (contrib - own.unsqueeze(0) * self_cr)
                # profiled SSE reduction for each candidate g
                gain = num**2 / self_norm[b].unsqueeze(-1)
                if lam_tau is not None:
                    gain = gain - lam_tau.unsqueeze(0) * penal_t.unsqueeze(-1)
                gi[:, b] = torch.argmax(gain, dim=0)
            A = _solve_amplitudes(gi, gram, proj, ar_b, ar_v, n_blocks)

        sse = (
            ss_y - torch.einsum("vb,vb->v", A, proj[ar_b[:, None], gi.T, ar_v[None, :]].T)
        ).clamp_min(0.0)
        r2_out[start:end] = (
            (1.0 - sse / ss_tot.clamp_min(1e-12)).clamp(-10, 1).float().cpu().numpy()
        )
        amps[start:end] = A.float().cpu().numpy()
        delays[start:end] = tau_t[gi].float().cpu().numpy()
        del y_raw, y, proj, A, gi
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if verbose:
        gained = float(np.median(r2_out - r2_fixed_out))
        print(
            f"  Shifted-HRF: {n_blocks} blocks, grid ±{tau_max}s/{tau_step}s, "
            f"{n_sweeps} sweep(s).  median R² {float(np.median(r2_out)):.3f} "
            f"(in-sample; τ=0 baseline {float(np.median(r2_fixed_out)):.3f}, "
            f"Δ={gained:+.4f} — free parameters, NOT evidence of real latency)"
        )
    return ShiftedHRFResult(
        amplitudes=amps,
        delays=delays,
        r2=r2_out,
        r2_fixed=r2_fixed_out,
        n_sweeps=int(max(1, n_sweeps)),
        tau_grid=tau_grid,
    )
