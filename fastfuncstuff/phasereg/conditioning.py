"""Phase time-series conditioning ahead of the magnitude-on-phase fit.

Phase regression needs each voxel's *temporal* phase change only, so two things
that are irrelevant to the fit but large in real data are removed here:

* **2π temporal wraps.** ``temporal_unwrap`` adds a multiple of 2π at each
  step that looks like a wrap. On already-unwrapped input it changes nothing; on
  wrapped input (complex-domain resampling, raw scanner phase) it is what makes
  the series fittable at all. No spatial unwrapping is needed or attempted.

* **Per-volume global field fluctuation.** Breathing, frequency drift and
  recon-level phase offsets shift the whole brain's phase together from one
  volume to the next. That is variance in the regressor with no magnitude
  counterpart, so it attenuates the slope (Menon 2002 avoided it with a
  navigator echo; Bodenschatz & Rowe 2026 fit and remove a low-order spatial
  polynomial per volume). Vessel phase is a local dipole field, which a
  brain-wide low-order polynomial cannot represent, so it survives.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch

from fastfuncstuff import policy_linalg as plinalg


def _segments(n_tp: int, run_lengths: list[int] | None) -> list[tuple[int, int]]:
    lengths = run_lengths if run_lengths else [n_tp]
    if sum(lengths) != n_tp:
        raise ValueError(f"run_lengths sum to {sum(lengths)}, data has {n_tp} timepoints")
    bounds = np.cumsum([0, *lengths])
    return [(int(a), int(b)) for a, b in itertools.pairwise(bounds)]


def temporal_unwrap(phase: torch.Tensor, run_lengths: list[int] | None = None) -> torch.Tensor:
    """Remove 2π wraps along time, per run.

    A step of at least π is corrected by the multiple of 2π that brings it under
    π/2, and left alone if no such multiple exists. A genuine wrap of a smooth
    series lands near zero once corrected. A step that does not (motion,
    interpolation across a spatial unwrap seam, phase at the noise floor) is not
    a wrap, and "unwrapping" it would only add a random 2π level shift. np.unwrap
    corrects every step and so does exactly that; the phaseprep parity data has
    such steps in a quarter of its voxels.

    Parameters
    ----------
    phase : Tensor (n_timepoints, n_voxels), radians.
    run_lengths : per-run lengths; each run is unwrapped on its own so a jump
        across a run boundary is never "corrected".

    Returns
    -------
    Tensor like ``phase``, equal to it modulo 2π at every sample.
    """
    out = phase.clone()
    two_pi = 2.0 * torch.pi
    for a, b in _segments(phase.shape[0], run_lengths):
        step = phase[a + 1 : b] - phase[a : b - 1]
        wrapped_step = torch.remainder(step + torch.pi, two_pi) - torch.pi
        is_wrap = (step.abs() >= torch.pi) & (wrapped_step.abs() < torch.pi / 2)
        correction = torch.where(is_wrap, wrapped_step - step, 0.0)
        out[a + 1 : b] += torch.cumsum(correction, dim=0)
    return out


def _legendre_basis(coords: torch.Tensor, order: int) -> torch.Tensor:
    """Total-degree <= order products of Legendre polynomials in each coordinate."""
    c = coords.double()
    lo, hi = c.min(0).values, c.max(0).values
    u = (2.0 * (c - lo) / (hi - lo).clamp(min=1.0) - 1.0).numpy()
    cols = []
    for degs in itertools.product(range(order + 1), repeat=u.shape[1]):
        if sum(degs) <= order:
            col = np.ones(u.shape[0])
            for axis, deg in enumerate(degs):
                col = col * np.polynomial.legendre.Legendre.basis(deg)(u[:, axis])
            cols.append(col)
    return torch.from_numpy(np.stack(cols, axis=1))


def _fit_term(centred: torch.Tensor, keep: torch.Tensor, order: int, coords) -> torch.Tensor:
    """Per-volume term fit on ``keep`` voxels, evaluated at every column."""
    if order == 0:
        g = centred[:, keep].median(dim=1).values
        return g.unsqueeze(1).expand_as(centred)
    basis = _legendre_basis(coords, order).to(centred.device)
    coef = plinalg.lstsq(basis[keep], centred[:, keep].T.double()).solution
    return (basis @ coef).T.to(centred.dtype)


def remove_global_phase(
    phase: torch.Tensor,
    order: int,
    fit_mask: torch.Tensor,
    coords: torch.Tensor | None = None,
    run_lengths: list[int] | None = None,
    groups: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subtract a per-volume spatial polynomial from temporally-centred phase.

    Each voxel is first centred on its own temporal mean (per run), which strips
    the static field — many radians across a brain — and leaves only the
    dynamic part, where a per-volume common term is visible. The term is fit on
    ``fit_mask`` voxels, excluding the 10% with the largest temporal variance
    (edge, seam and vessel voxels) so it describes bulk tissue; it is then
    subtracted from every voxel.

    With ``groups`` (one integer label per voxel, e.g. the acquired slice) the
    term is fit and removed within each group separately. That is the right
    unit for 2-D multi-slice EPI: slices are read at different moments of the
    breathing cycle, so one volume carries as many field states as it has
    acquisition times, and only native-space (pre-resampling) data keeps them
    apart.

    Parameters
    ----------
    phase : Tensor (n_timepoints, n_voxels), temporally continuous radians.
    order : 0 = robust (median) offset; N >= 1 = Legendre polynomial of total
        degree N in ``coords``.
    fit_mask : Tensor (n_voxels,) bool — voxels allowed to inform the fit.
    coords : Tensor (n_voxels, D) coordinates, required for order >= 1; with
        ``groups``, pass the in-group (in-plane) coordinates only.
    run_lengths : per-run lengths for the temporal centring.
    groups : Tensor (n_voxels,) int or None.

    Returns
    -------
    cleaned : Tensor like ``phase``.
    term : the fitted term averaged over each group's ``fit_mask`` voxels —
        (n_timepoints,) without groups, (n_timepoints, n_groups) with them, in
        ``torch.unique(groups)`` order. A QC trace of what was removed.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    if order >= 1 and coords is None:
        raise ValueError("order >= 1 needs voxel coordinates")
    centred = phase.clone()
    for a, b in _segments(phase.shape[0], run_lengths):
        centred[a:b] -= centred[a:b].mean(0)

    fm = fit_mask.bool()
    labels = torch.zeros(phase.shape[1], dtype=torch.long) if groups is None else groups.long()
    fitted = torch.zeros_like(phase)
    traces = []
    for lab in torch.unique(labels):
        sel = labels == lab
        sub = centred[:, sel]
        sub_fm = fm[sel]
        if int(sub_fm.sum()) < 10:
            if groups is None:
                raise ValueError("too few voxels to estimate the global phase term")
            traces.append(torch.zeros(phase.shape[0], dtype=phase.dtype))
            continue
        var = sub[:, sub_fm].var(0)
        keep = torch.zeros_like(sub_fm)
        keep[sub_fm] = var <= torch.quantile(var.double(), 0.9).to(var.dtype)
        term = _fit_term(sub, keep, order, None if coords is None else coords[sel])
        fitted[:, sel] = term
        traces.append(term[:, sub_fm].mean(1))
    trace = traces[0] if groups is None else torch.stack(traces, dim=1)
    return phase - fitted, trace
