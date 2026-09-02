"""Is a 4-D map task-locked? — the referee for BOLD leaking into a motion estimate.

A displacement estimator that closes a brightness-constancy data term (optical flow,
a windowed-correlation searchlight, qwarp) cannot tell "this boundary got brighter"
from "this boundary moved".  Under a block design the brightness change is large,
tissue-specific and sits on exactly the edges the estimator tracks, so the estimated
field acquires a task-locked component that is not motion.  See the wiki note
``Frame brightness and brightness constancy`` for why no purely intensity-based
defence closes this.

This module measures the contamination rather than assuming it.  Four decisions carry
the result, and the first two are corrections of a first version that got them wrong:

* **Signed correlation, not R2.**  A block design is essentially ONE frequency, which
  carries two degrees of freedom.  Any narrowband signal at that frequency with a
  random phase captures ``cos^2(dphi)`` of it, so its R2 averages **0.5** — R2 is
  structurally unusable here whatever null is wrapped around it.  Signed ``r`` has
  mean 0 under the same surrogate, and its sign is the information that matters:
  real contamination pushes a boundary in a CONSISTENT direction when the stimulus
  is on.
* **No per-voxel significance at all.**  It cannot be done for a block design, and
  two attempts to do it anyway were wrong.  A circular-shift null is invalid outright
  (a P/2 shift negates the design; ``|r|`` is unchanged, so the null draws ARE the
  alternative).  A phase-randomized null is valid in construction but useless in
  practice: the field's own task component sits at the design's frequency, so a
  surrogate captures ``|cos(dphi)|`` of it, and the measured p95 of the surrogate
  equals the true ``|r|`` to three decimals at every noise level tested.  With a
  handful of blocks a single voxel simply does not carry the degrees of freedom.  The
  inference therefore lives ACROSS voxels, in ``contamination_slope`` — which needs no
  null because it predicts a VALUE (kappa = 1), not merely "more than chance".
* **A response-stratified summary.**  A whole-brain median is dominated by tissue with
  no task response — most of an automask — so a real effect confined to responding
  cortex vanishes into it.  Every statistic is reported inside the top decile of the
  data's own task response and outside it.
* **Co-location with the BOLD response.**  Run the same measure on the image series and
  you get the task map of the DATA.  If the field's coupling lands on top of the
  data's, the task is entering the field through the intensity change — contamination.
  If the field is task-locked where the data shows no response (white matter, skull),
  the subject is moving with the task, and no intensity invariance will fix it.  That
  distinction decides which remedy applies, and neither map alone reveals it.

The primitive takes any ``(nx, ny, nz, T)`` map, not just a locomoco field.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field

import numpy as np
import torch

__all__ = [
    "TaskCoupling",
    "default_polort",
    "task_coupling",
    "co_location",
    "contamination_slope",
    "pe_gradient",
    "responding_mask",
    "task_enrichment",
    "map_enrichment",
    "enrichment_curve",
    "project_task_out",
    "design_notch_bins",
    "notch_basis",
    "filter_task_band",
    "component_variance_in_data",
    "format_task_coupling_report",
]


def default_polort(n_timepoints: int, tr: float) -> int:
    """AFNI's drift rule: ``1 + floor(run_seconds / 150)``.

    The same rule ``3dDeconvolve`` uses.  Degree 0 alone is never enough here — a slow
    displacement drift over the run would otherwise be charged to a long block.
    """
    return int(1 + np.floor(max(1.0, n_timepoints * float(tr)) / 150.0))


def _quantile(x: torch.Tensor, q: float) -> float:
    """``torch.quantile`` refuses tensors past ~16M elements; subsample first.

    Same guard as ``locomoco._sd_floor`` — a masked brain at 0.8 mm clears the limit.
    """
    v = x.float().reshape(-1)
    if v.numel() > 1_000_000:
        v = v[:: v.numel() // 1_000_000 + 1]
    return float(torch.quantile(v, q))


def _orthonormal_basis(
    mat: torch.Tensor, scale: float | None = None, tol: float = 1e-6
) -> torch.Tensor:
    """Orthonormal columns spanning ``mat``, rank-deficient columns dropped.

    ``scale`` must be the magnitude of the matrix BEFORE the projection that produced
    ``mat``.  Judging rank against ``mat``'s own leading singular value is the trap: a
    design the drift basis annihilates leaves a residual of pure roundoff, whose
    largest singular value is then declared full rank and handed back as a regressor
    of noise.
    """
    if mat.numel() == 0 or mat.shape[1] == 0:
        return mat.new_zeros((mat.shape[0], 0))
    u, s, _ = torch.linalg.svd(mat, full_matrices=False)
    ref = float(s[0]) if scale is None else scale
    keep = s > tol * ref if ref > 0 else s > 0
    return u[:, keep].contiguous()


def _unit_columns(mat: torch.Tensor) -> torch.Tensor:
    """Each column scaled to unit norm, so a dot product with it IS a correlation."""
    return mat / mat.norm(dim=0, keepdim=True).clamp(min=1e-30)


@dataclass
class TaskCoupling:
    """Per-voxel task coupling of one 4-D map, plus its empirical null."""

    r: torch.Tensor  # (nx,ny,nz,K) SIGNED partial correlation, one per condition
    beta: torch.Tensor  # (nx,ny,nz,K) SIGNED slope, MAP UNITS per unit of regressor
    task_rms: torch.Tensor  # (nx,ny,nz) rms of the task-explained part, MAP UNITS
    total_rms: torch.Tensor  # (nx,ny,nz) rms of the detrended map, MAP UNITS
    valid: torch.Tensor  # (nx,ny,nz) voxels with real detrended variance
    labels: list[str]
    polort: int
    n_timepoints: int = 0
    n_nuisance: int = 0  # extra columns residualized out alongside the drift basis
    summary: dict = dc_field(default_factory=dict)

    def summarize(self, mask: torch.Tensor | None = None) -> dict:
        """Re-summarize the SAME maps over any voxel subset, at no compute cost.

        The reason the per-voxel null maps are kept: a whole-brain median is dominated
        by tissue with no task response, so the report needs these statistics
        restricted to where the data responds.  Refitting per stratum would be
        wasteful and would invite the strata to drift out of sync with the maps
        actually written to disk.
        """
        m = self.valid.reshape(-1)
        if mask is not None:
            m = m & (torch.as_tensor(np.asarray(mask)).reshape(-1).to(m.device) > 0)
        n = int(m.sum())
        if n == 0:
            return {"n_voxels": 0, "conditions": []}
        n_k = self.r.shape[-1]
        r = self.r.reshape(-1, n_k)[m]
        # NOTE: no significance and no sign-agreement statistic. Neither is attainable
        # per voxel for a block design -- see the module docstring. These are
        # DESCRIPTIVE: they say what the map looks like where you are scrubbing it.
        conds = [
            {
                "label": self.labels[k],
                "r_median": float(r[:, k].median()),
                "abs_r_median": float(r[:, k].abs().median()),
                "abs_r_p95": _quantile(r[:, k].abs(), 0.95),
            }
            for k in range(n_k)
        ]
        return {
            "n_voxels": n,
            "conditions": conds,
            "task_rms_median": float(self.task_rms.reshape(-1)[m].median()),
            "total_rms_median": float(self.total_rms.reshape(-1)[m].median()),
        }

    @property
    def chance_share(self) -> float:
        """Task-explained rms share a field with NO task relation would show anyway.

        A K-dimensional projection of a random T-vector keeps ``sqrt(K/df)`` of its
        norm, so the share is never 0 and reading it without this reference overstates
        every result.  It is arithmetic, not a simulation — no surrogate needed, which
        matters here because every surrogate-based null failed (see the module
        docstring).

        A paired reference can land BELOW chance: registering within a task-state bin
        leaves the estimator unable to express a between-bin difference, so the field
        comes out closer to orthogonal to the design than chance would give.
        """
        df = max(1, self.n_timepoints - self.polort - 1 - self.n_nuisance)
        return float(np.sqrt(len(self.labels) / df))

    @property
    def strongest(self) -> dict:
        """The condition with the largest median |r| — what the verdict is judged on."""
        conds = self.summary.get("conditions") or [{}]
        return max(conds, key=lambda c: c.get("abs_r_median", 0.0))


def task_coupling(
    field: torch.Tensor | np.ndarray,
    design: torch.Tensor | np.ndarray,
    *,
    polort: int = 2,
    mask: torch.Tensor | np.ndarray | None = None,
    labels: list[str] | None = None,
    device: torch.device | None = None,
    nuisance: torch.Tensor | np.ndarray | None = None,
) -> TaskCoupling:
    """Signed partial correlation between a 4-D map and each task regressor.

    Parameters
    ----------
    field : (nx, ny, nz, T)
        The map under test, time LAST.  For locomoco this is a signed PE displacement
        in voxels, so ``task_rms`` comes out in voxels.
    design : (T, K)
        Task regressors, already HRF-convolved.
    polort : int
        Legendre drift degree removed from BOTH the map and the design first, so ``r``
        is a partial correlation and slow drift cannot be charged to a long block.
    nuisance : (T, P), optional
        Extra columns removed alongside the drift basis, so ``r`` and ``beta`` become
        partial with respect to those too.  This is how a candidate nuisance set is
        judged: fit the task WITH the regressors in the model and see what happens to
        the response, rather than inferring it from the regressors' properties.
    mask : (nx, ny, nz), optional
        Where the default summary is computed.  The maps are always full-FoV, and
        :meth:`TaskCoupling.summarize` re-summarizes over any other subset for free.
    Notes
    -----
    Each condition's ``r`` is partial with respect to DRIFT only, not to the other
    conditions.  Correlated conditions therefore share credit, which is the honest
    reading for a contamination question ("does the field follow this regressor?")
    rather than an estimation one.

    ``r`` is set to 0 where the detrended map has no variance.  Constant voxels
    scoring a perfect fit is a bug this codebase has shipped twice (the MELODIC/GGM
    parity hunt, the denoisatorial PC selection); it is guarded explicitly.
    """
    from fastfuncstuff.glm.core import construct_polynomial_matrix
    from fastfuncstuff.memory import estimate_chunk_size

    f = torch.as_tensor(np.asarray(field) if isinstance(field, np.ndarray) else field)
    if f.ndim != 4:
        raise ValueError(f"field must be 4-D (nx,ny,nz,T), got shape {tuple(f.shape)}")
    if device is None:
        device = f.device
    x = torch.as_tensor(np.asarray(design), dtype=torch.float64, device=device)
    if x.ndim == 1:
        x = x[:, None]
    n_t, n_k = f.shape[3], x.shape[1]
    if x.shape[0] != n_t:
        raise ValueError(f"design has {x.shape[0]} rows but the map has {n_t} timepoints")
    labels = list(labels) if labels else [f"cond{k + 1}" for k in range(n_k)]

    spatial = tuple(f.shape[:3])
    n_vox = int(np.prod(spatial))

    # float64 for the T-sized factorizations: they are tiny, and the map's own mean can
    # be orders of magnitude above the task effect we are trying to resolve.
    nuis = construct_polynomial_matrix(n_t, polort, device=device, dtype=torch.float64)
    n_extra = 0
    if nuisance is not None:
        extra = torch.as_tensor(np.asarray(nuisance), dtype=torch.float64, device=device)
        if extra.ndim == 1:
            extra = extra[:, None]
        if extra.shape[0] != n_t:
            raise ValueError(f"nuisance has {extra.shape[0]} rows but the map has {n_t}")
        n_extra = extra.shape[1]
        nuis = torch.cat([nuis, extra], dim=1)
    # One basis for drift AND the extra columns: they are projected out together, so a
    # nuisance column that duplicates drift is absorbed rather than counted twice.
    q_n = _orthonormal_basis(nuis)
    x_scale = float(torch.linalg.matrix_norm(x, 2))

    def _detrended_units(cols: torch.Tensor) -> torch.Tensor:
        return _unit_columns(cols - q_n @ (q_n.T @ cols))

    u_x = _detrended_units(x)
    if float(torch.linalg.matrix_norm(x - q_n @ (q_n.T @ x), 2)) < 1e-6 * x_scale:
        raise ValueError(
            f"every task column is collinear with the polort-{polort} drift basis"
            + (f" plus {n_extra} nuisance column(s)" if n_extra else "")
            + ". Lower -task_polort, or the design is not separable from drift at this "
            "block length."
        )
    # Omnibus subspace, only for the task-explained rms (a variance, so sign-free).
    q_x = _orthonormal_basis(x - q_n @ (q_n.T @ x), scale=x_scale)

    dots = torch.zeros(n_vox, n_k, dtype=torch.float64, device=device)
    ss_task = torch.zeros(n_vox, dtype=torch.float64, device=device)
    ss_tot = torch.zeros(n_vox, dtype=torch.float64, device=device)
    ss_raw = torch.zeros(n_vox, dtype=torch.float64, device=device)

    chunk = estimate_chunk_size(n_vox, n_t, n_k + polort + 1, device, operation="glm")
    flat = f.reshape(n_vox, n_t)
    for start in range(0, n_vox, chunk):
        y = flat[start : start + chunk].to(device=device, dtype=torch.float64)
        stop = start + y.shape[0]
        ss_raw[start:stop] = (y * y).sum(dim=1)
        # Residualize against drift explicitly rather than differencing sums of
        # squares: the map carries a large mean and the difference form cancels.
        y = y - (y @ q_n) @ q_n.T
        ss_tot[start:stop] = (y * y).sum(dim=1)
        ss_task[start:stop] = ((y @ q_x) ** 2).sum(dim=1)
        dots[start:stop] = y @ u_x
        del y

    # A constant voxel leaves float64 roundoff behind, not an exact zero, so an
    # `ss_tot > 0` test would still divide two dust piles and report a real-looking
    # correlation. The threshold is relative to the voxel's own magnitude.
    valid = ss_tot > 1e-12 * ss_raw
    norm = ss_tot.clamp(min=1e-30).sqrt()
    zero = torch.zeros_like(ss_tot)
    r = torch.where(valid[:, None], dots / norm[:, None], zero[:, None])
    # dots are against UNIT columns, so beta = dot / ||x_detrended|| puts the slope in
    # map units per unit of regressor -- the quantity the contamination test needs.
    x_norm = (x - q_n @ (q_n.T @ x)).norm(dim=0).clamp(min=1e-30)
    beta = torch.where(valid[:, None], dots / x_norm[None, :], zero[:, None])

    # Zero the rms maps wherever r was zeroed, so all maps agree about which voxels
    # carry no measurement (a 1e-17 rms in a viewer reads as a real number).
    task_rms = torch.where(valid, (ss_task / n_t).sqrt(), zero)
    total_rms = torch.where(valid, (ss_tot / n_t).sqrt(), zero)

    tc = TaskCoupling(
        r=r.reshape(*spatial, n_k),
        beta=beta.reshape(*spatial, n_k),
        task_rms=task_rms.reshape(spatial),
        total_rms=total_rms.reshape(spatial),
        valid=valid.reshape(spatial),
        labels=labels,
        polort=polort,
        n_timepoints=n_t,
        n_nuisance=n_extra,
    )
    tc.summary = tc.summarize(mask)
    if tc.summary["n_voxels"] == 0:
        raise ValueError("mask selects no voxel with temporal variance")
    return tc


def contamination_slope(
    field_beta: torch.Tensor,
    data_beta: torch.Tensor,
    pe_gradient: torch.Tensor,
    mask: torch.Tensor,
    condition: int = 0,
) -> dict:
    """Does the estimated displacement quantitatively ACCOUNT for the intensity change?

    The physical test, and the only one here that survives negative BOLD and an edge.
    A brightness-constancy estimator explains an intensity change ``dI`` at a voxel by
    a displacement ``d`` satisfying the first-order relation

        g * d = dI ,      g = dS/d(PE)   (the local gradient along the encode axis)

    Take both sides' response to the task: if the estimator is absorbing the BOLD
    response as motion, then ``g * beta_field`` should EQUAL ``beta_data`` voxel by
    voxel.  Regressing one on the other across voxels gives a slope ``kappa`` — near 1
    means the displacement is exactly accounting for the intensity change; near 0
    means the field's task response has nothing to do with the BOLD response, so it is
    something else (real task-correlated motion, or nothing).

    Why this and not a correlation of signs:

    * **Negative BOLD is real.**  A deactivating voxel has ``beta_data < 0``, which
      flips the predicted displacement too.  The ratio is unchanged, so deactivation
      is scored as contamination exactly like activation — as it should be.
    * **An edge flips the gradient.**  ``g`` reverses across a boundary, so the
      contaminated displacement reverses with it.  Again the ratio is unchanged, while
      any statistic built on the field's sign alone would call this incoherent.

    Written multiplicatively (``beta_data ~ kappa * g * beta_field``) rather than as
    ``d ~ dI/g``, so voxels with no gradient contribute nothing instead of exploding.
    That is the same ``|grad|^2`` weighting the Lucas-Kanade normal equations already use.

    Returns ``kappa``, the through-origin ``r2``, the centred Pearson ``r``, and the
    voxel count.
    """
    m = mask.reshape(-1) > 0
    pred = (pe_gradient * field_beta[..., condition]).reshape(-1)[m].double()
    obs = data_beta[..., condition].reshape(-1)[m].double()
    sxx, sxy, syy = pred.dot(pred), pred.dot(obs), obs.dot(obs)
    if float(sxx) <= 0 or float(syy) <= 0:
        return {"kappa": 0.0, "r2": 0.0, "r": 0.0, "n": int(m.sum())}
    a, b = pred - pred.mean(), obs - obs.mean()
    denom = float(a.norm() * b.norm())
    return {
        "kappa": float(sxy / sxx),
        "r2": float(sxy * sxy / (sxx * syy)),
        "r": float(a.dot(b) / denom) if denom > 0 else 0.0,
        "n": int(m.sum()),
    }


def pe_gradient(reference: torch.Tensor, axis: int) -> torch.Tensor:
    """Central-difference gradient of a 3-D reference along ``axis``, per voxel.

    The scale that converts a displacement into the intensity change it would cause,
    which is what :func:`contamination_slope` regresses through.
    """
    return torch.gradient(reference, dim=axis)[0]


def co_location(field_r: torch.Tensor, data_r: torch.Tensor, mask: torch.Tensor) -> float:
    """Spatial correlation of |r| maps — the contamination-vs-motion test.

    High means the field is task-locked exactly where the data responds, i.e. the
    response is entering the estimator as intensity.  Near zero with a task-locked
    field means the coupling lives where the BOLD response does not, which is
    task-correlated head motion and a different problem entirely.

    Magnitudes, not signed values: the field's sign is a direction of displacement and
    the data's is a direction of signal change, so their signed product means nothing.
    Where the two land is the question.
    """
    m = mask.reshape(-1) > 0
    a = field_r.abs().amax(dim=-1).reshape(-1)[m].double()
    b = data_r.abs().amax(dim=-1).reshape(-1)[m].double()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return float(a.dot(b) / denom) if float(denom) > 0 else 0.0


def project_task_out(
    field: torch.Tensor,
    design: torch.Tensor | np.ndarray,
    polort: int = 2,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove the task-locked part of a 4-D field, KEEPING drift and everything else.

    Returns ``(cleaned, removed)``, both ``(nx,ny,nz,T)``.

    The polynomial question, answered explicitly because it is easy to get backwards:
    the drift basis goes into the FIT but not into the SUBTRACTION.  Fitting task and
    drift jointly is what keeps a slow drift from biasing a long block regressor's
    beta.  Subtracting only ``X_task @ beta_task`` is what keeps the drift itself in
    the field — a slowly drifting displacement is REAL residual motion, and locomoco's
    own warp PCs routinely show a polynomial-like component that is exactly that.
    Removing it would be discarding the correction, not cleaning it.

    The intended consumer is the PC set.  A task-correlated nuisance regressor is
    worse than a slightly wrong one: dropped into a GLM it removes real BOLD along
    with the motion.
    """
    from fastfuncstuff.glm.core import construct_polynomial_matrix

    f = torch.as_tensor(field)
    if device is None:
        device = f.device
    n_t = f.shape[3]
    x = torch.as_tensor(np.asarray(design), dtype=torch.float64, device=device)
    if x.ndim == 1:
        x = x[:, None]

    q_n = _orthonormal_basis(
        construct_polynomial_matrix(n_t, polort, device=device, dtype=torch.float64)
    )
    # Orthogonalize the task against drift, so the joint fit reduces to two
    # independent projections and the task part carries no drift with it.
    q_x = _orthonormal_basis(x - q_n @ (q_n.T @ x), scale=float(torch.linalg.matrix_norm(x, 2)))
    flat = f.reshape(-1, n_t).to(dtype=torch.float64, device=device)
    removed = (flat @ q_x) @ q_x.T
    cleaned = flat - removed
    shape = tuple(f.shape)
    return (
        cleaned.reshape(shape).to(f.dtype),
        removed.reshape(shape).to(f.dtype),
    )


def design_notch_bins(
    design: torch.Tensor | np.ndarray,
    polort: int = 2,
    *,
    peak_frac: float = 0.01,
    widen: int = 0,
    max_frac: float = 0.50,
    warn_frac: float = 0.15,
) -> tuple[list[int], dict]:
    """Which rFFT bins carry the design — the band a notch has to remove.

    Selection is CONTRAST AGAINST THE FLOOR, not cumulative power, and the difference
    is not cosmetic. Measured on a 15-cycle 20 s block design (120 frames, TR 2.5):

        bin 15 = 0.0500 Hz : 91.13% of the design's power   (191x the median bin)
        bin 45 = 0.1500 Hz :  0.49%                         (1.04x -- leakage floor)
        ...a flat floor of ~0.47% per bin...

        90% of design power -> 1 bin   (1.6% of the spectrum,  2 DoF)
        95% of design power -> 10 bins (16.4%,                20 DoF)
        99% of design power -> 23 bins (37.7%,                46 DoF)

    A cumulative threshold is a cliff: past the fundamental you are buying bins off a
    flat leakage floor, so 90% is free and 95% costs ten times as much for nothing
    real. Keeping every bin that carries at least ``peak_frac`` of the STRONGEST
    line picks the one line here, and would keep a genuine harmonic at 5% of the
    fundamental. Contrast against the median bin was tried first and is not
    equivalent: this design's median bin sits 2048x below its fundamental, so a
    10x-the-median cut lands inside the low-frequency shoulder that drift removal
    leaves behind and selects nine bins instead of one.

    ``widen`` adds bins either side. The reason is amplitude NON-STATIONARITY across
    blocks -- adaptation or attention drift modulates block amplitude and puts
    sidebands at +/- 1/tau for an envelope timescale tau. It is NOT the HRF: convolution
    multiplies spectra, so a wider HRF makes the design's spectrum narrower, never
    wider, and cannot spread energy into bins the stimulus does not occupy.

    Raises if the selection exceeds ``max_frac`` of the spectrum: a jittered
    event-related design is broadband, and notching it would remove the data rather
    than the task.
    """
    from fastfuncstuff.glm.core import construct_polynomial_matrix

    x = torch.as_tensor(np.asarray(design), dtype=torch.float64)
    if x.ndim == 1:
        x = x[:, None]
    n_t = x.shape[0]
    q_n = _orthonormal_basis(
        construct_polynomial_matrix(n_t, polort, device=x.device, dtype=torch.float64)
    )
    xd = x - q_n @ (q_n.T @ x)
    # Pooled across conditions: the notch is one band for the whole design, since the
    # estimator sees the sum of every response at once.
    power = (torch.fft.rfft(xd, dim=0).abs() ** 2).sum(dim=1)
    n_bins = power.numel()
    # Bin 0 is the mean, which polort already owns; including it would notch the drift.
    body = power[1:]
    peak = float(body.max()) if body.numel() else 0.0
    # Relative to the PEAK line, not to the median bin. A median-contrast threshold
    # looked equivalent and is not: this design's median bin sits 2048x below its
    # fundamental, so "10x the median" lands inside the low-frequency shoulder the
    # drift-residualized boxcar leaves behind and selects nine bins instead of one.
    # A peak-relative cut is scale-free and says something a reader can check --
    # "every line carrying at least 1% of the strongest one".
    keep = {int(b) + 1 for b in torch.nonzero(body > peak_frac * peak).flatten().tolist()}
    if not keep:
        raise ValueError(
            "no frequency bin carries the design above the leakage floor. The design is "
            "either flat after drift removal or too broadband to notch."
        )
    for b in list(keep):
        for d in range(1, int(widen) + 1):
            if 1 <= b - d < n_bins:
                keep.add(b - d)
            if 1 <= b + d < n_bins:
                keep.add(b + d)
    bins = sorted(keep)
    frac = len(bins) / max(1, n_bins - 1)
    if frac > max_frac:
        raise ValueError(
            f"the design occupies {len(bins)} of {n_bins - 1} frequency bins "
            f"({frac:.0%} of the spectrum, over the {max_frac:.0%} limit). This is a "
            "broadband design -- a notch would remove the data, not the task. Use "
            "-detask (project the design out of the field) instead."
        )
    # Between warn_frac and max_frac the notch is expensive but not obviously wrong.
    # Bulk head motion is spectrally BROAD, so removing a sixth of the spectrum can
    # still leave it estimable -- what breaks is not knowable from the design alone,
    # only from whether the resulting field still tracks the motion. Hence a loud
    # warning and a permissive ceiling rather than a hard cut at the cheap end.
    warning = None
    if frac > warn_frac:
        warning = (
            f"the notch removes {frac:.0%} of the spectrum ({len(bins)} of "
            f"{n_bins - 1} bins, {2 * len(bins)} DoF). Bulk motion is spectrally broad "
            "so this may still be fine, but CHECK that the field still tracks real "
            "motion -- compare the flow rms and the warp PCs against an unfiltered run."
        )
    info = {
        "bins": bins,
        "n_bins": n_bins - 1,
        "spectrum_frac": frac,
        "peak_frac": peak_frac,
        "warning": warning,
        "widen": int(widen),
        "peak_over_median": float(peak / body.median())
        if float(body.median()) > 0
        else float("inf"),
    }
    return bins, info


def notch_basis(
    n_t: int,
    bins: list[int],
    polort: int = 2,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Orthonormal time-domain basis spanning ``bins``, with drift already removed.

    One cos/sin pair per bin (a single column for DC and Nyquist), orthogonalized
    against the drift basis so removing it cannot take the drift with it — the same
    split :func:`project_task_out` makes, and for the same reason: a slow drift is real
    residual motion, not something the task filter should be eating.
    """
    from fastfuncstuff.glm.core import construct_polynomial_matrix

    t = torch.arange(n_t, dtype=torch.float64, device=device)
    cols = []
    for b in bins:
        w = 2.0 * np.pi * b / n_t
        cols.append(torch.cos(w * t))
        # At Nyquist on an even-length series the sine column is identically zero;
        # _orthonormal_basis would otherwise hand back a normalised pile of roundoff.
        if not (n_t % 2 == 0 and 2 * b == n_t):
            cols.append(torch.sin(w * t))
    mat = torch.stack(cols, dim=1)
    q_n = _orthonormal_basis(
        construct_polynomial_matrix(n_t, polort, device=device, dtype=torch.float64)
    )
    resid = mat - q_n @ (q_n.T @ mat)
    return _orthonormal_basis(resid, scale=float(torch.linalg.matrix_norm(mat, 2)))


def design_fit_basis(
    design: torch.Tensor | np.ndarray,
    polort: int = 2,
    n_deriv: int = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Orthonormal basis spanning the task DESIGN itself, with drift already removed.

    The design-space counterpart to :func:`notch_basis`, and the third way to keep the
    task out of the estimator. The notch works in the frequency domain and costs two
    degrees of freedom per line it removes; this costs one per regressor. On a real
    5-condition 18 s block design (TR 3.5, 225 frames) the notch took 43 bins = 86 DoF
    where the design spans 5, so for anything but a tightly periodic paradigm this is
    the far cheaper cut -- and unlike the notch it has no periodicity requirement at
    all, which is what makes it the only estimator-side option for a broadband design.

    What it cannot do, on the record so this is not rediscovered. Projecting the design
    out removes the part of the response the CANONICAL shape explains and leaves the
    rest: latency and width mismatch put real task variance in the residual, and the
    estimator still sees it. ``n_deriv`` widens the subspace with successive time
    derivatives of each regressor (1 = the temporal derivative, absorbing a few hundred
    ms of latency; 2 adds curvature) at one more DoF per regressor per derivative,
    which is the standard trade and the reason the knob exists.

    The deeper circularity is not fixable here and is worth stating plainly: a
    contamination severe enough to matter also distorts the very fit being projected
    out, so the residual keeps a share of the artifact proportional to how bad the
    problem was. This is a mitigation, not a proof of removal -- read the enrichment
    diagnostic afterwards rather than assuming the cut worked.

    Derivatives are finite differences of the CONVOLVED regressor, which is the
    discrete form of convolving with the HRF's derivative -- no second design build,
    and it stays correct for any basis the design was built from.
    """
    from fastfuncstuff.glm.core import construct_polynomial_matrix

    x = torch.as_tensor(np.asarray(design), dtype=torch.float64, device=device)
    if x.ndim == 1:
        x = x[:, None]
    n_t = x.shape[0]
    if int(n_deriv) < 0:
        raise ValueError(f"n_deriv must be >= 0, got {n_deriv}")
    cols = [x]
    cur = x
    for _ in range(int(n_deriv)):
        # Central difference, edges held: a forward difference would shift the
        # derivative half a frame relative to the regressor it is meant to accompany.
        d = torch.zeros_like(cur)
        d[1:-1] = 0.5 * (cur[2:] - cur[:-2])
        d[0], d[-1] = cur[1] - cur[0], cur[-1] - cur[-2]
        cols.append(d)
        cur = d
    mat = torch.cat(cols, dim=1)
    q_n = _orthonormal_basis(
        construct_polynomial_matrix(n_t, polort, device=device, dtype=torch.float64)
    )
    resid = mat - q_n @ (q_n.T @ mat)
    scale = float(torch.linalg.matrix_norm(mat, 2))
    basis = _orthonormal_basis(resid, scale=scale)
    if basis.shape[1] == 0:
        raise ValueError(
            f"the design is entirely collinear with the polort-{polort} drift basis: "
            "there is nothing to project out. Lower -task_polort."
        )
    return basis


def filter_task_band(
    data: torch.Tensor,
    basis: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Remove ``basis`` from every voxel's time course of a ``(nx,ny,nz,T)`` series.

    Applied to the images the estimator SEES, not to the output. That is the whole
    point of doing it here rather than with ``-detask``: the field is never
    contaminated, so nothing can bleed through an iterative solve into other frames or
    voxels, and the corrected series is still resampled from the raw input.
    """
    from fastfuncstuff.memory import estimate_chunk_size

    f = torch.as_tensor(data)
    if f.ndim != 4:
        raise ValueError(f"data must be 4-D (nx,ny,nz,T), got {tuple(f.shape)}")
    n_t = f.shape[3]
    if basis.shape[0] != n_t:
        raise ValueError(f"basis has {basis.shape[0]} rows but the series has {n_t} frames")
    if device is None:
        device = f.device
    q = basis.to(device=device, dtype=torch.float64)
    spatial = tuple(f.shape[:3])
    n_vox = int(np.prod(spatial))
    flat = f.reshape(n_vox, n_t)
    out = torch.empty_like(flat)
    chunk = estimate_chunk_size(n_vox, n_t, q.shape[1] + 2, device, operation="glm")
    for start in range(0, n_vox, chunk):
        y = flat[start : start + chunk].to(device=device, dtype=torch.float64)
        y = y - (y @ q) @ q.T
        out[start : start + y.shape[0]] = y.to(dtype=out.dtype, device=out.device)
        del y
    return out.reshape(*spatial, n_t)


def component_variance_in_data(
    scores: torch.Tensor,
    data: torch.Tensor | np.ndarray,
    *,
    polort: int = 2,
    design: torch.Tensor | np.ndarray | None = None,
    mask: torch.Tensor | np.ndarray | None = None,
    device: torch.device | None = None,
) -> dict:
    """What share of the DATA each component's time course explains — and of the task.

    A component's own variance ratio says how much of the WARP it is. That is not the
    question a nuisance regressor raises. Two different questions matter and neither is
    answered by the warp spectrum:

    * ``var_data`` — the share of the data's drift-residualized variance this component
      explains. A component can be 50% of the warp and explain a tenth of a percent of
      the images, which is the normal and harmless case; the warp is small.
    * ``var_task`` — the share of the data's TASK-EXPLAINED variance it explains. This
      is the one that decides whether a regressor is safe: a nuisance column that eats
      task variance removes real BOLD in the GLM it is added to.
    * ``task_frac`` — the share of the component's OWN time course lying in the task
      subspace, i.e. the collinearity with the design, independent of the data. This is
      the SAME statistic :func:`component_task_fit` thresholds (its omnibus ``r2``),
      reported here on an interpretable scale and without the surrogate null; it is not
      a second, independent piece of evidence.

    ``joint_var_data`` / ``joint_var_task`` are the figures for all components TOGETHER
    -- projections onto the component SPAN, not sums. They coincide with the sum of the
    marginals only when the drift-residualized components are orthogonal; otherwise
    correlated components can suppress as well as reinforce, so the sum can land on
    either side of the joint and is not a meaningful quantity. Compare the joint against
    its OWN floor, ``k/(T-polort-1)``, which is k times the per-component one: k
    directions capture that share of any fixed direction by construction.

    Components are orthogonalized against the drift basis and re-orthonormalized first,
    so ``var_data`` is a partial share w.r.t. the polynomials the GLM will carry anyway,
    and the per-component shares are additive rather than double-counting a common
    trend.

    Cheap by construction: nothing four-dimensional is ever formed. The task-projected
    series is never materialized — ``u_k' (Q_x Q_x' y) = (Q_x' u_k)' (Q_x' y)`` — so one
    chunked pass over the data with two small matmuls gives every column.

    ``data`` is ``(nx, ny, nz, T)``, ideally the CORRECTED series: the regressors will
    be used on the images that come out of this tool, not the ones that went in. That
    also makes ``var_data`` the RESIDUAL motion-linked variance -- the correction has
    already removed what it could, so what these components still explain is what a
    regressor could still take out. Scored on the raw series it would read higher and
    mean something different.

    Note the ``var_task``/``task_frac`` pair collapses when the design has ONE column:
    the task subspace is then a single direction, so the share of the data's task
    variance a component takes is exactly the share of the component lying in that
    direction. They only carry separate information with multiple conditions, where the
    data decides which task directions matter.

    Compare against chance, which is not zero: one component orthonormalized into a
    ``T - polort - 1`` dimensional residual space takes ``1/(T-polort-1)`` of the
    variance and ``K/(T-polort-1)`` of the task subspace by construction.
    """
    from fastfuncstuff.glm.core import construct_polynomial_matrix
    from fastfuncstuff.memory import estimate_chunk_size

    f = torch.as_tensor(np.asarray(data) if isinstance(data, np.ndarray) else data)
    if f.ndim != 4:
        raise ValueError(f"data must be 4-D (nx,ny,nz,T), got {tuple(f.shape)}")
    u = torch.as_tensor(scores, dtype=torch.float64)
    if u.ndim != 2 or u.shape[0] != f.shape[3]:
        raise ValueError(f"scores must be (T, k) with T={f.shape[3]}, got {tuple(u.shape)}")
    if device is None:
        device = f.device
    n_t, k = u.shape
    u = u.to(device)

    q_n = _orthonormal_basis(
        construct_polynomial_matrix(n_t, polort, device=device, dtype=torch.float64)
    )
    # Partial w.r.t. drift, on BOTH sides: the GLM these enter carries polynomials, so
    # the slow variance they share with drift is not the component's to claim.
    #
    # Each component is normalized ON ITS OWN, never orthogonalized against the others.
    # An earlier version ran the whole set through _orthonormal_basis, which returns the
    # SVD directions of the set -- an orthogonal ROTATION of the component space. Row k
    # was then labelled "component k" while describing a direction that was a mixture of
    # all of them. Harmless for PCA (already orthonormal, barely rotated), wrong for ICA,
    # whose mixing is not orthogonal at all. The price of doing it per component is that
    # the shares OVERLAP where components correlate and so do not sum to the joint
    # figure, which is returned separately as ``joint_var_data``.
    u_d = u - q_n @ (q_n.T @ u)
    u_d = u_d / u_d.norm(dim=0, keepdim=True).clamp(min=1e-30)
    # The joint span, for the "all of them together" figure only.
    u_j = _orthonormal_basis(u_d)

    q_x = None
    if design is not None:
        x = torch.as_tensor(np.asarray(design), dtype=torch.float64, device=device)
        if x.ndim == 1:
            x = x[:, None]
        if x.shape[0] != n_t:
            raise ValueError(f"design has {x.shape[0]} rows but the data has {n_t}")
        q_x = _orthonormal_basis(x - q_n @ (q_n.T @ x), scale=float(torch.linalg.matrix_norm(x, 2)))

    spatial = tuple(f.shape[:3])
    n_vox = int(np.prod(spatial))
    keep = None
    if mask is not None:
        keep = torch.as_tensor(np.asarray(mask)).reshape(-1) > 0

    ss_pc = torch.zeros(k, dtype=torch.float64, device=device)
    ss_joint = torch.zeros((), dtype=torch.float64, device=device)
    ss_task_pc = torch.zeros(k, dtype=torch.float64, device=device)
    ss_task_joint = torch.zeros((), dtype=torch.float64, device=device)
    ss_tot = torch.zeros((), dtype=torch.float64, device=device)
    ss_task = torch.zeros((), dtype=torch.float64, device=device)
    a = None if q_x is None else q_x.T @ u_d  # (Kx, k)
    a_j = None if q_x is None else q_x.T @ u_j

    flat = f.reshape(n_vox, n_t)
    chunk = estimate_chunk_size(n_vox, n_t, k + polort + 2, device, operation="glm")
    for start in range(0, n_vox, chunk):
        stop = min(start + chunk, n_vox)
        if keep is not None:
            sel = keep[start:stop]
            if not bool(sel.any()):
                continue
            y = flat[start:stop][sel].to(device=device, dtype=torch.float64).T  # (T, v)
        else:
            y = flat[start:stop].to(device=device, dtype=torch.float64).T
        y = y - q_n @ (q_n.T @ y)
        ss_tot += (y * y).sum()
        ss_pc += (u_d.T @ y).pow(2).sum(dim=1)
        ss_joint += (u_j.T @ y).pow(2).sum()
        if q_x is not None and a is not None and a_j is not None:
            b = q_x.T @ y  # (Kx, v)
            ss_task += (b * b).sum()
            ss_task_pc += (a.T @ b).pow(2).sum(dim=1)
            ss_task_joint += (a_j.T @ b).pow(2).sum()
        del y

    tot = float(ss_tot.clamp(min=1e-30))
    out = {
        "var_data": (ss_pc / tot).cpu(),
        "joint_var_data": float(ss_joint / tot),
        "total_var": tot,
        "task_frac": None if a is None else (a * a).sum(dim=0).cpu(),
        "var_task": None,
        "joint_var_task": None,
        "total_task_var": None,
    }
    if q_x is not None:
        tt = float(ss_task.clamp(min=1e-30))
        out["var_task"] = (ss_task_pc / tt).cpu()
        out["joint_var_task"] = float(ss_task_joint / tt)
        out["total_task_var"] = tt
    return out


def component_task_fit(
    mixing: torch.Tensor,
    design: torch.Tensor | np.ndarray,
    *,
    polort: int = 2,
    n_surrogates: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """How well the task explains each component's TIME COURSE, against its own null.

    The temporal counterpart to :func:`map_enrichment`, and the criterion that has
    power exactly where the spatial one does not: a component that is genuinely
    task-locked but activates a handful of voxels scores ~1.0x on an energy share,
    because its energy is dominated by wherever else it lives.

    Why this is allowed here when the module docstring says a temporal correlation
    cannot be thresholded. That verdict was measured on a 20 s periodic BLOCK design
    and is true of it -- a component merely sharing that design's spectrum correlates
    ~0.92 with it by chance, roughly 2 effective degrees of freedom. It is not a
    general property. Measured with 2000 phase-randomised surrogates:

        design                    null median   p95     eff. DoF
        block, 20 s periodic          0.640     0.924      ~2
        jittered event-related        0.102     0.284     ~46
        5-cond 18 s blocks, 20 s SOA  0.12      0.34      ~31-38   (per condition)

    The third row is a real acquisition (OHBMPilot04) and is the reason this exists:
    its pooled on/off is periodic, but the scorer works per condition and each
    condition's onsets are irregular, so the statistic has ~35 DoF rather than 2. The
    only honest way to know which regime a run is in is to measure it, which is what
    the surrogates here do -- there is no need to classify the design by eye.

    The statistic is the OMNIBUS R^2 on the whole design, not a per-condition
    correlation. Conditions activate different tissue and the strongest one is not
    knowable in advance, so a per-condition maximum would need a multiplicity
    correction over K on top of the one over components; the joint fit needs neither
    and is the quantity "does the task explain this time course" actually asks about.

    Null and correction. Each component is phase-randomised against ITS OWN amplitude
    spectrum -- the right null direction, since a component's autocorrelation is what
    makes a spurious fit possible, and randomising the design instead would test a
    different hypothesis. Per-component R^2 is standardised to a z against that
    component's surrogates, then the threshold is the ``1 - alpha`` quantile of the
    MAX z across components per surrogate draw: familywise control, which matters
    because a 60-component decomposition tested at a nominal 0.05 flags three by
    chance. The max-statistic null treats components as independent, which ICA makes
    approximately true and PCA exactly true in the temporal basis.

    Returns a dict with per-component ``r2`` / ``z`` / ``p`` arrays, the ``z_cut``
    actually applied, and ``flagged`` (indices, most-fit first).
    """
    from fastfuncstuff.glm.core import construct_polynomial_matrix

    m = torch.as_tensor(mixing, dtype=torch.float64)
    if m.ndim != 2:
        raise ValueError(f"mixing must be (T, k), got {tuple(m.shape)}")
    n_t, k = m.shape
    x = torch.as_tensor(np.asarray(design), dtype=torch.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] != n_t:
        raise ValueError(f"design has {x.shape[0]} rows but the components have {n_t}")

    q_n = _orthonormal_basis(
        construct_polynomial_matrix(n_t, polort, dtype=torch.float64, device=m.device)
    )
    q_x = _orthonormal_basis(x - q_n @ (q_n.T @ x), scale=float(torch.linalg.matrix_norm(x, 2)))
    if q_x.shape[1] == 0:
        raise ValueError(
            f"the design is entirely collinear with the polort-{polort} drift basis; "
            "no temporal criterion is available. Lower -task_polort."
        )

    def _r2(y: torch.Tensor) -> torch.Tensor:
        """Omnibus R^2 of each column of ``y`` (T, n) on the task subspace."""
        yd = y - q_n @ (q_n.T @ y)
        ss_tot = (yd * yd).sum(dim=0)
        ss_fit = ((q_x.T @ yd) ** 2).sum(dim=0)
        return ss_fit / ss_tot.clamp(min=1e-30)

    r2 = _r2(m)

    # Phase randomisation, all components at once: same amplitude spectrum, random
    # phases. Bin 0 (and Nyquist on an even series) must keep phase 0 or the surrogate
    # is complex -- irfft would silently discard the imaginary part otherwise.
    g = torch.Generator().manual_seed(int(seed))
    f = torch.fft.rfft(m - m.mean(dim=0, keepdim=True), dim=0)  # (F, k)
    amp = f.abs()
    n_f = amp.shape[0]
    null = torch.empty(n_surrogates, k, dtype=torch.float64)
    # Chunked over surrogates: (S, F, k) complex is the only large intermediate here.
    step = max(1, min(n_surrogates, int(4e6 // max(1, n_f * k))))
    for start in range(0, n_surrogates, step):
        n_s = min(step, n_surrogates - start)
        ph = torch.rand(n_s, n_f, k, generator=g, dtype=torch.float64) * (2 * np.pi)
        ph[:, 0, :] = 0.0
        if n_t % 2 == 0:
            ph[:, -1, :] = 0.0
        surr = torch.fft.irfft(amp[None] * torch.exp(1j * ph), n=n_t, dim=1)  # (S, T, k)
        for i in range(n_s):
            null[start + i] = _r2(surr[i])
        del ph, surr

    mu, sd = null.mean(dim=0), null.std(dim=0).clamp(min=1e-12)
    z = (r2 - mu) / sd
    # Effective degrees of freedom, read off the null rather than assumed. For an
    # omnibus R^2 on K regressors, E[R^2] under the null is about K/df, so df is about
    # K/mean(null). This is the number that says which REGIME the design is in -- ~2
    # for a periodic block design, ~30-45 for a jittered or irregular one -- and it is
    # measured per run, so no design ever has to be classified by eye.
    null_mean = float(mu.mean())
    eff_dof = float(q_x.shape[1] / null_mean) if null_mean > 0 else float("inf")
    # Empirical p with the +1 correction: a surrogate set can never license p = 0.
    p = (null >= r2[None, :]).sum(dim=0).double().add(1.0) / (n_surrogates + 1)
    max_z = ((null - mu[None, :]) / sd[None, :]).amax(dim=1)
    z_cut = float(torch.quantile(max_z, 1.0 - float(alpha)))
    flagged = [i for i in torch.argsort(z, descending=True).tolist() if float(z[i]) > z_cut]

    # The R^2 a component would need to be flagged at all. When the null already sits
    # near 1 -- a periodic block design, where a component sharing the design's
    # spectrum fits it by construction -- this cut is unreachable and "nothing flagged"
    # means "no power", not "clean". Reporting the two as the same silence is the trap
    # this field spent a long time in; the caller is told which it is and falls back to
    # the spatial criterion, which needs no null.
    r2_needed = float((mu + z_cut * sd).median())
    # Two ways to have no power, and both must gate. The cut being unreachable is one.
    # The other is a low effective DoF: a component whose OWN spectrum coincides with
    # the design's can fit it by accident however high the cut is set, and the
    # phase-randomised surrogates of a narrowband component are all near-equally good
    # fits, so the max-z correction (which assumes components are independent) is
    # anti-conservative exactly there. Measured: broadband components against a
    # periodic block design give ~195 effective DoF and are safe, while NARROWBAND
    # components at that design's own frequency give ~5 and flag random-phase
    # sinusoids that have no task relation at all. 10 separates the two regimes with
    # room to spare -- the real designs measured sit at 30-85.
    informative = r2_needed < 0.9 and eff_dof >= 10.0
    return {
        "r2": r2,
        "z": z,
        "p": p,
        "z_cut": z_cut,
        "alpha": float(alpha),
        "n_surrogates": int(n_surrogates),
        "null_r2_median": mu,
        "null_r2_mean": null_mean,
        "eff_dof": eff_dof,
        "r2_needed": r2_needed,
        "informative": informative,
        "uninformative_reason": (
            None
            if informative
            else (
                f"the null already reaches R²={r2_needed:.2f}"
                if r2_needed >= 0.9
                else (
                    f"only {eff_dof:.0f} effective DoF — these components share the "
                    "design's own frequency band, so a fit proves nothing"
                )
            )
        ),
        "flagged": flagged if informative else [],
    }


def map_enrichment(
    values: torch.Tensor,
    active: torch.Tensor,
    mask: torch.Tensor,
) -> dict:
    """Share of a map's ENERGY inside ``active``, over the share of voxels it occupies.

    1.0 means the map is spread like the brain, i.e. unrelated to where the task is.
    Well above 1.0 means it is concentrated on activated tissue. No null is needed --
    the no-relation value is 1 by construction, not by simulation -- which is why this
    is the right scorer for a question with ~2 degrees of freedom in time.

    Used for two different maps: a field's task-explained rms (:func:`task_enrichment`)
    and a warp PC's spatial loading. The second is the one that matters for rejection:
    a component's correlation with the design is a 2-DoF statistic and cannot be
    thresholded, but WHERE its weights live can.

    What this is NOT, so the limits are on the record. The active mask is BINARY, so
    every voxel in the decile counts the same and the graded ``r`` values are discarded.
    It measures concentration, not SHAPE -- a component whose energy sits inside the
    mask arranged nothing like the response scores the same as one that traces it. And
    the denominator is the whole brain, so large loadings anywhere else dilute the
    score, which is what every real motion component has.

    :func:`co_location` is the shape-matching alternative (Pearson between ``|field r|``
    and ``|data r|``), and it was measured against this one rather than assumed worse.
    On the 0.8mm checkerboard run neither finds anything, because the information is
    not in the components:

        comp 0 (31.7% var):  enrichment 1.18x,  corr(|loading|, |r|_data) +0.068
        ...no component exceeds 1.25x or |corr| 0.07...
        the FIELD's own task-r map:            corr(|r|_field, |r|_data) +0.386

    The field is plainly co-located with the response; its principal components are
    not. That gap is the supervised/unsupervised one -- isolating a signal worth 0.7%
    of the field's variance needs a projection that knows what it is looking for, which
    is :func:`project_task_out`, not a variance decomposition. Do not re-try this by
    swapping the scorer.
    """
    m = mask.reshape(-1) > 0
    a = (active.reshape(-1) > 0) & m
    energy = (values.reshape(-1)[m].double()) ** 2
    inside = (values.reshape(-1)[a].double()) ** 2
    total = float(energy.sum())
    vox_share = float(a.sum()) / max(1, int(m.sum()))
    e_share = float(inside.sum()) / total if total > 0 else 0.0
    return {
        "n_active": int(a.sum()),
        "voxel_share": vox_share,
        "energy_share": e_share,
        "enrichment": e_share / vox_share if vox_share > 0 else 0.0,
    }


def task_enrichment(
    field: TaskCoupling,
    active: torch.Tensor,
    mask: torch.Tensor,
) -> dict:
    """How much of the field's task-locked displacement sits inside the active mask?

    The most direct form of the question, and the one to read first: threshold the
    DATA's task map to get "where the task is", then ask what share of the field's
    task-coupled energy falls in there — against the share of voxels it occupies.

        enrichment = (share of task-locked displacement inside) / (share of voxels inside)

    1.0 means the field's task coupling is spread like the brain, i.e. unrelated to
    where the task actually is.  Well above 1.0 means it is concentrated on activated
    tissue, which is what BOLD-driven displacement looks like.  It needs no null: the
    no-relation value is 1 by construction, not by simulation.
    """
    return map_enrichment(field.task_rms, active, mask)


def enrichment_curve(
    field: TaskCoupling,
    active: torch.Tensor,
    mask: torch.Tensor,
    quantiles: tuple[float, ...] = (0.5, 0.9, 0.99, 0.999),
) -> list[dict]:
    """Enrichment on active tissue as a function of how task-locked the FIELD is.

    The statistic that actually detects contamination, and the reason the first
    version of this module reported "nothing to fix" on a run that was visibly
    contaminated.  Every other summary here conditions on the DATA — take the decile
    where the response lives, then average the field over it.  That is the wrong way
    round.  Contamination obeys ``g * d = dI``, so it can only exist where the encode
    gradient is non-zero; the active decile is mostly flat voxel interiors where
    ``g ~ 0`` and no displacement is induced at all.  A median over it is a median
    over the voxels that *cannot* be contaminated, and it reads clean no matter how
    bad the tail is.

    Condition on the FIELD instead: take the voxels where the field is most
    task-locked and ask what share of THEM sit on activated tissue.  Measured on a
    contaminated 0.8 mm checkerboard run, where the median-based verdict said
    "no appreciable coupling":

        |r_field| > 0.2  ->  3.8x      |r_field| > 0.5  ->  7.4x
        |r_field| > 0.3  ->  5.6x      |r_field| > 0.7  ->  8.9x
        |r_field| > 0.4  ->  6.6x      |r_field| > 0.8  ->  9.6x

    against a ceiling of 10x (the active mask is a decile).  A field whose task
    coupling is unrelated to where the task is sits at 1.0 in every row; a field
    reading BOLD as motion climbs toward the ceiling.  It needs no null — the
    no-relation value is 1 by construction — and no per-voxel significance, which
    is unattainable for a block design anyway.

    Returns one row per quantile with enough voxels to mean anything, each carrying
    the ``|r|`` cut it corresponds to, the voxel count, and the enrichment.
    """
    m = mask.reshape(-1) > 0
    a = (active.reshape(-1) > 0)[m]
    strength = field.r.abs().amax(dim=-1).reshape(-1)[m]
    vox_share = float(a.double().mean())
    if vox_share <= 0:
        return []
    rows: list[dict] = []
    for q in quantiles:
        cut = _quantile(strength, q)
        sel = strength > cut
        n = int(sel.sum())
        # Below ~200 voxels the share is quantized coarsely enough that a single blob
        # sets it; reporting that as an enrichment invites reading noise as a trend.
        if n < 200:
            continue
        frac = float((a & sel).sum()) / n
        rows.append(
            {
                "quantile": q,
                "r_cut": cut,
                "n": n,
                "in_active": frac,
                "enrichment": frac / vox_share,
                "ceiling": 1.0 / vox_share,
            }
        )
    return rows


def responding_mask(
    data_r: torch.Tensor,
    mask: torch.Tensor,
    top_frac: float = 0.1,
    thresh: float | None = None,
):
    """The voxels where the DATA responds to the task, and their complement.

    ``thresh`` cuts |r| at an absolute value; otherwise the top ``top_frac`` of voxels
    by |r| are taken.  An absolute cut is the honest one when you know what counts as
    activated; the quantile is the fallback that always yields a non-empty mask.

    A whole-brain median answers the wrong question.  Most of an automask is tissue
    with no task response, so it drags every summary toward zero and a real effect
    confined to responding cortex vanishes — which is how the first real run reported
    "nothing to fix" while the map showed r ~ 0.58 where the data fits.
    """
    strength = data_r.abs().amax(dim=-1)
    m = mask > 0
    cut = float(thresh) if thresh is not None else _quantile(strength[m], 1.0 - top_frac)
    active = (strength > cut) & m
    if not bool(active.any()):
        raise ValueError(
            f"the active mask is empty at |r| > {cut:g}: the data shows no task response "
            "anywhere. Lower -task_thresh, or check that the TR and events describe this "
            "run (an all-zero data coupling map means the design never lined up with it)."
        )
    return active, (strength <= cut) & m, cut


def format_task_coupling_report(
    field: TaskCoupling,
    data: TaskCoupling | None = None,
    coloc: float | None = None,
    *,
    units: str = "voxels",
    label: str = "displacement field",
    responding: dict | None = None,
    quiet: dict | None = None,
    top_frac: float = 0.1,
    slope: dict | None = None,
    enrichment: dict | None = None,
    active_thresh: float | None = None,
    curve: list[dict] | None = None,
) -> str:
    """Compact verdict, for the saved .txt and a one-line stdout echo.

    Numbers only. The reasoning behind each statistic — why signed ``r`` and not R2,
    why there is no per-voxel significance, why every summary is stratified by the
    data's own response — is in the module docstring and the wiki, not on stdout once
    per axis per run.
    """
    chance = field.chance_share
    n_k = len(field.labels)
    lines = [
        f"ffs task-coupling — {label}",
        f"  design      : {n_k} condition(s) ({', '.join(field.labels)}), "
        f"{field.n_timepoints} frames, drift polort {field.polort}",
    ]
    if enrichment is not None:
        thr = f" at |r|_data > {active_thresh:.3f}" if active_thresh is not None else ""
        lines.append(
            f"  active mask : {enrichment['n_active']} vox "
            f"({enrichment['voxel_share'] * 100:.1f}% of brain){thr}"
        )

    strata = [(f"active(top{top_frac * 100:.0f}%)", responding), ("quiet", quiet)]
    strata = [(n, s) for n, s in strata if s is not None] + [("whole mask", field.summary)]
    lines += [
        "",
        f"  {'stratum':<18}{'cond':<14}{'|r| med':>9}{'|r| p95':>9}"
        f"{'task-expl':>11}  (chance {chance * 100:.0f}%)",
    ]
    for name, st in strata:
        if not st or st.get("n_voxels", 0) == 0:
            lines.append(f"  {name:<18}(empty)")
            continue
        share = f"{_share(st) * 100:.0f}%"
        for i, c in enumerate(st["conditions"]):
            lines.append(
                f"  {name if i == 0 else '':<18}{c['label'][:13]:<14}"
                f"{c['abs_r_median']:>9.3f}{c['abs_r_p95']:>9.3f}"
                f"{share if i == 0 else '':>11}" + (f"  [{st['n_voxels']} vox]" if i == 0 else "")
            )

    if curve:
        lines += [
            "",
            f"  IS THE FIELD'S TASK COUPLING WHERE THE TASK IS?  "
            f"(1.0x = unrelated, ceiling {curve[0]['ceiling']:.0f}x)",
            f"  {'field |r| >':<18}{'n vox':>9}{'on active':>11}{'enrichment':>12}",
        ]
        for row in curve:
            lines.append(
                f"  {row['r_cut']:<18.3f}{row['n']:>9d}"
                f"{row['in_active'] * 100:>10.1f}%{row['enrichment']:>11.2f}x"
            )

    tail = []
    if enrichment is not None:
        tail.append(f"enrichment {enrichment['enrichment']:.2f}x")
    if coloc is not None:
        tail.append(f"co-location r {coloc:+.3f}")
    if slope is not None:
        tail.append(f"kappa {slope['kappa']:+.3f} (centred r {slope['r']:+.3f})")
    if tail:
        lines += ["", "  " + "  ·  ".join(tail)]
    if data is not None:
        dc = data.strongest
        lines.append(
            f"  data response: |r| med {dc.get('abs_r_median', 0):.3f} "
            f"p95 {dc.get('abs_r_p95', 0):.3f}   ({units} rms for the field columns)"
        )
    if coloc is not None:
        lines += ["", "  " + _verdict(field, coloc, responding, slope, enrichment, curve)]
    return "\n".join(lines) + "\n"


def _share(summary: dict) -> float:
    """Task-locked rms as a fraction of the field's rms — the least abstract number here.

    A correlation is unitless and a variance ratio reads small; a field that is 0.009
    of 0.013 voxels rms task-locked is two-thirds not-motion, which is the sentence a
    user acts on.
    """
    total = summary.get("total_rms_median", 0.0)
    return summary.get("task_rms_median", 0.0) / total if total > 0 else 0.0


def _verdict(
    field: TaskCoupling,
    coloc: float,
    responding: dict | None = None,
    slope: dict | None = None,
    enrichment: dict | None = None,
    curve: list[dict] | None = None,
) -> str:
    """One sentence naming which of the two mechanisms the numbers point at.

    The TAIL of :func:`enrichment_curve` decides.  Every earlier version decided on a
    median — first over the whole brain, then over the responding decile — and both
    are medians over voxels that carry no encode gradient and therefore cannot be
    contaminated at all.  On the run this was rewritten for, the responding-decile
    median put the ratio at 1.28 against a gate of 1.3 and printed "nothing to fix"
    while the field's top percentile was 9x concentrated on activated tissue.  A
    statistic that a threshold nudge of 0.02 flips is not measuring the effect.

    kappa is corroboration, and now actually enters the decision instead of being
    printed beside it: the physical relation it tests is exact only for an unpooled
    estimator, so a weak kappa cannot clear a field, but a kappa of order 1 alongside
    a rising tail is independent evidence of the same mechanism.
    """
    st = responding if responding and responding.get("n_voxels") else field.summary
    conds = st.get("conditions") or []
    if not conds:
        return "No conditions to judge."
    best = max(conds, key=lambda c: c["abs_r_median"])
    where = "where the data responds" if responding else "in the mask"
    share, chance = _share(st), field.chance_share
    ratio = share / chance if chance > 0 else 0.0

    tail = curve[-1] if curve else None
    # kappa near +/-1 is the physical prediction; the sign depends on the field's
    # displacement convention, so magnitude is what carries the information.
    phys = bool(
        slope and slope["n"] > 100 and abs(slope["r"]) > 0.2 and 0.3 < abs(slope["kappa"]) < 3.0
    )
    kap = (
        f" Physical test agrees (kappa {slope['kappa']:+.2f}, centred r {slope['r']:+.2f})."
        if phys
        else ""
    )

    if tail is not None:
        e_tail = tail["enrichment"]
        top = f"the top {(1 - tail['quantile']) * 100:g}% of the field (|r| > {tail['r_cut']:.2f})"
        if e_tail >= 2.0 or (e_tail >= 1.5 and phys):
            return (
                f"CONTAMINATION: {top} is {e_tail:.1f}x concentrated on activated tissue "
                f"(ceiling {tail['ceiling']:.0f}x, no-relation 1.0x).{kap} "
                "See -detask in -help."
            )
        if ratio >= 2.0:
            return (
                f"TASK-LOCKED {where}: {share * 100:.0f}% task-explained, {ratio:.1f}x "
                f"chance, but the field's own tail is only {e_tail:.1f}x concentrated on "
                "activated tissue — real task-correlated motion, not BOLD read as motion. "
                "Do NOT project it out without looking at the maps."
            )
        return (
            f"NO APPRECIABLE COUPLING: {top} is {e_tail:.1f}x concentrated on activated "
            f"tissue (no-relation 1.0x); {share * 100:.0f}% task-explained {where} vs "
            f"{chance * 100:.0f}% chance."
        )

    # No curve (a caller that did not supply the masks): fall back to the share, and
    # say which statistic is being used so the weaker verdict is not read as the strong one.
    e = enrichment["enrichment"] if enrichment else None
    if ratio >= 2.0 or (e is not None and e >= 1.5 and ratio >= 1.3):
        conc = f", {e:.2f}x concentrated on active tissue" if e is not None else ""
        return (
            f"CONTAMINATION {where} (share-based): {share * 100:.0f}% task-explained, "
            f"{ratio:.1f}x chance{conc}.{kap} See -detask in -help."
        )
    if ratio < 0.8:
        return (
            f"CLEAR {where} (share-based): {share * 100:.0f}% task-explained, BELOW the "
            f"{chance * 100:.0f}% chance level (where a paired reference lands)."
        )
    if best["abs_r_median"] > 0.25:
        return (
            f"TASK-LOCKED {where} (share-based, |r| {best['abs_r_median']:.2f}) at only "
            f"{ratio:.1f}x chance — real task-correlated motion, or a slow field aliasing "
            "onto a slow design. Check the maps before projecting."
        )
    return (
        f"NO APPRECIABLE COUPLING {where} (share-based): {share * 100:.0f}% "
        f"task-explained vs {chance * 100:.0f}% chance."
    )
