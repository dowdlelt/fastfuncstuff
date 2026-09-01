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
    "project_task_out",
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
        df = max(1, self.n_timepoints - self.polort - 1)
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
    q_n = _orthonormal_basis(
        construct_polynomial_matrix(n_t, polort, device=device, dtype=torch.float64)
    )
    x_scale = float(torch.linalg.matrix_norm(x, 2))

    def _detrended_units(cols: torch.Tensor) -> torch.Tensor:
        return _unit_columns(cols - q_n @ (q_n.T @ cols))

    u_x = _detrended_units(x)
    if float(torch.linalg.matrix_norm(x - q_n @ (q_n.T @ x), 2)) < 1e-6 * x_scale:
        raise ValueError(
            f"every task column is collinear with the polort-{polort} drift basis. "
            "Lower -task_polort, or the design is not separable from drift at this "
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
    m = mask.reshape(-1) > 0
    a = (active.reshape(-1) > 0) & m
    energy = (field.task_rms.reshape(-1)[m] ** 2).double()
    inside = (field.task_rms.reshape(-1)[a] ** 2).double()
    tot = float(energy.sum())
    vox_share = float(a.sum()) / max(1, int(m.sum()))
    e_share = float(inside.sum()) / tot if tot > 0 else 0.0
    return {
        "n_active": int(a.sum()),
        "voxel_share": vox_share,
        "energy_share": e_share,
        "enrichment": e_share / vox_share if vox_share > 0 else 0.0,
    }


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
        lines += ["", "  " + _verdict(field, coloc, responding, slope, enrichment)]
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
) -> str:
    """One sentence naming which of the two mechanisms the numbers point at.

    ENRICHMENT decides: it asks the question directly — are we moving voxels, in a
    task-correlated way, where voxels are task-correlated — and its no-relation value
    is 1 by construction rather than by simulation.  The kappa slope corroborates when
    its centred r is real, but it cannot be the arbiter: it predicts a per-voxel
    relation that a window-pooled estimator blurs.  Detection is judged on the
    RESPONDING stratum; a whole-brain median is the statistic that hid the effect in
    the first place.
    """
    st = responding if responding and responding.get("n_voxels") else field.summary
    conds = st.get("conditions") or []
    if not conds:
        return "No conditions to judge."
    best = max(conds, key=lambda c: c["abs_r_median"])
    where = "where the data responds" if responding else "in the mask"

    # The task-explained SHARE is the robust detector, not enrichment. Enrichment is a
    # ratio of shares, so when real motion dominates the field it is dividing two noisy
    # numbers and tracks nothing -- measured 0.89x on a phantom whose field was 21%
    # task-explained, and 0.77x after a fix that cut that to 1%.
    share, chance = _share(st), field.chance_share
    ratio = share / chance if chance > 0 else 0.0
    e = enrichment["enrichment"] if enrichment else None
    kap = ""
    if slope is not None and slope["n"] > 100 and abs(slope["r"]) > 0.2:
        kap = f" Physical test agrees (kappa {slope['kappa']:+.2f}, centred r {slope['r']:+.2f})."

    if ratio >= 2.0 or (e is not None and e >= 1.5 and ratio >= 1.3):
        conc = f", {e:.2f}x concentrated on active tissue" if e is not None else ""
        return (
            f"CONTAMINATION {where}: {share * 100:.0f}% task-explained, {ratio:.1f}x "
            f"chance{conc}.{kap} Fix: -ref paired, or -detask on this field."
        )
    if ratio < 0.8:
        return (
            f"CLEAR {where}: {share * 100:.0f}% task-explained, BELOW the "
            f"{chance * 100:.0f}% chance level (where a paired reference lands)."
        )
    if best["abs_r_median"] > 0.25:
        return (
            f"TASK-LOCKED {where} (|r| {best['abs_r_median']:.2f}) at only {ratio:.1f}x "
            "chance and not on active tissue — real task-correlated motion, or a slow "
            "field aliasing onto a slow design. Check the maps before projecting."
        )
    return (
        f"NO APPRECIABLE COUPLING {where}: {share * 100:.0f}% task-explained vs "
        f"{chance * 100:.0f}% chance."
    )
