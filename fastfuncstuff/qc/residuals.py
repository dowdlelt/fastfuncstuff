"""What a GLM left behind, measured for QC from the data the fit already holds.

Everything here is ordinary least squares on the model's shared design, not the
ARMA-whitened fit: whitening exists to make residuals look white, so whitened
residuals hide exactly the structure a QC figure is for (motion bands,
breathing, leaked task). Two noise levels come out of one pass over the voxels:

- **input**: the SD left after the drift columns alone, which is the series the
  model was handed;
- **residual**: the SD left after the whole design.

Both divide by their own degrees of freedom (time points minus the rank of what
was removed). Without that a model gains tSNR just by having more columns, and
"the residual is quieter" would say nothing about the regressors.

Nothing is materialised whole: voxels stream through in chunks sized by the
memory module, and only per-voxel sums come back.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ResidualStats:
    """Per-voxel noise levels of one fit, over the time points it kept."""

    mean: np.ndarray  # (V,) temporal mean
    sd_in: np.ndarray  # (V,) SD after drift only, dof-corrected
    sd_res: np.ndarray  # (V,) SD after the full design, dof-corrected
    run_sd_in: np.ndarray  # (R, V)
    run_sd_res: np.ndarray  # (R, V)
    rank_drift: int
    rank_full: int
    n_kept: int
    run_kept: np.ndarray  # (R,) kept time points per run

    def tsnr_in(self) -> np.ndarray:
        return _ratio(np.abs(self.mean), self.sd_in)

    def tsnr_res(self) -> np.ndarray:
        return _ratio(np.abs(self.mean), self.sd_res)

    def run_tsnr(self, sd: np.ndarray) -> np.ndarray:
        return _ratio(np.abs(self.mean)[None], sd)


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.zeros(np.broadcast_shapes(num.shape, den.shape), np.float32)
    np.divide(num, den, out=out, where=den > 0)
    return out


def orthonormal_basis(a: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    """Orthonormal columns spanning ``a``'s column space (float64, via SVD).

    SVD rather than QR because a design can be rank deficient without being
    wrong: a condition with no events in the kept time points is a zero column,
    and its rank must not be charged to the degrees of freedom.
    """
    a = np.asarray(a, np.float64)
    if a.shape[1] == 0:
        return np.zeros((a.shape[0], 0))
    u, s, _ = np.linalg.svd(a, full_matrices=False)
    return u[:, s > rtol * max(s.max(), 1e-300)]


def run_of_timepoint(run_starts, n_time: int) -> np.ndarray:
    """Run index of every time point."""
    starts = np.asarray(list(run_starts) or [0], int)
    return np.searchsorted(starts, np.arange(n_time), side="right") - 1


def residual_stats(
    data,
    voxels: np.ndarray,
    design: np.ndarray,
    drift_cols,
    run_starts,
    keep: np.ndarray | None = None,
    device=None,
) -> ResidualStats:
    """Input and residual noise per voxel and per run.

    ``data`` is any array whose last axis is time; ``voxels`` are flat indices
    into its leading axes. ``design`` is ``(time, columns)`` on the full
    timeline and ``keep`` marks the time points the fit used.
    """
    import torch

    from fastfuncstuff.memory import estimate_chunk_size
    from fastfuncstuff.utils import get_device

    dev = torch.device(device) if device is not None else get_device()
    x = np.asarray(design, np.float64)
    n_time = x.shape[0]
    keep = np.ones(n_time, bool) if keep is None else np.asarray(keep, bool)
    rows = np.flatnonzero(keep)
    xk = x[rows]
    q_drift = orthonormal_basis(xk[:, list(drift_cols)])
    q_full = orthonormal_basis(xk)
    n = rows.size
    run_id = run_of_timepoint(run_starts, n_time)[rows]
    n_runs = len(list(run_starts) or [0])
    run_kept = np.bincount(run_id, minlength=n_runs).astype(np.float64)

    flat = np.asarray(data).reshape(-1, n_time)
    voxels = np.asarray(voxels, np.int64)
    nv = voxels.size
    chunk = estimate_chunk_size(nv, n, max(q_full.shape[1], 1), dev, operation="glm")
    qd = torch.from_numpy(q_drift).to(dev).to(torch.float32)
    qf = torch.from_numpy(q_full).to(dev).to(torch.float32)
    rows_t = torch.from_numpy(rows).to(dev)
    onehot = torch.zeros(n, n_runs, device=dev, dtype=torch.float32)
    onehot[torch.arange(n, device=dev), torch.from_numpy(run_id).to(dev)] = 1.0

    mean = np.zeros(nv, np.float32)
    ss = np.zeros((2, n_runs, nv), np.float32)
    for a in range(0, nv, chunk):
        idx = voxels[a : a + chunk]
        # (voxels, time) -> kept time points; the gather is the only host copy.
        y = torch.from_numpy(np.ascontiguousarray(flat[idx])).to(dev).to(torch.float32)
        y = y[:, rows_t]
        mean[a : a + chunk] = y.mean(1).cpu().numpy()
        for k, q in enumerate((qd, qf)):
            r = y - (y @ q) @ q.T
            ss[k, :, a : a + chunk] = ((r * r) @ onehot).T.cpu().numpy()
            del r
        del y

    def sd(k: int, rank: int) -> tuple[np.ndarray, np.ndarray]:
        # A run's share of the model's rank is its share of the time points.
        whole = np.sqrt(ss[k].sum(0) / max(n - rank, 1))
        dof_run = np.maximum(run_kept * (1.0 - rank / max(n, 1)), 1.0)
        return whole.astype(np.float32), np.sqrt(ss[k] / dof_run[:, None]).astype(np.float32)

    sd_in, run_in = sd(0, q_drift.shape[1])
    sd_res, run_res = sd(1, q_full.shape[1])
    return ResidualStats(
        mean, sd_in, sd_res, run_in, run_res, q_drift.shape[1], q_full.shape[1], n, run_kept
    )


def fit_parts(y_tv: np.ndarray, design: np.ndarray, drift_cols, keep: np.ndarray | None = None):
    """``(data, fitted, residual)`` time courses for a few voxels, drift removed.

    Betas come from the kept time points and are applied to every time point, so
    a censored volume shows in the residual as what the model could not explain
    rather than as a hole. ``data = fitted + residual``; the drift part belongs
    to neither.
    """
    x = np.asarray(design, np.float64)
    keep = np.ones(x.shape[0], bool) if keep is None else np.asarray(keep, bool)
    drift = list(drift_cols)
    rest = [i for i in range(x.shape[1]) if i not in set(drift)]
    # The pseudo-inverse is the sensitive step (float64); the products are not, and
    # float32 keeps three voxel-by-time arrays from costing six.
    y = np.asarray(y_tv, np.float32)
    beta = np.linalg.pinv(x[keep]).astype(np.float32) @ y[keep]
    x32 = x.astype(np.float32)
    data = y - x32[:, drift] @ beta[drift]
    fitted = x32[:, rest] @ beta[rest]
    return data, fitted, data - fitted


def heldout_r2_by_run(
    y_vt,
    design: np.ndarray,
    run_starts,
    task_cols,
    nuisance_cols,
    keep: np.ndarray | None = None,
    device=None,
    top_frac: float = 0.05,
) -> np.ndarray:
    """One number per run: how well the other runs' task betas predict it.

    Each run is held out in turn (nuisance projected fold-locally by
    :func:`fastfuncstuff.glm.xval.compute_xval_r2`), giving an R² per voxel per
    run. The run's number is the median over the voxels the OTHER runs' scores
    rank in the top ``top_frac``: picking the voxels on the run being scored
    would be selecting on its own noise. NaN for a run that cannot be scored
    (fewer than two runs, no task columns).
    """
    import torch

    from fastfuncstuff.glm.reml_xval import censor_xval_inputs
    from fastfuncstuff.glm.xval import compute_xval_r2

    starts = [int(s) for s in (list(run_starts) or [0])]
    n_runs = len(starts)
    out = np.full(n_runs, np.nan)
    task = list(task_cols)
    if n_runs < 2 or not task:
        return out
    dev = torch.device(device) if device is not None else None
    y = torch.as_tensor(np.asarray(y_vt), dtype=torch.float32)
    good = None if keep is None else np.flatnonzero(np.asarray(keep, bool)).tolist()
    y, x, kept_starts, _ = censor_xval_inputs(y, np.asarray(design, np.float64), starts, good)
    if len(kept_starts) != n_runs:  # a wholly censored run: no stable run index
        return out
    r2 = np.full((n_runs, y.shape[0]), np.nan, np.float32)
    for r in range(n_runs):
        others = [i for i in range(n_runs) if i != r]
        res = compute_xval_r2(
            y, x, kept_starts, task, list(nuisance_cols), [(others, [r])], device=dev, verbose=False
        )
        r2[r] = torch.as_tensor(res["r2"]).float().cpu().numpy()
    ok = np.isfinite(r2)
    k = max(1, round(top_frac * y.shape[0]))
    for r in range(n_runs):
        rest = np.delete(np.arange(n_runs), r)
        score = np.where(ok[rest], r2[rest], 0.0).sum(0) / np.maximum(ok[rest].sum(0), 1)
        vals = r2[r, np.argsort(-score)[:k]]
        if np.isfinite(vals).any():
            out[r] = float(np.nanmedian(vals))
    return out


# SPM's tissue order: what a bare posterior stack is assumed to be.
SPM_CLASSES = ("GM", "WM", "CSF")
TISSUE_MIN_P = 0.9


def tissue_groups(
    tissue: np.ndarray, names, mask: np.ndarray, min_p: float = TISSUE_MIN_P
) -> dict[str, np.ndarray]:
    """Boolean masks per tissue name from a label volume or a stack of posteriors.

    ``tissue`` is ``(z, y, x)`` integer labels (1 = first name) or
    ``(classes, z, y, x)`` probabilities on the mask's grid. A voxel joins a
    class only where that class holds ``min_p`` of it: at EPI resolution most
    of cortex is partial volume, and a carpet band named "GM" that is a third
    white matter would be read as grey matter. Several volumes may share a
    name (cortex + cerebellum + subcortex are all "GM"); a name of ``-`` drops
    its volume.
    """
    t = np.asarray(tissue)
    names = list(names)
    if t.ndim == 3:
        lab = np.rint(t).astype(int)
        prob = [(lab == i + 1).astype(np.float32) for i in range(len(names))]
    else:
        prob = [t[i].astype(np.float32) for i in range(min(len(names), t.shape[0]))]
        peak = max((float(p.max()) for p in prob), default=1.0)
        if peak > 1.5:  # stored as 0..255 or percent
            prob = [p / peak for p in prob]
    summed: dict[str, np.ndarray] = {}
    for name, p in zip(names, prob, strict=False):
        if name != "-":
            summed[name] = summed[name] + p if name in summed else p
    out = {k: (v >= min_p) & mask for k, v in summed.items()}
    return {k: v for k, v in out.items() if v.any()}
