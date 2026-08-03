"""Multi-echo T2*/S0 decay fitting, optimal combination, and leave-one-echo-out QC.

GPU-first rebuild of tedana's ``t2smap`` (decay.py + combine.py). The accurate
nonlinear path replaces tedana's per-voxel ``scipy.optimize.curve_fit`` with a single
batched Levenberg-Marquardt over every voxel at once, which is what makes the fit fast.

Everything works on masked samples shaped ``(V, E, T)`` (voxels, echoes, time) so the
math is plain batched linear algebra. Echo times are handled in milliseconds internally
(the same convention as tedana); the CLI converts user seconds -> ms and the output T2*
map ms -> seconds.

Two ideas drive the design:

* The adaptive mask (how many echoes have good signal at each voxel) is encoded as a
  per-echo weight matrix. A weight of 0 drops an echo, so variable echo counts and
  robust down-weighting flow through the same weighted solver instead of tedana's
  per-echo-count Python grouping.
* Leave-one-echo-out (LOEO): on the per-echo temporal mean, refit with E-1 echoes and
  predict the held-out echo. The predicted-vs-actual residual is a jackknife that flags
  per-echo dropout/ghosting far more locally than a global RMSE, and drives an optional
  robust (Tukey-biweight IRLS) refit.
"""

from __future__ import annotations

import torch
from torch import Tensor
from tqdm.auto import tqdm

from fastfuncstuff.memory import estimate_chunk_size
from fastfuncstuff.utils import warn_mps_float32_precision

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def monoexp(tes: Tensor, s0: Tensor, t2star: Tensor) -> Tensor:
    """Monoexponential decay ``S0 * exp(-TE / T2*)``.

    Args:
        tes: ``(M,)`` echo times (any consistent unit).
        s0: ``(B,)`` initial-signal estimates.
        t2star: ``(B,)`` T2* estimates (same unit as ``tes``).

    Returns:
        ``(B, M)`` predicted signal.
    """
    return s0.unsqueeze(-1) * torch.exp(-tes.unsqueeze(0) / t2star.unsqueeze(-1))


# ---------------------------------------------------------------------------
# Adaptive mask (torch port of tedana.utils.make_adaptive_mask)
# ---------------------------------------------------------------------------


def make_adaptive_mask(
    data: Tensor,
    methods: tuple[str, ...] = ("dropout",),
    threshold: int = 1,
) -> tuple[Tensor, Tensor]:
    """Map of how many echoes have usable signal at each voxel.

    Torch port of ``tedana.utils.make_adaptive_mask``. Combines a base mask (flag any
    NaN/<=0 sample), an optional "dropout" criterion (echo intensity below 1/3 of an
    exemplar 33rd-percentile voxel), and an optional "decay" criterion (signal stops
    decreasing across echoes), taking the element-wise minimum.

    Args:
        data: ``(V, E, T)`` masked multi-echo data.
        methods: any of ``"dropout"``, ``"decay"``, ``"none"``.
        threshold: minimum good-echo count to retain (values below set to 0).

    Returns:
        ``(mask, adaptive_mask)`` where ``mask`` is boolean ``(V,)`` and
        ``adaptive_mask`` is integer ``(V,)`` good-echo counts.
    """
    methods = tuple(m.lower() for m in methods)
    if not all(m in ("dropout", "decay", "none") for m in methods):
        raise ValueError(f"Unknown adaptive-mask method in {methods}")

    n_samples, n_echos, _ = data.shape
    masks = []

    # Base mask: longest prefix of echoes with no NaN/<=0 sample at any timepoint.
    bad = torch.isnan(data) | (data <= 0)
    good_vox_echoes = (~bad.any(dim=-1)).to(torch.int64)  # (V, E)
    # cumprod over a 0/1 row is 1 up to the first bad echo and 0 after, so its
    # sum is exactly that prefix length — same integers as the sequential
    # where() chain, without one kernel per echo.
    base = torch.cumprod(good_vox_echoes, dim=1).sum(dim=1)
    masks.append(base)

    if "dropout" in methods or "decay" in methods:
        echo_means = data.mean(dim=-1)  # (V, E)

    if "dropout" in methods:
        first_echo = echo_means[echo_means[:, 0] != 0, 0]
        # 33rd percentile via the "higher" interpolation, like tedana.
        perc = torch.quantile(first_echo, 0.33, interpolation="higher")
        perc_val = echo_means[:, 0] == perc
        lthrs = echo_means[perc_val].T / 3.0  # (E, n_exemplar)
        if lthrs.ndim > 1:
            lthrs = lthrs[:, lthrs.sum(dim=0).argmax()]
        dropout = torch.zeros(n_samples, dtype=torch.int64, device=data.device)
        for e in range(n_echos):
            dropout[echo_means[:, e].abs() > lthrs[e]] = e + 1
        masks.append(dropout)

    if "decay" in methods:
        diffs = torch.diff(echo_means, dim=1)
        diffs = torch.cat([torch.full((n_samples, 1), -1.0, device=data.device), diffs], dim=1)
        not_decreasing = diffs >= 0
        last_dec = not_decreasing.to(torch.int64).argmax(dim=1)
        last_dec[last_dec == 0] = n_echos  # never increased -> all echoes good
        masks.append(last_dec)

    adaptive = torch.stack(masks, dim=0).amin(dim=0)
    adaptive[adaptive < threshold] = 0
    return adaptive.to(torch.bool), adaptive


def availability_weights(adaptive_mask: Tensor, n_echos: int) -> Tensor:
    """Per-echo binary weights from an adaptive mask.

    A voxel with ``k`` good echoes uses its first ``k`` echoes; voxels with one good
    echo borrow the first two (matching tedana's ``echo_num == 2`` grouping). Voxels
    with zero good echoes get all-zero weights.

    Args:
        adaptive_mask: ``(V,)`` good-echo counts.
        n_echos: total number of echoes ``E``.

    Returns:
        ``(V, E)`` float weights in ``{0, 1}``.
    """
    n_use = torch.where(adaptive_mask >= 2, adaptive_mask, torch.full_like(adaptive_mask, 2))
    n_use = torch.where(adaptive_mask == 0, torch.zeros_like(adaptive_mask), n_use)
    idx = torch.arange(n_echos, device=adaptive_mask.device).unsqueeze(0)  # (1, E)
    return (idx < n_use.unsqueeze(1)).to(torch.float32)


# ---------------------------------------------------------------------------
# Weighted log-linear fit (closed form, fully batched)
# ---------------------------------------------------------------------------


def fit_loglinear(y: Tensor, tes: Tensor, weights: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Weighted log-linear monoexponential fit.

    Fits ``log(|S| + 1) = log(S0) - TE / T2*`` per row by weighted least squares,
    solving the 2x2 normal equations in closed form. An echo weight of 0 removes that
    observation, so this transparently handles per-voxel echo counts.

    Args:
        y: ``(B, M)`` signal (echo, or echo*time, observations).
        tes: ``(M,)`` echo times.
        weights: ``(B, M)`` non-negative weights. Default all ones.

    Returns:
        ``(t2s, s0)`` each ``(B,)``. Raw estimates without floors/ceilings; callers
        apply :func:`modify_t2s_s0_maps`.
    """
    w = torch.ones_like(y) if weights is None else weights
    logy = torch.log(y.abs() + 1.0)
    # Design columns c0 = 1, c1 = -TE. Work in float64 for the closed-form solve;
    # MPS has no float64, so it computes in float32 there (full-batch over voxels,
    # so a CPU fallback would be a large transfer — use -device cpu for full precision).
    dtype = torch.float32 if y.device.type == "mps" else torch.float64
    te = tes.to(dtype)
    w = w.to(dtype)
    logy = logy.to(dtype)

    a00 = w.sum(dim=1)
    a01 = (w * -te).sum(dim=1)
    a11 = (w * te * te).sum(dim=1)
    b0 = (w * logy).sum(dim=1)
    b1 = (w * -te * logy).sum(dim=1)

    det = a00 * a11 - a01 * a01
    det = torch.where(det.abs() < 1e-12, torch.full_like(det, 1e-12), det)
    beta0 = (a11 * b0 - a01 * b1) / det
    beta1 = (a00 * b1 - a01 * b0) / det

    t2s = 1.0 / beta1
    s0 = torch.exp(beta0)
    return t2s.to(y.dtype), s0.to(y.dtype)


# ---------------------------------------------------------------------------
# Batched Levenberg-Marquardt curve fit (the accelerator)
# ---------------------------------------------------------------------------


def _lm_curvefit_chunk(
    y: Tensor,
    tes: Tensor,
    s0: Tensor,
    t2s: Tensor,
    weights: Tensor,
    n_iter: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Batched LM for one chunk. Returns t2s, s0, failures, t2s_var, s0_var, covar."""
    # MPS has no float64; the LM fit runs in float32 there (use -device cpu for
    # full float64 precision). The fit is full-batch over voxels, so a CPU
    # fallback would mean a large host transfer per chunk.
    if y.device.type == "mps":
        warn_mps_float32_precision("multi-echo T2*/S0 LM fit")
        dtype = torch.float32
    else:
        dtype = torch.float64
    te = tes.to(dtype)
    y = y.to(dtype)
    w = weights.to(dtype)
    s0 = s0.to(dtype).clone()
    t2s = t2s.to(dtype).clone()

    eps = 1e-8
    # Guard the log-linear seed: T2* must be positive, S0 finite/positive.
    bad_seed = ~torch.isfinite(t2s) | (t2s <= eps) | ~torch.isfinite(s0) | (s0 <= 0)
    t2s = torch.where(bad_seed, torch.full_like(t2s, te.mean()), t2s)
    s0 = torch.where(bad_seed, y.abs().amax(dim=1).clamp_min(eps), s0)

    def cost(s0_, t2s_):
        r = s0_.unsqueeze(1) * torch.exp(-te.unsqueeze(0) / t2s_.unsqueeze(1)) - y
        return (w * r * r).sum(dim=1)

    lam = torch.full_like(s0, 1e-3)
    prev_cost = cost(s0, t2s)

    for _ in range(n_iter):
        exp_term = torch.exp(-te.unsqueeze(0) / t2s.unsqueeze(1))  # (B, M)
        pred = s0.unsqueeze(1) * exp_term
        r = pred - y
        # Jacobian columns
        j0 = exp_term
        j1 = s0.unsqueeze(1) * te.unsqueeze(0) / (t2s.unsqueeze(1) ** 2) * exp_term
        wj0, wj1 = w * j0, w * j1

        g0 = (wj0 * r).sum(dim=1)
        g1 = (wj1 * r).sum(dim=1)
        h00 = (wj0 * j0).sum(dim=1)
        h01 = (wj0 * j1).sum(dim=1)
        h11 = (wj1 * j1).sum(dim=1)

        # LM damping on the diagonal.
        d00 = h00 * (1.0 + lam)
        d11 = h11 * (1.0 + lam)
        det = d00 * d11 - h01 * h01
        det = torch.where(det.abs() < 1e-20, torch.full_like(det, 1e-20), det)
        step_s0 = -(d11 * g0 - h01 * g1) / det
        step_t2s = -(d00 * g1 - h01 * g0) / det

        new_s0 = (s0 + step_s0).clamp_min(eps)
        new_t2s = (t2s + step_t2s).clamp_min(eps)
        new_cost = cost(new_s0, new_t2s)

        improved = new_cost < prev_cost
        s0 = torch.where(improved, new_s0, s0)
        t2s = torch.where(improved, new_t2s, t2s)
        # Accepted steps shrink lambda (toward Gauss-Newton); rejected grow it.
        lam = torch.where(improved, (lam * 0.3).clamp_min(1e-9), (lam * 3.0).clamp_max(1e9))
        prev_cost = torch.where(improved, new_cost, prev_cost)
        # No early break: under heavy damping an accepted step can be tiny while still
        # far from the optimum, so a step-size criterion stops voxels short. The fit is
        # fully batched and converges from the log-linear seed in well under n_iter
        # iterations, so a fixed budget is both cheaper to reason about and correct.

    # Covariance from (J^T W J)^-1 * sigma^2 at the solution.
    exp_term = torch.exp(-te.unsqueeze(0) / t2s.unsqueeze(1))
    j0 = exp_term
    j1 = s0.unsqueeze(1) * te.unsqueeze(0) / (t2s.unsqueeze(1) ** 2) * exp_term
    h00 = (w * j0 * j0).sum(dim=1)
    h01 = (w * j0 * j1).sum(dim=1)
    h11 = (w * j1 * j1).sum(dim=1)
    det = h00 * h11 - h01 * h01
    failures = ~torch.isfinite(t2s) | ~torch.isfinite(s0) | (det.abs() < 1e-20)
    det_safe = torch.where(det.abs() < 1e-20, torch.full_like(det, 1e-20), det)

    n_obs = (w > 0).sum(dim=1).clamp_min(1)
    dof = (n_obs - 2).clamp_min(1)
    resid = s0.unsqueeze(1) * exp_term - y
    sigma2 = (w * resid * resid).sum(dim=1) / dof
    # Inverse of symmetric 2x2 [[h00,h01],[h01,h11]].
    inv00 = h11 / det_safe
    inv11 = h00 / det_safe
    inv01 = -h01 / det_safe
    s0_var = (inv00 * sigma2).to(y.dtype)
    t2s_var = (inv11 * sigma2).to(y.dtype)
    covar = (inv01 * sigma2).to(y.dtype)

    return t2s.to(y.dtype), s0.to(y.dtype), failures, t2s_var, s0_var, covar


def fit_curvefit_gpu(
    y: Tensor,
    tes: Tensor,
    s0_init: Tensor,
    t2s_init: Tensor,
    weights: Tensor | None = None,
    n_iter: int = 50,
    device: torch.device | None = None,
    verbose: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Batched nonlinear monoexponential fit (Levenberg-Marquardt).

    Drop-in accurate alternative to tedana's per-voxel ``scipy.optimize.curve_fit``,
    solving every row simultaneously with an analytic Jacobian. The 2x2 normal
    equations are formed and solved in float64 (an ill-conditioned 2x2 loses the step
    in float32), then cast back.

    Args:
        y: ``(B, M)`` signal.
        tes: ``(M,)`` echo times.
        s0_init, t2s_init: ``(B,)`` seeds (typically from :func:`fit_loglinear`).
        weights: ``(B, M)`` non-negative weights. Default all ones.
        n_iter: number of LM iterations (fixed budget; converges well within it).
        device: compute device (defaults to ``y.device``).
        verbose: show a progress bar over voxel chunks.

    Returns:
        ``(t2s, s0, failures, t2s_var, s0_var, t2s_s0_covar)`` each ``(B,)``.
    """
    device = device or y.device
    w = torch.ones_like(y) if weights is None else weights
    b = y.shape[0]
    m = y.shape[1]

    chunk = estimate_chunk_size(
        n_voxels=b,
        n_timepoints=m,
        n_regressors=2,
        device=device,
        operation="glm",
        use_double=True,
    )
    outs = [torch.empty(b, dtype=y.dtype, device=device) for _ in range(5)]
    failures = torch.empty(b, dtype=torch.bool, device=device)

    for start in tqdm(
        range(0, b, chunk), desc="curvefit", leave=False, disable=not verbose or b <= chunk
    ):
        sl = slice(start, min(start + chunk, b))
        t2s_c, s0_c, fail_c, t2v_c, s0v_c, cov_c = _lm_curvefit_chunk(
            y[sl], tes, s0_init[sl], t2s_init[sl], w[sl], n_iter
        )
        outs[0][sl], outs[1][sl] = t2s_c, s0_c
        outs[2][sl], outs[3][sl], outs[4][sl] = t2v_c, s0v_c, cov_c
        failures[sl] = fail_c

    return outs[0], outs[1], failures, outs[2], outs[3], outs[4]


# ---------------------------------------------------------------------------
# Decay-fit orchestration (all / ts, mean / all-timepoints)
# ---------------------------------------------------------------------------


def _flatten_for_fit(
    data_cat: Tensor,
    tes: Tensor,
    avail: Tensor,
    fitmode: str,
    fit_all_timepoints: bool,
    extra_weights: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, tuple]:
    """Reshape ``(V, E, T)`` data into ``(B, M)`` observations for the batched solvers.

    Returns ``(y, tes_long, weights, restore)`` where ``restore`` carries the shape
    metadata to put fitted ``(B,)`` params back into voxel (and possibly time) shape.
    """
    v, e, t = data_cat.shape
    if extra_weights is None:
        extra_weights = torch.ones_like(avail)

    if fitmode == "ts":
        # One fit per voxel per timepoint: B = V*T, M = E.
        y = data_cat.permute(0, 2, 1).reshape(v * t, e)
        w = (avail * extra_weights).unsqueeze(1).expand(v, t, e).reshape(v * t, e)
        return y, tes, w, ("ts", v, t)

    # fitmode == "all"
    if fit_all_timepoints:
        # Every (echo, timepoint) is an observation sharing one S0/T2* per voxel.
        y = data_cat.reshape(v, e * t)
        tes_long = tes.repeat_interleave(t)
        w = (avail * extra_weights).repeat_interleave(t, dim=1)
        return y, tes_long, w, ("all", v, 1)

    # Default "all": collapse time to the per-echo temporal mean.
    y = data_cat.mean(dim=2)
    w = avail * extra_weights
    return y, tes, w, ("all", v, 1)


def fit_decay(
    data_cat: Tensor,
    tes: Tensor,
    adaptive_mask: Tensor,
    fittype: str = "loglin",
    fitmode: str = "all",
    fit_all_timepoints: bool = False,
    weights: Tensor | None = None,
    device: torch.device | None = None,
    verbose: bool = False,
) -> dict[str, Tensor]:
    """Fit voxel-wise (and optionally timepoint-wise) monoexponential decay.

    Args:
        data_cat: ``(V, E, T)`` masked multi-echo data.
        tes: ``(E,)`` echo times (ms).
        adaptive_mask: ``(V,)`` good-echo counts.
        fittype: ``"loglin"`` or ``"curvefit"``.
        fitmode: ``"all"`` (one estimate per voxel) or ``"ts"`` (per voxel per TR).
        fit_all_timepoints: with ``fitmode="all"``, fit jointly across all TRs instead
            of the per-echo temporal mean (tedana-exact, slower).
        weights: optional ``(V, E)`` extra per-echo weights (e.g. robustness weights),
            multiplied into the availability weights.
        device: compute device.
        verbose: progress bars on the heavy loops.

    Returns:
        Dict with ``t2s``/``s0`` ``(V,)`` for ``"all"`` or ``(V, T)`` for ``"ts"``,
        plus (curvefit only) ``failures``, ``t2s_var``, ``s0_var``, ``t2s_s0_covar``.
    """
    device = device or data_cat.device
    avail = availability_weights(adaptive_mask, data_cat.shape[1])
    y, tes_long, w, restore = _flatten_for_fit(
        data_cat, tes, avail, fitmode, fit_all_timepoints, weights
    )

    t2s_init, s0_init = fit_loglinear(y, tes_long, w)
    if fittype == "loglin":
        t2s, s0 = t2s_init, s0_init
        extra: dict[str, Tensor] = {}
    elif fittype == "curvefit":
        t2s, s0, failures, t2s_var, s0_var, covar = fit_curvefit_gpu(
            y, tes_long, s0_init, t2s_init, weights=w, device=device, verbose=verbose
        )
        extra = {
            "failures": failures,
            "t2s_var": t2s_var,
            "s0_var": s0_var,
            "t2s_s0_covar": covar,
        }
    else:
        raise ValueError(f"Unknown fittype: {fittype!r}")

    kind, v, t = restore
    if kind == "ts":
        t2s = t2s.reshape(v, t)
        s0 = s0.reshape(v, t)
        extra = {k: val.reshape(v, t) for k, val in extra.items()}

    out = {"t2s": t2s, "s0": s0}
    out.update(extra)
    return out


# ---------------------------------------------------------------------------
# Floors / ceilings (port of tedana.decay.modify_t2s_s0_maps)
# ---------------------------------------------------------------------------


def modify_t2s_s0_maps(
    t2s: Tensor, s0: Tensor, adaptive_mask: Tensor, tes: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Apply tedana's floors/ceilings and build full vs limited maps.

    inf T2* -> 500, T2* <= 0 -> 1, a small positive floor to avoid divide-by-zero in
    optimal combination, NaN S0 -> 0, and a hard cap at 10x the 99.5th percentile.
    "Limited" maps zero out voxels with only one good echo.

    Args:
        t2s, s0: ``(V,)`` raw estimates.
        adaptive_mask: ``(V,)`` good-echo counts.
        tes: ``(E,)`` echo times (ms).

    Returns:
        ``(t2s_full, s0_full, t2s_limited, s0_limited)``.
    """
    t2s = t2s.clone()
    s0 = s0.clone()
    t2s[torch.isinf(t2s)] = 500.0
    t2s[t2s <= 0] = 1.0
    # Floor very small positive T2* so exp(-TE/T2*) doesn't underflow to 0.
    eps = torch.finfo(t2s.dtype).eps
    min_te = tes.min()
    floor = (-min_te / torch.log(torch.tensor(eps, device=t2s.device, dtype=t2s.dtype))).abs()
    nonzero = t2s != 0
    underflow = nonzero & (torch.exp(-tes.max() / t2s) == 0)
    t2s[underflow] = floor
    s0[torch.isnan(s0)] = 0.0

    t2s_limited = t2s.clone()
    s0_limited = s0.clone()
    t2s_limited[adaptive_mask == 1] = 0
    s0_limited[adaptive_mask == 1] = 0

    valid = t2s_limited[t2s_limited > 0]
    if valid.numel() > 0:
        cap = torch.quantile(valid.to(torch.float32), 0.995, interpolation="lower")
        t2s_limited[t2s_limited > cap * 10] = cap
    return t2s, s0, t2s_limited, s0_limited


# ---------------------------------------------------------------------------
# Optimal combination (port of tedana.combine.make_optcom)
# ---------------------------------------------------------------------------


def make_optcom(
    data_cat: Tensor,
    tes: Tensor,
    adaptive_mask: Tensor,
    t2s: Tensor | None = None,
    combmode: str = "t2s",
) -> Tensor:
    """Optimally combine echoes into a single timeseries.

    ``t2s`` weighting (Posse 1999): ``w_e = TE_e * exp(-TE_e / T2*)``.
    ``paid`` weighting (Poser 2006): ``w_e = (mean/std)_e * TE_e``.

    Args:
        data_cat: ``(V, E, T)`` masked multi-echo data.
        tes: ``(E,)`` echo times (ms).
        adaptive_mask: ``(V,)`` good-echo counts (zeros out unused echoes).
        t2s: ``(V,)`` or ``(V, T)`` T2* estimates; required for ``combmode="t2s"``.
        combmode: ``"t2s"`` or ``"paid"``.

    Returns:
        ``(V, T)`` combined data.
    """
    avail = availability_weights(adaptive_mask, data_cat.shape[1])  # (V, E)
    if combmode == "paid":
        mean_sig = data_cat.mean(dim=-1)
        std_sig = data_cat.std(dim=-1)
        snr = torch.where(std_sig != 0, mean_sig / std_sig, torch.zeros_like(mean_sig))
        alpha = snr * tes.unsqueeze(0)  # (V, E)
    elif combmode == "t2s":
        if t2s is None:
            raise ValueError("combmode='t2s' requires a t2s map")
        if t2s.ndim == 2:
            t2s_e = t2s.unsqueeze(1)  # (V, 1, T)
            alpha = tes.view(1, -1, 1) * torch.exp(-tes.view(1, -1, 1) / t2s_e)  # (V, E, T)
            alpha = alpha * avail.unsqueeze(-1)
            norm = alpha.sum(dim=1, keepdim=True).clamp_min(torch.finfo(alpha.dtype).eps)
            return (data_cat * alpha).sum(dim=1) / norm.squeeze(1)
        alpha = tes.unsqueeze(0) * torch.exp(-tes.unsqueeze(0) / t2s.unsqueeze(1))  # (V, E)
    else:
        raise ValueError(f"Unknown combmode: {combmode!r}")

    alpha = alpha * avail
    norm = alpha.sum(dim=1, keepdim=True).clamp_min(torch.finfo(alpha.dtype).eps)
    return (data_cat * alpha.unsqueeze(-1)).sum(dim=1) / norm


# ---------------------------------------------------------------------------
# Model-fit RMSE
# ---------------------------------------------------------------------------


def rmse_of_fit(
    data_cat: Tensor,
    tes: Tensor,
    adaptive_mask: Tensor,
    t2s: Tensor,
    s0: Tensor,
) -> Tensor:
    """Per-voxel RMSE of the decay fit, averaged across time.

    Args:
        data_cat: ``(V, E, T)`` data.
        tes: ``(E,)`` echo times (ms).
        adaptive_mask: ``(V,)`` good-echo counts.
        t2s, s0: ``(V,)`` or ``(V, T)`` estimates.

    Returns:
        ``(V,)`` RMSE map (NaN where no good echoes).
    """
    avail = availability_weights(adaptive_mask, data_cat.shape[1])  # (V, E)
    if t2s.ndim == 1:
        pred = (s0.unsqueeze(1) * torch.exp(-tes.unsqueeze(0) / t2s.unsqueeze(1))).unsqueeze(-1)
    else:
        pred = s0.unsqueeze(1) * torch.exp(-tes.view(1, -1, 1) / t2s.unsqueeze(1))
    resid = (data_cat - pred) * avail.unsqueeze(-1)
    n_good = avail.sum(dim=1).clamp_min(1).unsqueeze(-1)
    rmse_t = torch.sqrt((resid * resid).sum(dim=1) / n_good)  # (V, T)
    rmse = rmse_t.mean(dim=1)
    rmse[avail.sum(dim=1) == 0] = float("nan")
    return rmse


# ---------------------------------------------------------------------------
# Leave-one-echo-out QC and robust fitting
# ---------------------------------------------------------------------------


def leave_one_echo_out(
    mean_echo: Tensor,
    tes: Tensor,
    avail: Tensor,
    fittype: str = "loglin",
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Jackknife each echo: fit with the others, predict the held-out echo.

    For each echo ``k``, refit the decay using the remaining good echoes and predict
    echo ``k`` from its echo time. The residual ``predicted - actual`` flags echoes that
    disagree with the monoexponential model (dropout, ghosting, spikes). All E folds and
    all voxels are fit in one batched call.

    Args:
        mean_echo: ``(V, E)`` per-echo temporal mean.
        tes: ``(E,)`` echo times (ms).
        avail: ``(V, E)`` availability weights.
        fittype: ``"loglin"`` or ``"curvefit"`` for the fold fits.
        device: compute device.

    Returns:
        ``(resid, resid_frac)`` each ``(V, E)``. ``resid`` is predicted-minus-actual on
        the held-out echo; ``resid_frac`` divides by the actual echo magnitude (a
        fractional error). Both are NaN where a fold has fewer than 2 remaining echoes.
    """
    device = device or mean_echo.device
    v, e = mean_echo.shape
    eye = torch.eye(e, dtype=torch.bool, device=device)  # (E, E): fold k drops echo k

    # Tile to (E*V, E): fold-major.
    y = mean_echo.unsqueeze(0).expand(e, v, e).reshape(e * v, e)
    w = avail.unsqueeze(0).expand(e, v, e).clone()
    w[eye.unsqueeze(1).expand(e, v, e)] = 0.0  # zero the held-out echo per fold
    w = w.reshape(e * v, e)

    t2s_init, s0_init = fit_loglinear(y, tes, w)
    if fittype == "curvefit":
        t2s, s0, *_ = fit_curvefit_gpu(y, tes, s0_init, t2s_init, weights=w, device=device)
    else:
        t2s, s0 = t2s_init, s0_init
    t2s = t2s.reshape(e, v)
    s0 = s0.reshape(e, v)

    # Prediction at the held-out echo: fold k uses te_k.
    te_held = tes.view(e, 1)  # (E, 1) -> fold k
    pred_held = (s0 * torch.exp(-te_held / t2s)).T  # (V, E)
    resid = pred_held - mean_echo

    # A fold is only estimable if >= 2 echoes remain good after dropping the held one.
    good = (avail > 0).to(torch.int64)
    remaining = good.sum(dim=1, keepdim=True) - good  # (V, E) good echoes other than this one
    estimable = (avail > 0) & (remaining >= 2)
    resid = torch.where(estimable, resid, torch.full_like(resid, float("nan")))
    resid_frac = resid / (mean_echo.abs() + torch.finfo(mean_echo.dtype).eps)
    return resid, resid_frac


def robustness_weights(resid: Tensor, c: float = 4.685) -> Tensor:
    """Tukey biweight per-echo weights from LOEO residuals.

    Scale is a robust per-voxel MAD across echoes. Non-estimable echoes (NaN residual)
    keep weight 1 so they are not spuriously dropped.

    Args:
        resid: ``(V, E)`` leave-one-echo-out residuals.
        c: Tukey tuning constant (4.685 gives ~95% efficiency at the Gaussian).

    Returns:
        ``(V, E)`` weights in ``[0, 1]``.
    """
    finite = torch.isfinite(resid)
    r = torch.where(finite, resid, torch.zeros_like(resid))
    med = torch.nanmedian(
        torch.where(finite, resid.abs(), torch.full_like(resid, float("nan"))), dim=1, keepdim=True
    ).values
    sigma = (1.4826 * med).clamp_min(torch.finfo(resid.dtype).eps)
    u = r / (c * sigma)
    w = torch.where(u.abs() < 1.0, (1.0 - u * u) ** 2, torch.zeros_like(u))
    # Echoes without a residual estimate are left untouched.
    return torch.where(finite, w, torch.ones_like(w))


def fit_robust(
    data_cat: Tensor,
    tes: Tensor,
    adaptive_mask: Tensor,
    fitmode: str = "all",
    fit_all_timepoints: bool = False,
    n_irls: int = 2,
    device: torch.device | None = None,
    verbose: bool = False,
) -> dict[str, Tensor]:
    """Robust decay fit: iteratively down-weight echoes that fail the LOEO check.

    Derives per-echo Tukey-biweight weights from leave-one-echo-out residuals on the
    per-echo temporal mean, then refits with those weights folded into the availability
    mask. Repeats a couple of times (IRLS). The per-echo weight map is returned as a
    voxel-wise diagnostic.

    Returns:
        Same dict as :func:`fit_decay` (curvefit branch) plus ``echo_weight`` ``(V, E)``.
    """
    device = device or data_cat.device
    avail = availability_weights(adaptive_mask, data_cat.shape[1])
    mean_echo = data_cat.mean(dim=2)
    weight = torch.ones_like(avail)

    for _ in range(max(1, n_irls)):
        resid, _ = leave_one_echo_out(
            mean_echo, tes, avail * weight, fittype="curvefit", device=device
        )
        weight = robustness_weights(resid)

    out = fit_decay(
        data_cat,
        tes,
        adaptive_mask,
        fittype="curvefit",
        fitmode=fitmode,
        fit_all_timepoints=fit_all_timepoints,
        weights=weight,
        device=device,
        verbose=verbose,
    )
    out["echo_weight"] = weight
    return out
