"""Variational Laplace vectorised over many datasets that share one model.

:func:`~fastfuncstuff.laminar.inversion.variational_laplace_lockstep` shares the
integration across fits but runs each fit's arithmetic in its own coroutine, so
past a few hundred fits the per-fit Python of the E/M-steps is the bill. Here the
scheme itself is batched: one model (one prior, so one set of free parameters)
inverted against ``B`` datasets -- cortical columns, parcels, bootstrap draws --
with every step a batched tensor operation and every branch (accept/retreat, the
M-step's early exit, convergence) a per-fit mask.

The arithmetic is :func:`_vl_coroutine`'s, step for step, so each fit follows the
serial trajectory; batched linear algebra reorders sums, so agreement is to
rounding, not to the bit (pinned by ``tests/test_laminar_batched.py``). Models with
different free parameters cannot share a batch without changing their numbers
(padding the prior precision moves ``spm_logdet``), so a model space is one batch
per model.

Where it pays: the integration is ~1,800 sequential microsteps whatever the batch
width, and on a GPU the width is nearly free, so thousands of columns cost about
what one does. On a CPU the step turns compute-bound past ~128 rows and the
serial or lockstep paths are as good.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import torch

from fastfuncstuff.laminar.integrate import batch_bucket, integrate
from fastfuncstuff.laminar.inversion import (
    FINDIFF_STEP,
    Priors,
    VLResult,
    spm_logdet,
    unvec,
    vec,
)
from fastfuncstuff.laminar.params import ModelSpec

_TOL = 1e-16


def logdet_b(C: torch.Tensor) -> torch.Tensor:
    """:func:`~fastfuncstuff.laminar.inversion.spm_logdet` over a batch ``(B, n, n)``.

    Same branch per matrix: Cholesky when ``C`` passes ``torch.allclose(C, C.mT)``
    and is positive definite, else the sum of log singular values in
    ``[TOL, 1/TOL]``. A matrix with a zero diagonal (which the serial version
    drops rows for) falls back to the serial function.
    """
    zero_diag = (torch.diagonal(C, dim1=-2, dim2=-1) == 0).any(-1)
    sym = ((C - C.mT).abs() <= _TOL + 1e-5 * C.mT.abs()).all(-1).all(-1)
    chol, info = torch.linalg.cholesky_ex(C)
    use_chol = sym & (info == 0)
    ld_chol = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
    out = ld_chol
    if not bool(use_chol.all()):
        s = torch.linalg.svdvals(C)
        keep = (s > _TOL) & (s < 1.0 / _TOL)
        ld_svd = torch.where(keep, torch.log(torch.where(keep, s, 1.0)), 0.0).sum(-1)
        out = torch.where(use_chol, ld_chol, ld_svd)
    if bool(zero_diag.any()):
        out = out.clone()
        for i in torch.nonzero(zero_diag).flatten().tolist():
            out[i] = spm_logdet(C[i])
    return out


def inv_b(A: torch.Tensor) -> torch.Tensor:
    """:func:`~fastfuncstuff.laminar.inversion.spm_inv` over a batch: each matrix
    gets its own automatic ridge."""
    n = A.shape[-1]
    norm_inf = A.abs().sum(-1).amax(-1)
    eps = torch.finfo(A.dtype).eps * torch.clamp(norm_inf, min=1.0)
    tol = torch.clamp(eps * n, min=math.exp(-32))
    eye = torch.eye(n, dtype=A.dtype, device=A.device)
    return torch.linalg.inv(A + eye * tol[:, None, None])


def dx_b(dfdx: torch.Tensor, f: torch.Tensor, v: torch.Tensor | float) -> torch.Tensor:
    """:func:`~fastfuncstuff.laminar.inversion.spm_dx` over a batch; ``v`` per fit
    or shared."""
    B, n = f.shape
    v = torch.as_tensor(v, dtype=f.dtype, device=f.device).expand(B)
    t = torch.exp(v - logdet_b(dfdx) / n)
    big = t > math.exp(16)
    out = torch.empty_like(f)
    small = ~big
    if bool(small.any()):
        ts = t[small]
        aug = torch.zeros(int(small.sum()), n + 1, n + 1, dtype=f.dtype, device=f.device)
        aug[:, 1:, 0] = ts[:, None] * f[small]
        aug[:, 1:, 1:] = ts[:, None, None] * dfdx[small]
        out[small] = torch.linalg.matrix_exp(aug)[:, 1:, 0]
    if bool(big.any()):
        out[big] = -(torch.linalg.pinv(dfdx[big]) @ f[big][..., None])[..., 0]
    return out


@dataclass
class BatchedVL:
    """Posteriors of ``B`` inversions of one model; index with :meth:`result`."""

    Ep_vec: torch.Tensor  # noqa: N815  (B, n_params) posterior means, full structure
    Cp: torch.Tensor  # (B, np, np) in the reduced space
    Eh: torch.Tensor  # noqa: N815  (B, K)
    F: torch.Tensor  # (B,)
    L: torch.Tensor  # (B, 3)
    y_pred: torch.Tensor  # (B, n_scans, K), PSF applied
    free_idx: torch.Tensor
    n_iter: torch.Tensor  # (B,)
    converged: torch.Tensor  # (B,) bool
    stalled: torch.Tensor  # (B,) bool
    failed: torch.Tensor  # (B,) bool: the Jacobian diverged past four retreats
    spec: ModelSpec

    def result(self, i: int) -> VLResult:
        return VLResult(
            Ep=unvec(
                self.Ep_vec[i].cpu(), dataclasses.replace(self.spec, device=torch.device("cpu"))
            ),
            Cp=self.Cp[i].cpu(),
            Eh=self.Eh[i].cpu(),
            F=self.F[i].cpu(),
            L=self.L[i].cpu(),
            y_pred=self.y_pred[i].cpu(),
            free_idx=self.free_idx.cpu(),
            n_iter=int(self.n_iter[i]),
            converged=bool(self.converged[i]),
            stalled=bool(self.stalled[i]),
        )


def _replace_fits(res: BatchedVL, idx: torch.Tensor, sub: BatchedVL) -> BatchedVL:
    """``res`` with fits ``idx`` replaced by ``sub``'s."""
    out = dataclasses.replace(res)
    for f in dataclasses.fields(BatchedVL):
        if f.name in ("free_idx", "spec"):
            continue
        whole = getattr(res, f.name).clone()
        whole[idx] = getattr(sub, f.name).to(whole.device)
        setattr(out, f.name, whole)
    return out


def _psf_after(preds: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
    """Per-fit depth PSF on ``(B, P, n_scans, K)`` predictions: a ``(B, K, K)`` matrix
    per fit. Linear and applied per scan, so applying it after the integration is
    the same map ``g_obs`` applies inside it."""
    return torch.einsum("bpsk,bjk->bpsj", preds, kernels.to(preds.dtype))


def variational_laplace_batched(
    spec: ModelSpec,
    priors: Priors,
    u: torch.Tensor,
    Y: torch.Tensor,
    *,
    rows: torch.Tensor | None = None,
    kernel: torch.Tensor | None = None,
    max_iter: int = 128,
    hyper_from_rows: bool = False,
    device: torch.device | str | None = None,
    progress: bool = False,
    rescue: int = 0,
    start: tuple[torch.Tensor, torch.Tensor] | None = None,
    width: int | None = None,
    **integrate_kwargs,
) -> BatchedVL:
    """Invert one model against ``Y`` ``(B, n_scans, K)``, all fits at once.

    ``rescue`` restarts fits that stalled (ended on rejected steps) from their best
    point with the damping reset, up to that many times, as
    :func:`~fastfuncstuff.laminar.inversion.variational_laplace` does: a restarted
    fit takes the restarted run's result. ``start`` is ``(p, h)`` per fit (reduced
    parameters, log-precisions) and ``width`` the compiled batch width; both exist
    for the restarts.

    ``kernel`` is the depth PSF: ``None``, one kernel for every fit (1-D or
    ``(K, K)``, applied inside the integration exactly as the serial path does),
    or ``(B, K, K)``, one measured sampling matrix per fit. ``u`` and ``rows`` are
    shared: every dataset has the same design. ``device`` moves the whole
    inversion (default: ``spec.device``).
    """
    if rescue:
        kw = dict(rows=rows, max_iter=max_iter, hyper_from_rows=hyper_from_rows, device=device,
                  progress=progress, **integrate_kwargs)  # fmt: skip
        res = variational_laplace_batched(spec, priors, u, Y, kernel=kernel, **kw)
        pe_free = vec(priors.pE).to(res.Ep_vec)[res.free_idx]
        n_probe = int(res.free_idx.numel()) + 1
        for _ in range(int(rescue)):
            redo = torch.nonzero(res.stalled).flatten()
            if redo.numel() == 0:
                break
            per_fit = kernel is not None and kernel.ndim == 3
            sub = variational_laplace_batched(
                spec, priors, u, Y[redo.cpu()], kernel=kernel[redo.cpu()] if per_fit else kernel,
                start=(res.Ep_vec[redo][:, res.free_idx] - pe_free, res.Eh[redo]),
                width=batch_bucket(Y.shape[0] * n_probe), **kw,
            )  # fmt: skip
            res = _replace_fits(res, redo, sub)
        return res
    device = torch.device(device) if device is not None else spec.device
    spec = dataclasses.replace(spec, device=device)
    dtype = spec.dtype
    Y = Y.to(dtype=dtype, device=device)
    u = u.to(dtype=dtype, device=device)
    B, n_scans, K = Y.shape
    if K != spec.K:
        raise ValueError(f"data has {K} depths, model has {spec.K}")
    if rows is None:
        rows = torch.ones(n_scans, dtype=torch.bool)
    rows = rows.to(torch.bool).to(device)
    per_fit_kernel = kernel is not None and kernel.ndim == 3
    if per_fit_kernel:
        if kernel.shape != (B, K, K):
            raise ValueError(
                f"per-fit kernels must be (B, K, K) = {(B, K, K)}, got {tuple(kernel.shape)}"
            )
        kernel = kernel.to(dtype=dtype, device=device)
        shared_kernel = None
    else:
        shared_kernel = None if kernel is None else kernel.to(dtype=dtype, device=device)

    ns_kept = int(rows.sum())
    yv = Y[:, rows].mT.reshape(B, -1)  # depth-major, as masked_vec
    ny = yv.shape[1]

    pE_vec = vec(priors.pE).to(dtype=dtype, device=device)
    pC_vec = vec(priors.pC).to(dtype=dtype, device=device)
    free_idx = torch.nonzero(pC_vec > 0, as_tuple=False).squeeze(-1)
    np_ = free_idx.numel()
    if np_ == 0:
        raise ValueError("every parameter has zero prior variance; nothing to estimate")
    ipC = torch.diag(1.0 / pC_vec[free_idx])

    nh = K
    if priors.hE is None:
        var = yv.var(dim=1) if hyper_from_rows else Y.reshape(B, -1).var(dim=1)
        hE = (4.0 - torch.log(var))[:, None].expand(B, nh).clone()
    else:
        hE = priors.hE.to(dtype=dtype, device=device).expand(B, nh).clone()
    if priors.hC is None:
        ihC = torch.eye(nh, dtype=dtype, device=device) * math.exp(4)
    else:
        ihC = inv_b(priors.hC.to(dtype=dtype, device=device)[None])[0]

    width = width or batch_bucket(B * (np_ + 1))
    eye_np = torch.eye(np_, dtype=dtype, device=device) * FINDIFF_STEP

    def predict(p: torch.Tensor, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Jacobian and prediction at ``p`` ``(A, np)`` for fits ``idx``."""
        A = p.shape[0]
        probes = p[:, None, :].expand(A, np_ + 1, np_).clone()
        probes[:, 1:] += eye_np
        full = pE_vec.expand(A, np_ + 1, -1).clone()
        full[:, :, free_idx] += probes
        flat = full.reshape(A * (np_ + 1), -1)
        outs = []
        # One compiled width for the whole run (see variational_laplace_lockstep).
        for s in range(0, flat.shape[0], width):
            chunk = flat[s : s + width]
            n = chunk.shape[0]
            if n < width:
                chunk = torch.cat([chunk, chunk[-1:].expand(width - n, -1)], dim=0)
            pr = integrate(
                u, unvec(chunk, spec), spec, n_scans, kernel=shared_kernel, **integrate_kwargs
            )
            outs.append(pr[:n])
        preds = torch.cat(outs, dim=0).reshape(A, np_ + 1, n_scans, K)
        if per_fit_kernel:
            preds = _psf_after(preds, kernel[idx])
        mv = preds[:, :, rows].mT.reshape(A, np_ + 1, -1)  # (A, P, ny)
        dfdp = (mv[:, 1:] - mv[:, :1]).mT / FINDIFF_STEP  # (A, ny, np)
        return dfdp, preds[:, 0]

    p = torch.zeros(B, np_, dtype=dtype, device=device)
    h = hE.clone()
    if start is not None:  # a restart: the damping resets, the position does not
        p = start[0].to(dtype=dtype, device=device).clone()
        h = start[1].to(dtype=dtype, device=device).clone()
    v_rate = torch.full((B,), -4.0, dtype=dtype, device=device)
    best_F = torch.full((B,), -math.inf, dtype=dtype, device=device)
    best_p, best_h = p.clone(), h.clone()
    best_Cp = torch.eye(np_, dtype=dtype, device=device).expand(B, np_, np_).clone()
    dFdp = torch.zeros(B, np_, dtype=dtype, device=device)
    dFdpp = -torch.eye(np_, dtype=dtype, device=device).expand(B, np_, np_).clone()
    crit = torch.zeros(B, 4, dtype=torch.bool, device=device)
    L_all = torch.zeros(B, 3, dtype=dtype, device=device)
    y_pred_all = torch.zeros(B, n_scans, K, dtype=dtype, device=device)
    n_iter = torch.zeros(B, dtype=torch.long, device=device)
    converged = torch.zeros(B, dtype=torch.bool, device=device)
    failed = torch.zeros(B, dtype=torch.bool, device=device)
    rejected_tail = torch.zeros(B, dtype=torch.long, device=device)
    active = torch.ones(B, dtype=torch.bool, device=device)
    big = math.exp(32)

    bar = None
    if progress:
        from tqdm.auto import tqdm

        bar = tqdm(total=max_iter, desc=f"laminar batch x{B}", leave=True)

    for k in range(1, max_iter + 1):
        idx = torch.nonzero(active).flatten()
        if idx.numel() == 0:
            break
        n_iter[idx] = k
        dfdp, y_pred = predict(p[idx], idx)
        bad = ~torch.isfinite(dfdp).all(-1).all(-1) | (dfdp.abs().amax((-1, -2)) > big)
        if bool(bad.any()):
            # The reference retreats up to four times toward the best point.
            for _ in range(4):
                bi = idx[bad]
                v_rate[bi] = torch.clamp(v_rate[bi] - 2.0, max=-4.0)
                p[bi] = best_p[bi] + dx_b(dFdpp[bi], dFdp[bi], v_rate[bi])
                dj, yj = predict(p[bi], bi)
                dfdp[bad], y_pred[bad] = dj, yj
                bad = ~torch.isfinite(dfdp).all(-1).all(-1) | (dfdp.abs().amax((-1, -2)) > big)
                if not bool(bad.any()):
                    break
            if bool(bad.any()):
                # Serial raises; one column's failure must not sink the batch.
                failed[idx[bad]] = True
                active[idx[bad]] = False
                keep = ~bad
                idx, dfdp, y_pred = idx[keep], dfdp[keep], y_pred[keep]
                if idx.numel() == 0:
                    continue
        y_pred_all[idx] = y_pred
        A = idx.numel()
        e = yv[idx] - y_pred[:, rows].mT.reshape(A, -1)
        J = -dfdp
        pa, ha, hEa = p[idx], h[idx], hE[idx]

        # --- M-step ---------------------------------------------------------
        Jb = J.reshape(A, nh, ns_kept, np_)
        JtJ = Jb.mT @ Jb  # (A, nh, np, np)
        ee = (e.reshape(A, nh, ns_kept) ** 2).sum(-1)
        m_on = torch.ones(A, dtype=torch.bool, device=device)
        Pp = Cp = Ch = d = d_iS = None
        for _ in range(8):
            lam = math.exp(-32) + torch.exp(ha)
            d_iS_n = lam.repeat_interleave(ns_kept, dim=1)
            eps = torch.finfo(dtype).eps * torch.clamp(d_iS_n.amax(1), min=1.0)
            tol = torch.clamp(eps * ny, min=math.exp(-32))
            d_S = 1.0 / (d_iS_n + tol[:, None])
            Pp_n = (J.mT * d_iS_n[:, None, :]) @ J
            Cp_n = inv_b(Pp_n + ipC)
            eh = torch.exp(ha)
            tr_PS = eh * d_S.reshape(A, nh, ns_kept).sum(-1)
            tr_CJ = eh * (Cp_n.mT[:, None] * JtJ).sum((-1, -2))
            dFdh = tr_PS / 2 - eh * ee / 2 - tr_CJ / 2
            dFdhh_d = -(eh**2) * (d_S.reshape(A, nh, ns_kept) ** 2).sum(-1) / 2
            d_n = ha - hEa
            dFdh = dFdh - d_n @ ihC.mT
            dFdhh = torch.diag_embed(dFdhh_d) - ihC
            Ch_n = inv_b(-dFdhh)
            dh = torch.clamp(dx_b(dFdhh, dFdh, 4.0), -1.0, 1.0)
            if Pp is None:
                Pp, Cp, Ch, d, d_iS = Pp_n, Cp_n, Ch_n, d_n, d_iS_n
            else:
                m3 = m_on[:, None, None]
                Pp = torch.where(m3, Pp_n, Pp)
                Cp = torch.where(m3, Cp_n, Cp)
                Ch = torch.where(m3, Ch_n, Ch)
                d = torch.where(m_on[:, None], d_n, d)
                d_iS = torch.where(m_on[:, None], d_iS_n, d_iS)
            ha = torch.where(m_on[:, None], ha + dh, ha)
            m_on = m_on & ~((dFdh * dh).sum(-1) < 1e-2)
            if not bool(m_on.any()):
                break
        assert (
            Pp is not None
            and Cp is not None
            and Ch is not None
            and d is not None
            and d_iS is not None
        )

        # --- free energy ------------------------------------------------------
        L = torch.stack(
            [
                torch.log(d_iS).sum(1) / 2
                - (d_iS * e * e).sum(1) / 2
                - ny * math.log(2 * math.pi) / 2,
                logdet_b(ipC @ Cp) / 2 - ((pa @ ipC) * pa).sum(1) / 2,
                logdet_b(ihC @ Ch) / 2 - ((d @ ihC) * d).sum(1) / 2,
            ],
            dim=1,
        )
        F = L.sum(1)
        L_all[idx] = L

        acc = (F > best_F[idx]) | (k < 3)
        ai, ri = idx[acc], idx[~acc]
        best_F[ai] = F[acc]
        best_p[ai] = pa[acc]
        best_h[ai] = ha[acc]
        best_Cp[ai] = Cp[acc]
        dFdp[ai] = -(J[acc].mT @ (d_iS[acc] * e[acc])[..., None])[..., 0] - pa[acc] @ ipC.mT
        dFdpp[ai] = -Pp[acc] - ipC
        v_rate[ai] = torch.clamp(v_rate[ai] + 0.5, max=4.0)
        rejected_tail[ai] = 0
        pa = torch.where(acc[:, None], pa, best_p[idx])
        ha = torch.where(acc[:, None], ha, best_h[idx])
        v_rate[ri] = torch.clamp(v_rate[ri] - 2.0, max=-4.0)
        rejected_tail[ri] += 1

        dp = dx_b(dFdpp[idx], dFdp[idx], v_rate[idx])
        p[idx] = pa + dp
        h[idx] = ha
        dF_pred = (dFdp[idx] * dp).sum(1)
        crit[idx] = torch.cat([(dF_pred < 1e-1)[:, None], crit[idx, :-1]], dim=1)
        done = crit[idx].all(1)
        converged[idx[done]] = True
        active[idx[done]] = False
        if bar is not None:
            bar.update(1)
            bar.set_postfix(active=int(active.sum()))
    if bar is not None:
        bar.close()

    Ep_vec = pE_vec.expand(B, -1).clone()
    Ep_vec[:, free_idx] += best_p
    return BatchedVL(
        Ep_vec=Ep_vec,
        Cp=best_Cp,
        Eh=best_h,
        F=best_F,
        L=L_all,
        y_pred=y_pred_all,
        free_idx=free_idx,
        n_iter=n_iter,
        converged=converged,
        stalled=converged & (rejected_tail >= 3),
        failed=failed,
        spec=spec,
    )


@dataclass
class BatchedBPA:
    """One model's posteriors averaged across K, per dataset (``spm_dcm_bpa``)."""

    Ep_vec: torch.Tensor  # noqa: N815  (B, n_params)
    Vp_vec: torch.Tensor  # noqa: N815  (B, n_params) posterior variance, 0 where fixed
    Pp_vec: torch.Tensor  # noqa: N815  (B, n_params) P(parameter differs from 0)
    F: torch.Tensor  # (B,)
    per_k_F: torch.Tensor  # noqa: N815  (B, n_K)


def bpa_batched(runs: list[BatchedVL], *, free_energy: str = "reference") -> BatchedBPA:
    """:func:`~fastfuncstuff.laminar.experiment.bayesian_parameter_average` for
    ``B`` datasets at once; ``runs`` is one :class:`BatchedVL` per K."""
    if not runs:
        raise ValueError("no inversions to average")
    free_idx = runs[0].free_idx
    for r in runs[1:]:
        if not torch.equal(r.free_idx, free_idx):
            raise ValueError("inversions must share a parameter structure to be averaged")
    precisions = torch.stack([inv_b(r.Cp) for r in runs])  # (nK, B, np, np)
    Cp = inv_b(precisions.sum(0))
    reduced = torch.stack([r.Ep_vec[:, free_idx] for r in runs])  # (nK, B, np)
    Ep_red = (Cp @ (precisions @ reduced[..., None]).sum(0))[..., 0]
    Ep = torch.stack([r.Ep_vec for r in runs]).mean(0)
    Ep[:, free_idx] = Ep_red
    Vp = torch.zeros_like(Ep)
    Vp[:, free_idx] = torch.diagonal(Cp, dim1=-2, dim2=-1)
    sd = torch.sqrt(torch.clamp(Vp, min=0))
    z = torch.where(sd > 0, Ep.abs() / torch.clamp(sd, min=1e-300), torch.zeros_like(sd))
    Pp = torch.where(sd > 0, 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0))), torch.zeros_like(sd))
    per_k = torch.stack([r.F for r in runs], dim=1)
    if free_energy == "reference":
        F = per_k[:, 0]
    elif free_energy == "sum":
        F = per_k.sum(1)
    elif free_energy == "mean":
        F = per_k.mean(1)
    else:
        raise ValueError(f"free_energy must be 'reference', 'sum' or 'mean', got {free_energy!r}")
    return BatchedBPA(Ep, Vp, Pp, F, per_k)


_LOG2PI = math.log(2.0 * math.pi)


def predict_batched(
    spec: ModelSpec,
    Ep_vec: torch.Tensor,
    u: torch.Tensor,
    n_scans: int,
    *,
    kernel: torch.Tensor | None = None,
    **integrate_kwargs,
) -> torch.Tensor:
    """Predicted BOLD ``(B, n_scans, K)`` at parameter vectors ``Ep_vec`` (one forward
    pass, PSF as in :func:`variational_laplace_batched`)."""
    dev = Ep_vec.device
    spec = dataclasses.replace(spec, device=dev)
    B = Ep_vec.shape[0]
    width = batch_bucket(B)
    rows = Ep_vec if width == B else torch.cat([Ep_vec, Ep_vec[-1:].expand(width - B, -1)])
    per_fit = kernel is not None and kernel.ndim == 3
    shared = None if (kernel is None or per_fit) else kernel.to(dev, Ep_vec.dtype)
    pred = integrate(u.to(dev, Ep_vec.dtype), unvec(rows, spec), spec, n_scans, kernel=shared,
                     **integrate_kwargs)[:B]  # fmt: skip
    if per_fit:
        pred = _psf_after(pred[:, None], kernel.to(dev))[:, 0]
    return pred


def reduce_batched(
    full: BatchedVL,
    full_priors: Priors,
    reduced_priors: Priors,
) -> BatchedVL:
    """Bayesian model reduction: what inverting under ``reduced_priors`` would give,
    from the full model's posterior alone ([[Friston & Penny 2011]]).

    The free energy and posterior of a model whose prior differs from the full
    model's -- here a modulation B pinned at zero at some depths, or a tighter
    prior on a vascular parameter -- follow from the full posterior in closed form
    under the Laplace assumption, with the likelihood (and the noise estimate) held
    at the full model's. In two steps:

    1. parameters the reduced prior pins (variance 0) are conditioned on their
       pinned value: the evidence changes by log q(θ_s = r_s) − log p(θ_s = r_s),
       the Savage–Dickey ratio, and the rest take the Gaussian conditional;
    2. a changed prior on the parameters that stay free is the general reduction
       of ``spm_log_evidence_reduce``.

    Because every nested model comes from the same posterior, nested evidences are
    consistent by construction: there is no separate optimisation to stall in.
    """
    dev, dtype = full.Ep_vec.device, full.Ep_vec.dtype
    free = full.free_idx.to(dev)
    pE = vec(full_priors.pE).to(dev, dtype)[free]
    pC = vec(full_priors.pC).to(dev, dtype)[free]
    rE_all = vec(reduced_priors.pE).to(dev, dtype)
    rC_all = vec(reduced_priors.pC).to(dev, dtype)
    outside = torch.ones_like(rC_all, dtype=torch.bool)
    outside[free] = False
    if bool((rC_all[outside] > 0).any()):
        raise ValueError("the reduced model frees a parameter the full model fixed")
    rE, rC = rE_all[free], rC_all[free]
    s = rC <= 0
    k = ~s
    q = full.Ep_vec[:, free]  # absolute posterior means
    C = full.Cp
    B = q.shape[0]
    dF = torch.zeros(B, dtype=dtype, device=dev)
    qk, Ck = q[:, k], C[:, k][:, :, k]
    if bool(s.any()):
        Css = C[:, s][:, :, s]
        Csk = C[:, s][:, :, k]
        d = q[:, s] - rE[s]
        L = torch.linalg.cholesky(Css)
        alpha = torch.cholesky_solve(d[..., None], L)[..., 0]
        logdet_ss = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
        ns = int(s.sum())
        log_q = -0.5 * ((d * alpha).sum(-1) + logdet_ss + ns * _LOG2PI)
        dp = rE[s] - pE[s]
        log_p = -0.5 * ((dp * dp / pC[s]).sum() + torch.log(pC[s]).sum() + ns * _LOG2PI)
        dF = dF + log_q - log_p
        qk = qk - (Csk.mT @ alpha[..., None])[..., 0]
        Ck = Ck - Csk.mT @ torch.cholesky_solve(Csk, L)
    pEk, pCk, rEk, rCk = pE[k], pC[k], rE[k], rC[k]
    if not (torch.equal(pEk, rEk) and torch.equal(pCk, rCk)):
        qP = inv_b(Ck)
        pP = torch.diag(1.0 / pCk)
        rP = torch.diag(1.0 / rCk)
        sP = qP + rP - pP
        sC = inv_b(sP)
        sE = (sC @ ((qP @ qk[..., None])[..., 0] + rP @ rEk - pP @ pEk)[..., None])[..., 0]
        quad = (
            (qk * (qP @ qk[..., None])[..., 0]).sum(-1)
            + rEk @ rP @ rEk
            - pEk @ pP @ pEk
            - (sE * (sP @ sE[..., None])[..., 0]).sum(-1)
        )
        dF = dF + 0.5 * (logdet_b(qP) + torch.log(1.0 / rCk).sum() - logdet_b(sP)
                         - torch.log(1.0 / pCk).sum()) - 0.5 * quad  # fmt: skip
        qk, Ck = sE, sC
    Ep = full.Ep_vec.clone()
    Ep[:, free[s]] = rE[s]
    Ep[:, free[k]] = qk
    return dataclasses.replace(
        full, Ep_vec=Ep, Cp=Ck, F=full.F + dF, free_idx=full.free_idx[k.cpu()]
    )


__all__ = [
    "BatchedBPA",
    "BatchedVL",
    "bpa_batched",
    "reduce_batched",
    "dx_b",
    "inv_b",
    "logdet_b",
    "predict_batched",
    "variational_laplace_batched",
]
