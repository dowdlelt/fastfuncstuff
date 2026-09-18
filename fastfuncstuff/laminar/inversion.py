"""
Variational Laplace inversion of the laminar generative model.

A port of SPM's ``spm_nlsi_GN.m`` as modified by Faes et al. (the
``_laminar_mask`` variant), which differs from stock SPM in one respect: only
the masked data points enter the expectation-maximisation, so the white-noise
padding inserted between two concatenated conditions is fitted by nothing.

Free energy is the deliverable, not the fit. The scientific claim these models
support is a *ranking* -- which cortical depth carries the modulatory effect --
and a port that reproduced predicted BOLD but not F would not have reproduced
the science. See ``../fmri_wiki/sources/Friston 2007.md``.

Scheme, per iteration:

1. **E-step.** Linearise the predicted response about the current parameter
   estimate, by forward differences in the reduced parameter space.
2. **M-step.** Eight Fisher-scoring steps on the log-precision hyperparameters,
   one per depth.
3. **Free energy.** Accuracy + parameter complexity + hyperparameter complexity.
4. **Accept or retreat.** F improved (or we are still in the first three
   iterations) -> take a Gauss-Newton step and relax the regularisation;
   otherwise roll back and tighten it.

References
----------
Friston K et al. (2007). Variational free energy and the Laplace approximation.
    NeuroImage 34(1):220-234.
Faes LK et al. (2026). Nat Commun. doi:10.1038/s41467-026-73540-z
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from fastfuncstuff.laminar.integrate import batch_bucket, integrate
from fastfuncstuff.laminar.params import PARAM_SHAPES, ModelSpec, zero_params

#: Deterministic flatten order. Only needs to be self-consistent -- the free
#: energy is invariant to how the reduced parameter space is coordinatised.
PARAM_ORDER: tuple[str, ...] = tuple(PARAM_SHAPES)

#: SPM's finite-difference step (``spm_diff``). Part of the specification, not a
#: tuning knob: the Jacobian feeds the posterior covariance and hence F.
FINDIFF_STEP = math.exp(-8)


def vec(P: dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten a parameter dict to ``(..., n_params)``."""
    parts = []
    for name in PARAM_ORDER:
        t = P[name]
        batch = t.shape[: t.ndim - _trailing_ndim(name)]
        parts.append(t.reshape(*batch, -1))
    return torch.cat(parts, dim=-1)


def unvec(v: torch.Tensor, spec: ModelSpec) -> dict[str, torch.Tensor]:
    """Inverse of :func:`vec`, preserving leading batch dimensions."""
    template = zero_params(spec)
    out: dict[str, torch.Tensor] = {}
    i = 0
    batch = v.shape[:-1]
    for name in PARAM_ORDER:
        shape = template[name].shape
        n = int(torch.tensor(shape).prod()) if len(shape) else 1
        out[name] = v[..., i : i + n].reshape(tuple(batch) + tuple(shape))
        i += n
    if i != v.shape[-1]:
        raise ValueError(f"parameter vector has {v.shape[-1]} entries, expected {i}")
    return out


def _trailing_ndim(name: str) -> int:
    return {"scalar": 0, "N": 1, "K": 1, "NxN": 2, "NxNxM": 3, "NxU": 2, "NxM": 2, "1xM": 2}[
        PARAM_SHAPES[name]
    ]


# --------------------------------------------------------------------------
# SPM numerical primitives. Reimplemented rather than approximated: each one
# has behaviour on singular input that the stock library functions do not.
# --------------------------------------------------------------------------


def spm_logdet(C: torch.Tensor) -> torch.Tensor:
    """Log-determinant that tolerates rank deficiency.

    Drops rows/columns with a zero diagonal, then sums the logs of the singular
    values inside ``[TOL, 1/TOL]``. Plain ``logdet`` returns ``-inf`` on exactly
    the matrices this scheme hands it.
    """
    TOL = 1e-16
    d = torch.diagonal(C)
    keep = d != 0
    if not bool(keep.any()):
        return torch.zeros((), dtype=C.dtype, device=C.device)
    C = C[keep][:, keep]
    if torch.allclose(C, C.mT, atol=TOL):
        chol, info = torch.linalg.cholesky_ex(C)
        if int(info) == 0:
            return 2.0 * torch.log(torch.diagonal(chol)).sum()
    s = torch.linalg.svdvals(C)
    s = s[(s > TOL) & (s < 1.0 / TOL)]
    return torch.log(s).sum()


def spm_inv(A: torch.Tensor) -> torch.Tensor:
    """Inverse with SPM's automatic ridge, so singular input is survivable."""
    n = max(A.shape)
    norm_inf = A.abs().sum(dim=-1).max()
    eps = torch.finfo(A.dtype).eps * torch.clamp(norm_inf, min=1.0)
    tol = torch.clamp(eps * n, min=math.exp(-32))
    return torch.linalg.inv(A + torch.eye(A.shape[0], dtype=A.dtype, device=A.device) * tol)


def spm_dx(dfdx: torch.Tensor, f: torch.Tensor, v: float) -> torch.Tensor:
    """Regularised ascent step ``(expm(t J) - I) J^-1 f``.

    ``t = exp(v - logdet(J)/n)`` is the integration time: large ``t`` gives the
    full Gauss-Newton step, small ``t`` a short gradient step. The scheme moves
    ``v`` up on success and down on failure, which is how it retreats out of a
    bad linearisation without a separate line search.

    Evaluated by exponentiating the augmented matrix ``[[0, 0], [t f, t J]]``
    and taking its first column -- the standard trick that avoids inverting a
    ``J`` that may be singular.
    """
    n = f.shape[0]
    t = torch.exp(torch.as_tensor(v, dtype=f.dtype, device=f.device) - spm_logdet(dfdx) / n)
    if float(t) > math.exp(16):
        return -torch.linalg.pinv(dfdx) @ f
    aug = torch.zeros(n + 1, n + 1, dtype=f.dtype, device=f.device)
    aug[1:, 0] = t * f
    aug[1:, 1:] = t * dfdx
    return torch.linalg.matrix_exp(aug)[1:, 0]


def _trace_prod(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """``trace(A @ B)`` without forming the product (SPM's ``spm_trace``)."""
    return (A.mT * B).sum()


# --------------------------------------------------------------------------


@dataclass
class Priors:
    """Prior means and variances over the estimated log-deviations.

    A *model* in this framework is a pattern of prior variances, not a different
    set of equations: setting ``pC["B"]`` to zero everywhere but the superficial
    diagonal entry is the "modulation targets superficial layers" hypothesis.
    Parameters with zero prior variance are projected out entirely, so the model
    comparison is over genuinely different parameter counts.
    """

    # SPM's names, kept verbatim so the port reads against spm_nlsi_GN.m.
    pE: dict[str, torch.Tensor]  # noqa: N815
    pC: dict[str, torch.Tensor]  # noqa: N815
    hE: torch.Tensor | None = None  # noqa: N815
    hC: torch.Tensor | None = None  # noqa: N815


@dataclass
class VLResult:
    """Posterior of one inversion."""

    Ep: dict[str, torch.Tensor]  # posterior means (full parameter structure)
    Cp: torch.Tensor  # posterior covariance in the reduced space
    Eh: torch.Tensor  # posterior log-precisions, one per depth
    F: torch.Tensor  # free energy: the log-evidence bound
    L: torch.Tensor  # its three terms (accuracy, param, hyperparam)
    y_pred: torch.Tensor  # predicted response, (n_scans, K)
    free_idx: torch.Tensor  # which flat parameters were estimated
    n_iter: int
    converged: bool
    history: list[float] = field(default_factory=list)

    def posterior_variance(self, spec: ModelSpec) -> dict[str, torch.Tensor]:
        """Posterior variances scattered back into the parameter structure."""
        full = torch.zeros(vec(zero_params(spec)).shape[-1], dtype=self.Cp.dtype)
        full[self.free_idx] = torch.diagonal(self.Cp)
        return unvec(full, spec)


def masked_vec(y: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Flatten the retained time points depth-major, as ``spm_vec`` does.

    The reference's mask is a ``(n_kept, K)`` array of column-major linear
    indices, so the resulting vector runs all kept time points of depth 0, then
    depth 1, and so on. The noise-precision components below assume that
    blocking, one hyperparameter per contiguous block.
    """
    return y[rows].mT.reshape(-1)


def _vl_coroutine(
    spec: ModelSpec,
    priors: Priors,
    y: torch.Tensor,
    *,
    rows: torch.Tensor | None = None,
    max_iter: int = 128,
    verbose: bool = False,
):
    """The Variational Laplace scheme, with the forward model lifted out.

    A generator: it yields a ``(n_probe, n_params)`` matrix of parameter vectors
    it needs predictions for, and receives the corresponding
    ``(n_probe, n_scans, K)`` predictions back via ``send``. It never calls
    :func:`~fastfuncstuff.laminar.integrate.integrate` itself.

    That inversion of control is the whole point. Integrating is ~99% of the
    cost and is host-bound, so one call carrying many fits' probes costs little
    more than one carrying a single fit's. Driving several of these coroutines
    in lockstep -- :func:`variational_laplace_lockstep` -- shares that call
    across fits while leaving each one's arithmetic bit-identical to running it
    alone, because nothing here changes.
    """
    n_scans, K = y.shape
    if K != spec.K:
        raise ValueError(f"data has {K} depths, model has {spec.K}")
    if rows is None:
        rows = torch.ones(n_scans, dtype=torch.bool)
    rows = rows.to(torch.bool)

    dtype, device = spec.dtype, spec.device
    y = y.to(dtype=dtype, device=device)
    yv = masked_vec(y, rows)
    ny = yv.numel()  # masked response variables
    ns_kept = int(rows.sum())

    # --- reduce to the parameters that have prior variance -----------------
    pE_vec = vec(priors.pE)
    pC_vec = vec(priors.pC)
    free_idx = torch.nonzero(pC_vec > 0, as_tuple=False).squeeze(-1)
    np_ = free_idx.numel()
    if np_ == 0:
        raise ValueError("every parameter has zero prior variance; nothing to estimate")
    ipC = torch.diag(1.0 / pC_vec[free_idx])

    # --- noise model: one log-precision per depth --------------------------
    nh = K
    if priors.hE is None:
        # SPM's default: centre the precision on the data's own variance.
        hE = torch.full((nh,), 4.0 - float(torch.log(y.var())), dtype=dtype, device=device)
    else:
        hE = priors.hE.to(dtype=dtype, device=device).expand(nh).clone()
    ihC = (
        torch.eye(nh, dtype=dtype, device=device) * math.exp(4)
        if priors.hC is None
        else spm_inv(priors.hC.to(dtype=dtype, device=device))
    )
    # SPM writes the error precision as sum_i exp(h_i) Q_i, with Q[i] the
    # identity on depth i's block of the (depth-major) response vector. Forming
    # those as dense (nh, ny, ny) tensors and multiplying through costs ny^3 per
    # term; exploiting the structure costs ns_kept^2. Nothing here is an
    # approximation -- the structure is exact:
    #
    #   * iS is *diagonal*: exp(h_i) repeated ns_kept times. So is S.
    #   * PS[i] is supported on block i alone, so trace(PS[i] PS[j]) vanishes
    #     for i != j and dFdhh is diagonal.
    #   * J' Pi[i] J touches only block i's rows of J.
    #
    # At K=9 this replaces 8 x 9 x 126^3 flops per iteration with 8 x 9 x 14^2.

    def probe_matrix(p: torch.Tensor) -> torch.Tensor:
        """Parameter vectors whose predictions the next step needs.

        ``spm_diff`` walks the parameters one at a time; here the unperturbed
        point and all ``np`` perturbations are one batch. Row 0 is the current
        point, rows 1..np are its forward differences.

        Returned unpadded: the *driver* pads to a bucketed width, because in
        lockstep it is the combined width across fits that has to hit a bucket,
        not each fit's separately.
        """
        probes = p.expand(np_ + 1, -1).clone()
        probes[1:] += torch.eye(np_, dtype=dtype, device=device) * FINDIFF_STEP
        full = pE_vec.expand(np_ + 1, -1).clone()
        full[:, free_idx] += probes
        return full

    def finish_linearise(preds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward-difference Jacobian from the predictions just received."""
        f0 = preds[0]
        dfdp = torch.stack(
            [masked_vec(preds[j + 1], rows) - masked_vec(f0, rows) for j in range(np_)], dim=1
        )
        return dfdp / FINDIFF_STEP, f0

    p = torch.zeros(np_, dtype=dtype, device=device)
    h = hE.clone()
    v_rate = -4.0
    best = {
        "F": torch.tensor(-math.inf, dtype=dtype, device=device),
        "p": p.clone(),
        "h": h.clone(),
        "Cp": torch.eye(np_, dtype=dtype, device=device),
    }
    dFdp = torch.zeros(np_, dtype=dtype, device=device)
    dFdpp = -torch.eye(np_, dtype=dtype, device=device)
    criterion = [False] * 4
    history: list[float] = []
    converged = False
    L = torch.zeros(3, dtype=dtype, device=device)
    y_pred = torch.zeros(n_scans, K, dtype=dtype, device=device)
    Cp = best["Cp"]
    k = 0

    for k in range(1, max_iter + 1):
        preds = yield probe_matrix(p)
        dfdp, y_pred = finish_linearise(preds)
        if not torch.isfinite(dfdp).all() or float(dfdp.abs().max()) > math.exp(32):
            # The reference retreats up to four times here before giving up. We
            # do the same rather than erroring: a diverged linearisation early
            # on is normal, and the retreat is what recovers from it.
            recovered = False
            for _ in range(4):
                v_rate = min(v_rate - 2.0, -4.0)
                p = best["p"] + spm_dx(dFdpp, dFdp, v_rate)
                preds = yield probe_matrix(p)
                dfdp, y_pred = finish_linearise(preds)
                if torch.isfinite(dfdp).all() and float(dfdp.abs().max()) <= math.exp(32):
                    recovered = True
                    break
            if not recovered:
                raise RuntimeError(
                    "laminar VL: convergence failure -- the Jacobian diverged and four "
                    "regularisation retreats did not recover it"
                )

        e = yv - masked_vec(y_pred, rows)
        J = -dfdp

        # --- M-step: Fisher scoring on the log-precisions ------------------
        # Depth-major, so J and e split into nh contiguous blocks of ns_kept.
        Jb = J.reshape(nh, ns_kept, np_)
        eb = e.reshape(nh, ns_kept)
        for _ in range(8):
            lam = math.exp(-32) + torch.exp(h)  # (nh,) precision per depth
            d_iS = lam.repeat_interleave(ns_kept)  # diagonal of iS
            # spm_inv's ridge, on what is here a diagonal matrix.
            eps = torch.finfo(dtype).eps * torch.clamp(d_iS.max(), min=1.0)
            tol = torch.clamp(eps * ny, min=math.exp(-32))
            d_S = 1.0 / (d_iS + tol)

            Pp = (J.mT * d_iS) @ J
            Cp = spm_inv(Pp + ipC)

            eh = torch.exp(h)
            # PS[i] lives on block i with value eh_i * d_S there.
            tr_PS = eh * d_S.reshape(nh, ns_kept).sum(1)
            JPJ = eh[:, None, None] * (Jb.mT @ Jb)

            dFdh = torch.stack(
                [
                    tr_PS[i] / 2 - (eh[i] * (eb[i] * eb[i]).sum()) / 2 - _trace_prod(Cp, JPJ[i]) / 2
                    for i in range(nh)
                ]
            )
            # trace(PS[i] PS[j]) = 0 for i != j: disjoint blocks.
            dFdhh = torch.diag(-(eh**2) * (d_S.reshape(nh, ns_kept) ** 2).sum(1) / 2)

            d = h - hE
            dFdh = dFdh - ihC @ d
            dFdhh = dFdhh - ihC
            Ch = spm_inv(-dFdhh)

            dh = torch.clamp(spm_dx(dFdhh, dFdh, 4.0), -1.0, 1.0)
            h = h + dh
            if float(dFdh @ dh) < 1e-2:
                break

        # --- free energy ---------------------------------------------------
        # Accuracy, then the two complexity terms: how far the parameters and
        # the hyperparameters had to move from their priors, each weighted by
        # how much the posterior narrowed. The second is what makes F sensitive
        # to model size and is the part most easily got wrong.
        L = torch.stack(
            [
                torch.log(d_iS).sum() / 2
                - (d_iS * e * e).sum() / 2
                - ny * math.log(2 * math.pi) / 2,
                spm_logdet(ipC @ Cp) / 2 - (p @ ipC @ p) / 2,
                spm_logdet(ihC @ Ch) / 2 - (d @ ihC @ d) / 2,
            ]
        )
        F = L.sum()
        history.append(float(F))

        if bool(F > best["F"]) or k < 3:
            best = {"F": F, "p": p.clone(), "h": h.clone(), "Cp": Cp.clone()}
            dFdp = -(J.mT @ (d_iS * e)) - ipC @ p
            dFdpp = -Pp - ipC
            v_rate = min(v_rate + 0.5, 4.0)
            tag = "EM:(+)"
        else:
            p = best["p"].clone()
            h = best["h"].clone()
            Cp = best["Cp"].clone()
            v_rate = min(v_rate - 2.0, -4.0)
            tag = "EM:(-)"

        dp = spm_dx(dFdpp, dFdp, v_rate)
        p = p + dp
        dF_pred = float(dFdp @ dp)
        if verbose:
            print(f"{tag}: {k:3d}  F: {float(best['F']):12.4f}  dF predicted: {dF_pred:.3e}")

        criterion = [dF_pred < 1e-1, *criterion[:-1]]
        if all(criterion):
            converged = True
            break

    Ep_vec = pE_vec.clone()
    Ep_vec[free_idx] += best["p"]
    return VLResult(
        Ep=unvec(Ep_vec, spec),
        Cp=best["Cp"],
        Eh=best["h"],
        F=best["F"],
        L=L,
        y_pred=y_pred,
        free_idx=free_idx,
        n_iter=k,
        converged=converged,
        history=history,
    )


def _pad_to_bucket(full: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Pad a probe matrix up to a compiled batch width.

    Every distinct batch width is its own static compile of the integration step
    (~20 s), so the width must come from a small set. The padding rows repeat
    the last probe and their predictions are discarded.
    """
    n = full.shape[0]
    width = batch_bucket(n)
    if width == n:
        return full, n
    return torch.cat([full, full[-1:].expand(width - n, -1)], dim=0), n


def variational_laplace(
    spec: ModelSpec,
    priors: Priors,
    u: torch.Tensor,
    y: torch.Tensor,
    *,
    rows: torch.Tensor | None = None,
    kernel: torch.Tensor | None = None,
    max_iter: int = 128,
    verbose: bool = False,
    **integrate_kwargs,
) -> VLResult:
    """Invert the laminar model against one depth-resolved data matrix.

    Parameters
    ----------
    y : ``(n_scans, K)`` observed laminar BOLD.
    rows : boolean mask over time points. ``None`` fits every point. Faes et al.
        concatenate two conditions separated by white-noise padding and mask the
        padding out, which is the only reason it exists.

    Returns
    -------
    :class:`VLResult`, whose ``F`` is the quantity model comparison ranks.
    """
    n_scans = int(y.shape[0])
    co = _vl_coroutine(spec, priors, y, rows=rows, max_iter=max_iter, verbose=verbose)
    try:
        full = next(co)
        while True:
            padded, n = _pad_to_bucket(full)
            preds = integrate(
                u, unvec(padded, spec), spec, n_scans, kernel=kernel, **integrate_kwargs
            )
            full = co.send(preds[:n])
    except StopIteration as stop:
        return stop.value


def variational_laplace_lockstep(
    spec: ModelSpec,
    priors_list: list[Priors],
    u: torch.Tensor,
    y_list: list[torch.Tensor],
    *,
    rows: torch.Tensor | None = None,
    kernel: torch.Tensor | None = None,
    max_iter: int = 128,
    progress: bool = False,
    **integrate_kwargs,
) -> list[VLResult]:
    """Invert many models, or many datasets, sharing the forward model.

    Each fit runs its own Variational Laplace scheme, unchanged -- the results
    are bit-identical to calling :func:`variational_laplace` on each in turn.
    What is shared is the expensive part: at every iteration the probes of all
    still-running fits are concatenated into a *single* integration.

    That pays because the integration is host-bound, not compute-bound. Measured
    at K=9 on CPU, one probe row costs 205 us per step and 64 rows cost 286 us,
    so a batch of sixty-four fits' worth of probes costs about what one fit's
    does. The plateau breaks past ~128 rows, so the speedup is large but well
    short of the row count.

    ``priors_list`` and ``y_list`` are zipped: pass one ``y`` repeated to fit a
    model space against fixed data, or one set of priors repeated to fit the
    same model across parcels, seeds or conditions. ``u`` is shared, so every
    fit must have the same design and the same number of scans.

    Fits converge at different iteration counts, and the batch runs until the
    slowest finishes. Finished fits stop contributing probes, but the batch
    **width is held fixed** for the whole run.

    Bug of record: letting the width shrink as fits drop out is the obvious
    thing and it made lockstep *slower than serial*, 0.94x. Each narrower width
    crosses a bucket boundary into a shape that has never been compiled, and at
    ~20 s per compile a single run paid three or four of them -- more than the
    batching saved. Holding the width costs almost nothing, because the step is
    host-bound and the padding rows are close to free, which is the same
    property that makes the batching worth doing at all.
    """
    if len(priors_list) != len(y_list):
        raise ValueError(f"{len(priors_list)} priors against {len(y_list)} datasets")
    n_scans = int(y_list[0].shape[0])
    for i, y in enumerate(y_list):
        if int(y.shape[0]) != n_scans:
            raise ValueError(f"dataset {i} has {int(y.shape[0])} scans, expected {n_scans}")

    coroutines = [
        _vl_coroutine(spec, pr, y, rows=rows, max_iter=max_iter)
        for pr, y in zip(priors_list, y_list, strict=True)
    ]
    results: list[VLResult | None] = [None] * len(coroutines)
    pending: dict[int, torch.Tensor] = {}
    for i, co in enumerate(coroutines):
        try:
            pending[i] = next(co)
        except StopIteration as stop:  # pragma: no cover - max_iter=0
            results[i] = stop.value

    bar = None
    if progress:
        try:
            from tqdm.auto import tqdm

            bar = tqdm(total=len(coroutines), desc="laminar lockstep", leave=True)
        except ImportError:  # pragma: no cover - tqdm is a soft dependency
            bar = None

    # One width for the whole run, set by the opening batch. See the docstring.
    width = batch_bucket(sum(int(m.shape[0]) for m in pending.values()))

    while pending:
        idx = sorted(pending)
        sizes = [int(pending[i].shape[0]) for i in idx]
        stacked = torch.cat([pending[i] for i in idx], dim=0)
        n_real = int(stacked.shape[0])
        if n_real < width:
            stacked = torch.cat([stacked, stacked[-1:].expand(width - n_real, -1)], dim=0)
        elif n_real > width:  # pragma: no cover - the opening batch is the widest
            stacked, n_real = _pad_to_bucket(stacked)
        preds = integrate(u, unvec(stacked, spec), spec, n_scans, kernel=kernel, **integrate_kwargs)
        preds = preds[:n_real]

        offset = 0
        for i, n in zip(idx, sizes, strict=True):
            chunk = preds[offset : offset + n]
            offset += n
            try:
                pending[i] = coroutines[i].send(chunk)
            except StopIteration as stop:
                results[i] = stop.value
                del pending[i]
                if bar is not None:
                    bar.update(1)
    if bar is not None:
        bar.close()
    return [r for r in results if r is not None]
