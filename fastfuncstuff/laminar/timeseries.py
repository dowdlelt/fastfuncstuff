"""Continuous BOLD deveining with the linearized Havlicek–Uludag vasculature.

The unknown inputs are local CBF deviations at K vascular depths, not fast
neuronal activity. Exact Jacobians of the existing vascular ODE give the
frequency response. A regularized inverse estimates the transported signal;
subtracting that prediction preserves the unexplained residual. See the wiki's
Model-derived static deveining and Laminar timeseries deveining notes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from fastfuncstuff import policy_linalg as plinalg
from fastfuncstuff.laminar.forward import build_cache, g_obs, rhs
from fastfuncstuff.laminar.params import ModelSpec, zero_params
from fastfuncstuff.memory import estimate_chunk_size
from fastfuncstuff.utils import accum_dtype, cpu_if_mps, linalg_device


@dataclass
class VascularSystem:
    A: torch.Tensor  # (4K, 4K), vascular log-state Jacobian, per second
    B: torch.Tensor  # (4K, K), local relative-CBF inputs
    C: torch.Tensor  # (K, 4K), GE PSC observation
    micro_c: torch.Tensor  # venule-only observation, SAME H0 as full GE


def linearized_vasculature(
    spec: ModelSpec, params: dict[str, torch.Tensor] | None = None
) -> VascularSystem:
    """Linearize at rest; omit the neuronal/NVC front end from the inverse.

    K independent vascular inputs avoid interpreting the Gaussian neuronal-depth
    basis as drainage. This is a local approximation, not a nonlinear inversion.
    The small physiological Jacobian is generated in float64.
    """
    work = linalg_device(spec.device)
    s = replace(spec, dtype=torch.float64, device=work)
    p = (
        zero_params(s)
        if params is None
        else {k: v.to(device=work, dtype=s.dtype) for k, v in params.items()}
    )
    cache = build_cache(s, p)
    x = torch.zeros(s.n_states, device=work, dtype=s.dtype)
    u = torch.zeros(s.n_inputs, device=work, dtype=s.dtype)
    base, K = 4 * s.N, s.K
    # Input directly at vascular CBF, leaving local metabolism coupled by nr.
    cache = replace(cache, n2k=torch.eye(K, device=work, dtype=s.dtype))
    # rhs's neuronal blocks still need N states; use a K-input specification.
    if s.N != K:
        raise ValueError(
            "the continuous vascular system requires N=K (independent local CBF inputs)"
        )
    jac = torch.func.jacfwd(lambda z: rhs(z, u, cache, s.N, K))(x)
    obs = torch.func.jacfwd(lambda z: g_obs(z, p, s, cache=cache))(x)
    micro_cache = replace(
        cache,
        k1d=torch.zeros_like(cache.k1d),
        k2d=torch.zeros_like(cache.k2d),
        k3d=torch.zeros_like(cache.k3d),
    )
    micro_obs = torch.func.jacfwd(lambda z: g_obs(z, p, s, cache=micro_cache))(x)
    return VascularSystem(
        jac[base:, base:], jac[base:, 3 * s.N : 4 * s.N], obs[:, base:], micro_obs[:, base:]
    )


def vascular_transfer(
    system: VascularSystem, frequencies: torch.Tensor, target: str = "local"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return full and target responses (F,K,K), at frequencies in Hz.

    'local' retains each depth's response to its OWN CBF (including its local AV
    contribution). 'micro' retains venules only. Neither changes vascular flow
    equations or sets an ascending-vein compartment's volume to zero.
    """
    if target not in ("local", "micro"):
        raise ValueError("target must be local or micro")
    A, B = system.A, system.B
    freq = frequencies.to(device=A.device, dtype=A.dtype)
    eye = torch.eye(A.shape[0], device=A.device, dtype=torch.complex128)
    states = plinalg.solve(
        2j * torch.pi * freq[:, None, None] * eye - A,
        B.to(torch.complex128).expand(freq.numel(), -1, -1),
    )
    full = system.C.to(torch.complex128) @ states
    if target == "micro":
        desired = system.micro_c.to(torch.complex128) @ states
    else:
        desired = torch.diag_embed(full.diagonal(dim1=-2, dim2=-1))
    return full, desired


def stabilized_cov(cov: torch.Tensor) -> torch.Tensor:
    """Bin noise covariance made positive definite by an explicit, scale-relative ridge."""
    eye = torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype)
    return 0.95 * cov + 0.05 * cov.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] * eye


def cbf_kernel(
    spec: ModelSpec,
    params: dict[str, torch.Tensor] | None = None,
    dt: float = 0.1,
    seconds: float = 40.0,
) -> torch.Tensor:
    """Stimulus -> local CBF impulse response of the model's neuronal and NVC blocks.

    Linearised at rest and peak-normalised, sampled every ``dt`` s. Depths share
    scalar parameters, so one depth's response is every depth's; this is the
    response shape a task design gives each depth's CBF source.
    """
    work = linalg_device(spec.device)
    s = replace(spec, dtype=torch.float64, device=work)
    p = (
        zero_params(s)
        if params is None
        else {k: v.to(device=work, dtype=s.dtype) for k, v in params.items()}
    )
    cache = replace(build_cache(s, p), n2k=torch.eye(s.K, device=work, dtype=s.dtype))
    n4 = 4 * s.N
    x = torch.zeros(s.n_states, device=work, dtype=s.dtype)
    u = torch.zeros(s.n_inputs, device=work, dtype=s.dtype)
    jac = torch.func.jacfwd(lambda z: rhs(z, u, cache, s.N, s.K))(x)[:n4, :n4]
    step = torch.linalg.matrix_exp(jac * dt)
    state = torch.zeros(n4, device=work, dtype=s.dtype)
    state[0] = 1.0  # excitatory drive at depth 0
    out = torch.empty(int(round(seconds / dt)), dtype=s.dtype)
    for i in range(out.numel()):
        out[i] = state[3 * s.N]
        state = step @ state
    peak = out[out.abs().argmax()]
    return (out / peak).float()


def design_crossval(
    profiles: torch.Tensor,
    designs: list[torch.Tensor],
    full: torch.Tensor,
    n_fft: int,
    *,
    nuisance: list[torch.Tensor],
    sampling: torch.Tensor | None = None,
    noise_cov: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Leave-one-run-out prediction of depth profiles from each run's own events.

    Each patch's CBF sources are its design (``designs[r]``, ``(T, C)``, CBF-shaped)
    times free amplitudes per depth and condition -- one response shape at every
    depth -- seen through ``full`` (``(F, K, K)`` at ``rfftfreq(n_fft, TR)``). The
    amplitudes are fit on the other runs and predict the held-out one, so the runs'
    schedules may all differ. ``n_fft - T`` zero samples take each response's tail.
    Regressors are detrended with ``nuisance[r]`` as the profiles were. Free
    amplitudes absorb much of a drainage-strength change, so differences between
    physiologies are small: compare candidates on the same profiles, pooled.

    Returns per-patch held-out whitened SSE and total sum of squares, ``(U,)`` each.
    """
    R, U, T, K = profiles.shape
    if R < 2 or len(designs) != R or len(nuisance) != R or n_fft < T:
        raise ValueError("need >=2 runs, a design and nuisance per run, and n_fft >= T")
    dev = device or profiles.device
    dtype = accum_dtype(dev)
    H = full.to(device=dev, dtype=torch.complex128 if dtype == torch.float64 else torch.complex64)
    zs, grams = [], []
    for X, nuis in zip(designs, nuisance, strict=True):
        Xf = torch.fft.rfft(X.to(device=dev, dtype=dtype), n=n_fft, dim=0)
        Z = torch.fft.irfft(H[..., None] * Xf[:, None, None, :], n=n_fft, dim=0)[:T]
        Z = Z.reshape(T, K, -1)  # (time, physiological depth, source depth x condition)
        q, _ = plinalg.qr(nuis.to(device=dev, dtype=dtype))
        Z = Z - (q @ (q.T @ Z.reshape(T, -1))).reshape(Z.shape)
        zs.append(Z)
        grams.append(torch.einsum("tjp,tkq->jkpq", Z, Z))
    P = zs[0].shape[-1]
    sse = torch.empty(U, dtype=torch.float64)
    total = torch.empty(U, dtype=torch.float64)
    chunk = estimate_chunk_size(U, T, K, dev, operation="laminar_design_cv", n_runs=R, n_designs=P)
    eye_k = torch.eye(K, device=dev, dtype=dtype)
    eye_p = torch.eye(P, device=dev, dtype=dtype)
    solve_device = cpu_if_mps(dev, "solve")
    for start in range(0, U, chunk):
        stop = min(U, start + chunk)
        b = stop - start
        S = (
            eye_k.expand(b, -1, -1)
            if sampling is None
            else sampling[start:stop].to(device=dev, dtype=dtype)
        )
        cov = (
            eye_k.expand(b, -1, -1)
            if noise_cov is None
            else noise_cov[start:stop].to(device=dev, dtype=dtype)
        )
        cw = torch.cholesky_inverse(
            plinalg.cholesky(stabilized_cov(cov).to(cpu_if_mps(dev, "cholesky")))
        ).to(dev)
        metric = S.mT @ cw @ S  # whitened fit, in physiological depths
        G = torch.stack([torch.einsum("bjk,jkpq->bpq", metric, g) for g in grams])
        y = profiles[:, start:stop].to(device=dev, dtype=dtype)
        v = y @ (cw @ S)[None]  # (R, b, T, K)
        rhs_ = torch.stack([torch.einsum("tjp,btj->bp", z, v[r]) for r, z in enumerate(zs)])
        yy = torch.einsum("rbtm,bmn,rbtn->rb", y, cw, y)
        err = torch.zeros(b, device=dev, dtype=dtype)
        for r in range(R):
            g_tr, b_tr = G.sum(0) - G[r], rhs_.sum(0) - rhs_[r]
            eps = 1e-6 * g_tr.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-30)
            beta = plinalg.solve(
                (g_tr + eps[:, None, None] * eye_p).to(solve_device),
                b_tr[..., None].to(solve_device),
            ).to(dev)[..., 0]
            err += (
                yy[r]
                - 2 * (beta * rhs_[r]).sum(-1)
                + (beta[:, None] @ G[r] @ beta[..., None])[:, 0, 0]
            )
        sse[start:stop] = err.cpu().double()
        total[start:stop] = yy.sum(0).cpu().double()
    return sse, total


def loro_prior_ridge(
    xtx: torch.Tensor,
    xty: torch.Tensor,
    yy: torch.Tensor,
    prior: torch.Tensor,
    penalized: torch.Tensor,
    lambdas: tuple[float, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-voxel ridge toward a prior mean, its strength chosen by leave-one-run-out.

    ``xtx`` ``(R, b, P, P)``, ``xty`` ``(R, b, P)`` and ``yy`` ``(R, b)`` are each
    run's normal equations. Coefficients flagged in ``penalized`` are pulled toward
    ``prior`` by ``lambda * scale``, where ``scale`` is their mean all-run Gram
    diagonal, so lambda is unitless: large returns the prior, small a free fit.
    Unpenalized coefficients are fit freely and compete for the same variance.

    Returns coefficients refit on all runs at each voxel's chosen lambda, the
    chosen index ``(b,)``, and held-out SSE per lambda ``(L, b)``.
    """
    R = xtx.shape[0]
    if R < 2 or not lambdas or any(x <= 0 for x in lambdas):
        raise ValueError("need >=2 runs and positive lambdas")
    dev, dtype = xtx.device, xtx.dtype
    pen = penalized.to(device=dev, dtype=dtype)
    mu = prior.to(device=dev, dtype=dtype)
    tot_a, tot_b = xtx.sum(0), xty.sum(0)
    diag = tot_a.diagonal(dim1=-2, dim2=-1)
    scale = (diag * pen).sum(-1) / pen.sum().clamp_min(1)
    # A regressor that is all zero (the deepest depth has nothing below it)
    # must still give a solvable system; it then returns its prior, times zero.
    floor = 1e-6 * diag.mean(-1) + torch.finfo(dtype).tiny
    scale = scale.clamp_min(floor)
    solve_device = cpu_if_mps(dev, "solve")

    def solve(a, b, lam):
        d = lam[:, None] * scale[:, None] * pen + floor[:, None]
        return plinalg.solve(
            (a + torch.diag_embed(d)).to(solve_device), (b + d * mu)[..., None].to(solve_device)
        ).to(dev)[..., 0]

    sse = torch.zeros(len(lambdas), xtx.shape[1], device=dev, dtype=dtype)
    for li, lam in enumerate(lambdas):
        lam_b = torch.full_like(scale, lam)
        for r in range(R):
            beta = solve(tot_a - xtx[r], tot_b - xty[r], lam_b)
            sse[li] += (
                yy[r]
                - 2 * (beta * xty[r]).sum(-1)
                + (beta[:, None] @ xtx[r] @ beta[..., None])[:, 0, 0]
            )
    best = sse.argmin(0)
    lam_best = torch.as_tensor(lambdas, device=dev, dtype=dtype)[best]
    return solve(tot_a, tot_b, lam_best), best, sse


def _padded_spectrum(y: torch.Tensor, pad: int) -> torch.Tensor:
    """``(R, b, T, K)`` -> reflect-padded rfft, ``(R, b, F, K)``."""
    R, b, T, K = y.shape
    yp = F.pad(y.permute(0, 1, 3, 2).reshape(-1, 1, T), (pad, pad), mode="reflect")
    return torch.fft.rfft(yp.reshape(R, b, K, T + 2 * pad), dim=-1).permute(0, 1, 3, 2)


def _crop(z: torch.Tensor, T: int, pad: int) -> torch.Tensor:
    """Inverse of :func:`_padded_spectrum`: ``(R, b, F, K)`` -> ``(R, b, T, K)``."""
    return torch.fft.irfft(z.permute(0, 1, 3, 2), n=T + 2 * pad, dim=-1)[
        ..., pad : pad + T
    ].permute(0, 1, 3, 2)


def _inverse_operators(H, b, sampling, noise_cov, strengths, penalty, dev, dtype, cdtype):
    """Sampling ``S`` ``(b, 1, K, K)`` (complex), the sampled forward ``S H`` and one
    ridge inverse per strength.

    ``sampling`` / ``noise_cov`` are ``(b, K, K)`` or None (identity) for ``b`` units. Strengths are
    relative to the unit's own whitened model scale, so a unit measured with more
    voxels (a larger whitened gram) is penalised in proportion.
    """
    K = H.shape[-1]
    eye = torch.eye(K, device=dev, dtype=cdtype)
    S = eye.real.expand(b, -1, -1) if sampling is None else sampling.to(device=dev, dtype=dtype)
    cov = eye.real.expand(b, -1, -1) if noise_cov is None else noise_cov.to(device=dev, dtype=dtype)
    chol = plinalg.cholesky(stabilized_cov(cov).to(cpu_if_mps(dev, "cholesky"))).to(dev)
    whiten = plinalg.solve_triangular(chol, eye.real.expand(b, -1, -1), upper=False).to(cdtype)
    Sc = S.to(cdtype)[:, None]
    hs = Sc @ H
    hw = whiten[:, None] @ hs
    gram = hw.mH @ hw
    scale = gram[:, 0].diagonal(dim1=-2, dim2=-1).real.mean(-1).clamp_min(torch.finfo(dtype).eps)
    solve_device = cpu_if_mps(dev, "solve")
    rhs_solve = (hw.mH @ whiten[:, None]).to(solve_device)
    inverses = [
        plinalg.solve(
            (gram + st * scale[:, None, None, None] * penalty[None, :, None, None] * eye).to(
                solve_device
            ),
            rhs_solve,
        ).to(dev)
        for st in strengths
    ]
    return Sc, hs, inverses


@dataclass
class DynamicResult:
    correction: torch.Tensor  # (R,U,T,K), transported contribution, sample depths
    predicted: torch.Tensor  # full model reconstruction: a smoothing-only control
    grid_correction: torch.Tensor  # physiological depth grid, before sampling
    grid_predicted: torch.Tensor
    regularization: torch.Tensor  # (U,), selected on GE runs alone
    cv_r2: torch.Tensor  # (U,), leave-run-out full-model prediction
    noise_gain: torch.Tensor  # (U,), RMS gain of residual-preserving correction


def dynamic_inverse(
    profiles: torch.Tensor,
    full: torch.Tensor,
    target: torch.Tensor,
    *,
    sampling: torch.Tensor | None = None,
    noise_cov: torch.Tensor | None = None,
    strengths: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0),
    tr: float,
    pad: int,
    cutoff_hz: float = 0.1,
    device: torch.device | None = None,
    scores: torch.Tensor | None = None,
    verbose: bool = True,
) -> DynamicResult:
    """Regularized spectral inverse, selecting shrinkage per unit by GE LORO.

    ``scores`` ``(len(strengths), U)`` selects instead (the largest wins), e.g.
    :func:`split_half_scores`, which needs no repeated runs; they are reported
    as ``cv_r2``.

    profiles are detrended PSC, (runs, units, time, depths). Runs used for LORO
    MUST have the same task schedule and length. With one run a single supplied
    strength is required. A fixed single strength also supports unrelated runs.
    full/target use rfftfreq(T+2*pad, TR). Reflect padding reduces wraparound;
    first/last pad samples remain boundary-sensitive and should be excluded in QC.
    sampling maps physiological depths to measured bins. noise_cov accounts for
    shared voxels in those bins; a small diagonal term stabilizes the whitening.
    """
    if profiles.ndim != 4:
        raise ValueError("profiles must be (runs, units, time, depths)")
    R, U, T, K = profiles.shape
    if U < 1 or T < 3 or not 0 <= pad < T or tr <= 0 or cutoff_hz <= 0:
        raise ValueError("need units, >=3 samples, 0<=pad<T, and positive TR/cutoff")
    if not strengths or any(x <= 0 for x in strengths):
        raise ValueError("positive strengths required")
    if scores is None and R < 2 and len(strengths) > 1:
        raise ValueError("selection needs >=2 repeated runs, or scores")
    if scores is not None and tuple(scores.shape) != (len(strengths), U):
        raise ValueError("scores must be (len(strengths), units)")
    nf = (T + 2 * pad) // 2 + 1
    if full.shape != (nf, K, K) or target.shape != full.shape:
        raise ValueError("transfer grid must match rfftfreq(T+2*pad, tr) and K depths")
    if not torch.isfinite(profiles).all():
        raise ValueError("profiles must be finite")
    dev = device or profiles.device
    dtype = accum_dtype(dev) if profiles.dtype == torch.float64 else torch.float32
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    H, G = full.to(device=dev, dtype=cdtype), target.to(device=dev, dtype=cdtype)
    correction, predicted = torch.empty_like(profiles), torch.empty_like(profiles)
    grid_correction, grid_predicted = torch.empty_like(profiles), torch.empty_like(profiles)
    chosen, cv, gain = [
        torch.empty(U, dtype=profiles.dtype, device=profiles.device) for _ in range(3)
    ]
    chunk = estimate_chunk_size(
        U, T + 2 * pad, K, dev, operation="laminar_dynamic", n_runs=R, n_designs=len(strengths)
    )
    freq = torch.fft.rfftfreq(T + 2 * pad, tr, device=dev)
    penalty = 1 + (freq / cutoff_hz) ** 4
    eye = torch.eye(K, device=dev, dtype=cdtype)
    weights_f = torch.full((nf,), 2.0, device=dev, dtype=dtype)
    weights_f[0] = 1
    if (T + 2 * pad) % 2 == 0:
        weights_f[-1] = 1
    # CV is scored after cropping reflection padding, in observed BOLD units.
    for start in tqdm(
        range(0, U, chunk), desc="Dynamic inverse", leave=True, disable=not verbose or U <= chunk
    ):
        stop = min(U, start + chunk)
        b = stop - start
        y = profiles[:, start:stop].to(device=dev, dtype=dtype)
        yf = _padded_spectrum(y, pad)
        Sc, hs, inverses = _inverse_operators(
            H,
            b,
            None if sampling is None else sampling[start:stop],
            None if noise_cov is None else noise_cov[start:stop],
            strengths, penalty, dev, dtype, cdtype,
        )  # fmt: skip
        total = y.square().sum((0, 2, 3)).clamp_min(torch.finfo(dtype).eps)
        if scores is not None:
            scores_t = scores[:, start:stop].to(device=dev, dtype=dtype)
        else:
            rows = []
            for inverse in inverses:
                if R > 1:
                    train = (yf.sum(0, keepdim=True) - yf) / (R - 1)
                    held = torch.einsum("bfij,rbfj->rbfi", hs @ inverse, train)
                    rows.append(1 - (_crop(held, T, pad) - y).square().sum((0, 2, 3)) / total)
                else:
                    rows.append(torch.full((b,), float("nan"), device=dev, dtype=dtype))
            scores_t = torch.stack(rows)
        best = (
            scores_t.nan_to_num(float("-inf")).argmax(0)
            if len(strengths) > 1
            else torch.zeros(b, device=dev, dtype=torch.long)
        )
        inv = torch.stack(inverses)[best, torch.arange(b, device=dev)]
        remove = (hs - Sc @ G) @ inv
        transported = torch.einsum("bfij,rbfj->rbfi", remove, yf)
        fit = torch.einsum("bfij,rbfj->rbfi", hs @ inv, yf)

        def crop(z):
            return _crop(z, T, pad).to(profiles.device).to(profiles.dtype)

        correction[:, start:stop], predicted[:, start:stop] = crop(transported), crop(fit)
        grid_correction[:, start:stop] = crop(torch.einsum("bfij,rbfj->rbfi", (H - G) @ inv, yf))
        grid_predicted[:, start:stop] = crop(torch.einsum("bfij,rbfj->rbfi", H @ inv, yf))
        chosen[start:stop] = (
            torch.tensor(strengths, device=dev, dtype=dtype)[best]
            .to(chosen.device)
            .to(chosen.dtype)
        )
        cv[start:stop] = scores_t[best, torch.arange(b, device=dev)].to(cv.device).to(cv.dtype)
        filt = eye - remove
        # Integrated white-noise gain; this is a conditional diagnostic, not a CI.
        ng = (filt.abs().square().sum((-1, -2)) * weights_f).sum(-1) / (K * weights_f.sum())
        gain[start:stop] = ng.sqrt().to(gain.device).to(gain.dtype)
    return DynamicResult(correction, predicted, grid_correction, grid_predicted, chosen, cv, gain)


def split_half_scores(
    half_a: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None],
    half_b: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None],
    full: torch.Tensor,
    *,
    strengths: tuple[float, ...],
    tr: float,
    pad: int,
    cutoff_hz: float = 0.1,
    factor: float = 2.0,
    device: torch.device | None = None,
    verbose: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score ridge strengths across voxels instead of across runs: no repeated timing.

    Each half is ``(profiles (R,U,T,K), sampling, noise_cov)`` from disjoint voxels of
    the same units: the same blood-flow histories under independent noise.  The
    inverse fit to one half at each strength predicts the other through the forward
    model, at the same moment, and both directions are pooled into a held-out R²
    ``(len(strengths), U)``.  Every fluctuation the halves share counts as signal,
    task-locked or not -- which is what the correction removes -- where predicting
    a held-out run rewards only the task-locked part, i.e. smoothing.

    A half carries ``factor`` times the noise of the full profile, and strengths are
    relative to the unit's whitened model scale, so each half is fit at ``factor x``
    the strength being scored.  Also returns each unit's normalised A-B
    cross-spectrum above the model's band (``leak``, ~0 when the halves' noise is
    independent): shared resampling noise would read as signal and favour too
    little shrinkage.  Physiological signal up there inflates it, so it is an
    upper bound on the leak, not a measure of it.
    """
    ya_all, yb_all = half_a[0], half_b[0]
    if ya_all.shape != yb_all.shape or ya_all.ndim != 4:
        raise ValueError("both halves need (runs, units, time, depths) profiles of one shape")
    R, U, T, K = ya_all.shape
    nf = (T + 2 * pad) // 2 + 1
    if full.shape != (nf, K, K):
        raise ValueError("transfer grid must match rfftfreq(T+2*pad, tr) and K depths")
    if not strengths or any(x <= 0 for x in strengths) or factor <= 0:
        raise ValueError("positive strengths and factor required")
    dev = device or ya_all.device
    dtype = torch.float32
    cdtype = torch.complex64
    H = full.to(device=dev, dtype=cdtype)
    freq = torch.fft.rfftfreq(T + 2 * pad, tr, device=dev)
    penalty = 1 + (freq / cutoff_hz) ** 4
    band = freq >= min(2 * cutoff_hz, 0.8 * float(freq[-1]))
    scaled = tuple(st * factor for st in strengths)
    scores = torch.empty(len(strengths), U, dtype=dtype)
    leak = torch.empty(U, dtype=dtype)
    chunk = estimate_chunk_size(
        U, T + 2 * pad, K, dev, operation="laminar_dynamic", n_runs=2 * R,
        n_designs=2 * len(strengths),
    )  # fmt: skip

    def part(t, a, b):
        return None if t is None else t[a:b]

    for start in tqdm(
        range(0, U, chunk), desc="Split-half ridge", leave=True, disable=not verbose or U <= chunk
    ):
        stop = min(U, start + chunk)
        b = stop - start
        ya = ya_all[:, start:stop].to(device=dev, dtype=dtype)
        yb = yb_all[:, start:stop].to(device=dev, dtype=dtype)
        fa, fb = _padded_spectrum(ya, pad), _padded_spectrum(yb, pad)
        sa, _, inv_a = _inverse_operators(
            H, b, part(half_a[1], start, stop), part(half_a[2], start, stop), scaled, penalty,
            dev, dtype, cdtype,
        )  # fmt: skip
        sb, _, inv_b = _inverse_operators(
            H, b, part(half_b[1], start, stop), part(half_b[2], start, stop), scaled, penalty,
            dev, dtype, cdtype,
        )  # fmt: skip
        total = (ya.square().sum((0, 2, 3)) + yb.square().sum((0, 2, 3))).clamp_min(1e-30)
        for i in range(len(strengths)):
            # a -> b and b -> a, each through the OTHER half's own depth sampling
            pb = _crop(torch.einsum("bfij,rbfj->rbfi", sb @ H @ inv_a[i], fa), T, pad)
            pa = _crop(torch.einsum("bfij,rbfj->rbfi", sa @ H @ inv_b[i], fb), T, pad)
            err = (pb - yb).square().sum((0, 2, 3)) + (pa - ya).square().sum((0, 2, 3))
            scores[i, start:stop] = (1 - err / total).cpu()
        cross = (fa[:, :, band] * fb[:, :, band].conj()).real.sum((0, 2, 3))
        norm = (fa[:, :, band].abs().square().sum((0, 2, 3))
                * fb[:, :, band].abs().square().sum((0, 2, 3))).sqrt().clamp_min(1e-30)  # fmt: skip
        leak[start:stop] = (cross / norm).cpu()
    return scores, leak
