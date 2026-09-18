"""
Model-based deveining: laminar neuronal activity from depth-sampled BOLD.

Once a model is inverted, the fitted parameters define a map from neuronal
activity at ``N`` depths to BOLD at ``K`` depths. Deveining is the inverse of
that map. This module provides it in three forms of decreasing fidelity, because
what a downstream analysis can actually *use* is usually not the full one:

``deveined_timecourses``
    The exact answer: re-integrate the fitted model and read the neuronal states
    directly. No approximation, but it is tied to the stimulus and the fit.

``laminar_impulse_response``
    The neuronal -> BOLD transfer, ``(K, N, n_scans)``. Depth ``k`` of the output
    responds to depth ``n`` of the input with a kernel, not a number: the
    ascending vein imposes a transit delay and smears the response, which is why
    a scalar correction cannot be exact.

``static_deveining_matrix``
    The transfer collapsed to a single ``(K, N)`` amplitude matrix, and its
    pseudo-inverse ``(N, K)``. This is the form that multiplies betas, FIR peaks
    or event-related averages -- a per-ROI, per-depth correction applied as
    linear algebra. It supersedes assumed static-deconvolution weights (Menon
    2002 and descendants) by *estimating* them per ROI from physiology.

Three properties of the true operator constrain how far these reductions can be
trusted, and all three are measurable with :func:`deveining_fidelity`:

1. **It is triangular, not diagonal.** Blood drains superficially, so depth ``k``
   carries contributions from every depth below it. A per-depth scalar can
   rescale a depth against itself but can never remove what leaked in from
   below. The off-diagonal mass is the error a scalar correction commits.
2. **It has a time axis.** The inter-laminar transit delay is real and a scalar
   cannot represent a delay, so the amplitude reduction is exact only for a
   summary statistic (peak, or area), not for a timecourse.
3. **It is amplitude-dependent.** The balloon and dHb terms are nonlinear, so an
   operator derived at one effect size is wrong at another. Deriving it from a
   *contrast* rather than a single condition cancels much of this.

Deconvolution also amplifies noise: recovering deep-free superficial activity
means subtracting a *predicted* contribution that carries posterior uncertainty.
A scalar multiplier hides that entirely, rescaling signal and noise together and
leaving tSNR apparently untouched. :func:`deveining_fidelity` reports it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fastfuncstuff.laminar.integrate import batch_bucket, integrate
from fastfuncstuff.laminar.params import ModelSpec


@dataclass
class DeveinedResult:
    """Measured, predicted and inferred laminar timecourses on one time axis."""

    y_measured: torch.Tensor | None  # (n_scans, K) depth-sampled BOLD, if supplied
    y_predicted: torch.Tensor  # (n_scans, K) the fitted model's BOLD
    neuronal: torch.Tensor  # (n_scans, N) inferred excitatory activity
    inhibitory: torch.Tensor  # (n_scans, N)
    cbf: torch.Tensor  # (n_scans, N) CBF at the *neuronal* depths
    # The vascular-depth CBF is formed inside the ODE by the depth mapping and
    # never stored, so what comes back here is pre-mapping, at N not K.

    @property
    def n_scans(self) -> int:
        return int(self.y_predicted.shape[-2])


def deveined_timecourses(
    spec: ModelSpec,
    P: dict[str, torch.Tensor],
    u: torch.Tensor,
    n_scans: int,
    *,
    y_measured: torch.Tensor | None = None,
    kernel: torch.Tensor | None = None,
    **integrate_kwargs,
) -> DeveinedResult:
    """Neuronal timecourses implied by a fitted model, alongside its BOLD.

    This is deveining without approximation: the neuronal states are what the
    generative model says produced the measured BOLD, so reading them out *is*
    the vein-free estimate. The cost is that it is specific to this stimulus and
    this fit -- it is not an operator you can carry to other data. For that, see
    :func:`static_deveining_matrix`.

    ``P`` is normally a posterior mean (``VLResult.Ep`` or ``BPAResult.Ep``).
    """
    y_pred, states = integrate(
        u, P, spec, n_scans, kernel=kernel, return_states=True, **integrate_kwargs
    )
    N = spec.N
    # Excitatory and inhibitory are stored linearly; CBF is log-scaled.
    return DeveinedResult(
        y_measured=y_measured,
        y_predicted=y_pred,
        neuronal=states[..., 0:N],
        inhibitory=states[..., N : 2 * N],
        cbf=torch.exp(states[..., 3 * N : 4 * N]),
    )


def laminar_impulse_response(
    spec: ModelSpec,
    P: dict[str, torch.Tensor],
    n_scans: int,
    *,
    amplitude: float = 1.0,
    duration_bins: int = 1,
    kernel: torch.Tensor | None = None,
    **integrate_kwargs,
) -> torch.Tensor:
    """BOLD response at every depth to neuronal drive at each depth alone.

    Returns ``(K, N, n_scans)``: entry ``[k, n, t]`` is the BOLD at vascular
    depth ``k``, ``t`` scans after a brief drive delivered to neuronal depth
    ``n``. Depth 0 is superficial in both axes.

    The ``N`` drives are a single batched integration, which costs what one
    costs -- the step is host-bound, so the batch axis is close to free.

    ``amplitude`` matters: the model is nonlinear, so this operator is a local
    linearisation and is only valid near the effect size it was derived at. Pass
    something comparable to the modulation actually fitted.
    """
    N = spec.N
    n_micro = int(round(n_scans * spec.TR / spec.dt))
    width = batch_bucket(N)

    # One parameter set per driven depth: C is zeroed except on that depth, so
    # the driving input reaches exactly one neuronal depth.
    P_batch = {}
    for key, value in P.items():
        v = torch.as_tensor(value, dtype=spec.dtype, device=spec.device)
        P_batch[key] = v.expand((width,) + v.shape).clone()
    c = torch.zeros(width, N, spec.n_inputs, dtype=spec.dtype, device=spec.device)
    for n in range(N):
        c[n, n, -1] = amplitude  # last column is the driving input
    P_batch["C"] = c

    u = torch.zeros(width, n_micro, spec.n_inputs, dtype=spec.dtype, device=spec.device)
    u[:, :duration_bins, -1] = 1.0

    y = integrate(u, P_batch, spec, n_scans, kernel=kernel, **integrate_kwargs)
    # (width, n_scans, K) -> (K, N, n_scans), discarding the padded rows.
    return y[:N].permute(2, 0, 1).contiguous()


def static_deveining_matrix(
    impulse_response: torch.Tensor,
    *,
    mode: str = "peak",
    rcond: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse the transfer to an amplitude matrix and its inverse.

    Parameters
    ----------
    impulse_response : ``(K, N, n_scans)`` from :func:`laminar_impulse_response`.
    mode : ``"peak"`` takes each kernel's extremum (the summary an FIR peak or a
        single-basis beta reports); ``"auc"`` integrates it (the summary a
        sustained-response or area measure reports). They differ exactly to the
        extent the vein smears the response, so disagreement between them is
        itself a readout of how badly a static correction fits.

    Returns
    -------
    ``W`` of shape ``(K, N)``, the forward mixing -- how neuronal activity at
    each depth appears in BOLD at each depth -- and ``W_inv`` of shape
    ``(N, K)``, its pseudo-inverse, which is what multiplies measured
    depth-resolved betas to deveined ones.

    ``K > N`` in every published use, so ``W`` is tall and ``W_inv`` is a
    least-squares solution rather than an exact inverse. That is a feature: the
    extra vascular depths over-determine the neuronal ones, which is what makes
    the estimate stable.
    """
    if mode == "peak":
        flat = impulse_response.flatten(start_dim=2)
        idx = flat.abs().argmax(dim=-1, keepdim=True)
        W = flat.gather(-1, idx).squeeze(-1)
    elif mode == "auc":
        W = impulse_response.sum(dim=-1)
    else:
        raise ValueError(f"mode must be 'peak' or 'auc', got {mode!r}")
    return W, torch.linalg.pinv(W, rcond=rcond)


def apply_deveining(values: torch.Tensor, W_inv: torch.Tensor) -> torch.Tensor:
    """Apply a deveining operator to depth-resolved amplitudes.

    ``values`` is ``(..., K)`` -- betas, FIR peaks, event-related averages, one
    entry per vascular depth. Returns ``(..., N)`` neuronal-depth estimates.

    This is the form that answers "can it be applied to the raw data, or to the
    betas": yes, as a matrix multiply, provided the amplitudes are in the same
    units the model was fitted in (percent signal change against the same
    baseline) and the effect size is near the one the operator was derived at.
    """
    if values.shape[-1] != W_inv.shape[-1]:
        raise ValueError(
            f"values have {values.shape[-1]} depths, operator expects {W_inv.shape[-1]}"
        )
    return values @ W_inv.transpose(-1, -2)


def deveining_fidelity(
    impulse_response: torch.Tensor,
    *,
    mode: str = "peak",
) -> dict[str, torch.Tensor]:
    """How much a static, and then a scalar, correction gives up.

    Returns diagnostics that decide whether the cheap form is defensible for a
    given ROI, rather than assuming it:

    ``from_deeper`` / ``from_shallower``
        Fraction of each output depth's amplitude contributed by neuronal depths
        below / above the one lying at that cortical position. These must be
        read as a pair, because two different mechanisms mix depths and only one
        of them is a vein:

        * the neuronal -> vascular Gaussian basis spreads activity across
          neighbouring depths **symmetrically**, and is not contamination at all
          -- it is the depth mapping;
        * venous drainage adds mass **directionally**, from deeper depths only.

        So ``from_shallower`` is the symmetric-spread floor, and the asymmetry
        between them is the part a vein caused. A metric that merely summed
        off-diagonal mass would conflate the two and report a boundary depth as
        heavily contaminated when it is simply straddling two neuronal depths.
    ``drainage_asymmetry``
        ``from_deeper - from_shallower``: the directional, vein-attributable
        fraction. This is what a per-depth scalar cannot remove, because
        rescaling a depth against itself cannot subtract what leaked in from
        below. Expect it near zero at the deepest depth and largest at the
        surface.

        ``K > N``, so "its own depth" is not the matrix diagonal: vascular depth
        ``k`` is paired with the neuronal depth covering that fraction of
        cortex, ``k * N // K``.
    ``peak_lag``
        Scans from drive to peak, per (K, N) pair. Spread across the column is
        the transit delay a static operator discards.
    ``lag_spread``
        Max minus min ``peak_lag`` within each output depth. Non-zero means the
        contributions arrive at different times and no single number can align
        them.
    """
    W, _ = static_deveining_matrix(impulse_response, mode=mode)
    K, N = W.shape
    mag = W.abs()
    total = mag.sum(dim=1).clamp(min=1e-300)
    own = (torch.arange(K, device=W.device) * N // K).clamp(max=N - 1)
    col = torch.arange(N, device=W.device)[None, :].expand(K, N)
    own_col = own[:, None]
    from_deeper = torch.where(col > own_col, mag, mag.new_zeros(())).sum(dim=1) / total
    from_shallower = torch.where(col < own_col, mag, mag.new_zeros(())).sum(dim=1) / total

    peak_lag = impulse_response.abs().argmax(dim=-1).to(impulse_response.dtype)
    return {
        "from_deeper": from_deeper,
        "from_shallower": from_shallower,
        "drainage_asymmetry": from_deeper - from_shallower,
        "own_depth": own,
        "peak_lag": peak_lag,
        "lag_spread": peak_lag.max(dim=1).values - peak_lag.min(dim=1).values,
    }
