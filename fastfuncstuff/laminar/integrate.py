"""
Fixed-step integration and TR sampling for the laminar generative model.

The reference (``spm_int_IT.m``) advances the states with a second-order
Ito-Taylor update

.. math:: x_{t+\\Delta t} = x_t + \\Delta t\\, f + \\tfrac{1}{2}\\Delta t^2 J f

where :math:`J = \\partial f/\\partial x`, supplied there by generated symbolic
files (one pair per (N, K) combination). Matching this update is a parity gate
in its own right: an adaptive solver is scientifically defensible but will not
reproduce the published numbers, so it is not allowed in until this does.

We never form :math:`J`. The update needs only the Jacobian-vector product
:math:`Jf`, which ``torch.func.jvp`` computes exactly in a single forward-mode
pass -- O(1) in the state count instead of O(n). That removes the generated
symbolic files entirely and is exact, not a finite difference.

Bug of record
-------------
The reference's symbolic Jacobian is taken with respect to the *exponentiated*
states, because ``LBR_gen_fx_fcn.m`` differentiates against the symbols that
receive ``exp(x)``. Most of the state vector is stored in log space, so a
chain-rule factor is missing and the reference's correction term is not
:math:`Jf` but :math:`J D^{-1} f`, with :math:`D = \\mathrm{diag}(dz/dx)`.

We reproduce that by passing ``f / d`` as the JVP tangent (verified against
MATLAB to 1e-14), because the published fits and free energies were produced
with it. ``jacobian="exact"`` uses the true Jacobian instead. The difference is
O(dt^2) per step and does not vanish -- it changes the trajectory, and therefore
the fit and the model ranking.
"""

from __future__ import annotations

import torch
from torch.func import jvp

from fastfuncstuff.laminar.forward import f_ode, g_obs
from fastfuncstuff.laminar.params import ModelSpec


def sample_indices(n_micro: int, n_scans: int, delay_bins: int) -> torch.Tensor:
    """Microtime index at which each output scan is observed.

    Replicates ``spm_int_IT``: ``ceil((0:v-1)*u/v) + D``, where ``D`` is the
    slice-timing delay in microtime bins (``TR/2`` in every published use).
    """
    n = torch.arange(n_scans, dtype=torch.float64)
    return torch.ceil(n * n_micro / n_scans).long() + delay_bins


def _dz_dx(x: torch.Tensor, N: int) -> torch.Tensor:
    """d(exponentiated state)/d(stored state): 1 for the linear states, exp(x)
    for the log-scaled ones (CBF and all four vascular states)."""
    d = torch.ones_like(x)
    d[..., 3 * N :] = torch.exp(x[..., 3 * N :])
    return d


def delay_bins(spec: ModelSpec, delay_seconds: float | None = None) -> int:
    if delay_seconds is None:
        delay_seconds = spec.TR / 2.0
    return max(int(round(delay_seconds / spec.dt)), 1)


def integrate(
    u: torch.Tensor,
    P: dict[str, torch.Tensor],
    spec: ModelSpec,
    n_scans: int,
    *,
    kernel: torch.Tensor | None = None,
    delay_seconds: float | None = None,
    x0: torch.Tensor | None = None,
    return_states: bool = False,
    shift_depths: bool = False,
    jacobian: str = "reference",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Integrate the model and return predicted BOLD at ``n_scans`` TRs.

    Parameters
    ----------
    u : ``(..., n_micro, n_inputs)`` input at microtime resolution ``spec.dt``.
    P : parameter dict with leading batch dimensions broadcastable against ``u``.
    n_scans : number of output volumes.
    jacobian : ``"reference"`` reproduces the MATLAB's chain-rule-free Jacobian
        (required for parity with published fits); ``"exact"`` uses the true
        Jacobian of the state equation.

    Returns
    -------
    ``(..., n_scans, K)`` predicted laminar BOLD in percent signal change.
    """
    n_micro = u.shape[-2]
    idx = sample_indices(n_micro, n_scans, delay_bins(spec, delay_seconds))
    if int(idx.max()) >= n_micro:
        raise ValueError(
            f"sampling runs past the input: need {int(idx.max()) + 1} microtime bins, "
            f"have {n_micro}. Extend the input or reduce the delay."
        )
    # Map microtime step -> output scan, so the integration loop can stay a
    # single pass and does not need a search per step.
    scan_at = torch.full((n_micro,), -1, dtype=torch.long)
    scan_at[idx] = torch.arange(n_scans)

    batch = torch.broadcast_shapes(u.shape[:-2], P["sigma"].shape)
    if x0 is None:
        # Zero is the resting state: the log-scaled states exponentiate to 1.
        x = torch.zeros(*batch, spec.n_states, dtype=spec.dtype, device=spec.device)
    else:
        x = x0.expand(*batch, spec.n_states).clone()

    if jacobian not in ("reference", "exact"):
        raise ValueError(f"jacobian must be 'reference' or 'exact', got {jacobian!r}")

    dt = spec.dt
    y = torch.zeros(*batch, n_scans, spec.K, dtype=spec.dtype, device=spec.device)
    states = (
        torch.zeros(*batch, n_scans, spec.n_states, dtype=spec.dtype, device=spec.device)
        if return_states
        else None
    )

    for i in range(n_micro):
        u_i = u[..., i, :]

        def _f(xx: torch.Tensor, _u=u_i) -> torch.Tensor:
            return f_ode(xx, _u, P, spec, shift_depths=shift_depths)

        fx = _f(x)
        # The observation is read *before* the state advances, matching the
        # reference's ordering inside the integration loop.
        s = int(scan_at[i])
        if s >= 0:
            y[..., s, :] = g_obs(x, P, spec, kernel=kernel)
            if states is not None:
                states[..., s, :] = x

        tangent = fx if jacobian == "exact" else fx / _dz_dx(x, spec.N)
        _, Jf = jvp(_f, (x,), (tangent,))
        x = x + dt * fx + 0.5 * dt * dt * Jf

    if states is not None:
        return y, states
    return y


def build_input(
    spec: ModelSpec,
    onsets: list[list[float]],
    durations: list[list[float]],
    n_micro: int,
) -> torch.Tensor:
    """Boxcar input matrix at microtime resolution.

    ``onsets[c]`` / ``durations[c]`` are in seconds for input column ``c``.
    Modulatory columns come first, then driving columns -- ``B[m]`` multiplies
    column ``m``, and ``C`` is usually zero on the modulatory columns so the
    modulation acts only through the self-connection.
    """
    if len(onsets) != len(durations):
        raise ValueError("onsets and durations must have the same number of columns")
    u = torch.zeros(n_micro, len(onsets), dtype=spec.dtype, device=spec.device)
    for c, (ons, durs) in enumerate(zip(onsets, durations, strict=True)):
        if len(durs) == 1 and len(ons) > 1:
            durs = list(durs) * len(ons)
        for o, d in zip(ons, durs, strict=True):
            start = int(round(o / spec.dt))
            # A zero-duration event still occupies one bin; otherwise an
            # impulse specified as duration 0 would vanish entirely.
            stop = max(start + int(round(d / spec.dt)), start + 1)
            u[start : min(stop, n_micro), c] = 1.0
    return u
