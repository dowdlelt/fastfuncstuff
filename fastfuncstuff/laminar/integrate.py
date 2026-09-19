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

from typing import Literal, overload

import torch
from torch.func import jvp

from fastfuncstuff._compile import safe_compile
from fastfuncstuff.laminar.forward import ForwardCache, build_cache, g_obs, rhs
from fastfuncstuff.laminar.params import ModelSpec


def sample_indices(n_micro: int, n_scans: int, delay_bins: int) -> torch.Tensor:
    """Microtime index at which each output scan is observed.

    Replicates ``spm_int_IT``: ``ceil((0:v-1)*u/v) + D``, where ``D`` is the
    slice-timing delay in microtime bins (``TR/2`` in every published use).

    The reference indexes its microtime axis from 1, so the same instant is one
    bin earlier here. Getting this wrong samples every TR 50 ms late, which is
    invisible in the shape of the response and enough to move the free energy
    by ~18 nats.
    """
    n = torch.arange(n_scans, dtype=torch.float64)
    return torch.ceil(n * n_micro / n_scans).long() + delay_bins - 1


def _dz_dx(x: torch.Tensor, N: int) -> torch.Tensor:
    """d(exponentiated state)/d(stored state): 1 for the linear states, exp(x)
    for the log-scaled ones (CBF and all four vascular states)."""
    d = torch.ones_like(x)
    d[..., 3 * N :] = torch.exp(x[..., 3 * N :])
    return d


def _taylor_step(
    x: torch.Tensor,
    u_i: torch.Tensor,
    cache: ForwardCache,
    N: int,
    K: int,
    dt: float,
    reference_jacobian: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One Ito-Taylor step. Returns the advanced state and the derivative."""

    def _f(xx: torch.Tensor) -> torch.Tensor:
        return rhs(xx, u_i, cache, N, K)

    fx = _f(x)
    tangent = fx / _dz_dx(x, N) if reference_jacobian else fx
    _, Jf = jvp(_f, (x,), (tangent,))
    return x + dt * fx + 0.5 * dt * dt * Jf, fx


# Eager, this step costs ~34 ms: PyTorch has no native forward-mode rule for most
# of these ops and falls back to Python `_refs` decompositions at ~230 us apiece,
# against ~200 ops. Compiled, dynamo traces the dual computation and the unrolled
# depth loop into one fused graph -- ~245 us, a 137x speedup, agreeing with eager
# to 1e-13. This is the difference between an inversion taking minutes and hours.
#
# dynamic=False is load-bearing. Measured on this step, `dynamic=True` compiles
# once and never recompiles, but runs at 28-57 ms -- eager speed. Generalizing the
# shapes defeats the fusion entirely, so the only fast path is a static
# specialization per shape, and the shape count is managed by `batch_bucket`.
_compiled_step = safe_compile(_taylor_step, dynamic=False)


def batch_bucket(n: int) -> int:
    """Round a batch width up to the next power of two, floored at 16.

    Every distinct batch width is a separate static compile of ``_taylor_step``
    (~20 s each), so callers that vary their batch -- the model space, where the
    width is one per free parameter, or per-parcel fitting -- must not pass the
    raw count through. Bucketing trades padded rows for compiled variants.

    The padding is close to free because the step is host-bound -- ~200 tiny ops
    on a few dozen states, so per-op dispatch dominates the arithmetic. That
    regime exists on any machine; only its width is hardware-specific. Measured
    at K=9, float64, on a 6-core Xeon W-2133: batch 1 costs 205 us and batch 64
    costs 286 us, so 64x the rows cost 1.4x the time; the plateau breaks around
    128 and per-row cost turns back upward by 512 as it becomes compute-bound.

    Those numbers justify the padding but are deliberately not encoded here: the
    bucket has no cap and no measured constant, so it stays correct wherever a
    given machine's plateau sits. Code that needs to *choose* a batch size -- per
    parcel fitting, say -- should measure or derive it rather than hardcode one,
    the way ``memory.py`` derives chunk sizes instead of assuming them.
    """
    if n <= 16:
        return 16
    return 1 << (n - 1).bit_length()


def delay_bins(spec: ModelSpec, delay_seconds: float | None = None) -> int:
    if delay_seconds is None:
        delay_seconds = spec.TR / 2.0
    return max(int(round(delay_seconds / spec.dt)), 1)


@overload
def integrate(
    u: torch.Tensor,
    P: dict[str, torch.Tensor],
    spec: ModelSpec,
    n_scans: int,
    *,
    kernel: torch.Tensor | None = None,
    delay_seconds: float | None = None,
    x0: torch.Tensor | None = None,
    shift_depths: bool = False,
    jacobian: str = "reference",
    compile: bool = True,
    return_states: Literal[False] = ...,
) -> torch.Tensor: ...


@overload
def integrate(
    u: torch.Tensor,
    P: dict[str, torch.Tensor],
    spec: ModelSpec,
    n_scans: int,
    *,
    kernel: torch.Tensor | None = None,
    delay_seconds: float | None = None,
    x0: torch.Tensor | None = None,
    shift_depths: bool = False,
    jacobian: str = "reference",
    compile: bool = True,
    return_states: Literal[True],
) -> tuple[torch.Tensor, torch.Tensor]: ...


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
    compile: bool = True,
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
    compile : route the hot step through ``torch.compile`` (a ~137x speedup;
        falls back to eager automatically if inductor is unavailable).

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

    # Built once: the baseline hemodynamics, the depth mapping and every
    # P0*exp(P) expansion are identical at all n_micro steps.
    cache = build_cache(spec, P, shift_depths=shift_depths)

    dt = spec.dt
    y = torch.zeros(*batch, n_scans, spec.K, dtype=spec.dtype, device=spec.device)
    states = (
        torch.zeros(*batch, n_scans, spec.n_states, dtype=spec.dtype, device=spec.device)
        if return_states
        else None
    )

    step = _taylor_step if not compile else _compiled_step
    ref_jac = jacobian == "reference"
    for i in range(n_micro):
        # The observation is read *before* the state advances, matching the
        # reference's ordering inside the integration loop.
        s = int(scan_at[i])
        if s >= 0:
            y[..., s, :] = g_obs(x, P, spec, kernel=kernel, cache=cache)
            if states is not None:
                states[..., s, :] = x
        x, _ = step(x, u[..., i, :], cache, spec.N, spec.K, dt, ref_jac)

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
            # SPM's onsets and offsets are both *inclusive*, so a 1.6 s event at
            # dt = 0.05 occupies 33 bins, not 32. Getting this wrong shortens
            # every event by one bin, which is small enough to look like a fit
            # that nearly converged and large enough to move sigma by 4%.
            stop = start + int(round(d / spec.dt)) + 1
            u[start : min(stop, n_micro), c] = 1.0
    return u
