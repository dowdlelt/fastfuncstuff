"""
Parameter structures for the laminar BOLD generative model.

Two kinds of parameter live here, and keeping them apart is the whole point of
the module:

``P0`` -- fixed baseline values (physiology, MR constants, priors' anchor
points). These are the numbers in Table 2 of Havlicek & Uludag (2020).

``P``  -- the estimated deviations. Every physiological parameter enters as
``P0 * exp(P)``, so ``P = 0`` recovers the baseline and the estimation problem
is unconstrained-positive on a log scale. This is the SPM convention and it is
what makes the Gaussian priors of Variational Laplace sensible.

References
----------
Havlicek M & Uludag K (2020). A dynamical model of the laminar BOLD response.
    NeuroImage 204:116209.
Havlicek M et al. (2015). Physiologically informed dynamic causal modeling of
    fMRI data. NeuroImage 122:355-372.

Reference implementation: ``LBR_param_priors.m`` in the laminar_BOLD_model /
predictive_tones MATLAB trees.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import torch

# The model is inverted by Gauss-Newton ascent on a free energy whose curvature
# is the posterior covariance; float32 loses the Hessian. Unlike the rest of the
# toolbox this module is float64 by default (see [[Float32 vs float64]]).
DEFAULT_DTYPE = torch.float64


@dataclass
class P0:
    """Fixed baseline parameters. Values from LBR_param_priors.m."""

    # Neuronal (Havlicek 2015). Note sigma=3 here, not the 0.5 Hz of the 2015
    # paper -- the laminar reparametrisation differs, priors do not transfer.
    sigma: float = 3.0
    mu: float = 1.5
    lam: float = 0.1

    # Neurovascular coupling
    c1: float = 0.6
    c2: float = 1.5
    c3: float = 0.6

    # Width of the Gaussian neuronal->vascular depth mapping, as a fraction
    nsig: float = 0.005

    # Baseline hemodynamics
    V0t: float = 3.5  # total GM CBV0 (mL)
    w_v: float = 0.5  # fraction of CBV0 in venules vs ascending vein
    s_v: float = 0.0  # slope of CBV0 increase toward surface, venules
    s_d: float = 0.4  # ... ascending vein. The parameter that matters most.
    s_d0: float = 0.0  # knee position for the piecewise-linear AV slope
    s_d2: float = 0.0  # extra slope past the knee
    t0v: float = 1.0  # venule transit time (s)
    E0v: float = 0.35
    E0d: float = 0.35

    # Steady-state couplings
    al_v: float = 0.3  # Grubb exponent, venules
    al_d: float = 0.2  # ... ascending vein
    nr: float = 4.0  # n-ratio, (cbf-1)/(cmro2-1)

    # Viscoelastic CBF-CBV uncoupling time constants
    tau_v_in: float = 2.0
    tau_v_de: float = 2.0
    tau_d_in: float = 2.0
    tau_d_de: float = 2.0

    # Whether inflation and deflation share a time constant. The reference
    # implementation switches tau on the sign of dv/dt using the *previous*
    # call's derivative held in a MATLAB `persistent`, which makes the right-
    # hand side non-Markovian. Faes et al. set both flags to 1, which disables
    # the branch entirely; we only support that case (see forward.py).
    tau_v_same: bool = True
    tau_d_same: bool = True

    # BOLD signal equation (7T, GE, TE ~ 28 ms)
    Hct_v: float = 0.35
    Hct_d: float = 0.38
    gyro: float = 2 * torch.pi * 42.6e6
    suscep: float = 0.264e-6
    rho_t: float = 0.89
    R2s_t: float = 34.0
    R2s_v: float = 80.0
    R2s_d: float = 85.0
    r0v: float = 228.0
    r0d: float = 232.0

    @property
    def rho_v(self) -> float:
        return 0.95 - self.Hct_v * 0.22

    @property
    def rho_d(self) -> float:
        return 0.95 - self.Hct_d * 0.22


#: Every estimable parameter, with its shape rule.
#: "scalar"  -- one value
#: "N"       -- one per neuronal depth
#: "K"       -- one per vascular depth
#: "NxN"     -- neuronal connectivity
#:
#: Note the reference also allows fully depth-specific CBV0 fractions (`x_v`,
#: `x_d`, one value per depth). They are deliberately absent: they defeat the
#: slope parameterisation that keeps model complexity independent of K, and no
#: published application uses them.
PARAM_SHAPES: dict[str, str] = {
    # neuronal
    "A": "NxN",
    "B": "NxNxM",
    "C": "NxU",
    "sigma": "scalar",
    "mu": "scalar",
    "lam": "scalar",
    "Bmu": "NxM",
    "Blam": "1xM",
    # neurovascular coupling
    "c1": "scalar",
    "c2": "scalar",
    "c3": "scalar",
    # neuronal -> vascular depth mapping
    "s": "scalar",
    "nsig": "scalar",
    "nb": "scalar",
    # baseline hemodynamics
    "V0t": "scalar",
    "w_v": "scalar",
    "s_v": "scalar",
    "s_d": "scalar",
    "s_d0": "scalar",
    "s_d2": "scalar",
    "t0v": "scalar",
    "E0v": "scalar",
    "E0d": "scalar",
    # couplings
    "al_v": "scalar",
    "al_d": "scalar",
    "nr": "scalar",
    "tau_v_in": "scalar",
    "tau_v_de": "scalar",
    "tau_d_in": "scalar",
    "tau_d_de": "scalar",
}


@dataclass
class ModelSpec:
    """Structure of one laminar model: sizes, timing, and the fixed baselines.

    This is the ``M`` struct of the reference implementation, minus the priors.
    """

    N: int  # neuronal depths (3 in every published application)
    K: int  # vascular / BOLD depths (7-11; see Uludag & Havlicek 2021)
    n_inputs: int = 1  # columns of the driving-input matrix u
    n_mod: int = 1  # modulatory inputs (columns of B's third axis)
    TE: float = 0.028
    B0: float = 7.0
    dt: float = 0.05  # microtime step (s) used by the integrator
    TR: float = 1.6
    p0: P0 = field(default_factory=P0)
    dtype: torch.dtype = DEFAULT_DTYPE
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    @property
    def n_states(self) -> int:
        return 4 * self.N + 4 * self.K

    @property
    def depths(self) -> torch.Tensor:
        """Depth-bin centres as a percentage, superficial (k=0) first.

        ``linspace(0, 100, 2K+1)[1::2]`` -- the midpoints of K equal bins.
        """
        edges = torch.linspace(0, 100, 2 * self.K + 1, dtype=self.dtype, device=self.device)
        return edges[1::2]


def zero_params(spec: ModelSpec, batch: tuple[int, ...] = ()) -> dict[str, torch.Tensor]:
    """A full parameter set at the prior mean (all log-deviations zero).

    ``batch`` prepends batch dimensions; every entry is broadcastable against
    the state tensor, so one call to the forward model can evaluate a whole
    grid of parameter sets at once. The forward model is batch-first by
    construction because the GPU only pays off across models, depths, subjects
    and restarts -- never within a single 50-state ODE.
    """
    kw = {"dtype": spec.dtype, "device": spec.device}
    N, K, M, U = spec.N, spec.K, spec.n_mod, spec.n_inputs
    out: dict[str, torch.Tensor] = {}
    for name, shape in PARAM_SHAPES.items():
        if shape == "scalar":
            trailing: tuple[int, ...] = ()
        elif shape == "N":
            trailing = (N,)
        elif shape == "K":
            trailing = (K,)
        elif shape == "NxN":
            trailing = (N, N)
        elif shape == "NxNxM":
            trailing = (M, N, N)  # leading M so batched matmuls stay natural
        elif shape == "NxU":
            trailing = (N, U)
        elif shape == "NxM":
            trailing = (N, M)
        elif shape == "1xM":
            trailing = (1, M)
        else:  # pragma: no cover - guarded by the table above
            raise ValueError(f"unknown shape rule {shape!r} for {name!r}")
        out[name] = torch.zeros(batch + trailing, **kw)
    return out


def p0_tensors(spec: ModelSpec) -> dict[str, torch.Tensor]:
    """P0 scalars as tensors on the spec's device/dtype, for broadcasting."""
    kw = {"dtype": spec.dtype, "device": spec.device}
    out = {}
    for f in fields(spec.p0):
        val = getattr(spec.p0, f.name)
        if isinstance(val, bool):
            continue
        out[f.name] = torch.as_tensor(float(val), **kw)
    out["rho_v"] = torch.as_tensor(spec.p0.rho_v, **kw)
    out["rho_d"] = torch.as_tensor(spec.p0.rho_d, **kw)
    return out
