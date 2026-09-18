"""
Parameter recovery: does a laminar model comparison mean what it claims?

Parity testing proves we compute the same numbers the reference does. It cannot
prove those numbers identify anything, because the reference is the thing being
checked against -- both implementations could be faithfully, identically unable
to tell a middle-layer effect from a deep one at realistic SNR. This module asks
the separate question: simulate data from a *known* generator at the *measured*
noise level, invert, and see whether the truth wins.

Three experiments, in increasing order of how much they can hurt:

:func:`recover_model_space`
    Simulate a known modulation pattern, fit the whole model space, check the
    true model wins. Repeat over noise draws for a recovery rate. This is the
    gate on reading a layer assignment off a real dataset.

:func:`recover_drive_profile`
    The same for a *single condition*, where the hypothesis is which depths are
    driven rather than which are modulated. Harder, because fitting two
    conditions jointly lets systematic vascular error cancel between them and a
    single condition has nothing to cancel against.

:func:`s_d_misspecification_profile`
    The falsification. Uludag & Havlicek report an asymmetry: *underestimating*
    the ascending-vein baseline-CBV slope gives qualitatively wrong laminar
    profiles, while overestimating is safe. If our port does not fail in that
    direction, it is wrong somewhere no parity test can see -- so a *passing*
    result here is the one that would be suspicious.

The noise level is not invented. SPM's hyperparameters scale the precision of
each depth's error, so the fitted ``Eh`` from a real inversion gives the noise
standard deviation directly as ``exp(-Eh/2)``. On the published ROI that is
~0.075% signal change per depth against a data standard deviation of ~0.62.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from fastfuncstuff.laminar.experiment import (
    drive_priors,
    faes_priors,
    layer_model_names,
    layer_model_targets,
    posterior_model_probabilities,
)
from fastfuncstuff.laminar.integrate import integrate
from fastfuncstuff.laminar.inversion import variational_laplace
from fastfuncstuff.laminar.params import ModelSpec


def noise_sd_from_hyperparameters(Eh: torch.Tensor) -> torch.Tensor:
    """Per-depth noise standard deviation implied by a fitted inversion.

    SPM parameterises the error *precision*: ``iS = sum_i exp(h_i) Q_i``, with
    ``Q_i`` selecting depth ``i``. So the variance at depth ``i`` is
    ``exp(-h_i)`` and the standard deviation is ``exp(-h_i/2)``.

    Getting this backwards inflates the simulated noise by the square of the
    precision -- on the published ROI, ``sqrt(exp(Eh))`` gives ~13% signal
    change per depth against data whose own standard deviation is ~0.62%, which
    is obviously wrong and is the check that catches the error.
    """
    return torch.exp(-torch.as_tensor(Eh) / 2.0)


@dataclass
class RecoveryResult:
    """One simulated dataset, inverted against the whole model space."""

    true_index: int
    won_index: int
    F: list[float]
    probs: list[float]
    names: list[str] = field(default_factory=list)

    @property
    def correct(self) -> bool:
        return self.true_index == self.won_index

    @property
    def margin(self) -> float:
        """Free energy of the winner minus that of the true model.

        Zero when the truth wins. Positive values are how many nats of evidence
        the wrong answer had -- a large margin is a confident error, which is
        worse than an uncertain one.
        """
        return self.F[self.won_index] - self.F[self.true_index]


def simulate_laminar_data(
    spec: ModelSpec,
    P_true: dict[str, torch.Tensor],
    u: torch.Tensor,
    n_scans: int,
    *,
    noise_sd: torch.Tensor | float,
    kernel: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    **integrate_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Noiseless prediction plus an independent noise draw.

    ``noise_sd`` is per depth (shape ``(K,)``) or a scalar. Noise is white across
    time and independent across depths, matching the error model the inversion
    assumes -- which makes this a test of identifiability, not of robustness to
    a misspecified error structure. Correlated or drifting noise is a harder and
    separate question.
    """
    clean = integrate(u, P_true, spec, n_scans, kernel=kernel, **integrate_kwargs)
    sd = torch.as_tensor(noise_sd, dtype=spec.dtype, device=spec.device)
    noise = torch.randn(clean.shape, dtype=spec.dtype, device=spec.device, generator=generator)
    return clean + noise * sd, clean


def _true_params(
    spec: ModelSpec,
    base: dict[str, torch.Tensor],
    target: tuple[int, ...],
    amplitude: float,
    *,
    on: str = "B",
) -> dict[str, torch.Tensor]:
    """Ground-truth parameters: a realistic fit with one pattern imposed."""
    P = {k: v.clone() for k, v in base.items()}
    t = torch.as_tensor(list(target), dtype=spec.dtype, device=spec.device)
    if on == "B":
        P["B"] = torch.diag(t * amplitude)[None]
    elif on == "C":
        C = torch.zeros_like(P["C"])
        C[:, -1] = t * amplitude
        P["C"] = C
    else:
        raise ValueError(f"on must be 'B' or 'C', got {on!r}")
    return P


def recover_model_space(
    spec: ModelSpec,
    base_params: dict[str, torch.Tensor],
    u: torch.Tensor,
    n_scans: int,
    *,
    noise_sd: torch.Tensor | float,
    true_target: tuple[int, ...],
    amplitude: float,
    rows: torch.Tensor | None = None,
    kernel: torch.Tensor | None = None,
    seed: int = 0,
    on: str = "B",
    **vl_kwargs,
) -> RecoveryResult:
    """Simulate one dataset from a known generator and fit the whole space.

    ``on="B"`` tests modulation recovery (two conditions); ``on="C"`` tests
    drive recovery from a single condition.
    """
    targets = layer_model_targets(spec.N)
    names = layer_model_names(spec.N)
    true_index = targets.index(tuple(true_target))

    P_true = _true_params(spec, base_params, tuple(true_target), amplitude, on=on)
    gen = torch.Generator(device=spec.device).manual_seed(seed)
    y, _ = simulate_laminar_data(
        spec, P_true, u, n_scans, noise_sd=noise_sd, kernel=kernel, generator=gen
    )

    priors_fn = faes_priors if on == "B" else drive_priors
    F = []
    for target in targets:
        res = variational_laplace(
            spec, priors_fn(spec, target), u, y, rows=rows, kernel=kernel, **vl_kwargs
        )
        F.append(float(res.F))
    probs = posterior_model_probabilities(F).tolist()
    return RecoveryResult(
        true_index=true_index,
        won_index=int(max(range(len(F)), key=lambda i: F[i])),
        F=F,
        probs=probs,
        names=names,
    )


def recover_with_fixed_s_d(
    spec: ModelSpec,
    y: torch.Tensor,
    u: torch.Tensor,
    *,
    true_s_d: float,
    offsets: list[float],
    target: tuple[int, ...] = (1, 1, 1),
    rows: torch.Tensor | None = None,
    kernel: torch.Tensor | None = None,
    **vl_kwargs,
) -> dict[str, torch.Tensor]:
    """Invert with the ascending-vein CBV slope *pinned* at a wrong value.

    This is the falsification proper. :func:`s_d_misspecification_profile` only
    perturbs the forward model; here ``s_d`` is fixed (prior variance zero) at
    ``true_s_d + offset`` and the inversion has to explain the data without it,
    so the error is absorbed by the parameters that are still free -- including
    the modulation ``B`` that a result would be read off.

    ``s_d`` is log-scaled: an offset of -0.5 assumes a slope ``exp(-0.5)`` times
    the fitted one.

    Returns the recovered per-depth ``B`` at each offset, the free energy, and
    the correlation of each recovered profile with the one from the unbiased
    fit. The documented asymmetry is that underestimating the slope distorts the
    recovered laminar profile more than overestimating it does.
    """
    n_scans = int(y.shape[-2])
    B, F = [], []
    for off in offsets:
        priors = faes_priors(spec, target)
        priors.pE["s_d"] = torch.tensor(true_s_d + off, dtype=spec.dtype, device=spec.device)
        priors.pC["s_d"] = torch.zeros((), dtype=spec.dtype, device=spec.device)
        res = variational_laplace(spec, priors, u, y, rows=rows, kernel=kernel, **vl_kwargs)
        B.append(torch.diagonal(res.Ep["B"][0]))
        F.append(res.F)
    stacked = torch.stack(B)
    zero_at = offsets.index(0.0) if 0.0 in offsets else len(offsets) // 2
    ref = stacked[zero_at]

    def _corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a - a.mean()
        b = b - b.mean()
        return (a @ b) / torch.clamp(a.norm() * b.norm(), min=1e-300)

    _ = n_scans
    return {
        "offsets": torch.as_tensor(offsets, dtype=spec.dtype),
        "B": stacked,
        "F": torch.stack(F),
        "reference": ref,
        "correlation": torch.stack([_corr(p, ref) for p in stacked]),
        "peak_depth": stacked.argmax(dim=-1),
    }


def s_d_misspecification_profile(
    spec: ModelSpec,
    base_params: dict[str, torch.Tensor],
    u: torch.Tensor,
    n_scans: int,
    *,
    offsets: list[float],
    kernel: torch.Tensor | None = None,
    **integrate_kwargs,
) -> dict[str, torch.Tensor]:
    """Laminar BOLD profiles when the ascending-vein CBV slope is wrong.

    ``s_d`` is log-scaled, so an ``offset`` of -0.5 means the assumed slope is
    ``exp(-0.5)`` times the true one.

    Returns the peak depth profile at each offset, plus the correlation of each
    against the true-``s_d`` profile. The documented asymmetry is that negative
    offsets (underestimating the slope) distort the profile much more than
    positive ones of the same size.
    """
    profiles = []
    for off in offsets:
        P = {k: v.clone() for k, v in base_params.items()}
        P["s_d"] = P["s_d"] + off
        y = integrate(u, P, spec, n_scans, kernel=kernel, **integrate_kwargs)
        profiles.append(y.abs().max(dim=-2).values)
    stacked = torch.stack(profiles)

    zero_at = offsets.index(0.0) if 0.0 in offsets else len(offsets) // 2
    reference = stacked[zero_at]

    def _corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a - a.mean()
        b = b - b.mean()
        return (a @ b) / torch.clamp(a.norm() * b.norm(), min=1e-300)

    return {
        "offsets": torch.as_tensor(offsets, dtype=spec.dtype),
        "profiles": stacked,
        "reference": reference,
        "correlation": torch.stack([_corr(p, reference) for p in stacked]),
        "peak_depth": stacked.argmax(dim=-1),
    }
