"""
The study layer: model spaces, Bayesian parameter averaging, model comparison.

A *model* in this framework is a pattern of prior variances, not a different set
of equations. The eight-model space of Faes et al. asks which of the three
neuronal depths carry a modulatory effect on the excitatory self-connection --
the null, each depth alone, each pair, and all three -- by letting exactly those
entries of ``B`` have prior variance and pinning the rest at zero.

That shape is what makes the whole thing batchable: every member integrates the
same equations, so the model space is a batch dimension, not a loop over
different code.

Ported from ``apply_laminar_BOLD_model.m`` and SPM's ``spm_dcm_bpa.m``.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import torch

from fastfuncstuff.laminar.inversion import (
    Priors,
    VLResult,
    spm_inv,
    unvec,
    variational_laplace,
    vec,
)
from fastfuncstuff.laminar.params import ModelSpec, zero_params

DEPTH_NAMES = ("superficial", "middle", "deep")


def layer_model_names(N: int = 3) -> list[str]:
    """Names of the 2^N modulation-target hypotheses, null first."""
    names = []
    for r in range(N + 1):
        for combo in itertools.combinations(range(N), r):
            names.append("null" if not combo else "+".join(DEPTH_NAMES[i] for i in combo))
    return names


def layer_model_targets(N: int = 3) -> list[tuple[int, ...]]:
    """Binary modulation-target patterns, aligned with :func:`layer_model_names`."""
    out = []
    for r in range(N + 1):
        for combo in itertools.combinations(range(N), r):
            out.append(tuple(1 if i in combo else 0 for i in range(N)))
    return out


def faes_priors(
    spec: ModelSpec,
    target: tuple[int, ...] | list[int],
    *,
    b_variance: float = math.exp(0.5),
) -> Priors:
    """Priors for one member of the Faes et al. model space.

    Estimated: the driving-input strength per neuronal depth, the depth-mapping
    width ``nsig``, the neuronal time constant ``sigma``, the ascending vein's
    Grubb exponent ``al_d`` and baseline-CBV slope ``s_d``, and the modulation
    ``B`` at whichever depths ``target`` marks.

    ``mu`` and ``lam`` carry non-zero prior *means* but zero variance: the
    adaptation they produce shapes the post-stimulus undershoot, which a short
    inter-trial interval cannot observe, so fitting them is not identifiable.
    """
    N = spec.N
    pE = zero_params(spec)
    pE["C"] = torch.tensor([[0.0, 1.0]] * N, dtype=spec.dtype, device=spec.device)
    pE["mu"] = torch.tensor(-0.8, dtype=spec.dtype, device=spec.device)
    pE["lam"] = torch.tensor(1.8, dtype=spec.dtype, device=spec.device)

    pC = zero_params(spec)
    pC["C"] = torch.tensor([[0.0, 1.0]] * N, dtype=spec.dtype, device=spec.device)
    pC["nsig"] = torch.tensor(math.exp(-4), dtype=spec.dtype, device=spec.device)
    pC["sigma"] = torch.tensor(math.exp(-2), dtype=spec.dtype, device=spec.device)
    pC["al_d"] = torch.tensor(math.exp(-5), dtype=spec.dtype, device=spec.device)
    pC["s_d"] = torch.tensor(math.exp(-1), dtype=spec.dtype, device=spec.device)
    pC["B"] = (
        torch.diag(torch.tensor(list(target), dtype=spec.dtype, device=spec.device))[None]
        * b_variance
    )
    return Priors(pE=pE, pC=pC)


def drive_priors(
    spec: ModelSpec,
    target: tuple[int, ...] | list[int],
    *,
    c_variance: float = 1.0,
) -> Priors:
    """Priors for a **single-condition** model space, on the drive ``C``.

    :func:`faes_priors` asks which depths are *modulated* between two conditions,
    which needs two conditions to compare. This asks which depths are *driven* at
    all, from one condition alone -- "where does the input arrive?" rather than
    "what changed?". ``C`` is a linear per-depth gain on the driving input, so
    freeing it at a subset of depths is the same shape of hypothesis, moved from
    ``B`` to ``C``.

    Written for experiments where the conditions cannot be contrasted because
    they are not the same kind of event -- bottom-up perception against top-down
    imagery, say, where one has an enormous driving input and the other a weak
    endogenous signal, and their difference is not interpretable as a modulation.

    **This is the harder inference and it is worth knowing why.** Fitting two
    conditions jointly lets systematic error in the vascular model cancel between
    them. A single condition has nothing to cancel against, so the inferred
    laminar profile rests entirely on the ascending-vein parameters being right
    -- and the documented failure mode is asymmetric: *underestimating* ``s_d``
    gives qualitatively wrong laminar profiles, while overestimating is safe.

    For a strong and a weak condition in the same session there is usually a
    better route than fitting the weak one: the vasculature belongs to the tissue
    and not to the task, so fit the strong condition, then take the operator from
    :func:`~fastfuncstuff.laminar.devein.laminar_impulse_response` at the weak
    condition's amplitude and apply it. That spends none of the weak condition's
    SNR on estimating physiology.
    """
    N = spec.N
    t = torch.as_tensor(list(target), dtype=spec.dtype, device=spec.device)

    pE = zero_params(spec)
    pC = zero_params(spec)
    # The driving input is the last column; a modulatory column, if present, is
    # left at zero because a single condition has nothing to modulate.
    drive_mean = torch.zeros(N, spec.n_inputs, dtype=spec.dtype, device=spec.device)
    drive_var = torch.zeros_like(drive_mean)
    drive_mean[:, -1] = t
    drive_var[:, -1] = t * c_variance
    pE["C"] = drive_mean
    pC["C"] = drive_var

    pE["mu"] = torch.tensor(-0.8, dtype=spec.dtype, device=spec.device)
    pE["lam"] = torch.tensor(1.8, dtype=spec.dtype, device=spec.device)

    pC["nsig"] = torch.tensor(math.exp(-4), dtype=spec.dtype, device=spec.device)
    pC["sigma"] = torch.tensor(math.exp(-2), dtype=spec.dtype, device=spec.device)
    pC["al_d"] = torch.tensor(math.exp(-5), dtype=spec.dtype, device=spec.device)
    pC["s_d"] = torch.tensor(math.exp(-1), dtype=spec.dtype, device=spec.device)
    return Priors(pE=pE, pC=pC)


@dataclass
class BPAResult:
    """Posterior averaged over vascular resolutions."""

    Ep: dict[str, torch.Tensor]
    Cp: torch.Tensor
    Vp: dict[str, torch.Tensor]  # posterior variances, in parameter structure
    Pp: dict[str, torch.Tensor]  # P(parameter differs from zero)
    F: torch.Tensor
    free_idx: torch.Tensor
    per_k_F: list[float] = field(default_factory=list)  # noqa: N815  (SPM's name)


def bayesian_parameter_average(
    results: list[VLResult],
    spec: ModelSpec,
    *,
    free_energy: str = "reference",
) -> BPAResult:
    """Combine one model's inversions across vascular resolutions.

    A precision-weighted average: each K's posterior contributes in proportion to
    how sharply it determined the parameters. Faes et al. average *then* compare,
    not the other way round, so the model comparison ranks averaged models.

    ``free_energy`` selects which F the averaged model carries:

    ``"reference"``
        the first K's, which is what ``spm_dcm_bpa`` actually leaves in place.
        Its caller's comment says "accumulated over different number of BOLD
        depths", but the field is never touched after ``BPA = DCM`` on the first
        iteration. Use this for parity with published results.
    ``"sum"`` / ``"mean"``
        what that comment describes. Note neither is a proper joint log-evidence:
        each K is a different sampling of the same voxels, so the Ks are not
        independent observations of new data.
    """
    if not results:
        raise ValueError("no inversions to average")
    free_idx = results[0].free_idx
    for r in results[1:]:
        if not torch.equal(r.free_idx, free_idx):
            raise ValueError("inversions must share a parameter structure to be averaged")

    precisions = torch.stack([spm_inv(r.Cp) for r in results])
    Cp = spm_inv(precisions.sum(0))
    reduced = torch.stack([vec(r.Ep)[free_idx] for r in results])
    Ep_reduced = Cp @ (precisions @ reduced.unsqueeze(-1)).sum(0).squeeze(-1)

    # Parameters outside the estimated set have no posterior to weight by, so
    # they carry the plain mean across K (identical in practice: they are fixed).
    mean_full = torch.stack([vec(r.Ep) for r in results]).mean(0)
    Ep_full = mean_full.clone()
    Ep_full[free_idx] = Ep_reduced

    Vp_full = torch.zeros_like(Ep_full)
    Vp_full[free_idx] = torch.diagonal(Cp)
    # P(parameter != 0) under the Gaussian posterior, SPM's one-sided form.
    sd = torch.sqrt(torch.clamp(Vp_full, min=0))
    z = torch.where(sd > 0, Ep_full.abs() / torch.clamp(sd, min=1e-300), torch.zeros_like(sd))
    Pp_full = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    Pp_full = torch.where(sd > 0, Pp_full, torch.zeros_like(Pp_full))

    per_k = [float(r.F) for r in results]
    if free_energy == "reference":
        F = results[0].F
    elif free_energy == "sum":
        F = torch.stack([r.F for r in results]).sum()
    elif free_energy == "mean":
        F = torch.stack([r.F for r in results]).mean()
    else:
        raise ValueError(f"free_energy must be 'reference', 'sum' or 'mean', got {free_energy!r}")

    return BPAResult(
        Ep=unvec(Ep_full, spec),
        Cp=Cp,
        Vp=unvec(Vp_full, spec),
        Pp=unvec(Pp_full, spec),
        F=F,
        free_idx=free_idx,
        per_k_F=per_k,
    )


def posterior_model_probabilities(F: torch.Tensor | list[float]) -> torch.Tensor:
    """Softmax of the free energies -- the form a model comparison is reported in.

    Free energy is a log-evidence bound, so differences are log Bayes factors and
    a difference of 3 is already decisive.

    Shifted by the **maximum**, not the minimum. The reference subtracts the
    minimum, which leaves the best model's exponent as large as the whole spread
    of F and overflows to ``inf`` once that spread exceeds ~709 nats, returning
    ``nan`` for every model. Subtracting the maximum makes every exponent <= 0
    and cannot overflow. The two agree exactly wherever the reference does not
    overflow, so this is strictly safer and changes no published number: on the
    example ROI the spread is 232 nats, comfortably inside the safe range, but
    a stronger effect or a wider model space would not be.
    """
    f = torch.as_tensor(F, dtype=torch.float64)
    d = f - f.max()
    e = torch.exp(d)
    return e / e.sum()


def depth_inclusion_probabilities(
    F: torch.Tensor | list[float],
    targets: list[tuple[int, ...]] | None = None,
    N: int = 3,
) -> torch.Tensor:
    """P(depth is modulated), marginalised over the whole model space.

    The posterior probability of each depth being in the true model, summing
    every model that contains it -- Bayesian model averaging over the model
    space rather than winner-takes-all.

    **Prefer this to reporting the winning model.** Parameter recovery at the
    measured noise level of the published ROI shows the single-winner readout
    over-selects: across twelve simulations from a known single-depth
    generator, the true depth was inside the winning model 12/12 times, but the
    winner carried a *spurious extra depth* 4/12 times, on margins of 0.6-1.2
    nats. So localisation is reliable and parsimony is not, and a winner-takes-
    all report converts a weak preference into a categorical claim.

    Free energy over-selecting complexity is not new here -- ``ffs_bsds`` hit
    the same thing, where raw free energy chose an extra state and the fix was a
    different criterion rather than more restarts. Marginalising is the cheap
    half of that fix; held-out validation is the other half.
    """
    f = torch.as_tensor(F, dtype=torch.float64)
    if targets is None:
        targets = layer_model_targets(N)
    probs = posterior_model_probabilities(f)
    membership = torch.tensor([[float(v) for v in t] for t in targets], dtype=torch.float64)
    return probs @ membership


def fit_model_space(
    spec_for_k: dict[int, ModelSpec],
    data_for_k: dict[int, torch.Tensor],
    u_for_k: dict[int, torch.Tensor],
    *,
    kernel_for_k: dict[int, torch.Tensor] | None = None,
    rows: torch.Tensor | None = None,
    targets: list[tuple[int, ...]] | None = None,
    names: list[str] | None = None,
    free_energy: str = "reference",
    progress: bool = True,
    **vl_kwargs,
) -> tuple[list[str], list[BPAResult], list[list[VLResult]]]:
    """Invert every model at every vascular resolution, then average across K.

    Returns the model names, one :class:`BPAResult` per model, and the raw
    per-K inversions.
    """
    ks = sorted(spec_for_k)
    n_neuronal = spec_for_k[ks[0]].N
    if targets is None:
        targets = layer_model_targets(n_neuronal)
    if names is None:
        names = layer_model_names(n_neuronal)

    bar = None
    if progress:
        try:
            from tqdm.auto import tqdm

            bar = tqdm(total=len(targets) * len(ks), desc="laminar model space", leave=True)
        except ImportError:  # pragma: no cover - tqdm is a soft dependency
            bar = None

    per_model: list[list[VLResult]] = []
    averaged: list[BPAResult] = []
    for target in targets:
        runs = []
        for k in ks:
            spec = spec_for_k[k]
            res = variational_laplace(
                spec,
                faes_priors(spec, target),
                u_for_k[k],
                data_for_k[k],
                rows=rows,
                kernel=None if kernel_for_k is None else kernel_for_k[k],
                **vl_kwargs,
            )
            runs.append(res)
            if bar is not None:
                bar.update(1)
        per_model.append(runs)
        averaged.append(
            bayesian_parameter_average(runs, spec_for_k[ks[0]], free_energy=free_energy)
        )
    if bar is not None:
        bar.close()
    return names, averaged, per_model
