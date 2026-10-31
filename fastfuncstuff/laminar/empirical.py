"""Empirical priors on the vascular parameters, shared between neighbouring columns.

Each column is inverted on its own, so its ascending-vein slope ``s_d`` rests on
that column's data alone, and on real columns it spreads far below its prior:
the direction in which [[Uludag & Havlicek 2021]] show the laminar profile goes
wrong. The vasculature belongs to the tissue, not to a 3 mm column, so the
parameters that describe it are better estimated together.

Following [[Friston 2016 BMR PEB]] (empirical Bayes by model reduction), with
columns in the place of subjects: each column's prior on the pooled parameters
becomes N(mu_c, f * prior variance), where mu_c is the random-effects mean of
the posteriors of the columns within ``radius_mm`` of it and the shrinkage ``f``
is the value on a grid that maximises the summed reduced free energy over all
columns. The new posteriors and evidences then come from
:func:`~fastfuncstuff.laminar.batched.reduce_batched`, with no re-inversion.
The prior only ever narrows (f <= 1), which keeps every reduction well posed.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from fastfuncstuff.laminar.batched import BatchedVL, reduce_batched
from fastfuncstuff.laminar.inversion import Priors, vec
from fastfuncstuff.laminar.params import zero_params

#: Parameters of the vascular tree that the column fit estimates.
VASCULAR = ("s_d", "al_d")
#: Shrinkage grid, as a fraction of the prior variance.
SHRINK_GRID = tuple(float(v) for v in np.logspace(-3, 0, 13))


def batch_priors(
    priors: Priors, n: int, overrides: dict[str, torch.Tensor] | None = None
) -> Priors:
    """``priors`` with a leading dataset axis of ``n``; ``overrides`` replace pE/pC
    entries per dataset (keys ``"pE:<name>"`` / ``"pC:<name>"``, values ``(n,)``)."""

    def expand(d: dict[str, torch.Tensor], tag: str) -> dict[str, torch.Tensor]:
        out = {}
        for name, t in d.items():
            key = f"{tag}:{name}"
            if overrides and key in overrides:
                v = overrides[key].to(t)
                out[name] = v.reshape(n, *t.shape) if v.numel() == n * max(t.numel(), 1) else v
            else:
                out[name] = t.expand(n, *t.shape).clone()
        return out

    return dataclasses.replace(priors, pE=expand(priors.pE, "pE"), pC=expand(priors.pC, "pC"))


def _flat_index(spec, name: str) -> int:
    template = zero_params(spec)
    probe = {k: torch.zeros_like(v) for k, v in template.items()}
    probe[name] = torch.ones_like(probe[name])
    hits = torch.nonzero(vec(probe)).flatten()
    if hits.numel() != 1:
        raise ValueError(f"{name} is not a scalar parameter")
    return int(hits[0])


def neighbourhoods(
    centres: np.ndarray, hemi: np.ndarray, radius_mm: float | None
) -> list[np.ndarray]:
    """For each column, the columns whose centres are within ``radius_mm`` of its own
    (same hemisphere; None: every column of the hemisphere)."""
    from scipy.spatial import KDTree

    out: list[np.ndarray] = [np.zeros(0, np.int64)] * len(centres)
    for h in np.unique(hemi):
        idx = np.flatnonzero(hemi == h)
        if radius_mm is None:
            for i in idx:
                out[i] = idx
            continue
        tree = KDTree(centres[idx])
        for j, nb in enumerate(tree.query_ball_point(centres[idx], float(radius_mm))):
            out[idx[j]] = idx[np.asarray(nb, np.int64)]
    return out


@dataclasses.dataclass
class EmpiricalPrior:
    """The empirical prior per column and how it was chosen."""

    names: tuple[str, ...]
    mean: np.ndarray  # (C, P) prior mean per column and parameter
    shrink: float  # prior variance as a fraction of the original
    grid_free_energy: np.ndarray  # (G,) summed reduced F per grid value
    priors: Priors  # per-column reduced priors for the full model


def empirical_vascular_prior(
    full: BatchedVL,
    full_priors: Priors,
    centres: np.ndarray,
    hemi: np.ndarray,
    *,
    radius_mm: float | None = 10.0,
    names: tuple[str, ...] = VASCULAR,
    grid: tuple[float, ...] = SHRINK_GRID,
) -> EmpiricalPrior:
    """Choose each column's empirical prior on ``names`` from its neighbours' posteriors.

    ``full`` is the full model's inversion at one K for every column (rows in the
    order of ``centres``). The mean is the random-effects (precision-weighted) mean
    of the neighbours' posterior means, at the candidate prior variance; the
    variance is the grid value with the largest summed reduced free energy.
    """
    spec = full.spec
    free = full.free_idx.cpu().numpy()
    flat = [_flat_index(spec, n) for n in names]
    pos = [int(np.flatnonzero(free == f)[0]) for f in flat]
    m = full.Ep_vec[:, flat].cpu().numpy()  # (C, P)
    v = np.stack([full.Cp[:, p, p].cpu().numpy() for p in pos], 1)
    pC0 = vec(full_priors.pC)[flat].cpu().numpy()
    nbs = neighbourhoods(np.asarray(centres, np.float64), np.asarray(hemi), radius_mm)
    # Leave each column out of its own mean, or it pulls its prior toward itself.
    nbs = [nb[nb != i] for i, nb in enumerate(nbs)]
    alone = np.array([nb.size == 0 for nb in nbs])
    pE0 = vec(full_priors.pE)[flat].cpu().numpy()
    best = None
    grid_free_energy = np.full(len(grid), -np.inf)
    n = m.shape[0]
    for g, f in enumerate(grid):
        tau2 = f * pC0  # (P,)
        w = 1.0 / (v + tau2)  # random-effects weights
        mu = np.stack([(w[nb] * m[nb]).sum(0) / w[nb].sum(0) if nb.size else pE0 for nb in nbs])
        var = np.where(alone[:, None], pC0, tau2)  # a column with no neighbours keeps its prior
        over = {}
        for j, name in enumerate(names):
            over[f"pE:{name}"] = torch.as_tensor(mu[:, j])
            over[f"pC:{name}"] = torch.as_tensor(var[:, j])
        pri = batch_priors(full_priors, n, over)
        red = reduce_batched(full, full_priors, pri)
        grid_free_energy[g] = float((red.F - full.F).sum())
        if best is None or grid_free_energy[g] > grid_free_energy[best[0]]:
            best = (g, mu, pri)
    assert best is not None
    return EmpiricalPrior(tuple(names), best[1], float(grid[best[0]]), grid_free_energy, best[2])


__all__ = [
    "VASCULAR",
    "EmpiricalPrior",
    "batch_priors",
    "empirical_vascular_prior",
    "neighbourhoods",
]
