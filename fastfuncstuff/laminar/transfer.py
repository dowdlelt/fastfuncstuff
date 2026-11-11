"""Fit one condition, carry its physiology to the other: two drive profiles, no modulation.

The published design (:mod:`~fastfuncstuff.laminar.experiment`) says the second condition
is the first one's drive with the excitatory self-connection changed at some depths. That
is a statement about mechanism, and it cannot express a condition whose input arrives at
different depths: given one, it answers in gains of opposite sign.

This is the other reading. The vasculature belongs to the tissue and not to the task, so:

1. fit the FIRST condition alone, with everything the published model estimates except
   the modulation -- the drive at each neuronal depth, the neuronal time constant, the
   depth-mapping width and the ascending vein's slope and Grubb exponent;
2. fit the SECOND condition alone with all of that held at the first fit's values, and
   only its drive at each depth free.

What comes back is a drive per depth for each condition and their difference. Nothing is
compared by free energy and no layer hypothesis is tested.

Three things to know before reading the result:

* The second condition's intervals are CONDITIONAL on the first fit's physiology being
  right. Its uncertainty is not carried over, so they are too narrow by an amount this
  does not measure.
* The neuronal time constant is carried too, so the second condition may differ in how
  much input each depth receives and in nothing else. A response with a different time
  course at the same amplitude will fit worse; the two ``r2`` say whether it did.
* A single condition has nothing to cancel vascular error against (Uludag & Havlicek
  2021 recommend differences for that reason). Both profiles rest on the first fit's
  ascending-vein slope, and a slope fitted too low gives profiles that are wrong in kind.
  The DIFFERENCE of the two profiles shares that error and is the safer thing to read.

The first condition should be the one with more signal: it is the one that has to pay for
the physiology.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.laminar.experiment import bayesian_parameter_average, faes_priors
from fastfuncstuff.laminar.inversion import Priors, VLResult, variational_laplace
from fastfuncstuff.laminar.io import LaminarData
from fastfuncstuff.laminar.params import ModelSpec, zero_params


def stage_rows(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The fitted-row mask split into the first condition's rows and the second's."""
    from fastfuncstuff.laminar.pipeline import condition_blocks

    blocks = condition_blocks(rows.numpy())
    if len(blocks) != 2:
        raise ValueError(f"expected two fitted stretches, found {len(blocks)}")
    out = []
    for a, b in blocks:
        m = torch.zeros_like(rows, dtype=torch.bool)
        m[a:b] = True
        out.append(m)
    return out[0], out[1]


def source_priors(spec: ModelSpec, *, free: Sequence[str] = ()) -> Priors:
    """The published model with no modulation: what the first condition is fitted with."""
    return faes_priors(spec, (0,) * spec.N, free=free)


def target_priors(spec: ModelSpec, source: dict[str, torch.Tensor]) -> Priors:
    """Everything at the first fit's posterior mean and fixed, except the drive.

    The drive starts from the same neutral prior the first condition had (mean 1,
    variance 1), not from the first condition's estimate: the two profiles are then shrunk
    alike, and their difference is not pulled toward zero by construction.
    """
    neutral = source_priors(spec)
    pE = {k: v.clone() for k, v in source.items()}
    pE["C"] = neutral.pE["C"].clone()
    pC = zero_params(spec)
    pC["C"] = neutral.pC["C"].clone()
    return Priors(pE=pE, pC=pC)


def _r2(y: torch.Tensor, pred: torch.Tensor, rows: torch.Tensor) -> float:
    obs = y[rows]
    resid = float(((obs - pred[rows]) ** 2).sum())
    return 1.0 - resid / max(float(((obs - obs.mean()) ** 2).sum()), 1e-30)


@dataclass
class TransferFit:
    """Two conditions' drive per neuronal depth, the second on the first's physiology."""

    roi: str
    name: str
    hemi: str
    k_values: list[int]
    drive: np.ndarray  # (2, N) posterior mean, averaged across K; superficial first
    drive_sd: np.ndarray  # (2, N)
    r2: np.ndarray  # (2, n_K) variance explained on each condition's own rows
    physiology: list[dict]  # per K: what the first fit found and the second was given
    stalled: np.ndarray  # (2, n_K)
    warnings: list[str] = field(default_factory=list)

    @property
    def difference(self) -> np.ndarray:
        return self.drive[1] - self.drive[0]

    @property
    def difference_sd(self) -> np.ndarray:
        # The two fits read different rows, so their errors add. Conditional on the
        # carried physiology, like the second condition's own interval.
        return np.sqrt(self.drive_sd[0] ** 2 + self.drive_sd[1] ** 2)


def _physiology(spec: ModelSpec, Ep: dict[str, torch.Tensor]) -> dict[str, float]:
    p0 = spec.p0
    return {
        "K": spec.K,
        "sigma_hz": p0.sigma * float(torch.exp(Ep["sigma"])),
        "s_d": p0.s_d * float(torch.exp(Ep["s_d"])),
        "al_d": p0.al_d * float(torch.exp(Ep["al_d"])),
        "nsig_logdev": float(Ep["nsig"]),
        "mu_hz": p0.mu * float(torch.exp(Ep["mu"])),
        "lam_hz": p0.lam * float(torch.exp(Ep["lam"])),
        "tau_v_s": p0.tau_v_in * float(torch.exp(Ep["tau_v_in"])),
        "tau_d_s": p0.tau_d_in * float(torch.exp(Ep["tau_d_in"])),
    }


def fit_transfer(
    data_for_k: dict[int, LaminarData],
    kernel_for_k: dict[int, torch.Tensor | None] | None = None,
    *,
    free: Sequence[str] = (),
    max_iter: int = 128,
    rescue: int = 0,
    roi: str = "",
    name: str = "",
    hemi: str = "",
) -> TransferFit:
    """Fit the first condition, then the second on its physiology, at every K.

    ``data_for_k`` is the two-condition layout of
    :func:`~fastfuncstuff.laminar.io.two_condition_data`; each stage masks the other
    condition's rows. The modulatory input is never used: its ``B`` has no variance in
    either stage. ``free`` (``mu``, ``lam``, ``tau_v``, ``tau_d``) is estimated in the
    first stage and carried like the rest.

    The noise prior is centred on each stage's own rows, never on the padded series: with
    one condition masked, the published convention would let the masked condition's
    amplitude set the other's noise level.
    """
    ks = sorted(data_for_k)
    kernels = kernel_for_k or {}
    stages: tuple[list[VLResult], list[VLResult]] = ([], [])
    r2 = np.zeros((2, len(ks)))
    physiology = []
    for j, k in enumerate(ks):
        d = data_for_k[k]
        rows_a, rows_b = stage_rows(d.rows)
        kw = dict(
            kernel=kernels.get(k), max_iter=max_iter, rescue=rescue, hyper_from_rows=True,
            **d.integrate_kwargs,
        )  # fmt: skip
        first = variational_laplace(
            d.spec, source_priors(d.spec, free=free), d.u, d.y, rows=rows_a, **kw
        )
        second = variational_laplace(
            d.spec, target_priors(d.spec, first.Ep), d.u, d.y, rows=rows_b, **kw
        )
        stages[0].append(first)
        stages[1].append(second)
        r2[0, j] = _r2(d.y, first.y_pred, rows_a)
        r2[1, j] = _r2(d.y, second.y_pred, rows_b)
        physiology.append(_physiology(d.spec, first.Ep))

    spec0 = data_for_k[ks[0]].spec
    drive = np.zeros((2, spec0.N))
    drive_sd = np.zeros((2, spec0.N))
    for s in (0, 1):
        bpa = bayesian_parameter_average(stages[s], spec0)
        drive[s] = bpa.Ep["C"][:, -1].numpy()
        drive_sd[s] = bpa.Vp["C"][:, -1].clamp(min=0).sqrt().numpy()
    stalled = np.array([[bool(r.stalled) for r in runs] for runs in stages])
    warnings = []
    for s, label in enumerate(("first", "second")):
        if stalled[s].any():
            where = ", ".join(f"K={ks[j]}" for j in np.flatnonzero(stalled[s]))
            warnings.append(f"the {label} condition's fit stalled at {where}")
    if warnings and not rescue:
        warnings.append("rerun with -rescue 2 to restart stalled fits with the damping reset")
    return TransferFit(
        roi=roi, name=name, hemi=hemi, k_values=ks, drive=drive, drive_sd=drive_sd, r2=r2,
        physiology=physiology, stalled=stalled, warnings=warnings,
    )  # fmt: skip


def write_transfer(prefix: str, fits: list[TransferFit], provenance: dict | None = None) -> list:
    """``{prefix}_laminar_transfer.tsv`` (one line per ROI x neuronal depth) and ``.json``."""
    from fastfuncstuff.laminar.pipeline import depth_names

    paths = [f"{prefix}_laminar_transfer.tsv", f"{prefix}_laminar_transfer.json"]
    Path(paths[0]).parent.mkdir(parents=True, exist_ok=True)
    with open(paths[0], "w") as fh:
        fh.write(
            "roi\tname\themi\tneuronal_depth\tdrive_1\tdrive_1_sd\tdrive_2\tdrive_2_sd\t"
            "difference\tdifference_sd\tr2_1\tr2_2\n"
        )
        for f in fits:
            for i, d in enumerate(depth_names(f.drive.shape[1])):
                fh.write(
                    f"{f.roi}\t{f.name}\t{f.hemi}\t{d}\t{f.drive[0, i]:.6g}\t"
                    f"{f.drive_sd[0, i]:.6g}\t{f.drive[1, i]:.6g}\t{f.drive_sd[1, i]:.6g}\t"
                    f"{f.difference[i]:.6g}\t{f.difference_sd[i]:.6g}\t"
                    f"{f.r2[0].mean():.6g}\t{f.r2[1].mean():.6g}\n"
                )
    with open(paths[1], "w") as fh:
        json.dump(
            {
                "provenance": provenance or {},
                "rois": [
                    {
                        "roi": f.roi,
                        "name": f.name,
                        "hemi": f.hemi,
                        "K": f.k_values,
                        "r2": f.r2.tolist(),
                        "physiology": f.physiology,
                        "stalled": f.stalled.tolist(),
                        "warnings": f.warnings,
                    }  # fmt: skip
                    for f in fits
                ],
            },
            fh,
            indent=1,
        )
    return paths
