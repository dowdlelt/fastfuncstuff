"""Evaluate every alignment cost functional for one image pair.

The engine behind ``ffs_util_cost``, the analogue of ``3dAllineate -allcostX``:
given a base and a source (optionally moved by a transform), report all of the
cost functionals at that position rather than optimising any one of them.

Everything here is a thin arrangement of the existing cost primitives —
``cost.py`` (Pearson family), ``cost_hist.py`` (2-D histogram family) and
``cost_blok.py`` (local Pearson over bloks) — so the numbers printed are the
same numbers the optimiser sees.

**Sign convention**: values are reported in AFNI's convention, where *lower is
better* for every functional, so they can be compared against
``3dAllineate -allcostX`` directly. The ffs primitives use the opposite
(higher-is-better) convention internally; the conversion happens here, in one
place, per functional.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from . import cost_hist
from .cost import clipped_pearson_correlation, spearman_correlation
from .cost_blok import assign_bloks_points, local_pearson_value

# Ordered as 3dAllineate prints them, with AFNI's own descriptions of what the
# reported number is (see meth_costfunctional[] in 3dAllineate.c).
COST_INFO: dict[str, tuple[str, str]] = {
    "ls": ("leastsq", "1 - abs(Pearson correlation coefficient)"),
    "sp": ("spearman", "1 - abs(Spearman [rank] correlation coefficient)"),
    "mi": ("mutualinfo", "-Mutual Information = H(b,s) - H(b) - H(s)"),
    "crM": ("corratio_mul", "1 - abs[ CR(b,s) * CR(s,b) ]"),
    "nmi": ("norm_mutualinfo", "H(b,s) / [H(b) + H(s)]"),
    "je": ("jointentropy", "H(b,s) = joint entropy of the image pair"),
    "hel": ("hellinger", "-Hellinger distance(b,s)"),
    "crA": ("corratio_add", "1 - abs[ CR(b,s) + CR(s,b) ]"),
    "crU": ("corratio_uns", "1 - abs[ CR(s,b) ] (unsymmetrised)"),
    "lss": ("signedPcor", "Signed Pearson correlation coefficient"),
    "lpc": ("localPcorSigned", "Local Pearson correlation, signed"),
    "lpa": ("localPcorAbs", "1 - abs(Local Pearson correlation)"),
    "lpc+": ("localPcor+Others", "lpc + weighted hel/mi/nmi/crA (+ overlap)"),
    "lpa+": ("localPcorAbs+Others", "lpa + weighted hel/mi/nmi/crA (+ overlap)"),
}

ALL_COSTS = list(COST_INFO)

# Which functionals are computed from the same underlying quantity. Excluding a
# cost from a judging panel has to exclude its whole family, because "different
# name" is not "different evidence": lpc, lpa, lpc+ and lpa+ are all functions of
# the single ``lp_val`` computed once below, so dropping lpa while keeping lpa+
# lets the optimised cost keep voting under another name (measured rank
# correlation between them: 1.00).
COST_FAMILY: dict[str, str] = {
    "ls": "correlation",
    "lss": "correlation",
    "sp": "correlation",
    "mi": "histogram",
    "nmi": "histogram",
    "je": "histogram",
    "hel": "histogram",
    "crM": "histogram",
    "crA": "histogram",
    "crU": "histogram",
    "lpc": "localpearson",
    "lpa": "localpearson",
    "lpc+": "localpearson",
    "lpa+": "localpearson",
}

# Functionals that are *signed*: lower-is-better means they reward
# anti-correlation, which is what you want when the two images have inverted
# contrast (EPI vs T1) and exactly wrong when they do not. On same-modality data
# these rank the worst candidate first — measured at rho = -0.99 against every
# unsigned functional on a T1->MNI search, where they silently cancelled three of
# the eleven honest votes.
SIGNED_COSTS = ("lss", "lpc", "lpc+")

CONTRAST_REGIMES = ("same", "cross")

# AFNI's DEFAULT_MICHO_* weights for the "+" combination costs, as
# (hel, mi, nmi, crA, ov). lpa+ drops the MI term (3dAllineate.c, 27 May 2021).
MICHO_LPC = (0.4, 0.2, 0.2, 0.4, 0.4)
MICHO_LPA = (0.4, 0.0, 0.2, 0.4, 0.4)


@dataclass
class CostInputs:
    """The match points a cost is evaluated on: flat in-mask base/source/weight.

    Evaluation is restricted to the weight domain rather than the whole grid on
    purpose. A full-grid local-Pearson value is dominated by background bloks —
    they fail the variance test, and ``local_pearson_value`` scales by the
    fraction of populated bloks that contributed, so a *good* whole-brain
    alignment scores ~0.045 instead of ~0.57. AFNI has the same property and
    solves it the same way, by scoring only npt_match points inside the mask.
    """

    base: Tensor  # (M,) in-mask base values
    source: Tensor  # (M,) in-mask source values, on the base grid
    weight: Tensor | None = None  # (M,)
    coords_mm: Tensor | None = None  # (M, 3) physical coords, for blok assignment
    # Fraction of the smaller mask that both images cover; only meaningful when
    # the caller knows the source FOV (drives the lpc+/lpa+ overlap term).
    overlap: float | None = None


def build_cost_inputs(
    base: Tensor,
    source: Tensor,
    weight: Tensor | None,
    voxdims: tuple[float, float, float] = (1.0, 1.0, 1.0),
    n_match: float = 1.0,
    bloktype: str = "tohd",
    overlap: float | None = None,
    whole_volume: bool = False,
) -> CostInputs:
    """Gather the in-mask match points for a base/source pair on a common grid.

    Reuses allineate's own domain logic, so the points scored here are the same
    points the optimiser scores (``n_match`` follows the same unit-free rule:
    <=1.0 is a fraction of the domain, >1.0 an absolute count).

    ``whole_volume`` scores every voxel instead, which is what
    ``3dAllineate -allcostX`` does. It is a comparison mode, not a better one:
    outside the brain both images are zero, so the background agreement inflates
    every functional (whole-volume Pearson r = 0.93 where the in-brain r = 0.62
    for the same pair) and compresses the gap between candidates.
    """
    if whole_volume:
        dx, dy, dz = (float(v) for v in voxdims)
        nz, ny, nx = base.shape
        kk, jj, ii = torch.meshgrid(
            torch.arange(nz, dtype=torch.float32, device=base.device),
            torch.arange(ny, dtype=torch.float32, device=base.device),
            torch.arange(nx, dtype=torch.float32, device=base.device),
            indexing="ij",
        )
        coords = torch.stack(
            [(ii * dx).reshape(-1), (jj * dy).reshape(-1), (kk * dz).reshape(-1)], dim=1
        )
        return CostInputs(
            base=base.reshape(-1),
            source=source.reshape(-1),
            weight=None if weight is None else weight.reshape(-1),
            coords_mm=coords,
            overlap=overlap,
        )

    from .allineate import _build_sample_set

    sample = _build_sample_set(base, weight, voxdims, n_match, bloktype, base.device)
    if sample is None:  # tiny volume: score every voxel
        return CostInputs(base.reshape(-1), source.reshape(-1), None, None, overlap)

    idx = sample.idx_flat
    return CostInputs(
        base=base.reshape(-1)[idx],
        source=source.reshape(-1)[idx],
        weight=sample.weight_s,
        coords_mm=sample.coords_mm,
        overlap=overlap,
    )


def _overlap_term(overlap: float | None) -> float:
    """AFNI's (max(0, 9.95 - 10*overlap))^2 penalty; 0 when overlap is unknown."""
    if overlap is None:
        return 0.0
    return float(max(0.0, 9.95 - 10.0 * overlap) ** 2)


def evaluate_all_costs(
    inp: CostInputs,
    costs: list[str] | None = None,
    bloktype: str = "tohd",
    blokrad: float | None = None,
    ppow: float = 1.0,
    nbin: int | None = None,
) -> dict[str, float]:
    """Compute the requested cost functionals (AFNI convention, lower == better).

    Args:
        inp: base/source/weight on a common grid.
        costs: subset of :data:`ALL_COSTS` (default: all of them).
        bloktype, blokrad, ppow: local-Pearson blok geometry, as in allineate.
        nbin: 2-D histogram bin count (default: AFNI's data-driven choice).

    Returns:
        ``{cost_name: value}`` in the requested order.
    """
    want = list(costs) if costs else list(ALL_COSTS)
    unknown = [c for c in want if c not in COST_INFO]
    if unknown:
        raise ValueError(f"Unknown cost function(s): {', '.join(unknown)}")

    bflat, sflat, wflat = inp.base, inp.source, inp.weight
    base, source, weight = bflat, sflat, wflat

    out: dict[str, float] = {}

    # --- histogram family: one joint histogram serves mi/nmi/je/hel/cr* -------
    hist_names = {"mi", "nmi", "je", "hel", "crM", "crA", "crU"}
    needs_hist = bool(hist_names & set(want)) or bool({"lpc+", "lpa+"} & set(want))
    hist_vals: dict[str, float] = {}
    if needs_hist:
        hm = cost_hist.hist2d_measures(base, source, weight=weight, nbin=nbin)
        # 1 - association, matching the 1-|CR| AFNI reports for each flavour.
        cr_u = 1.0 - abs(1.0 - hm.cr_yx)
        cr_a = 1.0 - abs(1.0 - 0.5 * (hm.cr_yx + hm.cr_xy))
        cr_m = 1.0 - abs(1.0 - hm.cr_yx * hm.cr_xy)
        hist_vals = {
            "mi": -hm.mi,
            "nmi": hm.nmi,
            "je": hm.je,
            "hel": -(1.0 - hm.hel),  # AFNI reports -THD_hellinger = affinity - 1
            "crU": cr_u,
            "crA": cr_a,
            "crM": cr_m,
        }

    # --- local Pearson: one blok assignment serves lpc/lpa and both combos ----
    lp_val = None
    if {"lpc", "lpa", "lpc+", "lpa+"} & set(want):
        if inp.coords_mm is None:
            raise ValueError("The lpc/lpa family needs coords_mm; use build_cost_inputs()")
        blokset = assign_bloks_points(inp.coords_mm, bloktype, blokrad, device=base.device)
        lp_val = float(local_pearson_value(base, source, weight, blokset, ppow))

    for name in want:
        if name == "ls":
            r = float(clipped_pearson_correlation(bflat, sflat, wflat))
            out[name] = 1.0 - abs(r)
        elif name == "lss":
            out[name] = float(clipped_pearson_correlation(bflat, sflat, wflat))
        elif name == "sp":
            out[name] = 1.0 - abs(float(spearman_correlation(bflat, sflat)))
        elif name in hist_vals:
            out[name] = hist_vals[name]
        elif name == "lpc":
            out[name] = lp_val  # type: ignore[assignment]
        elif name == "lpa":
            out[name] = 1.0 - abs(lp_val)  # type: ignore[arg-type]
        elif name in ("lpc+", "lpa+"):
            w_hel, w_mi, w_nmi, w_cra, w_ov = MICHO_LPC if name == "lpc+" else MICHO_LPA
            val = lp_val if name == "lpc+" else 1.0 - abs(lp_val)  # type: ignore[arg-type]
            # Each extra term is exactly the standalone AFNI cost, weighted, so
            # the combination is a weighted sum of functionals we already report.
            val += (
                w_hel * hist_vals["hel"]
                + w_mi * hist_vals["mi"]
                + w_nmi * hist_vals["nmi"]
                + w_cra * hist_vals["crA"]
            )
            val += w_ov * _overlap_term(inp.overlap)
            out[name] = val

    return out


def format_cost_table(vals: dict[str, float], describe: bool = False) -> str:
    """Render an ``-allcostX``-style table (lower == better for every row)."""
    lines = []
    for name, v in vals.items():
        long, desc = COST_INFO[name]
        if describe:
            lines.append(f"   {name:<4s} {long:<20s} {v:>14.6f}   {desc}")
        else:
            lines.append(f"   {name:<4s} {long:<20s} {v:>14.6f}")
    return "\n".join(lines)


def cost_agreement(vals_a: dict[str, float], vals_b: dict[str, float]) -> dict[str, float]:
    """Per-functional improvement of ``a`` over ``b`` (positive == a is better).

    Used to compare two candidate alignments across every functional at once,
    rather than trusting the single one that was optimised.
    """
    common = [k for k in vals_a if k in vals_b]
    return {k: vals_b[k] - vals_a[k] for k in common}


def judge_panel(
    optimized: str | None = None,
    contrast: str = "same",
    exclude: tuple[str, ...] = (),
) -> list[str]:
    """The functionals allowed to judge a fit, given what produced it.

    Two filters, and both of them changed the answer on real data:

    * **Contrast regime.** On ``same``-modality pairs the signed functionals
      (:data:`SIGNED_COSTS`) are dropped, because "lower is better" means
      "more anti-correlated" for them and there is no legitimate anti-correlation
      between two T1s. Left in, they rank the *worst* warp first.
    * **Family-aware exclusion.** The optimised cost is excluded along with
      everything derived from the same quantity (:data:`COST_FAMILY`), so a cost
      cannot vote on its own fit through a sibling.

    Args:
        optimized: the cost the backend minimised, whose family is removed.
        contrast: ``"same"`` or ``"cross"`` modality.
        exclude: further functionals to drop by name.

    Returns:
        Cost names, in :data:`ALL_COSTS` order.
    """
    if contrast not in CONTRAST_REGIMES:
        raise ValueError(f"contrast must be one of {CONTRAST_REGIMES}, got {contrast!r}")

    drop = set(exclude)
    if optimized:
        if optimized not in COST_FAMILY:
            raise ValueError(f"unknown cost {optimized!r}")
        family = COST_FAMILY[optimized]
        drop |= {c for c, f in COST_FAMILY.items() if f == family}
    if contrast == "same":
        drop |= set(SIGNED_COSTS)

    panel = [c for c in ALL_COSTS if c not in drop]
    if not panel:
        raise ValueError(
            f"empty judging panel: optimizing {optimized!r} under contrast "
            f"{contrast!r} excludes every functional"
        )
    return panel


def consensus_rank(candidates: dict[str, dict[str, float]]) -> list[tuple[str, float]]:
    """Rank candidate alignments by mean per-functional rank (lower == better).

    Each cost functional votes by ranking the candidates it scores; the votes
    are averaged. A candidate that wins on its own cost but loses on the other
    thirteen is not a candidate that won — this is what turns "all the costs"
    into a single answer without privileging whichever one was optimised.
    """
    names = list(candidates)
    if not names:
        return []
    metrics = set(candidates[names[0]])
    for n in names[1:]:
        metrics &= set(candidates[n])

    totals = dict.fromkeys(names, 0.0)
    for m in sorted(metrics):
        order = sorted(names, key=lambda n: candidates[n][m])
        for rank, n in enumerate(order):
            totals[n] += rank
    n_m = max(1, len(metrics))
    scored = [(n, totals[n] / n_m) for n in names]
    scored.sort(key=lambda t: t[1])
    return scored
