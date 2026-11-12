"""
Global ETAC — Equitable Thresholding And Clustering (3dttest++ -ETAC).

ETAC thresholds a statistic map with *several* sub-tests at once — every
combination of blur case, voxelwise p-threshold and figure of merit (FOM =
sum |z|^h over a cluster) — and makes them share the false-positive budget
equitably: one tail fraction ``tfrac`` is common to every sub-test, each
sub-test's FOM threshold is its null max-FOM distribution at that tail, and
``tfrac`` is tuned until the fraction of null iterations in which *any*
sub-test passes equals the FPR goal.  A voxel survives if it sits in a cluster
that passes any sub-test.

What this does differently from ``3dXClustSim`` (read from its source):

* AFNI re-clusters every test iteration at every step of the ``tfrac``
  search.  A test iteration is a false positive exactly when some sub-test's
  largest null cluster reaches its threshold, so the search here runs on the
  stored ``[iteration, sub-test]`` max-FOM matrix and costs milliseconds.
* The search is a bisection for the largest ``tfrac`` whose FPR does not
  exceed the goal, instead of AFNI's secant iterations with a tolerance.
* For 1-sided tests AFNI draws its null alternately from the positive and the
  negative tail of successive iterations; here both tails of every iteration
  are used (twice the null clusters for the same permutations).

Kept identical: thresholds as ranked maxima with AFNI's plotting positions
and Gumbel interpolation (``inverse_interp_extreme``), the 5-voxel minimum
cluster, 2-sided clusters that join both signs, the optional split of
iterations into threshold-setting and FPR-testing halves (``splitfrac``).
Local (voxelwise-threshold) ETAC is deliberately absent — see
[[Group-level inference]].
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.stats import norm

from fastfuncstuff.stats.cluster_inference import NullSource, max_fom_null

DEFAULT_PTHR = (0.010, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001)
DEFAULT_NITER = 10080  # 3dttest++ NCSIM_NORM
MIN_CLUST = 5
TOPFRAC = 0.2468  # 3dXClustSim keeps the top 24.68% of FOMs; tfrac never exceeds 2/3 of it


@dataclass
class EtacOptions:
    pthr: tuple[float, ...] = DEFAULT_PTHR
    hpows: tuple[int, ...] = (2,)
    nn: int = 2
    sided: int = 2  # 1 or 2
    fpr: tuple[float, ...] = (5.0,)  # percent
    min_clust: int = MIN_CLUST
    splitfrac: float = 0.0  # 0 = AFNI default, no split

    def __post_init__(self) -> None:
        if self.sided not in (1, 2):
            raise ValueError("ETAC sidedness is 1 or 2")
        if self.nn not in (1, 2, 3):
            raise ValueError("NN must be 1, 2 or 3")
        bad = [h for h in self.hpows if h not in (0, 1, 2)]
        if bad or not self.hpows:
            raise ValueError("hpow values are 0, 1, 2")
        if any(not (0 < p <= 0.5) for p in self.pthr):
            raise ValueError("pthr values must lie in (0, 0.5]")
        if any(not (0 < f < 50) for f in self.fpr):
            raise ValueError("FPR goals are percentages in (0, 50)")
        if self.splitfrac and not (0.2 <= self.splitfrac <= 0.8):
            raise ValueError("splitfrac must be 0 (off) or within [0.2, 0.8]")

    @property
    def zthr(self) -> np.ndarray:
        p = np.asarray(self.pthr, dtype=np.float64)
        return norm.isf(p if self.sided == 1 else p / 2)

    @property
    def sided_key(self) -> str:
        return "1-sided" if self.sided == 1 else "2-sided"


@dataclass
class EtacNull:
    """Max-FOM null: ``fom[i, case, pthr, h]`` (0 where no cluster formed)."""

    fom: np.ndarray
    case_labels: tuple[str, ...]
    opts: EtacOptions
    n_train: int  # first n_train rows set thresholds
    test_rows: slice = field(default_factory=lambda: slice(None))
    _sorted: dict = field(default_factory=dict, repr=False)

    def ranked(self, idx: tuple[int, ...]) -> np.ndarray:
        """Descending nonzero training maxima of one sub-test (cached)."""
        got = self._sorted.get(idx)
        if got is None:
            col = self.fom[(slice(0, self.n_train), *idx)]
            keep = int(TOPFRAC * self.n_train)
            got = self._sorted[idx] = np.sort(col[col > 0])[::-1][:keep]
        return got


def build_null(
    sources: dict[str, NullSource],
    n_iter: int,
    mask: np.ndarray,
    opts: EtacOptions,
    device: torch.device,
    verbose: bool = True,
) -> EtacNull:
    """Run every blur case's null through the shared clustering driver.

    For 1-sided ETAC each iteration contributes its positive and its negative
    tail as two null samples (rows ``2i`` and ``2i+1``); the FPR is still
    measured on the positive tail of the test iterations only.
    """
    tc = {opts.sided_key: opts.zthr}
    n_pthr, n_h = len(opts.pthr), len(opts.hpows)
    rows = 2 * n_iter if opts.sided == 1 else n_iter
    fom = np.zeros((rows, len(sources), n_pthr, n_h), dtype=np.float32)
    for ic, (label, src) in enumerate(sources.items()):
        source = src
        if opts.sided == 1:

            def source(start, nb, _s=src):
                z = _s(start // 2, nb // 2)
                return torch.stack([z, -z], 1).reshape(-1, z.shape[1])

        got = max_fom_null(
            source,
            rows,
            mask,
            tc,
            nns=(opts.nn,),
            sideds=(opts.sided_key,),
            hpows=opts.hpows,
            min_clust=opts.min_clust,
            device=device,
            batch=_even_batch(rows, mask, opts, device) if opts.sided == 1 else None,
            desc=f"ETAC null {label}",
            verbose=verbose,
        )
        for ih, h in enumerate(opts.hpows):
            fom[:, ic, :, ih] = got[(opts.sided_key, opts.nn, h)]
    n_split = int(round(n_iter * opts.splitfrac)) if opts.splitfrac else 0
    per = 2 if opts.sided == 1 else 1
    if n_split:
        n_train, test = per * n_split, slice(per * n_split, rows, per)
    else:
        n_train, test = rows, slice(0, rows, per)
    return EtacNull(fom, tuple(sources), opts, n_train, test)


def _even_batch(rows: int, mask: np.ndarray, opts: EtacOptions, device: torch.device) -> int:
    """1-sided batches must hold whole (z, -z) pairs."""
    from fastfuncstuff.memory import cluster_fom_batch

    b = cluster_fom_batch(int(mask.sum()), len(opts.hpows), device)
    return max(2, min(rows, b) // 2 * 2)


# ---------------------------------------------------------------------------
# Thresholds and the FPR search
# ---------------------------------------------------------------------------


def _gumbel(a: np.ndarray) -> np.ndarray:
    return np.log(-np.log(1.0 - a))


def thresholds_at(null: EtacNull, tfrac: float) -> np.ndarray:
    """``[case, pthr, h]`` FOM thresholds at a common tail fraction.

    3dXClustSim's rule: from each sub-test's descending list of per-iteration
    maxima (iterations with no cluster excluded), take the ``ithresh``-th
    largest with plotting positions ``(j + 0.666) / (n + 0.333)`` and
    interpolate linearly in Gumbel space ``log(-log(1-a))``.
    """
    n = null.n_train
    nit33 = 1.0 / (n + 0.333)
    ithresh = int(np.rint(tfrac * (n - 0.666) + 0.333))
    out = np.zeros(null.fom.shape[1:], dtype=np.float64)
    for idx in np.ndindex(*null.fom.shape[1:]):
        vals = null.ranked(idx)
        npt = vals.size
        if npt < 2:
            # No null cluster survived often enough to rank: any observed
            # cluster at all would be beyond the null, so the bar is "exists".
            out[idx] = vals[0] if npt else 0.0
            continue
        j = min(ithresh, npt - 2)
        a0 = (j + 0.666) * nit33
        a1 = a0 + nit33
        x0, x1 = float(vals[j]), float(vals[j + 1])
        if (tfrac - a0) * (tfrac - a1) < 0:
            g0, g1, gt = _gumbel(np.array([a0, a1, tfrac]))
            out[idx] = x0 + (x1 - x0) / (g1 - g0) * (gt - g0)
        else:
            out[idx] = x0 if abs(tfrac - a0) <= abs(tfrac - a1) else x1
    return out


def fpr_at(null: EtacNull, thr: np.ndarray) -> float:
    """Percent of test iterations in which any sub-test's largest cluster passes."""
    test = null.fom[null.test_rows]  # [n_test, case, pthr, h]
    passed = (test >= thr[None]) & (test > 0)
    return 100.0 * float(passed.reshape(test.shape[0], -1).any(1).mean())


@dataclass
class EtacThresholds:
    fpr_goal: float
    fpr: float  # achieved on the test iterations
    tfrac: float
    thr: np.ndarray  # [case, pthr, h]


def search(null: EtacNull, fpr_goal: float, n_steps: int = 60) -> EtacThresholds:
    """Largest common tail fraction whose FPR does not exceed the goal.

    FPR rises monotonically with ``tfrac`` (thresholds only fall), so a
    bisection in log(tfrac) between AFNI's bounds finds it; the bounds are
    3dXClustSim's ``min_tfrac`` and ``0.666 * TOPFRAC``.
    """
    n = null.n_train
    lo = min(6.0 / n, 1e-4)
    hi = 0.666 * TOPFRAC
    best = EtacThresholds(fpr_goal, fpr_at(null, thresholds_at(null, lo)), lo,
                          thresholds_at(null, lo))  # fmt: skip
    if best.fpr > fpr_goal:
        return best  # even the strictest bar the null can rank is too lenient
    a, b = math.log(lo), math.log(hi)
    for _ in range(n_steps):
        mid = 0.5 * (a + b)
        tf = math.exp(mid)
        thr = thresholds_at(null, tf)
        f = fpr_at(null, thr)
        if f <= fpr_goal:
            a = mid
            best = EtacThresholds(fpr_goal, f, tf, thr)
        else:
            b = mid
        if b - a < 1e-4:
            break
    return best


# ---------------------------------------------------------------------------
# Applying thresholds to the observed maps
# ---------------------------------------------------------------------------


def apply(
    z_obs: dict[str, torch.Tensor],
    mask: np.ndarray,
    opts: EtacOptions,
    res: EtacThresholds,
    sign: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """``(union [V] bool, per-sub-test [case*pthr, V] bool)`` surviving voxels.

    ``sign=-1`` applies the same thresholds to ``-z`` (the 1-sided negative
    mask).  A voxel passes a (case, pthr) sub-test if its cluster's FOM for
    any requested h reaches that sub-test's threshold.
    """
    from fastfuncstuff.stats.cluster_gpu import build_neighbor_table, cluster_labels

    n_vox = int(mask.sum())
    cases = list(z_obs)
    all_mask = np.zeros((len(cases) * len(opts.pthr), n_vox), dtype=bool)
    hcol = {0: 0, 1: 1, 2: 2}
    for ic, case in enumerate(cases):
        z = z_obs[case] * sign
        nbr = build_neighbor_table(mask, opts.nn, z.device)
        for ip, zt in enumerate(opts.zthr):
            active, _, foms = cluster_labels(
                z.float(), nbr, opts.sided_key, float(zt), opts.min_clust
            )
            if active.numel() == 0:
                continue
            ok = torch.zeros(active.numel(), dtype=torch.bool, device=active.device)
            for ih, h in enumerate(opts.hpows):
                ok |= foms[hcol[h]] >= float(res.thr[ic, ip, ih])
            all_mask[ic * len(opts.pthr) + ip, active[ok].cpu().numpy()] = True
    return np.asarray(all_mask.any(0), dtype=bool), all_mask


# ---------------------------------------------------------------------------
# Group t-test driver
# ---------------------------------------------------------------------------


@dataclass
class EtacResult:
    """Everything one ETAC run produces, for every FPR goal."""

    case_labels: tuple[str, ...]
    opts: EtacOptions
    n_iter: int
    n_distinct: float
    z_obs: dict[str, torch.Tensor]
    thresholds: list[EtacThresholds]
    # masks[(fpr_goal, side)] = (union [V], per-sub-test [case*pthr, V]); side is
    # "2sid", "1pos" or "1neg" as in 3dttest++'s file names.
    masks: dict[tuple[float, str], tuple[np.ndarray, np.ndarray]]


def case_label(fwhm: float) -> str:
    return f"B{fwhm:.1f}"


def run_group_etac(
    design,
    ya: torch.Tensor,
    yb: torch.Tensor | None,
    mask: np.ndarray,
    voxdims: tuple[float, float, float],
    *,
    blurs: tuple[float, ...] = (0.0,),
    opts: EtacOptions | None = None,
    n_iter: int = DEFAULT_NITER,
    seed: int = 0,
    device: torch.device,
    verbose: bool = True,
) -> EtacResult:
    """Global ETAC for the main test of a :class:`group_ttest.TtestDesign`.

    Each blur case blurs the subject maps inside the mask (3dttest++
    ``-ETAC_blur``, applied on the device) and then shares one set of sign
    flips — and, for two independent sets, label permutations — so the cases
    see the same relabelings, as 3dttest++ does with a common ``-seed``.
    """
    from fastfuncstuff.stats.blur_to import MaskedGaussian
    from fastfuncstuff.stats.group_ttest import OlsNull, main_test, ols_contrasts, t_to_z_np

    opts = opts or EtacOptions()
    if design.unpooled:
        raise ValueError("ETAC with -unpooled is not implemented (use the pooled test)")
    permute = design.two_sets and not design.paired
    sources: dict[str, NullSource] = {}
    z_obs: dict[str, torch.Tensor] = {}
    n_distinct = math.inf
    for fwhm in blurs:
        label = case_label(fwhm)
        if label in sources:
            raise ValueError(f"blur {fwhm} given twice")
        if fwhm > 0:
            g = MaskedGaussian(mask, voxdims, fwhm, device=device)
            ya_b = g.rows(ya)
            yb_b = g.rows(yb) if yb is not None else None
        else:
            ya_b, yb_b = ya, yb
        y, x, c = main_test(design, ya_b, yb_b)
        null = OlsNull(y.float(), x, c, n_iter=n_iter, seed=seed, permute=permute)
        n_distinct = null.n_distinct
        fit = ols_contrasts(y.double(), x, c[None])
        z = t_to_z_np(fit.t[0].cpu().numpy(), fit.dof)
        z_obs[label] = torch.as_tensor(z, dtype=torch.float32, device=device)
        sources[label] = null

    null_fom = build_null(sources, n_iter, mask, opts, device, verbose=verbose)
    thresholds, masks = [], {}
    sides = [("2sid", 1)] if opts.sided == 2 else [("1pos", 1), ("1neg", -1)]
    for goal in opts.fpr:
        res = search(null_fom, goal)
        thresholds.append(res)
        for side, sign in sides:
            masks[(goal, side)] = apply(z_obs, mask, opts, res, sign)
    return EtacResult(
        case_labels=tuple(sources),
        opts=opts,
        n_iter=n_iter,
        n_distinct=n_distinct,
        z_obs=z_obs,
        thresholds=thresholds,
        masks=masks,
    )
