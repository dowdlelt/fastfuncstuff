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

from fastfuncstuff.stats.cluster_inference import NullSource, Where, max_fom_null

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
    mask: Where,
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
            batch=_even_batch(rows, mask, opts, device, _src_bytes(src))
            if opts.sided == 1
            else None,
            source_bytes=_src_bytes(src),
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


def _src_bytes(src) -> int:
    per_map = getattr(src, "bytes_per_map", None)
    return int(per_map()) if callable(per_map) else 0


def _even_batch(
    rows: int, mask: Where, opts: EtacOptions, device: torch.device, source_bytes: int = 0
) -> int:
    """1-sided batches must hold whole (z, -z) pairs."""
    from fastfuncstuff.memory import cluster_fom_batch
    from fastfuncstuff.stats.cluster_inference import as_geometry

    n = as_geometry(mask).n
    b = cluster_fom_batch(n, len(opts.hpows), device, source_bytes=source_bytes // 2)
    return max(2, min(rows, b) // 2 * 2)


# ---------------------------------------------------------------------------
# Thresholds and the FPR search
# ---------------------------------------------------------------------------


def _gumbel(a: np.ndarray) -> np.ndarray:
    return np.log(-np.log(1.0 - a))


def _thr_curve(vals: np.ndarray, n: int, tf: np.ndarray | float) -> np.ndarray:
    """One sub-test's FOM threshold at each ``tf`` (3dXClustSim's rule, below).

    ``vals`` is the sub-test's descending nonzero training maxima, ``n`` the
    number of training iterations.
    """
    tf = np.asarray(tf, dtype=np.float64)
    npt = vals.size
    if npt < 2:
        # No null cluster survived often enough to rank: any observed
        # cluster at all would be beyond the null, so the bar is "exists".
        return np.full(tf.shape, float(vals[0]) if npt else 0.0)
    nit33 = 1.0 / (n + 0.333)
    ithresh = np.rint(tf * (n - 0.666) + 0.333).astype(np.int64)
    j = np.minimum(ithresh, npt - 2)
    a0 = (j + 0.666) * nit33
    a1 = a0 + nit33
    x0, x1 = vals[j].astype(np.float64), vals[j + 1].astype(np.float64)
    inside = (tf - a0) * (tf - a1) < 0
    with np.errstate(invalid="ignore", divide="ignore"):
        g0, g1, gt = _gumbel(a0), _gumbel(a1), _gumbel(tf)
        interp = x0 + (x1 - x0) / (g1 - g0) * (gt - g0)
    near = np.where(np.abs(tf - a0) <= np.abs(tf - a1), x0, x1)
    return np.where(inside, interp, near)


def thresholds_at(null: EtacNull, tfrac: float) -> np.ndarray:
    """``[case, pthr, h]`` FOM thresholds at a common tail fraction.

    3dXClustSim's rule: from each sub-test's descending list of per-iteration
    maxima (iterations with no cluster excluded), take the ``ithresh``-th
    largest with plotting positions ``(j + 0.666) / (n + 0.333)`` and
    interpolate linearly in Gumbel space ``log(-log(1-a))``.
    """
    out = np.zeros(null.fom.shape[1:], dtype=np.float64)
    for idx in np.ndindex(*null.fom.shape[1:]):
        out[idx] = float(_thr_curve(null.ranked(idx), null.n_train, tfrac))
    return out


def _tfrac_bounds(n_train: int) -> tuple[float, float]:
    """3dXClustSim's ``min_tfrac`` and ``0.666 * TOPFRAC``."""
    return min(6.0 / n_train, 1e-4), 0.666 * TOPFRAC


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


def search(null: EtacNull, fpr_goal: float, n_steps: int = 80) -> EtacThresholds:
    """Largest common tail fraction whose FPR does not exceed the goal.

    FPR rises monotonically with ``tfrac`` (thresholds only fall), so a
    bisection in log(tfrac) between AFNI's bounds finds it; the bounds are
    3dXClustSim's ``min_tfrac`` and ``0.666 * TOPFRAC``.
    """
    lo, hi = _tfrac_bounds(null.n_train)
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
        # Tight on purpose: the FWE p map reads survival off the exact FPR curve,
        # and a looser stop leaves a sliver where p <= goal but the cluster is
        # not in that goal's mask.  Every tfrac in the FPR's flat step is
        # equally valid, so this only picks the most lenient one.
        if b - a < 1e-12:
            break
    return best


# ---------------------------------------------------------------------------
# Applying thresholds to the observed maps
# ---------------------------------------------------------------------------


def apply(
    z_obs: dict[str, torch.Tensor],
    mask: Where,
    opts: EtacOptions,
    res: EtacThresholds,
    sign: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """``(union [V] bool, per-sub-test [case*pthr, V] bool)`` surviving voxels.

    ``sign=-1`` applies the same thresholds to ``-z`` (the 1-sided negative
    mask).  A voxel passes a (case, pthr) sub-test if its cluster's FOM for
    any requested h reaches that sub-test's threshold.
    """
    from fastfuncstuff.stats.cluster_gpu import cluster_labels
    from fastfuncstuff.stats.cluster_inference import as_geometry

    geom = as_geometry(mask)
    n_vox = geom.n
    cases = list(z_obs)
    all_mask = np.zeros((len(cases) * len(opts.pthr), n_vox), dtype=bool)
    hcol = {0: 0, 1: 1, 2: 2}
    for ic, case in enumerate(cases):
        z = z_obs[case] * sign
        nbr = geom.neighbors(opts.nn, z.device)
        weight = geom.weights(z.device)
        for ip, zt in enumerate(opts.zthr):
            active, _, foms = cluster_labels(
                z.float(), nbr, opts.sided_key, float(zt), opts.min_clust, weight
            )
            if active.numel() == 0:
                continue
            ok = torch.zeros(active.numel(), dtype=torch.bool, device=active.device)
            for ih, h in enumerate(opts.hpows):
                ok |= foms[hcol[h]] >= float(res.thr[ic, ip, ih])
            all_mask[ic * len(opts.pthr) + ip, active[ok].cpu().numpy()] = True
    return np.asarray(all_mask.any(0), dtype=bool), all_mask


# ---------------------------------------------------------------------------
# Family-wise p per voxel, and the surviving clusters
# ---------------------------------------------------------------------------


def n_test_rows(null: EtacNull) -> int:
    return len(range(*null.test_rows.indices(null.fom.shape[0])))


def _first_pass(vals: np.ndarray, n: int, fom: np.ndarray, n_steps: int = 80) -> np.ndarray:
    """Smallest tfrac at which a cluster of FOM ``fom`` passes one sub-test.

    ``inf`` where it does not pass even at the most lenient tfrac the search
    may reach.  Thresholds only fall as tfrac grows, so a bisection in
    log(tfrac) finds it; the returned tfrac always passes.
    """
    lo, hi = _tfrac_bounds(n)
    fom = np.asarray(fom, dtype=np.float64)
    out = np.full(fom.shape, np.inf)
    at_lo = fom >= _thr_curve(vals, n, np.full(fom.shape, lo))
    todo = (fom >= _thr_curve(vals, n, np.full(fom.shape, hi))) & ~at_lo
    out[at_lo] = lo
    if todo.any():
        f = fom[todo]
        a = np.full(f.shape, math.log(lo))
        b = np.full(f.shape, math.log(hi))
        for _ in range(n_steps):
            m = 0.5 * (a + b)
            ok = f >= _thr_curve(vals, n, np.exp(m))
            b, a = np.where(ok, m, b), np.where(ok, a, m)
        out[todo] = np.exp(b)
    return out


def fwe_pvalues(
    null: EtacNull,
    z_obs: dict[str, torch.Tensor],
    mask: Where,
    opts: EtacOptions,
    sign: int = 1,
) -> np.ndarray:
    """Per voxel, the smallest family-wise FPR (a fraction) at which ETAC keeps it.

    Survival only grows with the FPR goal (a larger goal, a larger tfrac,
    lower thresholds everywhere), so each voxel has a first goal at which its
    cluster passes some sub-test; this is that goal.  Thresholding the map at
    q gives the mask ETAC would write for an FPR goal of q, for every q at
    once.  1.0 where no cluster reaches a passing level; floored at one test
    iteration (``1 / n_test``), the finest step the permutations resolve.
    """
    from fastfuncstuff.stats.cluster_gpu import cluster_labels
    from fastfuncstuff.stats.cluster_inference import as_geometry

    geom = as_geometry(mask)
    n = null.n_train
    tf_vox = np.full(geom.n, np.inf)
    hcol = {0: 0, 1: 1, 2: 2}
    for ic, case in enumerate(z_obs):
        z = z_obs[case] * sign
        nbr = geom.neighbors(opts.nn, z.device)
        weight = geom.weights(z.device)
        for ip, zt in enumerate(opts.zthr):
            active, label, foms = cluster_labels(
                z.float(), nbr, opts.sided_key, float(zt), opts.min_clust, weight
            )
            if active.numel() == 0:
                continue
            lab = label.cpu().numpy()
            _, first, inv = np.unique(lab, return_index=True, return_inverse=True)
            fo = foms.double().cpu().numpy()[:, first]
            tf_c = np.full(first.size, np.inf)
            for ih, h in enumerate(opts.hpows):
                tf_c = np.minimum(tf_c, _first_pass(null.ranked((ic, ip, ih)), n, fo[hcol[h]]))
            np.minimum.at(tf_vox, active.cpu().numpy(), tf_c[inv])
    p = np.ones(geom.n)
    fin = np.isfinite(tf_vox)
    if fin.any():
        uniq, inv = np.unique(tf_vox[fin], return_inverse=True)
        fpr = np.array([fpr_at(null, thresholds_at(null, t)) for t in uniq]) / 100.0
        p[fin] = np.maximum(fpr[inv], 1.0 / n_test_rows(null))
    return p


def survivor_labels(res: EtacResult, goal: float, mask: Where) -> np.ndarray:
    """``[V]`` int32 cluster ids of the voxels surviving at ``goal`` (0 = none).

    Positive and negative survivors are labelled apart, as the viewer does, and
    ids run from the largest cluster (1) down.
    """
    from fastfuncstuff.stats.cluster_gpu import cluster_labels
    from fastfuncstuff.stats.cluster_inference import as_geometry

    geom = as_geometry(mask)
    keep = res.survivors(goal)
    out = np.zeros(geom.n, dtype=np.int32)
    if not keep.any():
        return out
    device = res.z_obs[res.display_case].device
    field = torch.as_tensor(np.where(keep, res.fwe_sign, 0), dtype=torch.float32, device=device)
    weight = geom.weights(device)
    active, label, foms = cluster_labels(
        field, geom.neighbors(res.opts.nn, device), "bi-sided", 0.5, 1, weight
    )
    act, lab = active.cpu().numpy(), label.cpu().numpy()
    ids, first, inv = np.unique(lab, return_index=True, return_inverse=True)
    size = foms[0].cpu().numpy()[first]
    rank = np.empty(ids.size, dtype=np.int32)
    rank[np.argsort(-size, kind="stable")] = np.arange(1, ids.size + 1, dtype=np.int32)
    out[act] = rank[inv]
    return out


def describe_clusters(
    res: EtacResult,
    goal: float,
    labels: np.ndarray,
    coords: np.ndarray,
    weight: np.ndarray | None = None,
) -> list[dict]:
    """One row per surviving cluster: extent, peak, centre of mass, FWE p, sub-tests.

    ``coords`` are ``[V, 3]`` mm positions of the in-mask voxels (or vertices),
    ``weight`` their areas on a surface (``None``: each voxel counts 1).
    ``subtests`` are the (blur, p) sub-tests in which some of the cluster
    passed at ``goal`` -- a description of where it was found, not a further
    test: choosing among them after the fact is the choice ETAC exists to
    remove.
    """
    z = res.z_obs[res.display_case].float().cpu().numpy()
    per = np.zeros_like(next(iter(res.masks.values()))[1])
    for (g, _side), (_u, m) in res.masks.items():
        if g == goal:
            per |= m
    sub = [(c, p) for c in res.case_labels for p in res.opts.pthr]
    w_all = np.ones(labels.size) if weight is None else np.asarray(weight, np.float64)
    rows = []
    for k in range(1, int(labels.max(initial=0)) + 1):
        idx = np.flatnonzero(labels == k)
        if idx.size == 0:
            continue
        sgn = int(res.fwe_sign[idx[0]])
        peak = int(idx[np.argmax(z[idx] * sgn)])
        w = w_all[idx]
        cm = (coords[idx] * w[:, None]).sum(0) / w.sum()
        hit = [s for s, m in zip(sub, per[:, idx].any(1), strict=True) if m]
        rows.append({
            "cluster": k,
            "sign": "+" if sgn > 0 else "-",
            "n": int(idx.size),
            "size": float(w.sum()),
            "z_peak": float(z[peak]),
            "peak": peak,
            "peak_mm": tuple(float(v) for v in coords[peak]),
            "cm_mm": tuple(float(v) for v in cm),
            "fwe_p": float(res.fwe_p[idx].min()),
            "strictest_pthr": min((p for _, p in hit), default=float("nan")),
            "subtests": [f"{c}_p{p:.4g}" for c, p in hit],
        })  # fmt: skip
    return rows


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
    # Smallest FPR (fraction) at which each voxel survives, and the side it
    # survives on (+1/-1; for 2-sided, the sign of the displayed z).
    fwe_p: np.ndarray
    fwe_sign: np.ndarray
    display_case: str  # the z shown with the result: unblurred when it was run
    n_test: int

    def survivors(self, goal: float) -> np.ndarray:
        """``[V]`` bool: in the ETAC mask at ``goal`` on either side."""
        return np.logical_or.reduce([u for (g, _), (u, _) in self.masks.items() if g == goal])


def case_label(fwhm: float) -> str:
    return f"B{fwhm:.1f}"


def run_etac(
    cases: dict[str, tuple[torch.Tensor, NullSource]],
    mask: Where,
    opts: EtacOptions,
    n_iter: int,
    device: torch.device,
    verbose: bool = True,
    n_distinct: float = math.inf,
) -> EtacResult:
    """Global ETAC over ready-made cases: ``{label: (observed z [V], null source)}``."""
    sources = {k: src for k, (_, src) in cases.items()}
    z_obs = {k: z for k, (z, _) in cases.items()}
    null_fom = build_null(sources, n_iter, mask, opts, device, verbose=verbose)
    thresholds, masks = [], {}
    sides = [("2sid", 1)] if opts.sided == 2 else [("1pos", 1), ("1neg", -1)]
    for goal in opts.fpr:
        res = search(null_fom, goal)
        thresholds.append(res)
        for side, sign in sides:
            masks[(goal, side)] = apply(z_obs, mask, opts, res, sign)
    p_side = {side: fwe_pvalues(null_fom, z_obs, mask, opts, sign) for side, sign in sides}
    display = case_label(0.0) if case_label(0.0) in z_obs else next(iter(z_obs))
    zsign = np.where(z_obs[display].float().cpu().numpy() < 0, -1, 1).astype(np.int8)
    if opts.sided == 2:
        fwe_p, fwe_sign = p_side["2sid"], zsign
    else:
        pos, neg = p_side["1pos"], p_side["1neg"]
        fwe_p = np.minimum(pos, neg)
        fwe_sign = np.where(pos < neg, 1, np.where(neg < pos, -1, zsign)).astype(np.int8)
    return EtacResult(
        case_labels=tuple(sources),
        opts=opts,
        n_iter=n_iter,
        n_distinct=n_distinct,
        z_obs=z_obs,
        thresholds=thresholds,
        masks=masks,
        fwe_p=fwe_p,
        fwe_sign=fwe_sign,
        display_case=display,
        n_test=n_test_rows(null_fom),
    )


def run_group_etac(
    design,
    ya: torch.Tensor,
    yb: torch.Tensor | None,
    mask: Where,
    voxdims: tuple[float, float, float] | None,
    *,
    model: str = "ols",
    ta: torch.Tensor | None = None,
    tb: torch.Tensor | None = None,
    knha: bool = True,
    reml: str = "always",
    blurs: tuple[float, ...] = (0.0,),
    opts: EtacOptions | None = None,
    n_iter: int = DEFAULT_NITER,
    seed: int = 0,
    device: torch.device,
    verbose: bool = True,
) -> EtacResult:
    """Global ETAC for the main test of a group model (``ols`` or ``mema``).

    Each blur case blurs the subject maps inside the mask (3dttest++
    ``-ETAC_blur``, on the device) and shares one set of relabelings with the
    other cases, as 3dttest++ does with a common ``-seed``.  MEMA takes no
    blur: blurring a beta changes its within-subject variance in a way the t
    map does not record.
    """
    from fastfuncstuff.stats.cluster_inference import VolumeGeometry, as_geometry
    from fastfuncstuff.stats.group_inference import brick_nulls, brick_to_z

    geom = as_geometry(mask)
    if isinstance(geom, VolumeGeometry) and voxdims is not None:
        geom = VolumeGeometry(geom.mask, voxdims)

    opts = opts or EtacOptions()
    if model == "mema" and any(b > 0 for b in blurs):
        raise ValueError("ETAC blur cases are OLS-only (blurring betas changes their variances)")
    cases: dict[str, tuple[torch.Tensor, NullSource]] = {}
    n_distinct = math.inf
    for fwhm in blurs:
        label = case_label(fwhm)
        if label in cases:
            raise ValueError(f"blur {fwhm} given twice")
        if fwhm > 0:
            blur = geom.smoother(fwhm, device)
            ya_b, yb_b = blur(ya), (blur(yb) if yb is not None else None)
        else:
            ya_b, yb_b = ya, yb
        bricks = _bricks(design, model, ya_b, yb_b, ta, tb, knha, reml)
        (bn,) = brick_nulls(design, model=model, ya=ya_b, yb=yb_b, ta=ta, tb=tb, n_iter=n_iter,
                            seed=seed, knha=knha, reml=reml, main_only=True)  # fmt: skip
        cases[label] = (brick_to_z(bricks[bn.brick + 1]).to(device), bn.source)
        n_distinct = bn.source.n_distinct
    return run_etac(cases, geom, opts, n_iter, device, verbose, n_distinct)


def _bricks(design, model, ya, yb, ta, tb, knha, reml):
    if model == "mema":
        from fastfuncstuff.stats.group_mema import run_mema

        return run_mema(design, ya, ta, yb, tb, knha=knha, reml=reml)[0]
    from fastfuncstuff.stats.group_ttest import run_ttest

    return run_ttest(design, ya, yb)
