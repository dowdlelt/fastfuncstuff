"""Adaptive search over registration settings — the engine behind ``-search adaptive``.

The grid search this replaces has three costs, and they are separate problems:

1. **It spends fits proving what is already known.** On a T1->MNI run, 90 of 480
   fits re-confirmed a fold that the first subject's first two fits had already
   established. Screening on one subject and confirming only survivors fixes that.
2. **It cannot leave its own box.** Every optimum in that run sat on a range
   edge — ``total_var`` at its minimum, ``update_var`` and ``grad_step`` at their
   maxima. A search that always answers "go further" has not bracketed anything,
   and no amount of refining inside the box will find what is outside it.
3. **It treats the score and the gate as one surface.** They are not. The
   similarity score is smooth and nearly additive (measured: 97.5% main effects,
   2.1% two-way interaction), while the fold boundary is a cliff whose location
   *moves with other knobs* — ``optiwarp_hs`` at ``total_sigma=1.0`` is safe at
   ``update_sigma=0.5`` and folds at 1.0.

So this models the two separately and multiplies them: a Gaussian-process
surrogate for the score, a second GP for the continuous regularity margin
(:func:`warpqc.regularity_margin`), and an acquisition of expected improvement
times probability of feasibility. That is what makes it hunt the interesting
place — hard against the fold boundary without going over — rather than either
walking downhill into a folded warp or backing off into safe mediocrity.

A surrogate is also what handles interaction. Coordinate descent assumes the
knobs separate, which is exactly the assumption the ``hs`` boundary violates; a
GP over the joint space makes no such assumption and costs nothing extra at these
sizes (tens of points, a 3x3 solve).

Nothing here touches an image or a GPU: it consumes scored trials and emits
configs to try, so it is testable on synthetic response surfaces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np

from .tunespec import ParamSpec

# --- the search space -------------------------------------------------------


def _is_numeric(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


@dataclass
class Axis:
    """One knob as the searcher sees it, with a ladder that can grow.

    ``values`` starts as the ParamSpec grid but is not fixed: an optimum sitting
    on an end of the ladder is evidence the ladder is too short, not an answer.
    Only numeric axes can grow — there is no "one past ``gradmag``".
    """

    key: str
    values: list[Any]
    numeric: bool
    lo: float | None = None  # hard bound; regularization cannot go below zero
    hi: float | None = None

    @classmethod
    def from_param(cls, p: ParamSpec) -> Axis:
        vals = list(p.values)
        numeric = all(_is_numeric(v) for v in vals)
        lo = None
        if numeric:
            vals = sorted(vals)
            # A knob whose grid never goes negative is a magnitude (a variance, a
            # step size, a count), so zero is the floor rather than a value the
            # ladder should walk past into nonsense.
            if min(vals) >= 0:
                lo = 0.0
        return cls(key=p.key, values=vals, numeric=numeric, lo=lo)

    def grow(self, direction: int) -> bool:
        """Extend the ladder one step down (-1) or up (+1). True if it changed.

        The step matches the local spacing of the existing grid, and goes
        geometric when the grid is geometric — regularization ladders like
        ``0.5, 1.0, 2.0`` mean "double it", and extending those arithmetically
        would take smaller and smaller relative steps exactly where the
        interesting behaviour is coarsest.
        """
        if not self.numeric or len(self.values) < 2:
            return False
        v = self.values
        if direction < 0:
            step = v[1] - v[0]
            nxt = v[0] / 2.0 if (v[0] > 0 and v[1] / v[0] >= 1.8) else v[0] - step
            if self.lo is not None:
                nxt = max(nxt, self.lo)
        else:
            step = v[-1] - v[-2]
            nxt = v[-1] * 2.0 if (v[-2] > 0 and v[-1] / v[-2] >= 1.8) else v[-1] + step
            if self.hi is not None:
                nxt = min(nxt, self.hi)

        nxt = round(float(nxt), 6)
        if any(abs(nxt - x) < 1e-9 for x in v):
            return False  # already there, or clamped onto the bound
        if all(_is_numeric(x) and float(x).is_integer() for x in v):
            nxt = int(round(nxt))
            if nxt in v:
                return False
        self.values = sorted([*v, nxt])
        return True


@dataclass
class SearchSpace:
    """The axes of one backend, plus the encoding the surrogate sees."""

    axes: list[Axis]
    pins: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_params(
        cls, params: list[ParamSpec], pins: dict[str, Any] | None = None
    ) -> SearchSpace:
        pins = dict(pins or {})
        return cls(axes=[Axis.from_param(p) for p in params if p.key not in pins], pins=pins)

    @property
    def keys(self) -> list[str]:
        return [a.key for a in self.axes]

    def lattice(self, cap: int | None = None, rng: np.random.Generator | None = None) -> list[dict]:
        """Every config the current ladders allow, capped by random subsample.

        Enumerating is cheap because nothing is *fitted* here — the cap only
        exists so that a wide space with several expanded axes cannot make the
        acquisition step itself expensive.
        """
        if not self.axes:
            return [dict(self.pins)]
        grids = [[(a.key, v) for v in a.values] for a in self.axes]
        out = [{**self.pins, **dict(c)} for c in product(*grids)]
        if cap is not None and len(out) > cap:
            rng = rng or np.random.default_rng(0)
            idx = rng.choice(len(out), size=cap, replace=False)
            out = [out[i] for i in sorted(idx)]
        return out

    def encode(self, configs: list[dict]) -> np.ndarray:
        """Configs to a feature matrix in roughly [0, 1] per column.

        Numeric axes go through a signed log1p before normalising, because these
        knobs act multiplicatively: the gap from 0.0 to 0.5 regularization is a
        far bigger change than 2.0 to 2.5, and a linear encoding would tell the
        surrogate the opposite. Categorical axes become one-hot blocks, which
        keeps every level equidistant — there is no order to imply.
        """
        cols: list[np.ndarray] = []
        for a in self.axes:
            raw = [c[a.key] for c in configs]
            if a.numeric:
                pos = [abs(float(v)) for v in a.values if float(v) != 0.0]
                scale = float(np.median(pos)) if pos else 1.0
                t = np.array([_soft_log(float(v), scale) for v in raw])
                ref = np.array([_soft_log(float(v), scale) for v in a.values])
                span = float(ref.max() - ref.min()) or 1.0
                cols.append(((t - ref.min()) / span)[:, None])
            else:
                levels = list(a.values)
                oh = np.zeros((len(configs), len(levels)))
                for i, v in enumerate(raw):
                    if _hashable(v) in [_hashable(x) for x in levels]:
                        oh[i, [_hashable(x) for x in levels].index(_hashable(v))] = 1.0
                cols.append(oh)
        return np.hstack(cols) if cols else np.zeros((len(configs), 1))

    def edge_directions(self, config: dict) -> list[tuple[str, int]]:
        """Which axes this config sits at an end of, and which way is outward."""
        out = []
        for a in self.axes:
            if not a.numeric or a.key not in config:
                continue
            v = float(config[a.key])
            if abs(v - float(a.values[0])) < 1e-9:
                out.append((a.key, -1))
            elif abs(v - float(a.values[-1])) < 1e-9:
                out.append((a.key, +1))
        return out

    def grow_toward(self, config: dict) -> list[str]:
        """Extend every ladder this config is pinned against. Returns descriptions."""
        grown = []
        for key, direction in self.edge_directions(config):
            axis = next(a for a in self.axes if a.key == key)
            before = list(axis.values)
            if axis.grow(direction):
                new = [v for v in axis.values if v not in before]
                grown.append(f"{key} -> {new[0]}")
        return grown


def _soft_log(v: float, scale: float) -> float:
    return math.copysign(math.log1p(abs(v) / scale), v)


def _hashable(v: Any) -> Any:
    return tuple(v) if isinstance(v, list) else v


def config_key(config: dict) -> tuple:
    """Hashable identity for a config, for de-duplicating candidates."""
    return tuple(sorted((k, _hashable(v)) for k, v in config.items()))


# --- surrogate --------------------------------------------------------------


@dataclass
class GP:
    """A small Matern-5/2 Gaussian process. Exact, dense, and that is fine here.

    The whole point of the adaptive mode is that ``n`` stays in the tens, so an
    O(n^3) solve is microseconds and there is no reason to approximate anything.
    Hyperparameters come from a coarse marginal-likelihood grid rather than a
    gradient optimiser: with this little data the likelihood surface is broad,
    and a fragile optimiser is a worse failure mode than a coarse grid.
    """

    lengthscale: float
    noise: float
    x: np.ndarray
    alpha: np.ndarray
    chol: np.ndarray
    y_mean: float
    y_std: float

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray) -> GP:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        y_mean = float(y.mean())
        y_std = float(y.std()) or 1.0
        yz = (y - y_mean) / y_std

        best = None
        for ls in (0.15, 0.25, 0.4, 0.6, 0.9, 1.4, 2.2):
            for noise in (0.005, 0.02, 0.06, 0.15, 0.4):
                k = _matern52(x, x, ls) + noise * np.eye(len(x))
                try:
                    chol = np.linalg.cholesky(k)
                except np.linalg.LinAlgError:
                    continue
                alpha = _cho_solve(chol, yz)
                lml = -0.5 * float(yz @ alpha) - float(np.log(np.diag(chol)).sum())
                if best is None or lml > best[0]:
                    best = (lml, ls, noise, chol, alpha)
        if best is None:  # pragma: no cover - only if every jitter level fails
            raise np.linalg.LinAlgError("GP fit failed at every hyperparameter")
        _, ls, noise, chol, alpha = best
        return cls(ls, noise, x, alpha, chol, y_mean, y_std)

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and standard deviation, back on the original scale."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        ks = _matern52(x, self.x, self.lengthscale)
        mu = ks @ self.alpha
        v = np.linalg.solve(self.chol, ks.T)
        var = np.clip(1.0 + self.noise - (v**2).sum(axis=0), 1e-9, None)
        return mu * self.y_std + self.y_mean, np.sqrt(var) * self.y_std


def _matern52(a: np.ndarray, b: np.ndarray, ls: float) -> np.ndarray:
    d = np.sqrt(np.clip(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1), 0, None)) / ls
    s5 = math.sqrt(5.0)
    return (1.0 + s5 * d + (5.0 / 3.0) * d**2) * np.exp(-s5 * d)


def _cho_solve(chol: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.solve(chol.T, np.linalg.solve(chol, y))


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def _norm_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)


# --- acquisition ------------------------------------------------------------


@dataclass
class Observation:
    """One scored fit, as the optimiser sees it. One per *trial*, not per config.

    Deliberately not aggregated by config: several subjects at the same settings
    are several noisy reads of one point, and handing them to the GP separately
    lets it learn how much of the spread is between-subject noise. Averaging
    first would throw that away and make a config measured once look exactly as
    trustworthy as one measured five times.
    """

    config: dict
    score: float  # lower is better
    margin: float  # > 0 is a passing warp, per warpqc.regularity_margin


def propose(
    space: SearchSpace,
    observations: list[Observation],
    n: int,
    rng: np.random.Generator,
    cap: int = 20000,
    min_fit: int = 5,
) -> list[dict]:
    """The next ``n`` configs to try, by constrained expected improvement.

    Below ``min_fit`` observations there is nothing to fit, so the batch is
    space-filling instead — maximin over the lattice, which is the honest thing
    to do when the surrogate would just be interpolating noise.
    """
    tried = {config_key(o.config) for o in observations}
    pool = [c for c in space.lattice(cap=cap, rng=rng) if config_key(c) not in tried]
    if not pool:
        return []
    xc = space.encode(pool)

    if len(observations) < min_fit:
        seed = space.encode([o.config for o in observations]) if observations else None
        return [pool[i] for i in _maximin(xc, n, rng, seed=seed)]

    xo = space.encode([o.config for o in observations])
    score_gp = GP.fit(xo, np.array([o.score for o in observations]))
    mu, sd = score_gp.predict(xc)

    # Feasibility is a *separate* surface with its own geometry, so it gets its
    # own GP rather than being folded into the score as a penalty. A penalty
    # would let a very good score buy its way across the fold boundary, which is
    # the exact inversion this whole tool exists to prevent.
    margins = np.array([o.margin for o in observations])
    if margins.min() > 0 or margins.max() <= 0:
        p_feasible = np.ones(len(pool))  # no boundary observed yet; nothing to model
    else:
        m_mu, m_sd = GP.fit(xo, margins).predict(xc)
        p_feasible = _norm_cdf(m_mu / np.clip(m_sd, 1e-6, None))

    feasible_scores = [o.score for o in observations if o.margin > 0]
    best = (
        min(feasible_scores) if feasible_scores else float(np.min([o.score for o in observations]))
    )
    acq = _expected_improvement(mu, sd, best) * p_feasible
    return [pool[i] for i in _greedy_batch(acq, xc, n)]


def _expected_improvement(mu: np.ndarray, sd: np.ndarray, best: float) -> np.ndarray:
    sd = np.clip(sd, 1e-9, None)
    z = (best - mu) / sd
    return (best - mu) * _norm_cdf(z) + sd * _norm_pdf(z)


def _greedy_batch(acq: np.ndarray, x: np.ndarray, n: int) -> list[int]:
    """Top acquisition, but spread out.

    Picking the top ``n`` outright returns ``n`` neighbours of one point, which
    wastes a batch: they are one experiment measured repeatedly. After each pick
    the acquisition is damped around it, so the batch covers distinct claims.
    """
    acq = acq.copy()
    picked: list[int] = []
    radius = 0.25 * math.sqrt(x.shape[1])
    for _ in range(min(n, len(acq))):
        i = int(np.argmax(acq))
        if acq[i] <= 0 and picked:
            break
        picked.append(i)
        d = np.sqrt(((x - x[i]) ** 2).sum(axis=1))
        acq *= 1.0 - np.exp(-((d / max(radius, 1e-6)) ** 2))
    return picked


def _maximin(x: np.ndarray, n: int, rng: np.random.Generator, seed: np.ndarray | None) -> list[int]:
    """Greedy maximin subset: each pick is as far as possible from those taken."""
    picked: list[int] = []
    if seed is not None and len(seed):
        dist = np.sqrt(((x[:, None, :] - seed[None, :, :]) ** 2).sum(-1)).min(axis=1)
    else:
        first = int(rng.integers(len(x)))
        picked.append(first)
        dist = np.sqrt(((x - x[first]) ** 2).sum(axis=1))
    while len(picked) < n and len(picked) < len(x):
        i = int(np.argmax(dist))
        if i in picked:
            break
        picked.append(i)
        dist = np.minimum(dist, np.sqrt(((x - x[i]) ** 2).sum(axis=1)))
    return picked
