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
    max_values: int = 20  # ladders stop subdividing before the lattice explodes
    # Smallest magnitude the backend can act on, from ParamSpec.resolution. Zero
    # stays legal; the open interval (0, resolution) does not, because a value in
    # there computes the same answer as 0 while costing a fit to discover it.
    resolution: float | None = None

    @classmethod
    def from_param(cls, p: ParamSpec) -> Axis:
        vals = list(p.values)
        numeric = all(_is_numeric(v) for v in vals)
        lo, hi = p.bounds
        if numeric:
            vals = sorted(vals)
            # A knob whose grid never goes negative is a magnitude (a variance, a
            # step size, a count), so zero is the floor rather than a value the
            # ladder should walk past into nonsense.
            if lo is None and min(vals) >= 0:
                lo = 0.0
        return cls(key=p.key, values=vals, numeric=numeric, lo=lo, hi=hi, resolution=p.resolution)

    def _inert(self, value: float) -> bool:
        """True if ``value`` is inside the axis's dead zone just above zero."""
        return self.resolution is not None and 0.0 < abs(value) < self.resolution

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
        if self._inert(nxt):
            nxt = 0.0  # the dead zone collapses onto the value it is identical to
        if any(abs(nxt - x) < 1e-9 for x in v):
            return False  # already there, or clamped onto the bound
        if all(_is_numeric(x) and float(x).is_integer() for x in v):
            nxt = int(round(nxt))
            if nxt in v:
                return False
        self.values = sorted([*v, nxt])
        return True

    def absorb(self, value: Any) -> bool:
        """Put a value the ladder does not have back onto it. True if it changed.

        The counterpart to persistence: a resumed run rebuilds its ladders from the
        configs the store already holds, rather than restarting at the recipe grid
        and re-deriving every subdivision a batch of fits at a time.
        """
        if any(_hashable(v) == _hashable(value) for v in self.values):
            return False
        if not self.numeric:
            self.values = [*self.values, value]
            return True
        if not _is_numeric(value) or self._inert(float(value)):
            return False  # a dead-zone value from a store written before that rule
        self.values = sorted([*self.values, value])
        return True

    def drop(self, value: Any) -> bool:
        """Remove a value from the ladder. True if it was there.

        Never empties the ladder, and never removes the only value left on an axis:
        a knob with nothing to vary is worse than a knob with one bad setting.
        """
        if len(self.values) <= 1:
            return False
        keep = [v for v in self.values if _hashable(v) != _hashable(value)]
        if len(keep) == len(self.values):
            return False
        self.values = keep
        return True

    def refine(self, value: Any) -> bool:
        """Insert midpoints either side of ``value``. True if the ladder changed.

        The counterpart to :meth:`grow`, which only ever extends the ends. Without
        this the search can reach a listed value but never anything *between* two of
        them, so an optimum sitting between rungs is unreachable however long it runs.

        Not hypothetical. On a real T1->MNI run the score gap between neighbouring
        levels ran 130-290x the run-to-run noise -- optiwarp_hs ``update_sigma``
        0.5 to 1.0 moved lncc by 0.23 against a noise floor of 0.001 -- so a
        doubling ladder steps clean over most of the structure it is meant to map.

        Subdivision is geometric where the ladder is, and stops at
        :data:`MIN_RELATIVE_STEP`: past that two neighbours are closer together than
        one fit can distinguish, so a finer step would assert a difference the data
        cannot support.
        """
        if not self.numeric or len(self.values) < 2:
            return False
        if len(self.values) >= self.max_values:
            return False
        try:
            i = next(k for k, x in enumerate(self.values) if abs(float(x) - float(value)) < 1e-9)
        except (StopIteration, TypeError, ValueError):
            return False

        integral = all(_is_numeric(x) and float(x).is_integer() for x in self.values)
        # Both neighbours are read before anything is inserted. Doing it in the loop
        # shifts the indices under the second pass, which silently bisects the wrong
        # interval -- it produced 0.85 (between 0.71 and 1.0) where 1.41 (between 1.0
        # and 2.0) was intended.
        neighbours = [float(self.values[j]) for j in (i - 1, i + 1) if 0 <= j < len(self.values)]
        centre = float(self.values[i])
        # An interval that touches zero cannot be judged against its own endpoints:
        # with lo == 0, `hi - lo` *equals* max(|hi|, |lo|), so a relative test can
        # never fire and the interval bisects toward zero forever. The scale that
        # means something there is the ladder's own span -- the knob's resolution is
        # set by its dynamic range, not by how near zero you happen to have wandered.
        span = abs(float(self.values[-1]) - float(self.values[0]))
        added = False
        for other in neighbours:
            lo, hi = sorted((centre, other))
            if integral and hi - lo <= 1:
                continue  # no integer sits between them
            scale = span if lo == 0.0 else max(abs(hi), abs(lo))
            if scale > 0 and (hi - lo) <= MIN_RELATIVE_STEP * scale:
                continue  # finer than a single fit can resolve
            mid = math.sqrt(lo * hi) if lo > 0 and hi / lo >= 1.8 else 0.5 * (lo + hi)
            mid = int(round(mid)) if integral else round(mid, 6)
            if self._inert(mid):
                continue  # computes the same answer as 0.0; not worth a fit
            if any(abs(float(mid) - float(x)) < 1e-9 for x in self.values):
                continue
            self.values = sorted([*self.values, mid])
            added = True
        return added


BAND_FLOOR = 1e-3
"""How much of the band's own probability mass the smoothness aim must keep.

Below this the aim has effectively vetoed every candidate, which is a statement
that the band is mapped rather than a ranking of what to try next.
"""

MIN_RELATIVE_STEP = 0.08
"""Stop subdividing once neighbouring values are within 8% of each other.

Below that the difference stops being resolvable against the noise of a single
fit, so a finer step buys a distinction the data cannot support.
"""


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

    def seed_from(self, observations: list[Observation]) -> list[str]:
        """Rebuild the ladders from configs a previous run already tried.

        Without this a resumed run starts at the recipe grid and has to *re-derive*
        every subdivision the earlier run paid for -- and each grow or refine step
        is only justified by a batch of fits, so the cost is not the arithmetic, it
        is the fits. Measured on a 7T epi2epi study: 455 fits took a 78-value
        lattice to 211 values, and all 133 of those subdivisions were thrown away by
        the next invocation.

        Every value ever tried is in the trial table, so the ladder is implied by
        the data and does not need its own persisted state -- which also means a
        store written by an older build seeds correctly, and one written before a
        rule like `ParamSpec.resolution` existed has its dead-zone values dropped
        on the way in rather than resurrected.
        """
        added: list[str] = []
        for axis in self.axes:
            new = [
                v
                for o in observations
                if (v := o.config.get(axis.key)) is not None and axis.absorb(v)
            ]
            if new:
                added.append(f"{axis.key} +{len(new)}")
        return added

    def prune_infeasible(self, observations: list[Observation], min_trials: int = 2) -> list[str]:
        """Drop ladder values that have never once produced a warp that held together.

        The safe half of "do not waste time on the bad edges". Scoring *worse* is
        not grounds for removing a value -- the frontier is precisely the argument
        that a worse score can be the better setting -- but a value that folded on
        every one of several tries is not a trade-off, it is a dead end, and the
        lattice keeps offering it to the acquisition for exploration.

        Requires ``min_trials`` failures before acting, because one fold is a
        property of the config it was tried in, not of the value.
        """
        dropped: list[str] = []
        for axis in self.axes:
            for value in list(axis.values):
                seen = [
                    o for o in observations if _hashable(o.config.get(axis.key)) == _hashable(value)
                ]
                if len(seen) >= min_trials and all(o.margin <= 0 for o in seen):
                    if axis.drop(value):
                        dropped.append(f"{axis.key}={value}")
        return dropped

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

    def refine_around(self, config: dict) -> list[str]:
        """Subdivide every ladder around the value this config uses.

        Paired with :meth:`grow_toward`, which handles the other case: extend where
        the incumbent is pinned against an end, subdivide where it sits between two
        rungs. To each according to its need.
        """
        refined = []
        for axis in self.axes:
            if axis.key not in config:
                continue
            before = list(axis.values)
            if axis.refine(config[axis.key]):
                new = [v for v in axis.values if v not in before]
                refined.append(f"{axis.key} +{'/'.join(str(v) for v in new)}")
        return refined

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
    roughness: float = 0.0  # bending energy of the field; lower is better


def propose(
    space: SearchSpace,
    observations: list[Observation],
    n: int,
    rng: np.random.Generator,
    cap: int = 20000,
    min_fit: int = 5,
    band: tuple[float, float] | None = None,
) -> list[dict]:
    """The next ``n`` configs to try, by constrained expected improvement.

    Below ``min_fit`` observations there is nothing to fit, so the batch is
    space-filling instead — maximin over the lattice, which is the honest thing
    to do when the surrogate would just be interpolating noise.

    **The target is the frontier, not the score.** Fitting the GP to similarity
    alone asks the surrogate to find the settings that fit hardest, and it obliges:
    on a 7T epi2epi run the search walked to no smoothing and a two-sample stopping
    window, which is a demons solver taking three enormous unregularised steps and
    memorising thermal noise. The metric it was handed rewarded that, so the search
    was not malfunctioning -- it was being asked the wrong question.

    So each batch scalarises the two objectives -- similarity and field roughness --
    under a *randomly drawn* weight (Knowles' ParEGO). One batch chases one point on
    the accuracy/smoothness trade; successive batches sweep the weight and populate
    the whole frontier. This is deliberately not a fixed penalty: a penalty needs a
    constant nobody can defend and commits the whole run to one point on a curve
    whose interesting property is its *shape* -- the flat stretch where roughness
    grows 20x for a hundredth of similarity is the finding, and you only see it by
    having candidates along the length of it.

    Roughness enters as log1p: bending energy is heavy-tailed, and on raw values a
    single warp that came apart sets the normalising scale for every other config.

    ``band`` switches the question from "what is better" to "what else scores about
    *this* well, more smoothly" -- see :func:`score_bands`. Improvement is not a
    useful target once the corner has been found, and a study needs the settings
    beside the winner as much as the winner.
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

    if band is not None:
        acq = _band_acquisition(space, observations, pool, xo, xc, band) * p_feasible
        return [pool[i] for i in _greedy_batch(acq, xc, n)]

    target = _scalarize(observations, rng)
    score_gp = GP.fit(xo, target)
    mu, sd = score_gp.predict(xc)

    # The improvement reference has to be on the same scale the GP was fitted to,
    # so it comes from the scalarised target rather than the raw score.
    feasible = [t for t, o in zip(target, observations, strict=True) if o.margin > 0]
    best = float(min(feasible)) if feasible else float(np.min(target))
    acq = _expected_improvement(mu, sd, best) * p_feasible
    return [pool[i] for i in _greedy_batch(acq, xc, n)]


def score_bands(
    observations: list[Observation], n: int = 5, reach: float = 0.5
) -> list[tuple[float, float]]:
    """Score intervals worth filling in, from the best result to ``reach`` of the field.

    The default search asks one question -- what is the best setting -- and answers
    it by walking into the corner where the least regularization that still passes
    the gate lives. That corner is real, but a study also needs the settings *next
    to* it: the ones that give up a little similarity for a field you would believe
    on a subject the tuner never saw. Expected improvement will never propose those,
    because they are by construction not improvements.

    So the room is cut into bands and each is filled deliberately. The cut is into
    equal *widths* of score, not equal counts: quantile bands would hold the same
    number of configs by construction and so could never point at the place the
    search skipped over, which is the whole complaint -- a study walks from a score
    of 6 to a score of 16 with nothing in between and the gap is invisible to any
    rank-based cut. An empty band is a finding, and it is the band that most needs
    a fit spent in it.

    The ends come from the data: the best result, and the ``reach`` quantile of the
    field (the median by default). Beyond that lie settings already known to be
    worse, and filling those in is not backing off, it is regressing.

    Feasible configs only. A degenerate range yields no bands, since nothing can
    land strictly inside one.
    """
    scores = sorted(score for _, score, _ in feasible_configs(observations))
    if len(scores) < 2 or n < 1:
        return []
    lo, hi = scores[0], float(np.quantile(scores, reach))
    if not hi > lo:
        return []
    edges = [lo + (hi - lo) * k / n for k in range(n + 1)]
    return [(lo, hi) for lo, hi in zip(edges[:-1], edges[1:], strict=True) if hi > lo]


def feasible_configs(observations: list[Observation]) -> list[tuple[dict, float, float]]:
    """One (config, score, roughness) per feasible config, averaged over subjects.

    Several trials at the same settings are several noisy reads of one point, and
    everything that reasons about *where a config sits* -- which band it is in,
    which is the smoothest one there -- has to agree with how the bands were cut.
    Testing raw trials instead makes a narrow band read as empty because its
    members' individual fits scattered outside it. The surrogate is the one thing
    that keeps the trials apart: it is modelling that scatter on purpose (see
    :class:`Observation`).
    """
    agg: dict[tuple, list[Observation]] = {}
    for o in observations:
        if o.margin > 0:
            agg.setdefault(config_key(o.config), []).append(o)
    return [
        (
            os[0].config,
            float(np.mean([o.score for o in os])),
            float(np.mean([max(o.roughness, 0.0) for o in os])),
        )
        for os in agg.values()
    ]


def in_band(
    observations: list[Observation], band: tuple[float, float]
) -> list[tuple[dict, float, float]]:
    """The feasible configs whose mean score falls inside ``band``."""
    lo, hi = band
    return [c for c in feasible_configs(observations) if lo <= c[1] <= hi]


def _band_acquisition(
    space: SearchSpace,
    observations: list[Observation],
    pool: list[dict],
    xo: np.ndarray,
    xc: np.ndarray,
    band: tuple[float, float],
) -> np.ndarray:
    """Probability of landing in ``band``, times of being smoother than what is there.

    Two factors, because "fill this band" alone would be answered by re-measuring
    the configs that already sit in it. What the band is being asked for is the
    *smoothest* way to score around there, so the second factor is the probability
    of beating the roughness of the best field already known at that level -- which
    is expected improvement again, just aimed sideways along the frontier rather
    than down it.

    With nothing yet observed inside the band there is no roughness to beat, and
    the factor drops out: anything that lands there is new information. The factor
    also drops out once it has nothing left to offer -- when no candidate is
    credibly smoother than what the band already holds, keeping it would return an
    all-zero acquisition and the batch would be chosen by tie-breaking alone. The
    band still has a question at that point ("what else scores here"), so it falls
    back to asking that one.
    """
    lo, hi = band
    mu, sd = GP.fit(xo, np.array([o.score for o in observations], dtype=float)).predict(xc)
    sd = np.clip(sd, 1e-6, None)
    acq = _norm_cdf((hi - mu) / sd) - _norm_cdf((lo - mu) / sd)

    rough = np.array([max(o.roughness, 0.0) for o in observations], dtype=float)
    inside = [r for _, _, r in in_band(observations, band)]
    if not inside or not rough.any():
        return acq
    r_mu, r_sd = GP.fit(xo, np.log1p(rough)).predict(xc)
    reference = math.log1p(min(inside))
    aimed = acq * _norm_cdf((reference - r_mu) / np.clip(r_sd, 1e-6, None))
    return aimed if aimed.max() > BAND_FLOOR * max(acq.max(), 1e-12) else acq


# Chebyshev scalarisation keeps a candidate honest on its *worst* objective, which
# is what makes it reach the concave parts of a frontier that a weighted sum skips
# over entirely. The small linear term breaks ties between points the max cannot
# distinguish, and is why the augmented form is the one everybody uses.
CHEBYSHEV_RHO = 0.05


def _unit(values: np.ndarray) -> np.ndarray:
    """Rescale to [0, 1]. A constant objective becomes all-zero: it cannot discriminate."""
    lo, hi = float(values.min()), float(values.max())
    return (values - lo) / (hi - lo) if hi > lo else np.zeros_like(values)


def scalarize_weight(rng: np.random.Generator) -> float:
    """Draw one batch's similarity/roughness weight. 1.0 is pure similarity."""
    return float(rng.uniform())


HV_REFERENCE = 1.1
"""Where the hypervolume's reference corner sits, in units of the normalised range.

Outside [0, 1] on purpose. On the boundary the best and worst points of a frontier
land exactly on the corners and dominate zero area, so a sound two-point trade-off
would measure as no trade-off at all.
"""


def _frontier_points(observations: list[Observation]) -> list[tuple[float, float]]:
    """Un-dominated (similarity, roughness) pairs, one per config, feasible only.

    Aggregated per config first: two subjects at the same settings are two noisy
    reads of one point, not two points.
    """
    agg: dict[tuple, list[Observation]] = {}
    for o in observations:
        if o.margin > 0:
            agg.setdefault(config_key(o.config), []).append(o)
    pts = [
        (
            float(np.mean([o.score for o in os])),
            float(np.mean([max(o.roughness, 0.0) for o in os])),
        )
        for os in agg.values()
    ]
    return [
        p
        for i, p in enumerate(pts)
        if not any(j != i and q[0] <= p[0] and q[1] < p[1] for j, q in enumerate(pts))
    ]


def frontier_hypervolume(observations: list[Observation]) -> float:
    """How much of the objective square the frontier dominates, in [0, 1].

    The convergence signal, and the second attempt at one. Counting *points* on
    the frontier does not work: with two continuous objectives, almost any config
    that is not strictly dominated joins it, so "did this round add a point?" is
    nearly always yes and the search never stops. Measured on a real study, a
    two-knob space ran 60 fits without the point count ever going stale.

    Area is the honest question. It grows when a round finds something genuinely
    better on either objective, and barely moves when a round fills in between
    points already mapped -- which is exactly the difference between refining the
    answer and refining the fifth decimal place of it.

    Both objectives are normalised over the feasible points seen so far, so this
    is a fraction rather than a quantity in anybody's units. That makes the
    reference move when a new extreme appears; that is intended, since a moving
    reference means something genuinely new turned up.

    The reference sits *outside* the observed range (:data:`HV_REFERENCE`) rather
    than on the worst observed value. On the boundary the two extreme points of any
    frontier normalise exactly onto the corners and dominate nothing, so the area
    of a perfectly good two-point trade-off comes out zero.
    """
    pts = _frontier_points(observations)
    if len(pts) < 2:
        return 0.0
    # Scaled by the frontier's OWN extent, never by the spread of everything tried.
    # Normalising over all feasible points lets a config that is worse on both
    # objectives enlarge the box and so make the frontier look better: measured, one
    # strictly dominated point took the area from 0.17 to 0.66, which would have
    # reset the staleness counter for finding something bad.
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xlo, xhi = min(xs), max(xs)
    ylo, yhi = min(ys), max(ys)
    xr, yr = (xhi - xlo) or 1.0, (yhi - ylo) or 1.0
    norm = sorted(((p[0] - xlo) / xr, (p[1] - ylo) / yr) for p in pts)

    # Sorted by x ascending the frontier's y is non-increasing, so the strip
    # [x_i, x_i+1) is covered up to height 1 - y_i against the (1, 1) reference.
    ref = HV_REFERENCE
    hv, n = 0.0, len(norm)
    for i, (x, y) in enumerate(norm):
        nxt = norm[i + 1][0] if i + 1 < n else ref
        hv += (nxt - x) * (ref - y)
    return hv / (ref * ref)


def scalarize_balanced(observations: list[Observation]) -> np.ndarray:
    """The scalarisation at an even weight — for judging, not for proposing.

    ``propose`` draws its weight at random precisely so that the *search* does not
    commit to one point on the frontier. Anything that has to name a single best
    config so far (which is what steers ladder growth) needs the opposite: a fixed,
    reproducible weight, so the answer does not move because the RNG advanced.
    """
    return _weighted(observations, 0.5)


def _scalarize(observations: list[Observation], rng: np.random.Generator) -> np.ndarray:
    """Fold (similarity, roughness) into the one number the GP regresses on.

    See :func:`propose` for why the weight is random rather than chosen.
    """
    score = _unit(np.array([o.score for o in observations], dtype=float))
    rough = _unit(np.log1p(np.array([max(o.roughness, 0.0) for o in observations], dtype=float)))
    if not rough.any():
        return score  # no roughness recorded (or all identical); nothing to trade against
    return _weighted(observations, scalarize_weight(rng))


def _weighted(observations: list[Observation], w: float) -> np.ndarray:
    """Augmented Chebyshev over (similarity, roughness) at weight ``w``."""
    score = _unit(np.array([o.score for o in observations], dtype=float))
    rough = _unit(np.log1p(np.array([max(o.roughness, 0.0) for o in observations], dtype=float)))
    if not rough.any():
        return score
    a, b = w * score, (1.0 - w) * rough
    return np.maximum(a, b) + CHEBYSHEV_RHO * (a + b)


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
