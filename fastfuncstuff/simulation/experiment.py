"""Describe an experiment instead of listing its events, then draw realizations of it.

One abstraction covers event-related, block and miniblock designs: a run is a
sequence of **units**. A unit is one or more **items** -- a condition shown for
a duration -- separated by a *within-unit* interval, and units are separated by
a *between-unit* interval.

    event-related trial   one item          -trial A 2 20
    block                 one long item     -trial A 20 5
    block of events       one item x10      -miniblock A "A:1x10" 5
    miniblock             several items     -miniblock AB "A:2,B:2" 10

The reserved condition ``null`` occupies time without producing an event, which
is how blank trials are described. Intervals are measured offset-to-onset (the
gap between one stimulus ending and the next starting), and can be fixed or
jittered -- see :class:`Interval`. Every realization is one seed: a design that
is "20 trials with exponential jitter" is a distribution over event lists, and
power is a property of that distribution, not of one draw from it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

NULL = "null"


@dataclass(frozen=True)
class Interval:
    """A fixed or jittered interval in seconds.

    Spec strings: ``4`` (fixed), ``uniform:2,6``, ``exp:MEAN,MIN,MAX``
    (truncated exponential above MIN, mean-matched -- the usual jittered-ISI
    choice), ``poisson:MEAN,MIN,MAX`` (whole multiples of the TR).
    """

    kind: Literal["fixed", "uniform", "exp", "poisson"]
    mean: float
    low: float
    high: float

    @classmethod
    def parse(cls, spec: str | float) -> Interval:
        if isinstance(spec, (int, float)):
            return cls("fixed", float(spec), float(spec), float(spec))
        text = str(spec).strip()
        try:
            if ":" not in text:
                v = float(text)
                return cls("fixed", v, v, v)
            kind, args = text.split(":", 1)
            vals = [float(x) for x in args.split(",")]
        except ValueError:
            raise ValueError(f"cannot parse interval {spec!r}") from None
        kind = kind.lower()
        if kind == "uniform" and len(vals) == 2:
            lo, hi = sorted(vals)
            return cls("uniform", (lo + hi) / 2, lo, hi)
        if kind in ("exp", "poisson") and len(vals) == 3:
            mean, lo, hi = vals
            if not lo <= mean <= hi or lo >= hi:
                raise ValueError(f"{spec!r}: need MIN <= MEAN <= MAX and MIN < MAX")
            return cls(kind, mean, lo, hi)  # type: ignore[arg-type]
        raise ValueError(
            f"cannot parse interval {spec!r}: use N, uniform:LO,HI, exp:MEAN,MIN,MAX "
            "or poisson:MEAN,MIN,MAX"
        )

    def sample(self, n: int, rng: np.random.Generator, tr: float = 1.0) -> np.ndarray:
        if n <= 0:
            return np.zeros(0)
        if self.kind == "fixed":
            return np.full(n, self.mean)
        from fastfuncstuff.design.optimization import ISIConstraints, generate_isi_sequence

        dist = {"uniform": "uniform", "exp": "truncated_exponential", "poisson": "poisson"}[
            self.kind
        ]
        c = ISIConstraints(min_isi=self.low, max_isi=self.high, mean_isi=self.mean, tr=tr)
        return generate_isi_sequence(n + 1, c, dist, rng=rng)  # type: ignore[arg-type]

    def __str__(self) -> str:
        if self.kind == "fixed":
            return f"{self.mean:g}"
        if self.kind == "uniform":
            return f"uniform:{self.low:g},{self.high:g}"
        return f"{self.kind}:{self.mean:g},{self.low:g},{self.high:g}"


@dataclass(frozen=True)
class Item:
    condition: str
    duration: float


@dataclass
class Unit:
    """A trial, block or miniblock: items shown in order, ``count`` times per run."""

    name: str
    items: list[Item]
    count: int

    @classmethod
    def parse(cls, name: str, items: str, count: int) -> Unit:
        """``items`` is ``COND:DUR[xN]`` joined by commas, e.g. ``A:2,B:2`` or ``A:1x10``."""
        parsed: list[Item] = []
        for token in items.split(","):
            m = re.fullmatch(r"\s*([A-Za-z_][\w.]*)\s*:\s*([\d.]+)\s*(?:[x*]\s*(\d+))?\s*", token)
            if m is None:
                raise ValueError(f"cannot parse item {token!r} in {items!r}: use COND:DUR[xN]")
            reps = int(m.group(3) or 1)
            parsed += [Item(m.group(1), float(m.group(2)))] * reps
        if count < 1:
            raise ValueError(f"unit {name!r}: count must be >= 1")
        return cls(name, parsed, int(count))


@dataclass
class ExperimentSpec:
    tr: float
    units: list[Unit]
    n_runs: int = 1
    isi: Interval = field(default_factory=lambda: Interval.parse(0))
    within_isi: Interval = field(default_factory=lambda: Interval.parse(0))
    initial_fix: float = 0.0
    post_fix: float = 0.0
    order: Literal["random", "alternating", "blocked", "permuted_block"] = "random"

    @property
    def conditions(self) -> list[str]:
        seen: dict[str, None] = {}
        for u in self.units:
            for it in u.items:
                if it.condition != NULL:
                    seen.setdefault(it.condition)
        return list(seen)

    def durations(self) -> list[float]:
        """One duration per condition -- the design builder convolves per condition."""
        found: dict[str, float] = {}
        for u in self.units:
            for it in u.items:
                if it.condition == NULL:
                    continue
                if found.setdefault(it.condition, it.duration) != it.duration:
                    raise ValueError(
                        f"condition {it.condition!r} appears with durations "
                        f"{found[it.condition]:g} and {it.duration:g}; give each its own name"
                    )
        return [found[c] for c in self.conditions]

    def describe(self) -> str:
        lines = [
            f"TR {self.tr:g} s, {self.n_runs} run(s), fixation {self.initial_fix:g} s before / "
            f"{self.post_fix:g} s after, order {self.order}",
            f"between units: {self.isi}   within units: {self.within_isi}",
        ]
        for u in self.units:
            items = ", ".join(f"{it.condition}:{it.duration:g}" for it in u.items)
            lines.append(f"  unit {u.name:<10} x{u.count:<4} [{items}]")
        return "\n".join(lines)


@dataclass
class Realization:
    """One concrete draw of an experiment: event onsets in seconds, per run."""

    seed: int
    conditions: list[str]
    durations: list[float]
    onsets: list[list[np.ndarray]]  # [condition][run] -> seconds
    run_lengths: list[int]  # timepoints
    run_durations: list[float]  # seconds, before rounding up to whole TRs


def realize(spec: ExperimentSpec, seed: int) -> Realization:
    """Draw one realization: unit order and every jittered interval, per run."""
    from fastfuncstuff.design.optimization import generate_event_sequence

    rng = np.random.default_rng(seed)
    conds = spec.conditions
    if not conds:
        raise ValueError("the experiment has no non-null conditions")
    durations = spec.durations()
    onsets: dict[str, list[list[float]]] = {c: [[] for _ in range(spec.n_runs)] for c in conds}
    run_lengths, run_durations = [], []

    for run in range(spec.n_runs):
        order = generate_event_sequence(
            [u.count for u in spec.units], len(spec.units), ordering=spec.order, rng=rng
        )
        gaps = spec.isi.sample(len(order) - 1, rng, spec.tr)
        t = spec.initial_fix
        for k, unit_idx in enumerate(order):
            unit = spec.units[int(unit_idx)]
            within = spec.within_isi.sample(len(unit.items) - 1, rng, spec.tr)
            for j, item in enumerate(unit.items):
                if item.condition != NULL:
                    onsets[item.condition][run].append(t)
                t += item.duration
                if j < len(unit.items) - 1:
                    t += within[j]
            if k < len(order) - 1:
                t += gaps[k]
        t += spec.post_fix
        run_durations.append(t)
        run_lengths.append(int(np.ceil(t / spec.tr - 1e-9)))

    return Realization(
        seed=seed,
        conditions=conds,
        durations=durations,
        onsets=[[np.asarray(r) for r in onsets[c]] for c in conds],
        run_lengths=run_lengths,
        run_durations=run_durations,
    )


def parse_contrast(expr: str, conditions: list[str]) -> np.ndarray:
    """``A-B``, ``A+B-2*C``, ``0.5*A - 0.5*B`` -> weights over ``conditions``."""
    w = np.zeros(len(conditions))
    text = expr.replace(" ", "")
    pos = 0
    for m in re.finditer(r"([+-]?)(\d*\.?\d*)\*?([A-Za-z_][\w.]*)", text):
        if m.start() != pos:
            break
        pos = m.end()
        name = m.group(3)
        if name not in conditions:
            raise ValueError(f"contrast {expr!r}: unknown condition {name!r} ({conditions})")
        coef = float(m.group(2)) if m.group(2) else 1.0
        w[conditions.index(name)] += -coef if m.group(1) == "-" else coef
    if pos != len(text) or not w.any():
        raise ValueError(f"cannot parse contrast {expr!r}")
    return w


def default_contrasts(conditions: list[str]) -> dict[str, np.ndarray]:
    """Each condition against baseline, then every pairwise difference (Liu & Frank's set)."""
    out = {c: parse_contrast(c, conditions) for c in conditions}
    for i, a in enumerate(conditions):
        for b in conditions[i + 1 :]:
            out[f"{a}-{b}"] = parse_contrast(f"{a}-{b}", conditions)
    return out
