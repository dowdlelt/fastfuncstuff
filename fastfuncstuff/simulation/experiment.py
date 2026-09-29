"""Describe an experiment instead of listing its events, then draw realizations of it.

One abstraction covers event-related, block and miniblock designs: a run is a
sequence of **units**, and a unit is one or more **items**. Each item is

    LABEL:DUR[:OFF][xN]

a condition shown for DUR seconds, then OFF seconds of nothing before whatever
comes next, repeated N times. OFF is any :class:`Interval` spec, so it can be
fixed or jittered per position:

    -miniblock ABC "A:0.5:0, B:2:2, C:3:uniform:2,4" 10

is A for 0.5 s, straight into B for 2 s, 2 s off, C for 3 s, then 2-4 s before
the next unit. The last item's OFF is the gap to the next unit; items without
an OFF fall back to ``-within_isi`` inside a unit and ``-isi`` after it, so

    event-related trial   -trial A 2 20              (gap from -isi)
    per-type ITI          -miniblock A "A:2:exp:4,2,12" 20
    block                 -trial A 20 5
    block of events       -miniblock A "A:1:0.5x10" 5

The reserved condition ``null`` occupies time without producing an event, and
``uniform``/``exp``/``poisson`` cannot be condition names (they start interval
specs). Intervals are offset-to-onset. Every realization is one seed: a
jittered design is a distribution over event lists, and power is a property
of that distribution, not of one draw from it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

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
    off: Interval | None = None  # gap after this item; None -> -within_isi / -isi


_INTERVAL_KINDS = ("uniform", "exp", "poisson")
_ITEM = re.compile(r"([A-Za-z_][\w.]*):([\d.]+)(?::(.+?))?(?:[x*](\d+))?")


def _split_items(text: str) -> list[str]:
    """Split on commas, except those inside an OFF spec such as ``uniform:2,4``.

    A new item starts only at ``NAME:`` where NAME is not an interval kind; any
    other comma-separated piece belongs to the item before it.
    """
    items: list[str] = []
    for piece in (p.strip() for p in text.replace(";", ",").split(",")):
        head = re.match(r"([A-Za-z_][\w.]*):", piece)
        starts_item = head is not None and head.group(1).lower() not in _INTERVAL_KINDS
        if starts_item or not items:
            items.append(piece)
        else:
            items[-1] += "," + piece
    return [i for i in items if i]


@dataclass
class Unit:
    """A trial, block or miniblock: items shown in order, ``count`` times per run."""

    name: str
    items: list[Item]
    count: int

    @classmethod
    def parse(cls, name: str, items: str, count: int) -> Unit:
        """``items`` is ``LABEL:DUR[:OFF][xN]`` joined by commas (see module docstring)."""
        parsed: list[Item] = []
        for token in _split_items(items):
            m = _ITEM.fullmatch(token.replace(" ", ""))
            if m is None:
                raise ValueError(
                    f"cannot parse item {token!r} in {items!r}: use LABEL:DUR[:OFF][xN]"
                )
            label = m.group(1)
            if label.lower() in _INTERVAL_KINDS:
                raise ValueError(f"{label!r} starts an interval spec and cannot name a condition")
            off = Interval.parse(m.group(3)) if m.group(3) else None
            parsed += [Item(label, float(m.group(2)), off)] * int(m.group(4) or 1)
        if not parsed:
            raise ValueError(f"unit {name!r} has no items")
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
            items = ", ".join(
                f"{it.condition}:{it.duration:g}" + (f":{it.off}" if it.off else "")
                for it in u.items
            )
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
        order = [
            int(u)
            for u in generate_event_sequence(
                [u.count for u in spec.units], len(spec.units), ordering=spec.order, rng=rng
            )
        ]
        # Every gap slot in the run: (unit position k, item j) -> the interval it
        # draws from. The last unit's trailing gap is replaced by -post_fix.
        slots: dict[Any, list[tuple[int, int]]] = {}
        specs: dict[Any, Interval] = {}
        for k, ui in enumerate(order):
            items = spec.units[ui].items
            for j, item in enumerate(items):
                last = j == len(items) - 1
                if last and k == len(order) - 1:
                    continue
                if item.off is not None:
                    key: Any = ("item", ui, j)
                    specs[key] = item.off
                else:
                    key = "isi" if last else "within"
                    specs[key] = spec.isi if last else spec.within_isi
                slots.setdefault(key, []).append((k, j))
        # Draw each slot family in one call over the whole run. The mean-matched
        # generators force a single draw to the mean, so drawing gap by gap (as
        # the within-unit gaps once were) silently removed all jitter.
        gap: dict[tuple[int, int], float] = {}
        for key, where in slots.items():
            for pos, value in zip(where, specs[key].sample(len(where), rng, spec.tr), strict=True):
                gap[pos] = float(value)

        t = spec.initial_fix
        for k, ui in enumerate(order):
            for j, item in enumerate(spec.units[ui].items):
                if item.condition != NULL:
                    onsets[item.condition][run].append(t)
                t += item.duration + gap.get((k, j), 0.0)
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
