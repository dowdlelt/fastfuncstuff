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
    """A trial, block or miniblock: items shown in order, ``count`` times per run.

    ``family`` says which total governs it: ``"event"`` units (trials and null
    trials) are scaled by ``num_events``, ``"block"`` units (blocks and
    miniblocks) by ``num_blocks``. When a total or ``scan_time`` decides the
    counts, ``count`` is only a relative weight.
    """

    name: str
    items: list[Item]
    count: float  # whole for trials and blocks; a null unit's may be a fractional weight
    family: Literal["event", "block"] = "event"

    @property
    def is_null(self) -> bool:
        return all(it.condition == NULL for it in self.items)

    @classmethod
    def parse(
        cls, name: str, items: str, count: float, family: Literal["event", "block"] = "event"
    ) -> Unit:
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
        null = all(it.condition == NULL for it in parsed)
        if count <= 0 or (not null and count < 1):
            raise ValueError(f"unit {name!r}: count must be >= 1")
        return cls(name, parsed, float(count) if null else int(count), family)


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
    scan_time: float | None = None  # seconds per run; fixes the number of volumes
    num_events: int | None = None  # per-run total of non-null event units
    num_blocks: int | None = None  # per-run total of block units

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

    def _mean_gap(self, unit: Unit, j: int) -> float:
        item = unit.items[j]
        if item.off is not None:
            return item.off.mean
        return self.isi.mean if j == len(unit.items) - 1 else self.within_isi.mean

    def _expected_length(self, unit: Unit) -> tuple[float, float]:
        """(seconds from the unit's onset to the next unit's, its trailing gap), on average."""
        body = sum(it.duration + self._mean_gap(unit, j) for j, it in enumerate(unit.items))
        return body, self._mean_gap(unit, len(unit.items) - 1)

    def expected_duration(self, counts: list[float]) -> float:
        """Expected run length for unit ``counts``: the last unit's gap is -post_fix instead."""
        lengths = [self._expected_length(u) for u in self.units]
        n = sum(counts)
        if n == 0:
            return self.initial_fix + self.post_fix
        trailing = sum(c * t for c, (_, t) in zip(counts, lengths, strict=True)) / n
        body = sum(c * b for c, (b, _) in zip(counts, lengths, strict=True))
        return self.initial_fix + body - trailing + self.post_fix

    def resolve_counts(self) -> list[int]:
        """Units per run, after -num_events / -num_blocks / -scan_time.

        Each family is fixed by exactly one thing. A total (num_events,
        num_blocks) turns that family's counts into weights and scales them to
        it -- null trials scale with the events. Without a total, a family's
        counts stand as given, unless scan_time is set: then every family
        without a total is scaled, by one common factor, to fill the scan on
        average. If scan_time is set and nothing is left free, it only fixes the
        run length, and content that cannot fit is an error.
        """
        weights = [float(u.count) for u in self.units]
        counts = list(weights)

        def scale(idx: list[int], factor: float) -> None:
            for i in idx:
                counts[i] = weights[i] * factor

        events = [i for i, u in enumerate(self.units) if u.family == "event" and not u.is_null]
        nulls = [i for i, u in enumerate(self.units) if u.family == "event" and u.is_null]
        blocks = [i for i, u in enumerate(self.units) if u.family == "block"]
        free: list[int] = []
        if self.num_events is not None:
            if not events:
                raise ValueError("-num_events given but there are no -trial units")
            scale(events + nulls, self.num_events / sum(weights[i] for i in events))
        else:
            free += events + nulls
        if self.num_blocks is not None:
            if not blocks:
                raise ValueError("-num_blocks given but there are no -block/-miniblock units")
            scale(blocks, self.num_blocks / sum(weights[i] for i in blocks))
        else:
            free += blocks

        if self.scan_time is not None and free:
            fixed = [c if i not in free else 0.0 for i, c in enumerate(counts)]
            unit_free = [weights[i] if i in free else 0.0 for i in range(len(weights))]
            # expected_duration is affine in a common scale of the free counts
            # (up to the trailing-gap average, which is a small correction), so
            # solve it by bisection rather than algebra.
            lo, hi = 0.0, 1.0
            while (
                self.expected_duration([f + hi * w for f, w in zip(fixed, unit_free, strict=True)])
                < self.scan_time
            ):
                hi *= 2
                if hi > 1e6:
                    raise ValueError("cannot fill -scan_time: the units take no time")
            for _ in range(60):
                mid = (lo + hi) / 2
                d = self.expected_duration(
                    [f + mid * w for f, w in zip(fixed, unit_free, strict=True)]
                )
                lo, hi = (mid, hi) if d < self.scan_time else (lo, mid)
            scale(free, lo)

        def whole(down: set[int]) -> list[int]:
            """Each family to whole units, preserving its total (largest remainder).

            Families in ``down`` take the floor of their total instead of the
            nearest whole number.
            """
            out = [0] * len(counts)
            for fam in (events, nulls, blocks):
                if not fam:
                    continue
                exact = sum(counts[i] for i in fam)
                total = int(np.floor(exact + 1e-9)) if set(fam) & down else int(round(exact))
                floors = {i: int(np.floor(counts[i])) for i in fam}
                spare = total - sum(floors.values())
                for i in sorted(fam, key=lambda i: counts[i] - floors[i], reverse=True)[
                    : max(spare, 0)
                ]:
                    floors[i] += 1
                out = [floors.get(i, c) for i, c in enumerate(out)]
            return out

        out = whole(set())
        if (
            self.scan_time is not None
            and free
            and self.expected_duration([float(c) for c in out]) > self.scan_time + 1e-6
        ):
            # The families scan_time sized were rounded up past the scan (5.73
            # blocks -> 6, needing 345 s of a 330 s run). Refusing a count the
            # tool chose itself is wrong: take the largest that fits. The fit is
            # strict -- the content *and* the full -post_fix: a 2% allowance let 8
            # 30 s blocks into 330 s by cutting the final fixation from 15 s to 10.
            out = whole(set(free))
        for i, u in enumerate(self.units):
            if out[i] < 1 and not u.is_null:
                raise ValueError(
                    f"unit {u.name!r} rounds to 0 per run -- -scan_time or the totals are "
                    "too small for the design"
                )
        if self.scan_time is not None:
            need = self.expected_duration([float(c) for c in out])
            if need > self.scan_time + 1e-6:
                raise ValueError(
                    f"the units need ~{need:.0f} s per run on average, more than "
                    f"-scan_time {self.scan_time:g} s"
                )
        return out

    def scan_sized(self) -> bool:
        """Whether -scan_time chose any family's count (a family without its own total)."""
        if self.scan_time is None:
            return False
        fams = {u.family for u in self.units}
        return ("event" in fams and self.num_events is None) or (
            "block" in fams and self.num_blocks is None
        )

    def run_seconds(self, counts: list[int]) -> float | None:
        """Seconds per run: -scan_time, trimmed to the content when -scan_time sized it.

        A count sized to fill the scan is rounded, usually down (5.73 blocks
        -> 5), and padding the remainder with fixation would understate the
        design: per minute it is better than the padded scan says, and the
        time freed on every run adds up to more runs. So the run ends when
        its content does (rounded up to a whole TR), if that saves at least 5%
        of the scan -- a TR or two is not worth it, and a jittered realization
        that runs long would lose events to it. Counts the user fixed keep
        -scan_time's padding: that is chosen.
        """
        if self.scan_time is None:
            return None
        if not self.scan_sized():
            return self.scan_time
        need = self.expected_duration([float(c) for c in counts])
        trimmed = float(np.ceil(need / self.tr - 1e-9) * self.tr)
        return trimmed if self.scan_time - trimmed >= 0.05 * self.scan_time else self.scan_time

    def describe(self) -> str:
        counts = self.resolve_counts()
        lines = [
            f"TR {self.tr:g} s, {self.n_runs} run(s), fixation {self.initial_fix:g} s before / "
            f"{self.post_fix:g} s after, order {self.order}",
            f"between units: {self.isi}   within units: {self.within_isi}",
        ]
        if self.scan_time is not None:
            run_s = self.run_seconds(counts)
            assert run_s is not None
            lines.append(
                f"scan time {run_s:g} s per run ({int(round(run_s / self.tr))} volumes); "
                f"expected content {self.expected_duration([float(c) for c in counts]):.0f} s"
            )
            if run_s < self.scan_time:
                saved = self.scan_time - run_s
                lines.append(
                    f"  note: trimmed from -scan_time {self.scan_time:g} s -- the whole units "
                    f"that fit end at {run_s:g} s. {saved:g} s saved per run, "
                    f"{saved * self.n_runs:g} s over {self.n_runs} run(s)"
                    + (
                        f" ({saved * self.n_runs / run_s:.2f} of a run)"
                        if saved * self.n_runs < run_s
                        else f" -- enough for {int(saved * self.n_runs // run_s)} more run(s)"
                    )
                    + "."
                )
        for u, c in zip(self.units, counts, strict=True):
            items = ", ".join(
                f"{it.condition}:{it.duration:g}" + (f":{it.off}" if it.off else "")
                for it in u.items
            )
            lines.append(f"  {u.family:<5} {u.name:<10} x{c:<4} [{items}]")
        for fam, flag in (("event", "-num_events"), ("block", "-num_blocks")):
            idx = [i for i, u in enumerate(self.units) if u.family == fam and not u.is_null]
            got = {counts[i] for i in idx}
            if len({self.units[i].count for i in idx}) == 1 and len(got) > 1:
                k, total = len(idx), sum(counts[i] for i in idx)
                options = []
                for n in (total // k * k, (total // k + 1) * k):
                    if n == 0:
                        continue
                    trial = [float(c) for c in counts]
                    for i in idx:
                        trial[i] = n / k
                    run = float(np.ceil(self.expected_duration(trial) / self.tr - 1e-9) * self.tr)
                    options.append(f"{flag} {n} (runs of {run:g} s)")
                lines.append(
                    f"  note: equally weighted {fam} units got uneven counts "
                    f"({'/'.join(str(counts[i]) for i in idx)} per run): {total} is not a "
                    f"multiple of {k}. Balanced: " + " or ".join(options) + "."
                )
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
    counts: list[int] = field(default_factory=list)  # units per run, as resolved
    n_dropped: int = 0  # events of units that did not end -post_fix before a fixed scan's end


def realize(spec: ExperimentSpec, seed: int) -> Realization:
    """Draw one realization: unit order and every jittered interval, per run."""
    from fastfuncstuff.design.optimization import generate_event_sequence

    rng = np.random.default_rng(seed)
    conds = spec.conditions
    if not conds:
        raise ValueError("the experiment has no non-null conditions")
    durations = spec.durations()
    counts = spec.resolve_counts()
    onsets: dict[str, list[list[float]]] = {c: [[] for _ in range(spec.n_runs)] for c in conds}
    run_lengths, run_durations = [], []
    n_dropped = 0

    run_s = spec.run_seconds(counts)
    for run in range(spec.n_runs):
        order = [
            int(u)
            for u in generate_event_sequence(counts, len(spec.units), ordering=spec.order, rng=rng)
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

        # A fixed scan has a fixed number of volumes, and its last -post_fix
        # seconds are fixation: a unit whose events do not all end by then is
        # dropped whole (a cycle never loses its E2 and keeps its E1). Only
        # dropping events that *started* past the scan let jitter that is not
        # mean-matched (uniform) eat the final fixation -- 3.5 s of a 15 s one.
        limit = None if run_s is None else run_s - spec.post_fix
        t = spec.initial_fix
        for k, ui in enumerate(order):
            unit_events = []
            for j, item in enumerate(spec.units[ui].items):
                if item.condition != NULL:
                    unit_events.append((item.condition, t, item.duration))
                t += item.duration + gap.get((k, j), 0.0)
            if limit is not None and unit_events:
                if max(on + d for _, on, d in unit_events) > limit + 1e-9:
                    n_dropped += len(unit_events)
                    continue
            for cond, on, _ in unit_events:
                onsets[cond][run].append(on)
        t += spec.post_fix
        if run_s is not None:
            run_durations.append(run_s)
            run_lengths.append(int(round(run_s / spec.tr)))
        else:
            run_durations.append(t)
            run_lengths.append(int(np.ceil(t / spec.tr - 1e-9)))

    return Realization(
        seed=seed,
        conditions=conds,
        durations=durations,
        onsets=[[np.asarray(r) for r in onsets[c]] for c in conds],
        run_lengths=run_lengths,
        run_durations=run_durations,
        counts=counts,
        n_dropped=n_dropped,
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
