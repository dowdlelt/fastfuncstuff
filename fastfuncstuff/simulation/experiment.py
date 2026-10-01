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

Items may be separated by commas or spaces, and ``isi:SPEC`` is an item that
is only a gap (a zero-length ``null`` with that OFF), so a unit can open with
one: ``"isi:uniform:2,8 A:0.5 B:2"``. OFF can also come from a list:
``uniform:(2,5,9)`` (any of them each time) or ``even:(2,5,9)`` (equally often
within a run). A list can also be named once and shared, ``-isi_list SP
(4,9,11)``, then used as ``even:SP`` (or just ``SP``) and ``uniform:SP``,
optionally shifted, ``SP-3``. Every ``even:SP`` gap in a run -- whichever unit
or item it follows -- is balanced as one pool, and the leftovers are dealt
across runs so the experiment is balanced too. The reserved condition ``null`` occupies time without
producing an event, and ``uniform``/``even``/``exp``/``poisson``/``isi``
cannot be condition names (they start interval specs or gaps). Intervals are offset-to-onset. Every realization is one seed: a
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
    choice), ``poisson:MEAN,MIN,MAX`` (whole multiples of the TR), and from a
    list: ``uniform:(2,5,9)`` draws each gap from the values, ``even:(2,5,9)``
    uses them equally often within a run (the remainder, when the count does
    not divide, drawn without replacement) in shuffled order.

    A named list (``lists``, from -isi_list) is ``even:NAME``, ``uniform:NAME``
    or bare ``NAME`` (even), with an optional shift, ``NAME-3`` / ``NAME+0.5``
    (inline lists take one too). ``values`` keeps the unshifted list; mean, low
    and high are of the shifted gaps. A named even interval is a ``pool``:
    :func:`draw_plans` balances every gap that names it together, not per item.
    """

    kind: Literal["fixed", "uniform", "exp", "poisson", "choice", "even"]
    mean: float
    low: float
    high: float
    values: tuple[float, ...] = ()
    pool: str = ""  # the -isi_list name an even interval is balanced across
    shift: float = 0.0  # added to every value drawn from ``values``

    @classmethod
    def parse(
        cls, spec: str | float, lists: dict[str, tuple[float, ...]] | None = None
    ) -> Interval:
        if isinstance(spec, (int, float)):
            return cls("fixed", float(spec), float(spec), float(spec))
        text = str(spec).strip().replace(" ", "")
        listed = re.fullmatch(
            r"(?:(uniform|even):)?(\(.+\)|[A-Za-z_]\w*)([+-][\d.]+)?", text, flags=re.IGNORECASE
        )
        if listed and listed.group(2).startswith("(") and not listed.group(1):
            listed = None  # a bare (2,5,9) is not a spec
        if listed:
            body, name = listed.group(2), ""
            if not body.startswith("("):
                name = body
                if name not in (lists or {}):
                    known = ", ".join(lists or {}) or "none defined"
                    raise ValueError(
                        f"cannot parse interval {spec!r}: {name!r} is not an -isi_list "
                        f"({known}); define it with -isi_list {name} (4,9,11)"
                    )
            try:
                vals = (
                    list((lists or {})[name])
                    if name
                    else [float(x) for x in body[1:-1].split(",") if x]
                )
                shift = float(listed.group(3) or 0)
            except ValueError:
                raise ValueError(f"cannot parse interval {spec!r}") from None
            if not vals:
                raise ValueError(f"cannot parse interval {spec!r}: the list is empty")
            if min(vals) + shift < 0:
                raise ValueError(
                    f"{spec!r}: shifted by {shift:g}, the smallest gap is "
                    f"{min(vals) + shift:g} s -- gaps cannot be negative"
                )
            even = (listed.group(1) or "even").lower() == "even"
            return cls(
                "even" if even else "choice",
                float(np.mean(vals)) + shift,
                min(vals) + shift,
                max(vals) + shift,
                tuple(vals),
                name if even else "",
                shift,
            )
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
        if kind in ("uniform", "even") and len(vals) != 2:
            a = ",".join(f"{v:g}" for v in vals)
            raise ValueError(
                f"{spec!r}: uniform:LO,HI is a range of two values. To draw from a list, "
                f"write uniform:({a}) (each gap any of them) or even:({a}) (all equally "
                "often within a run)"
            )
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
        vals = np.asarray(self.values, dtype=float)
        if self.kind == "choice":
            return rng.choice(vals, n) + self.shift
        if self.kind == "even":
            whole, rest = divmod(n, len(vals))
            out = np.concatenate([np.repeat(vals, whole), rng.choice(vals, rest, replace=False)])
            return rng.permutation(out) + self.shift
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
        if self.kind in ("choice", "even"):
            kind = "uniform" if self.kind == "choice" else "even"
            body = self.pool or f"({','.join(f'{v:g}' for v in self.values)})"
            return f"{kind}:{body}" + (f"{self.shift:+g}" if self.shift else "")
        return f"{self.kind}:{self.mean:g},{self.low:g},{self.high:g}"


@dataclass(frozen=True)
class Item:
    condition: str
    duration: float
    off: Interval | None = None  # gap after this item; None -> -within_isi / -isi


_INTERVAL_KINDS = ("uniform", "exp", "poisson", "even")
_GAP = "isi"  # an item that is only a gap: isi:4, isi:uniform:2,8
_ITEM = re.compile(r"([A-Za-z_][\w.]*):([\d.]+)(?::(.+?))?(?:[x*](\d+))?")


def reserved_name(name: str) -> bool:
    """Names that start interval specs or gap items, and so cannot label anything."""
    return name.lower() in (*_INTERVAL_KINDS, _GAP, NULL)


def _split_items(text: str) -> list[str]:
    """Split on commas, semicolons or spaces, except inside an OFF spec.

    Separators inside parentheses (``even:(2,5,9)``) never split. A new item
    starts only at ``NAME:`` where NAME is not an interval kind; any other
    piece belongs to the item before it (the ``4`` of ``uniform:2,4``, the
    ``x15`` of ``A:1:4 x15``).
    """
    pieces, depth, cur = [], 0, ""
    for ch in text:
        depth += (ch == "(") - (ch == ")")
        if depth == 0 and (ch in ",;" or ch.isspace()):
            pieces.append(cur)
            cur = ""
        else:
            cur += ch
    pieces.append(cur)
    items: list[str] = []
    for piece in (p.strip() for p in pieces):
        if not piece:
            continue
        head = re.match(r"([A-Za-z_][\w.]*):", piece)
        starts_item = head is not None and head.group(1).lower() not in _INTERVAL_KINDS
        if starts_item or not items:
            items.append(piece)
        elif re.fullmatch(r"[x*]\d+", piece):
            items[-1] += piece  # "A:1:J x15": a repeat written apart from its item
        else:
            items[-1] += "," + piece
    return [i for i in items if i]


@dataclass
class Unit:
    """A trial, block or miniblock: items shown in order, ``count`` times per run.

    ``shuffle`` (-shuffle_items) draws a new order of its items every time
    the unit occurs: the conditions, with their durations, move; each gap
    belongs to its position, so the last written item's OFF is always the gap
    after the unit. Gap items (``isi:``) stay where they are written.

    ``family`` says which total governs it: ``"event"`` units (trials and null
    trials) are scaled by ``num_events``, ``"block"`` units (blocks and
    miniblocks) by ``num_blocks``. When a total or ``scan_time`` decides the
    counts, ``count`` is only a relative weight.
    """

    name: str
    items: list[Item]
    count: float  # whole for trials and blocks; a null unit's may be a fractional weight
    family: Literal["event", "block"] = "event"
    shuffle: bool = False  # items in a fresh order each time; gaps stay by position

    @property
    def is_null(self) -> bool:
        return all(it.condition == NULL for it in self.items)

    @classmethod
    def parse(
        cls,
        name: str,
        items: str,
        count: float,
        family: Literal["event", "block"] = "event",
        lists: dict[str, tuple[float, ...]] | None = None,
    ) -> Unit:
        """``items`` is ``LABEL:DUR[:OFF][xN]`` joined by commas (see module docstring);
        ``lists`` are the named gap lists an OFF may use."""
        parsed: list[Item] = []
        for token in _split_items(items):
            gap = re.fullmatch(rf"{_GAP}:(.+?)(?:[x*](\d+))?", token.replace(" ", ""), re.I)
            if gap:
                # Time with no event, e.g. before the first item: a zero-length
                # null item whose OFF is the gap.
                try:
                    off = Interval.parse(gap.group(1), lists)
                except ValueError as exc:
                    raise ValueError(
                        f"{token!r}: isi is reserved for a gap item (isi:SPEC) and cannot "
                        f"name a condition -- {exc}"
                    ) from None
                parsed += [Item(NULL, 0.0, off)] * int(gap.group(2) or 1)
                continue
            m = _ITEM.fullmatch(token.replace(" ", ""))
            if m is None:
                raise ValueError(
                    f"cannot parse item {token!r} in {items!r}: use LABEL:DUR[:OFF][xN]"
                )
            label = m.group(1)
            if label.lower() in _INTERVAL_KINDS:
                raise ValueError(f"{label!r} starts an interval spec and cannot name a condition")
            off = Interval.parse(m.group(3), lists) if m.group(3) else None
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
    tr_lock: bool = False  # every onset on a TR boundary (gaps snapped to TR steps)

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

        A scan is whole volumes, so -scan_time is rounded *up* to a whole TR: 331 s
        at TR 2 is 166 volumes, 332 s -- it had rounded the volumes but kept the
        seconds, so a run's length and its duration disagreed.
        """
        if self.scan_time is None:
            return None
        full = float(np.ceil(self.scan_time / self.tr - 1e-9) * self.tr)
        if not self.scan_sized():
            return full
        need = self.expected_duration([float(c) for c in counts])
        trimmed = float(np.ceil(need / self.tr - 1e-9) * self.tr)
        return trimmed if full - trimmed >= 0.05 * full else full

    def describe(self) -> str:
        counts = self.resolve_counts()
        lines = [
            f"TR {self.tr:g} s, {self.n_runs} run(s), fixation {self.initial_fix:g} s before / "
            f"{self.post_fix:g} s after, order {self.order}",
            f"between units: {self.isi}   within units: {self.within_isi}"
            + ("   onsets locked to the TR grid" if self.tr_lock else ""),
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
            lines.append(
                f"  {u.family:<5} {u.name:<10} x{c:<4} [{items}]"
                + ("  items shuffled each time" if u.shuffle else "")
            )
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


@dataclass
class RunPlan:
    """One run before it becomes a timeline: its units in order, each with the gap
    after each of its items.

    Every entry carries its full gap list, the last unit's trailing gap
    included (it is replaced by -post_fix when assembled), so reordering keeps
    each unit's gaps with it -- which is what a design search mutates -- and
    the order its items are shown in (``range(n_items)`` unless the unit is
    shuffled; the gaps are by position, not by item).
    """

    entries: list[tuple[int, list[float], list[int]]]  # (unit, gap after each slot, items)


def draw_plans(spec: ExperimentSpec, seed: int) -> tuple[list[int], list[RunPlan]]:
    """Resolved counts and one drawn :class:`RunPlan` per run (order and every gap)."""
    from fastfuncstuff.design.optimization import generate_event_sequence

    rng = np.random.default_rng(seed)
    # The last unit's trailing gap is not part of the run (-post_fix replaces it);
    # a design search still needs one per entry, drawn from its own stream so the
    # main one -- and every realization drawn before RunPlan existed -- is unchanged.
    spare_rng = np.random.default_rng([seed, 1])
    counts = spec.resolve_counts()
    plans = []
    decks: dict[Any, list[int]] = {}  # a pooled list's leftovers, dealt across runs
    for _ in range(spec.n_runs):
        order = [
            int(u)
            for u in generate_event_sequence(counts, len(spec.units), ordering=spec.order, rng=rng)
        ]
        # Every gap slot in the run: (unit position k, item j) -> the interval it
        # draws from. A named even list is one slot family wherever it is used.
        slots: dict[Any, list[tuple[int, int]]] = {}
        specs: dict[Any, Interval] = {}
        shift: dict[tuple[int, int], float] = {}  # pooled slots: each its own shift
        spare: tuple[Interval, tuple[int, int]] | None = None
        for k, ui in enumerate(order):
            items = spec.units[ui].items
            for j, item in enumerate(items):
                last = j == len(items) - 1
                if item.off is not None:
                    key: Any = ("item", ui, j)
                    off = item.off
                else:
                    key = "isi" if last else "within"
                    off = spec.isi if last else spec.within_isi
                if last and k == len(order) - 1:
                    spare = (off, (k, j))
                    continue
                if off.pool:
                    key = ("pool", off.pool)
                    shift[(k, j)] = off.shift
                specs[key] = off
                slots.setdefault(key, []).append((k, j))
        # Draw each slot family in one call over the whole run. The mean-matched
        # generators force a single draw to the mean, so drawing gap by gap (as
        # the within-unit gaps once were) silently removed all jitter.
        gap: dict[tuple[int, int], float] = {}
        for key, where in slots.items():
            if isinstance(key, tuple) and key[0] == "pool":
                vals = specs[key].values
                drawn = _deal(len(vals), len(where), decks.setdefault(key, []), rng)
                values = np.asarray(vals)[drawn] + [shift[pos] for pos in where]
            else:
                values = specs[key].sample(len(where), rng, spec.tr)
            for pos, value in zip(where, values, strict=True):
                gap[pos] = float(value)
        if spare is not None:
            # One value from a mean-matched generator can fail its own constraints
            # and be padded with the mean, with a warning; for a gap that is only
            # used if a search moves this unit, the mean is fine.
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                gap[spare[1]] = float(spare[0].sample(1, spare_rng, spec.tr)[0])
        plans.append(
            RunPlan(
                [
                    (
                        ui,
                        [gap.get((k, j), 0.0) for j in range(len(spec.units[ui].items))],
                        item_order(spec.units[ui], rng),
                    )
                    for k, ui in enumerate(order)
                ]
            )
        )
    return counts, plans


def item_order(unit: Unit, rng: np.random.Generator) -> list[int]:
    """The order a unit's items are shown in: as written, or (``shuffle``) the
    non-gap items permuted among their own positions."""
    order = list(range(len(unit.items)))
    if unit.shuffle:
        movable = [j for j, it in enumerate(unit.items) if it.condition != NULL]
        for j, src in zip(movable, rng.permutation(movable), strict=True):
            order[j] = int(src)
    return order


def _deal(n_values: int, n: int, deck: list[int], rng: np.random.Generator) -> np.ndarray:
    """``n`` indices into a list of ``n_values``, each used equally often, shuffled.

    The whole multiples are this run's; the remainder comes off ``deck``, a
    shuffled stack of every index refilled as it runs out and kept across
    runs, so leftovers rotate: 10 gaps over (4,9,11) in each of 3 runs is
    3/3/3 per run plus one extra that is a different value each run, 10 of
    each overall -- drawing each run's remainder afresh could give 4 every time.
    """
    whole, rest = divmod(n, n_values)
    extra: list[int] = []
    while len(extra) < rest:
        if not deck:
            deck.extend(rng.permutation(n_values).tolist())
        # a refill can hold an index this run already took: skip past it
        pick = next((d for d in deck if d not in extra), None)
        if pick is None:
            deck.extend(rng.permutation(n_values).tolist())
            continue
        deck.remove(pick)
        extra.append(pick)
    out = np.concatenate([np.repeat(np.arange(n_values), whole), np.asarray(extra, dtype=int)])
    return rng.permutation(out).astype(int)


def assemble(
    spec: ExperimentSpec, counts: list[int], plans: list[RunPlan], seed: int = 0
) -> Realization:
    """The timeline of drawn (or searched) run plans: onsets, run lengths, drops."""
    conds = spec.conditions
    onsets: dict[str, list[list[float]]] = {c: [[] for _ in plans] for c in conds}
    run_lengths, run_durations = [], []
    n_dropped = 0
    run_s = spec.run_seconds(counts)
    for run, plan in enumerate(plans):
        # A fixed scan has a fixed number of volumes, and its last -post_fix
        # seconds are fixation: a unit whose events do not all end by then is
        # dropped whole (a cycle never loses its E2 and keeps its E1). Only
        # dropping events that *started* past the scan let jitter that is not
        # mean-matched (uniform) eat the final fixation -- 3.5 s of a 15 s one.
        limit = None if run_s is None else run_s - spec.post_fix
        t = _lock(spec.initial_fix, spec.initial_fix, spec) if spec.tr_lock else spec.initial_fix
        for k, (ui, gaps, shown) in enumerate(plan.entries):
            unit_events = []
            items = [spec.units[ui].items[i] for i in shown]
            for j, item in enumerate(items):
                if item.condition != NULL:
                    unit_events.append((item.condition, t, item.duration))
                final = k == len(plan.entries) - 1 and j == len(items) - 1
                nxt = t + item.duration + (0.0 if final else gaps[j])
                t = _lock(nxt, t + item.duration, spec) if spec.tr_lock and not final else nxt
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
        durations=spec.durations(),
        onsets=[[np.asarray(r) for r in onsets[c]] for c in conds],
        run_lengths=run_lengths,
        run_durations=run_durations,
        counts=counts,
        n_dropped=n_dropped,
    )


def _lock(t: float, earliest: float, spec: ExperimentSpec) -> float:
    """-tr_lock: the TR boundary nearest ``t``, but not before ``earliest``.

    Nearest, not next, so the mean gap stays what was asked for; never before
    the previous item has ended, so a snap cannot make items overlap.
    """
    snapped = round(t / spec.tr) * spec.tr
    if snapped < earliest - 1e-9:
        snapped = float(np.ceil(earliest / spec.tr - 1e-9)) * spec.tr
    return float(snapped)


def realize(spec: ExperimentSpec, seed: int) -> Realization:
    """Draw one realization: unit order and every jittered interval, per run."""
    if not spec.conditions:
        raise ValueError("the experiment has no non-null conditions")
    counts, plans = draw_plans(spec, seed)
    return assemble(spec, counts, plans, seed)


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
