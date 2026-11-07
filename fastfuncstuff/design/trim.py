"""Dropping leading/trailing TRs, and the event-timing shift that must follow.

Dropping the first N TRs of a run moves the run's time origin forward by
``N * TR`` seconds. Every onset in that run's timing file is expressed relative
to the *original* first TR, so the timing has to be shifted back by the same
amount or the whole design is misaligned by exactly the amount dropped -- a
silent, plausible-looking error. Dropping trailing TRs needs no shift (the
origin does not move) but can strand events past the new run end.

Data trimmed *upstream* -- before the file reached the fitting tool -- moves
the origin the same way, by however many seconds were cut. ``onset_offset``
carries that: positive means that many seconds were removed from the start, so
onsets move back by it. It is seconds rather than volumes because upstream
trims need not be whole output TRs: ``ffs_moco -vaso`` cuts raw volumes of two
interleaved contrasts and reports the result as ``StartTimeOffset``.

Shifting can legitimately produce a *negative* onset: an event that began
before the retained window but is still ongoing when it starts. That event is
real and must contribute a truncated boxcar, not be discarded, so the onset is
kept negative and clamped at paint time by
``design/builder.py:create_onset_matrix_microtime``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TrimSpec:
    """How many TRs to drop from each end of every run.

    ``drop_first``/``drop_last`` are counts of TRs (not seconds): data can only
    be dropped in whole volumes. The seconds figure users care about is derived
    as :attr:`shift_sec`.
    """

    drop_first: int = 0
    drop_last: int = 0
    tr: float | None = None
    # Seconds already removed from the start of each run upstream (one value for
    # every run, or one per run). Moves the timing only; the data are untouched.
    onset_offset: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.drop_first < 0 or self.drop_last < 0:
            raise ValueError(
                f"-drop_first/-drop_last must be >= 0 (got {self.drop_first}/{self.drop_last})"
            )

    @property
    def active(self) -> bool:
        """Volumes are dropped from the data (the loader and design length change)."""
        return self.drop_first > 0 or self.drop_last > 0

    @property
    def shifts_timing(self) -> bool:
        """Onsets move: leading volumes dropped here, or an upstream offset."""
        return self.drop_first > 0 or any(o != 0 for o in self.onset_offset)

    def offset_for_run(self, run: int) -> float:
        if not self.onset_offset:
            return 0.0
        if len(self.onset_offset) == 1:
            return float(self.onset_offset[0])
        if run >= len(self.onset_offset):
            raise ValueError(
                f"-onset_offset has {len(self.onset_offset)} values; run {run + 1} has none"
            )
        return float(self.onset_offset[run])

    def run_shift(self, run: int) -> float:
        """Seconds run *run*'s onsets move back: dropped TRs plus the upstream offset."""
        return self.shift_sec + self.offset_for_run(run)

    @property
    def total(self) -> int:
        return self.drop_first + self.drop_last

    @property
    def shift_sec(self) -> float:
        """Seconds every onset must move back. Zero unless leading TRs are dropped."""
        if self.drop_first == 0:
            return 0.0
        if self.tr is None:
            raise ValueError("TrimSpec.shift_sec needs a TR; construct with tr=...")
        return self.drop_first * self.tr

    def with_tr(self, tr: float) -> TrimSpec:
        """Return a copy carrying *tr* (CLIs learn the TR only after loading)."""
        return TrimSpec(
            drop_first=self.drop_first,
            drop_last=self.drop_last,
            tr=tr,
            onset_offset=self.onset_offset,
        )

    def trimmed_length(self, n_tr: int) -> int:
        """Length of a run of *n_tr* TRs after trimming. Raises if nothing survives."""
        out = n_tr - self.total
        if out <= 0:
            raise ValueError(
                f"-drop_first {self.drop_first} + -drop_last {self.drop_last} removes all "
                f"{n_tr} TRs of a run; nothing would be left to fit."
            )
        return out

    def describe(self) -> str:
        parts = []
        if self.drop_first:
            parts.append(f"first {self.drop_first} TR{'s' if self.drop_first != 1 else ''}")
        if self.drop_last:
            parts.append(f"last {self.drop_last} TR{'s' if self.drop_last != 1 else ''}")
        return " and ".join(parts) if parts else "nothing"


@dataclass
class TrimTimingReport:
    """What the onset shift actually did, for printing and for tests."""

    shift_sec: float  # run 1's shift; the others are in run_shifts
    n_shifted: int
    n_straddling: int  # now start before the scan, still overlap it (kept, truncated)
    n_before: int  # ended before the retained window began (dropped)
    n_after: int  # start at/after the new run end (dropped)
    straddle_conditions: list[str]
    dropped_conditions: list[str]
    run_shifts: tuple[float, ...] = ()

    def lines(self) -> list[str]:
        if len(set(self.run_shifts)) > 1:
            by = ", ".join(f"{s:.3f}" for s in self.run_shifts)
            how = f"per run ({by})s"
        else:
            how = f"{self.shift_sec:.3f}s"
        out = [
            f"  Timing shifted back by {how} "
            f"({self.n_shifted} event{'s' if self.n_shifted != 1 else ''} across all runs)"
        ]
        if self.n_straddling:
            out.append(
                f"    ⚠️  {self.n_straddling} event(s) now begin before the scan starts "
                f"({', '.join(self.straddle_conditions)}); kept with a truncated response"
            )
        if self.n_before:
            out.append(
                f"    {self.n_before} event(s) ended before the retained window and were dropped"
            )
        if self.n_after:
            out.append(f"    {self.n_after} event(s) fell past the new run end and were dropped")
        if self.dropped_conditions:
            out.append(f"    dropped events came from: {', '.join(self.dropped_conditions)}")
        return out


def shift_onsets_for_trim(
    all_onsets: list[list[np.ndarray]],
    durations: list[float],
    condition_labels: list[str],
    trimmed_run_lengths_sec: list[float],
    spec: TrimSpec,
) -> tuple[list[list[np.ndarray]], TrimTimingReport]:
    """Shift every onset back by ``spec.run_shift(run)`` and drop what no longer fits.

    *trimmed_run_lengths_sec* is each run's length **after** trimming, so the
    late-event test uses the window the data actually covers.

    An event is kept when it overlaps the retained window at all. That
    deliberately includes events with a negative shifted onset (they began
    before the first retained TR but are still ongoing during it) -- discarding
    those would drop real signal and leave its variance in the residual. Events
    are only dropped when they end at or before the window start, or begin at or
    after its end.
    """
    n_runs = max((len(per_run) for per_run in all_onsets), default=0)
    run_shifts = tuple(spec.run_shift(r) for r in range(n_runs))
    n_shifted = 0
    n_straddling = 0
    n_before = 0
    n_after = 0
    straddle: set[str] = set()
    dropped: set[str] = set()

    out: list[list[np.ndarray]] = []
    for cidx, per_run in enumerate(all_onsets):
        # A zero-duration (impulse) event still occupies one microtime bin, but it
        # has no extent to straddle the window edge with: treat it as a point.
        dur = float(durations[cidx]) if cidx < len(durations) else 0.0
        label = condition_labels[cidx] if cidx < len(condition_labels) else f"cond{cidx}"

        new_per_run: list[np.ndarray] = []
        for r, ons in enumerate(per_run):
            arr = np.asarray(ons, dtype=float)
            if arr.size == 0:
                new_per_run.append(arr)
                continue
            n_shifted += int(arr.size)
            shifted = arr - run_shifts[r]

            run_end = (
                trimmed_run_lengths_sec[r] if r < len(trimmed_run_lengths_sec) else float("inf")
            )

            # Overlap test: [onset, onset+dur) must intersect [0, run_end).
            ends_before = (shifted + dur) <= 0.0
            starts_after = shifted >= run_end
            keep = ~(ends_before | starts_after)

            n_before += int(ends_before.sum())
            n_after += int(starts_after.sum())
            if ends_before.any() or starts_after.any():
                dropped.add(label)

            kept = shifted[keep]
            n_neg = int((kept < 0).sum())
            if n_neg:
                n_straddling += n_neg
                straddle.add(label)

            new_per_run.append(kept)
        out.append(new_per_run)

    report = TrimTimingReport(
        shift_sec=run_shifts[0] if run_shifts else spec.shift_sec,
        n_shifted=n_shifted,
        n_straddling=n_straddling,
        n_before=n_before,
        n_after=n_after,
        straddle_conditions=sorted(straddle),
        dropped_conditions=sorted(dropped),
        run_shifts=run_shifts,
    )
    return out, report


def trim_run_series(
    arr: np.ndarray,
    expected_trimmed: int,
    spec: TrimSpec,
    path: str | Path | None = None,
) -> np.ndarray:
    """Align a per-run regressor/censor series with trimmed data.

    Accepts the series either already trimmed (``expected_trimmed`` rows, left
    alone) or at the run's original length (trimmed here). Motion parameters and
    censor files are almost always estimated on the untrimmed run, so silently
    requiring the caller to pre-trim them would be a footgun; requiring the
    original length would break anyone who already trimmed. Anything else is an
    error, because guessing which end the extra rows belong to is exactly the
    kind of misalignment this module exists to prevent.
    """
    n = int(arr.shape[0])
    if n == expected_trimmed:
        return arr
    if n == expected_trimmed + spec.total:
        end = n - spec.drop_last
        return arr[spec.drop_first : end]
    where = f"{path}: " if path is not None else ""
    raise ValueError(
        f"{where}has {n} rows, but the run has {expected_trimmed} timepoints after "
        f"dropping {spec.describe()} (an untrimmed file would have "
        f"{expected_trimmed + spec.total} rows)."
    )
