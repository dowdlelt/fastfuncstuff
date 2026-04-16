"""
BIDS events TSV parser for fMRI design matrix construction.

Parses BIDS *_events.tsv files (one per run) into the onset/duration/label
structures expected by the design matrix builders (same format as
parse_afni_timing_file + parse_durations would produce).

Public API
----------
parse_bids_events(event_files, event_ignore, event_cols)
    → (all_onsets, durations, condition_labels)

sort_bids_event_files(paths)
    → list[Path] sorted by run number
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Run-number sorting
# ---------------------------------------------------------------------------

def _run_number(path: Path) -> int:
    """
    Extract the numeric run index from a BIDS filename.

    Handles both zero-padded (run-01) and non-padded (run-1) forms.
    Files with no run entity sort to position 0 (single-run datasets).
    """
    m = re.search(r'run-(\d+)', path.name, re.IGNORECASE)
    return int(m.group(1)) if m else 0


def sort_bids_event_files(paths: list[str | Path]) -> list[Path]:
    """
    Return *_events.tsv paths sorted by run number (ascending).

    Zero-padded and non-padded run numbers are handled identically:
    run-1, run-01, run-001 all sort to position 1.
    Files without a run entity (single-session data) are placed first.
    """
    return sorted((Path(p) for p in paths), key=_run_number)


# ---------------------------------------------------------------------------
# Low-level TSV reader
# ---------------------------------------------------------------------------

def _read_tsv(
    path: Path,
    onset_col: str,
    duration_col: str,
    trial_type_col: str,
) -> list[tuple[float, float, str]]:
    """
    Read one BIDS events TSV.

    Returns a list of (onset_s, duration_s, trial_type) tuples.
    Rows where trial_type is empty or 'n/a' are skipped.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If a required column is missing or a row cannot be parsed.
    """
    if not path.exists():
        raise FileNotFoundError(f"Events file not found: {path}")

    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])

        for col, role in (
            (onset_col, "onset"),
            (duration_col, "duration"),
            (trial_type_col, "trial_type"),
        ):
            if col not in fieldnames:
                raise ValueError(
                    f"Column '{col}' (mapped as '{role}') not found in {path}.\n"
                    f"  Available columns: {fieldnames}\n"
                    f"  Use -event_cols to specify custom column names."
                )

        events: list[tuple[float, float, str]] = []
        for row_num, row in enumerate(reader, start=2):  # row 1 is header
            trial_type = str(row[trial_type_col]).strip()
            if not trial_type or trial_type.lower() == "n/a":
                continue
            try:
                onset = float(row[onset_col])
                duration = float(row[duration_col])
            except ValueError as exc:
                raise ValueError(
                    f"Cannot parse numeric values on row {row_num} of {path}: {exc}"
                ) from exc
            events.append((onset, duration, trial_type))

    return events


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_bids_events(
    event_files: list[str | Path],
    event_ignore: list[str] | None = None,
    event_cols: tuple[str, str, str] | None = None,
    round_durations: int | None = None,
) -> tuple[list[list[np.ndarray]], list[float], list[str]]:
    """
    Parse BIDS *_events.tsv files into onset/duration/label structures.

    The returned structures match the format produced by::

        all_onsets  = [parse_afni_timing_file(f) for f in onset_files]
        durations   = parse_durations(duration_args, n_conditions, labels)

    so the rest of the pipeline is unaffected.

    Parameters
    ----------
    event_files : list of str or Path
        BIDS events TSV files, one per run.  Files are sorted by run number
        before processing (handles zero-padded and non-padded run indices).
    event_ignore : list of str, optional
        trial_type values to exclude entirely (e.g. ``['fixation', 'null']``).
        Ignored conditions do not appear in the returned labels or onsets.
    event_cols : tuple of (str, str, str), optional
        Custom column names mapping to ``(onset, duration, trial_type)``.
    round_durations : int or None, optional
        If given, round every event's duration to this many decimal places
        **before** collecting per-condition unique duration sets.  This
        prevents trivially different floating-point values (e.g. ``3.03``
        vs ``3.0``) from being treated as distinct durations.
        ``0`` rounds to integers, ``1`` to tenths, etc.
        Default: ``('onset', 'duration', 'trial_type')``.

    Returns
    -------
    all_onsets : list[list[ndarray]]
        ``all_onsets[condition_idx][run_idx]`` — onset times in seconds,
        sorted ascending.  Shape mirrors ``parse_afni_timing_file`` output.
    durations : list[float]
        One duration per condition.  Derived from unique durations observed
        in the TSV data:

        - If all events for a condition share the same duration, that value is used.
        - If durations vary within a condition, the median is used and a warning
          is printed to stderr.

        If every condition has the same single duration the list is still
        per-condition (consistent with ``parse_durations`` behaviour).
    condition_labels : list[str]
        Sorted unique trial_type values after applying *event_ignore*.

    Raises
    ------
    FileNotFoundError
        If any events file does not exist.
    ValueError
        If required columns are missing, a row cannot be parsed, or no
        conditions remain after filtering.
    """
    onset_col, duration_col, trial_type_col = (
        event_cols if event_cols is not None else ("onset", "duration", "trial_type")
    )
    ignore_set: set[str] = set(event_ignore or [])

    # ── Sort files by run number ─────────────────────────────────────────────
    sorted_files = sort_bids_event_files(event_files)
    n_runs = len(sorted_files)

    # ── Read every TSV ───────────────────────────────────────────────────────
    # run_events[run_idx] = list of (onset, duration, trial_type)
    run_events: list[list[tuple[float, float, str]]] = []
    all_conditions: set[str] = set()

    for tsv_path in sorted_files:
        events = _read_tsv(Path(tsv_path), onset_col, duration_col, trial_type_col)
        # Drop ignored conditions
        events = [(on, dur, ct) for on, dur, ct in events if ct not in ignore_set]
        run_events.append(events)
        all_conditions.update(ct for _, _, ct in events)

    # ── Build sorted condition list ──────────────────────────────────────────
    condition_labels = sorted(all_conditions)
    n_conditions = len(condition_labels)

    if n_conditions == 0:
        raise ValueError(
            "No conditions remain after applying event_ignore filter.\n"
            f"  Ignored: {sorted(ignore_set)}"
        )

    cond_to_idx: dict[str, int] = {c: i for i, c in enumerate(condition_labels)}

    # ── Populate all_onsets and collect durations ────────────────────────────
    all_onsets: list[list[np.ndarray]] = [
        [np.array([], dtype=np.float64) for _ in range(n_runs)]
        for _ in range(n_conditions)
    ]
    # cond_dur_sets[cond_idx] collects all observed durations for that condition
    cond_dur_sets: list[set[float]] = [set() for _ in range(n_conditions)]

    for run_idx, events in enumerate(run_events):
        cond_onset_lists: dict[int, list[float]] = {i: [] for i in range(n_conditions)}
        for onset, dur, cond in events:
            cidx = cond_to_idx[cond]
            cond_onset_lists[cidx].append(onset)
            effective_dur = (
                round(float(dur), round_durations)
                if round_durations is not None
                else float(dur)
            )
            cond_dur_sets[cidx].add(effective_dur)

        for cidx in range(n_conditions):
            onsets_arr = np.array(sorted(cond_onset_lists[cidx]), dtype=np.float64)
            all_onsets[cidx][run_idx] = onsets_arr

    # ── Determine per-condition durations ────────────────────────────────────
    durations: list[float] = []
    for cidx, cond in enumerate(condition_labels):
        dset = cond_dur_sets[cidx]
        if not dset:
            # Condition had no events in any run — use 0 as a safe default
            durations.append(0.0)
        elif len(dset) == 1:
            durations.append(next(iter(dset)))
        else:
            # Multiple durations observed — use median with a warning
            sorted_durs = sorted(dset)
            median_dur = float(np.median(sorted_durs))
            print(
                f"WARNING: condition '{cond}' has multiple durations "
                f"{sorted_durs}; using median ({median_dur:.3f}s).",
                file=sys.stderr,
            )
            durations.append(median_dur)

    return all_onsets, durations, condition_labels
