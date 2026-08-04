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
    m = re.search(r"run-(\d+)", path.name, re.IGNORECASE)
    return int(m.group(1)) if m else 0


def sort_bids_event_files(paths: list[str | Path]) -> list[Path]:
    """
    Return *_events.tsv paths sorted by run number (ascending).

    Zero-padded and non-padded run numbers are handled identically:
    run-1, run-01, run-001 all sort to position 1.
    Files without a run entity (single-session data) are placed first.

    .. warning::
       Run number alone is **not** a unique key across sessions. Do not use this
       to align event files with input runs -- see :func:`parse_bids_events`,
       which pairs positionally instead.
    """
    return sorted((Path(p) for p in paths), key=_run_number)


# Lenient entity extraction: matches BIDS ``ses-01`` / ``run-001`` and also the
# hyphen-less ``run01`` that shows up in derivative filenames.
_ENTITY_RE = {
    "sub": re.compile(r"sub-?([A-Za-z0-9]+)", re.IGNORECASE),
    "ses": re.compile(r"ses-?([A-Za-z0-9]+)", re.IGNORECASE),
    "task": re.compile(r"task-?([A-Za-z0-9]+)", re.IGNORECASE),
    "run": re.compile(r"run-?(\d+)", re.IGNORECASE),
}


def parse_path_entities(path: str | Path) -> dict[str, str]:
    """Extract whatever sub/ses/task/run entities a filename carries.

    Deliberately lenient: derivative filenames routinely drop ``sub-`` and write
    ``run01`` rather than ``run-01``. Missing entities are simply absent from the
    result, so callers can compare only the keys both sides actually have.
    Numeric entities are normalised (``run-001`` and ``run01`` both give ``"1"``)
    so zero-padding differences never register as a mismatch.
    """
    name = Path(path).name
    out: dict[str, str] = {}
    for key, rx in _ENTITY_RE.items():
        m = rx.search(name)
        if m:
            val = m.group(1)
            out[key] = str(int(val)) if val.isdigit() else val.lower()
    return out


def verify_events_match_inputs(
    input_files: list[str | Path],
    event_files: list[str | Path],
) -> list[str]:
    """Check that positionally paired input/event files describe the same run.

    Returns a list of human-readable mismatch lines (empty when consistent).
    Only entities present on *both* sides of a pair are compared, so a data file
    without a ``task`` entity never conflicts with an events file that has one.
    Pairs where nothing overlaps are skipped rather than guessed at.

    This exists because the failure it catches is otherwise silent: mispaired
    timing yields a plausible-looking design that is simply wrong, and the only
    downstream symptom is an occasional all-zero column when the borrowed onsets
    happen to overrun a shorter run.
    """
    if len(input_files) != len(event_files):
        return [f"{len(input_files)} input runs but {len(event_files)} events files"]

    problems: list[str] = []
    for i, (inp, ev) in enumerate(zip(input_files, event_files, strict=True)):
        ent_i = parse_path_entities(inp)
        ent_e = parse_path_entities(ev)
        shared = [k for k in ("sub", "ses", "task", "run") if k in ent_i and k in ent_e]
        if not shared:
            continue
        diff = [k for k in shared if ent_i[k] != ent_e[k]]
        if diff:
            desc_i = "/".join(f"{k}-{ent_i[k]}" for k in shared)
            desc_e = "/".join(f"{k}-{ent_e[k]}" for k in shared)
            problems.append(
                f"  slot {i:>3}: input {desc_i}  <->  events {desc_e}"
                f"   MISMATCH on {','.join(diff)}"
            )
    return problems


def run_sort_key(path: str | Path) -> tuple:
    """Nested (session, task, run) sort key for putting runs in acquisition order.

    ``task`` sits between session and run deliberately. A session that contains two
    tasks (say mvpsA run-01..02 then mvpsB run-01..03) would otherwise tie on
    ``(ses, run)`` and interleave the two tasks; ordering by task first keeps each
    task's runs contiguous, which is the order they were actually acquired in.

    Each component is ``(rank, value)`` so numeric, string, and absent entities stay
    mutually comparable: numeric entities sort before string ones, and anything the
    filename does not carry sorts last while a stable sort preserves the caller's
    order among the ties.
    """
    ent = parse_path_entities(path)

    def part(key: str) -> tuple:
        if key not in ent:
            return (2, "")
        val = ent[key]
        return (0, f"{int(val):09d}") if val.isdigit() else (1, val)

    return (part("ses"), part("task"), part("run"))


def sort_runs_by_entities(
    input_files: list[str],
    event_files: list[str] | None = None,
) -> tuple[list[int], list[str], list[str] | None]:
    """Reorder runs into (session, task, run) order, keeping input/event pairs together.

    Sorting the two lists independently could silently re-pair them, so the order is
    derived from the *inputs* alone and the same permutation is applied to both. Any
    pairing check must therefore run before this, not after.

    Returns ``(order, sorted_inputs, sorted_events)`` where ``order[i]`` is the
    original index now at position *i*.
    """
    if event_files is not None and len(event_files) != len(input_files):
        raise ValueError(
            f"cannot reorder: {len(input_files)} input runs but {len(event_files)} "
            "events files. The two lists are paired by position, so they must be the "
            "same length before anything is sorted."
        )
    order = sorted(range(len(input_files)), key=lambda i: run_sort_key(input_files[i]))
    sorted_inputs = [input_files[i] for i in order]
    sorted_events = [event_files[i] for i in order] if event_files is not None else None
    return order, sorted_inputs, sorted_events


_NIFTI_EXTS = (".nii.zst", ".nii.gz", ".nii", ".HEAD", ".BRIK", ".BRIK.gz")


def _strip_image_ext(path: str | Path) -> str:
    name = Path(path).name
    for ext in _NIFTI_EXTS:
        if name.endswith(ext):
            return name[: -len(ext)]
    return Path(name).stem


def find_duplicate_inputs(input_files: list[str]) -> list[list[int]]:
    """Group input indices that name the same run twice.

    Matching is on the filename with its image extension stripped, so
    ``run01_final.nii.gz`` and ``run01_final.nii.zst`` -- the same data kept in two
    compression formats -- collide. A glob like ``*_final.nii.*`` picks up both, and
    the duplicate is otherwise invisible: the run is simply concatenated twice,
    double-counting its timepoints and its events.

    Returns one list of indices per duplicated stem (only groups of size > 1).
    """
    groups: dict[str, list[int]] = {}
    for i, p in enumerate(input_files):
        groups.setdefault(_strip_image_ext(p), []).append(i)
    return [idx for idx in groups.values() if len(idx) > 1]


def find_late_events(
    all_onsets: list[list[np.ndarray]],
    run_lengths_sec: list[float],
    condition_labels: list[str],
) -> list[dict]:
    """Locate events whose onset falls at or after the end of their run.

    Such an event contributes an all-zero column to a single-trial design (and
    silently nothing to a condition-level one), which makes the design rank
    deficient. It nearly always means the timing was paired with the wrong run
    rather than that the scanner genuinely stopped early.

    Returns one dict per offending run: ``run``, ``length_sec``, ``n_late``,
    ``last_onset``, and ``conditions``.
    """
    n_runs = len(run_lengths_sec)
    out: list[dict] = []
    for r in range(n_runs):
        late_n = 0
        last = 0.0
        conds: set[str] = set()
        for cidx, per_run in enumerate(all_onsets):
            if r >= len(per_run):
                continue
            ons = np.asarray(per_run[r], dtype=float)
            if ons.size == 0:
                continue
            late = ons[ons >= run_lengths_sec[r]]
            if late.size:
                late_n += int(late.size)
                last = max(last, float(late.max()))
                conds.add(condition_labels[cidx])
        if late_n:
            out.append(
                {
                    "run": r,
                    "length_sec": run_lengths_sec[r],
                    "n_late": late_n,
                    "last_onset": last,
                    "conditions": sorted(conds),
                }
            )
    return out


def drop_late_events(
    all_onsets: list[list[np.ndarray]],
    run_lengths_sec: list[float],
) -> list[list[np.ndarray]]:
    """Remove events starting at or after their run's end (see :func:`find_late_events`)."""
    return [
        [
            np.asarray(ons, dtype=float)[np.asarray(ons, dtype=float) < run_lengths_sec[r]]
            if r < len(run_lengths_sec)
            else np.asarray(ons, dtype=float)
            for r, ons in enumerate(per_run)
        ]
        for per_run in all_onsets
    ]


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
    n_runs: int | None = None,
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
        BIDS events TSV files, one per run, **in the same order as the input
        runs they describe**.  The list is used exactly as given: file *i*
        supplies the timing for run *i*.  Nothing is sorted here.
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
    n_runs : int or None, optional
        Number of runs to produce.  When a **single** events file is passed and
        ``n_runs > 1``, the parsed onsets are broadcast (replicated) across all
        ``n_runs`` runs.  This supports datasets where every run shares the same
        stimulus timing and therefore ships one BIDS ``*_events.tsv`` for the
        whole task (a valid BIDS pattern).  When more than one file is given,
        ``n_runs`` (if set) must equal the file count; otherwise it is ignored.

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

    # ── Pair positionally with the caller's input runs ───────────────────────
    # These files are zipped against -input by position, so reordering them here
    # silently reassigns every run's timing. This used to sort by run number alone,
    # which ignores the session entity: across sessions that groups every run-001
    # together and pairs session N's timing with session 1's run N. Multi-session
    # datasets came out scrambled with no error, only a downstream singular design
    # from onsets landing past the end of a shorter run. Caller order is the contract;
    # cli_utils.verify_events_match_inputs checks it when filenames carry entities.
    sorted_files = [Path(p) for p in event_files]
    n_files = len(sorted_files)

    if n_runs is not None and n_files > 1 and n_runs != n_files:
        raise ValueError(
            f"n_runs={n_runs} but {n_files} events files were given; broadcasting "
            "is only supported from a single shared events file."
        )

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
        [np.array([], dtype=np.float64) for _ in range(n_files)] for _ in range(n_conditions)
    ]
    # cond_dur_sets[cond_idx] collects all observed durations for that condition
    cond_dur_sets: list[set[float]] = [set() for _ in range(n_conditions)]

    for run_idx, events in enumerate(run_events):
        cond_onset_lists: dict[int, list[float]] = {i: [] for i in range(n_conditions)}
        for onset, dur, cond in events:
            cidx = cond_to_idx[cond]
            cond_onset_lists[cidx].append(onset)
            effective_dur = (
                round(float(dur), round_durations) if round_durations is not None else float(dur)
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

    # ── Broadcast a single shared events file across all runs ─────────────────
    # A dataset with identical timing every run may ship one *_events.tsv for the
    # whole task; replicate its onsets so downstream code sees one run each.
    if n_runs is not None and n_files == 1 and n_runs > 1:
        all_onsets = [[run_onsets[0].copy() for _ in range(n_runs)] for run_onsets in all_onsets]

    return all_onsets, durations, condition_labels
