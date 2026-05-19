"""
Events loading and group/selection logic for [[ffs_perm]].

Single-trial 4D inputs are assumed to be in **run order** — trials
concatenated run-by-run with row order matching each run's events TSV.
This matches what ``ffs_fitbasis -single-trials`` writes.

We don't reuse :func:`fastfuncstuff.design.bids_events.parse_bids_events`
because it collapses to per-condition onset arrays and loses the
row-by-row context (custom column values, run-of-origin) we need for
arbitrary column selection and run-block exchangeability.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fastfuncstuff.design.bids_events import sort_bids_event_files


@dataclass
class EventsTable:
    """Flat events table concatenated across runs, preserving row order.

    Attributes
    ----------
    rows : list[dict[str, str]]
        One dict per trial, with TSV column values as strings.
    run_idx : np.ndarray
        Length-N int array: which run each row came from.
    columns : list[str]
        Union of column names seen across files (preserves first-seen order).
    """
    rows: list[dict[str, str]]
    run_idx: np.ndarray
    columns: list[str]

    def __len__(self) -> int:
        return len(self.rows)

    def column(self, name: str) -> np.ndarray:
        """Return column values as a length-N string ndarray."""
        if name not in self.columns:
            raise KeyError(
                f"Column '{name}' not found in events.\n"
                f"  Available: {self.columns}"
            )
        return np.array([r.get(name, "") for r in self.rows], dtype=object)


def load_events(paths: list[str | Path], drop_na: bool = True) -> EventsTable:
    """Load and concatenate BIDS events TSVs preserving row order.

    Files are sorted by run number (matching the rest of fastfuncstuff).
    Rows where the ``trial_type`` column is empty or 'n/a' are dropped
    when ``drop_na`` is True.
    """
    sorted_paths = sort_bids_event_files(paths)
    rows: list[dict[str, str]] = []
    run_idx_list: list[int] = []
    columns_seen: list[str] = []
    columns_set: set[str] = set()

    for run_i, p in enumerate(sorted_paths):
        with open(p, newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            file_cols = list(reader.fieldnames or [])
            for c in file_cols:
                if c not in columns_set:
                    columns_set.add(c)
                    columns_seen.append(c)
            for row in reader:
                tt = (row.get("trial_type") or "").strip()
                if drop_na and (not tt or tt.lower() == "n/a"):
                    continue
                rows.append({k: (v if v is not None else "") for k, v in row.items()})
                run_idx_list.append(run_i)

    return EventsTable(
        rows=rows,
        run_idx=np.asarray(run_idx_list, dtype=np.int64),
        columns=columns_seen,
    )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

@dataclass
class OneSampleSelection:
    """Indices of trials to test, plus run-block labels for those trials."""
    indices: np.ndarray   # int64, into the full N
    blocks: np.ndarray    # int64, run index per selected trial
    label: str            # condition name used (for output naming)


@dataclass
class TwoSampleSelection:
    indices: np.ndarray   # int64, concatenated [A, B]
    group: np.ndarray     # int8, 1 for A, 0 for B
    blocks: np.ndarray    # int64, run index per selected trial
    label_a: str
    label_b: str


def select_one_sample(events: EventsTable, col: str, label: str) -> OneSampleSelection:
    """Pick trials where ``events[col] == label``."""
    vals = events.column(col)
    mask = (vals == label)
    idx = np.where(mask)[0]
    if idx.size < 3:
        raise ValueError(
            f"Only {idx.size} trial(s) matched {col}={label!r}; need ≥ 3 for a "
            "1-sample test."
        )
    return OneSampleSelection(
        indices=idx.astype(np.int64),
        blocks=events.run_idx[idx].astype(np.int64),
        label=label,
    )


def select_two_sample(
    events: EventsTable, col: str, label_a: str, label_b: str,
) -> TwoSampleSelection:
    """Pick two distinct labels in the same column for a 2-sample test."""
    if label_a == label_b:
        raise ValueError("Two-sample test requires two distinct labels.")
    vals = events.column(col)
    idx_a = np.where(vals == label_a)[0]
    idx_b = np.where(vals == label_b)[0]
    if idx_a.size < 2 or idx_b.size < 2:
        raise ValueError(
            f"Need ≥ 2 trials per group; got {idx_a.size} for {label_a!r} and "
            f"{idx_b.size} for {label_b!r}."
        )
    idx = np.concatenate([idx_a, idx_b]).astype(np.int64)
    group = np.concatenate([
        np.ones(idx_a.size, dtype=np.int8),
        np.zeros(idx_b.size, dtype=np.int8),
    ])
    return TwoSampleSelection(
        indices=idx,
        group=group,
        blocks=events.run_idx[idx].astype(np.int64),
        label_a=label_a,
        label_b=label_b,
    )


def select_one_vs_all(
    events: EventsTable, col: str, label: str, drop_values: tuple[str, ...] = (),
) -> TwoSampleSelection:
    """Group A = trials with ``col == label``; group B = all other trials.

    Trials whose column value is in ``drop_values`` are excluded entirely
    (useful for skipping ``fixation`` / ``baseline`` events).
    """
    vals = events.column(col)
    keep = ~np.isin(vals, list(drop_values)) if drop_values else np.ones(vals.size, dtype=bool)
    idx_a = np.where(keep & (vals == label))[0]
    idx_b = np.where(keep & (vals != label))[0]
    if idx_a.size < 2 or idx_b.size < 2:
        raise ValueError(
            f"-onevsall got {idx_a.size} {label!r} trials and {idx_b.size} "
            "others; need ≥ 2 of each."
        )
    idx = np.concatenate([idx_a, idx_b]).astype(np.int64)
    group = np.concatenate([
        np.ones(idx_a.size, dtype=np.int8),
        np.zeros(idx_b.size, dtype=np.int8),
    ])
    other_label = f"not-{label}"
    return TwoSampleSelection(
        indices=idx,
        group=group,
        blocks=events.run_idx[idx].astype(np.int64),
        label_a=label,
        label_b=other_label,
    )
