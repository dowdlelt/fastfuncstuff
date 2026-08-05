"""Companion trial table for single-trial beta outputs.

Every ``ffs_*`` tool that writes single-trial betas writes one volume per event,
ordered chronologically across the concatenated runs. That order is obvious to the
code and opaque to the user: volume 417 is "some trial", and recovering *which*
trial means re-deriving the sort by hand from the events files.

This module writes the answer next to the betas — one TSV row per beta volume, in
beta order, carrying every column of the originating ``*_events.tsv`` plus the
``sub``/``ses``/``task``/``run`` entities parsed from its filename. Downstream tools
(``ffs_varpart``, any analysis script) can then select trials by their real
metadata instead of by index arithmetic.

The trial order here is not an independent guess: it replays exactly the
construction and sort that :func:`fastfuncstuff.glm.ridge.create_single_trial_design`
performs (condition-major enumeration, then a stable sort on absolute onset time),
so the row at index *i* is the trial in volume *i*. Any change to that sort has to
be mirrored here — ``tests/test_trial_table.py`` pins the two together.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from fastfuncstuff.design.bids_events import parse_path_entities, read_tsv_rows

# Entities parsed from the events filename, in the order they become columns.
_ENTITY_COLUMNS = ("sub", "ses", "task", "run")

# Column suffixes create_single_trial_design gives the SPM basis functions.
_DEFAULT_BASIS_NAMES = {
    1: ["canonical"],
    2: ["canonical", "timederiv"],
    3: ["canonical", "timederiv", "dispderiv"],
}


def canonicalize_label(value: str) -> str:
    """Normalise whitespace in a free-text label: strip, then collapse runs to one space.

    ``"location shown"`` and ``"location  shown"`` are the same level typed twice, and a
    stray double space is invisible in a spreadsheet — treating them as two levels
    halves the trials per cell and silently changes the design. Whitespace is the only
    difference collapsed here; every other character difference stays meaningful.
    """
    return " ".join(str(value).split())


def _sanitize(value: str) -> str:
    """Collapse a canonicalised label into a token safe for filenames and identifiers.

    Spaces and punctuation in ``trial_type`` are routine ("face, inverted") and
    become a problem the moment a level name is used as a column name, a sub-brick
    label, or part of an output path. Distinct inputs are kept distinct by the
    caller, not here — this only maps characters.
    """
    out = []
    for ch in value:
        out.append(ch if (ch.isalnum() or ch in "-_.") else "_")
    token = "".join(out).strip("_")
    return token or "unlabeled"


def sanitize_levels(values: list[str]) -> tuple[list[str], dict[str, str]]:
    """Map free-text labels to identifiers, merging only whitespace-equal labels.

    Returns ``(sanitized_values, mapping)`` where *mapping* goes from each original
    label to its identifier. Two labels that differ only in whitespace share an
    identifier (see :func:`canonicalize_label`); two that genuinely differ but happen
    to sanitize to the same token (``"a b"`` and ``"a-b"`` both give ``a_b``) get a
    numeric suffix instead, since merging real levels would change the design rather
    than just its naming.
    """
    mapping: dict[str, str] = {}
    by_canonical: dict[str, str] = {}
    used: set[str] = set()
    for raw in values:
        if raw in mapping:
            continue
        canonical = canonicalize_label(raw)
        token = by_canonical.get(canonical)
        if token is None:
            token = _sanitize(canonical)
            if token in used:
                n = 2
                while f"{token}__{n}" in used:
                    n += 1
                token = f"{token}__{n}"
            used.add(token)
            by_canonical[canonical] = token
        mapping[raw] = token
    return [mapping[v] for v in values], mapping


def read_run_event_rows(
    event_files: list[str | Path],
    *,
    event_ignore: list[str] | None = None,
    event_cols: tuple[str, str, str] | None = None,
    n_runs: int | None = None,
) -> tuple[list[list[dict[str, Any]]], list[str]]:
    """Read every events TSV, applying the same filtering as ``parse_bids_events``.

    Returns ``(rows_by_run, fieldnames)``. A single shared events file is broadcast
    across ``n_runs`` runs, matching the BIDS pattern ``parse_bids_events`` accepts.
    """
    onset_col, duration_col, trial_type_col = event_cols or ("onset", "duration", "trial_type")
    ignore = set(event_ignore or [])

    rows_by_run: list[list[dict[str, Any]]] = []
    fieldnames: list[str] = []
    for path in event_files:
        rows, names = read_tsv_rows(path, onset_col, duration_col, trial_type_col)
        rows = [r for r in rows if r["_trial_type"] not in ignore]
        for r in rows:
            r["_source_file"] = str(path)
        rows_by_run.append(rows)
        for name in names:
            if name not in fieldnames:
                fieldnames.append(name)

    if n_runs is not None and len(rows_by_run) == 1 and n_runs > 1:
        rows_by_run = [[dict(r) for r in rows_by_run[0]] for _ in range(n_runs)]

    return rows_by_run, fieldnames


def order_trials(
    rows_by_run: list[list[dict[str, Any]]],
    run_starts: list[int],
    tr: float,
    *,
    run_lengths_sec: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Return event rows in single-trial beta order.

    Replays ``create_single_trial_design``: enumerate condition-major (conditions in
    sorted order, runs in input order, trials by ascending onset within a run), then
    stable-sort by absolute onset ``run_starts[run] * tr + onset``.

    ``run_lengths_sec`` applies the same late-event drop as
    :func:`fastfuncstuff.design.bids_events.drop_late_events`, so a caller that ran
    with ``-allow_late_events`` still gets a table that lines up with its betas.
    """
    if run_lengths_sec is not None:
        rows_by_run = [
            [r for r in rows if float(r["_onset"]) < run_lengths_sec[i]]
            if i < len(run_lengths_sec)
            else rows
            for i, rows in enumerate(rows_by_run)
        ]

    conditions = sorted({str(r["_trial_type"]) for rows in rows_by_run for r in rows})

    ordered: list[dict[str, Any]] = []
    for cond in conditions:
        for run_idx, rows in enumerate(rows_by_run):
            selected = [r for r in rows if str(r["_trial_type"]) == cond]
            selected.sort(key=lambda r: float(r["_onset"]))
            for r in selected:
                entry = dict(r)
                entry["_run_index"] = run_idx
                ordered.append(entry)

    ordered.sort(key=lambda e: run_starts[e["_run_index"]] * tr + float(e["_onset"]))
    return ordered


def build_trial_table(
    event_files: list[str | Path],
    run_starts: list[int],
    tr: float,
    *,
    event_ignore: list[str] | None = None,
    event_cols: tuple[str, str, str] | None = None,
    n_runs: int | None = None,
    run_lengths_sec: list[float] | None = None,
    n_basis: int = 1,
    basis_names: list[str] | None = None,
) -> tuple[list[str], list[dict[str, str]]]:
    """Build the companion table: one row per single-trial beta volume, in order.

    With ``n_basis > 1`` the design emits one column per (trial, basis function) —
    interleaved, trial-major — so each trial's row is repeated once per basis with a
    ``basis`` column, keeping the table one row per volume.
    """
    rows_by_run, fieldnames = read_run_event_rows(
        event_files, event_ignore=event_ignore, event_cols=event_cols, n_runs=n_runs
    )
    ordered = order_trials(rows_by_run, run_starts, tr, run_lengths_sec=run_lengths_sec)

    entity_by_file: dict[str, dict[str, str]] = {}
    for rows in rows_by_run:
        for r in rows:
            src = str(r["_source_file"])
            if src not in entity_by_file:
                # Unnormalised: the table should say ses-02, not ses-2, because the
                # value a user will filter on is the one they see in the filename.
                entity_by_file[src] = parse_path_entities(src, normalize=False)

    present_entities = [
        key for key in _ENTITY_COLUMNS if any(key in ent for ent in entity_by_file.values())
    ]

    # Every column of every source TSV is carried through verbatim, whatever it is
    # named -- the point of the table is that nothing about a trial gets lost. Added
    # columns therefore yield the name on a collision (``ffs_condition`` etc.) rather
    # than shadowing a real events column.
    taken = set(fieldnames)

    def _add(name: str) -> str:
        if name not in taken:
            taken.add(name)
            return name
        alt = f"ffs_{name}"
        n = 2
        while alt in taken:
            alt = f"ffs_{name}_{n}"
            n += 1
        taken.add(alt)
        return alt

    col_trial_index = _add("trial_index")
    col_run_index = _add("run_index")
    col_condition = _add("condition")

    # 'session'/'task'/'run' are the names downstream tools look for (ffs_varpart
    # reads run/session for fold locality), so expand the BIDS abbreviations.
    entity_column_name = {"sub": "subject", "ses": "session", "task": "task", "run": "run"}
    entity_headers = {k: _add(entity_column_name[k]) for k in present_entities}

    col_basis = _add("basis") if n_basis > 1 else None
    col_events_file = _add("events_file")
    col_events_row = _add("events_row")

    header = [col_trial_index, col_run_index, col_condition, *entity_headers.values()]
    if col_basis is not None:
        header.append(col_basis)
    header += [col_events_file, col_events_row, *fieldnames]

    basis_labels = basis_names or _DEFAULT_BASIS_NAMES.get(
        n_basis, [f"basis{i}" for i in range(n_basis)]
    )

    table: list[dict[str, str]] = []
    for trial_index, entry in enumerate(ordered):
        src = str(entry["_source_file"])
        ent = entity_by_file[src]
        # Original columns first, so the added ones can never be clobbered by a
        # same-named key that slipped through.
        base = {col: str(entry.get(col, "")) for col in fieldnames}
        base[col_run_index] = str(entry["_run_index"])
        base[col_condition] = str(entry["_trial_type"])
        base[col_events_file] = src
        base[col_events_row] = str(entry["_source_row"])
        for key, name in entity_headers.items():
            base[name] = ent.get(key, "n/a")

        for basis_idx in range(n_basis):
            row = dict(base)
            row[col_trial_index] = str(trial_index * n_basis + basis_idx)
            if col_basis is not None:
                row[col_basis] = basis_labels[basis_idx]
            table.append(row)

    return header, table


def write_single_trial_event_table(
    output_prefix: str,
    event_files: list[str | Path] | None,
    run_starts: list[int],
    tr: float,
    *,
    event_ignore: list[str] | None = None,
    event_cols: tuple[str, str, str] | None = None,
    n_runs: int | None = None,
    run_lengths_sec: list[float] | None = None,
    n_basis: int = 1,
    basis_names: list[str] | None = None,
    verbose: bool = True,
) -> str | None:
    """Write ``{output_prefix}_single_trial_events.tsv``; return the path or ``None``.

    Returns ``None`` (writing nothing) when there are no ``-events`` files, or when
    no filename carries a single BIDS entity — without ``ses``/``task``/``run`` the
    table cannot say which run a row came from beyond its index, which is exactly the
    information it exists to add.
    """
    if not event_files:
        return None

    if not any(parse_path_entities(f) for f in event_files):
        if verbose:
            print(
                "  NOTE: events filenames carry no BIDS entities (sub/ses/task/run); "
                "skipping the single-trial events table."
            )
        return None

    try:
        header, table = build_trial_table(
            event_files,
            run_starts,
            tr,
            event_ignore=event_ignore,
            event_cols=event_cols,
            n_runs=n_runs,
            run_lengths_sec=run_lengths_sec,
            n_basis=n_basis,
            basis_names=basis_names,
        )
    except (OSError, ValueError) as exc:
        # The betas are already written and correct at this point; a metadata sidecar
        # is not worth failing the run over.
        if verbose:
            print(f"  NOTE: could not write single-trial events table: {exc}")
        return None

    if not any(parse_path_entities(f) for f in event_files):
        if verbose:
            print(
                "  NOTE: events filenames carry no BIDS entities (sub/ses/task/run); "
                "skipping the single-trial events table."
            )
        return None

    from fastfuncstuff.cli_utils import parse_prefix

    stem = parse_prefix(str(output_prefix)).stem
    path = f"{stem}_single_trial_events.tsv"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table)

    if verbose:
        print(f"  Saved: {path} ({len(table)} rows x {len(header)} columns)")
    return path
