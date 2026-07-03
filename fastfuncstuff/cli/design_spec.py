#!/usr/bin/env python
"""
ffs_design_spec — front-end to ffs_build_design that drives the whole design
(events, HRF-per-task, contrasts) from a single TOML file.

Subcommands
-----------
stub     Scan events.tsv files + NIfTI headers; write a populated design.spec
         skeleton ready to edit.
compile  Read a design.spec and emit an AFNI-compatible .xmat.1D.

The TOML schema lives in :mod:`fastfuncstuff.design.spec`. ``hrfopt:<lib>``
event models are deliberately rejected at compile time — they require
per-voxel design selection, which only ``ffs_reml -spec`` can consume.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

from fastfuncstuff.design.builder import (
    build_design_matrix,
    good_list_from_censor,
    write_afni_xmat,
)
from fastfuncstuff.design.spec import (
    EventSpec,
    MetaSpec,
    NuisanceSpec,
    RunSpec,
    Spec,
    load_spec,
    resolve_contrast,
    write_spec,
)
from fastfuncstuff.io.afni import get_tr_from_file, load_nifti

# ---------------------------------------------------------------------------
# Argparse plumbing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ffs_design_spec",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # stub
    p_stub = sub.add_parser(
        "stub",
        help="Scan events + BOLD headers and write a design.spec skeleton.",
    )
    p_stub.add_argument(
        "-input",
        nargs="+",
        required=True,
        metavar="FILE",
        dest="input",
        help="BOLD images, one per run (header read only).",
    )
    p_stub.add_argument(
        "-events",
        nargs="+",
        required=True,
        metavar="TSV",
        help="BIDS events.tsv, one per run, same order as -input.",
    )
    p_stub.add_argument("-out", required=True, metavar="SPEC", help="Output design.spec path.")
    p_stub.add_argument(
        "-TR", type=float, default=None, help="Override TR (otherwise read from input headers)."
    )
    p_stub.add_argument(
        "-event-cols",
        nargs=3,
        metavar=("ONSET", "DURATION", "TRIAL_TYPE"),
        help="Non-BIDS column names for the events.tsv files.",
    )
    p_stub.add_argument(
        "-drop-trial-types",
        nargs="*",
        default=["rest", "Rest", "REST", "baseline"],
        help="Trial types to exclude from the spec.",
    )
    p_stub.add_argument(
        "-default-hrf",
        default="SPMG1",
        help="HRF model used for every event in the stub. The "
        "duration is taken from the [[events]] 'duration' "
        "field, so the bare model name is what you want here.",
    )
    # Nuisance regressors — three input modes, matching ffs_reml. See the
    # generated [[nuisance]] section header in design.toml for the full
    # padding-semantics writeup.
    p_stub.add_argument(
        "-ortvec",
        action="append",
        nargs=2,
        metavar=("FILE", "LABEL"),
        help="Full-length nuisance (already concatenated across "
        "all runs, used as-is). Repeatable. Use this for "
        "AFNI mot_demean.r0N.1D files — each is already "
        "full-length and block-diagonal.",
    )
    p_stub.add_argument(
        "-ortvec_run",
        "-ortvec-run",
        action="append",
        nargs=3,
        metavar=("FILE", "LABEL", "RUN"),
        dest="ortvec_run",
        help="Per-run nuisance (file is one run long, "
        "zero-padded into the full grid). RUN is 1-indexed. "
        "Repeatable.",
    )
    p_stub.add_argument(
        "-ortvec_glob",
        "-ortvec-glob",
        action="append",
        nargs=2,
        metavar=("PATTERN", "LABEL"),
        dest="ortvec_glob",
        help="Glob matching per-run nuisance files; run index "
        "inferred from filename. Stored in the spec verbatim, "
        "re-resolved at compile time. Repeatable.",
    )
    p_stub.add_argument(
        "-ortvec_concat",
        "-ortvec-concat",
        action="append",
        nargs=2,
        metavar=("PATTERN", "LABEL"),
        dest="ortvec_concat",
        help="Glob matching N already-full-length per-run files "
        "(e.g. AFNI mot_demean.r0N.1D — each spans every run "
        "with zeros outside its own). Expanded into N "
        "scope='full' entries labelled LABEL01, LABEL02, … "
        "(width auto-padded from n_runs). Repeatable.",
    )
    p_stub.add_argument(
        "-overwrite",
        action="store_true",
        help="Overwrite -out if it already exists (no interactive prompt).",
    )

    # compile
    p_comp = sub.add_parser(
        "compile",
        help="Read a design.spec and emit an AFNI .xmat.1D.",
    )
    p_comp.add_argument(
        "-spec",
        required=True,
        metavar="SPEC",
        help="Input design TOML file. Extension auto-appended (.toml) if missing.",
    )
    p_comp.add_argument("-xmat", required=True, metavar="FILE", help="Output .xmat.1D path.")
    p_comp.add_argument(
        "-overwrite",
        action="store_true",
        help="Overwrite -xmat if it already exists (no interactive prompt).",
    )
    p_comp.add_argument("-verb", type=int, default=0, help="Verbosity level (0=quiet, 1=summary).")

    return parser


# ---------------------------------------------------------------------------
# stub
# ---------------------------------------------------------------------------


def _bold_header(path: Path) -> tuple[int, float]:
    """Return (n_timepoints, tr) by reading only the NIfTI header."""
    img = load_nifti(path)
    n_tp = img.shape[3] if len(img.shape) > 3 else 1
    tr = get_tr_from_file(str(path))
    return int(n_tp), float(tr)


def _auto_polort(durations_sec: list[float]) -> int | list[int]:
    """AFNI's rule: 1 + floor(duration_sec / 150). Collapse to scalar if
    every run agrees."""
    per_run = [int(1 + math.floor(d / 150.0)) for d in durations_sec]
    return per_run[0] if len(set(per_run)) == 1 else per_run


def _scan_trial_types(
    events_files: list[Path],
    cols: tuple[str, str, str],
    drop: set[str],
) -> tuple[list[str], dict[str, list[float]]]:
    """Read every events TSV and return:

    - sorted unique trial_types (after applying *drop*)
    - per-trial-type list of all observed event durations (across all runs).
    """
    onset_col, dur_col, tt_col = cols
    del onset_col
    durations: dict[str, list[float]] = {}
    for path in events_files:
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                tt = str(row.get(tt_col, "")).strip()
                if not tt or tt.lower() == "n/a" or tt in drop:
                    continue
                try:
                    dur = float(row[dur_col])
                except (KeyError, ValueError):
                    dur = float("nan")
                durations.setdefault(tt, []).append(dur)
    return sorted(durations.keys()), durations


def _duration_stats_comment(durations: list[float]) -> str:
    """One-line summary of a trial_type's durations for stub annotation.

    Purely informational — comments cannot influence compile output.
    """
    valid = [d for d in durations if d == d]  # drop NaN
    if not valid:
        return "no parseable durations"
    n = len(valid)
    lo, hi = min(valid), max(valid)
    mean = sum(valid) / n
    sorted_vals = sorted(valid)
    median = sorted_vals[n // 2] if n % 2 else 0.5 * (sorted_vals[n // 2 - 1] + sorted_vals[n // 2])
    uniq = sorted(set(round(d, 4) for d in valid))
    uniq_str = ", ".join(f"{u:g}" for u in uniq) if len(uniq) <= 5 else f"{len(uniq)} unique values"
    return (
        f"n={n}, range=[{lo:g}, {hi:g}], "
        f"mean={mean:.3g}, median={median:.3g}, unique={{{uniq_str}}}"
    )


def _build_nuisance_from_cli_args(
    args: argparse.Namespace,
    n_runs: int,
) -> list[NuisanceSpec]:
    """Turn -ortvec / -ortvec_run / -ortvec_glob / -ortvec_concat into
    NuisanceSpec rows. All four flags mirror ffs_reml's add_ortvec_arguments
    so users learn one API. The fourth (concat) is a convenience for
    AFNI-style already-padded per-run files (each is full length and
    block-diagonal — the demean output of afni_proc.py)."""
    from fastfuncstuff.cli_utils import expand_ortvec_concat

    out: list[NuisanceSpec] = []

    for file, label in getattr(args, "ortvec", None) or []:
        out.append(NuisanceSpec(file=str(file), label=label, scope="full"))

    for file, label, run in getattr(args, "ortvec_run", None) or []:
        out.append(NuisanceSpec(file=str(file), label=label, scope=f"run:{int(run)}"))

    for pattern, label in getattr(args, "ortvec_glob", None) or []:
        out.append(
            NuisanceSpec(
                file=None,
                label=label,
                scope="glob",
                pattern=str(pattern),
            )
        )

    for pattern, label in getattr(args, "ortvec_concat", None) or []:
        for path, suffixed_label in expand_ortvec_concat(pattern, label, n_runs):
            out.append(
                NuisanceSpec(
                    file=str(path),
                    label=suffixed_label,
                    scope="full",
                )
            )

    return out


def _resolve_spec_path(arg: str) -> Path:
    """Accept -spec with or without a .toml extension. Errors clearly if
    neither the given path nor its .toml-suffixed sibling exists."""
    p = Path(arg)
    if p.exists():
        return p
    if not p.suffix:
        p_toml = p.with_suffix(".toml")
        if p_toml.exists():
            return p_toml
    raise FileNotFoundError(
        f"Spec file not found: {arg}"
        + ("" if p.suffix else f" (also tried {p.with_suffix('.toml')})")
    )


def _confirm_overwrite(path: Path, force: bool, kind: str) -> bool:
    """Return True if we may write *path*. Asks y/n on a TTY when the file
    already exists and *force* is False; refuses non-interactively."""
    if not path.exists():
        return True
    if force:
        return True
    if not sys.stdin.isatty():
        print(
            f"ERROR: {kind} already exists: {path}\n"
            "Re-run with -overwrite, or remove the file first.",
            file=sys.stderr,
        )
        return False
    answer = input(f"{kind} {path} already exists. Overwrite? [y/N] ").strip().lower()
    return answer in ("y", "yes")


def _do_stub(args: argparse.Namespace) -> int:
    bold_paths = [Path(p) for p in args.input]
    events_paths = [Path(p) for p in args.events]
    if len(bold_paths) != len(events_paths):
        raise ValueError(
            f"-input and -events must have the same length "
            f"(got {len(bold_paths)} input vs {len(events_paths)} events)."
        )
    n_runs = len(bold_paths)

    # Header scan.
    n_tp_per_run: list[int] = []
    trs: list[float] = []
    for p in bold_paths:
        n_tp, tr = _bold_header(p)
        n_tp_per_run.append(n_tp)
        trs.append(tr)
    if args.TR is not None:
        tr_final = float(args.TR)
    elif len(set(trs)) > 1:
        raise ValueError(f"BOLD files report inconsistent TRs: {trs}. Use -TR.")
    else:
        tr_final = trs[0]

    cols = tuple(args.event_cols) if args.event_cols else ("onset", "duration", "trial_type")
    drop_set = set(args.drop_trial_types)
    trial_types, durations_per_tt = _scan_trial_types(events_paths, cols, drop_set)
    if not trial_types:
        raise ValueError("No trial_types survived the drop filter — nothing to model.")

    polort = _auto_polort([n * tr_final for n in n_tp_per_run])

    meta = MetaSpec(
        runs=[
            RunSpec(bold=str(b), events=str(e))
            for b, e in zip(bold_paths, events_paths, strict=True)
        ],
        tr=tr_final,
        n_timepoints_per_run=n_tp_per_run,
        polort=polort,
        drop_trial_types=list(args.drop_trial_types),
    )
    # Don't overwrite events_columns with defaults; only set if user customised.
    if args.event_cols:
        from fastfuncstuff.design.spec import EventsColumns

        meta.events_columns = EventsColumns(onset=cols[0], duration=cols[1], trial_type=cols[2])

    events = [
        EventSpec(trial_type=tt, duration="from_events", hrf=args.default_hrf, mode="condition")
        for tt in trial_types
    ]

    nuisance = _build_nuisance_from_cli_args(args, n_runs)

    # Stats per trial_type → informational comments above each [[events]] block.
    event_notes = {
        tt: (
            "observed durations (informational, no effect on compile): "
            + _duration_stats_comment(durations_per_tt[tt])
        )
        for tt in trial_types
    }

    spec = Spec(meta=meta, events=events, nuisance=nuisance, contrasts=[])

    out_path = Path(args.out)
    if not out_path.suffix:
        out_path = out_path.with_suffix(".toml")

    if not _confirm_overwrite(out_path, args.overwrite, "Spec"):
        return 1

    write_spec(
        spec,
        out_path,
        header_comment=(
            "Stub generated by ffs_design_spec. Edit before compiling.\n"
            "  - Adjust [[events]] hrf / mode / round_onset per task.\n"
            "  - Add [[nuisance]] entries for motion etc.\n"
            "  - Add [[contrasts]] (sym, balance). Globs and ALLOTHERS supported.\n"
        ),
        event_notes=event_notes,
        include_contrast_examples=True,
    )
    print(f"Wrote stub: {out_path}", file=sys.stderr)
    print(f"  {len(meta.runs)} runs, TR={tr_final}, polort={polort}", file=sys.stderr)
    print(f"  {len(events)} trial types: {trial_types}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# compile
# ---------------------------------------------------------------------------


def _round_value(value: float, mode: float | str | None, tr: float) -> float:
    if mode is None:
        return value
    if isinstance(mode, str):
        if mode.upper() == "TR":
            return round(value / tr) * tr
        raise ValueError(f"Unknown rounding mode: {mode!r}")
    # Numeric: decimals (0 = integers, 1 = tenths, …) following AFNI convention.
    decimals = int(mode)
    return round(value, decimals)


def _read_events_for_condition(
    events_path: Path,
    trial_type: str,
    cols: tuple[str, str, str],
    round_onset: float | str | None,
    round_duration: float | None,
    tr: float,
) -> list[tuple[float, float]]:
    """Return (onset, duration) pairs for one trial_type in one events file,
    with rounding applied per the event spec."""
    onset_col, dur_col, tt_col = cols
    out: list[tuple[float, float]] = []
    with open(events_path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            if str(row.get(tt_col, "")).strip() != trial_type:
                continue
            try:
                onset = float(row[onset_col])
                duration = float(row[dur_col])
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Could not parse onset/duration in {events_path}: {exc}") from exc
            onset = _round_value(onset, round_onset, tr)
            duration = _round_value(duration, round_duration, tr)
            out.append((onset, duration))
    return out


def _write_afni_timing(
    path: Path,
    per_run_onsets: list[list[float]],
) -> None:
    """Write a `-stim_times`-format file: one row per run, space-separated
    onsets, ``*`` for empty runs."""
    with open(path, "w") as fh:
        for run_onsets in per_run_onsets:
            if not run_onsets:
                fh.write("*\n")
            else:
                fh.write(" ".join(f"{o:.4f}" for o in sorted(run_onsets)) + "\n")


def _expand_event_to_stims(
    event: EventSpec,
    events_files: list[Path],
    cols: tuple[str, str, str],
    tr: float,
    tmpdir: Path,
) -> list[tuple[Path, str, str, bool]]:
    """
    Expand one EventSpec into one or more (timing_file, label, hrf, im_flag)
    tuples. ``from_events`` with multiple distinct rounded durations splits
    into per-duration regressors named ``{trial_type}_dur{d}``.
    """
    if event.hrf.startswith("hrfopt:"):
        raise ValueError(
            f"Event '{event.trial_type}' uses {event.hrf!r}. hrfopt models "
            "require ffs_reml -spec; ffs_design_spec compile cannot emit a "
            "single xmat for them."
        )

    # Per-run (onset, duration) lists.
    per_run = [
        _read_events_for_condition(
            ef,
            event.trial_type,
            cols,
            event.round_onset,
            event.round_duration,
            tr,
        )
        for ef in events_files
    ]

    # Resolve duration: explicit number vs from_events.
    if event.duration == "from_events":
        unique_durs = sorted({d for run in per_run for _, d in run})
        if not unique_durs:
            raise ValueError(f"Event '{event.trial_type}': no events found in any run.")
        if event.mode == "im" or len(unique_durs) == 1:
            # Single stim, single duration (median if im-mode and they differ).
            dur_value = unique_durs[0] if len(unique_durs) == 1 else float(np.median(unique_durs))
            timing_path = tmpdir / f"{event.trial_type}.1D"
            _write_afni_timing(timing_path, [[o for o, _ in run] for run in per_run])
            hrf = _inject_duration(event.hrf, dur_value)
            return [(timing_path, event.trial_type, hrf, event.mode == "im")]

        # condition mode + multiple durations -> split per duration.
        out: list[tuple[Path, str, str, bool]] = []
        for d in unique_durs:
            per_run_onsets = [[o for o, dd in run if dd == d] for run in per_run]
            label = f"{event.trial_type}_dur{_fmt_dur(d)}"
            timing_path = tmpdir / f"{label}.1D"
            _write_afni_timing(timing_path, per_run_onsets)
            hrf = _inject_duration(event.hrf, d)
            out.append((timing_path, label, hrf, False))
        return out

    # Explicit numeric duration.
    dur_value = float(event.duration)
    timing_path = tmpdir / f"{event.trial_type}.1D"
    _write_afni_timing(timing_path, [[o for o, _ in run] for run in per_run])
    hrf = _inject_duration(event.hrf, dur_value)
    return [(timing_path, event.trial_type, hrf, event.mode == "im")]


def _fmt_dur(d: float) -> str:
    """Render a duration for use inside a label: ``2.0 -> '2'``, ``2.5 -> '2p5'``."""
    if float(d).is_integer():
        return f"{int(d)}"
    return f"{d:g}".replace(".", "p")


def _inject_duration(hrf: str, duration: float) -> str:
    """Inject the resolved event duration into the HRF model string.

    The bare model form is the preferred input — ``hrf = "SPMG1"`` plus
    ``duration = <number>`` (or ``"from_events"``). When the user writes an
    explicit argument list (``SPMG1(3)``, ``BLOCK(20,1)``, ``TENT(0,20,11)``),
    that is treated as a *user override* and passed through unchanged.
    """
    if "(" in hrf:
        return hrf  # explicit form — user override wins.
    head = hrf
    if head == "BLOCK":
        return f"BLOCK({duration:g},1)"
    if head.startswith("SPMG"):
        return f"{head}({duration:g})"
    return hrf  # TENT/FIR / unknown — left alone (need explicit args anyway).


def _resolved_row_to_sym(
    row: dict[str, tuple[float, tuple[int, int] | None]],
) -> str:
    """Re-emit a resolved row dict as a SYM: string so it round-trips
    through write_afni_xmat without losing sub-ranges."""
    tokens: list[str] = []
    for label, (weight, rng) in row.items():
        sign = "+" if weight >= 0 else "-"
        mag = abs(weight)
        mag_str = f"{int(mag)}" if float(mag).is_integer() else f"{mag:g}"
        suffix = f"[{rng[0]}..{rng[1]}]" if rng is not None else ""
        tokens.append(f"{sign}{mag_str}*{label}{suffix}")
    return "SYM: " + " ".join(tokens)


def _maybe_demean_to_tempdir(path: Path, tmpdir: Path) -> Path:
    """Load a .1D file, subtract the per-column mean, write to *tmpdir*.
    Returns the new path. The original file is untouched."""
    arr = np.loadtxt(path, ndmin=2)
    arr = arr - arr.mean(axis=0, keepdims=True)
    out = tmpdir / f"{path.stem}_demean.1D"
    np.savetxt(out, arr, fmt="%.10g")
    return out


def _resolve_nuisance_for_compile(
    nuisance: list[NuisanceSpec],
    n_runs: int,
    n_timepoints_per_run: list[int],
    tmpdir: Path | None = None,
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str, int]]]:
    """Return (ortvec_files, padortvec_files) for build_design_matrix.

    - ``scope == "full"`` → ortvec_files (build_design_matrix loads as-is).
    - ``scope == "run:N"`` → padortvec_files (zero-padded into run N).
    - ``scope == "glob"`` → expanded into N padortvec_files entries using
      ffs_reml's filename → run-index inference. Files are validated to be
      one-run-length up front so compile fails fast with a clear message
      instead of crashing inside the builder.
    """
    import glob as glob_module

    from fastfuncstuff.cli_utils import _infer_run_indices_from_filenames

    ortvec_files: list[tuple[Path, str]] = []
    padortvec_files: list[tuple[Path, str, int]] = []

    def _maybe_rescale(src: Path, n_spec: NuisanceSpec) -> Path:
        if n_spec.rescale == "demean":
            if tmpdir is None:
                raise RuntimeError(
                    "rescale='demean' requires a tmpdir to write the preprocessed file to"
                )
            return _maybe_demean_to_tempdir(src, tmpdir)
        return src

    for n in nuisance:
        if n.scope == "full":
            ortvec_files.append((_maybe_rescale(Path(n.file or ""), n), n.label))
        elif n.scope.startswith("run:"):
            run_idx = int(n.scope.split(":", 1)[1])
            if run_idx < 1 or run_idx > n_runs:
                raise ValueError(f"nuisance '{n.label}': run {run_idx} out of range [1, {n_runs}]")
            padortvec_files.append((_maybe_rescale(Path(n.file or ""), n), n.label, run_idx))
        elif n.scope == "glob":
            if not n.pattern:
                raise ValueError(f"nuisance '{n.label}': scope='glob' but no pattern")
            matched = sorted(Path(p) for p in glob_module.glob(n.pattern))
            if not matched:
                raise ValueError(f"nuisance '{n.label}': glob {n.pattern!r} matched no files")
            run_indices_0 = _infer_run_indices_from_filenames(
                [p.name for p in matched],
                n_runs=n_runs,
            )
            # Pre-validate row counts so failures point at the glob source.
            for path, run_idx0 in zip(matched, run_indices_0, strict=True):
                expected = n_timepoints_per_run[run_idx0]
                n_rows = _count_1d_rows(path)
                if n_rows != expected:
                    raise ValueError(
                        f"nuisance '{n.label}': {path} has {n_rows} rows "
                        f"but run {run_idx0 + 1} expects {expected} "
                        "(glob mode requires one-run-length files)"
                    )
                padortvec_files.append((_maybe_rescale(path, n), n.label, run_idx0 + 1))
        else:
            raise ValueError(f"nuisance '{n.label}': unknown scope {n.scope!r}")

    return ortvec_files, padortvec_files


def _count_1d_rows(path: Path) -> int:
    """Count non-comment, non-blank rows in a plain-text .1D file."""
    n = 0
    with open(path) as fh:
        for line in fh:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                n += 1
    return n


def _reorder_columns_afni_style(
    design: np.ndarray,
    regressor_labels: list[str],
    metadata: dict,
) -> tuple[np.ndarray, list[str], dict]:
    """Permute columns from build_design_matrix's
    polort → padortvec → ortvec → stim → extra
    layout to AFNI's canonical
    polort → stim → padortvec → ortvec → extra.

    Updates the index lists in *metadata* so they point at the new positions.
    GLT contrasts resolve by label name, so they survive the permutation.
    """
    polort = list(metadata.get("polort_indices", []))
    padort = list(metadata.get("padortvec_indices", []))
    ort = list(metadata.get("ortvec_indices", []))
    stim = list(metadata.get("stim_indices", []))
    extra = list(metadata.get("extra_indices", []))

    perm = polort + stim + padort + ort + extra
    n = design.shape[1]
    if sorted(perm) != list(range(n)):
        # Defensive: if some indices weren't accounted for, fall through.
        return design, regressor_labels, metadata

    new_design = design[:, perm]
    new_labels = [regressor_labels[i] for i in perm]

    # Rebuild index lists pointing at new positions.
    new_polort = list(range(len(polort)))
    new_stim = list(range(len(polort), len(polort) + len(stim)))
    start = len(polort) + len(stim)
    new_padort = list(range(start, start + len(padort)))
    start += len(padort)  # noqa: E702
    new_ort = list(range(start, start + len(ort)))
    start += len(ort)  # noqa: E702
    new_extra = list(range(start, start + len(extra)))

    metadata = dict(metadata)
    metadata["polort_indices"] = new_polort
    metadata["stim_indices"] = new_stim
    metadata["padortvec_indices"] = new_padort
    metadata["ortvec_indices"] = new_ort
    metadata["extra_indices"] = new_extra
    metadata["nuisance_indices"] = new_polort + new_padort + new_ort + new_extra
    return new_design, new_labels, metadata


def _read_censor(path: Path) -> np.ndarray:
    """Read AFNI ``outcount.1D``-style keep mask: one number per line,
    nonzero = keep, zero = censor. Returns a 0/1 int array."""
    values = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values.append(int(float(line) != 0.0))
    return np.array(values, dtype=int)


def _do_compile(args: argparse.Namespace) -> int:
    spec_path = _resolve_spec_path(args.spec)
    spec = load_spec(spec_path)

    xmat_path = Path(args.xmat)
    if not _confirm_overwrite(xmat_path, args.overwrite, "xmat"):
        return 1
    cols = (
        spec.meta.events_columns.onset,
        spec.meta.events_columns.duration,
        spec.meta.events_columns.trial_type,
    )

    # polort: list form means per-run; build_design_matrix only supports a
    # scalar today. Collapse with the max and warn.
    if isinstance(spec.meta.polort, list):
        polort_scalar = max(spec.meta.polort)
        print(
            f"warn: per-run polort {spec.meta.polort} not yet supported by "
            f"build_design_matrix; using max={polort_scalar}.",
            file=sys.stderr,
        )
    elif spec.meta.polort == "auto":
        polort_scalar = max(
            int(1 + math.floor(n * spec.meta.tr / 150.0)) for n in spec.meta.n_timepoints_per_run
        )
    else:
        polort_scalar = int(spec.meta.polort)

    events_files = [Path(r.events) if r.events else None for r in spec.meta.runs]
    if any(ef is None for ef in events_files):
        raise ValueError("Every [meta].runs entry must have an 'events' field for compile.")

    # Expand events → AFNI timing files in a tempdir.
    tmpdir = Path(tempfile.mkdtemp(prefix="ffs_design_spec_"))
    timing_files: list[Path] = []
    stim_labels: list[str] = []
    hrf_models: list[str] = []
    im_modes: list[bool] = []
    for ev in spec.events:
        for path, label, hrf, im_flag in _expand_event_to_stims(
            ev, events_files, cols, spec.meta.tr, tmpdir
        ):
            timing_files.append(path)
            stim_labels.append(label)
            hrf_models.append(hrf)
            im_modes.append(im_flag)

    # Nuisance: resolve each [[nuisance]] spec into either an ortvec_files
    # entry (full-length, used as-is) or padortvec_files entries (per-run,
    # zero-padded). Glob scope is expanded here so the spec stays portable
    # but compile gets concrete files.
    ortvec_files, padortvec_files = _resolve_nuisance_for_compile(
        spec.nuisance,
        n_runs=len(spec.meta.runs),
        n_timepoints_per_run=spec.meta.n_timepoints_per_run,
        tmpdir=tmpdir,
    )

    design, regressor_labels, run_starts, metadata = build_design_matrix(
        timing_files=[str(p) for p in timing_files],
        stim_labels=stim_labels,
        n_timepoints_per_run=spec.meta.n_timepoints_per_run,
        tr=spec.meta.tr,
        polort=polort_scalar,
        hrf_models=hrf_models,
        im_mode=im_modes,
        padortvec_files=[(str(p), l, r) for p, l, r in padortvec_files] or None,
        ortvec_files=[(str(p), l) for p, l in ortvec_files] or None,
    )

    # Reorder columns to AFNI's canonical polort → stim → nuisance layout.
    # build_design_matrix emits polort → nuisance → stim, which makes the xmat
    # diff against AFNI's X.xmat.1D unreadable. Permute back here so downstream
    # tools / human inspection match the AFNI convention.
    design, regressor_labels, metadata = _reorder_columns_afni_style(
        design,
        regressor_labels,
        metadata,
    )

    # Resolve contrasts against stim_labels (the expanded labels we actually
    # built, including any ``_durN`` splits).
    glt_contrasts: list[tuple[str | list[str], str]] = []
    for c in spec.contrasts:
        rows = resolve_contrast(c, stim_labels)
        if len(rows) == 1:
            glt_contrasts.append((_resolved_row_to_sym(rows[0]), c.label))
        else:
            glt_contrasts.append(([_resolved_row_to_sym(r) for r in rows], c.label))

    # Censor.
    good_list_arg: list[int] | None = None
    n_row_full_arg: int | None = None
    if spec.meta.censor:
        keep = _read_censor(Path(spec.meta.censor))
        good_list_arg, n_row_full_arg = good_list_from_censor(keep)
        if len(good_list_arg) != design.shape[0]:
            raise ValueError(
                f"Censor mask kept {len(good_list_arg)} TRs but design has "
                f"{design.shape[0]} rows. Did you build the design from "
                "uncensored data?"
            )

    write_afni_xmat(
        args.xmat,
        design,
        regressor_labels,
        run_starts=run_starts,
        metadata=metadata,
        glt_contrasts=glt_contrasts or None,
        command_line=" ".join(sys.argv),
        good_list=good_list_arg,
        n_row_full=n_row_full_arg,
    )

    if args.verb >= 1:
        print(f"Compiled {args.spec} -> {args.xmat}", file=sys.stderr)
        print(f"  {design.shape[0]} TRs × {design.shape[1]} regressors", file=sys.stderr)
        print(f"  Stim labels: {stim_labels}", file=sys.stderr)
        if glt_contrasts:
            print(f"  Contrasts: {[lbl for _, lbl in glt_contrasts]}", file=sys.stderr)

    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.cmd == "stub":
        return _do_stub(args)
    if args.cmd == "compile":
        return _do_compile(args)
    parser.error(f"Unknown subcommand: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
