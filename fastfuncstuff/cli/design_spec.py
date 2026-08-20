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

from fastfuncstuff.cli_utils import add_ortvec_arguments
from fastfuncstuff.design.bids_events import check_events_pairing
from fastfuncstuff.design.builder import (
    build_design_matrix,
    good_list_from_censor,
    parse_afni_timing_file,
    write_afni_xmat,
)
from fastfuncstuff.design.spec import (
    EventSpec,
    NuisanceSpec,
    build_stub_spec,
    load_spec,
    resolve_contrast,
    write_spec,
)

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
        "-event_cols",
        "-event-cols",
        nargs=3,
        dest="event_cols",
        metavar=("ONSET", "DURATION", "TRIAL_TYPE"),
        help="Non-BIDS column names for the events.tsv files (as in ffs_reml -event_cols).",
    )
    p_stub.add_argument(
        "-event_ignore",
        "-event-ignore",
        "-drop-trial-types",
        "-drop_trial_types",
        nargs="*",
        dest="drop_trial_types",
        default=["rest", "Rest", "REST", "baseline"],
        help="Trial types to exclude from the spec (as in ffs_reml -event_ignore).",
    )
    p_stub.add_argument(
        "-default-hrf",
        default="SPMG1",
        help="HRF model used for every event in the stub. The "
        "duration is taken from the [[events]] 'duration' "
        "field, so the bare model name is what you want here.",
    )
    # Nuisance regressors — register the identical -ortvec / -ortvec_run /
    # -ortvec_glob / -ortvec_concat set that ffs_reml exposes, so a design
    # built here can be handed to ffs_reml (or compiled) without relearning
    # flags. See the generated [[nuisance]] section header in design.toml for
    # the full padding-semantics writeup.
    add_ortvec_arguments(p_stub)
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


def _build_nuisance_from_cli_args(
    args: argparse.Namespace,
    n_runs: int,
) -> list[NuisanceSpec]:
    """Turn -ortvec / -ortvec_run / -ortvec_glob / -ortvec_concat into
    NuisanceSpec rows. All four flags mirror ffs_reml's add_ortvec_arguments
    so users learn one API. The fourth (concat) is a convenience for
    AFNI-style already-padded per-run files (each is full length and
    block-diagonal — the demean output of afni_proc.py)."""
    from fastfuncstuff.cli_utils import expand_ortvec_concat, split_label_transform

    out: list[NuisanceSpec] = []

    for file, raw_label in getattr(args, "ortvec", None) or []:
        label, tf = split_label_transform(raw_label)
        out.append(NuisanceSpec(file=str(file), label=label, scope="full", transform=tf))

    for file, raw_label, run in getattr(args, "ortvec_run", None) or []:
        label, tf = split_label_transform(raw_label)
        out.append(NuisanceSpec(file=str(file), label=label, scope=f"run:{int(run)}", transform=tf))

    for pattern, raw_label in getattr(args, "ortvec_glob", None) or []:
        label, tf = split_label_transform(raw_label)
        out.append(
            NuisanceSpec(
                file=None,
                label=label,
                scope="glob",
                pattern=str(pattern),
                transform=tf,
            )
        )

    for pattern, raw_label in getattr(args, "ortvec_concat", None) or []:
        label, tf = split_label_transform(raw_label)
        for path, suffixed_label in expand_ortvec_concat(pattern, label, n_runs):
            out.append(
                NuisanceSpec(
                    file=str(path),
                    label=suffixed_label,
                    scope="full",
                    transform=tf,
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
    try:
        check_events_pairing(bold_paths, events_paths, n_runs=len(bold_paths))
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    spec, event_notes = build_stub_spec(
        bold_paths,
        events_paths,
        tr=args.TR,
        event_cols=tuple(args.event_cols) if args.event_cols else None,
        drop_trial_types=list(args.drop_trial_types),
        default_hrf=args.default_hrf,
        nuisance=_build_nuisance_from_cli_args(args, len(bold_paths)),
    )

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
    print(
        f"  {len(spec.meta.runs)} runs, TR={spec.meta.tr}, polort={spec.meta.polort}",
        file=sys.stderr,
    )
    tts = [e.trial_type for e in spec.events]
    print(f"  {len(tts)} trial types: {tts}", file=sys.stderr)
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


def _shift_events_for_trim(
    per_run: list[list[tuple[float, float]]],
    trim,
    run_lengths_sec: list[float],
) -> list[list[tuple[float, float]]]:
    """Shift (onset, duration) pairs onto the -drop_first retained window.

    The spec path carries a real duration per *event* (not one per condition),
    so the overlap test here is exact: keep anything whose ``[onset, onset+dur)``
    still intersects ``[0, run_end)``. Negative onsets are kept -- the event
    began before the retained window but is still running inside it, and the
    builder truncates its boxcar (see fastfuncstuff/design/trim.py).
    """
    shift = trim.shift_sec
    out: list[list[tuple[float, float]]] = []
    for r, run in enumerate(per_run):
        end = run_lengths_sec[r] if r < len(run_lengths_sec) else float("inf")
        kept = []
        for onset, dur in run:
            o = onset - shift
            if (o + dur) > 0.0 and o < end:
                kept.append((o, dur))
        out.append(kept)
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


def _warn_sparse_conditions(
    timing_files: list[Path],
    stim_labels: list[str],
    n_runs: int,
) -> None:
    """Report conditions that are absent from some runs, or so rare they will
    barely be estimable."""
    for path, label in zip(timing_files, stim_labels, strict=True):
        per_run = parse_afni_timing_file(path)
        missing = [i + 1 for i, onsets in enumerate(per_run) if len(onsets) == 0]
        n_events = sum(len(o) for o in per_run)
        if not missing:
            continue
        print(
            f"⚠️  Condition '{label}': {n_events} event(s) total; absent from "
            f"{len(missing)}/{n_runs} run(s) ({', '.join(str(r) for r in missing)}). "
            "The regressor is zero over those runs — legal, but the estimate "
            "rests on the runs that have it.",
            flush=True,
        )


def _expand_event_to_stims(
    event: EventSpec,
    events_files: list[Path],
    cols: tuple[str, str, str],
    tr: float,
    tmpdir: Path,
    trim=None,
    run_lengths_sec: list[float] | None = None,
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

    if trim is not None and trim.active:
        per_run = _shift_events_for_trim(per_run, trim, run_lengths_sec or [])

    # Resolve duration: explicit number vs from_events.
    if event.duration == "from_events":
        unique_durs = sorted({d for run in per_run for _, d in run})
        if "(" in event.hrf and unique_durs:
            # Bug-of-record: stubs used to default to "SPMG1(0)", which reads as
            # an override and modelled 10 s blocks as impulses. Warn loudly.
            print(
                f"⚠️  Event '{event.trial_type}': duration = \"from_events\" but hrf = "
                f"{event.hrf!r} carries explicit arguments, so the events-file durations "
                f"(observed {', '.join(f'{d:g}' for d in unique_durs[:5])}"
                f"{'…' if len(unique_durs) > 5 else ''} s) are IGNORED. "
                f"Write the bare model name (e.g. '{event.hrf.split('(')[0]}') to use them.",
                flush=True,
            )
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


def _materialize_nuisance(
    path: Path,
    n_spec: NuisanceSpec,
    tmpdir: Path | None,
    run_lengths: list[int] | None = None,
) -> Path:
    """Apply a nuisance block's transform + rescale, writing the result to
    *tmpdir* and returning its path. The original file is untouched; a block
    that asks for neither is passed straight through.

    ``run_lengths`` splits a scope="full" file into runs so the derivative never
    crosses a run boundary (per-run and glob files are one run by construction,
    so they pass None). This mirrors 1d_tool.py -set_nruns / -set_run_lens.
    """
    from fastfuncstuff.cli_utils import apply_nuisance_transform

    if n_spec.transform == "none" and n_spec.rescale == "as-is":
        return path
    if tmpdir is None:
        raise RuntimeError(
            f"nuisance '{n_spec.label}': transform/rescale requires a tmpdir to write to"
        )
    arr = np.loadtxt(path, ndmin=2)
    if n_spec.transform != "none":
        if run_lengths:
            if sum(run_lengths) != arr.shape[0]:
                raise ValueError(
                    f"nuisance '{n_spec.label}': {path} has {arr.shape[0]} rows but the "
                    f"design's runs total {sum(run_lengths)} — cannot split for transform"
                )
            parts, start = [], 0
            for n_rows in run_lengths:
                parts.append(
                    apply_nuisance_transform(arr[start : start + n_rows], n_spec.transform)
                )
                start += n_rows
            arr = np.vstack(parts)
        else:
            arr = apply_nuisance_transform(arr, n_spec.transform)
    if n_spec.rescale == "demean":
        arr = arr - arr.mean(axis=0, keepdims=True)
    tag = "_".join(t for t in (n_spec.transform, n_spec.rescale) if t not in ("none", "as-is"))
    out = tmpdir / f"{path.stem}_{tag}.1D"
    np.savetxt(out, arr, fmt="%.10g")
    return out


def _trim_1d_for_compile(
    path: Path,
    expected: list[int],
    trim,
    tmpdir: Path | None,
) -> Path:
    """Trim a nuisance .1D onto the retained window, if it is not already.

    Motion/physio regressors are produced from the untrimmed runs, so under
    -drop_first the file on disk is longer than the design. Accept either length
    (see fastfuncstuff/design/trim.py) and write the trimmed copy to *tmpdir*.
    """
    from fastfuncstuff.design.trim import trim_run_series

    if trim is None or not trim.active:
        return path
    arr = np.loadtxt(path, ndmin=2)
    if arr.shape[0] == sum(expected):
        return path
    blocks, off = [], 0
    for n_trimmed in expected:
        n_un = n_trimmed + trim.total
        blocks.append(trim_run_series(arr[off : off + n_un], n_trimmed, trim, path))
        off += n_un
    if off != arr.shape[0]:
        raise ValueError(
            f"{path}: has {arr.shape[0]} rows; expected {sum(expected)} (trimmed) or {off} "
            f"(untrimmed, before dropping {trim.describe()} per run)."
        )
    if tmpdir is None:
        raise RuntimeError(f"{path}: trimming needs a tmpdir to write to")
    out = tmpdir / f"trimmed_{path.name}"
    np.savetxt(out, np.concatenate(blocks))
    return out


def _build_stim_vec_for_compile(
    stim_vec_specs,
    n_timepoints_per_run: list[int],
    tr: float,
    hrf_models: list[str],
    trim=None,
):
    """Turn ``[[stim_vec]]`` blocks into design columns + labels.

    Returns ``(columns, labels)`` -- ``columns`` is
    ``(total_timepoints, k)`` or None when there is nothing to add.

    The default HRF follows the design: if every event uses the same bare model
    the vectors ride that one, otherwise SPMG1. A per-block ``hrf =`` overrides.
    Compiling to an xmat is what makes ``-stim_event_vec`` usable with
    ``ffs_reml -matrix``, which otherwise refuses the flag -- the columns have to
    reach the matrix somehow, and the spec is that somehow.
    """
    import glob as glob_module

    from fastfuncstuff.cli_utils import _infer_run_indices_from_filenames
    from fastfuncstuff.design.stim_vec import (
        build_stim_vec_design,
        load_stim_vec_block,
        resolve_stim_vec_hrf,
    )

    if not stim_vec_specs:
        return None, []

    n_runs = len(n_timepoints_per_run)
    total_tp = sum(n_timepoints_per_run)
    run_starts = [0]
    for n in n_timepoints_per_run[:-1]:
        run_starts.append(run_starts[-1] + n)

    # The design's own model, when the events agree on one.
    bases = {m.split("(", 1)[0].upper() for m in hrf_models}
    default_model = bases.pop() if len(bases) == 1 else "SPMG1"

    columns: list[np.ndarray] = []
    labels: list[str] = []
    for sv in stim_vec_specs:
        if sv.scope == "glob":
            matched = sorted(Path(x) for x in glob_module.glob(sv.pattern or ""))
            if not matched:
                raise ValueError(f"stim_vec '{sv.label}': pattern {sv.pattern!r} matched no files")
            order = _infer_run_indices_from_filenames(
                [m.name for m in matched], n_runs=n_runs, allow_sequential_fallback=True
            )
            paths = [p for _, p in sorted(zip(order, matched, strict=True))]
        else:
            paths = [Path(sv.file or "")]

        block = load_stim_vec_block(
            f"{sv.label}:{sv.mod}" if sv.mod != "none" else sv.label,
            paths,
            run_starts,
            total_tp,
            preconvolved=not sv.convolve,
            trim=trim,
        )
        model = sv.hrf or default_model
        hrf_bases, _note = resolve_stim_vec_hrf(
            model,
            is_fir_model=model.upper().startswith(("TENT", "FIR")),
            n_basis=None,
            microtime_dt=0.1,
            device=None,
        )
        design, block_labels, _groups = build_stim_vec_design(
            [block],
            n_timepoints=total_tp,
            tr=tr,
            microtime_dt=0.1,
            hrf_bases=hrf_bases,
            run_starts=run_starts,
            device=None,
        )
        columns.append(design.cpu().numpy())
        labels.extend(block_labels)

    return np.concatenate(columns, axis=1), labels


def _resolve_nuisance_for_compile(
    nuisance: list[NuisanceSpec],
    n_runs: int,
    n_timepoints_per_run: list[int],
    tmpdir: Path | None = None,
    trim=None,
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

    for n in nuisance:
        if n.scope == "full":
            ortvec_files.append(
                (
                    _materialize_nuisance(
                        _trim_1d_for_compile(
                            Path(n.file or ""), n_timepoints_per_run, trim, tmpdir
                        ),
                        n,
                        tmpdir,
                        run_lengths=n_timepoints_per_run,
                    ),
                    n.label,
                )
            )
        elif n.scope.startswith("run:"):
            run_idx = int(n.scope.split(":", 1)[1])
            if run_idx < 1 or run_idx > n_runs:
                raise ValueError(f"nuisance '{n.label}': run {run_idx} out of range [1, {n_runs}]")
            padortvec_files.append(
                (
                    _materialize_nuisance(
                        _trim_1d_for_compile(
                            Path(n.file or ""), [n_timepoints_per_run[run_idx - 1]], trim, tmpdir
                        ),
                        n,
                        tmpdir,
                    ),
                    n.label,
                    run_idx,
                )
            )
        elif n.scope == "glob":
            if not n.pattern:
                raise ValueError(f"nuisance '{n.label}': scope='glob' but no pattern")
            matched = sorted(Path(p) for p in glob_module.glob(n.pattern))
            if not matched:
                raise ValueError(f"nuisance '{n.label}': glob {n.pattern!r} matched no files")
            run_indices_0 = _infer_run_indices_from_filenames(
                [p.name for p in matched],
                n_runs=n_runs,
                allow_sequential_fallback=True,
            )
            # Pre-validate row counts so failures point at the glob source.
            for path, run_idx0 in zip(matched, run_indices_0, strict=True):
                expected = n_timepoints_per_run[run_idx0]
                path = _trim_1d_for_compile(path, [expected], trim, tmpdir)
                n_rows = _count_1d_rows(path)
                if n_rows != expected:
                    raise ValueError(
                        f"nuisance '{n.label}': {path} has {n_rows} rows "
                        f"but run {run_idx0 + 1} expects {expected} "
                        "(glob mode requires one-run-length files)"
                    )
                padortvec_files.append(
                    (_materialize_nuisance(path, n, tmpdir), n.label, run_idx0 + 1)
                )
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

    # -drop_first/-drop_last (passed through by `ffs_reml -spec`): the xmat must
    # be BUILT against the trimmed runs, since it is the design the trimmed data
    # will be fit with. Reducing the run lengths here makes every downstream
    # consumer -- polort, nuisance padding, run_starts -- describe the retained
    # window; the event shift rides along into _expand_event_to_stims.
    from fastfuncstuff.design.trim import TrimSpec

    trim = TrimSpec(
        drop_first=int(getattr(args, "drop_first", 0) or 0),
        drop_last=int(getattr(args, "drop_last", 0) or 0),
        tr=spec.meta.tr,
    )
    if trim.active:
        spec.meta.n_timepoints_per_run = [
            trim.trimmed_length(n) for n in spec.meta.n_timepoints_per_run
        ]
        print(
            f"✂️  Dropping {trim.describe()} per run → "
            f"{spec.meta.n_timepoints_per_run} TRs; "
            f"event timing shifted back by {trim.shift_sec:.3f}s",
            flush=True,
        )
    run_lengths_sec = [n * spec.meta.tr for n in spec.meta.n_timepoints_per_run]
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
            ev, events_files, cols, spec.meta.tr, tmpdir, trim, run_lengths_sec
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
        trim=trim,
    )

    # A condition that only fires in a subset of runs is legal (its column is
    # zero elsewhere) but is a common data-entry surprise and a weak estimate,
    # so say so before the design is built.
    _warn_sparse_conditions(timing_files, stim_labels, len(spec.meta.runs))

    stim_vec_cols, stim_vec_labels = _build_stim_vec_for_compile(
        spec.stim_vec,
        n_timepoints_per_run=spec.meta.n_timepoints_per_run,
        tr=spec.meta.tr,
        hrf_models=hrf_models,
        trim=trim,
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
        stim_vec_columns=stim_vec_cols,
        stim_vec_labels=stim_vec_labels or None,
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
    # Stim vectors go in as `extra_labels`: addressable by name, but kept out of
    # glob and ALLOTHERS expansion, because a `*` in a task contrast almost never
    # means "and the background".
    for c in spec.contrasts:
        rows = resolve_contrast(c, stim_labels, extra_labels=[sv.label for sv in spec.stim_vec])
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
        if stim_vec_labels:
            print(f"  Stim vectors: {stim_vec_labels}", file=sys.stderr)
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
