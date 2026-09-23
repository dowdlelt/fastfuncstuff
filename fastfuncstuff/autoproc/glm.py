"""GLM-stage inputs for ffs_autoproc: the design TOML and its nuisance blocks.

The GLM is the one stage whose *model* — conditions, HRF, contrasts, which
nuisance regressors — is a research decision, not a preprocessing default. So
instead of burying it in ffs_reml flags that are only discoverable by reading
the generated bash, ffs_autoproc writes a ``design.toml`` per task at script-
generation time and the script runs ``ffs_reml -spec``. The user edits one
annotated file (it lists the observed durations per trial type, the available
nuisance blocks, and commented contrast examples) before running anything.

Generating it early is the point: the spec describes runs that do not exist yet,
which works because the preprocessed series has the same TR and the same length
as the raw BIDS series (minus trimmed noise volumes) — see ``build_stub_spec``'s
``tr`` / ``n_timepoints_per_run`` overrides.
"""

from __future__ import annotations

from pathlib import Path

from fastfuncstuff.autoproc import config
from fastfuncstuff.autoproc.bids import find_events
from fastfuncstuff.autoproc.naming import STAGE_NUMBERS
from fastfuncstuff.autoproc.plan import Plan, PlanRun
from fastfuncstuff.design.spec import DEFAULT_EVENT_COLUMNS


def spec_path(task: str, opt=None) -> str:
    """The design TOML's name, relative to the script's working dir.

    A task with event filters gets the -glm_label in its name: the left- and
    right-hemifield fits of one task are different designs, and an existing TOML
    is kept rather than regenerated, so a shared name would fit the first
    variant's design under the second variant's label.
    """
    tag = ""
    if opt is not None and opt.event_filters.get(task) and opt.glm_label:
        tag = f"{opt.glm_label}."
    return f"stage{STAGE_NUMBERS['design']:02d}.design.{tag}task-{task}.toml"


def runs_by_task(plan: Plan) -> dict[str, list[PlanRun]]:
    tasks: dict[str, list[PlanRun]] = {}
    for pr in plan.runs:
        tasks.setdefault(pr.bold.task, []).append(pr)
    return tasks


#: Where the events TSVs are copied to, relative to the script's working dir.
STIMULI_DIR = "stimuli"


def stimuli_map(plan: Plan, bids_root: str | None) -> dict[str, str]:
    """``{absolute events TSV: work-dir-relative copy}`` for every task's events.

    The design TOMLs point at these copies, not at the BIDS tree, so the results
    directory is self-contained: the timing that produced a stat map travels with
    it, and re-running the GLM does not depend on the BIDS root still being
    mounted at the same path (or the events not having been edited since).

    Basenames normally carry the full BIDS entity set and are unique across
    tasks; a collision (two roots, same relative layout) is disambiguated by the
    source's parent directory so the mapping stays one-to-one and deterministic.
    """
    from fastfuncstuff.autoproc.emit import events_for_task

    mapping: dict[str, str] = {}
    used: set[str] = set()
    for task, prs in runs_by_task(plan).items():
        for src in events_for_task(task, prs, bids_root, plan.options):
            if src in mapping:
                continue
            name = Path(src).name
            if name in used:
                name = f"{Path(src).parent.name}_{name}"
                n = 2
                while name in used:
                    name = f"{Path(src).parent.name}_{n}_{Path(src).name}"
                    n += 1
            used.add(name)
            mapping[src] = f"{STIMULI_DIR}/{name}"
    return mapping


def copy_events(plan: Plan, bids_root: str | None, work_dir: str) -> list[str]:
    """Copy every events TSV into ``<work_dir>/stimuli/``. Returns the copies made.

    Overwrites: the copy is a mirror of the BIDS file, not an editable artifact
    (the *model* is the design TOML, which is never clobbered). A source that has
    since disappeared is skipped rather than fatal — preflight reports it.
    """
    import shutil

    made: list[str] = []
    for src, dest in stimuli_map(plan, bids_root).items():
        if not Path(src).is_file():
            continue
        target = Path(work_dir) / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        made.append(dest)
    return made


def requested_event_cols(task: str, opt) -> tuple[str, str, str] | None:
    """The ``(onset, duration, trial_type)`` column names asked for for ``task``.

    ``-sep_spec_event_cols TASK ...`` wins over the dataset-wide
    ``-spec_event_cols``; ``None`` means the BIDS defaults, which is also what
    keeps ``events_columns`` out of the generated TOML entirely.
    """
    per_task = (opt.sep_spec_event_cols or {}).get(task)
    return per_task or opt.spec_event_cols


def resolve_event_cols(
    task: str, events_paths: list, opt
) -> tuple[tuple[str, str, str] | None, str | None]:
    """``(columns to use, warning)`` for ``task``, after checking the files.

    A named column that no events TSV actually has would make the spec compile
    to an empty design (or crash) an hour of preprocessing later, so it is
    checked here against the real headers and falls back to the BIDS defaults
    with a warning. Falling back rather than erroring is deliberate: the whole
    point of ffs_autoproc is to finish and hand you an editable model — the
    column names are one line of the TOML to fix.
    """
    import csv

    want = requested_event_cols(task, opt)
    if want is None:
        return None, None

    missing: dict[str, list[str]] = {}
    for path in events_paths:
        try:
            with open(path, newline="") as fh:
                header = next(csv.reader(fh, delimiter="\t"), [])
        except OSError as exc:
            return None, f"task-{task}: cannot read {Path(path).name} ({exc}); using BIDS defaults"
        absent = [c for c in want if c not in header]
        if absent:
            missing[Path(path).name] = absent
    if not missing:
        return want, None

    name, absent = next(iter(missing.items()))
    more = f" (+{len(missing) - 1} more file(s))" if len(missing) > 1 else ""
    return None, (
        f"task-{task}: events column(s) {', '.join(absent)} not in {name}{more} — "
        f"falling back to the BIDS defaults {'/'.join(DEFAULT_EVENT_COLUMNS)}. "
        "Fix events_columns in the design TOML if that is wrong."
    )


def cut_event_warnings(task: str, runs_events: list, opt) -> list[str]:
    """Warnings for events a -cut_task_vols cut leaves past their run's new end.

    ``runs_events`` is ``[(BoldRun, events path or None)]`` for every run of
    ``task``. One line per cut run that loses events, plus a louder one for any
    condition left with no events in ANY run: ffs_reml -allow_late_events drops
    the late rows, and a condition with nothing left is an all-zero column.
    """
    from fastfuncstuff.design.bids_events import read_tsv_rows
    from fastfuncstuff.design.spec import bold_header

    n_keep = opt.cut_task_vols.get(task)
    paths = [ev for _, ev in runs_events if ev is not None]
    if n_keep is None or not paths:
        return []
    cols, _ = resolve_event_cols(task, paths, opt)
    onset_col, dur_col, tt_col = cols or DEFAULT_EVENT_COLUMNS
    filters = opt.event_filters.get(task)

    out: list[str] = []
    seen: set[str] = set()
    kept: set[str] = set()
    for run, ev in runs_events:
        if ev is None:
            continue
        tr = opt.tr if opt.tr is not None else run.tr
        n_raw, hdr_tr = bold_header(run.mag_path)
        tr = tr or hdr_tr
        end_sec = min(n_raw, n_keep) * tr
        try:
            rows, _ = read_tsv_rows(ev, onset_col, dur_col, tt_col, event_filters=filters)
        except (OSError, ValueError) as exc:
            out.append(f"-cut_task_vols task-{task}: cannot read {Path(ev).name} ({exc})")
            continue
        late: dict[str, int] = {}
        for row in rows:
            seen.add(row["_trial_type"])
            if row["_onset"] >= end_sec:
                late[row["_trial_type"]] = late.get(row["_trial_type"], 0) + 1
            else:
                kept.add(row["_trial_type"])
        if late:
            detail = ", ".join(f"{t} x{n}" for t, n in sorted(late.items()))
            # The flag is task-wide, so a run the cut never touched also has its
            # past-the-end events dropped rather than stopping the GLM.
            cut = n_raw > n_keep
            out.append(
                f"-cut_task_vols task-{task} {Path(ev).name}: {sum(late.values())} event(s) "
                f"start after {'the cut' if cut else 'the run end'} ({min(n_raw, n_keep)} "
                f"vols = {end_sec:g}s{'' if cut else ', run not cut'}) and are dropped: {detail}"
            )
    lost = sorted(seen - kept)
    if lost:
        out.append(
            f"-cut_task_vols task-{task}: condition(s) {', '.join(lost)} have NO events "
            "left before the cut in any run — each is an all-zero design column. Cut "
            "later, or remove the condition from the design TOML."
        )
    return out


def nuisance_specs(task: str, opt) -> tuple[list, list[str]]:
    """``([NuisanceSpec], [skipped_name])`` for the named sources in
    ``opt.glm_ortvec``. A source whose ``requires`` option is off is skipped —
    asking for locomoco PCs without locomoco should not produce a glob that
    matches nothing at GLM time."""
    from fastfuncstuff.design.spec import NuisanceSpec

    out, skipped = [], []
    for name in opt.glm_ortvec:
        entry = config.GLM_ORTVEC[name]
        req = entry.get("requires")
        if req and not getattr(opt, req, False):
            skipped.append(name)
            continue
        out.append(
            NuisanceSpec(
                label=name,
                scope="glob",
                pattern=entry["pattern"].format(task=task),
                transform=entry.get("transform", "none"),
            )
        )
    return out, skipped


def stim_vec_specs(task: str, opt) -> list:
    """``[StimVecSpec]`` for ``opt.glm_stim_vec``, with ``{task}`` substituted.

    A path containing a glob metacharacter becomes ``scope="glob"`` (one file per
    run, concatenated into one shared column); anything else is a single
    full-length file. No ``requires`` check as there is for nuisance sources:
    these files come from the user's stimulus code, not from a pipeline stage,
    so there is no stage whose absence would invalidate them.
    """
    from fastfuncstuff.design.spec import StimVecSpec
    from fastfuncstuff.design.stim_vec import split_label_mod

    out = []
    for raw_label, raw_path in opt.glm_stim_vec:
        label, mod = split_label_mod(raw_label)
        path = raw_path.format(task=task)
        is_glob = any(ch in path for ch in "*?[")
        out.append(
            StimVecSpec(
                label=label,
                file=None if is_glob else path,
                pattern=path if is_glob else None,
                scope="glob" if is_glob else "full",
                mod=mod,
            )
        )
    return out


def round_modes(task: str, opt) -> tuple[int | str | None, int | str | None]:
    """``(round_onset, round_duration)`` for ``task``; a per-task entry wins."""
    return (
        opt.sep_round_onsets.get(task, opt.round_onsets),
        opt.sep_round_durations.get(task, opt.round_durations),
    )


def _n_timepoints(pr: PlanRun, opt) -> int:
    """Timepoints the preprocessed run will have: the raw header's count minus
    any trailing noise volumes the pipeline trims up front, capped by
    -cut_task_vols for this run's task."""
    from fastfuncstuff.design.spec import bold_header

    n_tp, _ = bold_header(pr.bold.mag_path)
    n_tp = max(int(n_tp) - int(opt.noise_vols), 0)
    cut = opt.cut_task_vols.get(pr.bold.task)
    return min(n_tp, cut) if cut is not None else n_tp


def _kept_status(dest: Path, task: str, opt) -> str:
    """Status for a TOML left in place: a warning when it is not what was asked for."""
    from fastfuncstuff.design.spec import load_spec, parse_contrasts_text

    try:
        kept = load_spec(dest)
    except (OSError, ValueError):
        return "kept"
    problems = []
    requested = opt.event_filters.get(task, [])
    if [f.describe() for f in kept.meta.event_filters] != [f.describe() for f in requested]:
        problems.append("its event_filters differ from the ones requested")
    have = {c.label for c in kept.contrasts}
    missing = [
        c.label
        for path in opt.prebuilt_contrasts.get(task, [])
        for c in parse_contrasts_text(Path(path).read_text(), str(path))
        if c.label not in have
    ]
    if missing:
        problems.append(f"it lacks the prebuilt contrast(s) {', '.join(missing)}")
    if not problems:
        return "kept"
    return f"kept — WARNING: {'; '.join(problems)}; pass -glm_spec_overwrite to regenerate it"


def prebuilt_contrast_blocks(task: str, opt, spec, events_paths: list, event_cols) -> list[str]:
    """Validated ``-prebuilt_contrasts`` text for ``task``, ready to append verbatim.

    Raises ValueError listing every unresolvable label or duplicate at once.
    """
    from fastfuncstuff.design.spec import (
        DEFAULT_EVENT_COLUMNS,
        check_contrasts,
        parse_contrasts_text,
        predicted_stim_labels,
        scan_trial_types,
    )

    files = opt.prebuilt_contrasts.get(task, [])
    if not files:
        return []
    _, durations = scan_trial_types(
        events_paths,
        tuple(event_cols) if event_cols else DEFAULT_EVENT_COLUMNS,
        set(spec.meta.drop_trial_types),
        spec.meta.event_filters,
    )
    labels = predicted_stim_labels(spec.events, durations, spec.meta.tr)
    extra = [sv.label for sv in spec.stim_vec]
    blocks, problems, seen = [], [], []
    for path in files:
        text = Path(path).read_text()
        contrasts = parse_contrasts_text(text, str(path))
        problems += [
            f"{Path(path).name}: {p}" for p in check_contrasts(contrasts, labels, extra, seen)
        ]
        seen += [c.label for c in contrasts]
        blocks.append(
            f"\n# ---- prebuilt contrasts from {path} (-prebuilt_contrasts) ----\n"
            + text.rstrip()
            + "\n"
        )
    if problems:
        raise ValueError(
            "prebuilt contrasts do not match this design:\n    "
            + "\n    ".join(problems)
            + f"\n  design labels: {', '.join(labels + extra)}"
        )
    return blocks


def write_design_specs(
    plan: Plan,
    bids_root: str | None,
    work_dir: str,
) -> list[tuple[str, str, str]]:
    """Write one design TOML per task. Returns ``(task, path, status)`` rows,
    status in {"wrote", "kept", "skipped: <why>"}.

    An existing spec is never overwritten without ``-glm_spec_overwrite``:
    re-generating the script is routine, and silently discarding an edited model
    would be the worst bug this tool could have.
    """
    from fastfuncstuff.autoproc.emit import _frag
    from fastfuncstuff.design.spec import build_stub_spec, load_spec, write_spec

    opt = plan.options
    rows: list[tuple[str, str, str]] = []
    if not opt.run_glm:
        return rows

    out_dir = Path(work_dir)
    # The specs name the copies, so they have to exist before anything reads them.
    copy_events(plan, bids_root, work_dir)
    copies = stimuli_map(plan, bids_root)
    for task, prs in runs_by_task(plan).items():
        dest = out_dir / spec_path(task, opt)
        if opt.events:
            events = [Path(e) for e in opt.events]
            if len(events) == 1:
                events = events * len(prs)
        else:
            found = [find_events(pr.bold.mag_path, bids_root) for pr in prs]
            if any(e is None for e in found):
                rows.append((task, str(dest), "skipped: no events for every run"))
                continue
            events = [e for e in found if e is not None]
        if len(events) != len(prs):
            rows.append((task, str(dest), f"skipped: {len(events)} events for {len(prs)} runs"))
            continue

        if dest.exists() and not opt.glm_spec_overwrite:
            rows.append((task, str(dest), _kept_status(dest, task, opt)))
            continue

        # Scan the copies under work_dir, but record the work-dir-relative name:
        # the script cds there, so "stimuli/<f>.tsv" is what the GLM resolves.
        scan_paths, rel_paths = [], []
        for e in events:
            rel = copies.get(str(e))
            if rel and (out_dir / rel).is_file():
                scan_paths.append(out_dir / rel)
                rel_paths.append(rel)
            else:
                scan_paths.append(e)
                rel_paths.append(str(e))

        # -TR wins over the sidecar: for a 3D acquisition the header value is the
        # per-partition time, and the design must be sampled at the volume TR.
        trs = {opt.tr} if opt.tr is not None else {pr.bold.tr for pr in prs if pr.bold.tr}
        nuisance, _skipped = nuisance_specs(task, opt)
        stim_vecs = stim_vec_specs(task, opt)
        event_cols, _warn = resolve_event_cols(task, scan_paths, opt)
        round_onset, round_duration = round_modes(task, opt)
        try:
            spec, notes = build_stub_spec(
                [Path(f"stage10.final.{_frag(pr)}.nii{opt.final_fmt}") for pr in prs],
                scan_paths,
                tr=trs.pop() if len(trs) == 1 else None,
                n_timepoints_per_run=[_n_timepoints(pr, opt) for pr in prs],
                event_cols=event_cols,
                nuisance=nuisance,
                stim_vec=stim_vecs,
                round_onset=round_onset,
                round_duration=round_duration,
                event_filters=opt.event_filters.get(task),
            )
        except (ValueError, OSError) as exc:
            rows.append((task, str(dest), f"skipped: {exc}"))
            continue

        for run_spec, rel in zip(spec.meta.runs, rel_paths, strict=True):
            run_spec.events = rel

        # Prebuilt contrasts are checked against the labels compile WILL build
        # (filters, rounding and _dur splits applied), so a typo fails now rather
        # than an hour into the GLM. A bad one is an error, not a skip.
        try:
            prebuilt = prebuilt_contrast_blocks(task, opt, spec, scan_paths, event_cols)
        except ValueError as exc:
            rows.append((task, str(dest), f"error: {exc}"))
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        write_spec(
            spec,
            dest,
            header_comment=(
                f"Design for task-{task}, generated by ffs_autoproc. EDIT ME, then run the "
                "script.\n"
                "This file — not the ffs_reml command line — is the model. The GLM stage runs\n"
                "`ffs_reml -spec` on it, and re-running ffs_autoproc will NOT overwrite your\n"
                "edits (pass -glm_spec_overwrite if you want it regenerated).\n"
                "  - [[events]] one block per trial type found in the events TSVs; set hrf/mode.\n"
                "  - [[nuisance]] the regressors -glm_ortvec selected; patterns resolve at GLM\n"
                "    time, so the files they name do not exist yet.\n"
                "  - [[stim_vec]] continuous TR-locked stimulus vectors (-glm_stim_vec);\n"
                "    modelled as stimuli, not confounds. None unless you asked for them.\n"
                "  - [[contrasts]] none are guessed — add the ones your question needs.\n"
                "The runs listed below are this script's stage10 outputs; n_timepoints_per_run\n"
                "was read from the raw BIDS headers (preprocessing preserves run length).\n"
            ),
            event_notes=notes,
            include_contrast_examples=True,
        )
        if prebuilt:
            with open(dest, "a") as fh:
                fh.write("".join(prebuilt))
            load_spec(dest)  # the pasted text must leave a loadable design
        rows.append((task, str(dest), "wrote"))
    return rows
