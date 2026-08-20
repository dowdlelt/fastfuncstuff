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


def spec_path(task: str) -> str:
    """The design TOML's name, relative to the script's working dir."""
    return f"stage{STAGE_NUMBERS['design']:02d}.design.task-{task}.toml"


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


def _n_timepoints(pr: PlanRun, noise_vols: int) -> int:
    """Timepoints the preprocessed run will have: the raw header's count minus
    any trailing noise volumes the pipeline trims up front."""
    from fastfuncstuff.design.spec import bold_header

    n_tp, _ = bold_header(pr.bold.mag_path)
    return max(int(n_tp) - int(noise_vols), 0)


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
    from fastfuncstuff.design.spec import build_stub_spec, write_spec

    opt = plan.options
    rows: list[tuple[str, str, str]] = []
    if not opt.run_glm:
        return rows

    out_dir = Path(work_dir)
    # The specs name the copies, so they have to exist before anything reads them.
    copy_events(plan, bids_root, work_dir)
    copies = stimuli_map(plan, bids_root)
    for task, prs in runs_by_task(plan).items():
        dest = out_dir / spec_path(task)
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
            rows.append((task, str(dest), "kept"))
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
        try:
            spec, notes = build_stub_spec(
                [Path(f"stage10.final.{_frag(pr)}.nii{opt.final_fmt}") for pr in prs],
                scan_paths,
                tr=trs.pop() if len(trs) == 1 else None,
                n_timepoints_per_run=[_n_timepoints(pr, opt.noise_vols) for pr in prs],
                event_cols=event_cols,
                nuisance=nuisance,
                stim_vec=stim_vecs,
            )
        except (ValueError, OSError) as exc:
            rows.append((task, str(dest), f"skipped: {exc}"))
            continue

        for run_spec, rel in zip(spec.meta.runs, rel_paths, strict=True):
            run_spec.events = rel

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
        rows.append((task, str(dest), "wrote"))
    return rows
