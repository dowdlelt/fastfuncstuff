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


def spec_path(task: str) -> str:
    """The design TOML's name, relative to the script's working dir."""
    return f"stage{STAGE_NUMBERS['design']:02d}.design.task-{task}.toml"


def runs_by_task(plan: Plan) -> dict[str, list[PlanRun]]:
    tasks: dict[str, list[PlanRun]] = {}
    for pr in plan.runs:
        tasks.setdefault(pr.bold.task, []).append(pr)
    return tasks


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

        trs = {pr.bold.tr for pr in prs if pr.bold.tr}
        nuisance, _skipped = nuisance_specs(task, opt)
        try:
            spec, notes = build_stub_spec(
                [Path(f"stage10.final.{_frag(pr)}.nii{opt.final_fmt}") for pr in prs],
                events,
                tr=trs.pop() if len(trs) == 1 else None,
                n_timepoints_per_run=[_n_timepoints(pr, opt.noise_vols) for pr in prs],
                nuisance=nuisance,
            )
        except (ValueError, OSError) as exc:
            rows.append((task, str(dest), f"skipped: {exc}"))
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
                "  - [[contrasts]] none are guessed — add the ones your question needs.\n"
                "The runs listed below are this script's stage10 outputs; n_timepoints_per_run\n"
                "was read from the raw BIDS headers (preprocessing preserves run length).\n"
            ),
            event_notes=notes,
            include_contrast_examples=True,
        )
        rows.append((task, str(dest), "wrote"))
    return rows
