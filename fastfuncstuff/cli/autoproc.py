"""ffs_autoproc — scan a BIDS directory and emit a readable, resumable ffs
pipeline script (the ffs analogue of afni_proc.py). It writes bash; it does not
run anything. Read the script, edit it, then run it yourself.

    ffs_autoproc -bids_dir DIR -subject 001 -recipe simple -suma SUMA_DIR

The script is only written when a hard preflight passes — i.e. when it should
run first try. Missing anat/TPM/reference inputs are reported and nothing is
written (the point is: no broken scripts).
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import sys
from pathlib import Path

from fastfuncstuff.autoproc import config, optcheck
from fastfuncstuff.autoproc.bids import BoldRun, find_events, pair_undetermined, scan_subject
from fastfuncstuff.autoproc.emit import write_script
from fastfuncstuff.autoproc.glm import STIMULI_DIR, write_design_specs
from fastfuncstuff.autoproc.plan import Options, build_plan
from fastfuncstuff.design.spec import DEFAULT_EVENT_COLUMNS


def _opt_help(key: str) -> str:
    return f"(default: {config.DEFAULT_OPTS[key]!r})"


def _fmt_suffix(value: str) -> str:
    """Normalize a user format string to the suffix appended after ``.nii``.

    Filenames are built as ``name.nii$FMT``, so the emitted variable holds only
    the compression suffix: ``""`` (uncompressed), ``".gz"``, or ``".zst"``.
    Accepts nii | nii.gz/gz/.gz | nii.zst/.zst/zst/zstd (any case)."""
    v = value.strip().lower().removeprefix(".").removeprefix("nii").removeprefix(".")
    if v == "":
        return ""  # bare "nii" → uncompressed
    if v == "gz":
        return ".gz"
    if v in ("zst", "zstd"):
        return ".zst"
    raise argparse.ArgumentTypeError(
        f"unrecognized format {value!r}; use nii, nii.gz (gz/.gz), or nii.zst (zst/zstd/.zst)"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ffs_autoproc",
        description="Generate a readable, resumable ffs preprocessing script from a BIDS dataset.",
        epilog=config.recipe_help(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g = p.add_argument_group("inputs & scope")
    g.add_argument("-bids_dir", "-bids-dir", required=True, help="validated BIDS root")
    g.add_argument("-subject", required=True, help="subject id (e.g. 001 or sub-001)")
    g.add_argument("-session", nargs="+", help="restrict to these sessions (labels or ses-*)")
    g.add_argument("-task", nargs="+", help="restrict to these tasks")
    g.add_argument("-out", help="output script path (default: proc_sub-<id>.sh)")
    g.add_argument("-work_dir", "-work-dir", help="working dir baked into the script (OUT)")

    g = p.add_argument_group("recipe (preset defaults; see list below --help)")
    g.add_argument(
        "-recipe",
        choices=sorted(config.RECIPES),
        help="bare_bones | simple[_linear] | simple_nonlin | complete | extreme",
    )

    g = p.add_argument_group("anatomical & reference space")
    g.add_argument("-anat", help="skull-stripped T1w to align to (e.g. SUMA brain.nii.gz)")
    g.add_argument("-suma", help="FreeSurfer SUMA dir; brain.nii.gz + builds a TPM from aseg.auto")
    g.add_argument("-no_anat", "-no-anat", action="store_true", help="stay in EPI space (no anat)")
    g.add_argument(
        "-final_dxyz",
        "-final-dxyz",
        metavar="MM",
        help="final output voxel size in mm (isotropic); default = input EPI resolution",
    )
    g.add_argument(
        "-anat_nonlin",
        "-anat-nonlin",
        action="store_const",
        const=True,
        default=None,
        help="ffs_segment nonlinear anat warp (needs -tpm or -suma)",
    )
    g.add_argument(
        "-ref_image",
        "-ref-image",
        choices=["grandmean", "sbmean", "ref_fmap", "mean_fmap"],
        default=None,
        help="Which image REPRESENTS a level of data to the level above it — one "
        "choice for two places: each session's source for the cross-session "
        "alignment, and (unless -anat_source says otherwise) the image the anat "
        "step aligns. grandmean = the level's own mean of the data (a session mean "
        "at the session level); sbmean = the same, built from SBRefs; ref_fmap = "
        "that level's reference fieldmap, undistorted (sharpest, and it DEFINES the "
        "space); mean_fmap = ref_fmap averaged with the level's other aligned "
        "fieldmap means. A choice the data cannot supply degrades per level "
        "(no fieldmaps in a session → its own mean). Default: the data mean.",
    )
    g.add_argument(
        "-anat_source",
        "-anat-source",
        choices=["grandmean", "sbmean", "ref_fmap", "mean_fmap"],
        default=None,
        help="Override -ref_image for the anat step alone (same vocabulary). All "
        "four choices share the reference-fmap grid: grandmean (best SNR, only "
        "option w/o fieldmaps or SBRefs) | sbmean (the same average built from "
        "every run's SBRef: single-band contrast, one interpolation instead of "
        "two) | ref_fmap (reference fieldmap's undistorted mean; sharpest, defines "
        "the space) | mean_fmap (ref_fmap averaged with the other groups' aligned "
        "means). Default: whatever -ref_image is (grandmean).",
    )
    g.add_argument(
        "-anat_nonlin_input",
        "-anat-nonlin-input",
        choices=["grandmean", "sbmean", "ref_fmap", "mean_fmap", "blipfor", "blip_pair"],
        default="grandmean",
        help="ffs_segment input: grandmean (works w/o fieldmap) | sbmean | ref_fmap | "
        "mean_fmap | blipfor | blip_pair",
    )
    g.add_argument(
        "-tpm", help="tissue-probability template for ffs_segment (or auto-built from -suma)"
    )
    g.add_argument("-tpm_source", "-tpm-source", help="QC anat-in-EPI source for ffs_segment")
    g.add_argument("-ref_ses", "-ref-ses", help="reference session (default: first)")
    g.add_argument(
        "-grand_reference",
        "-grand-reference",
        metavar="RESULTS_DIR",
        help="another autoproc results dir whose anat matrix this data borrows",
    )
    g.add_argument(
        "-grand_reference_nonlin",
        "-grand-reference-nonlin",
        action="store_true",
        help="add a nonlinear step when aligning to the grand reference",
    )
    g.add_argument(
        "-ref_file", "-ref-file", help="explicit reference EPI-contrast image to align to"
    )
    g.add_argument(
        "-ref_transforms",
        "-ref-transforms",
        nargs="+",
        help="matrices/warps mapping the reference to anat, in nwarp order",
    )
    g.add_argument(
        "-ref_anat", "-ref-anat", help="reference anat (copied in; ffs_segment tpm-space anat + QC)"
    )

    g = p.add_argument_group("NORDIC / phase")
    g.add_argument(
        "-want_nordic",
        "-want-nordic",
        action="store_const",
        const=True,
        default=None,
        help="run NORDIC (needs phase)",
    )
    g.add_argument("-noise_vols", "-noise-vols", type=int, default=0, help="trailing noise volumes")
    g.add_argument(
        "-phase_proc",
        "-phase-proc",
        action="store_true",
        help="carry the phase timeseries through the pipeline: ROMEO-unwrap it up front "
        "(after NORDIC if NORDIC runs), then ride it along the magnitude's warp chain at "
        "the final resample → stage10.final.*.part-phase.*, unwrapped radians in the same "
        "space as the magnitude. Requires romeo (MRItools) on $PATH and a part-phase bold "
        "for every run.",
    )
    g.add_argument(
        "-nordic_save_resid",
        "-nordic-save-resid",
        action="store_true",
        help="save the NORDIC residual map",
    )

    g = p.add_argument_group("slice timing & motion")
    g.add_argument(
        "-slicetiming_method",
        "-slicetiming-method",
        choices=["integrate", "first", "none"],
        default=None,
        help="integrate STC into the final resample | first (before moco) | none. "
        "Falls back to none when no slice timing is available (no -slicetiming and no "
        "SliceTiming in the sidecars).",
    )
    g.add_argument(
        "-slicetiming",
        "-slice_timing",
        "-slice-timing",
        metavar="FILE",
        help="slice timing for EVERY run, in place of the per-run BIDS sidecar: a text "
        "file with one acquisition offset (seconds) per slice, or a JSON with a "
        "SliceTiming field. Use when the sidecars have no SliceTiming.",
    )
    g.add_argument(
        "-TR",
        "-tr",
        dest="tr",
        type=float,
        metavar="SECONDS",
        help="volume TR for every run, overriding the sidecar/header value. 3D "
        "acquisitions often store the per-partition time in the header; slice timing "
        "and the GLM (ffs_reml -TR) need the volume TR.",
    )
    g.add_argument(
        "-moco_ref",
        "-moco-ref",
        default="sbref",
        metavar="REF",
        help="moco base: sbref (else first) | first | last | <int>",
    )
    g.add_argument(
        "-locomoco",
        action="store_const",
        const=True,
        default=None,
        help="residual PE-axis nonlinear motion",
    )

    g = p.add_argument_group("cross-run / fmap / session alignment")
    g.add_argument(
        "-no_distortion",
        "-no-distortion",
        action="store_true",
        help="skip fieldmap distortion correction even if fmaps exist",
    )
    g.add_argument("-fmap_ref", "-fmap-ref", nargs="+", help="reference fmap id(s) per session")
    g.add_argument(
        "-fmap_pe_dir",
        "-fmap-pe-dir",
        help="dir label matching the BOLD runs' phase encoding (e.g. AP); pairs "
        "opposite-PE fmaps when the sidecars omit PhaseEncodingDirection",
    )
    g.add_argument(
        "-xrun_nonlin",
        "-xrun-nonlin",
        action="store_const",
        const=True,
        default=None,
        help="nonlinear cross-run refinement",
    )
    g.add_argument(
        "-xfmap_nonlin",
        "-xfmap-nonlin",
        action="store_const",
        const=True,
        default=None,
        help="nonlinear cross-fmap-group refinement",
    )
    g.add_argument(
        "-xses_nonlin",
        "-xses-nonlin",
        action="store_const",
        const=True,
        default=None,
        help="nonlinear cross-session refinement",
    )
    for _stage, _what in (
        ("xrun", "cross-run"),
        ("xfmap", "cross-fmap-group"),
        ("xses", "cross-session"),
    ):
        g.add_argument(
            f"-{_stage}_nonlin_in_source",
            f"-{_stage}-nonlin-in-source",
            action="store_const",
            const=True,
            default=None,
            help=f"estimate the {_what} nonlinear warp in the SOURCE's frame: pass the "
            "linear stage's matrix and its un-allineated input to ffs_formwarp, which "
            "inverts the matrix and pulls the base onto the source grid. The source is "
            "never resampled, and the warp swaps places with its affine in the nwarp "
            "chain. NOTE: this relocates a clipped FoV rather than recovering it (the "
            "base falls out of frame instead) and measured worse than the default at a "
            "clipped edge -- see ../fmri_wiki/concepts/SyN.md.",
        )

    g = p.add_argument_group("GLM")
    g.add_argument(
        "-no_glm", "-no-glm", action="store_true", help="don't emit the GLM stage enabled"
    )
    g.add_argument(
        "-glm_ortvec",
        "-glm-ortvec",
        nargs="*",
        default=None,
        metavar="NAME",
        help="nuisance regressor sources for the GLM, by name ("
        + " | ".join(config.GLM_ORTVEC)
        + "). Each becomes one [[nuisance]] block in the design TOML. Pass names "
        "to select exactly those; pass the flag with no names for the default "
        "set (" + " ".join(config.DEFAULT_GLM_ORTVEC) + "). A source whose stage "
        "is not in the pipeline is dropped (with a warning if you named it).",
    )
    g.add_argument(
        "-glm_opts",
        "-glm-opts",
        default=None,
        metavar="STR",
        help="extra flags appended verbatim to the ffs_reml command "
        "(e.g. '-jobs 4 -GOFORIT'). The design itself is edited in the TOML, "
        "not here.",
    )
    g.add_argument(
        "-glm_spec_overwrite",
        "-glm-spec-overwrite",
        action="store_true",
        help="rewrite the design TOML even if it already exists. Default is to "
        "leave it alone — regenerating the script must never silently discard "
        "your edits to the model.",
    )
    g.add_argument(
        "-events",
        nargs="+",
        help="events TSV(s) (bids); single = broadcast, or one per run. "
        "Overrides BIDS-discovered events.",
    )
    g.add_argument(
        "-spec_event_cols",
        "-spec-event-cols",
        nargs=3,
        default=None,
        metavar=("ONSET", "DURATION", "TRIAL_TYPE"),
        help="column names inside the events TSVs, for EVERY task (default: the "
        "BIDS " + "/".join(DEFAULT_EVENT_COLUMNS) + "). Written as events_columns "
        "in each design TOML. A name that is not actually in the file falls back "
        "to the defaults for that task, with a warning.",
    )
    g.add_argument(
        "-sep_spec_event_cols",
        "-sep-spec-event-cols",
        nargs=4,
        action="append",
        default=None,
        metavar=("TASK", "ONSET", "DURATION", "TRIAL_TYPE"),
        help="per-task version of -spec_event_cols; repeat once per task. Wins "
        "over -spec_event_cols for the task it names, so one oddly-columned task "
        "does not force the others to be spelled out.",
    )

    g = p.add_argument_group("QC")
    g.add_argument(
        "-no_qc",
        "-no-qc",
        action="store_true",
        help="Don't emit the stageNN.QC.* stacks. By default every alignment stage "
        "concatenates the set of images it claims to have brought into one space "
        "into a single 4-D file (labelled per sub-brick), so a misregistration is "
        "seen by scrolling the time axis instead of loading N files by hand. "
        "stage10.QC.final is every run's mean in output space — the main one.",
    )

    g = p.add_argument_group("batching")
    g.add_argument(
        "-batch_overwrite",
        "-batch-overwrite",
        action="store_true",
        help="The moco and final-resample stages run one batched process each "
        "(ffs_moco/ffs_nwarp -batch), amortizing startup over all runs. By default "
        "a re-run skips runs whose outputs already exist (-batch_skip); pass this "
        "to force every run to re-process. Sets skip_moco=skip_final=0 in the "
        "emitted script (both stay editable there).",
    )

    g = p.add_argument_group("output format (nii | nii.gz/gz | nii.zst/zst)")
    g.add_argument(
        "-format",
        "-fmt",
        dest="fmt",
        type=_fmt_suffix,
        default=None,
        metavar="EXT",
        help=f"compression for working intermediates → FMT (default: nii{config.DEFAULT_FMT}). "
        "These are read many times, so .zst is fast; use nii.gz for portability.",
    )
    g.add_argument(
        "-final_format",
        "-final-format",
        dest="final_fmt",
        type=_fmt_suffix,
        default=None,
        metavar="EXT",
        help=f"compression for the final timeseries → FINAL_FMT (default: nii{config.DEFAULT_FINAL_FMT}).",
    )
    g.add_argument(
        "-glm_format",
        "-glm-format",
        dest="glm_fmt",
        type=_fmt_suffix,
        default=None,
        metavar="EXT",
        help=f"compression for GLM stat buckets → GLM_FMT (default: nii{config.DEFAULT_GLM_FMT}).",
    )

    g = p.add_argument_group("per-stage option overrides (replace the default op string)")
    for key in config.STAGE_OPT_KEYS:
        dashed = key.replace("_", "-")
        g.add_argument(f"-{key}_opts", f"-{dashed}-opts", metavar="STR", help=_opt_help(key))
    return p


def _resolve_anat(args) -> tuple[str | None, str | None]:
    """Resolve the anat path and a default tpm_source from -anat/-suma."""
    if args.anat:
        return args.anat, args.tpm_source
    if args.suma:
        brain = str(Path(args.suma) / "brain.nii.gz")
        surfvol = sorted(Path(args.suma).glob("*SurfVol*.nii*"))
        tpm_src = args.tpm_source or (str(surfvol[0]) if surfvol else None)
        return brain, tpm_src
    return None, args.tpm_source


def preflight(args, opt: Options, anat_path: str | None, subject) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Non-empty errors ⇒ do not write the script."""
    errors: list[str] = []
    warnings: list[str] = []

    # Typo in an override: catch it now, not when the script reaches that stage.
    for key in (*config.STAGE_OPT_KEYS, "glm"):
        val = getattr(args, f"{key}_opts", None)
        if val:
            errors += optcheck.check_opts(key, val)

    needs_anat = opt.go_to_anat and not opt.grand_reference and not opt.ref_file
    if needs_anat and anat_path is None:
        errors.append(
            "this recipe aligns to an anatomical, but no anat is available. Pass -anat FILE "
            "(skull-stripped T1w) or -suma DIR (FreeSurfer SUMA; uses brain.nii.gz). "
            "The in-scope session(s) had no T1w — e.g. The T1w might be in a different session, "
            "not the functional session."
        )
    if opt.anat_nonlin and not opt.tpm and not opt.fs_tpm:
        errors.append(
            "-anat_nonlin (ffs_segment) needs a subject TPM: pass -tpm FILE, or -suma DIR "
            "to build one from FreeSurfer (aseg.auto + SurfVol)."
        )
    if opt.slicetiming_method != "none" and opt.tr is None:
        # Slice timing needs a TR per run; the sidecar is the only source here.
        no_tr = [r for s in subject.sessions for r in s.bold_runs if r.tr is None]
        if no_tr:
            errors.append(
                f"slice timing is on but {len(no_tr)} run(s) have no RepetitionTime in their "
                "sidecar. Pass -TR SECONDS (volume TR) or -slicetiming_method none."
            )
    if opt.phase_proc:
        # Phase is carried per run, so a single run without one would silently
        # produce a script that dies at stage00 — catch it here instead.
        missing = [
            f"{r.session or '-'}/{r.task}/{r.run or '-'}"
            for s in subject.sessions
            for r in s.bold_runs
            if r.phase_path is None
        ]
        if missing:
            errors.append(
                "-phase_proc needs a part-phase bold for every run; these have none: "
                + ", ".join(missing)
            )
        if shutil.which("romeo") is None:
            warnings.append(
                "-phase_proc: 'romeo' (MRItools) is not on $PATH. The script's preflight will "
                "refuse to run until it is — install it or add it to PATH."
            )
    if opt.fs_tpm:  # in-script TPM build needs the FS label volume
        aseg = Path(opt.suma_dir) / "aseg.auto.nii.gz"
        if not aseg.is_file():
            errors.append(f"-suma has no aseg.auto.nii.gz to build a TPM from: {aseg}")
    # File/dir existence for everything the user pointed at.
    for label, val in (
        ("-anat", args.anat),
        ("-tpm", args.tpm),
        ("-tpm_source", opt.tpm_source),
        ("-ref_file", opt.ref_file),
        ("-ref_anat", opt.ref_anat),
        ("-slicetiming", opt.slicetiming_file),
    ):
        if val and not Path(val).exists():
            errors.append(f"{label} not found: {val}")
    for ev in opt.events or []:
        if not Path(ev).exists():
            errors.append(f"-events not found: {ev}")
    if args.suma and not Path(args.suma).is_dir():
        errors.append(f"-suma dir not found: {args.suma}")
    if args.suma and anat_path and not Path(anat_path).exists():
        errors.append(f"-suma has no brain.nii.gz: {anat_path}")
    if opt.grand_reference and not Path(opt.grand_reference).is_dir():
        errors.append(f"-grand_reference dir not found: {opt.grand_reference}")
    for m in opt.ref_transforms or []:
        if not Path(m).exists():
            errors.append(f"-ref_transforms not found: {m}")
    if opt.ref_file and not (opt.ref_transforms or opt.ref_anat):
        warnings.append(
            "-ref_file given without -ref_transforms/-ref_anat: the reference is assumed to "
            "already be in anat space."
        )
    if opt.anat_source != "grandmean":
        if not opt.go_to_anat:
            warnings.append("-anat_source is ignored with -no_anat (there is no anat step).")
        elif opt.has_grand_ref:
            warnings.append(
                "-anat_source is ignored when the anat matrix is borrowed (-grand_reference) "
                "or overridden (-ref_file): no local anat alignment is computed."
            )
        elif opt.anat_source == "sbmean":
            # sbmean rides the SBRef lane, not the fieldmaps — different requirement.
            missing = [
                f"{r.task}/run-{r.run or '?'}"
                for s in subject.sessions
                for r in s.bold_runs
                if r.sbref_path is None
            ]
            if opt.moco_ref != "sbref":
                warnings.append(
                    f"-anat_source sbmean needs -moco_ref sbref (got {opt.moco_ref}): only "
                    "then is each SBRef the run's post-moco space. Falling back to grandmean."
                )
            elif missing:
                warnings.append(
                    "-anat_source sbmean needs an SBRef for every run; missing for "
                    + ", ".join(missing[:4])
                    + (" ..." if len(missing) > 4 else "")
                    + ". Falling back to grandmean."
                )
        elif not any(s.fmaps for s in subject.sessions) or not opt.distortion:
            warnings.append(
                f"-anat_source {opt.anat_source} needs fieldmaps; falling back to grandmean "
                "(ffs_segment is what recovers the distortion in that case)."
            )
    # A regressor named explicitly but not produced by this pipeline is a real
    # mismatch worth saying out loud; the same entry coming from the default set
    # is dropped silently (config.GLM_ORTVEC[...]["requires"]).
    for name in args.glm_ortvec or []:
        req = config.GLM_ORTVEC.get(name, {}).get("requires")
        if req and not getattr(opt, req, False):
            warnings.append(
                f"-glm_ortvec {name} needs -{req}, which is off — that nuisance block is "
                "dropped from the design."
            )
    # A -sep_spec_event_cols entry naming a task that is not in scope is almost
    # always a typo, and it would silently do nothing.
    all_tasks = {r.task for s in subject.sessions for r in s.bold_runs}
    for task in sorted(opt.sep_spec_event_cols or {}):
        if task not in all_tasks:
            warnings.append(
                f"-sep_spec_event_cols names task '{task}', which is not in this "
                f"subject/scope ({', '.join(sorted(all_tasks)) or 'none'}) — it does nothing."
            )
    # The requested event columns must exist in the files, or the design compiles
    # to nothing an hour later. Checked here; the spec writer falls back the same
    # way, so the warning and the TOML agree.
    if opt.run_glm and (opt.spec_event_cols or opt.sep_spec_event_cols):
        from fastfuncstuff.autoproc.glm import resolve_event_cols

        for task in sorted(all_tasks):
            if opt.events:
                paths = [Path(e) for e in opt.events]
            else:
                paths = [ev for _, ev in _events_by_run(args.bids_dir, task, subject) if ev]
            if not paths:
                continue
            _cols, warn = resolve_event_cols(task, paths, opt)
            if warn:
                warnings.append(warn)

    # Events: not a hard error (preprocessing still runs), but warn per task so
    # the user knows the GLM will fail without them.
    if opt.run_glm and not opt.events:
        tasks = all_tasks
        for task in sorted(tasks):
            pairs = _events_by_run(args.bids_dir, task, subject)
            missing = [r for r, ev in pairs if ev is None]
            if len(missing) == len(pairs):
                warnings.append(
                    f"no events found for task '{task}' (BIDS or -events). The GLM stage will "
                    "fail for it unless you pass -events; preprocessing (stages 0–10) is unaffected."
                )
            elif missing:
                names = ", ".join(f"run-{r.run or '?'}" for r in missing)
                warnings.append(
                    f"task '{task}': events found for some runs but not {names}. The GLM stage "
                    "will use what was found — check that this is what you want."
                )
    return errors, warnings


def _resolve_glm_ortvec(args, recipe: dict) -> list[str]:
    """Names of the GLM nuisance sources to use.

    ``-glm_ortvec`` with no names means "the default set"; with names it means
    exactly those. Unset falls back to the recipe. Unknown names are a hard
    error here rather than an empty glob buried in the script.
    """
    if args.glm_ortvec is None:
        names = list(recipe.get("glm_ortvec", []))
    elif args.glm_ortvec == []:
        names = list(config.DEFAULT_GLM_ORTVEC)
    else:
        names = list(args.glm_ortvec)
    unknown = [n for n in names if n not in config.GLM_ORTVEC]
    if unknown:
        raise SystemExit(
            f"ffs_autoproc: unknown -glm_ortvec name(s): {', '.join(unknown)}\n"
            f"  known: {', '.join(config.GLM_ORTVEC)}"
        )
    return names


def _resolve_event_cols(
    args,
) -> tuple[tuple[str, str, str] | None, dict[str, tuple[str, str, str]]]:
    """``(dataset-wide triple, {task: triple})`` from -spec_event_cols /
    -sep_spec_event_cols. A task named twice keeps the last spelling."""
    wide = tuple(args.spec_event_cols) if args.spec_event_cols else None
    per_task: dict[str, tuple[str, str, str]] = {}
    for task, *cols in args.sep_spec_event_cols or []:
        per_task[task[len("task-") :] if task.startswith("task-") else task] = tuple(cols)
    return wide, per_task  # type: ignore[return-value]


def _events_by_run(bids_dir: str, task: str, subject) -> list[tuple[BoldRun, Path | None]]:
    """(run, its events TSV or None) for every run of a task, in scan order.

    Resolution is ``bids.find_events`` — the same function the emitter writes
    into the script — so what preflight reports is exactly what the GLM stage
    gets, and the dataset-root ``task-<T>_events.tsv`` is the last fallback.
    """
    root_ev = Path(bids_dir) / f"task-{task}_events.tsv"
    pairs = []
    for sess in subject.sessions:
        for run in sess.bold_runs:
            if run.task != task:
                continue
            ev = find_events(run.mag_path, bids_dir)
            if ev is None and root_ev.is_file():
                ev = root_ev
            pairs.append((run, ev))
    return pairs


def _resolve_slicetiming(args, rget, subject) -> str:
    """The effective ``-slicetiming_method``, downgraded to ``none`` when there is
    no slice timing to apply.

    STC is on by default, but 3D acquisitions (and any converter that drops the
    field) leave the sidecars without ``SliceTiming`` — ffs_nwarp/ffs_slicetime
    would then die mid-run on a script that looked fine when it was written. Say
    so loudly at generation time instead, since a *missing* sidecar field is just
    as often a conversion mistake as a genuine 3D acquisition."""
    method = args.slicetiming_method or rget("slicetiming_method", "integrate")
    if method == "none" or args.slicetiming:
        return method
    runs = [r for s in subject.sessions for r in s.bold_runs]
    have = [r for r in runs if r.has_slice_timing]
    if len(have) == len(runs):
        return method
    which = (
        "no run has SliceTiming in its sidecar"
        if not have
        else f"{len(runs) - len(have)} of {len(runs)} runs have no SliceTiming in their sidecar"
    )
    print(
        f"warning: slice timing disabled (-slicetiming_method {method} → none): {which}.\n"
        "         Expected for a 3D acquisition. If it is NOT — the field was lost in "
        "conversion —\n"
        "         pass -slicetiming FILE (one offset per slice, seconds) and regenerate.",
        file=sys.stderr,
    )
    return "none"


def _fmt_time(sec: float | None) -> str:
    if sec is None:
        return "  --:--:-- "
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _report_fmap_assignment(subject) -> None:
    """Print the per-session fieldmap → run assignment (a useful diagnostic when
    it was inferred by AcquisitionTime rather than declared by IntendedFor)."""
    if not any(s.fmaps for s in subject.sessions):
        return
    print("== fieldmap → run assignment ==", file=sys.stderr)
    for sess in subject.sessions:
        if not sess.fmaps:
            continue
        multitask = len({r.task for r in sess.bold_runs}) > 1
        print(f"  ses-{sess.session}:", file=sys.stderr)
        for fg in sess.fmaps:
            # Disambiguate by task when the session mixes tasks (run numbers repeat).
            items = (
                ", ".join(f"{t}/{r}" for t, r in fg.intended_runs)
                if multitask
                else ", ".join(fg.run_ids)
            ) or "(none)"
            # A paired fmap (both PE polarities in fmap/) corrects itself; an
            # unpaired one borrows a data run as its forward image. Worth showing:
            # it's the difference between two blip inputs and one.
            kind = "pair" if fg.forward_path is not None else "solo"
            print(
                f"    fmap-{fg.fmap_id} ({kind})  t={_fmt_time(fg.acq_time)}  →  [{items}]",
                file=sys.stderr,
            )
        if undet := pair_undetermined(sess):
            print(
                f"    NOTE: {', '.join(undet)} look like opposite-PE pairs, but no sidecar"
                " gives a phase-encoding polarity, so each is corrected against a data"
                " run instead of its mate. Pass -fmap_pe_dir <label of the runs' PE dir>"
                " to pair them.",
                file=sys.stderr,
            )


def _report_events(args, opt, subject) -> None:
    """Print the events TSV each task's GLM will read — the same resolution the
    emitted script gets. Printed because the alternative is finding out at the
    very end of the pipeline, after every preprocessing stage has run."""
    if not opt.run_glm:
        return
    if opt.events:
        print("== events (from -events) ==", file=sys.stderr)
        for ev in opt.events:
            print(f"  {ev}", file=sys.stderr)
        return
    tasks = sorted({r.task for s in subject.sessions for r in s.bold_runs})
    print("== events → run assignment ==", file=sys.stderr)
    for task in tasks:
        for run, ev in _events_by_run(args.bids_dir, task, subject):
            coord = f"{task}/run-{run.run}" if run.run else task
            print(f"  {coord:<24} {ev if ev is not None else '(none found)'}", file=sys.stderr)


# Every input path the emitter bakes into the script verbatim. Relative ones are
# resolved against the CWD of *this* invocation before anything reads them.
# -out/-work_dir are not here: -out is where the script is written now, and
# work_dir is resolved separately (it becomes OUT inside the script).
_PATH_ARGS = (
    "bids_dir",
    "anat",
    "suma",
    "tpm",
    "tpm_source",
    "grand_reference",
    "ref_file",
    "ref_anat",
)
_PATH_LIST_ARGS = ("ref_transforms", "events")


def _absolutize_inputs(args) -> None:
    """Make user-supplied input paths absolute.

    The generated script hard-codes these, and it is routinely moved (or run from
    a different cwd) after generation — a relative ``-grand_reference
    floc_..results/`` would then point at nothing and the run dies at stage09.
    Resolving here also means preflight errors name the real path."""

    def abspath(p: str) -> str:
        return str(Path(p).expanduser().resolve())

    for name in _PATH_ARGS:
        val = getattr(args, name, None)
        if val:
            setattr(args, name, abspath(val))
    for name in _PATH_LIST_ARGS:
        vals = getattr(args, name, None)
        if vals:
            setattr(args, name, [abspath(v) for v in vals])


def _opt_flag_spellings() -> set[str]:
    """Both spellings of every ``-*_opts`` flag (stage overrides plus -glm_opts)."""
    keys = (*config.STAGE_OPT_KEYS, "glm")
    return {f"-{k}_opts" for k in keys} | {f"-{k.replace('_', '-')}-opts" for k in keys}


def _glue_opt_values(argv: list[str]) -> list[str]:
    """Rewrite ``-moco_opts <value>`` as ``-moco_opts=<value>``.

    Option strings for these flags are option-looking by nature. argparse only
    accepts a ``-``-leading value when it contains a space, so a single-flag
    override (``-nordic_opts '-save_numcomps'``) died with "expected one
    argument" while a multi-flag one sailed through. The ``=`` form is exempt
    from that check."""
    flags = _opt_flag_spellings()
    out: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] in flags and i + 1 < len(argv):
            out.append(f"{argv[i]}={argv[i + 1]}")
            i += 2
        else:
            out.append(argv[i])
            i += 1
    return out


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(_glue_opt_values(raw_argv))
    _absolutize_inputs(args)
    recipe = config.RECIPES.get(args.recipe, {}) if args.recipe else {}

    def rget(field, default):
        return recipe.get(field, default)

    def eff(argval, field, default=False):
        return argval if argval is not None else rget(field, default)

    # per-stage opt overrides mutate the shared defaults for this run.
    for key in config.STAGE_OPT_KEYS:
        val = getattr(args, f"{key}_opts", None)
        if val is not None:
            config.DEFAULT_OPTS[key] = val

    subject = scan_subject(
        args.bids_dir,
        args.subject,
        sessions=args.session,
        tasks=args.task,
        fmap_pe_dir=args.fmap_pe_dir,
    )
    if not subject.sessions:
        print("ERROR: no matching BOLD runs found", file=sys.stderr)
        return 1

    _report_fmap_assignment(subject)

    anat_path, tpm_source = _resolve_anat(args)
    if anat_path is None:  # fall back to a scanned in-scope T1w
        for sess in subject.sessions:
            if sess.anat is not None:
                anat_path = str(sess.anat)
                break

    go_to_anat = False if args.no_anat else rget("go_to_anat", True)
    anat_nonlin = eff(args.anat_nonlin, "anat_nonlin")
    event_cols, event_cols_by_task = _resolve_event_cols(args)

    # TPM resolution: an explicit -tpm wins; else, with -suma, build one in-script
    # from FreeSurfer outputs (no ahead-of-time SPM TPM needed).
    tpm = args.tpm
    fs_tpm = False
    if anat_nonlin and not tpm and args.suma:
        fs_tpm = True
        tpm = "tpm_work/sub_specific_fsTPM.nii.gz"  # produced by the FS-TPM stage

    opt = Options(
        recipe=args.recipe,
        want_nordic=eff(args.want_nordic, "want_nordic"),
        phase_proc=args.phase_proc,
        noise_vols=args.noise_vols,
        slicetiming_method=_resolve_slicetiming(args, rget, subject),
        slicetiming_file=args.slicetiming,
        tr=args.tr,
        distortion=(False if args.no_distortion else rget("distortion", True)),
        run_glm=(False if args.no_glm else rget("run_glm", True)),
        glm_ortvec=_resolve_glm_ortvec(args, recipe),
        glm_opts=args.glm_opts or "",
        glm_spec_overwrite=args.glm_spec_overwrite,
        spec_event_cols=event_cols,
        sep_spec_event_cols=event_cols_by_task,
        locomoco=eff(args.locomoco, "locomoco"),
        xrun_nonlin=eff(args.xrun_nonlin, "xrun_nonlin"),
        xfmap_nonlin=eff(args.xfmap_nonlin, "xfmap_nonlin"),
        xses_nonlin=eff(args.xses_nonlin, "xses_nonlin"),
        xrun_nonlin_in_source=eff(args.xrun_nonlin_in_source, "xrun_nonlin_in_source"),
        xfmap_nonlin_in_source=eff(args.xfmap_nonlin_in_source, "xfmap_nonlin_in_source"),
        xses_nonlin_in_source=eff(args.xses_nonlin_in_source, "xses_nonlin_in_source"),
        ref_ses=args.ref_ses,
        fmap_ref=args.fmap_ref,
        ref_image=args.ref_image,
        go_to_anat=go_to_anat,
        final_dxyz=args.final_dxyz,
        anat_nonlin=anat_nonlin,
        # -ref_image is the answer for every level; -anat_source overrides it for
        # the anat step alone.
        anat_source=args.anat_source or args.ref_image or "grandmean",
        anat_nonlin_input=args.anat_nonlin_input,
        anat_path=anat_path if go_to_anat else None,
        moco_ref=args.moco_ref,
        grand_reference=args.grand_reference,
        grand_reference_nonlin=args.grand_reference_nonlin,
        tpm=tpm,
        tpm_source=tpm_source,
        fs_tpm=fs_tpm,
        suma_dir=args.suma,
        ref_file=args.ref_file,
        ref_transforms=args.ref_transforms,
        ref_anat=args.ref_anat,
        events=args.events,
    )
    opt.nordic_save_resid = args.nordic_save_resid  # emitter reads via getattr
    opt.batch_overwrite = args.batch_overwrite
    opt.qc = not args.no_qc
    # Output formats: keep the Options (config) default unless the user set one.
    if args.fmt is not None:
        opt.fmt = args.fmt
    if args.final_fmt is not None:
        opt.final_fmt = args.final_fmt
    if args.glm_fmt is not None:
        opt.glm_fmt = args.glm_fmt

    _report_events(args, opt, subject)

    # Hard preflight: refuse to write a script that wouldn't run first try.
    errors, warnings = preflight(args, opt, anat_path, subject)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    if errors:
        print("ffs_autoproc: cannot generate a runnable script:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        print("nothing written.", file=sys.stderr)
        return 1

    plan = build_plan(subject, opt)

    # Absolute so OUT is pinned to where the user ran ffs_autoproc — moving the
    # script and running it elsewhere still writes outputs to the intended dir.
    work_dir = str(Path(args.work_dir or f"ffs_proc_sub-{subject.subject}.results").resolve())
    out_path = Path(args.out or f"proc_sub-{subject.subject}.sh")
    script = write_script(
        plan,
        work_dir,
        bids_root=args.bids_dir,
        script_stem=out_path.stem,
        invocation=shlex.join(["ffs_autoproc", *raw_argv]),
    )

    out_path.write_text(script)
    out_path.chmod(0o755)

    # The GLM's model is a file, not a command line: write it beside the script's
    # outputs so it can be reviewed (and edited) before anything runs.
    spec_rows = write_design_specs(plan, args.bids_dir, work_dir)

    tag = f" [{args.recipe}]" if args.recipe else ""
    print(f"wrote {out_path}{tag}  ({len(plan.runs)} run(s), ref session {plan.ref_session})")
    print(f"  working dir baked in: {work_dir}")
    for task, path, status in spec_rows:
        note = {
            "wrote": "design spec written — EDIT IT (contrasts, HRF) before the GLM stage",
            "kept": "design spec already exists, left untouched (-glm_spec_overwrite to replace)",
        }.get(status, status)
        print(f"  task-{task}: {Path(path).name}  [{note}]")
    n_stim = len(list((Path(work_dir) / STIMULI_DIR).glob("*.tsv")))
    if n_stim:
        print(f"  {n_stim} events TSV(s) copied to {STIMULI_DIR}/ (what the specs name)")
    print("  read it, edit it, then run it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
