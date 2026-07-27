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
import shutil
import sys
from pathlib import Path

from fastfuncstuff.autoproc import config
from fastfuncstuff.autoproc.bids import BoldRun, find_events, scan_subject
from fastfuncstuff.autoproc.emit import write_script
from fastfuncstuff.autoproc.glm import write_design_specs
from fastfuncstuff.autoproc.plan import Options, build_plan


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
        "-anat_source",
        "-anat-source",
        choices=["grandmean", "ref_fmap", "mean_fmap"],
        default="grandmean",
        help="EPI image the anat alignment uses (all share the reference-fmap grid): "
        "grandmean (best SNR, only option w/o fieldmaps) | ref_fmap (reference "
        "fieldmap's undistorted mean; sharpest, defines the space) | mean_fmap "
        "(ref_fmap averaged with the other groups' aligned means)",
    )
    g.add_argument(
        "-anat_nonlin_input",
        "-anat-nonlin-input",
        choices=["grandmean", "ref_fmap", "mean_fmap", "blipfor", "blip_pair"],
        default="grandmean",
        help="ffs_segment input: grandmean (works w/o fieldmap) | ref_fmap | "
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
        help="integrate STC into the final resample | first (before moco) | none",
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
    for key in ("moco", "locomoco", "blip", "xrun", "xfmap", "xses", "anat", "nwarp", "unwrap"):
        g.add_argument(f"-{key}_opts", f"-{key}-opts", metavar="STR", help=_opt_help(key))
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
    # Events: not a hard error (preprocessing still runs), but warn per task so
    # the user knows the GLM will fail without them.
    if opt.run_glm and not opt.events:
        tasks = {r.task for s in subject.sessions for r in s.bold_runs}
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
            print(
                f"    fmap-{fg.fmap_id}  t={_fmt_time(fg.acq_time)}  →  [{items}]",
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    recipe = config.RECIPES.get(args.recipe, {}) if args.recipe else {}

    def rget(field, default):
        return recipe.get(field, default)

    def eff(argval, field, default=False):
        return argval if argval is not None else rget(field, default)

    # per-stage opt overrides mutate the shared defaults for this run.
    for key in ("moco", "locomoco", "blip", "xrun", "xfmap", "xses", "anat", "nwarp", "unwrap"):
        val = getattr(args, f"{key}_opts", None)
        if val is not None:
            config.DEFAULT_OPTS[key] = val

    subject = scan_subject(args.bids_dir, args.subject, sessions=args.session, tasks=args.task)
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
        slicetiming_method=args.slicetiming_method or rget("slicetiming_method", "integrate"),
        distortion=(False if args.no_distortion else rget("distortion", True)),
        run_glm=(False if args.no_glm else rget("run_glm", True)),
        glm_ortvec=_resolve_glm_ortvec(args, recipe),
        glm_opts=args.glm_opts or "",
        glm_spec_overwrite=args.glm_spec_overwrite,
        locomoco=eff(args.locomoco, "locomoco"),
        xrun_nonlin=eff(args.xrun_nonlin, "xrun_nonlin"),
        xfmap_nonlin=eff(args.xfmap_nonlin, "xfmap_nonlin"),
        xses_nonlin=eff(args.xses_nonlin, "xses_nonlin"),
        ref_ses=args.ref_ses,
        fmap_ref=args.fmap_ref,
        go_to_anat=go_to_anat,
        final_dxyz=args.final_dxyz,
        anat_nonlin=anat_nonlin,
        anat_source=args.anat_source,
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
    script = write_script(plan, work_dir, bids_root=args.bids_dir, script_stem=out_path.stem)

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
    print("  read it, edit it, then run it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
