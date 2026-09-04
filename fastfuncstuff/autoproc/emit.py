"""Bash emitter: turn a :class:`~fastfuncstuff.autoproc.plan.Plan` into a
readable, resumable pipeline script.

Design: Python owns all the conditional *policy* (which run needs which
transform, in what order) and bakes the result into per-run **data arrays** at
the top of the script. The stage blocks are then simple, legible bash loops
over those arrays — no nested conditional warp-chain assembly in bash (the part
that made the hand-written AFNI generators unreadable). The composed transform
chain for every run is emitted verbatim into ``CHAIN[...]`` so a human can read
exactly what will be applied.

Every stage guards each output with ``[ -f "$OUTF" ] && continue`` and honours a
coarse ``skip_<stage>=0/1`` toggle, so the script is fully restartable.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

from fastfuncstuff.autoproc import config
from fastfuncstuff.autoproc.bids import find_events
from fastfuncstuff.autoproc.naming import STAGE_NUMBERS, NameKey, coord, stem
from fastfuncstuff.autoproc.plan import (
    Plan,
    PlanRun,
    effective_anat_source,
    ref_anchor,
    sbref_chain,
    session_fmap_ids,
    session_ref_mode,
)

_TOOLS = [
    "ffs_nordic",
    "ffs_slicetime",
    "ffs_moco",
    "ffs_locomoco",
    "ffs_blipflip",
    "ffs_util_b0fmap",
    "ffs_allineate",
    "ffs_segment",
    "ffs_nwarp",
    "ffs_util_3dmath",
    "ffs_util_autobox",
    "ffs_util_resample",
    "ffs_util_automask",
    "ffs_reml",
]


# ---------------------------------------------------------------------------
# key + stem helpers
# ---------------------------------------------------------------------------


def _key(pr: PlanRun) -> str:
    b = pr.bold
    return f"{b.session or '-'}:{b.task}:{b.run or '-'}"


def _frag(pr: PlanRun) -> str:
    """The per-run coordinate fragment (``ses-06.task-X.run-010``) — the SINGLE
    source of truth for run filenames. Emitted per run as ``FRAG[k]`` so the bash
    loops build exactly the same names the Python data table / CHAIN reference."""
    b = pr.bold
    return coord(NameKey("moco", session=b.session, task=b.task, run=b.run or None))


def _run_stem(pr: PlanRun, label: str, src: str | None = None) -> str:
    b = pr.bold
    return stem(NameKey(label, session=b.session, task=b.task, run=b.run or None, src=src))


def _fmap_stem(pr: PlanRun, label: str) -> str:
    fid = pr.fmap.fmap_id if pr.fmap else "x"
    return stem(NameKey(label, session=pr.bold.session, fmap=fid))


def _ses_stem(pr: PlanRun, label: str) -> str:
    return stem(NameKey(label, session=pr.bold.session))


def _phase_on(plan: Plan) -> bool:
    return bool(getattr(plan.options, "phase_proc", False))


def _nordic_mag(plan: Plan) -> str:
    """The NORDIC magnitude output. With -phase_proc it is written as plain gzip
    (ROMEO reads the matching phase file, and ROMEO cannot read .zst)."""
    fmt = "$PHASE_FMT" if _phase_on(plan) else "$FMT"
    return "stage00.nordic.${FRAG[$k]}.nii" + fmt


def _nordic_phase() -> str:
    """The NORDIC phase output — ffs_nordic appends ``_phase`` to the -prefix stem
    whenever phase input was used; that name is the tool's, not ours."""
    return "stage00.nordic.${FRAG[$k]}_phase.nii$PHASE_FMT"


def _phase_file(label: str, part: bool = True) -> str:
    """A per-run phase-side working file, e.g. ``stage00.unwrap.<frag>.part-phase``.
    Built in bash from ``${FRAG[$k]}`` so it matches the Python-side stems; ``part``
    is False for the magnitude member of a pair (the trimmed copies)."""
    tail = ".part-phase" if part else ""
    return f"stage{STAGE_NUMBERS[label]:02d}.{label}." + "${FRAG[$k]}" + f"{tail}.nii$PHASE_FMT"


def _unwrapped(label: str = "unwrap") -> str:
    return _phase_file(label)


def _phase_source(plan: Plan) -> str:
    """The phase series entering the final resample: the tshifted phase when slice
    timing ran up front (it must get the same temporal shift as its magnitude),
    else the unwrapped phase straight out of stage00."""
    if plan.options.slicetiming_method == "first":
        return _unwrapped("tshift")
    return _unwrapped()


def _tpattern(plan: Plan) -> str:
    """The ``-tpattern`` value: a single user-supplied slice-timing file for every
    run (``-slicetiming``), else that run's BIDS sidecar."""
    stf = plan.options.slicetiming_file
    return shlex.quote(str(Path(stf).resolve())) if stf else '"${JSON[$k]}"'


def _sidecar(nifti: Path) -> Path:
    return nifti.parent / (re.sub(r"\.(nii\.gz|nii|nii\.zst)$", "", nifti.name) + ".json")


def _split_flags(opts: str) -> list[str]:
    """Split an option string into per-flag tokens so each lands on its own line:
    ``'-rigid -cost lpa -autoweight'`` → ``['-rigid', '-cost lpa', '-autoweight']``.
    A token starting with ``-`` opens a new group; following values attach to it."""
    groups: list[str] = []
    for t in opts.split():
        if t.startswith("-") or not groups:
            groups.append(t)
        else:
            groups[-1] += " " + t
    return groups


def _nl_tool(opt, stage: str) -> str:
    """The command that estimates one nonlinear stage, per that stage's backend.

    The engines are interchangeable to a stage -- they all take -matrix,
    -warp_prefix and -batch -- so which one runs is a per-stage setting rather
    than something the emitter knows. What differs between them is the op string,
    which travels separately (config.nl_stage_opts / -<stage>_opts)."""
    return config.nl_command(getattr(opt, f"{stage}_backend", config.DEFAULT_NL_BACKEND))


def _ffs(tool: str, parts: list[str], indent: str = "  ") -> str:
    """Render an ffs_* call one flag per line with ``\\`` continuations, indented.
    ``parts`` are per-flag tokens (mix of literals and bash ``${...}``)."""
    flat = [p for p in parts if p]  # drop empties (e.g. a conditionally-absent flag)
    lines = [f"{indent}{tool}", *[f"{indent}  {p}" for p in flat]]
    return " \\\n".join(lines)


def _manifest_line(var: str, parts: list[str], indent: str = "  ") -> str:
    """A ``printf`` that appends one run's arguments to the batch manifest ``$var``.

    ``parts`` are the same per-flag tokens :func:`_ffs` takes, so a batched stage
    and a solo call are written from one list. The inner double quotes are escaped
    because the whole line is emitted inside a double-quoted printf argument —
    paths still expand ($FMT, ${FRAG[$k]}) but stay quoted for the tool."""
    flat = " ".join(p for p in parts if p).replace('"', '\\"')
    return f'{indent}printf \'%s\\n\' "{flat}" >> "${var}"'


def _batch_launch(tool: str, var: str, skip_var: str, indent: str = "") -> str:
    """One batched launch of ``tool`` over the manifest in ``$var``.

    Empty manifest → no call (the tools exit(1) on an empty -batch, and a stage
    can legitimately have nothing to do: single session, or every run its own
    anchor). ``skip_var`` is the stage's skip toggle, passed through as
    -batch_skip so a re-run only pays for jobs that are still missing outputs."""
    return (
        f'{indent}if [ -s "${var}" ]; then\n'
        f'{indent}  batch_skip=(); [ "${skip_var}" -eq 1 ] && batch_skip=(-batch_skip)\n'
        f'{indent}  {tool} -batch "${var}" "${{batch_skip[@]}}" -device "$DEVICE"\n'
        f"{indent}fi"
    )


def _math_batch(var: str, path: str, jobs: list[list[str]], indent: str = "") -> list[str]:
    """A manifest + one batched ``ffs_util_3dmath`` for a group of math jobs.

    Reductions are a few seconds of I/O each around ~2s of interpreter+torch
    startup, so a loop of them is nearly all fixed cost. ``jobs`` are the same
    per-flag token lists :func:`_ffs` takes, minus ``-device`` (the batch picks
    the device once)."""
    if not jobs:
        return []
    out = [f'{indent}{var}="{path}"; : > "${var}"']
    out += [_manifest_line(var, parts, indent=indent) for parts in jobs]
    out.append(
        f'{indent}if [ -s "${var}" ]; then ffs_util_3dmath -batch "${var}" -device "$DEVICE"; fi'
    )
    return out


def _anat_lin_files(opt) -> list[str]:
    """The anat-matrix link(s) at the chain head, per reference mode:
    * explicit ref_file → the user's ``-ref_transforms`` (nwarp order), if any;
    * -grand_reference DIR → the borrowed ``{DIR}/stage09.anat.aff12.1D``;
    * own anat → the locally-computed ``stage09.anat.aff12.1D``.
    """
    if opt is not None and opt.ref_file is not None:
        return list(opt.ref_transforms or [])
    anat = stem(NameKey("anat")) + ".aff12.1D"
    gr = getattr(opt, "grand_reference", None) if opt is not None else None
    return [f"{gr.rstrip('/')}/{anat}"] if gr else [anat]


# The pre-chain tokens: run-native → the session's common grid. Applied on their
# own to a run's lane image they build its "runmean" (see stage07). With
# fieldmaps the common grid is the reference fmap's undistorted space and all
# five tokens can appear; without, only the wxrun pair does and the grid is the
# session's anchor run.
_PRE_CHAIN_TOKENS = ("xfmap_nl", "xfmap_lin", "blip_half", "wxrun_nl", "wxrun_lin")


def _nl_source_args(in_source: bool, aligned: str, native: str, matrix: str) -> list[str]:
    """The -source (and maybe -matrix) args for a nonlinear refinement stage.

    Default: refine the image the linear stage already produced, on the base grid.
    ``in_source``: hand the backend the linear stage's matrix and the *un*-allineated
    input instead, and it inverts the matrix, pulls the base onto the source's grid and
    solves there -- the source is never resampled. The resulting warp lives in source
    space and therefore acts on the data BEFORE its affine, which is why plan.py swaps
    the pair in the chain. Both halves have to agree or the chain silently composes in
    the wrong order; they are driven by the same flag for exactly that reason.
    """
    if not in_source:
        return [f'-source "{aligned}"']
    return [f'-source "{native}"', f'-matrix "{matrix}"']


def _token_files(pr: PlanRun, tok: str, fmt: str, opt) -> list[str]:
    """The concrete file(s) one warp-chain token resolves to. Single source of
    truth for both the full chain and the fmap sub-chain (no drift)."""
    if tok == "anat_lin":
        return _anat_lin_files(opt)
    if tok == "xref_nl":
        return [stem(NameKey("xref")) + f"_nl_WARP{fmt}"]
    if tok == "xref_lin":
        return [stem(NameKey("xref")) + ".aff12.1D"]
    if tok == "anat_nl":
        return [stem(NameKey("nlanat")) + f"_invwarp{fmt}"]
    if tok == "xses_nl":
        return [_ses_stem(pr, "xses") + f"_nl_WARP{fmt}"]
    if tok == "xses_lin":
        return [_ses_stem(pr, "xses") + ".aff12.1D"]
    if tok == "xfmap_nl":
        return [_fmap_stem(pr, "xfmap") + f"_nl_WARP{fmt}"]
    if tok == "xfmap_lin":
        return [_fmap_stem(pr, "xfmap") + ".aff12.1D"]
    if tok == "blip_half":
        return [_fmap_stem(pr, "blip") + f"_warp{fmt}"]
    if tok == "wxrun_nl":
        return [_run_stem(pr, "xrun") + f"_nl_WARP{fmt}"]
    if tok == "wxrun_lin":
        return [_run_stem(pr, "xrun") + ".aff12.1D"]
    if tok == "locomoco":
        return [_run_stem(pr, "nlmoco") + f"_warp{fmt}"]
    if tok == "moco":
        return [_run_stem(pr, "moco") + ".aff12.1D"]
    return []


def chain_files(pr: PlanRun, fmt: str, opt=None, tokens: list[str] | None = None) -> list[str]:
    """Resolve a run's warp chain into filenames (nwarp-apply order).

    ``tokens`` overrides which links to resolve — used for the SBRef chain
    (``plan.sbref_chain``), which is the run's chain minus the within-run motion
    tokens. The files themselves are shared: an SBRef and its BOLD are corrected
    by exactly the same transforms from the fieldmap level up.
    """
    resolved: list[str] = []
    for tok in pr.warp_chain if tokens is None else tokens:
        resolved.extend(_token_files(pr, tok, fmt, opt))
    return resolved


def _pre_chain(pr: PlanRun, fmt: str) -> list[str]:
    """Just the pre-chain tokens of a run's chain (xfmap∘blip∘wxrun) — applied to a
    lane image to land it on the session's common grid (the runmean). Empty for
    the anchor run of a no-fieldmap session, which defines that grid."""
    out: list[str] = []
    for tok in pr.warp_chain:
        if tok in _PRE_CHAIN_TOKENS:
            out.extend(_token_files(pr, tok, fmt, None))
    return out


def _jac_spec(pr: PlanRun) -> str:
    """``AXIS:FIELDMAP`` for ``ffs_nwarp -jac``, or ``""`` for a run with no
    fieldmap in its chain.

    Both fieldmap tools emit a *geometry-only* displacement field: applying it
    unwarps the image but leaves the signal pile-up at compression edges, because
    the voxel that got squeezed still holds the sum of what was squeezed into it.
    The Jacobian ``1 + d(disp_pe)/d(pe)`` is what makes it quantitatively correct
    (FSL applytopup --method=jac). The stage04 mean carries it already (blipflip
    applies it to its own pair; the b0 branch asks ffs_nwarp for it); every
    *later* application of the same warp to raw data — the runmeans and the final
    resample — has to ask for it explicitly, or those two images disagree about
    intensity at exactly the places distortion was worst.

    The fieldmap is named (not left to ffs_nwarp's lone-static-warp auto-detect):
    the chain also carries locomoco's per-frame PE warp, and only the fieldmap's
    Jacobian belongs here.
    """
    if pr.fmap is None or "blip_half" not in pr.warp_chain:
        return ""
    axis = "".join(c for c in _fmap_pe(pr) if c.isalpha()) or "j"
    return f"{axis}:{_fmap_stem(pr, 'blip')}_warp.nii$FMT"


def _blip_forward(blip_stem: str, is_b0: bool) -> str:
    """A fieldmap group's undistorted FORWARD image — the space its runs land in,
    and therefore the target every transform of data is estimated against.

    For a pepolar pair that is sub-brick 0 of ``_unwarped`` (the corrected blip-up
    image), NOT the ``_mean``. The mean averages the corrected up and down images,
    and no data ever occupies it: a run's chain ends at the corrected *forward*
    space. Averaging also cancels half of each side's residual distortion, so two
    groups' means agree with each other far better than the images the transform
    actually has to serve — which made the mean a flattering, and wrong, target
    (measured on sub-3001: cross-group corrected-up agreement 0.906 for the
    mean-estimated rigid+nonlinear transform vs 0.927 for a rigid fit estimated
    on the forward images themselves).

    A GRE group has no pair to average: its ``_mean`` is one warped run rep, i.e.
    already the forward image, so that is what it returns.
    """
    return f"{blip_stem}_mean.nii$FMT" if is_b0 else f"{blip_stem}_unwarped.nii$FMT[0]"


def _fmap_forward(pr: PlanRun) -> str:
    """This run's OWN fmap group's undistorted forward image (xfmap source)."""
    assert pr.fmap is not None
    return _blip_forward(_fmap_stem(pr, "blip"), pr.fmap.is_b0)


def _ref_blip_forward(pr: PlanRun) -> str:
    """The reference fmap group's undistorted forward image — the image that
    defines the common space the runmeans (and cross-fmap alignment) land on."""
    return _blip_forward(
        stem(NameKey("blip", session=pr.bold.session, fmap=pr.ref_fmap_id)), pr.ref_fmap_is_b0
    )


def _shell_path(path: str) -> str:
    """``path`` with any AFNI sub-brick selector stripped — what a shell ``[ -f ]``
    or ``cp`` needs, since only the tools parse ``file.nii[0]``."""
    return path.split("[", 1)[0]


# ---------------------------------------------------------------------------
# lanes
#
# Parallel image lineages run the length of the mean pyramid, describing the same
# anatomy in the same spaces. One set of transforms is estimated (from the
# primary lane) and every lane is resampled by it, so any two lanes are directly
# comparable at every level:
#
#   LANE_MEAN   each run's motion-corrected BOLD mean (what this pipeline always
#               built). Multiband contrast, high SNR — but motion carries edge
#               voxels out of the FoV, and averaging a zero into those frames
#               DIMS the very edges registration needs.
#   LANE_SBREF  each run's SBRef. Single-band, no slice-leakage artefact, sharper
#               tissue contrast, ONE interpolation deep because the SBRef is the
#               moco base and needs no within-run transform at all — and, being a
#               single volume, it has no coverage deficit to repair.
#   LANE_MAX    each run's motion-corrected temporal MAX: the union of every voxel
#               imaged in any frame. Keeps the edges the mean loses, and it is the
#               lineage that COMPOSES — max-of-maxes up the pyramid keeps the most
#               brain in the most places, where a mean of means only ever loses.
#   LANE_MIN    the counterpart: 0 wherever any frame lost the voxel, so
#               min-of-mins is exactly the region with complete data everywhere.
#               Not a registration lane — a coverage floor, carried for masking.
#
# Compositing is per lane (``_lane_reduce``): means average, maxes take the max,
# mins take the min. Averaging a lane of mins would be meaningless.
# ---------------------------------------------------------------------------

LANE_MEAN = "mean"
LANE_SBREF = "sbref"
LANE_MAX = "max"
LANE_MIN = "min"

# How each lane composites when several of its images are combined into the level
# above (the ffs_util_3dmath reduction flag).
_LANE_REDUCE = {LANE_MEAN: "-mean", LANE_SBREF: "-mean", LANE_MAX: "-max", LANE_MIN: "-min"}


def _src(lane: str) -> str | None:
    """The ``src-`` naming token for a lane (None = the mean lane, untokenised so
    every pre-lane filename is unchanged)."""
    return None if lane == LANE_MEAN else lane


def _lanes(plan: Plan) -> tuple[str, ...]:
    """Lanes to build, primary first."""
    coverage = (LANE_MAX, LANE_MIN)
    return ((LANE_SBREF,) if plan.use_sbref else ()) + coverage + (LANE_MEAN,)


def _primary_lane(plan: Plan) -> str:
    """The lane that *estimates* the cross-run and cross-session transforms.

    The SBRef leads where it exists: it is a single full-FoV volume, so the edge
    loss LANE_MAX repairs never happened to it, and it is sharper besides. With no
    SBRef the max leads — same geometry as the mean, but with the moving edges
    still in it, which is exactly what the cost function needs at the FoV rim.
    """
    return LANE_SBREF if plan.use_sbref else LANE_MAX


def _lane_reduce(lane: str) -> str:
    return _LANE_REDUCE[lane]


def _lane_nwarp_flags(lane: str) -> list[str]:
    """The shared nwarp defaults with this lane's interpolation override applied.

    The min lane's information is its zero boundary; wsinc5 rings across it and
    smears the very edge the map exists to mark, so it resamples linearly. The
    override REPLACES the default ``-interp`` rather than trailing it — a line
    carrying two ``-interp`` flags parses fine and reads like a bug.
    """
    flags = _split_flags(config.DEFAULT_OPTS["nwarp"])
    if lane != LANE_MIN:
        return flags
    return ["-interp linear" if f.startswith("-interp ") else f for f in flags]


def _run_level(pr: PlanRun, lane: str) -> str:
    """A run's native-space image for this lane, in the run's post-moco space."""
    if lane == LANE_SBREF:
        return str(pr.bold.sbref_path)
    return _moco_reduction(pr, lane)


def _run_level_var(lane: str) -> str:
    """``_run_level`` as a bash data-table reference (for the stage loops)."""
    if lane == LANE_SBREF:
        return "${SBREF[$k]}"
    return f"${{MOCO_{lane.upper()}[$k]}}"


def _common_grid(pr: PlanRun, session_first: PlanRun, lane: str) -> str:
    """The grid this run's runmean lands on — the space its session averages in.

    In a session with fieldmaps that is the reference fmap's undistorted forward
    image (see :func:`_blip_forward`), for
    EVERY run of the session: a run with no fieldmap of its own still has to land
    there, or its runmean sits on a different grid from its siblings' and the
    session mean is averaging incompatible images (bug of record). Without
    fieldmaps the grid is the session's anchor run. Every lane's image shares a
    run's native geometry (an SBRef is acquired on its BOLD's grid), so this is one
    grid however many lineages ride it.
    """
    if pr.ref_fmap_id is not None:
        return _ref_blip_forward(pr)
    return _run_level(session_first, lane)


def _runmean(pr: PlanRun, lane: str = LANE_MEAN) -> str:
    """This run's lane image resampled onto the reference-fmap grid (sesmean
    input). Still called ``runmean`` in the SBRef lane: the label names the
    pyramid *level*, ``src-`` names the lineage."""
    return _run_stem(pr, "runmean", src=_src(lane)) + ".nii$FMT"


def _sesmean(session: str | None, lane: str = LANE_MEAN) -> str:
    """One session's mean: the average of its aligned run images, in that session's
    own space (the xses source for non-reference sessions)."""
    return stem(NameKey("sesmean", session=session, src=_src(lane))) + ".nii$FMT"


def _grandmean(lane: str = LANE_MEAN) -> str:
    """THE grandmean: every run of every session, after cross-session alignment.
    Built in stage08 (not 07) because it cannot exist until xses has put the
    sessions in a single space."""
    return stem(NameKey("grandmean", src=_src(lane))) + ".nii$FMT"


def _ref_marker(key: NameKey, source: str, note: str) -> str:
    """A QC-only copy of the image that *is* a level's reference, named like the
    level's other outputs plus ``role-ref``. Alignment stages skip their reference
    (it maps to itself), which otherwise leaves a browsable gap — N-1 files where
    the reader expects N. Carries ``_lin`` like the images it stands in for (the
    identity is a linear transform); has no ``_nl`` counterpart, by definition."""
    dst = stem(key) + "_lin.nii$FMT"
    if "[" in source:
        # Only the tools parse a sub-brick selector, so the extraction has to be a
        # tool call — cp would look for a file with brackets in its name.
        copy = _ffs(
            "ffs_util_3dmath",
            [f'-input "{source}"', "-tcat", f'-prefix "{dst}"', '-device "$DEVICE"'],
        )
        return f'# {note}\nif [ ! -f "{dst}" ]; then\n{copy}\nfi'
    return f'# {note}\n[ -f "{dst}" ] || cp -f "{source}" "{dst}"'


def _sessions(plan: Plan) -> list[str | None]:
    """Session labels in scan order (the None-session case yields ``[None]``)."""
    out: list[str | None] = []
    for pr in plan.runs:
        if pr.bold.session not in out:
            out.append(pr.bold.session)
    return out


def _fmapmean(plan: Plan, session: str | None) -> str:
    """The ``mean_fmap`` image for a session: the mean of its fmap group means."""
    return stem(NameKey("fmapmean", session=session)) + ".nii$FMT"


def _anchor_session(plan: Plan) -> str | None:
    anchor = ref_anchor(plan)
    return anchor.bold.session if anchor else None


def _session_ref_fmap_image(plan: Plan, session: str | None) -> str:
    """That session's reference fieldmap's undistorted forward image — the image
    that DEFINES its common grid (see :func:`_blip_forward`)."""
    ids = session_fmap_ids(plan, session)
    fid = ids[0] if ids else None
    is_b0 = any(
        pr.fmap is not None and pr.fmap.fmap_id == fid and pr.fmap.is_b0
        for pr in plan.runs
        if pr.bold.session == session
    )
    return _blip_forward(stem(NameKey("blip", session=session, fmap=fid)), is_b0)


def _fmapmean_inputs(plan: Plan, session: str | None) -> list[str]:
    """The per-group images averaged into one session's ``mean_fmap``, all already
    on that session's reference-fmap grid: the reference group's own forward image,
    plus each non-ref group's xfmap-aligned forward image (nonlinear result when
    ``-xfmap_nonlin`` ran). One lineage throughout — mixing a pair *mean* into an
    average of forward images would blur the composite with a different contrast."""
    ids = session_fmap_ids(plan, session)
    if not ids:
        return []
    nl = plan.options.xfmap_nonlin
    files = []
    for fid in ids:
        if fid == ids[0]:  # session_fmap_ids puts the reference group first
            files.append(_session_ref_fmap_image(plan, session))
        else:
            st = stem(NameKey("xfmap", session=session, fmap=fid))
            files.append(f"{st}_nl.nii$FMT" if nl else f"{st}_lin.nii$FMT")
    return files


def _fmapmean_sessions(plan: Plan) -> list[str | None]:
    """Sessions needing a ``mean_fmap`` built in stage05: the anat step's (or
    ffs_segment's) request, plus any session whose ``-ref_image`` representative
    for the cross-session alignment is ``mean_fmap``."""
    opt = plan.options
    want: list[str | None] = []
    if opt.go_to_anat:
        requests = [opt.anat_source]
        if opt.anat_nonlin:
            requests.append(opt.anat_nonlin_input)
        if any(effective_anat_source(plan, r) == "mean_fmap" for r in requests):
            want.append(_anchor_session(plan))
    if plan.multi_session:
        for ses in _sessions(plan):
            if session_ref_mode(plan, ses) == "mean_fmap" and ses not in want:
                want.append(ses)
    return want


def _anat_source_image(plan: Plan, requested: str | None = None) -> str:
    """The EPI-contrast image the anat step aligns (or ffs_segment consumes).

    All four choices sit on the reference-fmap grid, so the warp chain is
    identical whichever is picked — only the image content differs:
      grandmean  every run's moco mean averaged; best SNR, N interpolations deep,
                 and the only option without fieldmaps or SBRefs.
      sbmean     every run's SBRef averaged, same spaces as the grandmean: one
                 interpolation instead of two, single-band contrast, no multiband
                 slice-leakage — the best cross-modal ``lpc`` source there is when
                 SBRefs exist, and unlike ref_fmap it uses ALL the data.
      ref_fmap   the reference group's undistorted forward image; one interpolation,
                 sharpest, SBRef contrast — the image that *defines* the space.
      mean_fmap  ref_fmap averaged with the other groups' xfmap-aligned means;
                 ref_fmap's provenance with more SNR (multi-fieldmap only).
    """
    mode = effective_anat_source(plan, requested)
    if mode == "sbmean":
        return _grandmean(LANE_SBREF)
    if mode == "ref_fmap":
        anchor = ref_anchor(plan)
        if anchor is not None:
            return _ref_blip_forward(anchor)
    elif mode == "mean_fmap":
        return _fmapmean(plan, _anchor_session(plan))
    return _grandmean()


def _ref_image_file(plan: Plan, session: str | None, lane: str) -> str:
    """The image that represents ``session`` at the cross-session alignment.

    Same vocabulary as ``-anat_source``, one level down — see
    ``plan.session_ref_mode``. The fieldmap choices are lane-free (there is one
    fieldmap image, not one per lineage), so a lane only reaches this for the
    data-mean choices; that is what keeps every lane resampled by ONE transform.
    """
    mode = session_ref_mode(plan, session)
    if mode == "ref_fmap":
        return _session_ref_fmap_image(plan, session)
    if mode == "mean_fmap":
        return _fmapmean(plan, session)
    if mode == "sbmean":
        return _sesmean(session, LANE_SBREF)
    return _sesmean(session, lane)


# ---------------------------------------------------------------------------
# QC stacks
#
# Every alignment stage produces a SET of images that, if the stage worked, are
# in one space — the runs of a session on its common grid, the sessions in the
# reference session's space, every run's mean in the final output space. Each of
# those sets is concatenated along time into one 4-D ``stageNN.QC.*`` file, so
# the check is "scroll the time axis and watch for the brain to jump" instead of
# loading N files into a viewer and toggling them by hand.
#
# The grouping rule is exactly "these files SHOULD be voxel-for-voxel aligned",
# which is why the groups are not always one-per-stage: cross-run alignment in a
# multi-fieldmap session lands each group's runs on ITS forward image, so that
# stage gets one stack per fieldmap group, and the single per-session stack only
# appears at stage07 once the pre-chain has put them all on one grid. Where a
# level has a reference that maps to itself, it goes in as sub-brick 0 — the
# thing everything else was supposed to move onto.
#
# QC files are plain .nii.gz (never $FMT): they exist to be opened in a viewer,
# and stock AFNI/FSLeyes cannot read .nii.zst.
# ---------------------------------------------------------------------------

# What (file, label) pairs a QC stack is built from.
QCItems = list[tuple[str, str]]


def _qc_on(plan: Plan) -> bool:
    return bool(getattr(plan.options, "qc", True))


def _qc_stem(label: str, **coords) -> str:
    """``stageNN.QC.<label>[.<coords>]`` — the QC stack for a level.

    Same stage number and coordinate vocabulary as the images it stacks, with
    ``QC`` between them: an uppercase token sorts the stacks to the head of their
    stage in a directory listing, and ``ls *.QC.*`` is every one of them.
    """
    key = NameKey(label, **coords)
    c = coord(key)
    base = f"stage{STAGE_NUMBERS[label]:02d}.QC.{label}"
    return f"{base}.{c}" if c else base


def _qc_lanes(plan: Plan) -> tuple[str, ...]:
    """Lanes worth a QC stack: the anatomy-bearing ones.

    max/min are coverage maps resampled by the SAME transforms as the mean lane,
    so a stack of them shows the identical alignment twice — the extra files
    would only dilute the listing the QC exists to make readable.
    """
    return ((LANE_SBREF,) if plan.use_sbref else ()) + (LANE_MEAN,)


def _qc_call(out_stem: str, items: QCItems, indent: str = "") -> str:
    """One ``qc_tcat`` line, or "" when there is nothing to compare.

    A single-image group is dropped: a 1-volume "stack" answers no alignment
    question, and the stage's own output is already that file.
    """
    if len(items) < 2:
        return ""
    labels = " ".join(lab for _, lab in items)
    files = " ".join(f'"{f}"' for f, _ in items)
    return f'{indent}qc_tcat "{out_stem}.nii.gz" "{labels}" {files}'


def _qc_block(title: str, calls: list[str]) -> str:
    """A stage's QC lines under one echo, or "" when the stage has no groups.

    Ends with ``qc_flush`` so the stage's stacks are built by one batched
    ffs_util_3dmath instead of one process per stack."""
    live = [c for c in calls if c]
    if not live:
        return ""
    return "\n".join([f"echo {shlex.quote('== QC: ' + title + ' ==')}", *live, "qc_flush"])


def _qc_helper(plan: Plan, script_stem: str) -> str:
    """The ``qc_tcat`` / ``qc_flush`` shell functions, emitted once near the top.

    Missing inputs skip the stack rather than failing: a QC group can name a file
    an optional stage did not write (a nonlinear refinement that was turned off
    after a partial run, a fieldmap that failed), and `set -e` would take the
    whole pipeline down over an image nobody is going to analyse.
    """
    if not _qc_on(plan):
        return ""
    return f"""
# ============================ QC stacks =====================================
# qc_tcat OUT "LABELS..." INPUT... — concatenate images that SHOULD be aligned
# into one 4-D file. Scroll its time axis in a viewer: any jump between
# sub-bricks is a registration failure at that stage, and the sub-brick label
# names the run/session/fieldmap responsible. `ls *.QC.*` is all of them.
# qc_tcat only QUEUES the stack; the qc_flush at the end of each stage's QC
# block builds every stack queued since the last flush in ONE process. A stack
# is a few seconds of I/O around ~2s of interpreter+torch startup, so the loop
# was almost entirely fixed cost. Flushing per stage (not once at the end)
# keeps a stage's QC available before the next stage consumes its outputs.
skip_qc=1          # 1 = keep an existing QC stack; 0 = rebuild every one
qcbatch="{script_stem}_qcbatch.txt"; : > "$qcbatch"
qc_tcat() {{
  local out="$1" labs="$2"; shift 2
  [ "$skip_qc" -eq 1 ] && [ -f "$out" ] && return 0
  local f line
  for f in "$@"; do
    # ${{f%%[*}} drops any sub-brick selector: only the tools parse "file.nii[0]".
    [ -f "${{f%%[*}}" ] || {{ echo "  QC skip $(basename "$out"): missing $f"; return 0; }}
  done
  # $labs is deliberately unquoted: it is a space-separated label list.
  line="-tcat -overwrite -prefix \\"$out\\" -labels $labs -input"
  for f in "$@"; do line="$line \\"$f\\""; done
  printf '%s\\n' "$line" >> "$qcbatch"
}}
qc_flush() {{
  [ -s "$qcbatch" ] || return 0
  ffs_util_3dmath -batch "$qcbatch" -device "$DEVICE"
  : > "$qcbatch"
}}
"""


# ---------------------------------------------------------------------------
# per-run derived paths (computed in Python, emitted as data)
# ---------------------------------------------------------------------------


def _moco_reduction(pr: PlanRun, which: str) -> str:
    """One temporal reduction of a run's corrected series (locomoco's when enabled).

    Filenames match the tools' own output naming: ffs_moco writes the files we
    pass to ``-save_{mean,max,min}``; ffs_locomoco writes ``{stem}_locomoco_{which}``.
    """
    if pr.warp_chain and "locomoco" in pr.warp_chain:
        return _run_stem(pr, "nlmoco") + f"_locomoco_{which}.nii$FMT"
    return _run_stem(pr, "moco") + f"_{which}.nii$FMT"


def _moco_mean(pr: PlanRun) -> str:
    """The registration-target mean for a run."""
    return _moco_reduction(pr, LANE_MEAN)


def _xrun_base(pr: PlanRun, session_first: PlanRun, lane: str = LANE_MEAN) -> str | None:
    """Base image for this run's xrun alignment, or None if it needs no xrun.

    Fieldmap-anchored: the group's DISTORTED forward (blip_up) — blip_half
    undistorts *after* xrun in the chain, so xrun stays in distorted space. That
    forward image is already ``BoldRun.rep``, i.e. the SBRef when one exists, so
    the fieldmap path has always had an SBRef base; what the lane changes is the
    *source* it is matched against.
    Fieldmap session, run with no usable fieldmap (PE direction rules the
    reference field out): the reference group's UNdistorted forward image. Its chain has no
    blip_half, so this is the one base that still lands it on the session's common
    grid — cross-contrast (distorted source, undistorted base) but in the right
    space, which beats being correctly registered to the wrong one.
    No-fmap session: the session's first run, same lane as the source.
    """
    if "wxrun_lin" not in pr.warp_chain:
        return None
    if pr.fmap is not None:
        return pr.fmap_forward
    if pr.ref_fmap_id is not None:
        return _ref_blip_forward(pr)
    return _run_level(session_first, lane)  # first-run anchor


def _aligned_mean(pr: PlanRun, lane: str = LANE_MEAN) -> str:
    """The run's lane image in the session's common (sesmean) space.

    One rule for every run: push the lane image through the run's pre-chain
    (xfmap∘blip∘wxrun, whichever of those it has) onto the common grid — that is
    the runmean. A run with an empty pre-chain is the anchor and already sits in
    the common space, so it *is* its own aligned image.

    Both lanes ride the same pre-chain, which is what makes a runmean and its
    ``src-sbref`` sibling directly comparable.
    """
    if _pre_chain(pr, ".nii$FMT"):
        return _runmean(pr, lane)
    return _run_level(pr, lane)


def _first_by_session(plan: Plan) -> dict[str | None, PlanRun]:
    """First run of each session — the anchor a no-fieldmap session aligns to."""
    first: dict[str | None, PlanRun] = {}
    for pr in plan.runs:
        first.setdefault(pr.bold.session, pr)
    return first


# ---------------------------------------------------------------------------
# QC groups, one builder per stage. Each returns the stage's QC block, or "".
# ---------------------------------------------------------------------------


def _qc_xfmap(plan: Plan) -> str:
    """Per session: every fieldmap group's mean on that session's reference-fmap
    grid. Sub-brick 0 is the reference group's own undistorted forward image."""
    if not _qc_on(plan):
        return ""
    calls = []
    for ses in _sessions(plan):
        ids = session_fmap_ids(plan, ses)
        if len(ids) < 2:
            continue
        ref = (_session_ref_fmap_image(plan, ses), f"ref:fmap-{ids[0]}")
        for kind in ("lin", "nl") if plan.options.xfmap_nonlin else ("lin",):
            items: QCItems = [ref]
            for fid in ids[1:]:
                st = stem(NameKey("xfmap", session=ses, fmap=fid))
                items.append((f"{st}_{kind}.nii$FMT", f"fmap-{fid}"))
            calls.append(_qc_call(_qc_stem("xfmap", session=ses) + f"_{kind}", items))
    return _qc_block("cross-fmap (each group → the session reference fmap)", calls)


def _qc_xrun(plan: Plan) -> str:
    """Per alignment base: the runs that were aligned to it, plus the base itself.

    Grouped by base, not by session: in a multi-fieldmap session each group's runs
    land on ITS forward image, so they are only mutually comparable within a group
    until stage07 composes the rest of the pre-chain.
    """
    if not _qc_on(plan):
        return ""
    primary = _primary_lane(plan)
    first = _first_by_session(plan)
    kinds = ("lin", "nl") if plan.options.xrun_nonlin else ("lin",)
    groups: dict[tuple, QCItems] = {}
    for pr in plan.runs:
        base = _xrun_base(pr, first[pr.bold.session], primary)
        if base is None:
            continue  # this run IS the anchor; it enters as sub-brick 0 below
        fid = pr.fmap.fmap_id if pr.fmap is not None else None
        for kind in kinds:
            items = groups.setdefault((pr.bold.session, fid, base, kind), [(base, "base")])
            st = _run_stem(pr, "xrun", src=_src(primary))
            items.append((f"{st}_{kind}.nii$FMT", _frag(pr)))
    calls = [
        _qc_call(_qc_stem("xrun", session=ses, fmap=fid) + f"_{kind}", items)
        for (ses, fid, _base, kind), items in groups.items()
    ]
    return _qc_block("cross-run (runs → their alignment base)", calls)


def _qc_runmean(plan: Plan) -> str:
    """Per session, per lane: every run on that session's common grid. The first
    stack where all of a session's runs are directly comparable."""
    if not _qc_on(plan):
        return ""
    by_ses: dict[str | None, list[PlanRun]] = {}
    for pr in plan.runs:
        by_ses.setdefault(pr.bold.session, []).append(pr)
    calls = [
        _qc_call(
            _qc_stem("runmean", session=ses, src=_src(lane)),
            [(_aligned_mean(pr, lane), _frag(pr)) for pr in prs],
        )
        for lane in _qc_lanes(plan)
        for ses, prs in by_ses.items()
    ]
    return _qc_block("run means (all runs of a session, on its common grid)", calls)


def _qc_xses(plan: Plan) -> str:
    """Every session's representative image in the reference session's space."""
    if not _qc_on(plan) or not plan.multi_session:
        return ""
    primary = _primary_lane(plan)
    lane_tag = f".src-{_src(primary)}" if _src(primary) else ""
    ref = plan.ref_session
    nonref = [s for s in _sessions(plan) if s is not None and s != ref]
    calls = []
    for kind in ("lin", "nl") if plan.options.xses_nonlin else ("lin",):
        items: QCItems = [(_ref_image_file(plan, ref, primary), f"ref:ses-{ref}")]
        items += [
            (f"{stem(NameKey('xses', session=s))}{lane_tag}_{kind}.nii$FMT", f"ses-{s}")
            for s in nonref
        ]
        calls.append(_qc_call(_qc_stem("xses", src=_src(primary)) + f"_{kind}", items))
    return _qc_block("cross-session (each session → the reference session)", calls)


def _qc_grandmean(plan: Plan) -> str:
    """Every run of every session in grandmean space — the inputs to the grandmean,
    stacked. The one QC that covers the whole dataset before the anat step, and the
    place a cross-session failure shows up as a jump at a session boundary.

    Single session: grandmean space IS that session's common grid, so this stack
    would be stage07's per-session one under a second name. Skipped there.
    """
    if not _qc_on(plan) or not plan.multi_session:
        return ""
    calls = [
        _qc_call(
            _qc_stem("grandmean", src=_src(lane)),
            [(_gmrun_image(pr, lane), _frag(pr)) for pr in plan.runs],
        )
        for lane in _qc_lanes(plan)
    ]
    return _qc_block("grandmean inputs (every run, in one space)", calls)


def _qc_xref(plan: Plan) -> str:
    """This data's grandmean against the external reference it was aligned to."""
    opt = plan.options
    if not (_qc_on(plan) and opt.go_to_anat and opt.has_grand_ref):
        return ""
    items: QCItems = [("$REFGM_EXT", "ref")]
    for kind in ("lin", "nl") if opt.grand_reference_nonlin else ("lin",):
        items.append((f"stage09.xref_{kind}.nii$FMT", f"grandmean_{kind}"))
    return _qc_block("external reference", [_qc_call(_qc_stem("xref"), items)])


def _qc_anat(plan: Plan) -> str:
    """The EPI anchor on the anat grid, over the anat itself — the cross-modal
    alignment check, as the two images that are supposed to overlay."""
    if not (_qc_on(plan) and _own_anat(plan.options)):
        return ""
    src = effective_anat_source(plan)
    items: QCItems = [
        (_anat_box(), "anat"),
        (_al_anat(plan), f"{src}_al_anat"),
    ]
    if plan.options.anat_nonlin:
        # Both states, so the nonlinear link is visible as the step between them —
        # and so the one the final grid is built from is the one you can see.
        items.append((_al_anat_nl(plan), f"{src}_al_anat_nl"))
    return _qc_block("EPI → anat", [_qc_call(_qc_stem("anat"), items)])


def _qc_final(plan: Plan) -> str:
    """THE final QC: every run's mean, in output space, in one file.

    This is the whole dataset as the GLM will see it. Sub-brick 0 is the
    warpmaster, so the stack also answers "did the data land on the grid it was
    supposed to". Anything that moves between sub-bricks here was not fixed by any
    earlier stage, whatever those stages' own QC stacks showed.
    """
    if not _qc_on(plan):
        return ""
    wm = ("stage10.warpmaster.nii$FMT", "warpmaster")
    calls = [
        _qc_call(
            _qc_stem("final"),
            [wm]
            + [(f"mean_stage10.final.{_frag(pr)}.nii$FINAL_FMT", _frag(pr)) for pr in plan.runs],
        )
    ]
    if plan.use_sbref:
        calls.append(
            _qc_call(
                _qc_stem("final", src=LANE_SBREF),
                [wm]
                + [
                    (f"stage10.final.{_frag(pr)}.src-sbref.nii$FINAL_FMT", _frag(pr))
                    for pr in plan.runs
                ],
            )
        )
    return _qc_block("final space (per-run means — the one to look at)", calls)


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------


def _skip_default(opt) -> int:
    """Default value for the batched stages' skip toggle: 1 (skip complete runs,
    the resumable default) unless -batch_overwrite forces a full re-process."""
    return 0 if opt.batch_overwrite else 1


def _timing_header_note(plan: Plan) -> str:
    """Header lines for the timing decisions that are not obvious from the flags:
    where slice timing came from, and whether the TR was overridden."""
    opt = plan.options
    out = ""
    if opt.slicetiming_method == "none":
        out += "\n#   slice timing      : OFF (no timing available, or -slicetiming_method none)"
    elif opt.slicetiming_file:
        out += f"\n#   slice timing      : {opt.slicetiming_file} (same for every run)"
    if opt.tr is not None:
        out += f"\n#   TR                : {opt.tr:g} s for every run (-TR; overrides the header)"
    return out


def _sbref_header_note(plan: Plan) -> str:
    """What the SBRef lane is doing in this script, for the header block."""
    if not plan.use_sbref:
        return "  (no SBRef for every run, or -moco_ref is not sbref)"
    return (
        "\n#     SBRefs are the moco base, so they need no within-run transform and\n"
        "#     estimate the cross-run/cross-session alignment. `ls *src-sbref*` is\n"
        "#     the whole lane; the same files without that tag are the BOLD-mean\n"
        "#     lane, resampled by the SAME transforms — compare them as a check.\n"
        "#     They also feed the anat step: -anat_source auto (the default) aligns\n"
        "#     stage08.grandmean.src-sbref to the anat, not the BOLD-mean grandmean.\n"
        "#     -anat_source grandmean goes back to the BOLD mean."
    )


def _lane_header_note(plan: Plan) -> str:
    """What the coverage lanes are and which lineage estimates the transforms."""
    return (
        f"\n#     lanes: {', '.join(_lanes(plan))} — one set of transforms, estimated from"
        f"\n#     '{_primary_lane(plan)}', applied to every lineage. src-max is the temporal MAX of"
        "\n#     each run (the union of every voxel any frame saw), composited up the"
        "\n#     pyramid with max-of-maxes, so edges motion cost one run survive in the"
        "\n#     grandmean instead of being averaged away. src-min is its counterpart:"
        "\n#     min-of-mins, >0 exactly where every frame of every run has data."
    )


def _fmap_inherit_note(plan: Plan) -> str:
    """Runs no fieldmap claimed, which took the session's reference group instead —
    a guess worth stating out loud, since it silently undistorts real data."""
    borrowed = [_frag(pr) for pr in plan.runs if pr.fmap_inherited]
    if not borrowed:
        return ""
    shown = ", ".join(borrowed[:4]) + (" ..." if len(borrowed) > 4 else "")
    return (
        f"\n#   inherited fmap    : {len(borrowed)} run(s) matched no IntendedFor/acq-time and"
        f"\n#     took the session's REFERENCE fieldmap: {shown}."
        "\n#     Check that is right — fix IntendedFor if it is not."
    )


def _qc_header_note(plan: Plan) -> str:
    """Point the reader at the QC stacks and say what they are for."""
    if not _qc_on(plan):
        return "\n#   QC stacks         : OFF (-no_qc)"
    return (
        "\n#   QC stacks         : stageNN.QC.*.nii.gz — at each level, the images that"
        "\n#     level put in ONE space, concatenated over time. Scroll the time axis:"
        "\n#     a jump between sub-bricks is that stage failing, and the sub-brick"
        "\n#     label names the run/session/fieldmap. stage10.QC.final is the whole"
        "\n#     dataset as the GLM sees it — start there, work backwards."
    )


def _invocation_note(invocation: str | None) -> str:
    """The ffs_autoproc command that generated this script, commented out.

    Regenerating after a BIDS or flag change means retyping the original call;
    keeping it in the file it produced means uncommenting one line. Wrapped at a
    readable width with continuations so it can be pasted as-is."""
    if not invocation:
        return ""
    words = shlex.split(invocation)
    lines: list[str] = []
    cur = "# " + words[0]
    for w in words[1:]:
        # Keep a flag and its value on one line; break before the next flag.
        if len(cur) + len(w) + 1 > 84 and w.startswith("-"):
            lines.append(cur + " \\")
            cur = "#   " + w
        else:
            cur += " " + w
    lines.append(cur)
    body = "\n".join(lines)
    return (
        "\n# --- generated by (uncomment to regenerate) --------------------------------\n" + body
    )


def _header(plan: Plan, out_dir: str, invocation: str | None = None) -> str:
    opt = plan.options
    # Phase working files are plain gzip whatever FMT is: ROMEO (and every other
    # non-ffs reader) cannot open .nii.zst.
    phase_fmt = (
        "\nPHASE_FMT=.gz         # phase working files; ROMEO can't read .zst"
        if _phase_on(plan)
        else ""
    )
    phase_skip = "\nskip_unwrap=1" if _phase_on(plan) else ""
    return f"""#!/usr/bin/env bash
# =============================================================================
# ffs pipeline for sub-{plan.subject}  —  generated by ffs_autoproc
#
# Read me, edit me, run me. Every stage is guarded ([ -f ... ] && continue) so
# re-running resumes where it left off. Flip a skip_<stage> toggle to force a
# whole stage to re-run. The transform chain applied at the final resample is
# baked into CHAIN[...] below — inspect it before you trust it.
#
#   recipe            : {opt.recipe or "(none)"}
#   reference session : {plan.ref_session}
#   sessions          : {"multi" if plan.multi_session else "single"}
#   NORDIC : {opt.want_nordic}   locomoco : {opt.locomoco}   distortion : {opt.distortion}   slice-timing : {opt.slicetiming_method}
#   phase  : {_phase_on(plan)}{_timing_header_note(plan)}
#   sbref lane        : {plan.use_sbref}{_sbref_header_note(plan)}
#   image lanes       : {len(_lanes(plan))}{_lane_header_note(plan)}
#   ref image         : {opt.ref_image or "grandmean"} (session representative for xses; -ref_image){_fmap_inherit_note(plan)}{_qc_header_note(plan)}
# ============================================================================={_invocation_note(invocation)}
set -euo pipefail

OUT={shlex.quote(out_dir)}
DEVICE={shlex.quote(opt.device)}          # compute device for every ffs stage; -device
FMT={opt.fmt}           # working intermediates (read many times); -format
FINAL_FMT={opt.final_fmt}     # final timeseries (portable); -final_format
GLM_FMT={opt.glm_fmt}       # GLM stat buckets; -glm_format
NOISE_VOLS={opt.noise_vols}{phase_fmt}

mkdir -p "$OUT"; cd "$OUT"

# ---- wall-clock tic/toc ------------------------------------------------------
# SECONDS is bash's own monotonic counter, so this survives `set -e` and costs
# nothing. The trap fires on success, on error and on ^C, so a run that dies
# halfway still tells you how far it got before it did.
SECONDS=0
_ffs_toc() {{
  local rc=$? h=$((SECONDS/3600)) m=$(((SECONDS%3600)/60)) s=$((SECONDS%60))
  printf '\n== ffs_autoproc: %s after %02d:%02d:%02d (%ds) ==\n' \
    "$([ "$rc" -eq 0 ] && echo finished || echo "FAILED (exit $rc)")" \
    "$h" "$m" "$s" "$SECONDS"
  return "$rc"
}}
trap _ffs_toc EXIT

# ---- coarse per-stage re-run toggles (1 = skip stage output if it exists) ---
# The batched moco + final stages pass their toggle to the tool as -batch_skip.
skip_nordic=1 skip_moco={_skip_default(opt)} skip_locomoco=1 skip_blip=1
skip_xfmap=1  skip_xrun=1 skip_runmean=1 skip_xses=1 skip_anat=1
skip_final={_skip_default(opt)} skip_stats=1{phase_skip}
"""


def _data_arrays(plan: Plan, bids_root: str | None = None) -> str:
    lines = [
        "",
        "# ============================ per-run data table ============================",
    ]
    keys = [_key(pr) for pr in plan.runs]
    primary = _primary_lane(plan)
    lines.append("RUN_KEYS=(" + " ".join(shlex.quote(k) for k in keys) + ")")
    # One MOCO_<LANE> array per BOLD-derived lineage (MOCO_MEAN / MOCO_MAX /
    # MOCO_MIN); the SBRef lane reads SBREF directly. REFGRID is lane-free: every
    # lane's image is on the run's own geometry, so they share one target grid.
    moco_arrays = " ".join(f"MOCO_{ln.upper()}" for ln in _lanes(plan) if ln != LANE_SBREF)
    lines.append(
        "declare -A MAG PHASE SBREF TR PEDIR JSON FRAG CHAIN SBCHAIN XRUNBASE EVENTS "
        f"{moco_arrays} PRECHAIN REFGRID JAC"
    )

    # first run per session (the no-fmap anchor).
    first_by_ses: dict[str | None, PlanRun] = {}
    for pr in plan.runs:
        first_by_ses.setdefault(pr.bold.session, pr)

    for pr in plan.runs:
        b = pr.bold
        k = _key(pr)
        q = shlex.quote
        lines.append(f"# {k}")
        lines.append(f"MAG[{q(k)}]={q(str(b.mag_path))}")
        if b.phase_path:
            lines.append(f"PHASE[{q(k)}]={q(str(b.phase_path))}")
        if b.sbref_path:
            lines.append(f"SBREF[{q(k)}]={q(str(b.sbref_path))}")
        lines.append(f"JSON[{q(k)}]={q(str(_sidecar(b.mag_path)))}")
        tr = plan.options.tr if plan.options.tr is not None else b.tr
        lines.append(f"TR[{q(k)}]={q(str(tr if tr is not None else ''))}")
        lines.append(f"PEDIR[{q(k)}]={q(str(b.pe_dir or ''))}")
        # Filename coordinate fragment — the stage loops build run filenames as
        # stageNN.label.${FRAG[$k]}, matching every reference in the data table.
        lines.append(f"FRAG[{q(k)}]={q(_frag(pr))}")
        # The per-run task diagnostics read this -- NORDIC's task-leak report at
        # stage00, locomoco's task-coupling report at stage03. Both are per run, and a
        # run whose events did not resolve simply goes unscored rather than failing.
        if plan.options.locomoco or plan.options.want_nordic:
            ev = _events_script_path(plan, pr, bids_root)
            if ev:
                lines.append(f"EVENTS[{q(k)}]={q(ev)}")
        # These values contain the bash var $FMT and must expand at assignment,
        # so they are double-quoted (not shlex-quoted). The key and the values
        # are our own constructed names — no shell metacharacters beyond $FMT.
        for lane in _lanes(plan):
            if lane == LANE_SBREF:
                continue
            lines.append(f'MOCO_{lane.upper()}[{q(k)}]="{_moco_reduction(pr, lane)}"')
        # xrun base is lane-aware only in the no-fmap case; with a fieldmap it is
        # the group's forward image (already the SBRef when there is one).
        base = _xrun_base(pr, first_by_ses[b.session], primary)
        if base is not None:
            lines.append(f'XRUNBASE[{q(k)}]="{base}"')
        # Pre-chain (lane image → the session's common grid) + that grid. Empty for
        # a no-fmap session's anchor run: it defines the grid.
        pre = _pre_chain(pr, ".nii$FMT")
        if pre:
            lines.append(f'PRECHAIN[{q(k)}]="{" ".join(pre)}"')
        lines.append(f'REFGRID[{q(k)}]="{_common_grid(pr, first_by_ses[b.session], primary)}"')
        if plan.use_sbref:
            # The SBRef rides the run's chain minus the within-run motion tokens:
            # it IS the moco base, so it needs no transform of its own.
            sbchain = chain_files(pr, ".nii$FMT", plan.options, tokens=sbref_chain(pr))
            lines.append(f'SBCHAIN[{q(k)}]="{" ".join(sbchain)}"')
        chain = chain_files(pr, ".nii$FMT", plan.options)
        lines.append(f'CHAIN[{q(k)}]="{" ".join(chain)}"')
        # Jacobian modulation for the fieldmap link, wherever that chain is
        # applied to data (stage07, stage10). Empty for a run with no fmap.
        jac = _jac_spec(pr)
        if jac:
            lines.append(f'JAC[{q(k)}]="{jac}"')
    return "\n".join(lines) + "\n"


def _all_events(plan: Plan, bids_root: str | None) -> list[str]:
    """Every events TSV the GLM stage will read, across tasks. Empty when the GLM
    is off or nothing resolved (the emitted -events is then a TODO placeholder).

    These are the ``stimuli/`` copies ffs_autoproc writes beside the results, not
    the BIDS originals — the copies are what the design TOMLs name, so they are
    what preflight has to find."""
    from fastfuncstuff.autoproc.glm import stimuli_map

    if not plan.options.run_glm:
        return []
    return list(stimuli_map(plan, bids_root).values())


def _spec_files(plan: Plan, bids_root: str | None) -> list[str]:
    """Design TOMLs stage12 will read — one per task whose events resolved.
    Deleting a spec should stop the run at preflight, not at the GLM."""
    from fastfuncstuff.autoproc.glm import runs_by_task, spec_path

    if not plan.options.run_glm:
        return []
    return [
        spec_path(task)
        for task, prs in runs_by_task(plan).items()
        if events_for_task(task, prs, bids_root, plan.options)
    ]


def _preflight(plan: Plan, bids_root: str | None = None) -> str:
    inputs = sorted(
        {str(pr.bold.mag_path) for pr in plan.runs}
        | {str(pr.bold.phase_path) for pr in plan.runs if pr.bold.phase_path}
        | {str(pr.fmap.reverse_path) for pr in plan.runs if pr.fmap}
        | {str(pr.fmap.forward_path) for pr in plan.runs if pr.fmap and pr.fmap.forward_path}
        # A GRE fieldmap is several files, and reverse_path names only the first:
        # the magnitudes it is masked and matched with are just as load-bearing.
        | {
            str(p)
            for pr in plan.runs
            if pr.fmap
            for p in [*pr.fmap.phase_paths, *pr.fmap.magnitude_paths]
        }
        # SBRefs are load-bearing once the lane is on (moco base, xrun source,
        # the whole sbmean pyramid) — a missing one should fail here, not midway.
        | {str(pr.bold.sbref_path) for pr in plan.runs if pr.use_sbref}
        # The GLM is the last stage; a missing events file or design spec should
        # fail here, not after an hour of preprocessing. Only files we actually
        # resolved are checked (a task with no events warns at generation time).
        | set(_all_events(plan, bids_root))
        | set(_spec_files(plan, bids_root))
        # The stage00 and stage03 task diagnostics read events too, and with -no_glm
        # those are BIDS originals nothing else in this list covers.
        | {
            ev
            for pr in plan.runs
            if plan.options.locomoco or plan.options.want_nordic
            for ev in [_events_script_path(plan, pr, bids_root)]
            if ev
        }
    )
    checks = " \\\n".join(f"  {shlex.quote(p)}" for p in inputs)
    # romeo (MRItools) is an external dependency, only needed with -phase_proc.
    # Only the *command* is checked, not the flags that fix a backend's identity
    # ("ffs_optiwarp -force demons" is one executable), and only for the stages
    # this script actually runs -- a missing tool nobody calls is not a problem.
    opt = plan.options
    nl_tools = [
        _nl_tool(opt, stage).split()[0]
        for stage, on in (
            ("xrun_nl", opt.xrun_nonlin),
            ("xfmap_nl", opt.xfmap_nonlin),
            ("xses_nl", opt.xses_nonlin or opt.grand_reference_nonlin),
        )
        if on
    ]
    tools = [*_TOOLS, *dict.fromkeys(nl_tools)]
    if _phase_on(plan):
        tools = [*tools, "romeo"]
    return f"""
# =============================== stage: preflight ===========================
echo '== preflight: inputs + tools =='
_missing=0
for f in \\
{checks}
do [ -f "$f" ] || {{ echo "MISSING INPUT: $f"; _missing=1; }}; done
for t in {" ".join(tools)}; do
  command -v "$t" >/dev/null 2>&1 || {{ echo "MISSING TOOL: $t"; _missing=1; }}
done
[ "$_missing" -eq 0 ] || {{ echo 'preflight failed'; exit 1; }}
echo '   ok'
"""


def _pre_stc_source(plan: Plan, indent: str = "  ") -> str:
    """Bash setting ``raw`` to the series BEFORE any up-front slice timing — the
    NORDIC output, or the original BIDS magnitude read in place with an inline
    noise-vol sub-brick selector (no whole-dataset copy)."""
    if plan.options.want_nordic:
        return f'{indent}raw="{_nordic_mag(plan)}"'
    return (
        f'{indent}raw="${{MAG[$k]}}"\n'
        f'{indent}if [ "$NOISE_VOLS" -gt 0 ]; then '
        f'nv=$(ffs_info -nv "$raw"); raw="${{raw}}[0..$((nv - NOISE_VOLS - 1))]"; fi'
    )


def _raw_source(plan: Plan, indent: str = "  ") -> str:
    """Bash setting ``raw`` to the series that moco and the final resample act on.

    Precedence: NORDIC output → up-front tshift output (``-slicetiming_method
    first``) → BIDS magnitude (noise-vols trimmed). Nothing is copied wholesale.
    When STC is done first, the final ffs_nwarp reads the tshifted series and does
    NOT re-integrate timing (that only happens for ``-slicetiming_method
    integrate``)."""
    if plan.options.slicetiming_method == "first":
        return f'{indent}raw="stage01.tshift.${{FRAG[$k]}}.nii$FMT"'
    return _pre_stc_source(plan, indent)


def _stage_tshift(plan: Plan) -> str:
    if plan.options.slicetiming_method != "first":
        return ""
    cmd = _ffs(
        "ffs_slicetime",
        [
            '-input "$raw"',
            '-prefix "$outf"',
            f"-tpattern {_tpattern(plan)}",
            '-TR "${TR[$k]}"',
            *_split_flags(config.DEFAULT_OPTS["tshift"]),
            '-device "$DEVICE"',
        ],
    )
    # Phase must get the SAME temporal shift as its magnitude, or the two stop
    # describing the same instant. Unwrapped phase is smooth in time, so the same
    # temporal interpolation is valid on it (a wrapped phase would not be).
    phase_cmd = ""
    if _phase_on(plan):
        pc = _ffs(
            "ffs_slicetime",
            [
                f'-input "{_unwrapped()}"',
                '-prefix "$phout"',
                f"-tpattern {_tpattern(plan)}",
                '-TR "${TR[$k]}"',
                *_split_flags(config.DEFAULT_OPTS["tshift"]),
                '-device "$DEVICE"',
            ],
        )
        phase_cmd = f'  phout="{_unwrapped("tshift")}"\n  [ -f "$phout" ] || {{\n{pc}\n  }}\n'
    return f"""
# ============================ stage01: slice timing (first) =================
# STC applied up front (before moco); the final resample does NOT re-integrate
# timing. Reads the NORDIC/BIDS source in place; writes the tshifted series.
echo '== stage01: slice timing =='
for k in "${{RUN_KEYS[@]}}"; do
  outf="stage01.tshift.${{FRAG[$k]}}.nii$FMT"
  [ -z "${{TR[$k]}}" ] && {{ echo "no TR for $k; cannot slice-time"; exit 1; }}
{phase_cmd}  [ -f "$outf" ] && continue
{_pre_stc_source(plan)}
{cmd}
done
"""


def _stage_nordic(plan: Plan) -> str:
    opt = plan.options
    if not opt.want_nordic:
        # No stage00: nothing is copied. moco/final read the BIDS source directly.
        return ""
    resid = "-save-residual-map" if getattr(opt, "nordic_save_resid", False) else ""
    # ffs_nordic writes a denoised phase alongside the magnitude whenever phase
    # input was used, and it trims the noise volumes off both — so the -phase_proc
    # path needs nothing extra here beyond the plain-gzip output format.
    phase_note = (
        "\n# -phase_proc: the denoised phase lands beside the magnitude as\n"
        "# stage00.nordic.<frag>_phase.nii.gz (noise volumes already trimmed);\n"
        "# stage00 unwrap picks it up. Plain gzip because ROMEO can't read .zst.\n"
        if _phase_on(plan)
        else ""
    )
    rescue = opt.nordic_task_rescue
    nordic_cmd = _ffs(
        "ffs_nordic",
        [
            '-input_magn "${MAG[$k]}"',
            '"${phase_arg[@]}"',
            '-prefix "$outf"',
            '-noise-volume-last "$NOISE_VOLS"',
            *_split_flags(config.DEFAULT_OPTS["nordic"]),
            resid,
            # Diagnostic only: NORDIC decides what to discard from a patch's spectrum
            # alone, and nothing in that decision knows the task, so a component
            # carrying real response can fall under the threshold. -tr rides along for
            # the same reason it does at stage03 -- a 3D acquisition's pixdim[4] is the
            # per-partition time, not the volume TR the design is sampled at.
            '${ev:+-events "${ev}"}',
            '${ev:+${TR[$k]:+-tr "${TR[$k]}"}}',
            *([f"${{ev:+-task_rescue {rescue}}}"] if rescue else []),
            '-device "$DEVICE"',
        ],
    )
    return f"""
# ============================ stage00: NORDIC ==============================={phase_note}
# -events (per run, the same TSV the GLM models) turns on the task-leak report:
# <prefix>_taskleak.txt, the _taskfit_input/_kept/_lost buckets, the per-patch
# _taskpatch map, and a scan of the removed field's components. It MEASURES how much
# of the design went out with the noise and whether it landed on responding tissue;
# nothing about the denoising changes. This runs before motion correction, so read
# the enrichment and the component scan, not the per-voxel R maps.
# -nordic_task_rescue is what ACTS on it.
echo '== stage00: NORDIC denoise =='
for k in "${{RUN_KEYS[@]}}"; do
  outf="{_nordic_mag(plan)}"
  [ "$skip_nordic" -eq 1 ] && [ -f "$outf" ] && continue
  ph="${{PHASE[$k]:-}}"
  if [ -n "$ph" ]; then phase_arg=(-input_phase "$ph"); else phase_arg=(-magnitude-only); fi
  ev="${{EVENTS[$k]:-}}"   # unset for a run whose events did not resolve; set -u
{nordic_cmd}
done
"""


def _moco_reduction_flags() -> str:
    """``-save_mean/-save_max/-save_min`` for the stage02 manifest line (escaped for
    the printf that writes it). One resample, three images: the mean the pipeline
    always used, plus the coverage pair the max/min lanes are built from."""
    return " ".join(
        f'-save_{which} \\"${{mstem}}_{which}.nii$FMT\\"' for which in ("mean", "max", "min")
    )


def _stage_moco(plan: Plan, script_stem: str) -> str:
    moco_flags = " ".join(_split_flags(config.DEFAULT_OPTS["moco"]))
    batchfile = f"{script_stem}_mocobatch.txt"
    # locomoco estimates residual motion per volume: it needs the rigid-corrected
    # 4D, not the mean. Only then is it worth writing this intermediate.
    ts_arg = ' -prefix \\"${mstem}.nii$FMT\\"' if plan.options.locomoco else ""
    return f"""
# ============================ stage02: motion correction ====================
# Batched: ONE ffs_moco process motion-corrects every run, so the Python/CUDA/
# torch.compile startup is paid once, not per run. Each run's arguments are
# appended to {batchfile} (beside the script's outputs) and consumed with -batch.
# Reads the source series in place (no full-dataset copy). Saves the target mean
# + per-volume matrices + motion params; the corrected 4D is normally produced
# once by the final single-resample (stage10) — the exception is locomoco, which
# estimates residual per-volume PE motion and therefore needs a rigid-corrected
# 4D of its own (written here with -prefix, wsinc5).
# skip_moco=1 → -batch_skip (skip complete runs).
echo '== stage02: moco =='
MOCO_REF={plan.options.moco_ref}   # sbref | first | last | <int>
mocobatch="{batchfile}"
: > "$mocobatch"
for k in "${{RUN_KEYS[@]}}"; do
  mstem="stage02.moco.${{FRAG[$k]}}"
{_raw_source(plan)}
  sb="${{SBREF[$k]:-}}"
  case "$MOCO_REF" in
    sbref) if [ -n "$sb" ]; then base_str="-base \\"$sb\\""; else base_str="-base 0"; fi ;;
    first) base_str="-base 0" ;;
    last)  nv=$(ffs_info -nv "$raw"); base_str="-base $((nv - 1))" ;;
    *)     base_str="-base \\"$MOCO_REF\\"" ;;   # integer volume index
  esac
  printf '%s\\n' "-input \\"$raw\\" $base_str {moco_flags}{ts_arg} {_moco_reduction_flags()} -1Dmatrix_save \\"${{mstem}}.aff12.1D\\" -1Dfile \\"${{mstem}}.motion.1D\\"" >> "$mocobatch"
done
batch_skip=(); [ "$skip_moco" -eq 1 ] && batch_skip=(-batch_skip)
ffs_moco -batch "$mocobatch" "${{batch_skip[@]}}" -device "$DEVICE"
"""


def _stage_locomoco(plan: Plan, script_stem: str) -> str:
    if not plan.options.locomoco:
        return ""
    detask = plan.options.locomoco_detask
    parts = [
        '-input "stage02.moco.${FRAG[$k]}.nii$FMT"',
        '-prefix "${nlstem}.nii$FMT"',
        '-pe_dir "${pe:-y}"',
        *_split_flags(config.DEFAULT_OPTS["locomoco"]),
        # Diagnostic only: it measures how much of the estimated field the task
        # explains, which is exactly the failure mode a brightness-constancy data
        # term has on a block design. -tr rides along because a 3D acquisition's
        # header pixdim[4] is the per-partition time, not the volume TR the design
        # is sampled at, and the design is what this fit is against.
        '${ev:+-events "${ev}"}',
        '${ev:+${TR[$k]:+-tr "${TR[$k]}"}}',
        *([f"-detask {detask}"] if detask else []),
        "-warp_format 5d",
        "-save_mean",
        "-save_max",
        "-save_min",
    ]
    return f"""
# ============================ stage03: locomoco (residual NL motion) ========
# Input is stage02's rigid-corrected 4D (written only when this stage runs), NOT
# the moco mean: locomoco estimates a warp per volume, so it needs the time axis.
# Its own mean (_locomoco_mean) is what the rest of the chain aligns from.
# Batched: the loop only WRITES the manifest, then ONE ffs_locomoco does every
# run. The flow backend's torch.compile warmup is per-PROCESS, so a shell loop
# paid it N times. skip_locomoco=1 → -batch_skip; the tool checks the _warp AND
# the lane reductions, so a working directory from before the coverage lanes
# existed re-runs and produces them instead of resuming into a stage07 that
# cannot find its inputs.
#
# -events (per run, the same TSV the GLM models) turns on the task-coupling
# report: _locomoco_taskcoupling.txt plus the taskr/taskrms maps. It is a
# DIAGNOSTIC -- nothing about the correction changes -- and it is what to read
# before reaching for -locomoco_detask.
echo '== stage03: locomoco =='
lmbatch="{script_stem}_locomocobatch.txt"; : > "$lmbatch"
for k in "${{RUN_KEYS[@]}}"; do
  nlstem="stage03.nlmoco.${{FRAG[$k]}}"
  pe="${{PEDIR[$k]}}"; pe="${{pe//[!a-zA-Z]/}}"
  ev="${{EVENTS[$k]:-}}"   # unset for a run whose events did not resolve; set -u
{_manifest_line("lmbatch", parts)}
done
{_batch_launch("ffs_locomoco", "lmbatch", "skip_locomoco")}
"""


def _fmap_pe(pr: PlanRun) -> str:
    """The signed PE direction the fieldmap's warp is built for (``j-``, ``i``…).

    The group's own value when it has one — a reverse-PE fmap's polarity is a
    property of the acquisition, and a split GRE group carries its polarity in
    ``pe_override`` — else the run's."""
    return (pr.fmap.pe_dir if pr.fmap else None) or pr.bold.pe_dir or "j"


def _stage_blip(plan: Plan, script_stem: str) -> str:
    groups: dict[tuple, PlanRun] = {}
    for pr in plan.runs:
        if pr.fmap is None:
            continue
        groups.setdefault((pr.bold.session, pr.fmap.fmap_id), pr)
    if not groups:
        return ""
    kinds = {pr.fmap.kind for pr in groups.values() if pr.fmap}
    tool = "b0fmap" if kinds == {"b0"} else "blipflip"
    out = ["", f"# ============================ stage04: fieldmap ({tool}) " + "=" * 18]
    out.append("echo '== stage04: distortion correction =='")
    # The blip pairs are batched into one ffs_blipflip; GRE fieldmaps have no
    # batch mode and stay one call each.
    blip_jobs: list[list[str]] = []
    for pr in groups.values():
        assert pr.fmap is not None
        st = _fmap_stem(pr, "blip")
        if pr.fmap.is_b0:
            out.append(_blip_b0(pr, st))
            continue
        pe = "".join(c for c in _fmap_pe(pr) if c.isalpha()) or "j"
        ro = f"-readout {pr.fmap.readout}" if pr.fmap.readout else ""
        # blip_up: the fmap's own matched-PE image when the pair is self-contained,
        # else the borrowed rep the plan picked for the group (the run acquired
        # next to the fieldmap — NOT this loop's representative run, which is
        # whichever one the group dict happened to see first).
        up = pr.fmap_forward or pr.bold.rep
        blip_jobs.append(
            [
                f"-blip_up {shlex.quote(str(up))}",
                f"-blip_down {shlex.quote(str(pr.fmap.reverse_path))}",
                f"-pe_dir {pe}",
                ro,
                *_split_flags(config.DEFAULT_OPTS["blip"]),
                f'-prefix "{st}.nii$FMT"',
            ]
        )
    if blip_jobs:
        out.append(f'bfbatch="{script_stem}_blipbatch.txt"; : > "$bfbatch"')
        out += [_manifest_line("bfbatch", parts, indent="") for parts in blip_jobs]
        out.append(_batch_launch("ffs_blipflip", "bfbatch", "skip_blip"))
    return "\n".join(out) + "\n"


def _blip_b0(pr: PlanRun, st: str) -> str:
    """One GRE fieldmap group: ffs_util_b0fmap, then the ``_mean`` the rest of the
    pipeline expects from this stage.

    Two things a reverse-PE pair gives for free and a GRE fieldmap does not:

    * **The EPI geometry.** A GRE sidecar has echo times, not a phase-encode
      direction or a readout — the field is measured in a sequence that has no
      EPI distortion of its own. Both come from the run instead, and ``-readout``
      is not optional for a measured field (it sets the Hz→voxel scale), so a run
      whose sidecar omits ``TotalReadoutTime`` is reported here rather than
      failing inside the tool.
    * **An undistorted forward image.** blipflip corrects its own pair and writes
      both corrected images; sub-brick 0 of those, the corrected blip-up, defines
      the session's common space (every ``_ref_blip_forward`` consumer downstream).
      There is no EPI pair here, so we make the equivalent image the same way the
      final resample will: pull the run's rep through the warp with the Jacobian,
      which is what keeps this image and the runmeans agreeing about intensity
      where distortion was worst (see :func:`_jac_spec`). That single warped rep IS
      the forward image, so it is written as ``_mean`` and used directly.
    """
    fmap = pr.fmap
    assert fmap is not None
    pe = _fmap_pe(pr)
    axis = "".join(c for c in pe if c.isalpha()) or "j"
    epi = pr.fmap_forward or str(pr.bold.rep)
    readout = fmap.readout if fmap.readout is not None else pr.bold.json.get("TotalReadoutTime")
    if readout is None:
        return (
            f'echo "ffs_autoproc: fmap-{fmap.fmap_id} is a GRE fieldmap but '
            f'no TotalReadoutTime was found for its runs;" >&2\n'
            f'echo "  a measured field has an absolute Hz scale, so the readout '
            f'cannot be inferred. Aborting." >&2\nexit 1'
        )
    if fmap.phasediff_path is not None:
        src = [f"-phasediff {shlex.quote(str(fmap.phasediff_path))}"]
    elif fmap.fieldmap_path is not None:
        src = [f"-fieldmap {shlex.quote(str(fmap.fieldmap_path))}"]
    else:
        src = ["-phase " + " ".join(shlex.quote(str(p)) for p in fmap.phase_paths)]
    mags = " ".join(shlex.quote(str(p)) for p in fmap.magnitude_paths)
    cmd = _ffs(
        "ffs_util_b0fmap",
        [
            *src,
            f"-magnitude {mags}" if mags else "",
            f"-epi {shlex.quote(epi)}",
            f"-pe_dir {pe}",
            f"-readout {readout}",
            "-te " + " ".join(f"{t:g}" for t in fmap.te_ms) if fmap.te_ms else "",
            *_split_flags(config.DEFAULT_OPTS["b0fmap"]),
            f'-prefix "{st}.nii$FMT"',
            '-device "$DEVICE"',
        ],
    )
    # A static warp is applied identically to every volume, so collapsing before
    # or after it gives the same image — but only ffs_nwarp can collapse, so a 4-D
    # anchor pays for a full warped timeseries it then throws away. A 3-D anchor
    # (the SBRef case, and most fieldmap datasets have one) warps straight to the
    # mean with no intermediate at all.
    if _n_volumes(epi) > 1:
        tmp = f"{st}_unwarped.nii$FMT"
        mean = _ffs(
            "ffs_nwarp",
            [
                f"-source {shlex.quote(epi)}",
                f"-master {shlex.quote(epi)}",
                f'-nwarp "{st}_warp.nii$FMT"',
                f"-jac {axis}",
                "-save_mean",
                f'-prefix "{tmp}"',
                '-device "$DEVICE"',
            ],
        )
        mean += f'\n  mv -f "mean_{st}_unwarped.nii$FMT" "{st}_mean.nii$FMT"\n  rm -f "{tmp}"'
    else:
        mean = _ffs(
            "ffs_nwarp",
            [
                f"-source {shlex.quote(epi)}",
                f"-master {shlex.quote(epi)}",
                f'-nwarp "{st}_warp.nii$FMT"',
                f"-jac {axis}",
                f'-prefix "{st}_mean.nii$FMT"',
                '-device "$DEVICE"',
            ],
        )
    return (
        f'if [ "$skip_blip" -ne 1 ] || [ ! -f "{st}_warp.nii$FMT" ]; then\n{cmd}\nfi\n'
        f'if [ "$skip_blip" -ne 1 ] || [ ! -f "{st}_mean.nii$FMT" ]; then\n{mean}\nfi'
    )


def _n_volumes(path: str) -> int:
    """Volume count of an input, read from its header at emit time (the file is on
    disk — the scanner just read its sidecar). Unreadable → 2, i.e. take the branch
    that is correct for both cases."""
    try:
        import nibabel as nib

        shape: tuple[int, ...] = tuple(getattr(nib.load(path), "shape", ()))
    except Exception:
        return 2
    return int(shape[3]) if len(shape) > 3 else 1


def _stage_xfmap(plan: Plan, script_stem: str) -> str:
    """Align each NON-reference fmap group's undistorted forward image to the
    session's reference fmap forward image (once per group). This is what lets runs acquired under
    different fieldmaps share one session space; the per-run runmean composes it
    with blip+xrun (see stage07)."""
    opt = plan.options
    groups: dict[tuple, PlanRun] = {}
    for pr in plan.runs:
        if pr.fmap is None or "xfmap_lin" not in pr.warp_chain:
            continue  # reference fmap (or no fmap) → no cross-fmap alignment
        groups.setdefault((pr.bold.session, pr.fmap.fmap_id), pr)
    fmapmean_sessions = _fmapmean_sessions(plan)
    if not groups and not fmapmean_sessions:
        return ""
    out = ["", "# ============================ stage05: cross-fmap alignment ================="]
    out.append("echo '== stage05: cross-fmap alignment (→ reference fmap) =='")
    # The reference fmap group has no xfmap transform (it maps to itself) — emit its
    # forward image under the xfmap name so every group shows up in one listing.
    anchor = ref_anchor(plan)
    if anchor is not None and anchor.fmap is not None:
        out.append(
            _ref_marker(
                NameKey("xfmap", session=anchor.bold.session, fmap=anchor.fmap.fmap_id, role="ref"),
                _ref_blip_forward(anchor),
                f"QC: fmap-{anchor.fmap.fmap_id} is the reference group — no transform.",
            )
        )
    # Batched like stage06/stage08: manifests first, then one process each.
    if groups:
        out.append(f'albatch="{script_stem}_xfmapbatch.txt"; : > "$albatch"')
        if opt.xfmap_nonlin:
            out.append(f'fwbatch="{script_stem}_xfmapnlbatch.txt"; : > "$fwbatch"')
    for pr in groups.values():
        xstem = _fmap_stem(pr, "xfmap")
        src = _fmap_forward(pr)  # this fmap, undistorted forward image
        base = _ref_blip_forward(pr)  # reference fmap, same
        out.append(
            _manifest_line(
                "albatch",
                [
                    f'-base "{base}"',
                    f'-source "{src}"',
                    f'-prefix "{xstem}_lin.nii$FMT"',
                    f'-1Dmatrix_save "{xstem}.aff12.1D"',
                    *_split_flags(config.DEFAULT_OPTS["xfmap"]),
                ],
                indent="",
            )
        )
        if opt.xfmap_nonlin:
            out.append(
                _manifest_line(
                    "fwbatch",
                    [
                        f'-base "{base}"',
                        *_nl_source_args(
                            opt.xfmap_nonlin_in_source,
                            f"{xstem}_lin.nii$FMT",
                            src,
                            f"{xstem}.aff12.1D",
                        ),
                        f'-prefix "{xstem}_nl.nii$FMT"',
                        "-save_warp",
                        *_split_flags(config.DEFAULT_OPTS["xfmap_nl"]),
                    ],
                    indent="",
                )
            )
    if groups:
        out.append(_batch_launch("ffs_allineate", "albatch", "skip_xfmap"))
        if opt.xfmap_nonlin:
            out.append(_batch_launch(_nl_tool(opt, "xfmap_nl"), "fwbatch", "skip_xfmap"))
    if fmapmean_sessions:
        # mean_fmap: within a session every group's mean is now on that session's
        # reference-fmap grid, so averaging them is a straight voxelwise mean (more
        # SNR than any single group, without the grandmean's extra interpolation).
        out.append("echo '== stage05: mean of fieldmap means (mean_fmap) =='")
    for ses in fmapmean_sessions:
        fm = _fmapmean(plan, ses)
        inputs = " ".join(f'"{f}"' for f in _fmapmean_inputs(plan, ses))
        out.append(
            f'[ -f "{fm}" ] || \\\n'
            + _ffs(
                "ffs_util_3dmath",
                [f"-input {inputs}", "-mean", f'-prefix "{fm}"', '-device "$DEVICE"'],
            )
        )
    out.append(_qc_xfmap(plan))
    return "\n".join(p for p in out if p) + "\n"


def _stage_xrun(plan: Plan, script_stem: str) -> str:
    opt = plan.options
    primary = _primary_lane(plan)
    src_var = _run_level_var(primary)
    albatch = f"{script_stem}_xrunbatch.txt"
    fwbatch = f"{script_stem}_xrunnlbatch.txt"
    nl_append = ""
    nl_launch = ""
    if opt.xrun_nonlin:
        # Residual nonlinear refinement of the linear-aligned image → distinct
        # `_nl` output (never overwrite the linear source). The warp is a chain
        # link shared by BOTH lanes, so it keeps the lane-free stem (naming.py).
        nl_append = (
            _manifest_line(
                "fwbatch",
                [
                    '-base "$base"',
                    *_nl_source_args(
                        opt.xrun_nonlin_in_source,
                        "${xstem}${LANE}_lin.nii$FMT",
                        src_var,
                        "${xstem}.aff12.1D",
                    ),
                    '-prefix "${xstem}${LANE}_nl.nii$FMT"',
                    "-save_warp",
                    # The warp is shared by every lane; only the image it wrote
                    # belongs to the lane that produced it.
                    '-warp_prefix "${xstem}_nl"',
                    *_split_flags(config.DEFAULT_OPTS["xrun_nl"]),
                ],
            )
            + "\n"
        )
        nl_launch = "\n" + _batch_launch(_nl_tool(opt, "xrun_nl"), "fwbatch", "skip_xrun")
    # The aligned images the pyramid actually consumes are stage07's runmeans
    # (both lanes, one shared matrix); these -prefix outputs are alignment QC, so
    # they carry the lane that produced them. The matrix never does.
    lin_append = _manifest_line(
        "albatch",
        [
            '-base "$base"',
            f'-source "{src_var}"',
            '-prefix "${xstem}${LANE}_lin.nii$FMT"',
            '-1Dmatrix_save "${xstem}.aff12.1D"',
            *_split_flags(config.DEFAULT_OPTS["xrun"]),
        ],
    )
    # No-fmap mode: the session's first run IS the anchor, so it gets no xrun output.
    # Emit its lane image under the xrun name so the listing covers every run.
    markers = [
        _ref_marker(
            NameKey(
                "xrun",
                session=pr.bold.session,
                task=pr.bold.task,
                run=pr.bold.run or None,
                src=_src(primary),
                role="ref",
            ),
            _run_level(pr, primary),
            f"QC: {_frag(pr)} is the cross-run anchor — no transform.",
        )
        for pr in plan.runs
        if "wxrun_lin" not in pr.warp_chain
    ]
    marker_block = ("\n".join(markers) + "\n") if markers else ""
    lane_note = (
        "# Source is each run's SBRef: it is the moco base, so it is already in the\n"
        "# run's corrected space with no transform of its own, and it matches the\n"
        "# base's contrast (the fieldmap forward image is an SBRef too). Both lanes\n"
        "# are then resampled by the ONE matrix this saves.\n"
        if primary == LANE_SBREF
        else ""
    )
    return f"""
# ============================ stage06: cross-run alignment ==================
# Align each run to its anchor (fmap group forward image, or the session's first
# run when there are no fmaps). Saves a matrix that composes into the chain.
# Batched: the loop only WRITES the manifests, then ONE ffs_allineate (and one
# nonlinear backend) process does every run — Python/CUDA/torch.compile startup is
# paid once instead of once per run. The nonlinear batch runs after the linear
# one because its source is that run's linear output.
# skip_xrun=1 → -batch_skip (skip runs whose outputs already exist).
{lane_note}echo '== stage06: cross-run alignment =='
LANE="{f".src-{_src(primary)}" if _src(primary) else ""}"   # lineage tag on the QC images
albatch="{albatch}"; : > "$albatch"
fwbatch="{fwbatch}"; : > "$fwbatch"
{marker_block}for k in "${{RUN_KEYS[@]}}"; do
  base="${{XRUNBASE[$k]:-}}"
  [ -z "$base" ] && continue   # this run is the anchor (identity) — no xrun
  xstem="stage06.xrun.${{FRAG[$k]}}"
{lin_append}
{nl_append}done
{_batch_launch("ffs_allineate", "albatch", "skip_xrun")}{nl_launch}
{_qc_xrun(plan)}
"""


def _stage_grandmean(plan: Plan, script_stem: str) -> str:
    """stage07: the two *within*-session mean levels — runmean then sesmean.

    THE grandmean is not built here. Session means still sit in their own session
    spaces at this point, so averaging them would be meaningless; stage08 builds it
    once xses has brought them into one space. That is why ``grandmean`` carries
    stage number 8 — one producer, one stage, single- and multi-session alike.
    """
    by_ses: dict[str | None, list[PlanRun]] = {}
    for pr in plan.runs:
        by_ses.setdefault(pr.bold.session, []).append(pr)
    out = ["", "# ============================ stage07: run + session means =================="]
    # Every run first gets a "runmean": its lane image pushed through the run's
    # pre-chain (xfmap∘blip∘xrun, whichever it has) onto the session's common
    # grid, so runs from different fmap groups average in one space. A no-fmap
    # session's anchor run defines that grid and has no pre-chain, so it is
    # skipped here and feeds the session mean directly.
    # Batched: one manifest over EVERY (run, lane) pair, then ONE ffs_nwarp. With
    # four lanes this stage used to be 4*N processes whose per-image work — one
    # small 3-D resample — is dwarfed by interpreter + CUDA + torch.compile
    # startup. -batch_skip reproduces the old per-file [ -f ] guard.
    runbatch = f"{script_stem}_runmeanbatch.txt"
    out.append(f'rmbatch="{runbatch}"; : > "$rmbatch"')
    out.append('for k in "${RUN_KEYS[@]}"; do')
    out.append('  [ -z "${PRECHAIN[$k]:-}" ] && continue   # anchor run, already on the grid')
    for lane in _lanes(plan):
        suffix = f".src-{_src(lane)}" if _src(lane) else ""
        label = {LANE_SBREF: "SBRef", LANE_MEAN: "mean"}.get(lane, lane)
        out.append(f"  # {label} lane")
        out.append(
            _manifest_line(
                "rmbatch",
                [
                    f'-source "{_run_level_var(lane)}"',
                    '-nwarp "${PRECHAIN[$k]}"',
                    '${JAC[$k]:+-jac "${JAC[$k]}"}',
                    '-master "${REFGRID[$k]}"',
                    *_lane_nwarp_flags(lane),
                    f'-prefix "stage07.runmean.${{FRAG[$k]}}{suffix}.nii$FMT"',
                ],
            )
        )
    out.append("done")
    lanes_note = ", ".join(
        {LANE_SBREF: "SBRefs", LANE_MEAN: "means"}.get(ln, f"{ln}es") for ln in _lanes(plan)
    )
    out.append(f"echo '== stage07: run {lanes_note} (→ session common grid) =='")
    out.append(_batch_launch("ffs_nwarp", "rmbatch", "skip_runmean"))
    out.append("echo '== stage07: session means =='")
    out += _math_batch(
        "smbatch",
        f"{script_stem}_sesmeanbatch.txt",
        [
            [
                f"-input {' '.join(chr(34) + _aligned_mean(pr, lane) + chr(34) for pr in prs)}",
                # Per lane: means average, maxes take the max, mins the min.
                _lane_reduce(lane),
                f'-prefix "{_sesmean(ses, lane)}"',
                "-overwrite",
            ]
            for lane in _lanes(plan)
            for ses, prs in by_ses.items()
        ],
    )
    out.append(_qc_runmean(plan))
    return "\n".join(p for p in out if p) + "\n"


# Tokens that do NOT belong to the grandmean chain: the within-run motion the
# lane images already have baked in, and everything ABOVE grandmean space (which
# is by definition estimated from the grandmean, so it cannot act before it).
_GRANDMEAN_DROP = frozenset({"moco", "locomoco", "anat_lin", "anat_nl", "xref_lin", "xref_nl"})


def _grandmean_tokens(pr: PlanRun) -> list[str]:
    """The transform tokens taking this run's post-moco lane image all the way to
    grandmean space — pre-chain (xfmap∘blip∘xrun) plus xses, in one chain."""
    return [tok for tok in pr.warp_chain if tok not in _GRANDMEAN_DROP]


def _needs_gmrun(pr: PlanRun) -> bool:
    """True when reaching grandmean space costs this run a resample the pyramid has
    not already paid for — i.e. it has an xses link. A reference-session run's
    grandmean chain IS its pre-chain, so its runmean is already the image."""
    return _grandmean_tokens(pr) != [tok for tok in pr.warp_chain if tok in _PRE_CHAIN_TOKENS]


def _gmrun(pr: PlanRun, lane: str) -> str:
    """This run's lane image in grandmean space, one interpolation from moco."""
    return _run_stem(pr, "gmrun", src=_src(lane)) + ".nii$FMT"


def _gmrun_image(pr: PlanRun, lane: str) -> str:
    """The file this run contributes to the grandmean: the dedicated single-resample
    image when it needs one, else the runmean it already has."""
    return _gmrun(pr, lane) if _needs_gmrun(pr) else _aligned_mean(pr, lane)


def _stage_xses(plan: Plan, script_stem: str) -> str:
    """stage08: bring every session into the reference session's space, then build
    THE grandmean from the result.

    Always emitted, including single-session — the grandmean has exactly one
    producer that way, and it is always the post-alignment one. With one session
    the alignment loop is empty and the grandmean is just that session's mean.
    """
    opt = plan.options
    sessions = [s for s in _sessions(plan) if s is not None]
    ref = plan.ref_session
    primary = _primary_lane(plan)
    lane_tag = f".src-{_src(primary)}" if _src(primary) else ""
    out = ["", "# ============================ stage08: cross-session align + grandmean ======"]
    if plan.multi_session:
        out.append("echo '== stage08: cross-session alignment =='")
        note = {
            LANE_SBREF: "SBRef session means (sharper, single-band contrast)",
            LANE_MAX: "session MAX images (every voxel any frame of any run saw, "
            "so a run that lost an edge to motion does not drag the alignment)",
        }.get(primary)
        if note:
            out.append(
                f"# Estimated on the {note};\n"
                "# every other lane is resampled by that same transform."
            )
        if opt.ref_image:
            out.append(f"# -ref_image {opt.ref_image}: each session is represented by that image.")
    # Base is the reference session's representative; every candidate image sits on
    # that session's common grid, so the one transform lands any lane.
    out.append(f'REFGM="{_ref_image_file(plan, ref, primary)}"')
    # The grid the grandmean is built on (always a session mean — it exists for
    # every session and every lane, whatever -ref_image chose to align with).
    out.append(f'GMGRID="{_sesmean(ref, primary)}"')
    if plan.multi_session:
        # The reference session has no xses transform (it maps to itself) — emit its
        # representative under the xses name so `ls stage08.xses.*` shows every session.
        out.append(
            _ref_marker(
                NameKey("xses", session=ref, role="ref"),
                _ref_image_file(plan, ref, primary),
                f"QC: ses-{ref} is the reference session — no transform, shown for completeness.",
            )
        )
    # Same batching as stage06: the per-session manifests are written first, then
    # ONE ffs_allineate (and one nonlinear backend) process handles every session.
    nonref = [s for s in sessions if s != ref]
    if nonref:
        out.append(f'albatch="{script_stem}_xsesbatch.txt"; : > "$albatch"')
        if opt.xses_nonlin:
            out.append(f'fwbatch="{script_stem}_xsesnlbatch.txt"; : > "$fwbatch"')
    for s in nonref:
        xstem = stem(NameKey("xses", session=s))
        out.append(
            _manifest_line(
                "albatch",
                [
                    '-base "$REFGM"',
                    f'-source "{_ref_image_file(plan, s, primary)}"',
                    f'-prefix "{xstem}{lane_tag}_lin.nii$FMT"',
                    f'-1Dmatrix_save "{xstem}.aff12.1D"',
                    *_split_flags(config.DEFAULT_OPTS["xses"]),
                ],
                indent="",
            )
        )
        if opt.xses_nonlin:
            # Distinct `_nl` output — don't overwrite the linear-aligned source.
            out.append(
                _manifest_line(
                    "fwbatch",
                    [
                        '-base "$REFGM"',
                        *_nl_source_args(
                            opt.xses_nonlin_in_source,
                            f"{xstem}{lane_tag}_lin.nii$FMT",
                            _ref_image_file(plan, s, primary),
                            f"{xstem}.aff12.1D",
                        ),
                        f'-prefix "{xstem}{lane_tag}_nl.nii$FMT"',
                        "-save_warp",
                        # Lane-free warp: it is applied to every lane (see xrun).
                        f'-warp_prefix "{xstem}_nl"',
                        *_split_flags(config.DEFAULT_OPTS["xses_nl"]),
                    ],
                    indent="",
                )
            )
    if nonref:
        out.append(_batch_launch("ffs_allineate", "albatch", "skip_xses"))
        if opt.xses_nonlin:
            out.append(_batch_launch(_nl_tool(opt, "xses_nl"), "fwbatch", "skip_xses"))
    out.append(_qc_xses(plan))
    out.append(_stage_gmrun(plan, script_stem))

    # THE grandmean, composited from every run's image in grandmean space (one
    # interpolation from moco for all of them — see _stage_gmrun). Per lane: the
    # mean lanes average, the max lane takes the max of the maxes (the most brain
    # in the most places), the min lane the min of the mins (where every frame of
    # every run has data). This is the default target the anat alignment (stage09)
    # uses; the SBRef lane's sibling is what -anat_source sbmean selects.
    out.append("echo '== stage08: grandmean (all runs, in one space) =='")
    out += _math_batch(
        "gmeanbatch",
        f"{script_stem}_grandmeanbatch.txt",
        [
            [
                f"-input {' '.join(chr(34) + _gmrun_image(pr, lane) + chr(34) for pr in plan.runs)}",
                _lane_reduce(lane),
                f'-prefix "{_grandmean(lane)}"',
                "-overwrite",
            ]
            for lane in _lanes(plan)
        ],
    )
    out.append(_qc_grandmean(plan))
    return "\n".join(p for p in out if p) + "\n"


def _stage_gmrun(plan: Plan, script_stem: str) -> str:
    """The interpolation checkpoint: every run that needs an xses link is resampled
    from its post-moco lane image into grandmean space in ONE pass.

    Without this the grandmean is a mean of session means, and a non-reference
    session's contribution has been interpolated three times (moco resample →
    runmean → xses) before it lands. That blur is not cosmetic: the grandmean is
    what the anat step, ffs_segment and every -grand_reference align to, so it sets
    the sharpness ceiling for everything downstream. Composing the pre-chain with
    xses costs the same single resample the runmean already pays, so the whole
    grandmean sits one interpolation from moco, uniformly.

    Runs of the reference session have no xses link — their grandmean chain IS
    their pre-chain — so they contribute the runmean they already have, and a
    single-session dataset emits nothing here at all.
    """
    todo = [pr for pr in plan.runs if _needs_gmrun(pr)]
    if not todo:
        return ""
    batch = f"{script_stem}_gmrunbatch.txt"
    out = [
        "",
        "# ---------------------------- stage08b: runs → grandmean space -------------",
        "# One resample each: the whole pre-chain composed with this run's xses link,",
        "# straight from the post-moco image. Keeps the grandmean one interpolation",
        "# from moco instead of three. Batched like every other per-run stage.",
        "echo '== stage08b: runs → grandmean space (single resample) =='",
        f'gmbatch="{batch}"; : > "$gmbatch"',
    ]
    for pr in todo:
        chain = " ".join(chain_files(pr, ".nii$FMT", plan.options, tokens=_grandmean_tokens(pr)))
        jac = _jac_spec(pr)
        for lane in _lanes(plan):
            out.append(
                _manifest_line(
                    "gmbatch",
                    [
                        f'-source "{_run_level(pr, lane)}"',
                        f'-nwarp "{chain}"',
                        *([f'-jac "{jac}"'] if jac else []),
                        '-master "$GMGRID"',
                        *_lane_nwarp_flags(lane),
                        f'-prefix "{_gmrun(pr, lane)}"',
                    ],
                    indent="",
                )
            )
    out.append(_batch_launch("ffs_nwarp", "gmbatch", "skip_xses"))
    return "\n".join(out)


def _pe_axis(plan: Plan) -> str:
    for pr in plan.runs:
        pe = pr.bold.pe_dir
        if pe:
            letter = "".join(c for c in pe if c.isalpha())
            if letter:
                return {"i": "x", "j": "y", "k": "z"}.get(letter, letter)
    return "y"


def _ref_target(opt) -> str | None:
    """The EPI-contrast image this data's grandmean is aligned to (xref base):
    the explicit ``-ref_file`` if given, else the grand_reference dir's grandmean."""
    if opt.ref_file is not None:
        return opt.ref_file
    if opt.grand_reference is not None:
        return f"{opt.grand_reference.rstrip('/')}/{_grandmean()}"
    return None


def _stage_xref(plan: Plan) -> str:
    opt = plan.options
    if not (opt.go_to_anat and opt.has_grand_ref):
        return ""
    target = _ref_target(opt)
    nl = ""
    if opt.grand_reference_nonlin:
        nl = (
            _ffs(
                _nl_tool(opt, "xses_nl"),
                [
                    '-base "$REFGM_EXT"',
                    '-source "stage09.xref_lin.nii$FMT"',
                    '-prefix "stage09.xref_nl.nii$FMT"',
                    "-save_warp",
                    *_split_flags(config.DEFAULT_OPTS["xses_nl"]),
                    '-device "$DEVICE"',
                ],
            )
            + "\n"
        )
    lin = _ffs(
        "ffs_allineate",
        [
            '-base "$REFGM_EXT"',
            f'-source "{_grandmean()}"',
            '-prefix "stage09.xref_lin.nii$FMT"',
            '-1Dmatrix_save "stage09.xref.aff12.1D"',
            *_split_flags(config.DEFAULT_OPTS["xses"]),
            '-device "$DEVICE"',
        ],
    )
    return f"""
# ============================ stage09: align to reference ===================
# Align THIS data's grandmean to the reference EPI contrast, so the reference's
# anat matrix/transforms apply. This is how a filtered `primary`-only script
# anchors on a separately-processed floc run. For -grand_reference DIR, adjust
# the ext if that dir wrote a different working format.
echo '== stage09: align to reference =='
REFGM_EXT="{target}"
if [ "$skip_anat" -ne 1 ] || [ ! -f "stage09.xref.aff12.1D" ]; then
{lin}
{nl}fi
{_qc_xref(plan)}
"""


def _stage_anat(plan: Plan) -> str:
    opt = plan.options
    if not opt.go_to_anat:
        return (
            "\n# ============================ stage09: anat (skipped) ======================\n"
            "# -no_anat / no anat given: final space is the EPI grandmean.\n"
        )
    out = ["", "# ============================ stage09: anatomical alignment ================="]
    out.append("echo '== stage09: anat alignment =='")
    # tpm_source default: an explicit -ref_anat (copied in) doubles as the anat in
    # tpm space, useful for ffs_segment QC; else the user's -tpm_source.
    tpm_src_path = opt.tpm_source
    if opt.ref_file is not None:
        # Explicit reference override: transforms are user-supplied; copy ref_anat
        # in (tpm-space anat + QC). anat_lin links = -ref_transforms (see chain).
        refmats = " ".join(shlex.quote(m) for m in (opt.ref_transforms or []))
        out.append("# explicit reference: -ref_transforms map ref→anat (baked into CHAIN).")
        if opt.ref_anat:
            out.append(f"cp -f {shlex.quote(opt.ref_anat)} stage09.ref_anat.nii.gz")
            tpm_src_path = tpm_src_path or "stage09.ref_anat.nii.gz"
        seg_mats = f"{refmats} stage09.xref.aff12.1D".strip()
    elif opt.grand_reference:
        gr = opt.grand_reference.rstrip("/")
        # anat matrix is BORROWED from the reference; segment (if any) inits from
        # borrowed ref2anat ∘ computed xref (matches the primary reference chain).
        out.append(f"# anat matrix borrowed from {gr}/stage09.anat.aff12.1D (not recomputed).")
        seg_mats = f"{gr}/stage09.anat.aff12.1D stage09.xref.aff12.1D"
    else:
        anat_ph = (
            opt.anat_path
            or "${FFS_ANAT:?set FFS_ANAT to the skull-stripped T1w (SUMA brain.nii.gz)}"
        )
        src = _anat_source_image(plan)
        mode = effective_anat_source(plan)
        box = _anat_box()
        boxing = _ffs(
            "ffs_util_autobox",
            ['-input "$ANAT"', "-npad 3", f'-prefix "{box}"', '-device "$DEVICE"'],
        )
        lin = _ffs(
            "ffs_allineate",
            [
                f'-base "{box}"',
                f'-source "{src}"',
                f'-prefix "{_al_anat(plan)}"',
                '-1Dmatrix_save "stage09.anat.aff12.1D"',
                *_split_flags(config.DEFAULT_OPTS["anat"]),
                "-tbest 16",
                '-device "$DEVICE"',
            ],
        )
        note = ""
        if opt.anat_source not in (mode, "auto"):
            note = f"  # (-anat_source {opt.anat_source} unavailable here → {mode})\n"
        out.append(
            f'ANAT="{anat_ph}"\n'
            "# Crop the anat's air away first: the alignment base, every EPI-in-anat\n"
            "# output, and the final grid all inherit this FOV, and the aff12 matrix is\n"
            "# in DICOM mm — cropping the base moves nothing, it only shrinks the search\n"
            "# space and the volume-sized allocations. Doubles as the viewing underlay.\n"
            f'[ -f "{box}" ] || \\\n{boxing}\n'
            'if [ "$skip_anat" -ne 1 ] || [ ! -f "stage09.anat.aff12.1D" ]; then\n'
            "  # cross-modal rigid lpc: base=anat → matrix maps anat→EPI (chain head).\n"
            f"  # source = -anat_source {mode} (all choices share the reference-fmap grid).\n"
            f"{note}{lin}\nfi"
        )
        seg_mats = "stage09.anat.aff12.1D"

    if opt.anat_nonlin:
        # Optionally build a subject TPM from FreeSurfer (SUMA) first, so no SPM
        # TPM is needed ahead of time.
        if opt.fs_tpm:
            out.append(_fs_tpm_block(opt))
        # One last ffs_segment on the CURRENT data (PE-mode nonlinear anat), init
        # from the anat matrix/matrices. Needs a subject TPM.
        tpm = (
            shlex.quote(opt.tpm)
            if opt.tpm
            else '"${FFS_TPM:?set FFS_TPM to the subject tissue-probability template}"'
        )
        # The FS-derived TPM has 8 hard-edge classes → its own ngaus/cleanup params.
        seg_tune = _split_flags(config.DEFAULT_OPTS["segment_fstpm" if opt.fs_tpm else "segment"])
        seg = _ffs(
            "ffs_segment",
            [
                _segment_input(plan),
                f"-pe_axis {_pe_axis(plan)}",
                f"-tpm {tpm}",
                (f"-tpm_source {shlex.quote(tpm_src_path)}" if tpm_src_path else ""),
                f"-1Dmatrix {seg_mats}",
                '-prefix "stage09.nlanat.nii$FMT"',
                *seg_tune,
                '-device "$DEVICE"',
            ],
        )
        out.append(
            f'if [ "$skip_anat" -ne 1 ] || [ ! -f "stage09.nlanat_invwarp.nii$FMT" ]; then\n'
            f"{seg}\nfi"
        )
        out.append(_anat_nl_anchor_call(plan))
    out.append(_qc_anat(plan))
    return "\n".join(p for p in out if p) + "\n"


def _fs_tpm_block(opt) -> str:
    """Build a subject-specific 8-class TPM from FreeSurfer (SUMA) outputs using
    AFNI (aseg.auto label sets + a skull class from SurfVol). Hard-edge tissue
    priors in FS space — no ahead-of-time SPM TPM needed. Runs in a subshell so
    the ``cd`` doesn't leak. Output: tpm_work/sub_specific_fsTPM.nii.gz."""
    suma = shlex.quote(opt.suma_dir.rstrip("/")) if opt.suma_dir else "${SUMA_DIR:?set -suma}"
    labels = [
        ("tpm1_cortex", "3,42"),
        ("tpm2_cerebelgm", "8,47"),
        ("tpm3_cortwm", "2,41,28,60,16,252,251,253,254,255"),
        ("tpm4_cerebelwm", "7,46"),
        ("tpm5_subcortexgm", "5,10,11,12,13,17,18,49,50,51,52,53,54,44,58,26"),
        ("tpm6_vents", "4,14,43"),
        ("tpm7_outercsf", "24"),
    ]
    calc = "\n".join(
        f"  3dcalc -overwrite -a {suma}/aseg.auto.nii.gz'<{sel}>' -expr 'step(a)' -prefix {name}.nii.gz"
        for name, sel in labels
    )
    return f"""# --- build subject TPM from FreeSurfer (SUMA) ---
if [ ! -f "tpm_work/sub_specific_fsTPM.nii.gz" ]; then
  mkdir -p tpm_work
( cd tpm_work
{calc}
  3dMean -sum -overwrite -prefix sum_sub_tpm.nii.gz tpm[1-7]_*.nii.gz
  # skull class = whatever SurfVol has outside the labelled tissue.
  3dcalc -overwrite -a sum_sub_tpm.nii.gz -b {suma}/*_SurfVol.nii \\
    -expr 'abs(step(a)-1)*b' -prefix skull.nii.gz
  3dAutomask -overwrite -prefix tpm8_skull.nii.gz skull.nii.gz
  3dTcat -overwrite -prefix sub_specific_fsTPM.nii.gz tpm[1-8]_*.nii.gz
)
fi"""


def _segment_input(plan: Plan) -> str:
    """ffs_segment ``-input`` args per ``-anat_nonlin_input``.

    grandmean  → the EPI grandmean (works even with no fieldmap; segment does the
                 distortion correction implicitly against the anat/TPM).
    sbmean /
    ref_fmap /
    mean_fmap  → the same anchor images ``-anat_source`` selects (shared vocabulary).
    blipfor    → the first fmap group's undistorted forward frame.
    blip_pair  → forward ``[0]`` + reverse ``[1]`` of the blipflip unwarped pair
                 (topup-style; matches the reference `primary` script).
    Falls back to grandmean (with a note) when no fmap exists, and likewise for a
    GRE fieldmap: both modes read the blipflip unwarped *pair*, which a measured
    field has no counterpart to — there is only ever one EPI.
    """
    mode = plan.options.anat_nonlin_input
    blip = None
    for pr in plan.runs:
        if pr.fmap is not None and not pr.fmap.is_b0:
            blip = _fmap_stem(pr, "blip")
            break
    if mode in ("blipfor", "blip_pair") and blip is None:
        why = "no reverse-PE fmap" if any(pr.fmap for pr in plan.runs) else "no fmap"
        return f'-input "{_grandmean()}"  # ({why}; fell back from {mode})'
    if mode == "blipfor":
        return f'-input "{blip}_unwarped.nii$FMT[0]"'
    if mode == "blip_pair":
        return f'-input "{blip}_unwarped.nii$FMT[0]" -pe_reverse "{blip}_unwarped.nii$FMT[1]"'
    return f'-input "{_anat_source_image(plan, mode)}"'


def _own_anat(opt) -> bool:
    """True when this pipeline computes its OWN anat matrix (an ``$ANAT`` bash var
    is defined in stage09) — as opposed to borrowing (-grand_reference) or
    overriding (-ref_file) it. Gates the anat-derived underlays."""
    return opt.go_to_anat and opt.ref_file is None and opt.grand_reference is None


def _anat_box() -> str:
    """The anat cropped to its own brain — the alignment base (stage09), the final
    grid's ancestor, and the whole-brain viewing underlay.

    Plain .gz, not $FMT: this one is for looking at, and stock AFNI cannot open
    .nii.zst."""
    return "stage09.anat_autobox.nii.gz"


def _al_anat(plan: Plan) -> str:
    """The EPI anchor resampled into anat space by the stage09 alignment.

    Named for the source that made it (``stage09.grandmean_al_anat``,
    ``stage09.sbmean_al_anat``, …): it is EPI data sitting on the anat grid, not
    an anatomical, and the distinction decides what you should be looking at.

    This one is the AFFINE result — ffs_allineate's own output, which is what the
    cross-modal alignment QC wants to show. With -anat_nonlin it is NOT the state
    the data ends up in; see :func:`_al_anat_nl`."""
    return f"stage09.{effective_anat_source(plan)}_al_anat.nii$FMT"


def _al_anat_nl(plan: Plan) -> str:
    """The same anchor with the nonlinear anat link applied too (-anat_nonlin).

    ``anat_nl`` (the ffs_segment invwarp) is estimated *after* the affine, so
    ffs_allineate's ``_al_anat`` output is one link short of what every run's
    CHAIN actually does. That link is a real, PE-axis-only deformation of several
    voxels, so an ``_al_anat`` derived grid/mask/QC describes a brain shape the
    data never lands in. This image is the anchor pushed through the whole head
    of the chain, and is what stage10a boxes, masks and QCs against."""
    return f"stage09.{effective_anat_source(plan)}_al_anat_nl.nii$FMT"


def _anat_head_chain(plan: Plan) -> list[str]:
    """The chain links that act at or above ``anat_nl`` — the ones the anat-source
    anchor still needs.

    The anchor already has everything below ``anat_nl`` baked in (it lives on the
    grandmean/reference-session grid), so applying exactly this head slice puts it
    where the data ends up. Resolved through :func:`chain_files` off a run's own
    ``warp_chain``, so the borrowed-matrix and external-reference modes get their
    extra links (``xref_*``) for free instead of being special-cased."""
    if not plan.runs:
        return []
    pr = plan.runs[0]
    toks = list(pr.warp_chain)
    if "anat_nl" not in toks:
        return []
    head = toks[: toks.index("anat_nl") + 1]
    return chain_files(pr, ".nii$FMT", plan.options, tokens=head)


def _anat_nl_anchor_call(plan: Plan) -> str:
    """The ffs_nwarp that builds :func:`_al_anat_nl`, or "" when there is nothing
    to apply.

    Only for a locally-computed anat: the borrowing modes take their grid from the
    reference results dir, so the link that matters there is the *reference's*
    anat_nl, already baked into the image :func:`_final_master` globs."""
    if not (plan.options.anat_nonlin and _own_anat(plan.options)):
        return ""
    head = _anat_head_chain(plan)
    if not head:
        return ""
    out = _al_anat_nl(plan)
    call = _ffs(
        "ffs_nwarp",
        [
            f'-source "{_anat_source_image(plan)}"',
            f'-nwarp "{" ".join(head)}"',
            f'-master "{_anat_box()}"',
            *_split_flags(config.DEFAULT_OPTS["nwarp"]),
            f'-prefix "{out}"',
            '-device "$DEVICE"',
        ],
    )
    return (
        "# The anchor through the SAME head-of-chain the data gets (affine + the\n"
        "# segment invwarp). ffs_allineate's _al_anat stops at the affine, so it is\n"
        "# the wrong shape to box the output grid or automask from.\n"
        f'[ -f "{out}" ] || \\\n{call}'
    )


def _final_master(plan: Plan) -> str:
    """The dataset whose grid (space/FOV) the final output inherits — before the
    warpmaster autoboxes it to the EPI coverage and drops it to the EPI voxel size."""
    opt = plan.options
    if not opt.go_to_anat:
        return _grandmean()
    if opt.ref_file is not None:
        # explicit reference: the copied-in ref anat defines the shared grid.
        return "stage09.ref_anat.nii.gz" if opt.ref_anat else _grandmean()
    if opt.grand_reference:
        # Borrow the reference's anat-space grid so all subjects/tasks co-register.
        # Globbed because the reference's file is named for ITS -anat_source, which
        # this script has no way to know.
        gr = opt.grand_reference.rstrip("/")
        # `|| true` so a miss leaves MASTER empty for the guard below to report,
        # rather than tripping `set -o pipefail` with no explanation.
        # _al_anat_nl first: with -anat_nonlin that is the shape the reference's
        # own data landed in, and the affine-only _al_anat is one link short of it.
        return (
            f"$(ls {gr}/stage09.*_al_anat_nl.nii* {gr}/stage09.*_al_anat.nii* "
            "2>/dev/null | head -1 || true)"
        )
    return _al_anat_nl(plan) if opt.anat_nonlin else _al_anat(plan)


def _final_dxyz_default(plan: Plan) -> str:
    """Final output voxel size: -final_dxyz if given, else the input EPI resolution
    read from the first run's in-plane dim."""
    opt = plan.options
    if opt.final_dxyz:
        return opt.final_dxyz
    first_mag = shlex.quote(str(plan.runs[0].bold.mag_path)) if plan.runs else '"$raw"'
    return f"$(ffs_info -adi {first_mag})"


def _stage_warpmaster(plan: Plan) -> str:
    """Pin the final output grid and analysis mask BEFORE the resample (stage10).

    The warpmaster is the anat-space EPI target autoboxed to the EPI's own
    coverage and resampled to the EPI voxel size — it fixes the exact position +
    spacing every run's timeseries lands on (stage10 ``-master``). That target has
    to be the anchor in the state the *data* ends up in: with -anat_nonlin it is
    :func:`_al_anat_nl`, not ffs_allineate's affine-only ``_al_anat``, or the
    output FOV, the automask and the final QC's sub-brick 0 all describe a brain
    shape several PE-axis voxels away from the one the runs land in. The EPI FOV is
    usually a slab, not the whole head, so this crop is what keeps the output from
    carrying the anat's empty space. epi_mask is a dilated automask on that grid,
    used to mask the GLM. Two anat underlays come out of this: the whole-brain
    stage09.anat_autobox and stage10.anat_in_epi_fov, the same anat at anat
    resolution over the warpmaster's FOV — the one to overlay results on. MASTER
    and FINAL_DXYZ are defined here once and reused by stage10 (same shell); the
    FFS_MASTER / FFS_FINAL_DXYZ overrides are still honoured."""
    opt = plan.options
    box = "stage10.warpmaster_box.nii$FMT"
    wm = "stage10.warpmaster.nii$FMT"
    mask = "epi_mask.nii$FMT"
    anat_fov = "stage10.anat_in_epi_fov.nii.gz"

    def guarded(outfile: str, tool: str, parts: list[str]) -> str:
        return f'[ -f "{outfile}" ] || \\\n' + _ffs(tool, parts)

    out = [
        "",
        "# ============================ stage10a: warpmaster + mask ==================",
        "# Fix the final output grid and analysis mask before the resample. The",
        "# warpmaster is the anat-space EPI target autoboxed to the EPI's own coverage",
        "# and resampled to the EPI voxel size; stage10 lands every run on it (-master).",
        "# epi_mask is a dilated automask on that grid (masks the GLM).",
    ]
    if _own_anat(opt):
        out += [
            "# stage10.anat_in_epi_fov is the anat on that same FOV at anat resolution —",
            "# the underlay for results; stage09.anat_autobox is the whole-brain one.",
        ]
    out += [
        "echo '== stage10a: warpmaster + mask =='",
        # Defined once here and reused by stage10; FFS_* overrides still win.
        f'MASTER="${{FFS_MASTER:-{_final_master(plan)}}}"',
        '[ -n "$MASTER" ] || { echo "stage10a: no master dataset; set FFS_MASTER" >&2; exit 1; }',
        f'FINAL_DXYZ="${{FFS_FINAL_DXYZ:-{_final_dxyz_default(plan)}}}"',
    ]
    out.append(
        guarded(
            box,
            "ffs_util_autobox",
            ['-input "$MASTER"', "-npad 5", f'-prefix "{box}"', '-device "$DEVICE"'],
        )
    )
    if _own_anat(opt):
        # Pure crop: the box came from a dataset on the anat grid, so this shares
        # its voxel lattice and -rmode never fires.
        out.append(
            guarded(
                anat_fov,
                "ffs_util_resample",
                [
                    f'-input "{_anat_box()}"',
                    f'-master "{box}"',
                    f'-prefix "{anat_fov}"',
                    '-device "$DEVICE"',
                ],
            )
        )
    # resample -dxyz needs three values; FINAL_DXYZ is one number (isotropic).
    out.append(
        guarded(
            wm,
            "ffs_util_resample",
            [
                f'-input "{box}"',
                f'-prefix "{wm}"',
                "-dxyz $FINAL_DXYZ $FINAL_DXYZ $FINAL_DXYZ",
                "-rmode wsinc5",
                '-device "$DEVICE"',
            ],
        )
    )
    out.append(
        guarded(
            mask,
            "ffs_util_automask",
            [f'-input "{wm}"', f'-prefix "{mask}"', "-dilate 2", '-device "$DEVICE"'],
        )
    )
    return "\n".join(out) + "\n"


def _stage_final(plan: Plan, script_stem: str) -> str:
    opt = plan.options
    if opt.slicetiming_method == "integrate":
        st = (
            f"  tp={_tpattern(plan)}; " + 'tr="${TR[$k]}"\n'
            '  if [ -n "$tr" ]; then st_str="-tpattern \\"$tp\\" -TR \\"$tr\\" -tzero 0"; '
            'else st_str=""; fi'
        )
    else:
        st = '  st_str=""'
    nwarp_flags = " ".join(_split_flags(config.DEFAULT_OPTS["nwarp"]))
    batchfile = f"{script_stem}_nwarpbatch.txt"
    # Phase rides the magnitude's chain in the same single interpolation.
    # -phase_warp direct warps magnitude and phase independently: the magnitude
    # output is bit-for-bit what a phase-free run would produce, and it is the
    # correct mode for phase that is already unwrapped (no wraps to smear).
    # -batch_skip counts -phase_prefix among a job's required outputs.
    phase_args = ""
    phase_note = ""
    if _phase_on(plan):
        phase_args = (
            f' -phase \\"{_phase_source(plan)}\\" -phase_units rad -phase_warp direct'
            f' -phase_prefix \\"$phoutf\\"'
        )
        phase_note = (
            "# The unwrapped phase rides the same chain in the same pass\n"
            "# (-phase_warp direct, radians) → stage10.final.<frag>.part-phase.*\n"
        )
    phase_out = (
        '  phoutf="stage10.final.${FRAG[$k]}.part-phase.nii$FINAL_FMT"\n' if _phase_on(plan) else ""
    )
    # -save_mean is not optional: mean_stage10.final.<frag> is the per-run image
    # the final QC stack is built from, and it costs one temporal reduction of a
    # series ffs_nwarp already has in memory.
    return f"""
# ============================ stage10: compose + resample ===================
# Batched: ONE ffs_nwarp process resamples every run (startup paid once). Each
# run's arguments are appended to {batchfile} and consumed with -batch. One
# interpolation per run applies the whole CHAIN in a single pass; the source is
# read in place (NORDIC output, or BIDS magnitude with noise vols trimmed inline)
# so no raw copy is materialised. The chain lands every run on the stage10a
# warpmaster grid (MASTER/FINAL_DXYZ set there). skip_final=1 → -batch_skip.
# -save_mean writes mean_stage10.final.<frag> per run — the QC stack below.
{phase_note}echo '== stage10: final compose + resample =='
nwarpbatch="{batchfile}"
: > "$nwarpbatch"
for k in "${{RUN_KEYS[@]}}"; do
  outf="stage10.final.${{FRAG[$k]}}.nii$FINAL_FMT"
{phase_out}{st}
{_raw_source(plan)}
  printf '%s\\n' "-source \\"$raw\\" -nwarp \\"${{CHAIN[$k]}}\\"${{JAC[$k]:+ -jac \\"${{JAC[$k]}}\\"}} -master stage10.warpmaster.nii$FMT -dxyz \\"$FINAL_DXYZ\\" {nwarp_flags} $st_str -save_mean -prefix \\"$outf\\"{phase_args}" >> "$nwarpbatch"
done
{_sbref_final_jobs(plan)}batch_skip=(); [ "$skip_final" -eq 1 ] && batch_skip=(-batch_skip)
ffs_nwarp -batch "$nwarpbatch" "${{batch_skip[@]}}" -device "$DEVICE"
echo 'done → stage10.final.*'
{_qc_final(plan)}
"""


def _sbref_final_jobs(plan: Plan) -> str:
    """Extra stage10 batch entries putting every run's SBRef in the final space.

    Same batch, same master/grid, same transforms as its BOLD — minus the
    within-run motion tokens the SBRef defines rather than needs (SBCHAIN). No
    slice-timing argument: a 3-D image has no time axis to shift.

    These are the sharpest per-run images the pipeline can produce in output
    space, which makes them the QC of record for cross-run and cross-session
    alignment: overlay them and any residual misregistration is visible directly,
    not inferred from a motion-blurred mean.
    """
    if not plan.use_sbref:
        return ""
    nwarp_flags = " ".join(_split_flags(config.DEFAULT_OPTS["nwarp"]))
    # An empty SBCHAIN means this SBRef needs no transform at all — the anchor run
    # of a no-anat, no-fieldmap, single-session plan, whose own space IS the final
    # space. ffs_nwarp has no identity warp, so that one is a plain regrid.
    return (
        "# SBRefs into the final space (QC; same chain minus the moco tokens).\n"
        'for k in "${RUN_KEYS[@]}"; do\n'
        '  sboutf="stage10.final.${FRAG[$k]}.src-sbref.nii$FINAL_FMT"\n'
        '  if [ -z "${SBCHAIN[$k]:-}" ]; then\n'
        '    [ -f "$sboutf" ] || ffs_util_resample -input "${SBREF[$k]}" \\\n'
        "      -master stage10.warpmaster.nii$FMT -rmode wsinc5 \\\n"
        '      -prefix "$sboutf" -device "$DEVICE"\n'
        "    continue\n"
        "  fi\n"
        '  printf \'%s\\n\' "-source \\"${SBREF[$k]}\\" -nwarp \\"${SBCHAIN[$k]}\\"'
        '${JAC[$k]:+ -jac \\"${JAC[$k]}\\"} '
        '-master stage10.warpmaster.nii$FMT -dxyz \\"$FINAL_DXYZ\\" '
        f'{nwarp_flags} -prefix \\"$sboutf\\"" >> "$nwarpbatch"\n'
        "done\n"
    )


def _stage_unwrap(plan: Plan) -> str:
    """Temporal phase unwrapping (ROMEO) — the one thing done to phase before the
    final resample.

    Unwrapping happens as early as possible, in native space, while the phase is
    still voxelwise aligned to its own magnitude: ROMEO's spatial+temporal
    unwrapping wants the magnitude as a quality weight, and every later stage
    resamples. After this the phase is in **radians**, monotone in time, and is
    simply carried; the warp chain is estimated from magnitude alone.

    With NORDIC the input is the denoised mag+phase pair (NORDIC has already
    trimmed the noise volumes). Without NORDIC the raw BIDS pair is used, and the
    noise volumes must come off *both* first — ROMEO reads files, not sub-brick
    selectors, so this is the one stage that materialises a copy of the raw data.
    """
    if not _phase_on(plan):
        return ""
    opt = plan.options
    romeo_opts = " ".join(_split_flags(config.DEFAULT_OPTS["unwrap"]))
    uw = _unwrapped()
    if opt.want_nordic:
        src = f'  mag="{_nordic_mag(plan)}"; ph="{_nordic_phase()}"'
        note = "# Inputs: the NORDIC-denoised pair (noise volumes already trimmed)."
    else:
        trim = (
            f'    tm="{_phase_file("trim", part=False)}"\n'
            f'    tp="{_phase_file("trim")}"\n'
            f'    nv=$(ffs_info -nv "$mag"); last=$((nv - NOISE_VOLS - 1))\n'
            f'    [ -f "$tm" ] || ffs_util_3dmath -input "${{mag}}[0..$last]" -expr a '
            f'-prefix "$tm" -device cpu\n'
            f'    [ -f "$tp" ] || ffs_util_3dmath -input "${{ph}}[0..$last]" -expr a '
            f'-prefix "$tp" -device cpu\n'
            f'    mag="$tm"; ph="$tp"\n'
        )
        src = (
            '  mag="${MAG[$k]}"; ph="${PHASE[$k]}"\n'
            '  if [ "$NOISE_VOLS" -gt 0 ]; then\n'
            f"{trim}"
            "  fi"
        )
        note = (
            "# Inputs: the raw BIDS pair. With NOISE_VOLS>0 both are trimmed into\n"
            "# stage00.trim.* first (ROMEO takes files, not [0..n] selectors)."
        )
    return f"""
# ============================ stage00: phase unwrap (ROMEO) =================
# -phase_proc: temporally unwrap the phase up front, in native space, while it is
# still voxelwise aligned to its magnitude. Output is radians; nothing touches the
# phase again until the final resample (stage10) carries it along the magnitude's
# warp chain. ROMEO is external (MRItools) — it must be on $PATH.
{note}
echo '== stage00: phase unwrap (ROMEO) =='
for k in "${{RUN_KEYS[@]}}"; do
  uw="{uw}"
  [ "$skip_unwrap" -eq 1 ] && [ -f "$uw" ] && continue
{src}
  romeo -p "$ph" -m "$mag" -o "$uw" {romeo_opts}
done
"""


def _stage_stats(plan: Plan, bids_root: str | None) -> str:
    """One GLM per task.

    The model lives in that task's design TOML (written by ffs_autoproc at
    generation time, see autoproc/glm.py) and the command is `ffs_reml -spec` —
    so changing conditions, HRF, nuisance blocks or contrasts is an edit to one
    annotated file, not a hunt through generated bash. Tasks whose events could
    not be resolved fall back to the flag form with a TODO, because there is
    nothing to build a spec from.
    """
    from fastfuncstuff.autoproc.glm import nuisance_specs, runs_by_task, spec_path

    opt = plan.options
    tasks = runs_by_task(plan)
    gate = "1" if opt.run_glm else "0"
    out = ["", "# ============================ stage12: GLM (ffs_reml) ======================="]
    out.append(f"# One model per task. Runs when FFS_RUN_GLM=1 (default {gate} for this recipe).")
    out.append(f'if [ "${{FFS_RUN_GLM:-{gate}}}" = "1" ]; then')
    for task, prs in tasks.items():
        finals = " ".join(f'"stage10.final.{_frag(pr)}.nii$FINAL_FMT"' for pr in prs)
        resolved = events_for_task(task, prs, bids_root, opt)
        common = [
            f'-Obuck "stage12.stats-ols.task-{task}.nii$GLM_FMT"',
            f'-Rbuck "stage12.stats-reml.task-{task}.nii$GLM_FMT"',
            "-tout",
            "-fout",
            "-mask epi_mask.nii$FMT",
            "-do_scale",
            # -TR only when the user gave one: a 3D acquisition's header TR is the
            # per-partition time, not the volume TR the design is sampled at.
            *([f"-TR {opt.tr:g}"] if opt.tr is not None else []),
            *([f"-drop_first {opt.glm_drop_first}"] if opt.glm_drop_first else []),
            *([f"-drop_last {opt.glm_drop_last}"] if opt.glm_drop_last else []),
            *(_split_flags(opt.glm_opts) if opt.glm_opts else []),
            '-device "$DEVICE"',
        ]
        if resolved:
            spec = spec_path(task)
            _, skipped = nuisance_specs(task, opt)
            out.append(f"# task-{task}: model = {spec} (edit that, not this command).")
            if skipped:
                out.append(
                    f"# NOTE -glm_ortvec {' '.join(skipped)} skipped: the stage that writes "
                    "those regressors is not in this pipeline."
                )
            out.append(_ffs("ffs_reml", [f"-input {finals}", f"-spec {spec}", *common], indent=""))
            continue

        out.append(
            f"# TODO task-{task}: no events TSV found when this script was written, so no"
            "\n#      design spec could be built. The -events value below is a PLACEHOLDER"
            "\n#      GLOB that ffs_reml will not expand — replace it with the real file(s),"
            "\n#      or write a spec with `ffs_design_spec stub` and swap in -spec."
        )
        ort_parts = []
        for name, entry in config.GLM_ORTVEC.items():
            if name not in opt.glm_ortvec:
                continue
            req = entry.get("requires")
            if req and not getattr(opt, req, False):
                continue
            label = (
                name if entry.get("transform", "none") == "none" else f"{name}:{entry['transform']}"
            )
            ort_parts.append(f"-ortvec_glob '{entry['pattern'].format(task=task)}' {label}")
        out.append(
            _ffs(
                "ffs_reml",
                [
                    f"-input {finals}",
                    f"-events {_events_args(task, prs, bids_root, opt)}",
                    *ort_parts,
                    "-polort 3",
                    *common,
                ],
                indent="",
            )
        )
    out.append("fi")
    return "\n".join(out) + "\n"


def events_for_task(task: str, prs: list[PlanRun], bids_root: str | None, opt=None) -> list[str]:
    """Events TSV(s) for a task, one per run where they exist.

    Explicit ``-events`` wins; otherwise each run is resolved through BIDS
    inheritance (``bids.find_events``), which is what handles the entities that
    apply to the image but not the task (``part-mag`` and friends) and the
    shared ``task-<T>_events.tsv`` at a coarser level. Duplicates collapse, so a
    single shared file is emitted once and broadcast by ffs_reml.
    """
    if opt is not None and opt.events:
        return list(opt.events)
    found: list[str] = []
    for pr in prs:
        ev = find_events(pr.bold.mag_path, bids_root)
        if ev is not None and str(ev) not in found:
            found.append(str(ev))
    if not found and bids_root:
        root_ev = Path(bids_root) / f"task-{task}_events.tsv"
        if root_ev.is_file():
            found.append(str(root_ev))
    return found


def events_for_run(pr: PlanRun, bids_root: str | None, opt=None) -> str | None:
    """The events TSV for ONE run, or None — what a per-run tool needs.

    :func:`events_for_task` collapses a task's files into a set for ffs_reml to
    broadcast; ffs_locomoco's task-coupling diagnostic runs on a single series and
    needs the file for THAT run. An explicit ``-events`` is honoured only when it is
    one shared file: a longer list pairs with ``-input`` by position, and the per-run
    loop has already taken that pairing apart.
    """
    if opt is not None and opt.events:
        return str(opt.events[0]) if len(opt.events) == 1 else None
    ev = find_events(pr.bold.mag_path, bids_root)
    return str(ev) if ev is not None else None


def _events_script_path(plan: Plan, pr: PlanRun, bids_root: str | None) -> str | None:
    """:func:`events_for_run`, as the generated script should spell it.

    Prefers the ``stimuli/`` copy when the GLM stage is writing one, so the timing a
    diagnostic was computed against travels with the results directory and matches the
    file the design TOML names. Falls back to the BIDS original when there is no GLM.
    """
    from fastfuncstuff.autoproc.glm import stimuli_map

    src = events_for_run(pr, bids_root, plan.options)
    if src is None:
        return None
    if plan.options.run_glm:
        return stimuli_map(plan, bids_root).get(src, src)
    return src


def _events_args(task: str, prs: list[PlanRun], bids_root: str | None, opt=None) -> str:
    """The ``-events`` argument string, or a placeholder glob when nothing was
    found — the generator warns in that case, and the script carries a TODO."""
    found = events_for_task(task, prs, bids_root, opt)
    if found:
        return " ".join(shlex.quote(p) for p in found)
    root = str(bids_root).rstrip("/") if bids_root else "."
    return shlex.quote(f"{root}/**/*task-{task}*_events.tsv")


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------


def write_script(
    plan: Plan,
    out_dir: str,
    bids_root: str | None = None,
    script_stem: str = "proc",
    invocation: str | None = None,
) -> str:
    """Assemble the full pipeline script text for ``plan``.

    ``script_stem`` is the basename (no extension) of the script being written;
    every batched stage names its manifest file after it (``_mocobatch.txt``,
    ``_xrunbatch.txt`` / ``_xrunnlbatch.txt``, ``_runmeanbatch.txt``,
    ``_xsesbatch.txt`` / ``_xsesnlbatch.txt``, ``_nwarpbatch.txt``) so they sit beside the script's
    outputs and don't collide across sibling subjects.

    ``invocation`` is the ffs_autoproc command line that produced this script; it
    is written into the header, commented out."""
    parts = [
        _header(plan, out_dir, invocation),
        _data_arrays(plan, bids_root),
        _preflight(plan, bids_root),
        _qc_helper(plan, script_stem),
        _stage_nordic(plan),
        _stage_unwrap(plan),
        _stage_tshift(plan),
        _stage_moco(plan, script_stem),
        _stage_locomoco(plan, script_stem),
        _stage_blip(plan, script_stem),
        _stage_xfmap(plan, script_stem),
        _stage_xrun(plan, script_stem),
        _stage_grandmean(plan, script_stem),
        _stage_xses(plan, script_stem),
        _stage_xref(plan),
        _stage_anat(plan),
        _stage_warpmaster(plan),
        _stage_final(plan, script_stem),
        _stage_stats(plan, bids_root),
    ]
    return "\n".join(p for p in parts if p).rstrip() + "\n"
