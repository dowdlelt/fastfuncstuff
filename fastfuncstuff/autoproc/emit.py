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
from fastfuncstuff.autoproc.family import Family, family_dir
from fastfuncstuff.autoproc.naming import STAGE_NUMBERS, NameKey, coord, stem
from fastfuncstuff.autoproc.plan import (
    ABOVE_GRANDMEAN,
    Plan,
    PlanRun,
    effective_anat_source,
    medic_owns_inplane,
    ref_anchor,
    run_average_chain,
    run_noise_vols,
    session_fmap_ids,
    session_ref_mode,
    task_cut,
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


# ---------------------------------------------------------------------------
# Multi-echo. Every per-echo file of a multi-echo run carries ``.echo-N`` (echo 1
# included) and its phase ``.echo-N.part-phase``, as the stage10 finals do; files
# shared by the echoes (matrices, field map, warps, run means) stay unlabeled, and a
# single-echo run has no echo label at all (cli_utils.entity_stem, which every tool
# uses). E1[$k] is ".echo-1" for a multi-echo run and empty otherwise: the
# one-image stages read echo 1 through it. In bash, MAGE["$k|$e"] / PHASEE["$k|$e"]
# hold a multi-echo run's per-echo sources and fall back to MAG / PHASE for a
# single-echo run, so one loop serves both.
# ---------------------------------------------------------------------------
_ECHO_LOOP = (
    'for e in $(seq 1 "${NECHO[$k]:-1}"); do\n'
    '    esfx=""; [ "${NECHO[$k]:-1}" -gt 1 ] && esfx=".echo-$e"'
)
_ECHO_MAG = "${MAGE[$k|$e]:-${MAG[$k]}}"
_ECHO_PHASE = "${PHASEE[$k|$e]:-${PHASE[$k]:-}}"


def _any_me(plan: Plan) -> bool:
    return any(pr.bold.is_multiecho for pr in plan.runs)


def _me_3d(pr: PlanRun, opt) -> bool:
    """A multi-echo 3-D run: its partition shift differs per echo (TE-scaled), so
    moco and locomoco hand stage10 per-echo transforms."""
    return bool(opt is not None and opt.acq_3d and pr.bold.is_multiecho)


def _lm_phase(plan: Plan) -> bool:
    """Phase goes into locomoco only when asked for (-locomoco_phase, experimental)."""
    opt = plan.options
    return bool(opt.locomoco and opt.locomoco_phase and _phase_on(plan))


def _medic_on(plan: Plan) -> bool:
    return plan.options.medic and _phase_on(plan)


def _medic_stem() -> str:
    """A run's MEDIC stem: ffs_medic writes <stem>.echo-N_undist,
    <stem>.echo-N.part-phase_undist and the shared <stem>_warp."""
    return "stage01.medic.${FRAG[$k]}"


def _fmap_stem(pr: PlanRun, label: str) -> str:
    fid = pr.fmap.fmap_id if pr.fmap else "x"
    return stem(NameKey(label, session=pr.bold.session, fmap=fid))


def _ses_stem(pr: PlanRun, label: str) -> str:
    return stem(NameKey(label, session=pr.bold.session))


def _phase_on(plan: Plan) -> bool:
    return bool(getattr(plan.options, "phase_proc", False))


def _nordic_mag(plan: Plan) -> str:
    """The NORDIC magnitude output (echo 1's, for a multi-echo run)."""
    return "stage00.nordic.${FRAG[$k]}${E1[$k]:-}.nii$FMT"


def _nordic_fmt(plan: Plan) -> str:
    return "$FMT"


def _indent(block: str, n: int) -> str:
    pad = " " * n
    return "\n".join(pad + ln if ln.strip() else ln for ln in block.split("\n"))


def _nordic_phase() -> str:
    """The NORDIC phase output — ffs_nordic writes ``<prefix>.part-phase`` whenever
    phase input was used."""
    return "stage00.nordic.${FRAG[$k]}.part-phase.nii$FMT"


def _phase_file(label: str, part: bool = True) -> str:
    """A per-run phase-side working file, e.g. ``stage00.unwrap.<frag>.part-phase``.
    Built in bash from ``${FRAG[$k]}`` so it matches the Python-side stems; ``part``
    is False for a magnitude-side file."""
    tail = ".part-phase" if part else ""
    return f"stage{STAGE_NUMBERS[label]:02d}.{label}." + "${FRAG[$k]}" + f"{tail}.nii$FMT"


def _unwrapped(label: str = "unwrap") -> str:
    return _phase_file(label)


def _phase_clean(plan: Plan) -> str:
    """The stage00 phase every later stage reads: drift-cleaned unless
    -no_phase_drift, else the unwrap's output."""
    return _unwrapped("unwrapdrift") if plan.options.phase_drift else _unwrapped()


def _phase_source(plan: Plan) -> str:
    """The phase series entering the final resample: the tshifted phase when slice
    timing ran up front (it must get the same temporal shift as its magnitude),
    else the stage00 phase."""
    if plan.options.slicetiming_method == "first":
        return _unwrapped("tshift")
    return _phase_clean(plan)


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


def _mni_files(opt, tail: str) -> list[str]:
    """The anat→MNI link at the very head of the chain (-do_mni).

    Borrowed from the reference results dir in -grand_reference mode, for the same
    reason the anat matrix is: two datasets anchored on one reference must reach
    MNI through the SAME warp, or they are in two slightly different MNI spaces."""
    name = stem(NameKey("mni")) + tail
    gr = getattr(opt, "grand_reference", None) if opt is not None else None
    return [f"{gr.rstrip('/')}/{name}"] if gr else [name]


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


def _token_files(pr: PlanRun, tok: str, fmt: str, opt, echo: int | None = None) -> list[str]:
    """The concrete file(s) one warp-chain token resolves to. Single source of
    truth for both the full chain and the fmap sub-chain (no drift).

    ``echo``: resolve for that echo of a multi-echo 3-D run, whose moco matrices
    carry its own (TE-scaled) partition correction and whose locomoco warp is its
    own; every other link is shared by the echoes."""
    if echo is not None and _me_3d(pr, opt):
        if tok == "moco":
            return [_run_stem(pr, "moco") + f".echo-{echo}.aff12.1D"]
        if tok == "locomoco":
            # One shared warp (ffs_locomoco -me_shared_warp, echo 1's field); echo e's is
            # it with the partition (k) displacement times TE_e/TE_1 -- the TE law is
            # fixed (-me_fixed_scaling), so the factor is known here. ffs_nwarp's VSCALE
            # scales along the warp's voxel axes, exact on an oblique slab.
            warp = _run_stem(pr, "nlmoco") + f"_warp{fmt}"
            tes = pr.bold.echo_tes
            if echo == 1:
                return [warp]
            return [f"VSCALE:1,1,{tes[echo - 1] / tes[0]:.6g}:{warp}"]
    if tok == "mni_nl":
        # .gz, not $FMT: every backend takes the warp's extension from its -prefix,
        # and that prefix is the whole-brain MNI underlay, which has to be openable
        # in stock AFNI (it cannot read .nii.zst).
        return _mni_files(opt, "_nl_WARP.nii.gz")
    if tok == "mni_lin":
        return _mni_files(opt, ".aff12.1D")
    if tok == "anat_lin":
        return _anat_lin_files(opt)
    if tok == "refanat_lin":
        return [_refanat_matrix()]
    if tok == "refanat_nl":
        return [_refanat_warp(opt)]
    if tok == "xfam_lin":
        return [_xfam_matrix()]
    if tok == "xfam_nl":
        return [stem(NameKey("xfam")) + f"_nl_WARP{fmt}"]
    if tok == "xref_nl":
        return [stem(NameKey("xref")) + f"_nl_WARP{fmt}"]
    if tok == "xref_lin":
        return [stem(NameKey("xref")) + ".aff12.1D"]
    if tok == "anat_nl":
        if opt.anat_nonlin_method == "register":
            return [stem(NameKey("nlanat")) + f"_WARP{fmt}"]
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
    if tok == "medic":
        # one 5-D warp: in-plane PE distortion is the same at every echo
        return [_run_stem(pr, "medic") + f"_warp{fmt}"]
    return []


def chain_files(
    pr: PlanRun, fmt: str, opt=None, tokens: list[str] | None = None, echo: int | None = None
) -> list[str]:
    """Resolve a run's warp chain into filenames (nwarp-apply order).

    ``tokens`` overrides which links to resolve — used for the run-average chain
    (``plan.run_average_chain``), which is the run's chain minus the within-run
    motion tokens. The files themselves are shared: an SBRef and its BOLD are
    corrected by exactly the same transforms from the fieldmap level up.
    """
    resolved: list[str] = []
    for tok in pr.warp_chain if tokens is None else tokens:
        resolved.extend(_token_files(pr, tok, fmt, opt, echo=echo))
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
        if opt.anat_nonlin and opt.anat_nonlin_method == "segment":
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
    opt = plan.options
    if opt.ref_anat:
        # Two grids, so two stacks: the EPI's alignment on the ref (what anat_lin
        # and anat_nl were fitted against), and everything on the anat after the
        # ref links -- the ref affine-only, then warped, then the EPI anchor.
        refanat = stem(NameKey("refanat"))
        ref_items: QCItems = [(_refanat_brain(opt), "ref_anat"), (_al_anat(plan), f"{src}_al_ref")]
        items: QCItems = [
            (_anat_box(), "anat"),
            (f"{refanat}_lin.nii.gz", "ref_anat_affine"),
            (_refanat_in_anat(), "ref_anat_nonlin"),
            (_al_anat_nl(plan), f"{src}_al_anat_nl"),
        ]
        calls = [_qc_call(_qc_stem("refanat"), ref_items), _qc_call(_qc_stem("anat"), items)]
    else:
        items = [
            (_anat_box(), "anat"),
            (_al_anat(plan), f"{src}_al_anat"),
        ]
        if opt.anat_nonlin:
            # Both states, so the nonlinear link is visible as the step between them —
            # and so the one the final grid is built from is the one you can see.
            items.append((_al_anat_nl(plan), f"{src}_al_anat_nl"))
        calls = [_qc_call(_qc_stem("anat"), items)]
    if opt.anat_skull:
        # pre-strip vs brain on the input grid: the one stack that shows whether
        # the strip ate cortex. It cannot join the group above — that one lives on
        # the autoboxed grid, and qc_tcat stacks, it does not resample.
        pre = _anat_prestrip(opt)
        strip_items: QCItems = [
            (pre, "anat_avg" if opt.anat_extra else "anat_head"),
            (_anat_brain(), "anat_brain"),
        ]
        if opt.presurf:
            # UNI beside what MPRAGEise made of it. The presurf outputs and the
            # mean (based on the first) all sit on the first UNI's grid.
            strip_items.insert(0, (opt.presurf[0][0], "uni"))
        calls.insert(0, _qc_call(_qc_stem("anat_strip"), strip_items))
    if opt.anat_extra:
        # Every T1w that went into the mean, plus the mean: a rigid alignment that
        # missed shows up here as the one sub-brick that does not overlay.
        ins: QCItems = [(_anat_inputs(opt, opt.anat_path or "$ANAT")[0], "anat_in1")]
        ins += [(_anat_in_al(k), f"anat_in{k}") for k in range(2, len(opt.anat_extra) + 2)]
        ins.append((_anat_avg(), "anat_avg"))
        calls.insert(0, _qc_call(_qc_stem("anat_avg"), ins))
    return _qc_block("EPI → ref anat → anat" if opt.ref_anat else "EPI → anat", calls)


def _final_mean(frag: str) -> str:
    """The per-run temporal mean of the final series, for a ``_frag`` (or a bash
    expression that expands to one).

    ``_mean`` on the end, not ``mean_`` on the front: every other file in this
    directory starts with its stage, and a tool-appended suffix is how stage02's
    ``_mean``/``_max``/``_min`` already read. ffs_nwarp's bare ``-save_mean``
    would put the token in front and drop the file to the top of the listing,
    away from the run it belongs to."""
    return f"stage10.final.{frag}_mean.nii$FINAL_FMT"


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
            [wm] + [(_final_mean(_frag(pr)), _frag(pr)) for pr in plan.runs],
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


def _family_header_note(plan: Plan) -> str:
    """The acquisition families, when there is more than one to tell apart."""
    if len(plan.families) < 2:
        return ""
    lines = [f"\n#   acq families      : {len(plan.families)}"]
    for fam in plan.families:
        how = " (-family, MIXED acquisitions)" if fam.mixed else " (-family)" if fam.forced else ""
        lines.append(
            f"\n#     {fam.label}: {fam.signature.describe()}; {len(fam.members)} run(s){how}"
        )
        if fam.link:
            lines.append(f"\n#       reaches the anat {_family_link_note(fam)}")
    if plan.family_plans:
        lines.append(
            "\n#     Each family runs as its own complete pipeline in fam-<label>/: its own"
            "\n#     grandmean, anat alignment, anat_nl on its own PE axis, output grid at"
            "\n#     its own voxel size, mask and GLM. A family linked to the root then has"
            "\n#     its corrected anchor aligned to the root's (stage09 xfam), so the root's"
            "\n#     anat placement is shared; the root's folder runs first."
        )
    else:
        lines.append(
            "\n#     No anat for them to meet at, so they run as ONE pipeline, linked by"
            "\n#     cross-fieldmap alignment, at the first run's voxel size."
        )
    return "".join(lines)


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
    # Re-quoted, so a value with spaces (-anat_opts "-rigid -cost lpa") pastes
    # back as one word; a quoted value also never starts with "-", so it is
    # never mistaken for the next flag below.
    words = [shlex.quote(w) for w in shlex.split(invocation)]
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


def _cut_header_note(plan: Plan) -> str:
    cuts = plan.options.cut_task_vols
    if not cuts:
        return ""
    body = ", ".join(f"task-{t} {n}" for t, n in sorted(cuts.items()))
    noise = " — no noise vols (NOISE[...]=0)" if plan.options.noise_vols else ""
    return f"\n#   cut to N vols     : {body} (-cut_task_vols; MAG[...] selectors below){noise}"


def _header(plan: Plan, out_dir: str, invocation: str | None = None) -> str:
    opt = plan.options
    phase_skip = "\nskip_unwrap=1" if _phase_on(plan) else ""
    if _medic_on(plan):
        phase_skip += " skip_medic=1"
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
#   NORDIC : {opt.want_nordic}   locomoco : {opt.locomoco}   distortion : {
        opt.distortion
    }   slice-timing : {opt.slicetiming_method}
#   phase  : {_phase_on(plan)}   MEDIC : {_medic_on(plan)}{_timing_header_note(plan)}{
        _cut_header_note(plan)
    }
#   sbref lane        : {plan.use_sbref}{_sbref_header_note(plan)}
#   image lanes       : {len(_lanes(plan))}{_lane_header_note(plan)}
#   ref image         : {
        opt.ref_image or "grandmean"
    } (session representative for xses; -ref_image){
        "" if plan.family_plans else _fmap_inherit_note(plan)
    }{_family_header_note(plan)}{_qc_header_note(plan)}
# ============================================================================={
        _invocation_note(invocation)
    }
# ---- Bash version -----------------------------------------------------------
# macOS ships Bash 3.2, so hand over to a newer one if it is installed anywhere usual.
if [ -z "${{BASH_VERSION:-}}" ] || [ "${{BASH_VERSINFO[0]}}" -lt 4 ]; then
    if [ -f "$0" ]; then
        for _bash in "$(command -v bash)" /opt/homebrew/bin/bash /usr/local/bin/bash \
                /opt/local/bin/bash "${{CONDA_PREFIX:-}}/bin/bash"; do
            if [ -x "$_bash" ] && "$_bash" -c '[ "${{BASH_VERSINFO[0]}}" -ge 4 ]' 2>/dev/null; then
                echo "ffs_autoproc: continuing under $_bash" >&2
                exec "$_bash" "$0" "$@"
            fi
        done
    fi
    echo "ERROR: ffs_autoproc pipelines require Bash >= 4 (associative arrays)." >&2
    echo "On macOS: brew install bash, then run this script again." >&2
    exit 1
fi
set -euo pipefail

OUT={shlex.quote(out_dir)}
DEVICE={shlex.quote(opt.device)}          # compute device for every ffs stage; -device
FMT={opt.fmt}           # working intermediates (read many times); -format
FINAL_FMT={opt.final_fmt}     # final timeseries (portable); -final_format
GLM_FMT={opt.glm_fmt}       # GLM stat buckets; -glm_format

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
        "declare -A MAG PHASE NOISE SBREF TR PEDIR TRT JSON FRAG CHAIN AVGCHAIN XRUNBASE EVENTS "
        "EVOPTS "
        f"{moco_arrays} PRECHAIN REFGRID JAC"
        # Keys are "ses:task:run": undeclared, bash takes SURFCHAIN as an indexed
        # array and evaluates the key as arithmetic, failing every assignment.
        + (" SURFCHAIN" if plan.options.surface_target else "")
        # Declared even for single-echo data: the echo loops read ${NECHO[$k]:-1},
        # and an undeclared array would take the "ses:task:run" key as arithmetic.
        + " NECHO TES MAGE PHASEE ECHAIN E1"
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
        # -cut_task_vols rides on the source as a sub-brick selector, so every
        # reader (NORDIC, slice timing, moco, the final resample) sees one cut.
        cut = task_cut(pr, plan.options)
        sel = ""
        if cut:
            lines.append(f"# -cut_task_vols: {cut[0]} -> {cut[1]} volumes")
            sel = f"[0..{cut[1] - 1}]"
        lines.append(f"MAG[{q(k)}]={q(str(b.mag_path) + sel)}")
        lines.append(f"NOISE[{q(k)}]={run_noise_vols(pr, plan.options)}")
        if b.phase_path:
            lines.append(f"PHASE[{q(k)}]={q(str(b.phase_path) + sel)}")
        # -medic: the SBRef has no per-frame field, so it cannot be the moco base of
        # undistorted echoes; leaving it out falls the base back to a volume.
        if b.sbref_path and not plan.options.medic:
            lines.append(f"SBREF[{q(k)}]={q(str(b.sbref_path))}")
        if b.is_multiecho:
            lines.append(f"NECHO[{q(k)}]={b.n_echoes}")
            lines.append(f"E1[{q(k)}]=.echo-1")
            lines.append(f"TES[{q(k)}]={q(' '.join(f'{t:g}' for t in b.echo_tes))}")
            for e, (em, ep) in enumerate(zip(b.echo_mags, b.echo_phases, strict=True), start=1):
                lines.append(f"MAGE[{q(f'{k}|{e}')}]={q(str(em) + sel)}")
                if ep is not None:
                    lines.append(f"PHASEE[{q(f'{k}|{e}')}]={q(str(ep) + sel)}")
        lines.append(f"JSON[{q(k)}]={q(str(_sidecar(b.mag_path)))}")
        tr = plan.options.tr if plan.options.tr is not None else b.tr
        lines.append(f"TR[{q(k)}]={q(str(tr if tr is not None else ''))}")
        lines.append(f"PEDIR[{q(k)}]={q(str(b.pe_dir or ''))}")
        if plan.options.medic and b.json.get("TotalReadoutTime") is not None:
            lines.append(f"TRT[{q(k)}]={q(str(b.json['TotalReadoutTime']))}")
        # Filename coordinate fragment — the stage loops build run filenames as
        # stageNN.label.${FRAG[$k]}, matching every reference in the data table.
        lines.append(f"FRAG[{q(k)}]={q(_frag(pr))}")
        # The per-run task diagnostics read this -- NORDIC's task-leak report at
        # stage00, locomoco's task-coupling report at stage03. Both are per run, and a
        # run whose events did not resolve simply goes unscored rather than failing.
        if _events_consumers(plan.options):
            ev = _events_script_path(plan, pr, bids_root)
            if ev:
                lines.append(f"EVENTS[{q(k)}]={q(ev)}")
                shape = _event_shape_args(plan, b.task, bids_root)
                if shape:
                    lines.append(f"EVOPTS[{q(k)}]={q(shape)}")
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
        # The chain for images whose space is the whole run, not one volume of it
        # (the SBRef, the NORDIC dof map): the run's chain minus the within-run
        # motion tokens. Always emitted — the dof map needs it with or without an
        # SBRef lane.
        avg = chain_files(pr, ".nii$FMT", plan.options, tokens=run_average_chain(pr))
        lines.append(f'AVGCHAIN[{q(k)}]="{" ".join(avg)}"')
        chain = chain_files(pr, ".nii$FMT", plan.options)
        lines.append(f'CHAIN[{q(k)}]="{" ".join(chain)}"')
        if b.is_multiecho:
            # Each echo's chain: echo 1's for 2-D (one warp moves every echo alike);
            # for 3-D each echo's own moco matrices and locomoco warp (TE-scaled).
            for e in range(1, b.n_echoes + 1):
                ech = chain_files(pr, ".nii$FMT", plan.options, echo=e)
                lines.append(f'ECHAIN[{q(f"{k}|{e}")}]="{" ".join(ech)}"')
        if plan.options.surface_target:
            # The surfaces live in the anat's scanner space: the chain stops there.
            surf = chain_files(
                pr,
                ".nii$FMT",
                plan.options,
                tokens=[t for t in pr.warp_chain if t not in _MNI_TOKENS],
            )
            lines.append(f'SURFCHAIN[{q(k)}]="{" ".join(surf)}"')
            if b.is_multiecho:
                for e in range(1, b.n_echoes + 1):
                    surf = chain_files(
                        pr,
                        ".nii$FMT",
                        plan.options,
                        tokens=[t for t in pr.warp_chain if t not in _MNI_TOKENS],
                        echo=e,
                    )
                    lines.append(f'SURFCHAIN[{q(f"{k}|{e}")}]="{" ".join(surf)}"')
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
        spec_path(task, plan.options)
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
            if _events_consumers(plan.options)
            for ev in [_events_script_path(plan, pr, bids_root)]
            if ev
        }
        # The TPM build reads these before stage09 can produce a nonlinear anat.
        | {p for p in (plan.options.spm_tpm, plan.options.anat_t2) if p and plan.options.anat_tpm}
        | ({plan.options.ref_anat} if plan.options.ref_anat else set())
        | {p for p in (plan.options.ref_presurf or ())[1:] if p}
    )
    checks = " \\\n".join(f"  {shlex.quote(p)}" for p in inputs)
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
        tools = [*tools, "ffs_phase_unwrap"]
        if opt.phase_drift:
            tools = [*tools, "ffs_phasedrift"]
    if (opt.presurf or opt.ref_presurf) and _own_anat(opt):
        tools = [*tools, "ffs_util_presurf"]
    if opt.anat_skull and _own_anat(opt):
        tools = [*tools, "mri_synthstrip"]
    if opt.do_mni and _own_anat(opt):
        tools = [*tools, config.nl_command(opt.mni_backend).split()[0]]
    if opt.anat_nonlin and opt.anat_nonlin_method == "register":
        tools = [*dict.fromkeys([*tools, _nl_tool(opt, "anat_nl").split()[0]])]
    if opt.ref_anat and opt.ref_anat_nonlin_method == "register":
        tools = [*dict.fromkeys([*tools, _nl_tool(opt, "ref_anat_nl").split()[0]])]
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
        f'{indent}if [ "${{NOISE[$k]}}" -gt 0 ]; then '
        f'nv=$(ffs_info -nv "$raw"); raw="${{raw}}[0..$((nv - ${{NOISE[$k]}} - 1))]"; fi'
    )


def _raw_source(plan: Plan, indent: str = "  ") -> str:
    """Bash setting ``raw`` to the series that moco and the final resample act on.

    Precedence: NORDIC output → up-front tshift output (``-slicetiming_method
    first``) → BIDS magnitude (noise-vols trimmed). Nothing is copied wholesale.
    When STC is done first, the final ffs_nwarp reads the tshifted series and does
    NOT re-integrate timing (that only happens for ``-slicetiming_method
    integrate``)."""
    if plan.options.slicetiming_method == "first":
        return f'{indent}raw="stage01.tshift.${{FRAG[$k]}}${{E1[$k]:-}}.nii$FMT"'
    return _pre_stc_source(plan, indent)


def _tshift_flags(plan: Plan) -> list[str]:
    """-tshift_opts with the plan's tzero in force (-tzero wins over the opts)."""
    rest = [f for f in _split_flags(config.DEFAULT_OPTS["tshift"]) if f.split()[0] != "-tzero"]
    return [f"-tzero {plan.options.tzero:g}", *rest]


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
            *_tshift_flags(plan),
            '-device "$DEVICE"',
        ],
    )
    # Phase must get the SAME temporal shift as its magnitude, or the two stop
    # describing the same instant. Unwrapped phase is smooth in time, so the same
    # temporal interpolation is valid on it (a wrapped phase would not be).
    phase_cmd = pc = ""
    if _phase_on(plan):
        pc = _ffs(
            "ffs_slicetime",
            [
                f'-input "{_phase_clean(plan)}"',
                '-prefix "$phout"',
                f"-tpattern {_tpattern(plan)}",
                '-TR "${TR[$k]}"',
                *_tshift_flags(plan),
                '-device "$DEVICE"',
            ],
        )
        phase_cmd = (
            f'  phout="{_unwrapped("tshift")}"\n'
            f'  [ -z "${{PHASE[$k]:-}}" ] || [ -f "$phout" ] || {{\n{pc}\n  }}\n'
        )
    me_block = ""
    if _any_me(plan):
        # Every echo, and each echo's unwrapped phase, gets the same shift: echo e
        # was read out with the same slice order, a few ms after echo 1.
        me_cmd = cmd.replace('-input "$raw"', '-input "$mag"').replace(
            '-prefix "$outf"', '-prefix "$eoutf"'
        )
        me_phase = ""
        if _phase_on(plan):
            me_phase = (
                '\n      phout="' + _echo_stc_phase() + '"\n'
                '      [ -z "${PHASE[$k]:-}" ] || [ -f "$phout" ] || '
                + pc.replace(
                    f'"{_phase_clean(plan)}"',
                    '"' + _me_unwrap_stem() + '.echo-${e}.part-phase.nii$FMT"',
                ).strip()
            )
        me_block = (
            '  if [ "${NECHO[$k]:-1}" -gt 1 ]; then\n'
            '    for e in $(seq 1 "${NECHO[$k]}"); do\n'
            '      esfx=".echo-$e"\n'
            f'      eoutf="{_echo_stc_mag()}"'
            + me_phase
            + '\n      [ -f "$eoutf" ] && continue\n'
            + _echo_pair_source(plan)
            + "\n"
            + _indent(me_cmd, 4)
            + "\n    done\n    continue\n  fi\n"
        )
    return f"""
# ============================ stage01: slice timing (first) =================
# STC applied up front (before moco); the final resample does NOT re-integrate
# timing. Reads the NORDIC/BIDS source in place; writes the tshifted series.
echo '== stage01: slice timing =='
for k in "${{RUN_KEYS[@]}}"; do
  outf="stage01.tshift.${{FRAG[$k]}}.nii$FMT"
  [ -z "${{TR[$k]}}" ] && {{ echo "no TR for $k; cannot slice-time"; exit 1; }}
{me_block}{phase_cmd}  [ -f "$outf" ] && continue
{_pre_stc_source(plan)}
{cmd}
done
"""


def _nordic_events_on(opt) -> bool:
    """Does stage00 hand ffs_nordic the run's events? Only for a step that reads them:
    the task-leak QC, or -task_rescue, which acts on that QC's component scan."""
    return bool(opt.want_nordic and (opt.nordic_task_qc or opt.nordic_task_rescue))


# ffs_locomoco flags, possibly arriving through -locomoco_opts, that need the design.
_LOCOMOCO_EVENT_FLAGS = re.compile(r"(?:^|\s)-(?:(?:detask|reject|task_qc)\b|ref\s+paired)")


def _locomoco_events_on(opt) -> bool:
    """Does stage03 hand ffs_locomoco the run's events? Only for a step that reads
    them: the task-coupling QC, -detask, or a -reject / -ref paired in -locomoco_opts."""
    if not opt.locomoco:
        return False
    return bool(
        opt.locomoco_task_qc
        or opt.locomoco_detask
        or _LOCOMOCO_EVENT_FLAGS.search(config.DEFAULT_OPTS["locomoco"])
    )


def _events_consumers(opt) -> bool:
    """Does any stage in this pipeline read a per-run events TSV?

    Only the stages that ASK for it: the EVENTS array and the preflight path check
    follow the same rules the stages do, so a pipeline with no task step leaves no
    dangling reference to a file nothing opens.
    """
    return _locomoco_events_on(opt) or _nordic_events_on(opt)


def _dof_adjust_on(opt) -> bool:
    """Is the NORDIC dof correction wired end to end for this pipeline?"""
    return bool(opt.want_nordic and opt.nordic_dof_adjust)


def _numcomps_native(plan: Plan) -> str:
    """This run's NORDIC component-count map, on its own acquisition grid.

    ffs_nordic appends ``_numcomps`` to the -prefix stem, so it inherits whatever
    format the magnitude output was written in. A multi-echo run denoises each
    echo on its own; echo 1's map stands for the run.
    """
    return "stage00.nordic.${FRAG[$k]}${E1[$k]:-}_numcomps.nii$FMT"


def _numcomps_final(frag: str) -> str:
    """The same map after this run's OWN chain has put it in the final space."""
    return f"stage10.dofloss.{frag}.nii$FMT"


def _task_dofloss(task: str) -> str:
    return f"stage11.dofloss.task-{task}.nii$FMT"


def _stage_nordic(plan: Plan) -> str:
    opt = plan.options
    if not opt.want_nordic:
        # No stage00: nothing is copied. moco/final read the BIDS source directly.
        return ""
    resid = "-save-residual-map" if getattr(opt, "nordic_save_resid", False) else ""
    # ffs_nordic writes a denoised phase alongside the magnitude whenever phase
    # input was used, and it trims the noise volumes off both — so the -phase_proc
    # path needs nothing extra here.
    phase_note = (
        "\n# -phase_proc: the denoised phase lands beside the magnitude as\n"
        "# stage00.nordic.<frag>.part-phase (noise volumes already trimmed);\n"
        "# stage00 unwrap picks it up.\n"
        if _phase_on(plan)
        else ""
    )
    rescue = opt.nordic_task_rescue
    # The per-voxel count of components removed. stage10 carries it through this run's
    # own warp chain and stage12 charges it against the GLM's degrees of freedom.
    numcomps = "-save_num_comps" if _dof_adjust_on(opt) else ""
    nordic_cmd = _ffs(
        "ffs_nordic",
        [
            '-input_magn "$mag"',
            '"${phase_arg[@]}"',
            '-prefix "$outf"',
            '-noise-volume-last "${NOISE[$k]}"',
            *_split_flags(config.DEFAULT_OPTS["nordic"]),
            resid,
            numcomps,
            # The task QC is a diagnostic: NORDIC decides what to discard from a
            # patch's spectrum alone, and nothing in that decision knows the task, so a
            # component carrying real response can fall under the threshold. -tr rides
            # along for the same reason it does at stage03 -- a 3D acquisition's
            # pixdim[4] is the per-partition time, not the volume TR the design is
            # sampled at.
            *(
                ['${ev:+-events "${ev}"}', '"${evopts[@]}"', '${ev:+${TR[$k]:+-tr "${TR[$k]}"}}']
                if _nordic_events_on(opt)
                else []
            ),
            *(["${ev:+-task_qc}"] if opt.nordic_task_qc else []),
            *([f"${{ev:+-task_rescue {rescue}}}"] if rescue else []),
            '-device "$DEVICE"',
        ],
    )
    return f"""
# ============================ stage00: NORDIC ==============================={phase_note}
# -nordic_task_qc (per run, against the same events TSV the GLM models) writes the
# task-leak report: <prefix>_taskleak.txt, the _taskfit_input/_kept/_lost buckets,
# the per-patch _taskpatch map, and a scan of the removed field's components. It
# MEASURES how much of the design went out with the noise and whether it landed on
# responding tissue; nothing about the denoising changes. This runs before motion
# correction, so read the enrichment and the component scan, not the per-voxel R
# maps. -nordic_task_rescue is what ACTS on it.
echo '== stage00: NORDIC denoise =='
# Multi-echo: each echo denoised on its own for now (written .echo-N, its phase
# .echo-N.part-phase); a joint cross-echo denoising is still to be tested.
for k in "${{RUN_KEYS[@]}}"; do
  ev="${{EVENTS[$k]:-}}"   # unset for a run whose events did not resolve; set -u
  # EVOPTS is shell-quoted text (-event_cols / -event_filter_*); eval keeps its quoting.
  evopts=(); [ -n "$ev" ] && eval "evopts=(${{EVOPTS[$k]:-}})"
  {_ECHO_LOOP}
    outf="stage00.nordic.${{FRAG[$k]}}${{esfx}}.nii{_nordic_fmt(plan)}"
    mag="{_ECHO_MAG}"; ph="{_ECHO_PHASE}"
    # with phase, done means its .part-phase too (a dir from before that name has _phase)
    phout="stage00.nordic.${{FRAG[$k]}}${{esfx}}.part-phase.nii{_nordic_fmt(plan)}"
    [ "$skip_nordic" -eq 1 ] && [ -f "$outf" ] && {{ [ -z "$ph" ] || [ -f "$phout" ]; }} && continue
    if [ -n "$ph" ]; then phase_arg=(-input_phase "$ph"); else phase_arg=(-magnitude-only); fi
{_indent(nordic_cmd, 2)}
  done
done
"""


def _moco_reduction_flags() -> str:
    """``-save_mean/-save_max/-save_min`` for the stage02 manifest line (escaped for
    the printf that writes it). One resample, three images: the mean the pipeline
    always used, plus the coverage pair the max/min lanes are built from."""
    return " ".join(
        f'-save_{which} \\"${{mstem}}_{which}.nii$FMT\\"' for which in ("mean", "max", "min")
    )


def _censoring_on(opt) -> bool:
    return opt.censor_motion is not None or opt.censor_outliers is not None


def _moco_censor_flags(plan: Plan) -> str:
    """Per-run censoring outputs for the stage02 manifest line (escaped for its printf).

    Computed inside moco because the motion parameters and the raw series are
    both already in memory there; the traces ride along as QC whenever any
    censoring is on. stage12 concatenates the per-run keep masks per task.
    """
    opt = plan.options
    if not _censoring_on(opt):
        return ""
    parts = ['-enorm \\"${mstem}.enorm.1D\\"']
    if opt.censor_motion is not None:
        parts.append(f"-censor_motion {opt.censor_motion:g}")
    if opt.censor_outliers is not None:
        parts += [
            '-outcount \\"${mstem}.outcount.1D\\"',
            f"-censor_outliers {opt.censor_outliers:g}",
        ]
    parts.append('-censor \\"${mstem}.censor.1D\\"')
    return " " + " ".join(parts)


def _stage_moco(plan: Plan, script_stem: str) -> str:
    moco_flags = " ".join(_split_flags(config.DEFAULT_OPTS["moco"]))
    batchfile = f"{script_stem}_mocobatch.txt"
    # locomoco estimates residual motion per volume: it needs the rigid-corrected
    # 4D, not the mean. Only then is it worth writing this intermediate.
    # $mpfx: the run's stem, or echo 1's for a multi-echo run registered on echo 1
    # alone (2-D); a 3-D multi-echo run hands every echo in and resets it to the
    # run's stem, from which ffs_moco names each echo's series.
    ts_arg = ' -prefix \\"${mpfx}.nii$FMT\\"' if plan.options.locomoco else ""
    # MotSim rides along with the correction that produced the motion: the base
    # volume and the matrices are already in memory, and the backward variant's
    # re-registration inherits this run's settings, so it models the residual of
    # THIS correction. See config.GLM_ORTVEC["motsim"].
    motsim_arg = (
        f' -motsim {plan.options.motsim} -motsim_1D \\"${{mstem}}.motsim.1D\\"'
        if plan.options.motsim
        else ""
    )
    censor_arg = _moco_censor_flags(plan)
    me3d_block, me3d_links = _moco_me3d(plan)
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
{_medic_moco_source(plan)}  ins="\\"$raw\\""; me_args=""; mpfx="${{mstem}}${{E1[$k]:-}}"
{me3d_block}  printf '%s\\n' "-input $ins $base_str {moco_flags}{ts_arg} {_moco_reduction_flags()} -1Dmatrix_save \\"${{mstem}}.aff12.1D\\" -1Dfile \\"${{mstem}}.motion.1D\\"{motsim_arg}{censor_arg}$me_args" >> "$mocobatch"
done
batch_skip=(); [ "$skip_moco" -eq 1 ] && batch_skip=(-batch_skip)
ffs_moco -batch "$mocobatch" "${{batch_skip[@]}}" -device "$DEVICE"
{me3d_links}"""


def _medic_moco_source(plan: Plan) -> str:
    """-medic: moco registers the undistorted echo 1, not the raw series (the chain's
    rightmost medic warp puts the raw frame there at stage10)."""
    if not _medic_on(plan):
        return ""
    return (
        '  if [ -n "${PHASE[$k]:-}" ] && [ "${NECHO[$k]:-1}" -gt 1 ]; then '
        f'raw="{_medic_stem()}.echo-1_undist.nii$FMT"; fi\n'
    )


def _moco_echo_source(plan: Plan) -> str:
    """Bash setting ``mag`` to echo $e's moco input (in the manifest's echo loop)."""
    if _medic_on(plan):
        return f'      mag="{_medic_stem()}.echo-${{e}}_undist.nii$FMT"'
    return _echo_source(plan)


def _moco_phase_file(plan: Plan) -> str:
    """Echo $e's unwrapped phase for moco to carry: undistorted with the magnitude
    under -medic (a phase that moved differently from its magnitude is no input)."""
    if _medic_on(plan):
        return f"{_medic_stem()}.echo-${{e}}.part-phase_undist.nii$FMT"
    return _echo_phase_file(plan)


def _stage_medic(plan: Plan) -> str:
    """Frame-wise distortion correction from the unwrap stage's per-frame field map.

    ffs_medic -fieldmap: no second unwrap, the same receiver-drift-free field the
    phase carries, SVD-denoised as MEDIC does, then the PE pull warp. Writes the
    undistorted echoes (moco's input), the 5-D warp (the chain's rightmost link), and
    -- when 3-D multi-echo locomoco reads the phase -- the unwrapped phase undistorted
    with its magnitude.
    """
    if not _medic_on(plan):
        return ""
    opt = plan.options
    phase = ""
    if _lm_phase(plan) and opt.acq_3d:
        phase = (
            '\n  phs=(); for e in $(seq 1 "${NECHO[$k]}"); do '
            'esfx=".echo-$e"; '
            f'phs+=("{_echo_phase_file(plan)}"); done'
        )
        apply = ' -apply_phase -phase_is_unwrapped -phase "${phs[@]}"'
        # a work dir that gains -locomoco_phase has the warp but not the phase
        phase_done = ' && [ -f "${ms}.echo-${NECHO[$k]}.part-phase_undist.nii$FMT" ]'
    else:
        apply = ""
        phase_done = ""
    return f"""
# ============================ stage01: MEDIC (frame-wise distortion) ========
# -medic: each frame's field map (stage00's multi-echo unwrap) -> SVD denoising ->
# a pull warp along the PE axis. Moco registers the undistorted echoes; stage10
# composes the warp (stage01.medic.*_warp, one 5-D file shared by the echoes) at
# the right of every chain and still resamples once from the raw series. No
# fieldmap is used, and the SBRef is not the moco base.
echo '== stage01: MEDIC =='
for k in "${{RUN_KEYS[@]}}"; do
  [ -n "${{PHASE[$k]:-}}" ] && [ "${{NECHO[$k]:-1}}" -gt 1 ] || continue
  ms="{_medic_stem()}"; uwe="{_me_unwrap_stem()}"
  [ "$skip_medic" -eq 1 ] && [ -f "${{ms}}_warp.nii$FMT" ]{phase_done} && continue
  mags=()
  for e in $(seq 1 "${{NECHO[$k]}}"); do
    esfx=".echo-$e"
{_echo_source(plan, indent="    ")}
    mags+=("$mag")
  done{phase}
  ffs_medic -magnitude "${{mags[@]}}" -fieldmap "${{uwe}}_fieldmap.nii$FMT" -fieldmap_mask "${{uwe}}_mask.nii$FMT" -total_readout_time "${{TRT[$k]}}" -pe_dir "${{PEDIR[$k]}}" -save_warp -warp_5d -prefix "${{ms}}.nii$FMT" -device "$DEVICE" -verb 0{apply}
done
"""


def _moco_me3d(plan: Plan) -> tuple[str, str]:
    """stage02's multi-echo 3-D additions: (bash in the manifest loop, bash after the batch).

    Motion is estimated from echo 1, but every echo goes in: -me_3depi removes the
    TE-scaled partition shift (and stretch) from echo 1 BEFORE the rigid estimate, so
    the head's motion is not confused with it, and writes each echo's matrices with
    its own correction folded in (<mstem>.echo-N.aff12.1D, read by ECHAIN). The
    correction comes from stage00's phase echo-slope when the phase is processed,
    else from moco's own inter-echo cross-correlation of the magnitude. With locomoco,
    every echo's corrected series is written for its solve (and the phase, under
    -locomoco_phase).
    """
    opt = plan.options
    if not (opt.acq_3d and _any_me(plan)):
        return "", ""
    tables = (
        '      [ -n "${PHASE[$k]:-}" ] && [ -f "' + _pshift() + '" ] && '
        'me_args="$me_args -shift_table \\"'
        + _pshift()
        + '\\" -stretch_table \\"'
        + _pstretch()
        + '\\""\n'
        if _phase_on(plan)
        else ""
    )
    phase = ""
    if _lm_phase(plan):
        phase = (
            '      if [ -n "${PHASE[$k]:-}" ]; then\n'
            # esfx set per echo here too: the name can carry it (slice timing up front),
            # and this loop is not the magnitude loop that set it before.
            '        phs=""; for e in $(seq 1 "${NECHO[$k]}"); do esfx=".echo-$e"; '
            'phs="$phs \\"' + _moco_phase_file(plan) + '\\""; done\n'
            '        me_args="$me_args -phase$phs -phase_units rad"\n'
            "      fi\n"
        )
    block = (
        '  if [ "${NECHO[$k]:-1}" -gt 1 ]; then\n'
        '    ins=""\n'
        '    for e in $(seq 1 "${NECHO[$k]}"); do\n'
        '      esfx=".echo-$e"\n' + _moco_echo_source(plan) + '\n      ins="$ins \\"$mag\\""\n'
        "    done\n"
        '    me_args=" -reg_echo 1 -me_3depi -axis k -echo_times ${TES[$k]}'
        ' -1Dmatrix_shiftcorr \\"${mstem}.aff12.1D\\""\n'
        '    mpfx="$mstem"\n'
        + tables.replace("      ", "    ", 1)
        + phase.replace("      ", "    ")
        + "  fi\n"
    )
    links = (
        "# Multi-echo 3-D: echo 1 is the registration echo -- its .echo-1\n"
        "# reductions take the run's names every later stage reads.\n"
        'for k in "${RUN_KEYS[@]}"; do\n'
        '  [ "${NECHO[$k]:-1}" -gt 1 ] || continue\n'
        '  mstem="stage02.moco.${FRAG[$k]}"\n'
        "  for which in mean max min; do\n"
        '    ln -sf "${mstem}.echo-1_${which}.nii$FMT" "${mstem}_${which}.nii$FMT"\n'
        "  done\n"
        "done\n"
    )
    return block, links


def _stage_locomoco(plan: Plan, script_stem: str) -> str:
    if not plan.options.locomoco:
        return ""
    detask = plan.options.locomoco_detask
    parts = [
        # $lm_in / $lm_pe / $lm_me are set per run in the loop below (multi-echo 3-D
        # changes all three).
        "-input $lm_in",
        '-prefix "${nlstem}.nii$FMT"',
        "$lm_pe",
        "$lm_me",
        *_split_flags(config.DEFAULT_OPTS["locomoco"]),
        # The design, for whichever step reads it. -tr rides along because a 3D
        # acquisition's header pixdim[4] is the per-partition time, not the volume TR
        # the design is sampled at.
        # EVOPTS is shell-quoted text; the manifest is shlex-split, which unquotes it.
        *(
            [
                '${ev:+-events "${ev}"}',
                "${ev:+${EVOPTS[$k]:-}}",
                '${ev:+${TR[$k]:+-tr "${TR[$k]}"}}',
            ]
            if _locomoco_events_on(plan.options)
            else []
        ),
        # Diagnostic only: how much of the estimated field the task explains, which
        # is exactly the failure mode a brightness-constancy data term has on a block
        # design.
        *(["${ev:+-task_qc}"] if plan.options.locomoco_task_qc else []),
        # Behind ${ev:+}: -detask without a design is refused, and one run without
        # events would take the whole batch down with it.
        *([f"${{ev:+-detask {detask}}}"] if detask else []),
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
# -locomoco_task_qc (per run, against the same events TSV the GLM models) writes
# the task-coupling report: _locomoco_taskcoupling.txt, the taskr/taskrms maps,
# and a scan of the field's independent components (_locomoco_taskcomps.1D). All
# of it is a DIAGNOSTIC -- nothing about the correction changes -- and it is what
# to read before reaching for -locomoco_detask.
echo '== stage03: locomoco =='
lmbatch="{script_stem}_locomocobatch.txt"; : > "$lmbatch"
for k in "${{RUN_KEYS[@]}}"; do
  nlstem="stage03.nlmoco.${{FRAG[$k]}}"
  pe="${{PEDIR[$k]}}"; pe="${{pe//[!a-zA-Z]/}}"
  ev="${{EVENTS[$k]:-}}"   # unset for a run whose events did not resolve; set -u
{_locomoco_run_vars(plan)}
{_manifest_line("lmbatch", parts)}
done
{_batch_launch("ffs_locomoco", "lmbatch", "skip_locomoco")}
"""


def _locomoco_run_vars(plan: Plan) -> str:
    """Per-run locomoco inputs, axes and multi-echo flags (bash in the stage03 loop).

    Single echo: stage02's corrected series; the in-plane PE axis, plus the partition
    axis for 3-D EPI (a field change during the shot moves it too) in one solve.
    Multi-echo 3-D: every echo's corrected series, both axes -- the partition one
    TE-scaled (echo e moves TE_e/TE1 times as far), the in-plane one the same at every
    echo -- and, only under -locomoco_phase (experimental), the unwrapped phase, making
    the partition estimate the joint, field-blind magnitude+phase one; each echo gets
    its own warp (stage10's ECHAIN). Nothing else corrects in-plane distortion frame by frame here. Multi-echo 2-D: echo 1 alone -- the in-plane warp moves every
    echo alike, and stage10 applies it to each.
    """
    pe = '-pe_dir \\"${pe:-y}\\"' + (" -pe_dir2 k" if plan.options.acq_3d else "")
    out = [
        '  lm_in="\\"stage02.moco.${FRAG[$k]}${E1[$k]:-}.nii$FMT\\""',
        f'  lm_pe="{pe}"; lm_me=""',
    ]
    if plan.options.acq_3d and _any_me(plan):
        phase = ""
        if _lm_phase(plan):
            phase = (
                '    if [ -n "${PHASE[$k]:-}" ]; then\n'
                '      for e in $(seq 1 "${NECHO[$k]}"); do '
                'lm_ph="${lm_ph:-} \\"stage02.moco.${FRAG[$k]}.echo-${e}.part-phase.nii$FMT\\""; done\n'
                '      lm_me="$lm_me -phase$lm_ph -phase_units rad"; unset lm_ph\n'
                "    fi\n"
            )
        out.append(
            '  if [ "${NECHO[$k]:-1}" -gt 1 ]; then\n'
            '    lm_in=""; for e in $(seq 1 "${NECHO[$k]}"); do '
            'lm_in="$lm_in \\"stage02.moco.${FRAG[$k]}.echo-${e}.nii$FMT\\""; done\n'
            # The TE law fixed, not learned: learned on A01D (magnitude only) it came out
            # non-physical (1, 1.24, 0.78, 0.42, 0.36 against TE ratios to 7.9; linear-in-TE
            # r2 -2.7), and the learning pass holds a field per axis per echo.
            # -me_shared_warp: one warp for the run; each echo's chain scales it (VSCALE).
            '    lm_me="-me_3depi -echo_times ${TES[$k]} -me_fixed_scaling -me_shared_warp"\n'
            + phase
            # MEDIC already corrected each frame in-plane: only the TE-scaled partition
            # shift is left for the joint solve.
            + (
                '    lm_pe="-pe_dir2 k"\n'
                if _medic_on(plan) and medic_owns_inplane(plan.options)
                else ""
            )
            + "  fi"
        )
    return "\n".join(out)


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
                # The pair as the estimator saw it (collapsed, on blip_up's grid): the
                # QC answer to "what went into the fit", see _qc_distortion.
                "-save_pad_phase",
                f'-prefix "{st}.nii$FMT"',
            ]
        )
    if blip_jobs:
        out.append(f'bfbatch="{script_stem}_blipbatch.txt"; : > "$bfbatch"')
        out += [_manifest_line("bfbatch", parts, indent="") for parts in blip_jobs]
        out.append(_batch_launch("ffs_blipflip", "bfbatch", "skip_blip"))
    if _qc_on(plan):
        out.append(_qc_distortion(list(groups.values())))
    return "\n".join(out) + "\n"


def _qc_distortion(prs: list[PlanRun]) -> str:
    """QC stacks for each fieldmap group. A reverse-PE pair gets the pair blipflip
    fitted (``_inputs``); its corrected counterpart is blipflip's own
    ``_unwarped``. A GRE fieldmap gets a before/after stack of the EPI it
    corrected, on the same grid: a field with the wrong sign or scale shows as
    the brain stretching instead of settling.

    The "before" image is the anchor through an identity matrix with -save_mean,
    the same ffs_nwarp read that makes the corrected one, so the pair differs only
    by the warp. A 3-D anchor is its own before image."""
    lines: list[str] = []
    calls = []
    for pr in prs:
        assert pr.fmap is not None
        st = _fmap_stem(pr, "blip")
        qc = _qc_stem("blip", session=pr.bold.session, fmap=pr.fmap.fmap_id)
        if not pr.fmap.is_b0:
            # The pair as the estimator saw it (collapsed, on blip_up's grid). Its
            # blip_up volume is the uncorrected anchor, and {st}_unwarped already
            # holds both directions corrected, so a before/after stack would
            # only duplicate them.
            pair = f"{st}_padphase.nii$FMT"
            calls.append(
                _qc_call(qc + "_inputs", [(f"{pair}[0]", "blip_up"), (f"{pair}[1]", "blip_down")])
            )
            continue
        src = str(pr.fmap_forward or pr.bold.rep)  # what stage04 corrected, either lane
        before = f"{st}_before.nii$FMT"
        if _n_volumes(src) > 1:
            tmp = f"{st}_before_tmp.nii$FMT"
            if not lines:
                lines += [
                    'ident="stage04.identity.aff12.1D"',
                    '[ -f "$ident" ] || printf \'1 0 0 0 0 1 0 0 0 0 1 0\\n\' > "$ident"',
                ]
            lines.append(
                f'[ -f "{before}" ] || {{ ffs_nwarp -source {shlex.quote(src)} '
                f'-master {shlex.quote(src)} -nwarp "$ident" -save_mean "{before}" '
                f'-prefix "{tmp}" -device "$DEVICE" && rm -f "{tmp}"; }}'
            )
        else:
            before = src
        calls.append(_qc_call(qc, [(before, "uncorrected"), (_fmap_forward(pr), "corrected")]))
    head = "\n".join(lines) + "\n" if lines else ""
    return head + _qc_block("distortion", calls)


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
                f'-save_mean "{st}_mean.nii$FMT"',
                f'-prefix "{tmp}"',
                '-device "$DEVICE"',
            ],
        )
        mean += f'\n  rm -f "{tmp}"'
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
# lane images already have baked in, MEDIC (under -medic moco registers the
# undistorted echoes, so its lane images are already undistorted -- the warp again
# would correct them twice), and everything ABOVE grandmean space (which is by
# definition estimated from the grandmean, so it cannot act before it).
_GRANDMEAN_DROP = frozenset({"moco", "locomoco", "medic"}) | ABOVE_GRANDMEAN


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


_SHARED_ANAT_NOTE = (
    "# Anat prep (strip, autobox, subject TPM, ref-anat and MNI links) is shared by\n"
    "# every family: built once in the parent folder and linked in here (see the\n"
    "# shared section)."
)


def _anat_prep_blocks(opt) -> tuple[list[str], str]:
    """The anat-only preparation before the autobox — presurf, the T1w average,
    the skull strip — and the image ``$ANAT`` ends up naming. EPI-free, so a
    multi-family script runs it once for every family (:func:`_shared_anat_section`)."""
    blocks: list[str] = []
    if stage := _fs_stage_block(opt):
        blocks.append(stage)  # the anat itself is one of these: before anything reads it
    anat_ph = (
        opt.anat_path or "${FFS_ANAT:?set FFS_ANAT to the T1w to align to (SUMA brain.nii.gz)}"
    )
    if opt.presurf:
        blocks.append(_presurf_block(opt))
    inputs = _anat_inputs(opt, anat_ph)
    anat_ph = inputs[0]
    if opt.anat_extra:
        blocks.append(_anat_average_block(anat_ph, inputs[1:]))
        anat_ph = _anat_avg()
    if opt.anat_skull:
        blocks.append(_skullstrip_block(opt, anat_ph))
        anat_ph = _anat_brain()
    return blocks, anat_ph


def _anat_box_block() -> str:
    box = _anat_box()
    boxing = _ffs(
        "ffs_util_autobox",
        ['-input "$ANAT"', "-npad 3", f'-prefix "{box}"', '-device "$DEVICE"'],
    )
    return f'[ -f "{box}" ] || \\\n{boxing}'


def _needs_anat_tpm(opt) -> bool:
    """Whether some ffs_segment run reads the anat's subject TPM: the EPI fit
    (without -ref_anat; with it the EPI reads the ref's copy, made from this one)
    or the ref head's segmentation."""
    epi = opt.anat_nonlin and opt.anat_nonlin_method == "segment"
    return epi or bool(opt.ref_anat and _refanat_needs_seg(opt))


def _tpm_block(opt) -> str:
    """The subject-TPM build ffs_segment reads (the EPI fit, or -ref_anat's), if
    this run makes one."""
    if not _needs_anat_tpm(opt):
        return ""
    if opt.fs_tpm:
        return _fs_tpm_block(opt)
    if opt.anat_tpm:
        return _anat_tpm_block(opt)
    return ""


def _shared_anat_section(plan: Plan) -> str:
    """Multi-family: the anat-only work every family would otherwise redo — prep,
    autobox, subject TPM, the anat→MNI link — run once in the parent folder.

    Each family folder links these in (:func:`_link_shared_anat`) and its stage09
    leaves them out rather than relying on their exists-guards: under skip_anat=0 a
    family would otherwise re-estimate the MNI link through the symlink, after an
    earlier family had already been resampled with the old one."""
    opt = plan.options
    if not (plan.family_plans and _own_anat(opt)):
        return ""
    prep, anat_ph = _anat_prep_blocks(opt)
    parts = [
        "",
        "# ============================ shared: anat prep (all families) ==============",
        "echo '== shared anat prep =='",
        *prep,
        f'ANAT="{anat_ph}"',
        _anat_box_block(),
        _tpm_block(opt),
        # EPI-free like the MNI link, and shared for the same reason: families that
        # meet at the anat must reach it through ONE ref→anat warp.
        _stage_ref_anat(plan),
        _mni_estimate(plan),
    ]
    return "\n".join(p for p in parts if p) + "\n"


def _link_shared_anat(plan: Plan) -> str:
    """Link the parent folder's shared anat outputs into a family folder."""
    if not plan.shared_anat:
        return ""
    return (
        "# The shared anat prep, built once in the parent folder.\n"
        "for f in ../stage09.* ../presurf_work ../tpm_work; do\n"
        '  if [ -e "$f" ]; then ln -sfn "$f" .; fi\n'
        "done\n"
    )


def _stage_anat(plan: Plan) -> str:
    opt = plan.options
    if not opt.go_to_anat:
        return (
            "\n# ============================ stage09: anat (skipped) ======================\n"
            "# -no_anat / no anat given: final space is the EPI grandmean.\n"
        )
    out = ["", "# ============================ stage09: anatomical alignment ================="]
    out.append("echo '== stage09: anat alignment =='")
    # tpm_source default: an explicit -ref_file_anat (copied in) doubles as the anat in
    # tpm space, useful for ffs_segment QC; else the user's -tpm_source.
    tpm_src_path = opt.tpm_source
    if opt.ref_file is not None:
        # Explicit reference override: transforms are user-supplied; copy ref_file_anat
        # in (tpm-space anat + QC). anat_lin links = -ref_transforms (see chain).
        refmats = " ".join(shlex.quote(m) for m in (opt.ref_transforms or []))
        out.append("# explicit reference: -ref_transforms map ref→anat (baked into CHAIN).")
        if opt.ref_file_anat:
            out.append(f"cp -f {shlex.quote(opt.ref_file_anat)} stage09.ref_file_anat.nii.gz")
            tpm_src_path = tpm_src_path or "stage09.ref_file_anat.nii.gz"
        seg_mats = f"{refmats} stage09.xref.aff12.1D".strip()
    elif opt.grand_reference:
        gr = opt.grand_reference.rstrip("/")
        # anat matrix is BORROWED from the reference; segment (if any) inits from
        # borrowed ref2anat ∘ computed xref (matches the primary reference chain).
        out.append(f"# anat matrix borrowed from {gr}/stage09.anat.aff12.1D (not recomputed).")
        seg_mats = f"{gr}/stage09.anat.aff12.1D stage09.xref.aff12.1D"
    else:
        prep, anat_ph = _anat_prep_blocks(opt)
        if plan.shared_anat:
            out.append(_SHARED_ANAT_NOTE)
        else:
            out.extend(prep)
        src = _anat_source_image(plan)
        mode = effective_anat_source(plan)
        lin = _ffs(
            "ffs_allineate",
            [
                f'-base "{_anat_lin_base(opt)}"',
                f'-source "{src}"',
                f'-prefix "{_al_anat(plan)}"',
                '-1Dmatrix_save "stage09.anat.aff12.1D"',
                *_split_flags(config.DEFAULT_OPTS["anat"]),
                "-tbest 16",
                '-device "$DEVICE"',
            ],
        )
        # -ref_anat's segment reads the subject TPM, and the EPI matrix's base is
        # the ref, so both come first. The TPM build has its own exists-guard.
        ref = ""
        if opt.ref_anat and not plan.shared_anat:
            ref = "\n".join(p for p in (_tpm_block(opt), _stage_ref_anat(plan)) if p) + "\n"
        note = ""
        if opt.anat_source not in (mode, "auto"):
            note = f"  # (-anat_source {opt.anat_source} unavailable here → {mode})\n"
        out.append(
            f'ANAT="{anat_ph}"\n'
            "# Crop the anat's air away first: the alignment base, every EPI-in-anat\n"
            "# output, and the final grid all inherit this FOV, and it shrinks the search\n"
            "# space and the volume-sized allocations. Doubles as the viewing underlay.\n"
            "# The aff12 is in DICOM mm, but that is the BASE's cardinal frame, which\n"
            "# keeps the base's own origin — so on an oblique anat this crop is NOT a\n"
            "# free lunch: anything reading the matrix against the uncropped anat must\n"
            "# be told the crop was the base, or it slips by a fixed shift.\n"
            f"{'' if plan.shared_anat else _anat_box_block() + chr(10)}"
            f"{ref}"
            'if [ "$skip_anat" -ne 1 ] || [ ! -f "stage09.anat.aff12.1D" ]; then\n'
            "  # cross-modal rigid lpc: base=anat → matrix maps anat→EPI (chain head).\n"
            f"{"  # (-ref_anat: base = the ref anat, the EPI scanner's own.)" + chr(10) if opt.ref_anat else ''}"
            f"  # source = -anat_source {mode} (all choices share the reference-fmap grid).\n"
            f"{note}{lin}\nfi"
        )
        seg_mats = "stage09.anat.aff12.1D"

    if opt.anat_nonlin and opt.anat_nonlin_method == "register":
        out.append(_anat_nl_register(plan, seg_mats))
        out.append(_anat_nl_anchor_call(plan))
    elif opt.anat_nonlin:
        # Optionally build the subject TPM first: from FreeSurfer (SUMA), so no SPM
        # TPM is needed ahead of time, else from the anat's own segmentation.
        if not (plan.shared_anat or opt.ref_anat):
            out.append(_tpm_block(opt))
        if opt.ref_anat:
            # The EPI landed on the ref, so its priors are the ref's posteriors, on the
            # ref grid — whatever -tpm_source (or -suma's SurfVol) the anat TPM had.
            tpm_src_path = _refanat_box()
        elif opt.anat_tpm and not opt.fs_tpm:
            # The head is the image those priors were fitted to, so it is the
            # anat-in-EPI QC source that costs nothing extra to ask for.
            tpm_src_path = tpm_src_path or _anat_prestrip(opt)
        # One last ffs_segment on the CURRENT data (PE-mode nonlinear anat), init
        # from the anat matrix/matrices. Needs a subject TPM.
        tpm = (
            f'"{_refanat_tpm()}"'
            if opt.ref_anat
            else shlex.quote(opt.tpm)
            if opt.tpm
            else '"${FFS_TPM:?set FFS_TPM to the subject tissue-probability template}"'
        )
        # The FS-derived TPM has 8 hard-edge classes → its own ngaus/cleanup params.
        seg_tune = _split_flags(config.DEFAULT_OPTS["segment_fstpm" if opt.fs_tpm else "segment"])
        # The matrix's base is the anat autobox; the TPM is on the uncropped anat. On an
        # oblique anat those are different cardinal frames, so say which one it was.
        seg_base = _refanat_brain(opt) if opt.ref_anat else _anat_matrix_base(opt)
        seg = _ffs(
            "ffs_segment",
            [
                _segment_input(plan),
                f"-pe_axis {_pe_axis(plan)}",
                f"-tpm {tpm}",
                (f"-tpm_source {shlex.quote(tpm_src_path)}" if tpm_src_path else ""),
                f"-1Dmatrix {seg_mats}",
                (f'-matrix_base "{seg_base}"' if seg_base else ""),
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
    elif opt.ref_anat:
        # No anat_nl, but the ref links still sit above anat_lin.
        out.append(_anat_nl_anchor_call(plan))
    out.append(_stage_xfam(plan))
    out.append(_qc_anat(plan))
    out.append(_stage_mni(plan))
    return "\n".join(p for p in out if p) + "\n"


def _anat_nl_register(plan: Plan, matrix: str) -> str:
    """-anat_nonlin_method register: refine the anat anchor onto the anat with the
    anat_nl backend, along the PE axis only.

    In the anchor's own frame (-matrix): the base is pulled onto the anchor's grid,
    the native EPI grid every run's pre-chain ends on, so the PE axis is a grid
    axis there and -no?dis on the other two is exact rather than approximate on an
    oblique slab. A warp in that frame acts after anat_lin on the output
    coordinate, which is the anat_nl slot the chain already has. What it can fix is
    what a rigid anat matrix cannot: distortion left over along PE."""
    pe = _pe_axis(plan)
    frozen = [f"-no{a.upper()}dis" for a in "xyz" if a != pe]
    call = _ffs(
        _nl_tool(plan.options, "anat_nl"),
        [
            f'-base "{_anat_lin_base(plan.options)}"',
            f'-source "{_anat_source_image(plan)}"',
            f'-matrix "{matrix}"',
            '-prefix "stage09.nlanat.nii$FMT"',
            "-save_warp",
            '-warp_prefix "stage09.nlanat"',
            *frozen,
            *_split_flags(config.DEFAULT_OPTS["anat_nl"]),
            '-device "$DEVICE"',
        ],
    )
    return (
        f"# nonlinear anat by registration, PE axis ({pe}) only: {' '.join(frozen)}\n"
        f'if [ "$skip_anat" -ne 1 ] || [ ! -f "stage09.nlanat_WARP.nii$FMT" ]; then\n'
        f"{call}\nfi"
    )


def _refanat_box() -> str:
    """-ref_anat autoboxed — the grid every ref-side image shares. One lattice for
    the matrix's source, the segment input and the EPI matrix's base, because an
    .aff12.1D lives in its dataset's cardinal frame, and on an oblique ref a crop of
    it is a different frame ([[project_aff12_base_grid]])."""
    return "stage09.refanat_autobox.nii.gz"


def _refanat_presurf_image(opt) -> str:
    """-ref_anat's MPRAGEised head (ffs_util_presurf), named as _presurf_image is."""
    kind = "UNIreg" if opt.presurf_denoise == "reg" else "MPRAGEised"
    return f"presurf_work/refanat_{kind}.nii.gz"


def _refanat_brain(opt) -> str:
    """The ref anat the EPI aligns to: stripped on the box grid when the anat is
    being stripped too (-anat_skull), else the box itself, taken to be a brain."""
    return "stage09.refanat_brain.nii.gz" if opt.anat_skull else _refanat_box()


def _refanat_matrix() -> str:
    return stem(NameKey("refanat")) + ".aff12.1D"


def _refanat_warp(opt) -> str:
    """The refanat_nl link. .gz like the MNI warp: anat-sized, written once (in the
    parent folder when there are families), and opened beside the anat."""
    if opt.ref_anat_nonlin_method == "register":
        return stem(NameKey("refanat")) + "_nl_WARP.nii.gz"
    return _refanat_seg() + "_invwarp.nii.gz"


def _refanat_seg() -> str:
    """ffs_segment's prefix for the ref head against the anat's subject TPM."""
    return stem(NameKey("refanat")) + "_seg"


def _refanat_tpm() -> str:
    """The anat's subject TPM as the ref anat's own posteriors, on the ref grid —
    the priors -anat_nonlin's EPI fit reads when the EPI lands on the ref."""
    return "tpm_work/refanat_TPM.nii.gz"


def _refanat_in_anat() -> str:
    """The ref anat through both ref links onto the anat — the QC image that says
    whether the ref→anat warp did its job."""
    return stem(NameKey("refanat")) + "_in_anat.nii.gz"


def _anat_lin_base(opt) -> str:
    """The image the EPI's anat matrix is fitted against: the ref anat when there is
    one (the EPI's own scanner), else the anat autobox."""
    return _refanat_brain(opt) if opt.ref_anat else _anat_box()


def _refanat_needs_seg(opt) -> bool:
    """ffs_segment of the ref runs for the ref→anat warp (segment) and for the
    ref-space TPM a segment-driven -anat_nonlin reads, whichever warp is chained."""
    return opt.ref_anat_nonlin_method == "segment" or (
        opt.anat_nonlin and opt.anat_nonlin_method == "segment"
    )


def _stage_ref_anat(plan: Plan) -> str:
    """-ref_anat: the final anat → the same-scanner ref anat, rigid then full 3-D.

    The EPI and the ref share gradient coils (and usually a session), so whatever
    gradient nonlinearity the ref carries the EPI carries too, and anat_lin/anat_nl
    align the EPI to it as they would to -anat. The anat — another scanner, a
    better ventral brain — stays the final space, and this is the link between
    them: the same brain, so a rigid pose plus a deformation free on all three axes.

    The segment route reuses the anat's subject TPM: segmenting the ref head
    against it gives the warp (its source-side invwarp, chained after the matrix
    like anat_nl's) and, as posteriors, those priors on the ref grid. Needs $ANAT
    and the subject TPM, and runs before the EPI's anat matrix, whose base it makes."""
    opt = plan.options
    if not opt.ref_anat:
        return ""
    box, brain, mat = _refanat_box(), _refanat_brain(opt), _refanat_matrix()
    out = ["# --- -ref_anat: the anat → the ref anat (the EPI's own scanner) ---"]
    head = shlex.quote(opt.ref_anat)
    if opt.ref_presurf:
        # An MP2RAGE ref: UNI's background is amplified noise, which every cost
        # below would have to work around. Only the MPRAGEised head is used —
        # presurfer's own masks are not; the strip (if any) is synthstrip's.
        mprage = _refanat_presurf_image(opt)
        call = _presurf_call(opt, opt.ref_presurf, "presurf_work/refanat.nii.gz", ["-no_classes"])
        out.append(f'if [ ! -f "{mprage}" ]; then\n  mkdir -p presurf_work\n{call}\nfi')
        head = f'"{mprage}"'
    out.append(
        f'[ -f "{box}" ] || \\\n'
        + _ffs(
            "ffs_util_autobox",
            [
                f"-input {head}",
                "-npad 3",
                f'-prefix "{box}"',
                '-device "$DEVICE"',
            ],
        )
    )
    if opt.anat_skull:
        strip_opts = " ".join(_split_flags(config.DEFAULT_OPTS["synthstrip"]))
        out.append(f'[ -f "{brain}" ] || mri_synthstrip -i "{box}" -o "{brain}" {strip_opts}')
    # base = the anat, as for ffs_segment's -1Dmatrix: the matrix maps anat → ref,
    # which is also the direction the chain reads it in.
    lin = _ffs(
        "ffs_allineate",
        [
            f'-base "{_anat_box()}"',
            f'-source "{brain}"',
            f'-prefix "{stem(NameKey("refanat"))}_lin.nii.gz"',
            f'-1Dmatrix_save "{mat}"',
            *_split_flags(config.DEFAULT_OPTS["ref_anat"]),
            '-device "$DEVICE"',
        ],
    )
    out.append(f'if [ "$skip_anat" -ne 1 ] || [ ! -f "{mat}" ]; then\n{lin}\nfi')
    if _refanat_needs_seg(opt):
        tpm = shlex.quote(opt.tpm) if opt.tpm else '"${FFS_TPM:?set FFS_TPM}"'
        tune = [
            *(_split_flags(config.DEFAULT_OPTS["segment_fstpm"]) if opt.fs_tpm else []),
            *_split_flags(config.DEFAULT_OPTS["ref_anat_seg"]),
        ]
        # The head, skull on: the TPM's non-brain classes need the tissue they describe.
        seg = _ffs(
            "ffs_segment",
            [
                f'-input "{box}"',
                f"-tpm {tpm}",
                f'-1Dmatrix "{mat}"',
                f'-matrix_base "{_anat_box()}"',
                f'-prefix "{_refanat_seg()}.nii.gz"',
                *tune,
                '-device "$DEVICE"',
            ],
        )
        out.append(
            f'if [ "$skip_anat" -ne 1 ] || [ ! -f "{_refanat_seg()}_invwarp.nii.gz" ]; then\n'
            f"{seg}\nfi"
        )
        if opt.anat_nonlin and opt.anat_nonlin_method == "segment":
            # -V so c10 sorts after c9, as for the anat's own TPM. Only the TPM's own
            # classes: a posterior beyond them is the background class ffs_segment
            # synthesised (an FS TPM does not cover air), which the EPI fit adds back
            # itself — kept, it would be a "real" class -ngaus does not describe.
            out.append(
                f'[ -f "{_refanat_tpm()}" ] || {{\n  mkdir -p tpm_work\n'
                f"  tpm_n=$(ffs_info -nv {tpm})\n"
                + _ffs(
                    "ffs_util_3dmath",
                    [
                        f"-input $(ls {_refanat_seg()}_c*.nii* | grep -E '_c[0-9]+\\.nii' "
                        '| sort -V | head -n "$tpm_n")',
                        "-tcat",
                        f'-prefix "{_refanat_tpm()}"',
                        '-device "$DEVICE"',
                    ],
                )
                + "\n}"
            )
    if opt.ref_anat_nonlin_method == "register":
        # In the ref's frame (-matrix + the un-allineated ref): the warp then acts
        # after the matrix on the output coordinate, the same slot segment's
        # invwarp takes, so the chain does not depend on which method ran.
        warp = _refanat_warp(opt)
        nl = _ffs(
            _nl_tool(opt, "ref_anat_nl"),
            [
                f'-base "{_anat_box()}"',
                *_nl_source_args(True, "", brain, mat),
                f'-prefix "{stem(NameKey("refanat"))}_nl.nii.gz"',
                "-save_warp",
                f'-warp_prefix "{stem(NameKey("refanat"))}_nl"',
                *_split_flags(config.DEFAULT_OPTS["ref_anat_nl"]),
                '-device "$DEVICE"',
            ],
        )
        out.append(f'if [ "$skip_anat" -ne 1 ] || [ ! -f "{warp}" ]; then\n{nl}\nfi')
    links = f"{mat} {_refanat_warp(opt)}"
    out.append(
        f'[ -f "{_refanat_in_anat()}" ] || \\\n'
        + _ffs(
            "ffs_nwarp",
            [
                f'-source "{brain}"',
                f'-nwarp "{links}"',
                f'-master "{_anat_box()}"',
                *_split_flags(config.DEFAULT_OPTS["nwarp"]),
                f'-prefix "{_refanat_in_anat()}"',
                '-device "$DEVICE"',
            ],
        )
    )
    return "\n".join(out)


def _mni_lin_anat() -> str:
    """The anat affine-aligned to the template — the nonlinear step's source, and
    the image that says whether the affine alone was already in the right place."""
    return "stage09.anat_mni_lin.nii.gz"


def _anat_mni() -> str:
    """The anat in MNI: affine + nonlinear. The whole-brain underlay in the output
    space, and the grid stage10a crops the warpmaster out of."""
    return "stage09.anat_mni.nii.gz"


def _al_mni(plan: Plan) -> str:
    """The EPI anchor in MNI — the master the final grid is cut from.

    The anat lands in MNI by construction (it is what the warp was fitted on); the
    EPI has to be pushed through the whole head of the chain to get there, exactly
    as :func:`_al_anat_nl` does one link lower."""
    return f"stage09.{effective_anat_source(plan)}_al_mni.nii$FMT"


def _mni_head_chain(plan: Plan) -> list[str]:
    """Chain head down to the deepest anat-level link — mni_nl, mni_lin and
    everything between them and the space the anchor already sits in."""
    if not plan.runs:
        return []
    pr = plan.runs[0]
    toks = list(pr.warp_chain)
    last = "anat_nl" if "anat_nl" in toks else "anat_lin"
    if last not in toks:
        return []
    return chain_files(pr, ".nii$FMT", plan.options, tokens=toks[: toks.index(last) + 1])


def _stage_mni(plan: Plan) -> str:
    """-do_mni: anat → MNI template, affine then nonlinear (stage09b).

    Estimated on the SKULL-STRIPPED anat against the template's own stripped
    sub-brick — a skull is the loudest thing in a T1 and the least like the
    template's. Both links go at the head of every run's chain, so the data still
    makes exactly ONE trip through an interpolator: this stage warps nothing but
    the anat itself.

    Run late in stage09 because it needs $ANAT, and before stage10a because the
    warpmaster is cut out of what it produces."""
    opt = plan.options
    if not (opt.do_mni and opt.go_to_anat):
        return ""
    if opt.grand_reference:
        # Borrowed, like the anat matrix: nothing to compute, and the anchor image
        # comes from the reference dir too (see _final_master).
        gr = opt.grand_reference.rstrip("/")
        return (
            "\n# ============================ stage09b: MNI (borrowed) =====================\n"
            f"# anat→MNI warp borrowed from {gr}/stage09.mni* (not recomputed).\n"
        )
    return "\n".join(
        [
            "",
            "# ============================ stage09b: anat → MNI =========================",
            "# The last link in every run's chain, estimated here and applied once, at the",
            "# very end of stage10's single resample. Only the anat is warped in this",
            "# stage; the data never makes a second trip through an interpolator.",
            "echo '== stage09b: anat → MNI =='",
            _SHARED_ANAT_NOTE if plan.shared_anat else _mni_estimate(plan),
            _mni_anchor_call(plan),
            _qc_mni(plan),
        ]
    )


def _mni_estimate(plan: Plan) -> str:
    """The anat → MNI affine + warp. Estimated on the anat alone, so a multi-family
    script runs it once and every family reaches MNI through the same warp."""
    opt = plan.options
    if not (opt.do_mni and opt.go_to_anat and not opt.grand_reference):
        return ""
    # Sub-brick 0 of AFNI's SSW template is the skull-off volume; a plain template
    # has one volume and [0] is still it. Quoted whole: the selector's brackets are
    # a glob to the shell and belong to the tool.
    base = (
        shlex.quote(f"{opt.mni_template}[0]")
        if opt.mni_template
        else '"${FFS_MNI_TEMPLATE:?set -mni_template}[0]"'
    )
    lin = _ffs(
        "ffs_allineate",
        [
            f"-base {base}",
            f'-source "{_anat_box()}"',
            f'-prefix "{_mni_lin_anat()}"',
            f'-1Dmatrix_save "{stem(NameKey("mni"))}.aff12.1D"',
            *_split_flags(config.DEFAULT_OPTS["mni"]),
            '-device "$DEVICE"',
        ],
    )
    nl = _ffs(
        config.nl_command(opt.mni_backend),
        [
            f"-base {base}",
            f'-source "{_mni_lin_anat()}"',
            f'-prefix "{_anat_mni()}"',
            "-save_warp",
            f'-warp_prefix "{stem(NameKey("mni"))}_nl"',
            *_split_flags(config.DEFAULT_OPTS["mni_nonlin"]),
            '-device "$DEVICE"',
        ],
    )
    warp = _mni_files(opt, "_nl_WARP.nii.gz")[0]
    return "\n".join(
        [
            f'if [ "$skip_anat" -ne 1 ] || [ ! -f "{stem(NameKey("mni"))}.aff12.1D" ]; then',
            "  # base = the template's skull-off volume, source = the stripped anat.",
            f"{lin}",
            "fi",
            f'if [ "$skip_anat" -ne 1 ] || [ ! -f "{warp}" ]; then',
            f"{nl}",
            "fi",
        ]
    )


def _mni_anchor_call(plan: Plan) -> str:
    """Push the EPI anchor through the whole head of the chain into MNI.

    This is what stage10a boxes the output grid out of, so it has to be the anchor
    in the state the DATA ends up in — every link at or above the anat, MNI
    included — and on the same grid as the anat that will underlay it."""
    out = _al_mni(plan)
    head = _mni_head_chain(plan)
    if not head:
        return ""
    call = _ffs(
        "ffs_nwarp",
        [
            f'-source "{_anat_source_image(plan)}"',
            f'-nwarp "{" ".join(head)}"',
            f'-master "{_anat_mni()}"',
            *_split_flags(config.DEFAULT_OPTS["nwarp"]),
            f'-prefix "{out}"',
            '-device "$DEVICE"',
        ],
    )
    return f'[ -f "{out}" ] || \\\n{call}'


def _qc_mni(plan: Plan) -> str:
    """Template, anat-after-affine, anat-after-warp, and the EPI anchor — the four
    images that are all supposed to be the same brain in the same place."""
    opt = plan.options
    if not (_qc_on(plan) and opt.do_mni and not opt.grand_reference):
        return ""
    tmpl = opt.mni_template or "$FFS_MNI_TEMPLATE"
    items: QCItems = [
        (f"{tmpl}[0]", "template"),
        (_mni_lin_anat(), "anat_affine"),
        (_anat_mni(), "anat_nonlin"),
    ]
    if _mni_head_chain(plan):
        items.append((_al_mni(plan), f"{effective_anat_source(plan)}_al_mni"))
    return _qc_block("anat → MNI", [_qc_call(_qc_stem("mni"), items)])


def _anat_avg() -> str:
    """The mean of several T1w of one acquisition, aligned to the first.

    Averaging repeats is free SNR on the image every cross-modal cost function
    and (with -anat_skull yes) the strip has to work from, and it is the reason
    the anat can be left to the BIDS scan instead of pointed at by hand."""
    return "stage09.anat_avg.nii.gz"


def _anat_in_al(k: int) -> str:
    """One non-first T1w, rigidly aligned onto the first one's grid."""
    return f"stage09.anat_in{k}_al.nii.gz"


def _anat_average_block(base: str, extras: list[str]) -> str:
    """Align every extra T1w to the first and average them.

    Plain .gz throughout: these are images to look at when the mean comes out
    blurred, and stock AFNI cannot open .nii.zst."""
    al_opts = _split_flags(config.DEFAULT_OPTS["anat_avg"])
    avg = _anat_avg()
    steps = []
    for k, extra in enumerate(extras, start=2):
        steps.append(
            _ffs(
                "ffs_allineate",
                [
                    f'-base "{base}"',
                    f"-source {shlex.quote(extra)}",
                    f'-prefix "{_anat_in_al(k)}"',
                    *al_opts,
                    '-device "$DEVICE"',
                ],
            )
        )
    aligned = [_anat_in_al(k) for k in range(2, len(extras) + 2)]
    inputs = " ".join(f'"{p}"' for p in [base, *aligned])
    steps.append(
        _ffs(
            "ffs_util_3dmath",
            [f"-input {inputs}", "-mean", f'-prefix "{avg}"', '-device "$DEVICE"'],
        )
    )
    body = "\n".join(steps)
    n = len(extras) + 1
    return f"""# --- average {n} T1w of one acquisition (aligned to the first) ---
if [ ! -f "{avg}" ]; then
{body}
fi"""


def _presurf_image(opt, k: int) -> str:
    """The k-th (1-based) anat input's head image from ffs_util_presurf.

    Named by what -denoise made it: UNIreg is the regularised UNI, not an
    MPRAGEised one, and a filename that says otherwise misleads at the QC."""
    kind = "UNIreg" if opt.presurf_denoise == "reg" else "MPRAGEised"
    return f"presurf_work/anat{k}_{kind}.nii.gz"


def _anat_inputs(opt, first: str) -> list[str]:
    """What the average (or, with one input, the strip) starts from: the presurf
    heads when the inputs are MP2RAGE, else the T1w as given."""
    if opt.presurf:
        return [_presurf_image(opt, k) for k in range(1, len(opt.presurf) + 1)]
    return [first, *opt.anat_extra]


def _presurf_call(opt, mp2rage: tuple[str, str, str | None], prefix: str, flags: list[str]) -> str:
    """One ffs_util_presurf run: (uni, inv2, inv1-or-None) → ``prefix``'s head."""
    uni, inv2, inv1 = mp2rage
    tpm = shlex.quote(opt.presurf_tpm) if opt.presurf_tpm else '"${FFS_TPM:?set FFS_TPM}"'
    return _ffs(
        "ffs_util_presurf",
        [
            f"-uni {shlex.quote(uni)}",
            f"-inv2 {shlex.quote(inv2)}",
            f"-inv1 {shlex.quote(inv1)}" if inv1 else "",
            f"-tpm {tpm}",
            f'-prefix "{prefix}"',
            f"-denoise {opt.presurf_denoise}" if opt.presurf_denoise != "off" else "",
            *flags,
            '-device "$DEVICE"',
        ],
    )


def _presurf_block(opt) -> str:
    """MP2RAGE inv1/inv2/uni → an MPRAGE-like head, once per anat input.

    UNI is a ratio image: its background is noise amplified to full brightness,
    which synthstrip and every cross-modal cost then have to work around.
    MPRAGEising multiplies it by the bias-corrected INV2, which is dark there."""
    base_flags = _split_flags(config.DEFAULT_OPTS["presurf"])
    steps = []
    for k, (uni, inv2, inv1) in enumerate(opt.presurf, start=1):
        # The first anat's tissue classes are -phasereg's vein-mask tissue maps
        # (_tissue_files): keep them then, and rerun a presurf that skipped them.
        classes = k == 1 and _presurf_classes_needed(opt)
        flags = [f for f in base_flags if not (classes and f == "-no_classes")]
        call = _presurf_call(opt, (uni, inv2, inv1), f"presurf_work/anat{k}.nii.gz", flags)
        need = f'[ ! -f "{_presurf_image(opt, k)}" ]'
        if classes:
            need += f' || [ ! -f "{_tissue_files(opt)[0]}" ]'
        steps.append(f"if {need}; then\n{call}\nfi")
    return "# --- MP2RAGE → MPRAGEised head (presurfer) ---\n" + "\n".join(steps)


def _anat_head() -> str:
    """The anat as it arrived, with its skull (-anat_skull yes).

    Written even though nothing downstream reads it: a strip is a decision, and
    the only way to see what came off is to still have what went in. Plain .gz so
    it opens in stock AFNI next to the brain."""
    return "stage09.anat_head.nii.gz"


def _anat_brain() -> str:
    """The skull-stripped anat (-anat_skull yes) — what stage09 actually aligns
    to, standing in for the SUMA brain.nii.gz a FreeSurfer subject would have."""
    return "stage09.anat_brain.nii.gz"


def _anat_prestrip(opt) -> str:
    """The image mri_synthstrip reads — and the record of what went into the
    strip: the average when there were several T1w, else the copy kept of the one."""
    return _anat_avg() if opt.anat_extra else _anat_head()


def _skullstrip_block(opt, src: str) -> str:
    """Strip the anat in-script with FreeSurfer's mri_synthstrip (>= 8).

    A single input anat is first copied in beside its brain, so the results dir
    still holds what went into the strip — byte for byte when it is already a
    .nii.gz (keeping its integer dtype), else through ffs_util_3dmath, because
    the copy has to really be the .nii.gz its name claims for stock AFNI to open
    it. An averaged anat is already that record and is stripped where it lies."""
    strip_opts = " ".join(_split_flags(config.DEFAULT_OPTS["synthstrip"]))
    head, brain = _anat_prestrip(opt), _anat_brain()
    if opt.anat_extra:
        copy = ""
    elif src.endswith(".nii.gz"):
        copy = f'  cp -f "{src}" "{head}"\n'
    else:
        copy = (
            _ffs(
                "ffs_util_3dmath",
                [f'-input "{src}"', "-expr 'a'", f'-prefix "{head}"', '-device "$DEVICE"'],
            )
            + "\n"
        )
    return f"""# --- skull strip the anat (-anat_skull yes) ---
if [ ! -f "{brain}" ]; then
{copy}  mri_synthstrip -i "{head}" -o "{brain}" {strip_opts}
fi"""


# aseg.auto label sets per TPM class. aseg.auto's label 24 covers the sulcal CSF
# too, so the CSF classes reach the pial veins the vein mask looks for.
ASEG_TPM_LABELS = [
    ("tpm1_cortex", "3,42"),
    ("tpm2_cerebelgm", "8,47"),
    ("tpm3_cortwm", "2,41,28,60,16,252,251,253,254,255"),
    ("tpm4_cerebelwm", "7,46"),
    ("tpm5_subcortexgm", "5,10,11,12,13,17,18,49,50,51,52,53,54,44,58,26"),
    ("tpm6_vents", "4,14,43"),
    ("tpm7_outercsf", "24"),
]


#: Where the staged FreeSurfer volumes land, under the work dir.
FS_WORK_DIR = "fs_work"

#: What autoproc needs out of a FreeSurfer subject, as (staged NIfTI, mri/ source).
#: Each is byte-identical to the file @SUMA_Make_Spec_FS writes into SUMA/ -- checked
#: on a 0.8 mm recon: brain.nii.gz == brain.mgz, aseg.auto.nii.gz == aseg.auto.mgz,
#: sub_SurfVol.nii == orig.mgz, data and affine both. So -fs_subject and -suma read
#: the same voxels, and SUMA/ is only ever a NIfTI repackaging of these three.
FS_STAGED_VOLUMES = (
    ("brain.nii.gz", "brain.mgz"),  # the anat: FreeSurfer's normalised, stripped T1
    ("aseg.auto.nii.gz", "aseg.auto.mgz"),  # the 8-class TPM, and phasereg's tissue
    ("fs_SurfVol.nii.gz", "orig.mgz"),  # the TPM's skull class, and tpm_source
)


def _fs_stage_block(opt) -> str:
    """FreeSurfer's own volumes as NIfTI, so -fs_subject needs no SUMA directory.

    The conversion exists because of AFNI, not us: ``load_image`` reads .mgz
    directly, but the subject-TPM build is 3dcalc on aseg label sets and stock
    AFNI reports NO-DSET for .mgz. Staged once under exists-guards, into the
    absolute path ``suma_dir`` already names, so every FreeSurfer-priors path
    downstream -- the TPM, phasereg's tissue maps, the anat itself -- is the
    -suma path unchanged. Verified bit-identical through 3dcalc's value selector.
    """
    if not opt.fs_staged:
        return ""
    src = Path(opt.fs_subject) / "mri"
    out = Path(opt.suma_dir)
    lines = [
        "# --- FreeSurfer volumes -> NIfTI (-fs_subject without -suma) ---",
        f"mkdir -p {shlex.quote(str(out))}",
    ]
    for name, mgz in FS_STAGED_VOLUMES:
        dst = shlex.quote(str(out / name))
        call = _ffs(
            "ffs_util_3dmath",
            [
                f"-input {shlex.quote(str(src / mgz))}",
                "-expr a",
                f"-prefix {dst}",
                '-device "$DEVICE"',
            ],
        )
        lines.append(f"[ -f {dst} ] || \\\n{call}")
    return "\n".join(lines)


def _fs_tpm_block(opt) -> str:
    """Build a subject-specific 8-class TPM from FreeSurfer (SUMA) outputs using
    AFNI (aseg.auto label sets + a skull class from SurfVol). Hard-edge tissue
    priors in FS space — no ahead-of-time SPM TPM needed. Runs in a subshell so
    the ``cd`` doesn't leak. Output: tpm_work/sub_specific_fsTPM.nii.gz."""
    suma = shlex.quote(opt.suma_dir.rstrip("/")) if opt.suma_dir else "${SUMA_DIR:?set -suma}"
    calc = "\n".join(
        f"  3dcalc -overwrite -a {suma}/aseg.auto.nii.gz'<{sel}>' -expr 'step(a)' -prefix {name}.nii.gz"
        for name, sel in ASEG_TPM_LABELS
    )
    return f"""# --- build subject TPM from FreeSurfer (SUMA) ---
if [ ! -f "tpm_work/sub_specific_fsTPM.nii.gz" ]; then
  mkdir -p tpm_work
( cd tpm_work
{calc}
  3dMean -sum -overwrite -prefix sum_sub_tpm.nii.gz tpm[1-7]_*.nii.gz
  # skull class = whatever SurfVol has outside the labelled tissue.
  3dcalc -overwrite -a sum_sub_tpm.nii.gz -b {suma}/*_SurfVol.nii* \\
    -expr 'abs(step(a)-1)*b' -prefix skull.nii.gz
  3dAutomask -overwrite -prefix tpm8_skull.nii.gz skull.nii.gz
  3dTcat -overwrite -prefix sub_specific_fsTPM.nii.gz tpm[1-8]_*.nii.gz
)
fi"""


def _anat_tpm_file() -> str:
    """The subject TPM built from the anat itself (the no-FreeSurfer path)."""
    return "tpm_work/sub_specific_anatTPM.nii.gz"


def _anat_t2_al() -> str:
    """The T2w resampled onto the anat head's grid — a segmentation channel has to
    sit on the same lattice as the first one.

    Kept beside the head and brain rather than in tpm_work: it is a second
    contrast of this subject in the final anat space, which is a QC underlay and
    a tissue reference long after the TPM it was made for has been consumed."""
    return "stage09.anat_t2.nii.gz"


def _anat_tpm_block(opt) -> str:
    """Build a subject TPM by segmenting the anat against an SPM-style group TPM.

    The same trick as the FreeSurfer path, one step earlier: what ffs_segment
    needs in EPI mode is tissue priors *in anat space*, and a segmentation of the
    anat produces exactly that — its posteriors are the subject's own tissue, on
    the subject's own grid, which is all the EPI fit reads them as. Fewer and
    softer classes than aseg's hard labels, so -suma still wins when there is a
    FreeSurfer run; this is the path for a T1w straight off the scanner.

    The head goes in with its skull on: SPM's TPM carries bone/soft-tissue/air
    classes, and segmenting a stripped brain against them leaves those classes
    with nothing to claim — the EPI fit then has no prior for everything outside
    the brain, which in a distortion fit is where the signal it must not chase
    lives. A T2w channel is added when one was found, since bone-from-CSF and
    dura-from-grey are the separations a T1 alone cannot make."""
    head = _anat_prestrip(opt)
    tpm = shlex.quote(opt.spm_tpm) if opt.spm_tpm else '"${FFS_TPM:?set FFS_TPM}"'
    out = _anat_tpm_file()
    steps = []
    inputs = [head]
    if opt.anat_t2:
        inputs.append(_anat_t2_al())
        steps.append(
            _ffs(
                "ffs_allineate",
                [
                    f'-base "{head}"',
                    f"-source {shlex.quote(opt.anat_t2)}",
                    f'-prefix "{_anat_t2_al()}"',
                    *_split_flags(config.DEFAULT_OPTS["anat_t2"]),
                    '-device "$DEVICE"',
                ],
            )
        )
    steps.append(
        _ffs(
            "ffs_segment",
            [
                "-input " + " ".join(f'"{p}"' for p in inputs),
                f"-tpm {tpm}",
                '-prefix "tpm_work/anatseg"',
                *_split_flags(config.DEFAULT_OPTS["anat_tpm"]),
                '-device "$DEVICE"',
            ],
        )
    )
    # The class count is the group TPM's 4th dimension, which only the file knows,
    # so the stack is globbed at run time. -V so c10 sorts after c9, not after c1.
    steps.append(
        _ffs(
            "ffs_util_3dmath",
            [
                "-input $(ls tpm_work/anatseg_c*.nii* | grep -E '_c[0-9]+\\.nii' | sort -V)",
                "-tcat",
                f'-prefix "{out}"',
                '-device "$DEVICE"',
            ],
        )
    )
    body = "\n".join(steps)
    chan = " (T1 + T2)" if opt.anat_t2 else ""
    return f"""# --- build subject TPM from the anat{chan} ---
if [ ! -f "{out}" ]; then
  mkdir -p tpm_work
{body}
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
    of the chain, and is what stage10a boxes the output grid out of."""
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
    last = "anat_nl" if "anat_nl" in toks else "anat_lin"
    # Without anat_nl the anchor needs a head only when -ref_anat put links above
    # anat_lin; otherwise ffs_allineate's own output is already it.
    if last not in toks or (last == "anat_lin" and "refanat_lin" not in toks):
        return []
    # ...but NOT the -do_mni pair above it: this image is the anchor in ANAT
    # space (it is mastered on the anat), and stage09b's own anchor is the one
    # that goes on to MNI. Nor xfam_lin: it is estimated FROM this anchor, and its
    # own output (stage09.xfam_lin) is the anchor with that link applied.
    head = [t for t in toks[: toks.index(last) + 1] if not t.startswith(("mni_", "xfam_"))]
    return chain_files(pr, ".nii$FMT", plan.options, tokens=head)


def _anat_nl_anchor_call(plan: Plan) -> str:
    """The ffs_nwarp that builds :func:`_al_anat_nl`, or "" when there is nothing
    to apply.

    Only for a locally-computed anat: the borrowing modes take their grid from the
    reference results dir, so the link that matters there is the *reference's*
    anat_nl, already baked into the image :func:`_final_master` globs."""
    if not ((plan.options.anat_nonlin or plan.options.ref_anat) and _own_anat(plan.options)):
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
        return "stage09.ref_file_anat.nii.gz" if opt.ref_file_anat else _grandmean()
    if opt.grand_reference:
        # Borrow the reference's anat-space grid so all subjects/tasks co-register.
        # Globbed because the reference's file is named for ITS -anat_source, which
        # this script has no way to know.
        gr = opt.grand_reference.rstrip("/")
        if opt.do_mni:
            # ...and with -do_mni the shared space is MNI, one link further out.
            return f"$(ls {gr}/stage09.*_al_mni.nii* 2>/dev/null | head -1 || true)"
        # The trailing `true` lets a miss leave MASTER empty for the guard below to
        # report, rather than tripping `set -e` with no explanation.
        # _al_anat_nl first: with -anat_nonlin that is the shape the reference's
        # own data landed in, and the affine-only _al_anat is one link short of it.
        # stage09.anat is the same anchor under its pre-926966c5 name, so references
        # processed before that rename still resolve. First match in pattern order,
        # not `ls | head`: ls sorts its output, and "_al_anat." sorts before
        # "_al_anat_nl.", which silently picked the affine-only anchor.
        return (
            f"$(for f in {gr}/stage09.*_al_anat_nl.nii* {gr}/stage09.*_al_anat.nii* "
            f'{gr}/stage09.anat.nii*; do [ -e "$f" ] && {{ echo "$f"; break; }}; done; true)'
        )
    if opt.do_mni:
        return _al_mni(plan)
    if plan.xfam_root is not None:
        return _xfam_nl_anchor() if plan.options.xfam_nonlin else _xfam_image()
    return _anat_anchor(plan)


def _anat_anchor(plan: Plan) -> str:
    """This family's EPI anchor in anat space, in the state its data lands in
    below any cross-family link: through anat_nl when there is one."""
    opt = plan.options
    return _al_anat_nl(plan) if (opt.anat_nonlin or opt.ref_anat) else _al_anat(plan)


def _xfam_matrix() -> str:
    return stem(NameKey("xfam")) + ".aff12.1D"


def _xfam_image() -> str:
    """The child's anchor after xfam_lin, on the root anchor's grid (allineate's
    own output) — the image this family's output grid and mask are cut from."""
    return stem(NameKey("xfam")) + "_lin.nii$FMT"


def _stage_xfam(plan: Plan) -> str:
    """Align this family's corrected anchor to the root family's (stage09).

    Both anchors are EPI contrast in anat space, each already through its own
    family's anat_lin and anat_nl, so this is a same-contrast rigid fit (lpa) that
    pins the two families to each other far more tightly than two separate
    cross-modal fits to the anat can. Its matrix goes at the head of every run's
    chain (xfam_lin), so the root's anat placement is the one both families share.
    -source_automask keeps the child's out-of-slab zeros out of the cost, so only
    the brain the two slabs share drives it.
    """
    if plan.xfam_root is None:
        return ""
    fam, root_plan = plan.xfam_root
    base = f"../{family_dir(fam)}/{_anat_anchor(root_plan)}"
    call = _ffs(
        "ffs_allineate",
        [
            f'-base "{base}"',
            f'-source "{_anat_anchor(plan)}"',
            f'-prefix "{_xfam_image()}"',
            f'-1Dmatrix_save "{_xfam_matrix()}"',
            *_split_flags(config.DEFAULT_OPTS["xfam"]),
            '-device "$DEVICE"',
        ],
    )
    qc = ""
    if _qc_on(plan):
        qc = _qc_block(
            "cross-family",
            [
                _qc_call(
                    _qc_stem("xfam"),
                    [
                        (base, f"root_{fam.label}"),
                        (_anat_anchor(plan), "anchor_own"),
                        (_xfam_image(), "anchor_xfam"),
                    ],
                )
            ],
        )
    out = (
        f"# ---- stage09: align to the root family ({fam.label}) ----\n"
        "# This family's corrected anchor onto the root family's, both in anat space.\n"
        f'if [ "$skip_anat" -ne 1 ] || [ ! -f "{_xfam_matrix()}" ]; then\n{call}\nfi\n{qc}'
    )
    return out + _stage_xfam_nl(plan, base)


def _xfam_nl_anchor() -> str:
    """The child's anchor in anat space through its WHOLE head of chain, the
    cross-family warp included — what the output grid is cut from under
    -xfam_nonlin (allineate's stage09.xfam_lin stops at the rigid link)."""
    return stem(NameKey("xfam")) + "_nl_al_anat.nii$FMT"


def _stage_xfam_nl(plan: Plan, root_anchor: str) -> str:
    """-xfam_nonlin: warp this family onto the root along its own PE axis.

    Not on the anat grid: there both anchors have been through anat_lin, and this
    family's PE axis is a diagonal. So the root's corrected anchor is pulled BACK
    onto this family's EPI grid through the inverse of the two rigid links
    (INV(anat_lin) acts first on the output coordinate, then INV(xfam)), and fitted
    there against this family's own grandmean through its anat_nl — two images on
    one grid whose PE axis is a grid axis, so -no?dis on the other two is exact.
    The warp lands between anat_lin and anat_nl, in the frame anat_nl acts in.

    Where the root has no data the pulled-back image is zero, and the backends'
    default coverage handling drops it from the cost, so only the shared slab
    drives the fit; outside it the field is left to the regularisation.
    """
    if plan.xfam_root is None or not plan.options.xfam_nonlin or not plan.runs:
        return ""
    opt = plan.options
    pr = plan.runs[0]
    own = _anat_source_image(plan)
    anat_mats = _anat_lin_files(opt)
    pull = " ".join([*(f"INV({m})" for m in anat_mats), f"INV({_xfam_matrix()})"])
    root_in_epi = stem(NameKey("xfam")) + "_root_in_epi.nii$FMT"
    parts = [
        "# ---- stage09: cross-family warp, PE axis only (-xfam_nonlin) ----",
        "# The root's corrected anchor, pulled back onto this family's EPI grid.",
        f'if [ "$skip_anat" -ne 1 ] || [ ! -f "{root_in_epi}" ]; then',
        _ffs(
            "ffs_nwarp",
            [
                f'-source "{root_anchor}"',
                f'-nwarp "{pull}"',
                f'-master "{own}"',
                *_split_flags(config.DEFAULT_OPTS["nwarp"]),
                f'-prefix "{root_in_epi}"',
                '-device "$DEVICE"',
            ],
        ),
        "fi",
    ]
    corrected = own
    if "anat_nl" in pr.warp_chain:
        corrected = stem(NameKey("xfam")) + "_own_in_epi.nii$FMT"
        parts += [
            "# ...and this family's own grandmean through its anat_nl, on the same grid.",
            f'if [ "$skip_anat" -ne 1 ] || [ ! -f "{corrected}" ]; then',
            _ffs(
                "ffs_nwarp",
                [
                    f'-source "{own}"',
                    f'-nwarp "{" ".join(chain_files(pr, ".nii$FMT", opt, tokens=["anat_nl"]))}"',
                    f'-master "{own}"',
                    *_split_flags(config.DEFAULT_OPTS["nwarp"]),
                    f'-prefix "{corrected}"',
                    '-device "$DEVICE"',
                ],
            ),
            "fi",
        ]
    pe = _pe_axis(plan)
    frozen = [f"-no{a.upper()}dis" for a in "xyz" if a != pe]
    warp = chain_files(pr, ".nii$FMT", opt, tokens=["xfam_nl"])[0]
    parts += [
        f"# PE axis ({pe}) only: {' '.join(frozen)}",
        f'if [ "$skip_anat" -ne 1 ] || [ ! -f "{warp}" ]; then',
        _ffs(
            _nl_tool(opt, "xfam_nl"),
            [
                f'-base "{root_in_epi}"',
                f'-source "{corrected}"',
                f'-prefix "{stem(NameKey("xfam"))}_nl.nii$FMT"',
                "-save_warp",
                f'-warp_prefix "{stem(NameKey("xfam"))}_nl"',
                *frozen,
                *_split_flags(config.DEFAULT_OPTS["xfam_nl"]),
                '-device "$DEVICE"',
            ],
        ),
        "fi",
    ]
    # The anchor through the whole head, so the output grid and mask are cut from
    # the shape the data lands in.
    head = [
        t for t in pr.warp_chain[: pr.warp_chain.index("xfam_nl") + 1] if not t.startswith("mni_")
    ]
    if "anat_nl" in pr.warp_chain:
        head.append("anat_nl")
    anchor = _xfam_nl_anchor()
    parts += [
        f'if [ "$skip_anat" -ne 1 ] || [ ! -f "{anchor}" ]; then',
        _ffs(
            "ffs_nwarp",
            [
                f'-source "{own}"',
                f'-nwarp "{" ".join(chain_files(pr, ".nii$FMT", opt, tokens=head))}"',
                f'-master "{_anat_box()}"',
                *_split_flags(config.DEFAULT_OPTS["nwarp"]),
                f'-prefix "{anchor}"',
                '-device "$DEVICE"',
            ],
        ),
        "fi",
    ]
    if _qc_on(plan):
        parts.append(
            _qc_block(
                "cross-family warp",
                [
                    _qc_call(
                        _qc_stem("xfam", role="nl"),
                        [
                            (root_in_epi, "root_in_epi"),
                            (corrected, "own"),
                            (f"{stem(NameKey('xfam'))}_nl.nii$FMT", "own_xfam_nl"),
                        ],
                    )
                ],
            )
        )
    return "\n" + "\n".join(p for p in parts if p) + "\n"


def _final_anat(plan: Plan) -> str:
    """The whole-brain anatomical in the FINAL space — the ancestor of the results
    underlay. With -do_mni that is the anat warped to the template, not the anat's
    own cropped grid, or the underlay would be a brain the data never lands on."""
    return _anat_mni() if plan.options.do_mni else _anat_box()


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
    carrying the anat's empty space. No mask is made here — an automask of the
    anchor is not an automask of the data; that is stage10b's job. Two anat
    underlays come out of this: the whole-brain
    stage09.anat_autobox and stage10.anat_in_epi_fov, the same anat at anat
    resolution over the warpmaster's FOV — the one to overlay results on. MASTER
    and FINAL_DXYZ are defined here once and reused by stage10 (same shell); the
    FFS_MASTER / FFS_FINAL_DXYZ overrides are still honoured."""
    opt = plan.options
    box = "stage10.warpmaster_box.nii$FMT"
    wm = "stage10.warpmaster.nii$FMT"
    anat_fov = "stage10.anat_in_epi_fov.nii.gz"

    def guarded(outfile: str, tool: str, parts: list[str]) -> str:
        return f'[ -f "{outfile}" ] || \\\n' + _ffs(tool, parts)

    out = [
        "",
        "# ============================ stage10a: warpmaster ========================",
        "# Fix the final output grid before the resample. The warpmaster is the",
        "# anat-space EPI target autoboxed to the EPI's own coverage and resampled to",
        "# the EPI voxel size; stage10 lands every run on it (-master). The analysis",
        "# masks come after the data does, in stage10b.",
    ]
    if _own_anat(opt):
        out += [
            "# stage10.anat_in_epi_fov is the anat on that same FOV at anat resolution —",
            "# the underlay for results; stage09.anat_autobox is the whole-brain one.",
        ]
    out += [
        "echo '== stage10a: warpmaster =='",
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
                    f'-input "{_final_anat(plan)}"',
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
    return "\n".join(out) + "\n"


def _final_st_line(plan: Plan) -> str:
    """Bash setting ``st_str``: the -tpattern flags a final resample folds slice timing
    in with (``-slicetiming_method integrate``), or nothing."""
    opt = plan.options
    if opt.slicetiming_method != "integrate":
        return '  st_str=""'
    return (
        f"  tp={_tpattern(plan)}; " + 'tr="${TR[$k]}"\n'
        '  if [ -n "$tr" ]; then st_str="-tpattern \\"$tp\\" -TR \\"$tr\\" '
        f'-tzero {opt.tzero:g}"; '
        'else st_str=""; fi'
    )


def _stage_final(plan: Plan, script_stem: str) -> str:
    st = _final_st_line(plan)
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
        phase_args = "$pa"
        phase_note = (
            "# The unwrapped phase rides the same chain in the same pass\n"
            "# (-phase_warp direct, radians) → stage10.final.<frag>.part-phase.*\n"
        )
    # A run without phase (a spin-echo run beside gradient-echo ones) gets no
    # phase arguments: pa stays empty.
    phase_out = (
        '  phoutf="stage10.final.${FRAG[$k]}.part-phase.nii$FINAL_FMT"; pa=""\n'
        '  [ -n "${PHASE[$k]:-}" ] && pa=" -phase \\"'
        + _phase_source(plan)
        + '\\" -phase_units rad -phase_warp direct -phase_prefix \\"$phoutf\\""\n'
        if _phase_on(plan)
        else ""
    )
    meanf = _final_mean("${FRAG[$k]}")
    surf_args, surf_note = _surface_args(plan), ""
    if surf_args:
        surf_note = (
            "# -surface_target: the same process also reads each run at cortical surface\n"
            "# points (-surf_prefix stage10s.surf.<frag>), from the same read of the source,\n"
            "# through SURFCHAIN (CHAIN without the MNI links) onto $ANAT, whose scanner\n"
            "# space the FreeSurfer surfaces share: one interpolation for both outputs.\n"
        )
    # -save_mean is not optional: stage10.final.<frag>_mean is the per-run image
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
# -save_mean writes stage10.final.<frag>_mean per run — the QC stack below.
{phase_note}{surf_note}echo '== stage10: final compose + resample =='
nwarpbatch="{batchfile}"
: > "$nwarpbatch"
for k in "${{RUN_KEYS[@]}}"; do
  outf="stage10.final.${{FRAG[$k]}}.nii$FINAL_FMT"
  meanf="{meanf}"
{phase_out}{st}
{_raw_source(plan)}
{_final_me_jobs(plan, nwarp_flags)}  printf '%s\\n' "-source \\"$raw\\" -nwarp \\"${{CHAIN[$k]}}\\"${{JAC[$k]:+ -jac \\"${{JAC[$k]}}\\"}} -master stage10.warpmaster.nii$FMT -chain_base \\"$MASTER\\" -dxyz \\"$FINAL_DXYZ\\" {nwarp_flags} $st_str -save_mean \\"$meanf\\" -prefix \\"$outf\\"{phase_args}{surf_args}" >> "$nwarpbatch"
done
{_numcomps_final_jobs(plan)}{_sbref_final_jobs(plan)}batch_skip=(); [ "$skip_final" -eq 1 ] && batch_skip=(-batch_skip)
ffs_nwarp -batch "$nwarpbatch" "${{batch_skip[@]}}" -device "$DEVICE"
{_final_me_combine(plan)}echo 'done → stage10.final.*'
{_qc_final(plan)}
"""


def _final_me_jobs(plan: Plan, nwarp_flags: str) -> str:
    """stage10 manifest lines for a multi-echo run: one resample per echo, each with
    its own chain (ECHAIN) and phase, written as stage10.final.<frag>.echo-N; the
    run's own name is left for the optimal combination (_final_me_combine). Echo 1's
    job writes the QC mean. Ends the iteration for a multi-echo run."""
    if not _any_me(plan):
        return ""
    phase = ""
    if _phase_on(plan):
        phase = (
            '      [ -n "${PHASE[$k]:-}" ] && pe_args=" -phase \\"'
            + _echo_phase_file(plan)
            + '\\" -phase_units rad -phase_warp direct -phase_prefix '
            '\\"stage10.final.${FRAG[$k]}.echo-$e.part-phase.nii$FINAL_FMT\\""\n'
        )
    return (
        '  if [ "${NECHO[$k]:-1}" -gt 1 ]; then\n'
        '    for e in $(seq 1 "${NECHO[$k]}"); do\n'
        '      esfx=".echo-$e"\n'
        + _echo_source(plan)
        + '\n      eoutf="stage10.final.${FRAG[$k]}.echo-$e.nii$FINAL_FMT"; pe_args=""; em=""\n'
        + phase
        + _me_surface_args(plan)
        + '      [ "$e" -eq 1 ] && em=" -save_mean \\"$meanf\\""\n'
        '      printf \'%s\\n\' "-source \\"$mag\\" -nwarp \\"${ECHAIN[$k|$e]}\\"'
        '${JAC[$k]:+ -jac \\"${JAC[$k]}\\"} -master stage10.warpmaster.nii$FMT '
        '-chain_base \\"$MASTER\\" -dxyz \\"$FINAL_DXYZ\\" '
        + nwarp_flags
        + ' $st_str$em -prefix \\"$eoutf\\"$pe_args$sa" >> "$nwarpbatch"\n'
        "    done\n"
        "    continue\n"
        "  fi\n"
    )


def _me_surface_args(plan: Plan) -> str:
    """Bash setting ``sa``: echo $e's surface read in its stage10 job (its own chain, an
    .echo-N prefix). Every echo keeps the shared -surf_geom_prefix -- the geometry
    depends only on subject and mesh, and without it ffs_nwarp writes a copy under each
    echo's own stem; only the QC is echo 1's alone."""
    if not plan.options.surface_target:
        return '      sa=""\n'
    full = (
        _surface_args(plan)
        .replace('stage10s.surf.${FRAG[$k]}\\"', 'stage10s.surf.${FRAG[$k]}.echo-$e\\"')
        .replace("${SURFCHAIN[$k]}", "${SURFCHAIN[$k|$e]}")
    )
    lean = full.replace(" -surf_qc", "")
    return f'      sa="{lean}"; [ "$e" -eq 1 ] && sa="{full}"\n'


def _me_surface_combine(plan: Plan) -> str:
    """Per vertex, the same optimal combination as the volume, into the run's own
    surface names; mask and mean are echo 1's (as the volume QC mean is)."""
    from fastfuncstuff.processing.surface_projection import space_name

    if not plan.options.surface_target:
        return ""
    out = []
    for space in dict.fromkeys(space_name(m) for m in plan.options.surface_target):
        for hemi in ("lh", "rh"):
            tail = f"{space}.{hemi}"
            sopt = f"stage10s.surf.${{FRAG[$k]}}.{tail}.func.gii"
            out += [
                f'  if ! {{ [ "$skip_final" -eq 1 ] && [ -f "{sopt}" ]; }}; then',
                f'    sins=(); for e in $(seq 1 "${{NECHO[$k]}}"); do sins+=("stage10s.surf.${{FRAG[$k]}}.echo-$e.{tail}.func.gii"); done',
                f'    ffs_t2smap -input "${{sins[@]}}" -tes ${{TES[$k]}} -no_qc -prefix "stage10s.t2s.${{FRAG[$k]}}.{tail}_" -optcom "{sopt}" -device "$DEVICE" -verb 0',
                "  fi",
                "  for w in mask mean; do "
                f'ln -sf "stage10s.surf.${{FRAG[$k]}}.echo-1.{tail}.$w.shape.gii" '
                f'"stage10s.surf.${{FRAG[$k]}}.{tail}.$w.shape.gii"; done',
            ]
    return "\n".join(out) + "\n"


def _phasereg_field_on(plan: Plan) -> bool:
    """-phasereg_me field: phase regression runs on the combined echoes against one
    frequency-shift series, which the stage10 combine writes beside the run's final."""
    opt = plan.options
    return bool(opt.phasereg and opt.phasereg_me == "field" and _phase_on(plan))


def _final_me_combine(plan: Plan) -> str:
    """Optimal combination of each multi-echo run's per-echo finals into the run's own
    stage10.final name, which every later stage reads (ffs_t2smap: monoexponential
    T2*/S0 fit, T2*-weighted combination; maps kept as stage10.t2s.<frag>_*).

    The volume and surface combines resume independently: a work dir that gains
    -surface_target (or stopped between the two) must still combine its surfaces, so
    the volume's skip is a guard on its own command, not a ``continue``. Under
    -phasereg_me field the frequency-shift series rides along, for runs with phase."""
    if not _any_me(plan):
        return ""
    field = _phasereg_field_on(plan)
    lines = [
        "# Multi-echo: combine each run's warped echoes (optimal combination, T2*-weighted)",
        "# into stage10.final.<frag> -- the series every later stage reads. The per-echo",
        "# finals and the T2*/S0 maps (stage10.t2s.<frag>_*) stay beside it.",
        'for k in "${RUN_KEYS[@]}"; do',
        '  [ "${NECHO[$k]:-1}" -gt 1 ] || continue',
        '  outf="stage10.final.${FRAG[$k]}.nii$FINAL_FMT"; fa=(); fdone=1',
    ]
    if field:
        lines += [
            '  if [ -n "${PHASE[$k]:-}" ]; then',
            '    fieldf="stage10.field.${FRAG[$k]}.nii$FINAL_FMT"; [ -f "$fieldf" ] || fdone=0',
            '    phs=(); for e in $(seq 1 "${NECHO[$k]}"); do '
            'phs+=("stage10.final.${FRAG[$k]}.echo-$e.part-phase.nii$FINAL_FMT"); done',
            '    fa=(-phase "${phs[@]}" -field_hz "$fieldf")',
            "  fi",
        ]
    lines += [
        '  if ! { [ "$skip_final" -eq 1 ] && [ -f "$outf" ] && [ "$fdone" -eq 1 ]; }; then',
        '    ins=(); for e in $(seq 1 "${NECHO[$k]}"); do '
        'ins+=("stage10.final.${FRAG[$k]}.echo-$e.nii$FINAL_FMT"); done',
        '    ffs_t2smap -input "${ins[@]}" -tes ${TES[$k]} -automask -no_qc '
        '-prefix "stage10.t2s.${FRAG[$k]}_" -optcom "$outf" ${fa[@]+"${fa[@]}"} '
        '-device "$DEVICE" -verb 0',
        "  fi",
    ]
    return "\n".join(lines) + "\n" + _me_surface_combine(plan) + "done\n"


_MNI_TOKENS = ("mni_nl", "mni_lin")


def _surface_args(plan: Plan) -> str:
    """stage10's per-run surface flags (escaped for its printf), or "" without
    -surface_target. Footprint reads at equivolume depths, averaged over the ribbon."""
    opt = plan.options
    if not opt.surface_target:
        return ""
    assert opt.fs_subject is not None
    meshes = " ".join(shlex.quote(m) for m in opt.surface_target)
    base = _anat_matrix_base(opt)
    return (
        f" -surf {shlex.quote(opt.fs_subject)} -surf_mesh {meshes}"
        ' -surf_prefix \\"stage10s.surf.${FRAG[$k]}\\" -surf_master \\"$ANAT\\"'
        ' -surf_nwarp \\"${SURFCHAIN[$k]}\\" -surf_geom_prefix stage10s.surf'
        + (f' -surf_chain_base \\"{base}\\"' if base else "")
        + (" -surf_qc" if opt.surface_qc else "")
    )


def _anat_matrix_base(opt) -> str | None:
    """The grid the anat matrix was fitted on — every reader of it needs to be told.

    The matrix's base is the anat's autobox crop, while the things it gets read
    against are the raw $ANAT (the surface master) and the subject TPM (the
    uncropped anat). On an oblique anat those share a rotation, not an origin, so
    the cardinal mm frame an .aff12.1D lives in differs and the data lands mm away
    — 6.6 mm on a 5 degree anat; see [[project_oblique_frame_slip]]. Feeds
    ffs_nwarp -surf_chain_base and ffs_segment -matrix_base. None for -ref_file,
    whose transforms' base this script cannot know."""
    if opt.ref_file is not None:
        return None
    if opt.grand_reference:
        return f"{opt.grand_reference.rstrip('/')}/{_anat_box()}"
    return _anat_box()


def _stage_surface(plan: Plan, script_stem: str) -> str:
    """stage10s: the cross-run surface masks. The sampling itself rides stage10's
    nwarp line (:func:`_surface_args`), so it needs no pass of its own."""
    opt = plan.options
    if not opt.surface_target:
        return ""
    return f"""
# ============================ stage10s: surface masks =======================
# stage10 wrote every run onto {" + ".join(Path(m).name for m in opt.surface_target)}
# (stage10s.surf.<frag>.<SPACE>.?h.*.gii) in the same pass as its volume.
echo '== stage10s: surface masks =='
{_surface_masks(plan)}"""


def _surface_mask(space: str, hemi: str) -> str:
    """The cross-run statistics mask of one target and hemisphere (ffs_reml -mask)."""
    return f"stage10s.surf.allruns.{space}.{hemi}.mask.shape.gii"


def _surface_masks(plan: Plan) -> str:
    """stage10s masks: every run's cortex-and-coverage mask intersected, then clipped
    on the runs' mean signal -- stage10b's rules, on vertices (ffs_util_surfmask)."""
    from fastfuncstuff.processing.surface_projection import space_name

    opt = plan.options
    assert opt.surface_target
    out = [
        "# Masks: every run's (cortex AND full footprint coverage) intersected, then",
        "# AFNI's clip level on the runs' mean signal -- stage10b's rules, on vertices.",
        "# ffs_reml -mask on surface data takes these.",
    ]
    for space in dict.fromkeys(space_name(m) for m in opt.surface_target):
        for hemi in ("lh", "rh"):
            run = f"stage10s.surf.${{FRAG[$k]}}.{space}.{hemi}"
            target = _surface_mask(space, hemi)
            out += [
                f'if [ ! -f "{target}" ]; then',
                "  masks=(); means=()",
                '  for k in "${RUN_KEYS[@]}"; do',
                f'    masks+=("{run}.mask.shape.gii"); means+=("{run}.mean.shape.gii")',
                "  done",
                _ffs(
                    "ffs_util_surfmask",
                    [
                        '-mask "${masks[@]}"',
                        '-mean "${means[@]}"',
                        f'-prefix "{target.removesuffix(".mask.shape.gii")}"',
                    ],
                ),  # fmt: skip
                "fi",
            ]
    return "\n".join(out) + "\n"


def _numcomps_final_jobs(plan: Plan) -> str:
    """Carry each run's NORDIC component count into the final space.

    The dof a voxel lost is a property of THAT run, so each map rides THAT run's own
    chain and the same master its timeseries just took. Anything else lands the count
    on tissue that did not pay it.

    Three differences from the data's own resample, all deliberate:

    * **AVGCHAIN, not CHAIN.** NORDIC runs before motion correction on the raw run, and
      its count is pooled over every frame -- a property of the run, not of one volume.
      The two within-run motion links are time-varying, and ffs_nwarp resolves a
      time-varying transform at frame 0 for a 3-D source: the map would be placed at
      the pose of the first volume, chosen for no reason. Dropping them puts the count
      in the moco base's space, which is the run-average space it was measured in.
      (Bug of record: it rode the full CHAIN, so every dof map carried volume 0's
      rigid motion and volume 0's locomoco PE field.)

    * **linear, not wsinc5.** This is a count, not an image. A windowed-sinc kernel
      rings, and a negative or overshot dof is not a thing.
    * **no -jac.** The Jacobian modulation conserves signal density through a
      deformation, which is right for an intensity and wrong for a per-voxel count:
      compressing two voxels into one does not double the components removed there.

    Runs are appended to the same batch as the timeseries, so this costs no extra
    process. Outside a run's coverage the resample gives 0, which is what the sum in
    stage11 needs -- a voxel the run never saw is charged nothing for it.
    """
    if not _dof_adjust_on(plan.options):
        return ""
    return f"""# The dof map rides its run's chain minus the within-run motion links
# (AVGCHAIN): the count is pooled over the whole run, so it belongs in the
# run-average space, not at volume 0's pose. Linear and no -jac: it is a count,
# not an intensity. NOTE a working directory written before this
# existed has no _numcomps map, and skip_nordic=1 will not notice: delete the
# stage00 outputs for those runs, or pass -no_nordic_dof_adjust.
for k in "${{RUN_KEYS[@]}}"; do
  dofoutf="stage10.dofloss.${{FRAG[$k]}}.nii$FMT"
  if [ -z "${{AVGCHAIN[$k]:-}}" ]; then
    [ -f "$dofoutf" ] || ffs_util_resample -input "{_numcomps_native(plan)}" \\
      -master stage10.warpmaster.nii$FMT -rmode linear \\
      -prefix "$dofoutf" -device "$DEVICE"
    continue
  fi
  printf '%s\\n' "-source \\"{_numcomps_native(plan)}\\" -nwarp \\"${{AVGCHAIN[$k]}}\\" -master stage10.warpmaster.nii$FMT -chain_base \\"$MASTER\\" -dxyz \\"$FINAL_DXYZ\\" -interp linear -no_neg -prefix \\"$dofoutf\\"" >> "$nwarpbatch"
done
"""


def _sbref_final_jobs(plan: Plan) -> str:
    """Extra stage10 batch entries putting every run's SBRef in the final space.

    Same batch, same master/grid, same transforms as its BOLD — minus the
    within-run motion tokens the SBRef defines rather than needs (AVGCHAIN). No
    slice-timing argument: a 3-D image has no time axis to shift.

    These are the sharpest per-run images the pipeline can produce in output
    space, which makes them the QC of record for cross-run and cross-session
    alignment: overlay them and any residual misregistration is visible directly,
    not inferred from a motion-blurred mean.
    """
    if not plan.use_sbref:
        return ""
    nwarp_flags = " ".join(_split_flags(config.DEFAULT_OPTS["nwarp"]))
    # An empty AVGCHAIN means this SBRef needs no transform at all — the anchor run
    # of a no-anat, no-fieldmap, single-session plan, whose own space IS the final
    # space. ffs_nwarp has no identity warp, so that one is a plain regrid.
    return (
        "# SBRefs into the final space (QC; same chain minus the moco tokens).\n"
        'for k in "${RUN_KEYS[@]}"; do\n'
        '  sboutf="stage10.final.${FRAG[$k]}.src-sbref.nii$FINAL_FMT"\n'
        '  if [ -z "${AVGCHAIN[$k]:-}" ]; then\n'
        '    [ -f "$sboutf" ] || ffs_util_resample -input "${SBREF[$k]}" \\\n'
        "      -master stage10.warpmaster.nii$FMT -rmode wsinc5 \\\n"
        '      -prefix "$sboutf" -device "$DEVICE"\n'
        "    continue\n"
        "  fi\n"
        '  printf \'%s\\n\' "-source \\"${SBREF[$k]}\\" -nwarp \\"${AVGCHAIN[$k]}\\"'
        '${JAC[$k]:+ -jac \\"${JAC[$k]}\\"} '
        '-master stage10.warpmaster.nii$FMT -chain_base \\"$MASTER\\" -dxyz \\"$FINAL_DXYZ\\" '
        f'{nwarp_flags} -prefix \\"$sboutf\\"" >> "$nwarpbatch"\n'
        "done\n"
    )


def _stage_unwrap(plan: Plan) -> str:
    """Phase unwrapping (ffs_phase_unwrap) — the one thing done to phase before the
    final resample.

    Unwrapping happens as early as possible, in native space, while the phase is
    still voxelwise aligned to its own magnitude: the unwrap wants the magnitude as a
    quality weight, and every later stage resamples. After this the phase is in
    **radians**, continuous in time, and is simply carried; the warp chain is
    estimated from magnitude alone.

    ffs_phase_unwrap ROMEOs one reference (the temporal-mean complex image, checked
    against a few ROMEO'd frames) and anchors every frame to it on the GPU, following
    a voxel along time where its field moves further than half a period. It reads
    sub-brick selectors, so without NORDIC the noise volumes come off as a
    ``[0..n]`` selector on the raw BIDS pair rather than as a copy.
    """
    if not _phase_on(plan):
        return ""
    opt = plan.options
    # 3-D multi-echo: the per-echo partition shift + stretch, measured from the phase
    # echo-slope (head motion and whole-head field changes cancel across echoes),
    # for stage02 to fold into each echo's matrices.
    me_3d_args = (
        f' -partition_axis k -save_partition_shift "{_pshift()}"'
        f' -save_partition_stretch "{_pstretch()}"'
        if opt.acq_3d
        else ""
    )
    me_3d_note = ", and the per-echo partition shift + stretch for stage02" if opt.acq_3d else ""
    me_done = ""
    if _medic_on(plan):
        # the core/border mask MEDIC's SVD denoising of the field map needs; a work dir
        # unwrapped before -medic lacks it, so it re-runs
        me_3d_args += f' -save_mask "{_me_unwrap_stem()}_mask.nii$FMT"'
        me_done = ' && [ -f "${uwe}_mask.nii$FMT" ]'

    unwrap_opts = " ".join(_split_flags(config.DEFAULT_OPTS["unwrap"]))
    unwrap_opts = f" {unwrap_opts}" if unwrap_opts else ""
    uw = _unwrapped()
    drift_note = drift_vars = drift_done = drift_cmd = ""
    if opt.phase_drift:
        # Native space is the only place this can be done: once slice timing or
        # the final resample has mixed slices, the per-slice field states that
        # breathing imprints are no longer separable.
        drift_note = (
            "\n# Then ffs_phasedrift removes each slice's dynamic field offset + in-plane\n"
            "# gradient per volume (breathing, B0 drift) -> stage00.unwrapdrift.*;\n"
            "# that, not the unwrap's output, is the phase every later stage reads."
        )
        trace = "stage00.unwrapdrift.${FRAG[$k]}.trace.1D"
        drift_vars = f'\n  pd="{_unwrapped("unwrapdrift")}"'
        drift_done = ' && [ -f "$pd" ]'
        # 3-D EPI has no acquired slices: one field state per volume.
        by = "volume" if opt.acq_3d else "slice"
        drift_cmd = (
            '\n  [ -f "$pd" ] || ffs_phasedrift -input "$uw" -magnitude "$mag" '
            f'-prefix "$pd" -by {by} -save_trace "{trace}" -verb 0'
        )
    if opt.want_nordic:
        src = f'  mag="{_nordic_mag(plan)}"; ph="{_nordic_phase()}"'
        note = "# Inputs: the NORDIC-denoised pair (noise volumes already trimmed)."
    else:
        # A -cut_task_vols run arrives with its selector already on MAG/PHASE and
        # NOISE[$k]=0, so the two selectors never stack.
        src = (
            '  mag="${MAG[$k]}"; ph="${PHASE[$k]}"\n'
            '  if [ "${NOISE[$k]}" -gt 0 ]; then\n'
            '    nv=$(ffs_info -nv "$mag"); last=$((nv - ${NOISE[$k]} - 1))\n'
            '    mag="${mag}[0..$last]"; ph="${ph}[0..$last]"\n'
            "  fi"
        )
        note = (
            "# Inputs: the raw BIDS pair, noise volumes (NOISE[$k]) dropped by a [0..n]\n"
            "# selector on both."
        )
    return f"""
# ============================ stage00: phase unwrap (GPU) ===================
# -phase_proc: unwrap the phase up front, in native space, while it is still
# voxelwise aligned to its magnitude. Output is radians; nothing touches the
# phase again until the final resample (stage10) carries it along the magnitude's
# warp chain.
{note}{drift_note}
echo '== stage00: phase unwrap =='
for k in "${{RUN_KEYS[@]}}"; do
  [ -n "${{PHASE[$k]:-}}" ] || continue  # no phase for this run (e.g. spin echo)
  if [ "${{NECHO[$k]:-1}}" -gt 1 ]; then
    # Multi-echo: ONE call over every echo -- the echo ladder, the field map, the
    # removal of the head-wide drift of the phase at TE 0{me_3d_note}.
    uwe="{_me_unwrap_stem()}"
    [ "$skip_unwrap" -eq 1 ] && [ -f "${{uwe}}.echo-1.part-phase.nii$FMT" ]{me_done} && continue
    mags=(); phs=()
    for e in $(seq 1 "${{NECHO[$k]}}"); do
      esfx=".echo-$e"
{_echo_pair_source(plan)}
      mags+=("$mag"); phs+=("$ph")
    done
    ffs_phase_unwrap -phase "${{phs[@]}}" -magnitude "${{mags[@]}}" -echo_times ${{TES[$k]}} -prefix "${{uwe}}.nii$FMT" -device "$DEVICE" -verb 0{me_3d_args}{unwrap_opts}
    continue
  fi
  uw="{uw}"{drift_vars}
  [ "$skip_unwrap" -eq 1 ] && [ -f "$uw" ]{drift_done} && continue
{src}
  [ "$skip_unwrap" -eq 1 ] && [ -f "$uw" ] || ffs_phase_unwrap -phase "$ph" -magnitude "$mag" -prefix "$uw" -device "$DEVICE" -verb 0{unwrap_opts}{drift_cmd}
done
"""


def _me_unwrap_stem() -> str:
    """A multi-echo run's unwrap stem: ffs_phase_unwrap writes
    <stem>.echo-N.part-phase per echo plus the shared <stem>_fieldmap (and _mask)."""
    return "stage00.unwrap.${FRAG[$k]}"


def _pshift() -> str:
    """Per-echo partition shift table from the phase echo-slope (3-D multi-echo)."""
    return "stage00.unwrap.${FRAG[$k]}.pshift.1D"


def _pstretch() -> str:
    return "stage00.unwrap.${FRAG[$k]}.pstretch.1D"


def _echo_pair_source(plan: Plan, indent: str = "      ") -> str:
    """Bash setting ``mag``/``ph`` to echo $e's sources (esfx set by the echo loop):
    the NORDIC pair, or the raw BIDS pair with the noise volumes dropped by selector."""
    if plan.options.want_nordic:
        return (
            f'{indent}mag="stage00.nordic.${{FRAG[$k]}}${{esfx}}.nii$FMT"; '
            'ph="stage00.nordic.${FRAG[$k]}${esfx}.part-phase.nii$FMT"'
        )
    return (
        f'{indent}mag="{_ECHO_MAG}"; ph="{_ECHO_PHASE}"\n'
        f'{indent}if [ "${{NOISE[$k]}}" -gt 0 ]; then\n'
        f'{indent}  nv=$(ffs_info -nv "$mag"); last=$((nv - ${{NOISE[$k]}} - 1))\n'
        f'{indent}  mag="${{mag}}[0..$last]"; ph="${{ph}}[0..$last]"\n'
        f"{indent}fi"
    )


def _stc_first(plan: Plan) -> bool:
    return plan.options.slicetiming_method == "first"


def _echo_stc_mag() -> str:
    """Echo $e's up-front-slice-timed magnitude (esfx set by the echo loop)."""
    return "stage01.tshift.${FRAG[$k]}${esfx}.nii$FMT"


def _echo_stc_phase() -> str:
    return "stage01.tshift.${FRAG[$k]}${esfx}.part-phase.nii$FMT"


def _echo_source(plan: Plan, indent: str = "      ") -> str:
    """Bash setting ``mag`` to echo $e's series as moco and the final resample see it:
    the slice-timed echo when slice timing ran up front, else the NORDIC / raw pair."""
    if _stc_first(plan):
        return f'{indent}mag="{_echo_stc_mag()}"'
    return _echo_pair_source(plan, indent)


def _echo_phase_file(plan: Plan) -> str:
    """Echo $e's unwrapped phase as the final resample sees it (esfx set)."""
    if _stc_first(plan):
        return _echo_stc_phase()
    return _me_unwrap_stem() + ".echo-${e}.part-phase.nii$FMT"


def _mask_epi() -> str:
    """The EPI automask on the final grid: every voxel this dataset imaged."""
    return "stage10.mask_epi.nii$FMT"


def _mask_anat_hires() -> str:
    """The anat brain mask at ANAT resolution — what :func:`_mask_anat` is
    downsampled from.

    Masking happens here rather than after the resample because automask is only
    as good as the image under it: a 1 mm anat dropped to a 4 mm EPI grid is a
    partial-volume blur, and thresholding THAT loses exactly the thin cortex the
    anat was brought in to describe. Dilated by one ANAT voxel, which is a
    fraction of an EPI voxel — a millimetre of slack, not a gyrus of it."""
    return "stage10.mask_anat_hires.nii$FMT"


def _mask_anat() -> str:
    """The anat brain mask on the final grid (nearest-neighbour from the
    anat-resolution one — a binary mask has no business being interpolated)."""
    return "stage10.mask_anat.nii$FMT"


def _mask_brain() -> str:
    """EPI coverage ∩ anat brain — the tissue this dataset actually has.

    The EPI automask keeps whatever is bright in an EPI (eyes, neck fat, a rim of
    skull marrow); the anat mask has no such extras but covers a whole head the
    EPI slab never saw. The intersection is the only one of the three that is both
    brain and acquired, which is what a GLM should be fit inside."""
    return "stage10.mask_brain.nii$FMT"


def _glm_mask(plan: Plan) -> str:
    """The stage10b mask stage12 is fit inside (-glm_mask), or "" for no mask.

    The anat modes fall back to the EPI mask when this pipeline has no anat of its
    own to mask with (-grand_reference / -ref_file) — the generator warns about
    that at parse time rather than emitting a -mask naming a file nothing writes.
    """
    mode = plan.options.glm_mask
    if mode == "none":
        return ""
    if not _own_anat(plan.options):
        return _mask_epi()
    return {"epi": _mask_epi(), "anat": _mask_anat(), "brain": _mask_brain()}[mode]


def _stage_masks(plan: Plan) -> str:
    """Masks of the data as it actually came out, on the final grid.

    Deliberately stage10b and not stage10a: an automask made before the resample is
    an automask of the *anchor*, and what the GLM has to be masked against is the
    timeseries every run landed on. stage10.meanall is every run's own mean
    averaged — the whole dataset as one 3D image, which is what an automask wants.
    """
    opt = plan.options
    meanall = "stage10.meanall.nii$FMT"
    wm = "stage10.warpmaster.nii$FMT"
    anat_fov = "stage10.anat_in_epi_fov.nii.gz"
    have_anat = _own_anat(opt)

    def guarded(outfile: str, tool: str, parts: list[str]) -> str:
        return f'[ -f "{outfile}" ] || \\\n' + _ffs(tool, parts)

    out = [
        "",
        "# ============================ stage10b: masks ==============================",
        "# Masks in the FINAL space at the final voxel size, built from the data that",
        "# came out of stage10 (not from the pre-resample anchor). stage10.meanall is",
        "# the mean of every run's mean; mask_epi is its automask.",
    ]
    if have_anat:
        out.append(
            "# mask_anat is the anat brain masked at ANAT resolution and dropped onto this\n"
            "# grid nearest-neighbour (automasking a downsampled anat loses thin cortex),\n"
            "# and mask_brain is their intersection: acquired AND brain. -glm_mask picks\n"
            "# which one (if any) stage12 is fit inside; all three are written either way."
        )
    else:
        out.append(
            "# No anat of this pipeline's own (-grand_reference / -ref_file), so there is\n"
            "# no brain to intersect with and mask_epi is the only mask there can be."
        )
    out.append("echo '== stage10b: masks =='")
    # Built here rather than with ffs_nwarp -save_mean's outputs listed inline so a
    # 100-run recipe doesn't put 100 filenames on one line.
    out.append(f'if [ ! -f "{meanall}" ]; then')
    out.append("  allmeans=()")
    out.append(
        f'  for k in "${{RUN_KEYS[@]}}"; do allmeans+=("{_final_mean("${FRAG[$k]}")}"); done'
    )
    out.append(
        _ffs(
            "ffs_util_3dmath",
            ['-input "${allmeans[@]}"', "-mean", f'-prefix "{meanall}"', '-device "$DEVICE"'],
        )
    )
    out.append("fi")
    # -dilate 1, not the tool's 2: this mask gets intersected with the anat, and a
    # 2-voxel grow at EPI resolution is most of a gyrus.
    out.append(
        guarded(
            _mask_epi(),
            "ffs_util_automask",
            [f'-input "{meanall}"', f'-prefix "{_mask_epi()}"', "-dilate 1", '-device "$DEVICE"'],
        )
    )
    if have_anat:
        # Mask at anat resolution, THEN drop to the final grid: see _mask_anat_hires.
        out.append(
            guarded(
                _mask_anat_hires(),
                "ffs_util_automask",
                [
                    f'-input "{anat_fov}"',
                    f'-prefix "{_mask_anat_hires()}"',
                    "-dilate 1",
                    '-device "$DEVICE"',
                ],
            )
        )
        out.append(
            guarded(
                _mask_anat(),
                "ffs_util_resample",
                [
                    f'-input "{_mask_anat_hires()}"',
                    f'-master "{wm}"',
                    "-rmode NN",
                    f'-prefix "{_mask_anat()}"',
                    '-device "$DEVICE"',
                ],
            )
        )
        out.append(
            guarded(
                _mask_brain(),
                "ffs_util_3dmath",
                [
                    f'-input "{_mask_epi()}" "{_mask_anat()}"',
                    "-expr 'step(a)*step(b)'",
                    f'-prefix "{_mask_brain()}"',
                    '-device "$DEVICE"',
                ],
            )
        )
    glm_mask = _glm_mask(plan)
    if glm_mask:
        # One place the GLM's mask is named; FFS_GLM_MASK swaps it without
        # regenerating the script (the other three are sitting right there).
        out.append(f'GLM_MASK="${{FFS_GLM_MASK:-{glm_mask}}}"')
    else:
        out.append(
            "# -glm_mask none: stage12 is fit in every voxel. Re-generate with\n"
            "# -glm_mask brain (or epi / anat) to fit inside one of the masks above."
        )
    return "\n".join(out) + "\n" + _qc_masks(plan)


def _qc_masks(plan: Plan) -> str:
    """The three masks over the image they were made from, in one stack.

    A mask is only ever wrong in a way you can see: mask_epi reaching into the
    neck, mask_anat sitting a few voxels off the data because the anat link is
    off, mask_brain hollow where the two disagree."""
    if not _qc_on(plan):
        return ""
    items = [("stage10.meanall.nii$FMT", "meanall"), (_mask_epi(), "epi")]
    if _own_anat(plan.options):
        items += [(_mask_anat(), "anat"), (_mask_brain(), "brain")]
    return _qc_block("final-space masks", [_qc_call(_qc_stem("mask"), items)])


def _dofloss_sums(plan: Plan, tasks) -> list[str]:
    """One summed dof map per task: what the model built from those runs actually lost.

    The GLM concatenates a task's runs, so its error dof is the sum over runs -- and so
    is the dof NORDIC took out of it. Each run's map is already in the final space by
    now (stage10, through that run's OWN chain), so summing them is the whole
    correction, and a run that does not cover a voxel contributes 0 there rather than
    charging it for frames it never had.

    What this buys is the comparison. Without it a NORDIC bucket's t/F sub-bricks carry
    the dof the design implies, which the denoising has already spent -- so a denoised
    run beats an undenoised one partly on accounting. ffs_reml -adjust_dof converts each
    statistic to a z at the corrected dof, and the two become comparable.
    """
    opt = plan.options
    if not _dof_adjust_on(opt):
        return []
    out = [
        "",
        "# ---- NORDIC dof: sum each task's per-run maps (already in final space) ----",
        "# The GLM's error dof is the sum over the runs it concatenates, so the dof",
        "# NORDIC removed from it is too. -no_nordic_dof_adjust turns this off.",
    ]
    for task, prs in tasks.items():
        srcs = " ".join(f'"{_numcomps_final(_frag(pr))}"' for pr in prs)
        out.append(
            f'ffs_util_3dmath -input {srcs} -sum -prefix "{_task_dofloss(task)}" '
            '-overwrite -device "$DEVICE"'
        )
    return out


# Each distinct phase-regression setting gets its own numbered folder, chosen when the
# script runs (see _phasereg_dir_block): rerunning autoproc with other -phasereg_*
# options adds phasereg_02 instead of mixing outputs, and an identical rerun resumes in
# the folder it already made. Everything inside double quotes, so the variable expands.
PHASEREG_DIR = "$PHASEREG_DIR"
# Settings-independent inputs shared by every phasereg_NN (the aseg tissue maps).
PHASEREG_SHARED = "phasereg_tissue"


def _tissue_files(opt) -> list[str]:
    """The anat's GM/WM/CSF fractions for ffs_veinmask: -tissue, else presurfer's
    UNI classes (same scanner space as the anat it makes), else none."""
    if opt.tissue:
        return list(opt.tissue)
    if opt.presurf and _own_anat(opt):
        return [f"presurf_work/anat1_uni_c{k}.nii.gz" for k in (1, 2, 3)]
    if _aseg_tissue(opt):
        return [f"{PHASEREG_SHARED}/tissue_aseg_{c}.nii.gz" for c in ("gm", "wm", "csf")]
    return []


def _aseg_tissue(opt) -> bool:
    """The anat is FreeSurfer's own brain volume -- -suma's brain.nii.gz, or the
    same voxels staged from mri/brain.mgz by -fs_subject -- so aseg.auto sits on
    it and its labels are the tissue maps (the -suma TPM's classes)."""
    return bool(
        opt.suma_dir
        and opt.anat_path
        and not opt.anat_extra
        and Path(opt.anat_path).parent == Path(opt.suma_dir)
        and _own_anat(opt)
    )


def _aseg_tissue_block(opt) -> list[str]:
    """Binary GM/WM/CSF from aseg.auto into phasereg_tissue/; ffs_veinmask blurs them to
    the EPI voxel, which turns the hard labels into partial-volume fractions."""
    aseg = shlex.quote(str(Path(opt.suma_dir) / "aseg.auto.nii.gz"))
    sets = dict(ASEG_TPM_LABELS)
    groups = {
        "gm": ("tpm1_cortex", "tpm2_cerebelgm", "tpm5_subcortexgm"),
        "wm": ("tpm3_cortwm", "tpm4_cerebelwm"),
        "csf": ("tpm6_vents", "tpm7_outercsf"),
    }
    out = [
        "# Tissue maps from FreeSurfer's aseg (the anat is -suma's brain).",
        f"mkdir -p {PHASEREG_SHARED}",
    ]
    for cls, names in groups.items():
        sel = ",".join(sets[n] for n in names)
        f = f"{PHASEREG_SHARED}/tissue_aseg_{cls}.nii.gz"
        out.append(
            f"[ -f \"{f}\" ] || 3dcalc -overwrite -a {aseg}'<{sel}>' -expr 'step(a)' -prefix \"{f}\""
        )
    return out


def _presurf_classes_needed(opt) -> bool:
    """presurf must write its UNI classes: they are the vein mask's tissue maps."""
    return bool(opt.phasereg and opt.phasereg_veins == "auto" and not opt.tissue and opt.presurf)


def _phasereg_runs(plan: Plan) -> dict[str, list[PlanRun]]:
    """Per task, the runs phase regression corrects: those carrying phase."""
    from fastfuncstuff.autoproc.glm import runs_by_task

    out = {}
    for task, prs in runs_by_task(plan).items():
        with_phase = [pr for pr in prs if pr.bold.phase_path]
        if with_phase:
            out[task] = with_phase
    return out


def _phasereg_corrected(task: str, i: int) -> str:
    """ffs_phasereg's i-th (1-based, -magnitude order) output for ``task``."""
    return f"{PHASEREG_DIR}/task-{task}_corrected_run-{i:02d}.nii$FINAL_FMT"


def _veinmask_on(plan: Plan) -> bool:
    opt = plan.options
    return opt.phasereg and opt.phasereg_veins == "auto" and bool(_tissue_files(opt))


def _phasereg_dir_block(settings: str) -> list[str]:
    """Bash that sets $PHASEREG_DIR (see :func:`_numbered_dir_block`)."""
    return _numbered_dir_block("phasereg", "ffs_phasereg", settings)


def _numbered_dir_block(
    name: str, tool: str, settings: str, content_files: list[str] | None = None
) -> list[str]:
    """Bash that sets ``$<NAME>_DIR``: the ``<name>_NN`` whose settings.txt holds
    exactly ``settings``, else the next free number. settings.txt is written once,
    when the folder is made; history.txt gets a line every time a run uses the folder.

    ``content_files`` are read when the script runs and become part of the settings,
    so a file edited since (a design TOML) picks a new folder instead of resuming
    in one whose outputs came from the old version."""
    var = name.upper()
    out = [
        f"# Which {name}_NN: the one made with exactly these settings (resume), else a",
        "# new number. settings.txt says what was run; history.txt when.",
        f"read -r -d '' {var}_SETTINGS <<'FFS_{var}_SETTINGS' || true",
        settings.rstrip("\n"),
        f"FFS_{var}_SETTINGS",
    ]
    for f in content_files or []:
        out.append(f'{var}_SETTINGS+=$\'\\n\\n== {f} ==\\n\'"$(cat "{f}")"')
    return out + [
        f"select_{name}_dir() {{",
        "  local d n=1",
        f"  {var}_DIR=",
        f"  for d in {name}_[0-9][0-9]; do",
        '    [ -f "$d/settings.txt" ] || continue',
        f'    if [ "$(cat "$d/settings.txt")" = "${var}_SETTINGS" ]; then {var}_DIR=$d; break; fi',
        "  done",
        f'  if [ -z "${var}_DIR" ]; then',
        f'    while [ -e "$(printf \'{name}_%02d\' "$n")" ]; do n=$((n + 1)); done',
        f"    {var}_DIR=$(printf '{name}_%02d' \"$n\")",
        f'    mkdir -p "${var}_DIR"',
        f'    printf \'%s\\n\' "${var}_SETTINGS" > "${var}_DIR/settings.txt"',
        "  fi",
        f'  echo "$(date \'+%F %T\') $(command -v {tool})" >> "${var}_DIR/history.txt"',
        f'  echo "{name}: outputs in ${var}_DIR/ (settings: ${var}_DIR/settings.txt)"',
        "}",
        f'[ -n "${{{var}_DIR:-}}" ] || select_{name}_dir',
    ]


def _phasereg_settings(opt, tasks, commands: list[str]) -> str:
    """What settings.txt records: the options that shape phase regression, then the
    exact commands (paths under $PHASEREG_DIR left literal, so identical settings give
    identical text wherever the folder ends up)."""
    lines = [
        "ffs_autoproc phase regression settings",
        f"-phasereg_veins {opt.phasereg_veins}",
        f"-phasereg_opts {opt.phasereg_opts or '(none)'}",
        f"-tissue {' '.join(opt.tissue) if opt.tissue else '(none)'}",
        "tasks: " + ", ".join(f"{t} ({len(p)} runs)" for t, p in tasks.items()),
        "",
        *commands,
    ]
    return "\n".join(lines)


def _stage_phasereg(plan: Plan, bids_root: str | None = None) -> str:
    """Phase regression on the final data, into phasereg_NN/ (post-preprocessing, so
    a folder of its own rather than another stageNN prefix; one folder per setting)."""
    opt = plan.options
    if not opt.phasereg:
        return ""
    tasks = _phasereg_runs(plan)
    d = PHASEREG_DIR
    out = [
        "",
        "# ============================ phasereg: phase regression ====================",
        "# The phase-explained (macrovascular) part of each GE run's magnitude, removed",
        "# on the stage10 data (voxel-matched magnitude + phase) into phasereg_NN/. stage10",
        "# is left as it is, so the GLM below fits both and they compare like for like.",
        "echo '== phasereg =='",
    ]
    body: list[str] = []
    commands: list[str] = []
    spr: list[str] = []
    qc = []
    if _veinmask_on(plan):
        means = " ".join(f'"{_final_mean(_frag(pr))}"' for prs in tasks.values() for pr in prs)
        tissue = " ".join(shlex.quote(t) for t in _tissue_files(opt))
        if _aseg_tissue(opt) and not opt.tissue and not opt.presurf:
            out += _aseg_tissue_block(opt)
        vm_cmd = _ffs(
            "ffs_veinmask",
            [
                f"-epi {means}",
                f"-tissue {tissue}",
                f'-mask "{_mask_brain()}"',
                f'-prefix "{d}/veinmask"',
                '-device "$DEVICE"',
            ],
        )
        commands.append(vm_cmd)
        body += [
            "# Vein candidates for -spr: cortex and pial CSF darker in the EPI than the",
            "# anat's tissue fractions predict (ffs_veinmask). Only there may a voxel",
            "# borrow a neighbour's phase; elsewhere its own phase is used.",
            f'[ -f "{d}/veinmask_mask.nii$FINAL_FMT" ] || \\',
            vm_cmd,
        ]
        spr = ["-spr", f'-spr_mask "{d}/veinmask_mask.nii$FINAL_FMT"']
        first = next(iter(tasks.values()))[0]
        qc.append(
            f'qc_tcat "{d}/QC.veinmask.nii.gz" "epi expected ratio region mask" '
            f'"{_final_mean(_frag(first))}" "{d}/veinmask_expected.nii$FINAL_FMT" '
            f'"{d}/veinmask_ratio.nii$FINAL_FMT" "{d}/veinmask_region.nii$FINAL_FMT" '
            f'"{d}/veinmask_mask.nii$FINAL_FMT"'
        )
    elif opt.phasereg_veins == "auto":
        body.append("# No anat tissue maps (-tissue / -presurf): plain phase regression, no -spr.")
    extra = _split_flags(opt.phasereg_opts) if opt.phasereg_opts else []
    if spr:
        how = (
            "sPR inside the local vein mask (OLS on the borrowed phase), "
            "own phase and ODR-damped Deming elsewhere"
        )
    elif opt.phasereg_veins == "auto":
        how = "own phase only, no sPR: no anat tissue maps for a vein mask (-tissue)"
    else:
        how = "own phase only, no sPR (-phasereg_veins off)"
    body.append(
        f"echo 'phasereg: {how}; Deming slope damped by ODR (-shrink slope), phi from FFT noise, no cap'"
    )
    if extra:
        body.append(f"echo 'phasereg: extra options {shlex.quote(opt.phasereg_opts)[1:-1]}'")
    for task, prs in tasks.items():
        mags = " ".join(f'"stage10.final.{_frag(pr)}.nii$FINAL_FMT"' for pr in prs)
        phs = " ".join(f'"stage10.final.{_frag(pr)}.part-phase.nii$FINAL_FMT"' for pr in prs)
        body.append(f"# task-{task}: runs {', '.join(_frag(pr) for pr in prs)} -> run-01..")
        select: list[str] = []
        if spr:
            # Donors chosen on task-residual (spontaneous) coupling: chosen on the
            # task-included series, a neighbour wins by carrying the task and sPR then
            # removes tissue response (ds003427, 2026-10-14). Needs every run's events.
            evs = [events_for_run(pr, bids_root, opt) for pr in prs]
            if all(evs):
                select = [
                    "-spr_select_on residual -task_removal fit:2",
                    "-events " + " ".join(shlex.quote(str(e)) for e in evs),
                ]
            else:
                body.append(
                    f"echo 'phasereg task-{task}: no events for every run, so sPR donors "
                    "are chosen on the task-included series (can remove tissue response)'"
                )

        def pr_call(mags, phs, prefix, select=select, extra_flags=()):
            return _ffs(
                "ffs_phasereg",
                [
                    f"-magnitude {mags}",
                    f"-phase {phs}",
                    f'-prefix "{prefix}"',
                    "-keep_drift",
                    *extra_flags,
                    *spr,
                    *select,
                    "-verb 1",
                    *extra,
                    '-device "$DEVICE"',
                ],
            )

        if prs[0].bold.is_multiecho and _phasereg_field_on(plan):
            # The optimal combination (stage10.final, what the GLM reads) against the one
            # frequency-shift series fitted from every echo's phase (stage10.field): Hz,
            # not radians, so no temporal unwrap; one regression per voxel, not per echo.
            fields = " ".join(f'"stage10.field.{_frag(pr)}.nii$FINAL_FMT"' for pr in prs)
            body.append(f'[ -f "{_phasereg_corrected(task, len(prs))}" ] || \\')
            pr_cmd = pr_call(mags, fields, f"{d}/task-{task}", extra_flags=["-no_phase_unwrap"])
            commands.append(pr_cmd)
            body.append(pr_cmd)
        elif prs[0].bold.is_multiecho:
            # Each echo against its own phase (the vessel's phase and its magnitude
            # footprint both scale with TE), then the corrected echoes are combined
            # like the stage10 echoes were, into the names the GLM reads.
            ne = prs[0].bold.n_echoes
            tes = " ".join(f"{t:g}" for t in prs[0].bold.echo_tes)
            for e in range(1, ne + 1):
                emags = " ".join(
                    f'"stage10.final.{_frag(pr)}.echo-{e}.nii$FINAL_FMT"' for pr in prs
                )
                ephs = " ".join(
                    f'"stage10.final.{_frag(pr)}.echo-{e}.part-phase.nii$FINAL_FMT"' for pr in prs
                )
                stem_e = f"{d}/task-{task}.echo-{e}"
                body.append(f'[ -f "{stem_e}_corrected_run-{len(prs):02d}.nii$FINAL_FMT" ] || \\')
                pr_cmd = pr_call(emags, ephs, stem_e)
                commands.append(pr_cmd)
                body.append(pr_cmd)
            for i in range(1, len(prs) + 1):
                ins = " ".join(
                    f'"{d}/task-{task}.echo-{e}_corrected_run-{i:02d}.nii$FINAL_FMT"'
                    for e in range(1, ne + 1)
                )
                body.append(
                    f'[ -f "{_phasereg_corrected(task, i)}" ] || ffs_t2smap -input {ins} '
                    f'-tes {tes} -automask -no_qc -prefix "{d}/task-{task}_t2s_run-{i:02d}_" '
                    f'-optcom "{_phasereg_corrected(task, i)}" -device "$DEVICE" -verb 0'
                )
        else:
            body.append(f'[ -f "{_phasereg_corrected(task, len(prs))}" ] || \\')
            pr_cmd = pr_call(mags, phs, f"{d}/task-{task}")
            commands.append(pr_cmd)
            body.append(pr_cmd)
        # per-echo mode writes one r2 map per echo; echo 1's stands for the task
        per_echo = prs[0].bold.is_multiecho and not _phasereg_field_on(plan)
        r2 = f"{d}/task-{task}" + (".echo-1" if per_echo else "") + "_r2.nii$FINAL_FMT"
        qc.append(
            f'qc_tcat "{d}/QC.task-{task}.nii.gz" "before after r2" '
            f'"stage10.final.{_frag(prs[0])}.nii$FINAL_FMT[0]" '
            f'"{_phasereg_corrected(task, 1)}[0]" "{r2}"'
        )
    out += _phasereg_dir_block(_phasereg_settings(opt, tasks, commands))
    out += body
    if _qc_on(plan):
        out.append(_qc_block("phasereg", qc))
    return "\n".join(out) + "\n"


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
    from fastfuncstuff.autoproc.naming import glm_tag

    opt = plan.options
    tasks = runs_by_task(plan)
    gate = "1" if opt.run_glm else "0"
    out = ["", "# ============================ stage12: GLM (ffs_reml) ======================="]
    out.append(f"# One model per task. Runs when FFS_RUN_GLM=1 (default {gate} for this recipe).")
    blurs = {task: _task_blur(opt, task) for task in tasks}
    if any(blurs.values()):
        how = (
            f"{opt.glm_blur:g} mm FWHM"
            if opt.glm_blur
            else "no smoothing by default, and per task: "
        )
        per = ", ".join(
            f"task-{t} {f:g} mm" for t, f in sorted(opt.sep_glm_blur.items()) if t in tasks
        )
        out.append(
            f"# Smoothed inside the GLM (-glm_blur): {how}{'; ' + per if per and opt.glm_blur else per}."
            "\n# stage10 stays unsmoothed, so re-running this stage at another FWHM re-does"
            "\n# nothing upstream, and each FWHM's buckets carry their own blur token."
        )
    if opt.glm_blur_to or opt.glm_blur_tsnr:
        what = (
            f"to residual tSNR {opt.glm_blur_tsnr:g}"
            if opt.glm_blur_tsnr
            else f"to {opt.glm_blur_to:g} mm FWHM"
            + (", spatially varying" if opt.glm_blur_local else "")
        )
        out.append(
            f"# Smoothed inside the GLM {what} (ffs_reml -blur_to_*), measured on the"
            "\n# residuals of each task's own model inside the GLM mask. The blurfwhm map"
            "\n# beside each bucket shows how smooth each place ended up."
        )
    if opt.glm_label:
        out.append(
            f"# These buckets carry the '{opt.glm_label}' variant token, so this fit does not"
            "\n# overwrite another variant of the same task (the skip_stats guard keys on it)."
        )
    out.append(f'if [ "${{FFS_RUN_GLM:-{gate}}}" = "1" ]; then')
    out += _dofloss_sums(plan, tasks)
    for task, prs in tasks.items():
        finals = " ".join(f'"stage10.final.{_frag(pr)}.nii$FINAL_FMT"' for pr in prs)
        resolved = events_for_task(task, prs, bids_root, opt)
        # "stage12." or "stage12.blur6.noloco." — the token rides on the buckets,
        # not on the design, because -glm_blur does not change the xmat and
        # -glm_label only names a TOML you edited yourself. Per task, because the
        # blur is: two tasks smoothed differently must not share a bucket name.
        blur = blurs[task]
        tag = glm_tag(blur, opt.glm_label, smooth=_volume_smooth_tag(opt))
        stem = f"stage12.{tag}." if tag else "stage12."
        rbuck = f"{stem}stats-reml.task-{task}.nii$GLM_FMT"
        # Same guard every other stage has, and the toggle it reads was already in
        # the preamble -- it just had nothing to switch. Editing the design TOML
        # and re-running the script is the common case, so say what to flip.
        out.append(f'if [ "$skip_stats" -ne 1 ] || [ ! -f "{rbuck}" ]; then')
        common = _glm_common(plan, task, stem, blur)
        if _censoring_on(opt):
            out.append(_concat_censor(task, prs))
        if resolved:
            spec = spec_path(task, opt)
            _, skipped = nuisance_specs(task, opt)
            out.append(f"# task-{task}: model = {spec} (edit that, not this command).")
            if skipped:
                out.append(
                    f"# NOTE -glm_ortvec {' '.join(skipped)} skipped: the stage that writes "
                    "those regressors is not in this pipeline."
                )
            out.append(_ffs("ffs_reml", [f"-input {finals}", f"-spec {spec}", *common], indent=""))
            out.append(_stats_guard_close(task))
            out += _phasereg_glm(plan, task, prs, spec, tag, blur)
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
                    *(
                        f"-event_filter_{f.action} {shlex.quote(f.column)} "
                        + " ".join(shlex.quote(v) for v in f.values)
                        for f in opt.event_filters.get(task, [])
                    ),
                    *ort_parts,
                    "-polort 3",
                    *common,
                ],
                indent="",
            )
        )
        out.append(_stats_guard_close(task))
    out.append("fi")
    return "\n".join(out) + "\n"


def _glm_common(plan: Plan, task: str, stem: str, blur: float | None) -> list[str]:
    """stage12's ffs_reml flags for one task's buckets under ``stem`` (``stage12.``,
    ``stage12.blur6.``, ``$PHASEREG_DIR/``): outputs, mask, smoothing, cluster and dof."""
    opt = plan.options
    return [
        f'-Obuck "{stem}stats-ols.task-{task}.nii$GLM_FMT"',
        f'-Rbuck "{stem}stats-reml.task-{task}.nii$GLM_FMT"',
        *(['-mask "$GLM_MASK"'] if _glm_mask(plan) else []),
        *([f"-do_blur {blur:g}"] if blur else []),
        *_target_blur_flags(
            to=opt.glm_blur_to,
            tsnr=opt.glm_blur_tsnr,
            local=opt.glm_blur_local,
            max_fwhm=opt.glm_blur_max_fwhm,
            nbhd=opt.glm_blur_nbhd,
            fwhm_map=f"{stem}blurfwhm.task-{task}.nii.gz",
        ),
        # The ACF comes from the residuals this fit already has in memory,
        # so this needs no -Rerrts and no second pass over the data.
        *(["-clustsim"] if opt.clustsim else []),
        *(["-clustsim_null resid"] if opt.clustsim and opt.clustsim_null == "resid" else []),
        *([f'-adjust_dof "{_task_dofloss(task)}"'] if _dof_adjust_on(opt) else []),
        *_reml_shared(opt, task),
    ]


def _phasereg_glm(
    plan: Plan, task: str, prs: list[PlanRun], spec: str, tag: str, blur: float | None
) -> list[str]:
    """The same model fit on phasereg's corrected runs, buckets in the same phasereg_NN/."""
    corrected = _phasereg_runs(plan).get(task) if plan.options.phasereg else None
    if not corrected:
        return []
    if len(corrected) != len(prs):
        return [
            f"# task-{task}: no phasereg GLM -- only {len(corrected)} of {len(prs)} runs "
            "carry phase, and the design spec models them all."
        ]
    stem = f"{PHASEREG_DIR}/{tag}." if tag else f"{PHASEREG_DIR}/"
    finals = " ".join(f'"{_phasereg_corrected(task, i)}"' for i in range(1, len(prs) + 1))
    rbuck = f"{stem}stats-reml.task-{task}.nii$GLM_FMT"
    return [
        f"# task-{task} after phase regression: same spec, buckets in {PHASEREG_DIR}/.",
        '[ -n "${PHASEREG_DIR:-}" ] || select_phasereg_dir',
        f'if [ "$skip_stats" -ne 1 ] || [ ! -f "{rbuck}" ]; then',
        _ffs(
            "ffs_reml",
            [f"-input {finals}", f"-spec {spec}", *_glm_common(plan, task, stem, blur)],
            indent="",
        ),
        _stats_guard_close(task),
    ]


# One folder per deveining setting, chosen when the script runs, as for phasereg_NN.
HRFTDV_DIR = "$HRFTDV_DIR"


def _stage_hrftdv(plan: Plan, bids_root: str | None) -> str:
    """Temporal deveining on the final data, into hrftdv_NN/: each task's early/late
    HRF pair per cortical column (ffs_hrftdv, from the task's design TOML), then that
    model fit with them. A folder of its own, like phasereg's, because it is an
    analysis of stage10 rather than another step of it."""
    from fastfuncstuff.autoproc.glm import runs_by_task, spec_path
    from fastfuncstuff.autoproc.naming import glm_tag

    opt = plan.options
    if not opt.hrftdv:
        return ""
    assert opt.fs_subject is not None
    d = HRFTDV_DIR
    gate = "1" if opt.run_glm else "0"
    pooled = opt.hrftdv_model == "pooled_late"
    extra = _split_flags(opt.hrftdv_opts) if opt.hrftdv_opts else []
    body: list[str] = []
    commands: list[str] = []
    specs: list[str] = []
    qc: list[str] = []
    for task, prs in runs_by_task(plan).items():
        if not events_for_task(task, prs, bids_root, opt):
            body.append(f"# task-{task}: no events, so no design to devein -- skipped.")
            continue
        spec = spec_path(task, opt)
        specs.append(spec)
        finals = " ".join(f'"stage10.final.{_frag(pr)}.nii$FINAL_FMT"' for pr in prs)
        stem = f"{d}/task-{task}"
        cmd = _ffs(
            "ffs_hrftdv",
            [
                f"-input {finals}",
                f"-spec {spec}",
                f"-fs_subj {shlex.quote(opt.fs_subject)}",
                *_design_timing_flags(opt, tr_flag="-tr"),
                # Tiles (the brain off the ribbon) inside the brain this data has,
                # not an automask of a mean that is bright in the eyes and neck.
                f'-tile_mask "{_mask_brain()}"',
                *(["-pooled_late"] if pooled else []),
                f'-prefix "{stem}"',
                *extra,
                '-device "$DEVICE"',
            ],
        )
        commands.append(cmd)
        lib = f"{stem}_early" if pooled else stem
        # The last file ffs_hrftdv writes, so a fit killed part way is redone.
        last = f"{stem}_late_dsort.run-{len(prs):02d}.nii.gz" if pooled else f"{stem}_metadata.json"
        body += [
            f"# task-{task}: column pairs from {spec}, then the same model fit with them.",
            f'[ -f "{last}" ] || \\',
            cmd,
        ]
        blur = _task_blur(opt, task)
        tag = glm_tag(blur, opt.glm_label, smooth=_volume_smooth_tag(opt))
        bstem = f"{d}/{tag}." if tag else f"{d}/"
        rbuck = f"{bstem}stats-reml.task-{task}.nii$GLM_FMT"
        model = [f'-hrf_opt "{lib}"']
        if pooled:
            late = " ".join(
                f'"{stem}_late_dsort.run-{i:02d}.nii.gz"' for i in range(1, len(prs) + 1)
            )
            model += [f"-dsort {late}", "-dsort_betas yes"]
        body += [
            f'if [ "$skip_stats" -ne 1 ] || [ ! -f "{rbuck}" ]; then',
            _ffs(
                "ffs_reml",
                [
                    f"-input {finals}",
                    f"-spec {spec}",
                    *model,
                    *_glm_common(plan, task, bstem, blur),
                ],
                indent="",
            ),
            _stats_guard_close(task),
        ]
        tdv = f"{stem}_tdv.nii.gz"
        qc.append(
            f'qc_tcat "{d}/QC.task-{task}.nii.gz" "mean rel_lag lag_r2 late_lag depth" '
            f'"{_final_mean(_frag(prs[0]))}" "{tdv}[1]" "{tdv}[3]" "{tdv}[8]" "{tdv}[9]"'
        )
    if not commands:
        return "\n# hrftdv: no task has events, so there is nothing to devein.\n"
    settings = "\n".join(
        [
            "ffs_autoproc temporal deveining settings",
            f"-hrftdv_model {opt.hrftdv_model}",
            f"-hrftdv_opts {opt.hrftdv_opts or '(none)'}",
            f"-fs_subject {opt.fs_subject}",
            "",
            *commands,
        ]
    )
    out = [
        "",
        "# ============================ hrftdv: temporal deveining ====================",
        "# Veins deliver the response later than the capillaries they drain. Per",
        "# cortical column, an early and a late HRF (ffs_hrftdv) fit together, so the",
        "# late one takes the venous part; buckets in hrftdv_NN/, stage10 untouched.",
        "# The design TOMLs are part of the folder's settings: edit one and the next",
        "# run makes a new folder rather than reusing pairs fit to the old model.",
        f'if [ "${{FFS_RUN_GLM:-{gate}}}" = "1" ]; then',
        "echo '== hrftdv =='",
        *_numbered_dir_block("hrftdv", "ffs_hrftdv", settings, content_files=specs),
        *body,
    ]
    if _qc_on(plan):
        out.append(_qc_block("hrftdv", qc))
    out.append("fi")
    return "\n".join(out) + "\n"


def _design_timing_flags(opt, tr_flag: str = "-TR") -> list[str]:
    """The TR and time origin a design is sampled at, for every tool that builds one."""
    return [
        # -TR only when the user gave one: a 3D acquisition's header TR is the
        # per-partition time, not the volume TR the design is sampled at.
        *([f"{tr_flag} {opt.tr:g}"] if opt.tr is not None else []),
        # The slice timing above aligned every slice to tzero; sample the
        # model there too (explicit: the final resample's header may not
        # carry the time origin through).
        *(
            [f"-microtime_offset {opt.tzero:g}"]
            if opt.slicetiming_method != "none" and opt.tzero
            else []
        ),
    ]


def _reml_shared(opt, task: str) -> list[str]:
    """ffs_reml flags the voxel (stage12) and surface (stage12s) fits share: the
    statistics, scaling, timing, trims, censoring and -glm_opts. Space-specific
    choices -- mask, smoothing, cluster tables, the NORDIC dof map -- stay with
    each stage."""
    return [
        "-tout",
        "-fout",
        "-do_scale",
        *_design_timing_flags(opt),
        *([f"-drop_first {opt.glm_drop_first}"] if opt.glm_drop_first else []),
        *([f"-drop_last {opt.glm_drop_last}"] if opt.glm_drop_last else []),
        # Events past a deliberate -cut_task_vols end are expected, not the
        # mispaired-timing-file symptom ffs_reml otherwise stops for.
        *(["-allow_late_events"] if task in opt.cut_task_vols else []),
        *([f'-censor "{_task_censor_path(task)}"'] if _censoring_on(opt) else []),
        *(_split_flags(opt.glm_opts) if opt.glm_opts else []),
        '-device "$DEVICE"',
    ]


def _surface_glm_tag(opt) -> str:
    """``blur4`` / ``blurto6`` (and the -glm_label), or ``""``: the surface buckets'
    variant token, so fits at different smoothings coexist."""
    from fastfuncstuff.autoproc.naming import smooth_tag

    smooth = smooth_tag(
        by=opt.surface_glm_blur,
        to=opt.surface_glm_blur_to,
        tsnr=opt.surface_glm_blur_tsnr,
        local=opt.surface_glm_blur_local,
    )
    return ".".join(t for t in (smooth, (opt.glm_label or "").strip()) if t)


def _stage_surface_stats(plan: Plan, bids_root: str | None) -> str:
    """stage12s: each task's model on each surface target and hemisphere.

    The same design spec as stage12 (the model does not care where the data
    live), fit on the stage10s series inside the cross-run surface mask. Tasks
    with no spec (unresolved events) are skipped: stage12's placeholder command is
    for a human to fix, not to copy four more times.
    """
    from fastfuncstuff.autoproc.glm import runs_by_task, spec_path
    from fastfuncstuff.processing.surface_projection import space_name

    opt = plan.options
    if not opt.surface_glm:
        return ""
    gate = "1" if opt.run_glm else "0"
    tag = _surface_glm_tag(opt)
    stem = f"stage12s.{tag}." if tag else "stage12s."
    spaces = list(dict.fromkeys(space_name(m) for m in opt.surface_target or []))
    out = [
        "",
        "# ============================ stage12s: surface GLM (ffs_reml) =============",
        "# stage12's model per task, fit on every surface target and hemisphere: the",
        "# stage10s series inside stage10s's cross-run mask. Smoothing is along the",
        "# cortex (heat diffusion on the data's own midthickness, mask edge a wall);",
        "# cluster tables are SurfClustSim areas in mm^2. Gated with stage12 (FFS_RUN_GLM).",
    ]
    if _dof_adjust_on(opt):
        out.append("# NOTE: the NORDIC dof correction is a voxel map and is not applied here.")
    out.append(f'if [ "${{FFS_RUN_GLM:-{gate}}}" = "1" ]; then')
    for task, prs in runs_by_task(plan).items():
        if not events_for_task(task, prs, bids_root, opt):
            out.append(
                f"# task-{task}: no design spec (no events found) -- not fit on the surface."
            )
            continue
        if _censoring_on(opt):
            out.append(_concat_censor(task, prs))
        for space in spaces:
            for hemi in ("lh", "rh"):
                ins = " ".join(f'"stage10s.surf.{_frag(pr)}.{space}.{hemi}.func.gii"' for pr in prs)
                where = f"task-{task}.{space}.{hemi}"
                rbuck = f"{stem}stats-reml.{where}.func.gii"
                flags = [
                    f"-input {ins}",
                    f"-spec {spec_path(task, opt)}",
                    f'-Obuck "{stem}stats-ols.{where}.func.gii"',
                    f'-Rbuck "{rbuck}"',
                    f'-mask "{_surface_mask(space, hemi)}"',
                    *([f"-do_blur {opt.surface_glm_blur:g}"] if opt.surface_glm_blur else []),
                    *_target_blur_flags(
                        to=opt.surface_glm_blur_to,
                        tsnr=opt.surface_glm_blur_tsnr,
                        local=opt.surface_glm_blur_local,
                        max_fwhm=opt.surface_glm_blur_max_fwhm,
                        nbhd=opt.surface_glm_blur_nbhd,
                        fwhm_map=f"{stem}blurfwhm.{where}.shape.gii",
                    ),
                    *(
                        ["-clustsim", f'-save_acf "{stem}resid.{where}"']
                        if opt.surface_clustsim
                        else []
                    ),
                    *(
                        ["-clustsim_null resid"]
                        if opt.surface_clustsim and opt.clustsim_null == "resid"
                        else []
                    ),
                    *(
                        [f"-clustsim_niter {opt.surface_clustsim_niter}"]
                        if opt.surface_clustsim and opt.surface_clustsim_niter
                        else []
                    ),
                    *(_split_flags(opt.surface_glm_opts) if opt.surface_glm_opts else []),
                    *_reml_shared(opt, task),
                ]
                out.append(f'if [ "$skip_stats" -ne 1 ] || [ ! -f "{rbuck}" ]; then')
                out.append(_ffs("ffs_reml", flags, indent=""))
                out.append(
                    "else\n"
                    f'  echo "stage12s {where}: bucket exists -- skipping. '
                    'Set skip_stats=0 above (or delete it) to refit."\n'
                    "fi"
                )
    out.append("fi")
    return "\n".join(out) + "\n"


def _task_censor_path(task: str) -> str:
    return f"stage12.censor.task-{task}.1D"


def _concat_censor(task: str, prs) -> str:
    """Bash joining a task's per-run stage02 keep masks into the one file ffs_reml
    -censor reads, in the order the runs are given to -input (the order the
    design's runs are in). Lengths are the moco series', i.e. before -drop_first:
    ffs_reml trims the censor file the same way it trims the data."""
    srcs = " ".join(f'"stage02.moco.{_frag(pr)}.censor.1D"' for pr in prs)
    dest = _task_censor_path(task)
    return (
        f'cat {srcs} > "{dest}"\n'
        f'echo "task-{task}: censoring $(grep -c \'^0\' "{dest}") of $(wc -l < "{dest}") TRs"'
    )


def _volume_smooth_tag(opt) -> str | None:
    """stage12's smoothing token when it smooths TO a target (``blurto6local``,
    ``tsnr80``), else None: the per-task ``-glm_blur`` kernel names the bucket."""
    from fastfuncstuff.autoproc.naming import smooth_tag

    if not (opt.glm_blur_to or opt.glm_blur_tsnr):
        return None
    return smooth_tag(to=opt.glm_blur_to, tsnr=opt.glm_blur_tsnr, local=opt.glm_blur_local)


def _target_blur_flags(*, to, tsnr, local, max_fwhm, nbhd, fwhm_map: str) -> list[str]:
    """ffs_reml's blur-to flags for one fit, and the map of how smooth each place
    ended up (-save_blur_fwhm), which a target blur always writes: the per-place
    parameter a spatially varying smoothing chose is a diagnostic worth keeping."""
    if not (to or tsnr):
        return []
    flags = [f"-blur_to_tsnr {tsnr:g}"] if tsnr else [f"-blur_to_fwhm {to:g}"]
    if local and not tsnr:
        flags.append("-blur_to_local")
    if tsnr and max_fwhm:
        flags.append(f"-blur_tsnr_max_fwhm {max_fwhm:g}")
    if nbhd and (local or tsnr):
        flags.append(f"-blur_to_nbhd {nbhd:g}")
    return [*flags, f'-save_blur_fwhm "{fwhm_map}"']


def _task_blur(opt, task: str) -> float | None:
    """The GLM smoothing FWHM for one task: its own ``-glm_blur FWHM TASK`` entry,
    else the bare ``-glm_blur`` that stands for every task nothing else names."""
    return opt.sep_glm_blur.get(task, opt.glm_blur)


def _stats_guard_close(task: str) -> str:
    """Close one task's skip_stats guard, naming the way back in.

    The message is the point: a refit after editing the design TOML is the
    single most common reason to re-run this script, and without it a skipped
    stage12 looks like the GLM ran.
    """
    return (
        "else\n"
        f'  echo "stage12 task-{task}: bucket exists — skipping the GLM. '
        'Set skip_stats=0 above (or delete it) to refit."\n'
        "fi"
    )


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


def _event_shape_args(plan: Plan, task: str, bids_root: str | None) -> str:
    """``-event_cols`` / ``-event_filter_*`` for ``task``, shell-quoted, or "".

    The per-run stages (NORDIC's task QC, locomoco's -detask) read the same TSVs as
    the GLM, so they must read the same columns and rows: without these a task whose
    conditions live in a non-BIDS column was modelled from ``trial_type`` there.
    Resolved across the task's files exactly as the GLM stage does, fallback included.
    """
    from fastfuncstuff.autoproc.glm import requested_event_ignore, resolve_event_cols

    opt = plan.options
    prs = [pr for pr in plan.runs if pr.bold.task == task]
    cols, _warn = resolve_event_cols(task, events_for_task(task, prs, bids_root, opt), opt)
    q = shlex.quote
    out = [f"-event_cols {' '.join(q(c) for c in cols)}"] if cols else []
    # Only when asked for: the spec's own default drop list is not forwarded,
    # so a pipeline that never passed -event_ignore keeps fitting what it did.
    ignore = requested_event_ignore(task, opt)
    if ignore:
        out += [f"-event_ignore {' '.join(q(c) for c in ignore)}"]
    out += [
        f"-event_filter_{f.action} {q(f.column)} " + " ".join(q(v) for v in f.values)
        for f in opt.event_filters.get(task, [])
    ]
    return " ".join(out)


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
    if plan.family_plans:
        sections = [
            _family_section(fam, sub, bids_root, script_stem) for fam, sub in plan.family_plans
        ]
        shared = _shared_anat_section(plan)
        return (
            "\n".join(
                p for p in [_header(plan, out_dir, invocation), shared, *sections] if p
            ).rstrip()
            + "\n"
        )
    return _pipeline_body(plan, bids_root, script_stem, _header(plan, out_dir, invocation))


def _family_section(fam: Family, sub: Plan, bids_root: str | None, script_stem: str) -> str:
    """One family's whole pipeline, in its own folder and its own subshell.

    The subshell is what lets each family reuse every name the single-family
    script uses — the data-table arrays, MASTER, FINAL_DXYZ — without one family's
    values leaking into the next; the folder does the same for filenames. The skip
    toggles are set once in the header and inherited. ``set -e`` carries in too, so
    a failure inside a family still stops the script.
    """
    d = family_dir(fam)
    banner = (
        "\n# " + "#" * 77 + f"\n# family {fam.label}: {fam.signature.describe()}"
        f"\n#   {len(sub.runs)} run(s), task {', '.join(fam.tasks)}  →  {d}/"
        f"\n#   reference session : {sub.ref_session}"
        f"\n#   sbref lane        : {sub.use_sbref}{_fmap_inherit_note(sub)}"
        f"\n#   reaches the anat  : {_family_link_note(fam)}"
        "\n# " + "#" * 77
    )
    body = _pipeline_body(sub, bids_root, script_stem, None)
    return f'{banner}\n(\nmkdir -p "{d}" && cd "{d}"\n{_link_shared_anat(sub)}{body})'


def _family_link_note(fam: Family) -> str:
    if fam.link == "root":
        return f"ROOT — the other families align to it ({fam.link_reason})"
    if fam.link == "xfam":
        return f"via the root family, rigid EPI-EPI in anat space ({fam.link_reason})"
    return f"on its own, cross-modal to the anat ({fam.link_reason or 'no root'})"


def _pipeline_body(plan: Plan, bids_root: str | None, script_stem: str, header: str | None) -> str:
    parts = [
        header,
        _data_arrays(plan, bids_root),
        _preflight(plan, bids_root),
        _qc_helper(plan, script_stem),
        _stage_nordic(plan),
        _stage_unwrap(plan),
        _stage_tshift(plan),
        _stage_medic(plan),
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
        _stage_surface(plan, script_stem),
        _stage_masks(plan),
        _stage_phasereg(plan, bids_root),
        _stage_stats(plan, bids_root),
        _stage_hrftdv(plan, bids_root),
        _stage_surface_stats(plan, bids_root),
    ]
    return "\n".join(p for p in parts if p).rstrip() + "\n"
