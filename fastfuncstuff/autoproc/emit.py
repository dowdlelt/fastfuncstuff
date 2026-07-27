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
from fastfuncstuff.autoproc.naming import STAGE_NUMBERS, NameKey, coord, stem
from fastfuncstuff.autoproc.plan import (
    Plan,
    PlanRun,
    anchor_fmap_ids,
    effective_anat_source,
    ref_anchor,
)

_TOOLS = [
    "ffs_nordic",
    "ffs_slicetime",
    "ffs_moco",
    "ffs_locomoco",
    "ffs_blipflip",
    "ffs_allineate",
    "ffs_formwarp",
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


def _run_stem(pr: PlanRun, label: str) -> str:
    b = pr.bold
    return stem(NameKey(label, session=b.session, task=b.task, run=b.run or None))


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


def _ffs(tool: str, parts: list[str], indent: str = "  ") -> str:
    """Render an ffs_* call one flag per line with ``\\`` continuations, indented.
    ``parts`` are per-flag tokens (mix of literals and bash ``${...}``)."""
    flat = [p for p in parts if p]  # drop empties (e.g. a conditionally-absent flag)
    lines = [f"{indent}{tool}", *[f"{indent}  {p}" for p in flat]]
    return " \\\n".join(lines)


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


# The fmap-level tokens: run-native → reference-fmap undistorted space. Applied
# on their own to a run's moco-mean they build its "runmean" (see stage07).
_FMAP_TOKENS = ("xfmap_nl", "xfmap_lin", "blip_half", "wxrun_nl", "wxrun_lin")


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


def chain_files(pr: PlanRun, fmt: str, opt=None) -> list[str]:
    """Resolve a run's full warp chain into filenames (nwarp-apply order)."""
    resolved: list[str] = []
    for tok in pr.warp_chain:
        resolved.extend(_token_files(pr, tok, fmt, opt))
    return resolved


def _fmap_subchain(pr: PlanRun, fmt: str) -> list[str]:
    """Just the fmap-level tokens of a run's chain (xfmap∘blip∘wxrun) — applied to
    the moco-mean to land it on the reference-fmap grid (the runmean)."""
    out: list[str] = []
    for tok in pr.warp_chain:
        if tok in _FMAP_TOKENS:
            out.extend(_token_files(pr, tok, fmt, None))
    return out


def _ref_blip_mean(pr: PlanRun) -> str:
    """The reference fmap group's undistorted mean — the common grid the runmeans
    (and cross-fmap alignment) land on."""
    return stem(NameKey("blip", session=pr.bold.session, fmap=pr.ref_fmap_id)) + "_mean.nii$FMT"


def _runmean(pr: PlanRun) -> str:
    """This run's mean resampled onto the reference-fmap grid (sesmean input)."""
    return _run_stem(pr, "runmean") + ".nii$FMT"


def _sesmean(session: str | None) -> str:
    """One session's mean: the average of its aligned run means, in that session's
    own space (the xses source for non-reference sessions)."""
    return stem(NameKey("sesmean", session=session)) + ".nii$FMT"


def _grandmean() -> str:
    """THE grandmean: every run of every session, after cross-session alignment.
    Built in stage08 (not 07) because it cannot exist until xses has put the
    sessions in a single space."""
    return stem(NameKey("grandmean")) + ".nii$FMT"


def _ref_marker(key: NameKey, source: str, note: str) -> str:
    """A QC-only copy of the image that *is* a level's reference, named like the
    level's other outputs plus ``role-ref``. Alignment stages skip their reference
    (it maps to itself), which otherwise leaves a browsable gap — N-1 files where
    the reader expects N. Has no ``_nl`` counterpart, by definition."""
    dst = stem(key) + ".nii$FMT"
    return f'# {note}\n[ -f "{dst}" ] || cp -f "{source}" "{dst}"'


def _fmapmean(plan: Plan) -> str:
    """The ``mean_fmap`` image: mean of the anchor session's fmap means."""
    anchor = ref_anchor(plan)
    ses = anchor.bold.session if anchor else None
    return stem(NameKey("fmapmean", session=ses)) + ".nii$FMT"


def _fmapmean_inputs(plan: Plan) -> list[str]:
    """The per-group means averaged into ``mean_fmap``, all already on the
    reference-fmap grid: the reference group's own blip mean, plus each non-ref
    group's xfmap-aligned mean (nonlinear result when ``-xfmap_nonlin`` ran)."""
    anchor = ref_anchor(plan)
    if anchor is None:
        return []
    ses = anchor.bold.session
    ref_id = anchor.fmap.fmap_id if anchor.fmap else None
    nl = plan.options.xfmap_nonlin
    files = []
    for fid in anchor_fmap_ids(plan):
        if fid == ref_id:
            files.append(stem(NameKey("blip", session=ses, fmap=fid)) + "_mean.nii$FMT")
        else:
            st = stem(NameKey("xfmap", session=ses, fmap=fid))
            files.append(f"{st}_nl.nii$FMT" if nl else f"{st}.nii$FMT")
    return files


def _needs_fmapmean(plan: Plan) -> bool:
    """True when anything asks for the ``mean_fmap`` image — the anat linear step
    or (independently) ffs_segment's input — so stage05 knows to build it."""
    opt = plan.options
    if not opt.go_to_anat:
        return False  # no anat step, nothing consumes it
    requests = [opt.anat_source]
    if opt.anat_nonlin:
        requests.append(opt.anat_nonlin_input)
    return any(effective_anat_source(plan, r) == "mean_fmap" for r in requests)


def _anat_source_image(plan: Plan, requested: str | None = None) -> str:
    """The EPI-contrast image the anat step aligns (or ffs_segment consumes).

    All three choices sit on the reference-fmap grid, so the warp chain is
    identical whichever is picked — only the image content differs:
      grandmean  every run's mean averaged; best SNR, N interpolations deep, and
                 the only option without fieldmaps.
      ref_fmap   the reference group's undistorted blip mean; one interpolation,
                 sharpest, SBRef contrast — the image that *defines* the space.
      mean_fmap  ref_fmap averaged with the other groups' xfmap-aligned means;
                 ref_fmap's provenance with more SNR (multi-fieldmap only).
    """
    mode = effective_anat_source(plan, requested)
    if mode == "ref_fmap":
        anchor = ref_anchor(plan)
        if anchor is not None:
            return _ref_blip_mean(anchor)
    elif mode == "mean_fmap":
        return _fmapmean(plan)
    return _grandmean()


# ---------------------------------------------------------------------------
# per-run derived paths (computed in Python, emitted as data)
# ---------------------------------------------------------------------------


def _moco_mean(pr: PlanRun) -> str:
    """The registration-target mean for a run (locomoco mean if enabled).

    Filenames match the tools' own output naming: ffs_moco writes the file we
    pass to ``-save_mean``; ffs_locomoco writes ``{stem}_locomoco_mean``.
    """
    if pr.warp_chain and "locomoco" in pr.warp_chain:
        return _run_stem(pr, "nlmoco") + "_locomoco_mean.nii$FMT"
    return _run_stem(pr, "moco") + "_mean.nii$FMT"


def _xrun_base(pr: PlanRun, session_first: PlanRun) -> str | None:
    """Base image for this run's xrun alignment, or None if it needs no xrun.

    Fieldmap-anchored: the group's DISTORTED forward (blip_up) — blip_half
    undistorts *after* xrun in the chain, so xrun stays in distorted space.
    No-fmap: the session's first-run moco mean.
    """
    if "wxrun_lin" not in pr.warp_chain:
        return None
    if pr.fmap is not None:
        return pr.fmap_forward
    return _moco_mean(session_first)  # first-run anchor


def _aligned_mean(pr: PlanRun) -> str:
    """The run's mean in the session's common (sesmean) space.

    With fieldmaps: the runmean (moco-mean pushed through xfmap∘blip∘wxrun onto
    the reference-fmap grid). No fieldmap: the xrun-aligned mean (nonlinear result
    if xrun_nonlin ran, else linear), or the moco mean for the anchor run.
    """
    if pr.fmap is not None:
        return _runmean(pr)
    if "wxrun_nl" in pr.warp_chain:
        return _run_stem(pr, "xrun") + "_nl.nii$FMT"
    if "wxrun_lin" in pr.warp_chain:
        return _run_stem(pr, "xrun") + ".nii$FMT"
    return _moco_mean(pr)


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------


def _skip_default(opt) -> int:
    """Default value for the batched stages' skip toggle: 1 (skip complete runs,
    the resumable default) unless -batch_overwrite forces a full re-process."""
    return 0 if opt.batch_overwrite else 1


def _header(plan: Plan, out_dir: str) -> str:
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
#   phase  : {_phase_on(plan)}
# =============================================================================
set -euo pipefail

OUT={shlex.quote(out_dir)}
DEVICE={config.DEFAULT_DEVICE}
FMT={opt.fmt}           # working intermediates (read many times); -format
FINAL_FMT={opt.final_fmt}     # final timeseries (portable); -final_format
GLM_FMT={opt.glm_fmt}       # GLM stat buckets; -glm_format
NOISE_VOLS={opt.noise_vols}{phase_fmt}

mkdir -p "$OUT"; cd "$OUT"

# ---- coarse per-stage re-run toggles (1 = skip stage output if it exists) ---
# The batched moco + final stages pass their toggle to the tool as -batch_skip.
skip_nordic=1 skip_moco={_skip_default(opt)} skip_locomoco=1 skip_blip=1
skip_xfmap=1  skip_xrun=1 skip_xses=1 skip_anat=1 skip_final={_skip_default(opt)} skip_stats=1{phase_skip}
"""


def _data_arrays(plan: Plan) -> str:
    lines = [
        "",
        "# ============================ per-run data table ============================",
    ]
    keys = [_key(pr) for pr in plan.runs]
    lines.append("RUN_KEYS=(" + " ".join(shlex.quote(k) for k in keys) + ")")
    lines.append(
        "declare -A MAG PHASE SBREF TR PEDIR JSON FRAG CHAIN MOCOMEAN XRUNBASE ALIGNED "
        "PRECHAIN REFBLIP"
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
        lines.append(f"TR[{q(k)}]={q(str(b.tr if b.tr is not None else ''))}")
        lines.append(f"PEDIR[{q(k)}]={q(str(b.pe_dir or ''))}")
        # Filename coordinate fragment — the stage loops build run filenames as
        # stageNN.label.${FRAG[$k]}, matching every reference in the data table.
        lines.append(f"FRAG[{q(k)}]={q(_frag(pr))}")
        # These values contain the bash var $FMT and must expand at assignment,
        # so they are double-quoted (not shlex-quoted). The key and the values
        # are our own constructed names — no shell metacharacters beyond $FMT.
        lines.append(f'MOCOMEAN[{q(k)}]="{_moco_mean(pr)}"')
        base = _xrun_base(pr, first_by_ses[b.session])
        if base is not None:
            lines.append(f'XRUNBASE[{q(k)}]="{base}"')
        lines.append(f'ALIGNED[{q(k)}]="{_aligned_mean(pr)}"')
        if pr.fmap is not None:
            # runmean sub-chain (moco-mean → reference-fmap grid) + that grid.
            lines.append(f'PRECHAIN[{q(k)}]="{" ".join(_fmap_subchain(pr, ".nii$FMT"))}"')
            lines.append(f'REFBLIP[{q(k)}]="{_ref_blip_mean(pr)}"')
        chain = chain_files(pr, ".nii$FMT", plan.options)
        lines.append(f'CHAIN[{q(k)}]="{" ".join(chain)}"')
    return "\n".join(lines) + "\n"


def _preflight(plan: Plan) -> str:
    inputs = sorted(
        {str(pr.bold.mag_path) for pr in plan.runs}
        | {str(pr.bold.phase_path) for pr in plan.runs if pr.bold.phase_path}
        | {str(pr.fmap.reverse_path) for pr in plan.runs if pr.fmap}
    )
    checks = " \\\n".join(f"  {shlex.quote(p)}" for p in inputs)
    # romeo (MRItools) is an external dependency, only needed with -phase_proc.
    tools = [*_TOOLS, "romeo"] if _phase_on(plan) else _TOOLS
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
        f'nv=$(3dinfo -nv "$raw"); raw="${{raw}}[0..$((nv - NOISE_VOLS - 1))]"; fi'
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
            '-tpattern "${JSON[$k]}"',
            '-TR "${TR[$k]}"',
            "-tzero 0",
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
                '-tpattern "${JSON[$k]}"',
                '-TR "${TR[$k]}"',
                "-tzero 0",
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
    nordic_cmd = _ffs(
        "ffs_nordic",
        [
            '-input_magn "${MAG[$k]}"',
            '"${phase_arg[@]}"',
            '-prefix "$outf"',
            '-noise-volume-last "$NOISE_VOLS"',
            "-nordic",
            resid,
            '-device "$DEVICE"',
            "-verbose",
        ],
    )
    return f"""
# ============================ stage00: NORDIC ==============================={phase_note}
echo '== stage00: NORDIC denoise =='
for k in "${{RUN_KEYS[@]}}"; do
  outf="{_nordic_mag(plan)}"
  [ "$skip_nordic" -eq 1 ] && [ -f "$outf" ] && continue
  ph="${{PHASE[$k]:-}}"
  if [ -n "$ph" ]; then phase_arg=(-input_phase "$ph"); else phase_arg=(-magnitude-only); fi
{nordic_cmd}
done
"""


def _stage_moco(plan: Plan, script_stem: str) -> str:
    moco_flags = " ".join(_split_flags(config.DEFAULT_OPTS["moco"]))
    batchfile = f"{script_stem}_mocobatch.txt"
    return f"""
# ============================ stage02: motion correction ====================
# Batched: ONE ffs_moco process motion-corrects every run, so the Python/CUDA/
# torch.compile startup is paid once, not per run. Each run's arguments are
# appended to {batchfile} (beside the script's outputs) and consumed with -batch.
# Reads the source series in place (no full-dataset copy). Saves the target mean
# + per-volume matrices + motion params; the corrected 4D is produced once by the
# final single-resample (stage10). skip_moco=1 → -batch_skip (skip complete runs).
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
    last)  nv=$(3dinfo -nv "$raw"); base_str="-base $((nv - 1))" ;;
    *)     base_str="-base \\"$MOCO_REF\\"" ;;   # integer volume index
  esac
  printf '%s\\n' "-input \\"$raw\\" $base_str {moco_flags} -save_mean \\"${{mstem}}_mean.nii$FMT\\" -1Dmatrix_save \\"${{mstem}}.aff12.1D\\" -1Dfile \\"${{mstem}}.motion.1D\\"" >> "$mocobatch"
done
batch_skip=(); [ "$skip_moco" -eq 1 ] && batch_skip=(-batch_skip)
ffs_moco -batch "$mocobatch" "${{batch_skip[@]}}" -device "$DEVICE"
"""


def _stage_locomoco(plan: Plan) -> str:
    if not plan.options.locomoco:
        return ""
    return f"""
# ============================ stage03: locomoco (residual NL motion) ========
echo '== stage03: locomoco =='
for k in "${{RUN_KEYS[@]}}"; do
  nlstem="stage03.nlmoco.${{FRAG[$k]}}"
  [ "$skip_locomoco" -eq 1 ] && [ -f "${{nlstem}}_warp.nii$FMT" ] && continue
  pe="${{PEDIR[$k]}}"; pe="${{pe//[!a-zA-Z]/}}"
{_ffs("ffs_locomoco", ['-input "stage02.moco.${FRAG[$k]}_mean.nii$FMT"', '-prefix "${nlstem}.nii$FMT"', '-pe_dir "${pe:-y}"', *_split_flags(config.DEFAULT_OPTS["locomoco"]), "-warp_format 5d", "-save_mean", '-device "$DEVICE"'])}
done
"""


def _stage_blip(plan: Plan) -> str:
    groups: dict[tuple, PlanRun] = {}
    for pr in plan.runs:
        if pr.fmap is None:
            continue
        groups.setdefault((pr.bold.session, pr.fmap.fmap_id), pr)
    if not groups:
        return ""
    out = ["", "# ============================ stage04: fieldmap (blipflip) =================="]
    out.append("echo '== stage04: distortion correction =='")
    for pr in groups.values():
        st = _fmap_stem(pr, "blip")
        pe = "".join(c for c in (pr.fmap.pe_dir or pr.bold.pe_dir or "j") if c.isalpha()) or "j"
        ro = f"-readout {pr.fmap.readout}" if pr.fmap.readout else ""
        cmd = _ffs(
            "ffs_blipflip",
            [
                f"-blip_up {shlex.quote(str(pr.bold.rep))}",
                f"-blip_down {shlex.quote(str(pr.fmap.reverse_path))}",
                f"-pe_dir {pe}",
                ro,
                *_split_flags(config.DEFAULT_OPTS["blip"]),
                f'-prefix "{st}.nii$FMT"',
                '-device "$DEVICE"',
            ],
        )
        out.append(f'if [ "$skip_blip" -ne 1 ] || [ ! -f "{st}_warp.nii$FMT" ]; then\n{cmd}\nfi')
    return "\n".join(out) + "\n"


def _stage_xfmap(plan: Plan) -> str:
    """Align each NON-reference fmap group's undistorted mean to the session's
    reference fmap mean (once per group). This is what lets runs acquired under
    different fieldmaps share one session space; the per-run runmean composes it
    with blip+xrun (see stage07)."""
    opt = plan.options
    groups: dict[tuple, PlanRun] = {}
    for pr in plan.runs:
        if pr.fmap is None or "xfmap_lin" not in pr.warp_chain:
            continue  # reference fmap (or no fmap) → no cross-fmap alignment
        groups.setdefault((pr.bold.session, pr.fmap.fmap_id), pr)
    want_fmapmean = _needs_fmapmean(plan)
    if not groups and not want_fmapmean:
        return ""
    out = ["", "# ============================ stage05: cross-fmap alignment ================="]
    out.append("echo '== stage05: cross-fmap alignment (→ reference fmap) =='")
    # The reference fmap group has no xfmap transform (it maps to itself) — emit its
    # undistorted mean under the xfmap name so every group shows up in one listing.
    anchor = ref_anchor(plan)
    if anchor is not None and anchor.fmap is not None:
        out.append(
            _ref_marker(
                NameKey("xfmap", session=anchor.bold.session, fmap=anchor.fmap.fmap_id, role="ref"),
                _ref_blip_mean(anchor),
                f"QC: fmap-{anchor.fmap.fmap_id} is the reference group — no transform.",
            )
        )
    for pr in groups.values():
        xstem = _fmap_stem(pr, "xfmap")
        src = _fmap_stem(pr, "blip") + "_mean.nii$FMT"  # this fmap, undistorted
        base = _ref_blip_mean(pr)  # reference fmap, undistorted
        lin = _ffs(
            "ffs_allineate",
            [
                f'-base "{base}"',
                f'-source "{src}"',
                f'-prefix "{xstem}.nii$FMT"',
                f'-1Dmatrix_save "{xstem}.aff12.1D"',
                *_split_flags(config.DEFAULT_OPTS["xfmap"]),
                '-device "$DEVICE"',
            ],
        )
        out.append(f'if [ "$skip_xfmap" -ne 1 ] || [ ! -f "{xstem}.aff12.1D" ]; then\n{lin}')
        if opt.xfmap_nonlin:
            out.append(
                _ffs(
                    "ffs_formwarp",
                    [
                        f'-base "{base}"',
                        f'-source "{xstem}.nii$FMT"',
                        f'-prefix "{xstem}_nl.nii$FMT"',
                        "-save_warp",
                        *_split_flags(config.DEFAULT_OPTS["xfmap_nl"]),
                        '-device "$DEVICE"',
                    ],
                )
            )
        out.append("fi")
    if want_fmapmean:
        # -anat_source mean_fmap: every group's mean is now on the reference-fmap
        # grid, so averaging them is a straight voxelwise mean (more SNR than any
        # single group, without the grandmean's extra interpolation).
        fm = _fmapmean(plan)
        inputs = " ".join(f'"{f}"' for f in _fmapmean_inputs(plan))
        out.append("echo '== stage05: mean of fieldmap means (-anat_source mean_fmap) =='")
        out.append(
            f'[ -f "{fm}" ] || \\\n'
            + _ffs(
                "ffs_util_3dmath",
                [f"-input {inputs}", "-mean", f'-prefix "{fm}"', '-device "$DEVICE"'],
            )
        )
    return "\n".join(out) + "\n"


def _stage_xrun(plan: Plan) -> str:
    opt = plan.options
    nl = ""
    if opt.xrun_nonlin:
        # Residual nonlinear refinement of the linear-aligned mean → distinct
        # `_nl` output (never overwrite the linear source); this feeds grandmean.
        fw = _ffs(
            "ffs_formwarp",
            [
                '-base "$base"',
                '-source "${xstem}.nii$FMT"',
                '-prefix "${xstem}_nl.nii$FMT"',
                "-save_warp",
                *_split_flags(config.DEFAULT_OPTS["xrun_nl"]),
                '-device "$DEVICE"',
            ],
            indent="    ",
        )
        nl = f'  if [ ! -f "${{xstem}}_nl_WARP.nii$FMT" ]; then\n{fw}\n  fi\n'
    lin = _ffs(
        "ffs_allineate",
        [
            '-base "$base"',
            '-source "${MOCOMEAN[$k]}"',
            '-prefix "${xstem}.nii$FMT"',
            '-1Dmatrix_save "${xstem}.aff12.1D"',
            *_split_flags(config.DEFAULT_OPTS["xrun"]),
            '-device "$DEVICE"',
        ],
        indent="    ",
    )
    # No-fmap mode: the session's first run IS the anchor, so it gets no xrun output.
    # Emit its moco mean under the xrun name so the listing covers every run.
    markers = [
        _ref_marker(
            NameKey(
                "xrun",
                session=pr.bold.session,
                task=pr.bold.task,
                run=pr.bold.run or None,
                role="ref",
            ),
            _moco_mean(pr),
            f"QC: {_frag(pr)} is the cross-run anchor — no transform.",
        )
        for pr in plan.runs
        if "wxrun_lin" not in pr.warp_chain
    ]
    marker_block = ("\n".join(markers) + "\n") if markers else ""
    return f"""
# ============================ stage06: cross-run alignment ==================
# Align each run's mean to its anchor (fmap group mean, or the session's first
# run when there are no fmaps). Saves a matrix that composes into the chain.
echo '== stage06: cross-run alignment =='
{marker_block}for k in "${{RUN_KEYS[@]}}"; do
  base="${{XRUNBASE[$k]:-}}"
  [ -z "$base" ] && continue   # this run is the anchor (identity) — no xrun
  xstem="stage06.xrun.${{FRAG[$k]}}"
  if [ "$skip_xrun" -ne 1 ] || [ ! -f "${{xstem}}.aff12.1D" ]; then
{lin}
  fi
{nl}done
"""


def _stage_grandmean(plan: Plan) -> str:
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
    # Fieldmap runs first get a "runmean": their moco-mean pushed through
    # xfmap∘blip∘xrun onto the reference-fmap grid, so runs from different fmap
    # groups average in one common space. (No-fmap runs skip this — PRECHAIN unset.)
    if any(pr.fmap is not None for pr in plan.runs):
        nwarp_opts = config.DEFAULT_OPTS["nwarp"]
        runmean_cmd = _ffs(
            "ffs_nwarp",
            [
                '-source "${MOCOMEAN[$k]}"',
                '-nwarp "${PRECHAIN[$k]}"',
                '-master "${REFBLIP[$k]}"',
                *_split_flags(nwarp_opts),
                '-prefix "stage07.runmean.${FRAG[$k]}.nii$FMT"',
                '-device "$DEVICE"',
            ],
        )
        out.append("echo '== stage07: run means (→ reference-fmap grid) =='")
        out.append(
            'for k in "${RUN_KEYS[@]}"; do\n'
            '  [ -z "${PRECHAIN[$k]:-}" ] && continue   # no-fmap run, no runmean\n'
            '  pm="stage07.runmean.${FRAG[$k]}.nii$FMT"\n'
            '  [ -f "$pm" ] && continue\n'
            f"{runmean_cmd}\n"
            "done"
        )
    out.append("echo '== stage07: session means =='")
    for ses, prs in by_ses.items():
        aligned = " ".join(f'"{_aligned_mean(pr)}"' for pr in prs)
        out.append(
            _ffs(
                "ffs_util_3dmath",
                [
                    f"-input {aligned}",
                    "-mean",
                    f'-prefix "{_sesmean(ses)}"',
                    "-overwrite",
                    '-device "$DEVICE"',
                ],
                indent="",
            )
        )
    return "\n".join(out) + "\n"


def _xses_aligned(session: str, nonlin: bool) -> str:
    """The cross-session-aligned session mean for a non-ref session (nl if the
    nonlinear step ran, else the linear result)."""
    stem_ = stem(NameKey("xses", session=session))
    return f"{stem_}_nl.nii$FMT" if nonlin else f"{stem_}.nii$FMT"


def _stage_xses(plan: Plan) -> str:
    """stage08: bring every session into the reference session's space, then build
    THE grandmean from the result.

    Always emitted, including single-session — the grandmean has exactly one
    producer that way, and it is always the post-alignment one. With one session
    the alignment loop is empty and the grandmean is just that session's mean.
    """
    opt = plan.options
    sessions = []
    seen = set()
    for pr in plan.runs:
        s = pr.bold.session
        if s not in seen and s is not None:
            seen.add(s)
            sessions.append(s)
    ref = plan.ref_session
    out = ["", "# ============================ stage08: cross-session align + grandmean ======"]
    if plan.multi_session:
        out.append("echo '== stage08: cross-session alignment =='")
    out.append(f'REFGM="{_sesmean(ref)}"')
    aligned_means = [_sesmean(ref)]  # ref session is already in place
    if plan.multi_session:
        # The reference session has no xses transform (it maps to itself) — emit its
        # mean under the xses name so `ls stage08.xses.*` shows every session.
        out.append(
            _ref_marker(
                NameKey("xses", session=ref, role="ref"),
                _sesmean(ref),
                f"QC: ses-{ref} is the reference session — no transform, shown for completeness.",
            )
        )
    for s in sessions:
        if s == ref:
            continue
        xstem = stem(NameKey("xses", session=s))
        src = _sesmean(s)
        lin = _ffs(
            "ffs_allineate",
            [
                '-base "$REFGM"',
                f'-source "{src}"',
                f'-prefix "{xstem}.nii$FMT"',
                f'-1Dmatrix_save "{xstem}.aff12.1D"',
                *_split_flags(config.DEFAULT_OPTS["xses"]),
                '-device "$DEVICE"',
            ],
        )
        out.append(f'if [ "$skip_xses" -ne 1 ] || [ ! -f "{xstem}.aff12.1D" ]; then\n{lin}')
        if opt.xses_nonlin:
            # Distinct `_nl` output — don't overwrite the linear-aligned source.
            out.append(
                _ffs(
                    "ffs_formwarp",
                    [
                        '-base "$REFGM"',
                        f'-source "{xstem}.nii$FMT"',
                        f'-prefix "{xstem}_nl.nii$FMT"',
                        "-save_warp",
                        *_split_flags(config.DEFAULT_OPTS["xses_nl"]),
                        '-device "$DEVICE"',
                    ],
                )
            )
        out.append("fi")
        aligned_means.append(_xses_aligned(s, opt.xses_nonlin))

    # THE grandmean = mean of the reference session's mean + every cross-session-
    # aligned non-ref session mean (all now in reference-session space). This is the
    # default target the anat alignment (stage09) uses.
    out.append("echo '== stage08: grandmean (all sessions, post-alignment) =='")
    allm = " ".join(f'"{m}"' for m in aligned_means)
    out.append(
        _ffs(
            "ffs_util_3dmath",
            [
                f"-input {allm}",
                "-mean",
                '-prefix "stage08.grandmean.nii$FMT"',
                "-overwrite",
                '-device "$DEVICE"',
            ],
            indent="",
        )
    )
    return "\n".join(out) + "\n"


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
                "ffs_formwarp",
                [
                    '-base "$REFGM_EXT"',
                    '-source "stage09.xref.nii$FMT"',
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
            '-prefix "stage09.xref.nii$FMT"',
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
        lin = _ffs(
            "ffs_allineate",
            [
                '-base "$ANAT"',
                f'-source "{src}"',
                '-prefix "stage09.anat.nii$FMT"',
                '-1Dmatrix_save "stage09.anat.aff12.1D"',
                *_split_flags(config.DEFAULT_OPTS["anat"]),
                '-device "$DEVICE"',
            ],
        )
        note = ""
        if mode != opt.anat_source:
            note = f"  # (-anat_source {opt.anat_source} unavailable here → {mode})\n"
        out.append(
            f'ANAT="{anat_ph}"\n'
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
        # The FS-derived TPM has 8 hard-edge classes → these ngaus/cleanup params.
        seg_tune = (
            ["-ngaus 1 1 1 1 1 2 3 4", "-cleanup 0", "-samp 1.5"]
            if opt.fs_tpm
            else ["-niter 2000", "-samp 1.5"]
        )
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
    return "\n".join(out) + "\n"


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
    ref_fmap /
    mean_fmap  → the same anchor images ``-anat_source`` selects (shared vocabulary).
    blipfor    → the first fmap group's undistorted forward frame.
    blip_pair  → forward ``[0]`` + reverse ``[1]`` of the blipflip unwarped pair
                 (topup-style; matches the reference `primary` script).
    Falls back to grandmean (with a note) when no fmap exists.
    """
    mode = plan.options.anat_nonlin_input
    blip = None
    for pr in plan.runs:
        if pr.fmap is not None:
            blip = _fmap_stem(pr, "blip")
            break
    if mode in ("blipfor", "blip_pair") and blip is None:
        return f'-input "{_grandmean()}"  # (no fmap; fell back from {mode})'
    if mode == "blipfor":
        return f'-input "{blip}_unwarped.nii$FMT[0]"'
    if mode == "blip_pair":
        return f'-input "{blip}_unwarped.nii$FMT[0]" -pe_reverse "{blip}_unwarped.nii$FMT[1]"'
    return f'-input "{_anat_source_image(plan, mode)}"'


def _own_anat(opt) -> bool:
    """True when this pipeline computes its OWN anat matrix (an ``$ANAT`` bash var
    is defined in stage09) — as opposed to borrowing (-grand_reference) or
    overriding (-ref_file) it. Gates the viewing-only autobox3_brain."""
    return opt.go_to_anat and opt.ref_file is None and opt.grand_reference is None


def _final_master(plan: Plan) -> str:
    """The dataset whose grid (space/FOV) the final output inherits — before the
    warpmaster autoboxes it to the brain and drops it to the EPI voxel size."""
    opt = plan.options
    if not opt.go_to_anat:
        return _grandmean()
    if opt.ref_file is not None:
        # explicit reference: the copied-in ref anat defines the shared grid.
        return "stage09.ref_anat.nii.gz" if opt.ref_anat else _grandmean()
    if opt.grand_reference:
        # borrow the reference's anat-space grid so all subjects/tasks co-register.
        return f"{opt.grand_reference.rstrip('/')}/stage09.anat.nii$FMT"
    return "stage09.anat.nii$FMT"


def _final_dxyz_default(plan: Plan) -> str:
    """Final output voxel size: -final_dxyz if given, else the input EPI resolution
    read from the first run's in-plane dim."""
    opt = plan.options
    if opt.final_dxyz:
        return opt.final_dxyz
    first_mag = shlex.quote(str(plan.runs[0].bold.mag_path)) if plan.runs else '"$raw"'
    return f"$(3dinfo -ad3 {first_mag} | awk '{{print $1}}')"


def _stage_warpmaster(plan: Plan) -> str:
    """Pin the final output grid and analysis mask BEFORE the resample (stage10).

    The warpmaster is the anat-space target autoboxed to a tight brain FOV and
    resampled to the EPI voxel size — it fixes the exact position + spacing every
    run's timeseries lands on (stage10 ``-master``). epi_mask is a dilated automask
    on that grid, used to mask the GLM. autobox3_brain is a tight skull-stripped
    anat for nicer result overlays (viewing only; not used downstream). MASTER and
    FINAL_DXYZ are defined here once and reused by stage10 (same shell); the
    FFS_MASTER / FFS_FINAL_DXYZ overrides are still honoured."""
    opt = plan.options
    box = "stage10.warpmaster_box.nii$FMT"
    wm = "stage10.warpmaster.nii$FMT"
    mask = "epi_mask.nii$FMT"

    def guarded(outfile: str, tool: str, parts: list[str]) -> str:
        return f'[ -f "{outfile}" ] || \\\n' + _ffs(tool, parts)

    out = [
        "",
        "# ============================ stage10a: warpmaster + mask ==================",
        "# Fix the final output grid and analysis mask before the resample. The",
        "# warpmaster is the anat-space target autoboxed to a tight brain FOV and",
        "# resampled to the EPI voxel size; stage10 lands every run on it (-master).",
        "# epi_mask is a dilated automask on that grid (masks the GLM). autobox3_brain",
        "# is a tight skull-stripped anat for nicer result overlays (viewing only).",
        "echo '== stage10a: warpmaster + mask =='",
        # Defined once here and reused by stage10; FFS_* overrides still win.
        f'MASTER="${{FFS_MASTER:-{_final_master(plan)}}}"',
        f'FINAL_DXYZ="${{FFS_FINAL_DXYZ:-{_final_dxyz_default(plan)}}}"',
    ]
    if _own_anat(opt):
        out.append(
            guarded(
                "autobox3_brain.nii.gz",
                "ffs_util_autobox",
                [
                    '-input "$ANAT"',
                    "-npad 3",
                    '-prefix "autobox3_brain.nii.gz"',
                    '-device "$DEVICE"',
                ],
            )
        )
    out.append(
        guarded(
            box,
            "ffs_util_autobox",
            ['-input "$MASTER"', "-npad 5", f'-prefix "{box}"', '-device "$DEVICE"'],
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
            '  tp="${JSON[$k]}"; tr="${TR[$k]}"\n'
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
    return f"""
# ============================ stage10: compose + resample ===================
# Batched: ONE ffs_nwarp process resamples every run (startup paid once). Each
# run's arguments are appended to {batchfile} and consumed with -batch. One
# interpolation per run applies the whole CHAIN in a single pass; the source is
# read in place (NORDIC output, or BIDS magnitude with noise vols trimmed inline)
# so no raw copy is materialised. The chain lands every run on the stage10a
# warpmaster grid (MASTER/FINAL_DXYZ set there). skip_final=1 → -batch_skip.
{phase_note}echo '== stage10: final compose + resample =='
nwarpbatch="{batchfile}"
: > "$nwarpbatch"
for k in "${{RUN_KEYS[@]}}"; do
  outf="stage10.final.${{FRAG[$k]}}.nii$FINAL_FMT"
{phase_out}{st}
{_raw_source(plan)}
  printf '%s\\n' "-source \\"$raw\\" -nwarp \\"${{CHAIN[$k]}}\\" -master stage10.warpmaster.nii$FMT -dxyz \\"$FINAL_DXYZ\\" {nwarp_flags} $st_str -prefix \\"$outf\\"{phase_args}" >> "$nwarpbatch"
done
batch_skip=(); [ "$skip_final" -eq 1 ] && batch_skip=(-batch_skip)
ffs_nwarp -batch "$nwarpbatch" "${{batch_skip[@]}}" -device "$DEVICE"
echo 'done → stage10.final.*'
"""


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
            f'    nv=$(3dinfo -nv "$mag"); last=$((nv - NOISE_VOLS - 1))\n'
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
    # One GLM per task, over that task's final runs, broadcasting BIDS events.
    tasks: dict[str, list[PlanRun]] = {}
    for pr in plan.runs:
        tasks.setdefault(pr.bold.task, []).append(pr)
    opt = plan.options
    gate = "1" if opt.run_glm else "0"
    out = ["", "# ============================ stage12: GLM (ffs_reml) ======================="]
    out.append(f"# One model per task. Runs when FFS_RUN_GLM=1 (default {gate} for this recipe).")
    out.append("# Edit -events / -polort as needed.")
    out.append(f'if [ "${{FFS_RUN_GLM:-{gate}}}" = "1" ]; then')
    for task, prs in tasks.items():
        finals = " ".join(f'"stage10.final.{_frag(pr)}.nii$FINAL_FMT"' for pr in prs)
        events = _events_args(task, prs, bids_root, opt)
        ort_parts = []
        if opt.glm_ortvec:
            # motion + locomoco warp-PCs as nuisance; run index inferred from the
            # .run-NN. token in each filename.
            ort_parts.append(f"-ortvec_glob 'stage02.moco.*task-{task}*.motion.1D' motion")
            if opt.locomoco:
                ort_parts.append(
                    f"-ortvec_glob 'stage03.nlmoco.*task-{task}*_locomoco_pcs.1D' locomoco"
                )
        cmd = _ffs(
            "ffs_reml",
            [
                f"-input {finals}",
                f"-events {events}",
                *ort_parts,
                "-polort 3",
                f'-Obuck "stage12.stats-ols.task-{task}.nii$GLM_FMT"',
                f'-Rbuck "stage12.stats-reml.task-{task}.nii$GLM_FMT"',
                "-tout",
                "-fout",
                "-mask epi_mask.nii$FMT",
                "-do_scale",
                '-device "$DEVICE"',
            ],
            indent="",
        )
        out.append(cmd)
    out.append("fi")
    return "\n".join(out) + "\n"


def _events_args(task: str, prs: list[PlanRun], bids_root: str | None, opt=None) -> str:
    """Resolve events file(s) for a task. Explicit ``-events`` wins; else per-run
    BIDS siblings; else the dataset-root ``task-<T>_events.tsv``; else a
    placeholder glob (the user edits)."""
    if opt is not None and opt.events:
        return " ".join(shlex.quote(p) for p in opt.events)
    found: list[str] = []
    for pr in prs:
        sib = _sidecar(pr.bold.mag_path).with_name(
            re.sub(r"_bold$", "_events", _sidecar(pr.bold.mag_path).stem) + ".tsv"
        )
        if sib.is_file():
            found.append(str(sib))
    if not found and bids_root:
        root_ev = Path(bids_root) / f"task-{task}_events.tsv"
        if root_ev.is_file():
            found.append(str(root_ev))
    if found:
        return " ".join(shlex.quote(p) for p in found)
    return shlex.quote(f"{bids_root or '.'}/**/*task-{task}*_events.tsv")


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------


def write_script(
    plan: Plan,
    out_dir: str,
    bids_root: str | None = None,
    script_stem: str = "proc",
) -> str:
    """Assemble the full pipeline script text for ``plan``.

    ``script_stem`` is the basename (no extension) of the script being written;
    the batched moco / final stages name their manifest files after it
    (``{script_stem}_mocobatch.txt`` / ``_nwarpbatch.txt``) so they sit beside
    the script's outputs and don't collide across sibling subjects."""
    parts = [
        _header(plan, out_dir),
        _data_arrays(plan),
        _preflight(plan),
        _stage_nordic(plan),
        _stage_unwrap(plan),
        _stage_tshift(plan),
        _stage_moco(plan, script_stem),
        _stage_locomoco(plan),
        _stage_blip(plan),
        _stage_xfmap(plan),
        _stage_xrun(plan),
        _stage_grandmean(plan),
        _stage_xses(plan),
        _stage_xref(plan),
        _stage_anat(plan),
        _stage_warpmaster(plan),
        _stage_final(plan, script_stem),
        _stage_stats(plan, bids_root),
    ]
    return "\n".join(p for p in parts if p).rstrip() + "\n"
