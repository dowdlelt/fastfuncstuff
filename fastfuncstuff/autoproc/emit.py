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
from fastfuncstuff.autoproc.naming import NameKey, coord, stem
from fastfuncstuff.autoproc.plan import Plan, PlanRun

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


def chain_files(pr: PlanRun, fmt: str, opt=None) -> list[str]:
    """Resolve a run's warp-chain tokens into concrete filenames (OUT-relative),
    preserving nwarp-apply order.

    The ``anat_lin`` token expands per reference mode (see ``_anat_lin_files``);
    the ``xref_*`` links map this data's grandmean onto the reference's.
    """
    resolved: list[str] = []
    for tok in pr.warp_chain:
        if tok == "anat_lin":
            resolved.extend(_anat_lin_files(opt))
        elif tok == "xref_nl":
            resolved.append(stem(NameKey("xref")) + f"_nl_WARP{fmt}")
        elif tok == "xref_lin":
            resolved.append(stem(NameKey("xref")) + ".aff12.1D")
        elif tok == "anat_nl":
            resolved.append(stem(NameKey("nlanat")) + f"_invwarp{fmt}")
        elif tok == "xses_nl":
            resolved.append(_ses_stem(pr, "xses") + f"_nl_WARP{fmt}")
        elif tok == "xses_lin":
            resolved.append(_ses_stem(pr, "xses") + ".aff12.1D")
        elif tok == "xfmap_nl":
            resolved.append(_fmap_stem(pr, "xfmap") + f"_nl_WARP{fmt}")
        elif tok == "xfmap_lin":
            resolved.append(_fmap_stem(pr, "xfmap") + ".aff12.1D")
        elif tok == "blip_half":
            resolved.append(_fmap_stem(pr, "blip") + f"_warp{fmt}")
        elif tok == "wxrun_nl":
            resolved.append(_run_stem(pr, "xrun") + f"_nl_WARP{fmt}")
        elif tok == "wxrun_lin":
            resolved.append(_run_stem(pr, "xrun") + ".aff12.1D")
        elif tok == "locomoco":
            resolved.append(_run_stem(pr, "nlmoco") + f"_warp{fmt}")
        elif tok == "moco":
            resolved.append(_run_stem(pr, "moco") + ".aff12.1D")
    return resolved


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
    """Base image for this run's xrun alignment, or None if it needs no xrun."""
    if "wxrun_lin" not in pr.warp_chain:
        return None
    if pr.fmap is not None:
        return _fmap_stem(pr, "blip") + "_mean.nii$FMT"  # undistorted fmap mean
    return _moco_mean(session_first)  # first-run anchor


def _aligned_mean(pr: PlanRun) -> str:
    """The run's mean *after* xrun (feeds the session grandmean): the nonlinear
    result when xrun_nonlin ran, else the linear one, else the moco mean."""
    if "wxrun_nl" in pr.warp_chain:
        return _run_stem(pr, "xrun") + "_nl.nii$FMT"
    if "wxrun_lin" in pr.warp_chain:
        return _run_stem(pr, "xrun") + ".nii$FMT"
    return _moco_mean(pr)


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------


def _header(plan: Plan, out_dir: str) -> str:
    opt = plan.options
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
# =============================================================================
set -euo pipefail

OUT={shlex.quote(out_dir)}
DEVICE={config.DEFAULT_DEVICE}
FMT={config.DEFAULT_FMT}           # working intermediates (read many times)
FINAL_FMT={config.DEFAULT_FINAL_FMT}     # final timeseries (portable)
GLM_FMT={config.DEFAULT_GLM_FMT}
NOISE_VOLS={opt.noise_vols}

mkdir -p "$OUT"; cd "$OUT"

# ---- coarse per-stage re-run toggles (1 = skip stage output if it exists) ---
skip_nordic=1 skip_moco=1 skip_locomoco=1 skip_blip=1
skip_xfmap=1  skip_xrun=1 skip_xses=1 skip_anat=1 skip_final=1 skip_stats=1
"""


def _data_arrays(plan: Plan) -> str:
    lines = [
        "",
        "# ============================ per-run data table ============================",
    ]
    keys = [_key(pr) for pr in plan.runs]
    lines.append("RUN_KEYS=(" + " ".join(shlex.quote(k) for k in keys) + ")")
    lines.append("declare -A MAG PHASE SBREF TR PEDIR JSON FRAG CHAIN MOCOMEAN XRUNBASE ALIGNED")

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
    return f"""
# =============================== stage: preflight ===========================
echo '== preflight: inputs + tools =='
_missing=0
for f in \\
{checks}
do [ -f "$f" ] || {{ echo "MISSING INPUT: $f"; _missing=1; }}; done
for t in {" ".join(_TOOLS)}; do
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
        return f'{indent}raw="stage00.nordic.${{FRAG[$k]}}.nii$FMT"'
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
    return f"""
# ============================ stage01: slice timing (first) =================
# STC applied up front (before moco); the final resample does NOT re-integrate
# timing. Reads the NORDIC/BIDS source in place; writes the tshifted series.
echo '== stage01: slice timing =='
for k in "${{RUN_KEYS[@]}}"; do
  outf="stage01.tshift.${{FRAG[$k]}}.nii$FMT"
  [ -f "$outf" ] && continue
  [ -z "${{TR[$k]}}" ] && {{ echo "no TR for $k; cannot slice-time"; exit 1; }}
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
# ============================ stage00: NORDIC ===============================
echo '== stage00: NORDIC denoise =='
for k in "${{RUN_KEYS[@]}}"; do
  outf="stage00.nordic.${{FRAG[$k]}}.nii$FMT"
  [ "$skip_nordic" -eq 1 ] && [ -f "$outf" ] && continue
  ph="${{PHASE[$k]:-}}"
  if [ -n "$ph" ]; then phase_arg=(-input_phase "$ph"); else phase_arg=(-magnitude-only); fi
{nordic_cmd}
done
"""


def _stage_moco(plan: Plan) -> str:
    return f"""
# ============================ stage02: motion correction ====================
# Reads the source series in place (BIDS magnitude, or the NORDIC output) — no
# full-dataset copy. Saves the target mean + per-volume matrices + motion params;
# the corrected 4D is produced once by the final single-resample (stage10).
echo '== stage02: moco =='
MOCO_REF={plan.options.moco_ref}   # sbref | first | last | <int>
for k in "${{RUN_KEYS[@]}}"; do
  mstem="stage02.moco.${{FRAG[$k]}}"
  [ "$skip_moco" -eq 1 ] && [ -f "${{mstem}}_mean.nii$FMT" ] && continue
{_raw_source(plan)}
  sb="${{SBREF[$k]:-}}"
  case "$MOCO_REF" in
    sbref) if [ -n "$sb" ]; then base_arg=(-base "$sb"); else base_arg=(-base 0); fi ;;
    first) base_arg=(-base 0) ;;
    last)  nv=$(3dinfo -nv "$raw"); base_arg=(-base $((nv - 1))) ;;
    *)     base_arg=(-base "$MOCO_REF") ;;   # integer volume index
  esac
{_ffs("ffs_moco", ['-input "$raw"', '"${base_arg[@]}"', *_split_flags(config.DEFAULT_OPTS["moco"]), '-save_mean "${mstem}_mean.nii$FMT"', '-1Dmatrix_save "${mstem}.aff12.1D"', '-1Dfile "${mstem}.motion.1D"', '-device "$DEVICE"'])}
done
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
    return f"""
# ============================ stage06: cross-run alignment ==================
# Align each run's mean to its anchor (fmap group mean, or the session's first
# run when there are no fmaps). Saves a matrix that composes into the chain.
echo '== stage06: cross-run alignment =='
for k in "${{RUN_KEYS[@]}}"; do
  base="${{XRUNBASE[$k]:-}}"
  [ -z "$base" ] && continue   # this run is the anchor (identity) — no xrun
  xstem="stage06.xrun.${{FRAG[$k]}}"
  if [ "$skip_xrun" -ne 1 ] || [ ! -f "${{xstem}}.aff12.1D" ]; then
{lin}
  fi
{nl}done
"""


def _stage_grandmean(plan: Plan) -> str:
    # Per-session grandmeans (mean of that session's xrun-aligned run means). The
    # OVERALL grandmean is NOT built here for multi-session data — the session
    # grandmeans are still in their own session spaces; it is built in stage08
    # after cross-session alignment. Single-session: the overall == the session's.
    by_ses: dict[str | None, list[PlanRun]] = {}
    for pr in plan.runs:
        by_ses.setdefault(pr.bold.session, []).append(pr)
    out = ["", "# ============================ stage07: grandmeans ==========================="]
    out.append("echo '== stage07: per-session grandmeans =='")
    ses_means = []
    for ses, prs in by_ses.items():
        aligned = " ".join(f'"{_aligned_mean(pr)}"' for pr in prs)
        tag = f".ses-{ses}" if ses else ""
        gm = f"stage07.grandmean{tag}.nii$FMT"
        ses_means.append(gm)
        out.append(
            _ffs(
                "ffs_util_3dmath",
                [
                    f"-input {aligned}",
                    "-mean",
                    f'-prefix "{gm}"',
                    "-overwrite",
                    '-device "$DEVICE"',
                ],
                indent="",
            )
        )
    if not plan.multi_session and ses_means:
        # Single session: its grandmean IS the overall grandmean.
        out.append(f'cp -f "{ses_means[0]}" "stage07.grandmean.nii$FMT"')
    return "\n".join(out) + "\n"


def _xses_aligned(session: str, nonlin: bool) -> str:
    """The cross-session-aligned grandmean for a non-ref session (nl if the
    nonlinear step ran, else the linear result)."""
    stem = f"stage08.xses.ses-{session}"
    return f"{stem}_nl.nii$FMT" if nonlin else f"{stem}.nii$FMT"


def _stage_xses(plan: Plan) -> str:
    if not plan.multi_session:
        return ""
    opt = plan.options
    sessions = []
    seen = set()
    for pr in plan.runs:
        s = pr.bold.session
        if s not in seen and s is not None:
            seen.add(s)
            sessions.append(s)
    ref = plan.ref_session
    out = ["", "# ============================ stage08: cross-session alignment =============="]
    out.append("echo '== stage08: cross-session alignment =='")
    out.append(f'REFGM="stage07.grandmean.ses-{ref}.nii$FMT"')
    aligned_means = [f"stage07.grandmean.ses-{ref}.nii$FMT"]  # ref session is already in place
    for s in sessions:
        if s == ref:
            continue
        xstem = f"stage08.xses.ses-{s}"
        src = f"stage07.grandmean.ses-{s}.nii$FMT"
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

    # Overall grandmean = mean of the ref grandmean + the cross-session-aligned
    # non-ref grandmeans (all now in reference-session space). This is the target
    # the anat alignment (stage09) uses.
    out.append("echo '== stage08: overall grandmean (post-alignment) =='")
    allm = " ".join(f'"{m}"' for m in aligned_means)
    out.append(
        _ffs(
            "ffs_util_3dmath",
            [
                f"-input {allm}",
                "-mean",
                '-prefix "stage07.grandmean.nii$FMT"',
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
        return f"{opt.grand_reference.rstrip('/')}/stage07.grandmean.nii$FMT"
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
            '-source "stage07.grandmean.nii$FMT"',
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
        lin = _ffs(
            "ffs_allineate",
            [
                '-base "$ANAT"',
                '-source "stage07.grandmean.nii$FMT"',
                '-prefix "stage09.anat.nii$FMT"',
                '-1Dmatrix_save "stage09.anat.aff12.1D"',
                *_split_flags(config.DEFAULT_OPTS["anat"]),
                '-device "$DEVICE"',
            ],
        )
        out.append(
            f'ANAT="{anat_ph}"\n'
            'if [ "$skip_anat" -ne 1 ] || [ ! -f "stage09.anat.aff12.1D" ]; then\n'
            "  # cross-modal rigid lpc: base=anat → matrix maps anat→EPI (chain head).\n"
            f"{lin}\nfi"
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
        return '-input "stage07.grandmean.nii$FMT"  # (no fmap; fell back from ' + mode + ")"
    if mode == "blipfor":
        return f'-input "{blip}_unwarped.nii$FMT[0]"'
    if mode == "blip_pair":
        return f'-input "{blip}_unwarped.nii$FMT[0]" -pe_reverse "{blip}_unwarped.nii$FMT[1]"'
    return '-input "stage07.grandmean.nii$FMT"'


def _stage_final(plan: Plan) -> str:
    opt = plan.options
    if opt.slicetiming_method == "integrate":
        st = (
            '  tp="${JSON[$k]}"; tr="${TR[$k]}"\n'
            '  if [ -n "$tr" ]; then st_arg=(-tpattern "$tp" -TR "$tr" -tzero 0); else st_arg=(); fi'
        )
    else:
        st = "  st_arg=()"
    if not opt.go_to_anat:
        master = "stage07.grandmean.nii$FMT"
    elif opt.ref_file is not None:
        # explicit reference: the copied-in ref anat defines the shared grid.
        master = "stage09.ref_anat.nii.gz" if opt.ref_anat else "stage07.grandmean.nii$FMT"
    elif opt.grand_reference:
        # borrow the reference's anat-space grid so all subjects/tasks co-register.
        master = f"{opt.grand_reference.rstrip('/')}/stage09.anat.nii$FMT"
    else:
        master = "stage09.anat.nii$FMT"
    nwarp_cmd = _ffs(
        "ffs_nwarp",
        [
            '-source "$raw"',
            '-nwarp "${CHAIN[$k]}"',
            '-master "$MASTER"',
            *_split_flags(config.DEFAULT_OPTS["nwarp"]),
            '"${st_arg[@]}"',
            '-prefix "$outf"',
            '-device "$DEVICE"',
        ],
    )
    return f"""
# ============================ stage10: compose + resample ===================
# One interpolation per run applies the whole CHAIN in a single pass. The source
# is read in place — the NORDIC output, or the original BIDS magnitude (noise
# vols trimmed inline) — so no raw copy is ever materialised. The chain maps it
# to the final grid; override the grid with FFS_MASTER (e.g. EPI-res anat).
echo '== stage10: final compose + resample =='
MASTER="${{FFS_MASTER:-{master}}}"
for k in "${{RUN_KEYS[@]}}"; do
  outf="stage10.final.${{FRAG[$k]}}.nii$FINAL_FMT"
  [ "$skip_final" -eq 1 ] && [ -f "$outf" ] && continue
{st}
{_raw_source(plan)}
{nwarp_cmd}
done
echo 'done → stage10.final.*'
"""


def _stage_phase_stub(plan: Plan) -> str:
    if not getattr(plan.options, "phase_proc", False):
        return ""
    return """
# ============================ stage: phase processing (TODO) ================
# -phase_proc requested but not yet implemented in this milestone. Planned:
#   * romeo unwrap (-phase_first_unwrap) before NORDIC,
#   * NORDIC writes phase (-make-complex-nii),
#   * carry phase through stage10 via ffs_nwarp -phase/-phase_warp {split,direct}.
echo 'NOTE: phase processing is stubbed — see the TODO block in this script.'
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
                f'-Rbuck "stage12.stats.task-{task}.nii$GLM_FMT"',
                "-tout",
                "-fout",
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


def write_script(plan: Plan, out_dir: str, bids_root: str | None = None) -> str:
    """Assemble the full pipeline script text for ``plan``."""
    parts = [
        _header(plan, out_dir),
        _data_arrays(plan),
        _preflight(plan),
        _stage_phase_stub(plan),
        _stage_nordic(plan),
        _stage_tshift(plan),
        _stage_moco(plan),
        _stage_locomoco(plan),
        _stage_blip(plan),
        _stage_xrun(plan),
        _stage_grandmean(plan),
        _stage_xses(plan),
        _stage_xref(plan),
        _stage_anat(plan),
        _stage_final(plan),
        _stage_stats(plan, bids_root),
    ]
    return "\n".join(p for p in parts if p).rstrip() + "\n"
