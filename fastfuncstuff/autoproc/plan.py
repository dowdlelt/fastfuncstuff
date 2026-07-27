"""Processing plan: reference resolution + per-run warp-chain composition.

Turns a scanned :class:`~fastfuncstuff.autoproc.bids.Subject` plus the user's
options into a :class:`Plan` — the fully-resolved description the emitter walks
to write bash. All alignment *policy* lives here (which session/fmap/run is the
reference, which transform links a given run needs); the emitter only turns
tokens into filenames.

Warp-chain token order (output/anat space → native source; the same order
``ffs_nwarp -nwarp`` consumes, leftmost acting first on the output coordinate)::

    anat_lin  xref_nl xref_lin  xses_nl xses_lin  anat_nl  xfmap_nl xfmap_lin
        blip_half  wxrun_nl wxrun_lin  locomoco  moco

Verified against both reference ffs scripts and the AFNI final-apply block:
  * own-anat, single session (floc):   anat_lin  anat_nl  blip_half  wxrun*  ...
  * grand-reference (primary→floc):    anat_lin(borrowed)  xref_lin  anat_nl  blip_half  wxrun*  ...
``anat_nl`` (an ffs_segment invwarp computed on the *current* data) sits just
before the fieldmap block in both. The per-level lin/nl order is deliberately
*not* uniform (anat is lin-then-nl; the cross-* levels are nl-then-lin) — it
mirrors how each tool stores its warp, and matches the reference scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fastfuncstuff.autoproc import config
from fastfuncstuff.autoproc.bids import BoldRun, FmapGroup, Subject


@dataclass
class Options:
    """Resolved autoproc options relevant to plan/warp-chain construction.

    Populated from the CLI Namespace; kept as a plain dataclass so plan tests
    can construct it directly without argparse.
    """

    recipe: str | None = None  # named preset (for the header/provenance only)
    want_nordic: bool = False
    # Carry the phase timeseries through the pipeline: ROMEO-unwrap it up front
    # (after NORDIC when NORDIC runs), then ride it along the magnitude's warp
    # chain at the final resample. Nothing between those two points touches it.
    phase_proc: bool = False
    noise_vols: int = 0
    slicetiming_method: str = "integrate"  # integrate (fold into final resample) | first | none
    distortion: bool = True  # apply fieldmap distortion correction when fmaps exist
    run_glm: bool = True  # emit the GLM stage enabled (else behind FFS_RUN_GLM)
    glm_ortvec: bool = False  # add motion + locomoco-PC nuisance regressors to the GLM
    locomoco: bool = False
    xrun_nonlin: bool = False
    xfmap_nonlin: bool = False
    xses_nonlin: bool = False
    ref_ses: str | None = None
    fmap_ref: list[str] | None = None
    go_to_anat: bool = True  # False → final space is the EPI grandmean
    final_dxyz: str | None = None  # final output voxel size (mm); None → input EPI res
    anat_nonlin: bool = False  # segment/rbr nonlinear anat refinement
    anat_path: str | None = None  # skull-stripped T1w (baked into the script if found)
    moco_ref: str = "sbref"  # moco base: sbref|first|last|<int>  (sbref → sbref if present)
    # Which EPI-contrast image the anat linear step aligns to. All choices live on
    # the SAME grid (the reference fmap's undistorted space) — see
    # ``effective_anat_source``; they differ only in SNR/sharpness/contrast.
    anat_source: str = "grandmean"  # grandmean | ref_fmap | mean_fmap
    anat_nonlin_input: str = "grandmean"  # ffs_segment input: + blipfor|blip_pair
    # -grand_reference: path to ANOTHER autoproc results dir whose anat matrix
    # this run borrows; this data's grandmean is aligned to that ref (xref_*).
    # This is how a filtered `primary`-only script anchors on a floc run.
    grand_reference: str | None = None
    grand_reference_nonlin: bool = False
    tpm: str | None = None  # subject tissue-probability template for ffs_segment
    tpm_source: str | None = None  # QC anat-in-EPI source for ffs_segment
    fs_tpm: bool = False  # build the TPM in-script from FreeSurfer (SUMA) outputs
    suma_dir: str | None = None  # FreeSurfer SUMA dir (aseg.auto + SurfVol for the TPM)
    events: list[str] | None = None  # events TSV(s) passed directly (bids format)
    # Explicit reference override (alternative to -grand_reference DIR): an EPI
    # contrast image to align to, the nwarp-order matrices mapping it to anat, and
    # the anat itself (copied in; used as ffs_segment tpm-space / QC).
    ref_file: str | None = None
    ref_transforms: list[str] | None = None
    ref_anat: str | None = None
    # Batched stages (moco, final resample) skip already-complete runs by default
    # (via each tool's -batch_skip). True forces every run to re-process.
    batch_overwrite: bool = False
    # Output compression suffixes appended after ``.nii`` (emitted as FMT /
    # FINAL_FMT / GLM_FMT). "" = uncompressed .nii, ".gz", or ".zst". Working
    # intermediates default to .zst (read many times); final + GLM to .gz
    # (portable). See config.DEFAULT_*_FMT.
    fmt: str = config.DEFAULT_FMT
    final_fmt: str = config.DEFAULT_FINAL_FMT
    glm_fmt: str = config.DEFAULT_GLM_FMT

    @property
    def has_grand_ref(self) -> bool:
        """True when a reference anchor is set (either the results-dir or the
        explicit ref_file form)."""
        return self.grand_reference is not None or self.ref_file is not None


# Canonical chain order, reference-side first. Each token carries the level it
# belongs to so the drop rules read cleanly.
_CHAIN_ORDER = (
    "anat_lin",
    "xref_nl",
    "xref_lin",
    "xses_nl",
    "xses_lin",
    "anat_nl",
    "xfmap_nl",
    "xfmap_lin",
    "blip_half",
    "wxrun_nl",
    "wxrun_lin",
    "locomoco",
    "moco",
)


@dataclass
class PlanRun:
    bold: BoldRun
    fmap: FmapGroup | None
    is_ref_session: bool
    is_ref_fmap: bool
    is_ref_run: bool  # only meaningful in first-run-anchored (no-fmap) mode
    # The DISTORTED forward image of this run's fmap group (its blip_up input) —
    # the xrun base, since blip_half undistorts *after* xrun in the chain.
    fmap_forward: str | None = None
    # The session's reference fmap id — the common (undistorted) grid that the
    # cross-fmap alignment and the per-run premeans land on.
    ref_fmap_id: str | None = None
    warp_chain: list[str] = field(default_factory=list)


@dataclass
class Plan:
    subject: str
    options: Options
    runs: list[PlanRun]
    ref_session: str | None
    multi_session: bool


def ref_anchor(plan: Plan) -> PlanRun | None:
    """The run carrying the *reference* fieldmap of the *reference* session — the
    group whose undistorted mean defines the space everything else lands on.

    ``None`` when no run has a fieldmap (``-no_distortion``, or no fmaps scanned),
    which is what makes the fmap-based anat sources unreachable.
    """
    with_fmap = [pr for pr in plan.runs if pr.fmap is not None]
    for pr in with_fmap:
        if pr.is_ref_session and pr.is_ref_fmap:
            return pr
    # Reference session has no fmap of its own: fall back to any ref-fmap group.
    for pr in with_fmap:
        if pr.is_ref_fmap:
            return pr
    return with_fmap[0] if with_fmap else None


def anchor_fmap_ids(plan: Plan) -> list[str]:
    """Fieldmap ids in the anchor's session, reference first — the groups whose
    aligned means ``mean_fmap`` averages. Cross-*session* fmaps are excluded: they
    only reach the anchor grid via ``xses``, which is computed downstream of here.
    """
    anchor = ref_anchor(plan)
    if anchor is None:
        return []
    ref_id = anchor.fmap.fmap_id if anchor.fmap else None
    ids: list[str] = []
    for pr in plan.runs:
        if pr.fmap is None or pr.bold.session != anchor.bold.session:
            continue
        if pr.fmap.fmap_id not in ids:
            ids.append(pr.fmap.fmap_id)
    if ref_id in ids:  # reference group first
        ids.remove(ref_id)
        ids.insert(0, ref_id)
    return ids


def effective_anat_source(plan: Plan, requested: str | None = None) -> str:
    """Resolve an anat-source request against what the data can actually supply.

    ``ref_fmap`` / ``mean_fmap`` need fieldmaps; without them the grandmean is the
    only EPI-contrast image there is (and ``ffs_segment`` is then what recovers the
    distortion). ``mean_fmap`` with a single group degenerates to ``ref_fmap`` —
    averaging one image is just that image.
    """
    mode = requested if requested is not None else plan.options.anat_source
    if mode == "grandmean":
        return mode
    ids = anchor_fmap_ids(plan)
    if not ids:
        return "grandmean"
    if mode == "mean_fmap" and len(ids) == 1:
        return "ref_fmap"
    return mode


def _resolve_ref_session(subject: Subject, opt: Options) -> str | None:
    labels = [s.session for s in subject.sessions]
    if opt.ref_ses is not None:
        want = opt.ref_ses[len("ses-") :] if opt.ref_ses.startswith("ses-") else opt.ref_ses
        if want in labels:
            return want
    return labels[0] if labels else None


def _resolve_ref_fmap(session_fmaps: list[FmapGroup], opt: Options) -> FmapGroup | None:
    """Reference fmap group for a session: user-named id, else the first."""
    if not session_fmaps:
        return None
    if opt.fmap_ref:
        wanted = set(opt.fmap_ref)
        for fg in session_fmaps:
            if fg.fmap_id in wanted:
                return fg
    return session_fmaps[0]


def _fmap_for_run(run: BoldRun, session_fmaps: list[FmapGroup]) -> FmapGroup | None:
    # intended_runs holds (task, run) pairs, so this is collision-proof across
    # tasks that share run numbers (rest/run-01 vs skilled/run-01).
    for fg in session_fmaps:
        if (run.task, run.run) in fg.intended_runs:
            return fg
    return None


def build_warp_chain(pr: PlanRun, opt: Options, multi_session: bool) -> list[str]:
    """Compose the ordered transform tokens for one run, applying drop rules.

    Drop rules:
      * no anat target → drop ``anat_*`` (final space = EPI grandmean).
      * reference session (or single session) → drop ``xses_*``.
      * reference fmap group (or no fmaps) → drop ``xfmap_*``.
      * no fmap for this run → drop ``blip_half`` and ``xfmap_*``.
      * first-run-anchored mode, reference run → drop ``wxrun_*``.
      * nonlinear links present only when the matching ``*_nonlin`` opt is set.
      * ``locomoco`` present only with ``-locomoco``.
    """
    has_fmap = pr.fmap is not None
    grand_ref = opt.has_grand_ref
    include = {
        "anat_lin": opt.go_to_anat,
        # xref: align this data's grandmean to the external reference's grandmean.
        "xref_lin": opt.go_to_anat and grand_ref,
        "xref_nl": opt.go_to_anat and grand_ref and opt.grand_reference_nonlin,
        "anat_nl": opt.go_to_anat and opt.anat_nonlin,
        "xses_lin": multi_session and not pr.is_ref_session,
        "xses_nl": multi_session and not pr.is_ref_session and opt.xses_nonlin,
        "xfmap_lin": has_fmap and not pr.is_ref_fmap,
        "xfmap_nl": has_fmap and not pr.is_ref_fmap and opt.xfmap_nonlin,
        "blip_half": has_fmap,
        # fmap-anchored: every run aligns to its fmap ref (always present).
        # first-run-anchored (no fmap): drop for the reference run.
        "wxrun_lin": has_fmap or not pr.is_ref_run,
        "wxrun_nl": (has_fmap or not pr.is_ref_run) and opt.xrun_nonlin,
        "locomoco": opt.locomoco,
        "moco": True,
    }
    return [tok for tok in _CHAIN_ORDER if include.get(tok, False)]


def build_plan(subject: Subject, opt: Options) -> Plan:
    """Resolve references and build every run's warp chain."""
    ref_session = _resolve_ref_session(subject, opt)
    multi_session = len([s for s in subject.sessions if s.session is not None]) > 1

    runs: list[PlanRun] = []
    for sess in subject.sessions:
        ref_fmap = _resolve_ref_fmap(sess.fmaps, opt)
        # -distortion off (e.g. bare_bones): treat as no fmaps → no blip/xfmap,
        # xrun falls back to first-run anchoring.
        has_fmaps = bool(sess.fmaps) and opt.distortion
        # First run of the session anchors xrun when there are no fmaps.
        session_first_run = sess.bold_runs[0].run if sess.bold_runs else None
        # Each fmap group's DISTORTED forward = the (first intended run's) blip_up
        # image — the same image _stage_blip feeds to ffs_blipflip. This is the
        # xrun base for that group.
        forward_by_fmap: dict[str, str] = {}
        if has_fmaps:
            for fg in sess.fmaps:
                for b in sess.bold_runs:
                    if (b.task, b.run) in fg.intended_runs:
                        forward_by_fmap[fg.fmap_id] = str(b.rep)
                        break
        ref_fmap_id = ref_fmap.fmap_id if ref_fmap is not None else None
        for bold in sess.bold_runs:
            fmap = _fmap_for_run(bold, sess.fmaps) if has_fmaps else None
            pr = PlanRun(
                bold=bold,
                fmap=fmap,
                is_ref_session=(sess.session == ref_session),
                is_ref_fmap=(
                    fmap is not None and ref_fmap is not None and fmap.fmap_id == ref_fmap.fmap_id
                ),
                is_ref_run=(not has_fmaps and bold.run == session_first_run),
                fmap_forward=(forward_by_fmap.get(fmap.fmap_id) if fmap else None),
                ref_fmap_id=(ref_fmap_id if fmap else None),
            )
            pr.warp_chain = build_warp_chain(pr, opt, multi_session)
            runs.append(pr)

    return Plan(
        subject=subject.subject,
        options=opt,
        runs=runs,
        ref_session=ref_session,
        multi_session=multi_session,
    )
