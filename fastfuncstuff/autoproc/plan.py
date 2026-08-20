"""Processing plan: reference resolution + per-run warp-chain composition.

Turns a scanned :class:`~fastfuncstuff.autoproc.bids.Subject` plus the user's
options into a :class:`Plan` — the fully-resolved description the emitter walks
to write bash. All alignment *policy* lives here (which session/fmap/run is the
reference, which transform links a given run needs); the emitter only turns
tokens into filenames.

Warp-chain token order (output/anat space → native source; the same order
``ffs_nwarp -nwarp`` consumes, leftmost acting first on the output coordinate)::

    anat_lin  xref_nl xref_lin  anat_nl  xses_nl xses_lin  xfmap_nl xfmap_lin
        blip_half  wxrun_nl wxrun_lin  locomoco  moco

Verified against both reference ffs scripts and the AFNI final-apply block:
  * own-anat, single session (floc):   anat_lin  anat_nl  blip_half  wxrun*  ...
  * grand-reference (primary→floc):    anat_lin(borrowed)  xref_lin  anat_nl  blip_half  wxrun*  ...
``anat_nl`` (an ffs_segment invwarp computed on the *current* data) sits just
below the anat/xref block in both. It is estimated on the *grandmean* — which
lives in the reference session's space — so it must act on data that xses has
already brought there; listing it above ``xses_*`` is what does that (bug of
record: multi-session runs had it acting in session-native space). The
per-level lin/nl order is deliberately
*not* uniform (anat is lin-then-nl; the cross-* levels are nl-then-lin) — it
mirrors how each tool stores its warp, and matches the reference scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

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
    # One slice-timing file (text, one offset per slice in seconds, or a JSON with
    # SliceTiming) used for EVERY run, in place of each run's BIDS sidecar. For
    # data whose sidecars carry no SliceTiming. None → per-run sidecar.
    slicetiming_file: str | None = None
    # Volume TR (seconds) for every run, overriding the sidecar/header value. 3D
    # acquisitions often store the per-partition time in the header, not the
    # volume TR, which is the number slice timing and the GLM actually need.
    tr: float | None = None
    distortion: bool = True  # apply fieldmap distortion correction when fmaps exist
    run_glm: bool = True  # emit the GLM stage enabled (else behind FFS_RUN_GLM)
    # Named nuisance sources for the GLM, keys of config.GLM_ORTVEC (e.g.
    # ["motion", "motion_deriv", "locomoco"]). Empty = no nuisance regressors.
    glm_ortvec: list[str] = field(default_factory=list)
    # Continuous stimulus vectors for the GLM: (LABEL[:mod], path-or-glob) pairs,
    # each becoming one [[stim_vec]] block. `{task}` in the path is substituted
    # per task. Unlike glm_ortvec these are user-supplied files -- nothing in the
    # pipeline produces them -- so they are named directly rather than by key.
    glm_stim_vec: list[tuple[str, str]] = field(default_factory=list)
    # Extra ffs_reml flags appended to the GLM command, as one string.
    glm_opts: str = ""
    # TRs dropped from each end of every run at GLM time (ffs_reml
    # -drop_first/-drop_last). Dropping leading TRs shifts the event timing back
    # by N*TR automatically -- the design is compiled against the trimmed runs.
    glm_drop_first: int = 0
    glm_drop_last: int = 0
    # Write the design TOML even if one is already there (an edited spec is
    # otherwise never clobbered — that is the whole point of generating it).
    glm_spec_overwrite: bool = False
    # Column names inside the events TSVs: (onset, duration, trial_type). None =
    # the BIDS defaults. ``spec_event_cols`` applies to every task;
    # ``sep_spec_event_cols`` maps a task name to its own triple and wins for
    # that task — one non-BIDS task in a session does not force the others to
    # be spelled out. A requested column that is not in the file falls back to
    # the defaults for that task, with a warning.
    spec_event_cols: tuple[str, str, str] | None = None
    sep_spec_event_cols: dict[str, tuple[str, str, str]] = field(default_factory=dict)
    locomoco: bool = False
    xrun_nonlin: bool = False
    xfmap_nonlin: bool = False
    xses_nonlin: bool = False
    # Estimate the nonlinear refinement in the SOURCE's own frame instead of the
    # base's: hand ffs_formwarp the linear stage's matrix and its *un*-allineated
    # input, and it inverts the matrix, pulls the base onto the source grid, and
    # solves there. The source is never resampled. Per stage, because the trade
    # depends on which image has the fuller FoV -- see ../fmri_wiki/concepts/SyN.md,
    # where the default (base-frame) arrangement measured better at a clipped edge.
    xrun_nonlin_in_source: bool = False
    xfmap_nonlin_in_source: bool = False
    xses_nonlin_in_source: bool = False
    ref_ses: str | None = None
    fmap_ref: list[str] | None = None
    # -ref_image: which EPI-contrast image REPRESENTS a level. One vocabulary at
    # two levels — it picks each session's cross-session alignment source AND
    # (unless -anat_source overrides it) the image the anat step aligns. They are
    # the same question: "what stands in for this data in the space above it?".
    # None → the historical behaviour, i.e. the session mean for xses.
    ref_image: str | None = None
    go_to_anat: bool = True  # False → final space is the EPI grandmean
    final_dxyz: str | None = None  # final output voxel size (mm); None → input EPI res
    anat_nonlin: bool = False  # segment/rbr nonlinear anat refinement
    anat_path: str | None = None  # skull-stripped T1w (baked into the script if found)
    moco_ref: str = "sbref"  # moco base: sbref|first|last|<int>  (sbref → sbref if present)
    # Which EPI-contrast image the anat linear step aligns to. All choices live on
    # the SAME grid (the reference fmap's undistorted space) — see
    # ``effective_anat_source``; they differ only in SNR/sharpness/contrast.
    # auto = sbmean where the SBRef lane exists, else grandmean (see
    # ``effective_anat_source``).
    anat_source: str = "auto"  # auto | grandmean | sbmean | ref_fmap | mean_fmap
    anat_nonlin_input: str = "auto"  # ffs_segment input: + blipfor | blip_pair
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
    # Emit the ``stageNN.QC.*`` stacks: at every level, the set of images that
    # stage claims to have brought into one space, concatenated along time into a
    # single 4-D file. Cheap (one temporal concat per group) and it is what makes
    # a misregistration visible instead of inferred.
    qc: bool = True
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
    # Compute device handed to every ffs_* stage as ``-device $DEVICE``
    # (cuda | mps | cpu | auto, plus the "cuda,0" / "cpu,8" forms every ffs CLI
    # parses). Emitted as the DEVICE variable so it stays editable in the script.
    device: str = config.DEFAULT_DEVICE

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
    # anat_nl is estimated on the grandmean (reference-session space), so the
    # data must already be there when it acts — i.e. AFTER xses, which in this
    # leftmost-acts-first order means it is listed BEFORE the xses tokens.
    "anat_nl",
    "xses_nl",
    "xses_lin",
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
    # cross-fmap alignment and the per-run premeans land on. Set for EVERY run of
    # a session that has fieldmaps, including one with no fieldmap of its own:
    # that run still has to land on this grid or it drops out of the session mean
    # (bug of record: an unclaimed run aligned to the session's first run instead,
    # so its runmean sat on a different grid AND was never undistorted).
    ref_fmap_id: str | None = None
    # Kind of that reference group, which decides which stage04 output is its
    # forward image: a pepolar pair's is sub-brick 0 of ``_unwarped``, a GRE
    # group's is its ``_mean`` (one warped run rep — there is no pair to average).
    ref_fmap_is_b0: bool = False
    # This run was not claimed by any fieldmap's IntendedFor / acquisition time and
    # inherited the session's reference group. Header-note only.
    fmap_inherited: bool = False
    warp_chain: list[str] = field(default_factory=list)
    # This run's SBRef is usable as an alignment source: it exists, and the moco
    # base IS that SBRef — which is what puts it in the run's post-moco space
    # with no transform of its own (see ``sbref_chain``).
    use_sbref: bool = False


@dataclass
class Plan:
    subject: str
    options: Options
    runs: list[PlanRun]
    ref_session: str | None
    multi_session: bool

    @property
    def use_sbref(self) -> bool:
        """True when the SBRef lane is available for the *whole* dataset.

        All-or-nothing on purpose: the lane's value is that every run's image is
        directly comparable to every other's, and a sesmean silently mixing SBRef
        and BOLD-mean contrast would be worse than either lane alone. A dataset
        that has SBRefs at all normally has one per BOLD run.
        """
        return bool(self.runs) and all(pr.use_sbref for pr in self.runs)


def sbref_chain(pr: PlanRun) -> list[str]:
    """The warp chain for this run's SBRef: the run's chain minus the tokens that
    describe *within-run* motion.

    ``moco`` goes because the SBRef is the moco base — it already defines the
    space that token maps into, so applying it would be a double correction.
    ``locomoco`` goes for the same reason one step out: it converges to the mean
    of the rigid-corrected series, and its per-volume PE wiggles average to
    roughly nothing over a run, so that mean and the SBRef describe the same
    space to within the accuracy locomoco itself is correcting.
    """
    return [tok for tok in pr.warp_chain if tok not in ("moco", "locomoco")]


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


def session_fmap_ids(plan: Plan, session: str | None) -> list[str]:
    """Fieldmap ids in one session, that session's reference first — the groups
    whose aligned means its ``mean_fmap`` representative averages."""
    ids: list[str] = []
    ref_id: str | None = None
    for pr in plan.runs:
        if pr.fmap is None or pr.bold.session != session:
            continue
        if pr.fmap.fmap_id not in ids:
            ids.append(pr.fmap.fmap_id)
        if pr.is_ref_fmap:
            ref_id = pr.fmap.fmap_id
    if ref_id in ids:
        ids.remove(ref_id)
        ids.insert(0, ref_id)
    return ids


def effective_ref_image(plan: Plan, session: str | None, requested: str) -> str:
    """Resolve a representative-image request against what ONE session can supply.

    ``ref_fmap`` / ``mean_fmap`` need fieldmaps in *that* session; without them its
    own mean is the only image there is. ``mean_fmap`` with a single group
    degenerates to ``ref_fmap`` — averaging one image is just that image.
    ``sbmean`` needs the SBRef lane (dataset-wide, by construction).
    """
    # "grandmean" names the level's OWN mean of the data — the session mean at the
    # session level, THE grandmean at the top. One token, because it is one idea.
    if requested == "grandmean":
        return requested
    if requested == "sbmean":
        return requested if plan.use_sbref else "grandmean"
    ids = session_fmap_ids(plan, session)
    if not ids:
        return "grandmean"
    if requested == "mean_fmap" and len(ids) == 1:
        return "ref_fmap"
    return requested


def session_ref_mode(plan: Plan, session: str | None) -> str:
    """Which representative image stands in for ``session`` at the cross-session
    alignment, resolved against what that session has. Defaults to the session's
    own mean, which is what the pipeline always used."""
    return effective_ref_image(plan, session, plan.options.ref_image or "grandmean")


def effective_anat_source(plan: Plan, requested: str | None = None) -> str:
    """Resolve an anat-source request against what the data can actually supply.

    Same vocabulary as ``-ref_image``, resolved against the *anchor* session —
    with one extra condition the per-session resolver cannot know about: the anat
    step aligns an image that must live in **grandmean space**, i.e. the reference
    session's. When the reference session has no fieldmap of its own, ``ref_anchor``
    falls back to some other session's group, whose undistorted mean is NOT in that
    space (it only gets there through xses, which is estimated downstream). Taking
    it would estimate the anat matrix in the wrong frame, so the fieldmap-based
    choices degrade to the grandmean there (bug of record).
    """
    mode = requested if requested is not None else plan.options.anat_source
    # "auto" = use the SBRefs when they exist. They already lead every other
    # alignment (``emit._primary_lane`` estimates xrun/xses from the SBRef lane),
    # and the anat step is the one cross-modal ``lpc`` fit in the pipeline — the
    # place a sharp, single-interpolation, single-band image matters MOST. Falling
    # back to the BOLD grandmean there wasted the very images the rest of the
    # pipeline is anchored on.
    if mode == "auto":
        mode = "sbmean" if plan.use_sbref else "grandmean"
    if mode == "grandmean":
        return mode
    if mode == "sbmean":
        return mode if plan.use_sbref else "grandmean"
    anchor = ref_anchor(plan)
    if anchor is None or anchor.bold.session != plan.ref_session:
        return "grandmean"
    return effective_ref_image(plan, anchor.bold.session, mode)


def _resolve_ref_session(subject: Subject, opt: Options) -> str | None:
    """The session everything else is aligned to (its reference fieldmap defines
    the final EPI space).

    An unrecognised ``-ref_ses`` raises rather than quietly falling back to the
    first session: the reference is the one choice that changes every warp chain
    in the script, and a typo'd label would produce a plausible-looking pipeline
    anchored on the wrong session.
    """
    labels = [s.session for s in subject.sessions]
    if opt.ref_ses is not None:
        want = opt.ref_ses[len("ses-") :] if opt.ref_ses.startswith("ses-") else opt.ref_ses
        if want in labels:
            return want
        have = ", ".join(str(lab) for lab in labels) or "(none scanned)"
        raise ValueError(f"-ref_ses {opt.ref_ses!r}: no such session in scope. Scanned: {have}")
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


def _pe_compatible(run: BoldRun, fg: FmapGroup) -> bool:
    """True when ``fg``'s displacement field is applicable to ``run`` — same phase-
    encode axis AND polarity. An unknown direction on either side is permissive
    (the scan is quirk-tolerant; the common case is a sidecar that omits it).

    Polarity matters as much as the axis: applying an AP field to a PA run doubles
    the distortion instead of removing it. For a GRE group the constraint is on the
    *warp*, not the measurement — the Hz field has no polarity — so a b0 group has
    already been split per polarity by :func:`split_b0_by_polarity` and ``pe_dir``
    reports which split this one is.
    """
    a, b = run.pe_dir, fg.pe_dir
    return not a or not b or a == b


def split_b0_by_polarity(fmaps: list[FmapGroup], bold_runs: list[BoldRun]) -> list[FmapGroup]:
    """Give each PE polarity a GRE group serves its own group (and hence its own
    warp), leaving reverse-PE groups untouched.

    A measured GRE field is polarity-free: the same Hz map corrects AP and PA runs
    alike. The *displacement* it implies is not — it flips sign — so one group
    cannot own one warp for both. Splitting here rather than emitting two warps per
    group keeps the rest of the pipeline's "one fmap_id, one blip stem" invariant,
    and the two halves land in different undistorted spaces, which cross-fmap
    alignment (stage05) already exists to reconcile.

    Groups whose runs are all one polarity (the overwhelmingly common case) come
    back unchanged apart from ``pe_override``, so no ids churn.
    """
    pe_by_run = {(r.task, r.run): r.pe_dir for r in bold_runs}
    out: list[FmapGroup] = []
    for fg in fmaps:
        if not fg.is_b0:
            out.append(fg)
            continue
        by_pe: dict[str | None, list[tuple[str, str]]] = {}
        for key in fg.intended_runs:
            by_pe.setdefault(pe_by_run.get(key), []).append(key)
        multi = len([pe for pe in by_pe if pe]) > 1
        for pe, keys in by_pe.items():
            # A suffix only when there is something to disambiguate; PE strings
            # carry a '-' that would read as a separator in the filename fragment.
            tag = (pe or "").replace("-", "neg") if multi else ""
            out.append(
                replace(
                    fg,
                    fmap_id=f"{fg.fmap_id}-{tag}" if tag else fg.fmap_id,
                    intended_runs=keys,
                    pe_override=pe,
                )
            )
    return out


def _fmap_for_run(
    run: BoldRun, session_fmaps: list[FmapGroup], ref_fmap: FmapGroup | None = None
) -> tuple[FmapGroup | None, bool]:
    """This run's fieldmap group, and whether it was inherited rather than claimed.

    ``IntendedFor`` (resolved in bids.py) is authoritative. A run no group claims
    falls back to the session's *reference* group when that group's field is
    applicable — one unclaimed run is far more likely to be a sidecar omission
    than a run genuinely acquired with no usable fieldmap, and the alternative
    (no fieldmap at all) strands it off the session's common grid. A run whose PE
    direction rules the reference field out gets no group: it still lands on the
    common grid (see ``_xrun_base``), just without undistortion.
    """
    # intended_runs holds (task, run) pairs, so this is collision-proof across
    # tasks that share run numbers (rest/run-01 vs skilled/run-01).
    for fg in session_fmaps:
        if (run.task, run.run) in fg.intended_runs:
            return fg, False
    if ref_fmap is not None and _pe_compatible(run, ref_fmap):
        return ref_fmap, True
    # A GRE field measured for this session applies to this run whatever its
    # polarity; only the warp differs, and the polarity split already built one
    # per direction. So an unclaimed run whose PE rules the reference split out
    # can still inherit its sibling instead of going uncorrected.
    if ref_fmap is not None and ref_fmap.is_b0:
        for fg in session_fmaps:
            if fg.is_b0 and _pe_compatible(run, fg):
                return fg, True
    return None, False


def _borrowed_forward(fg: FmapGroup, bold_runs: list[BoldRun]) -> str | None:
    """The run rep that stands in as this fieldmap's forward (blip-up) image.

    The one it was acquired NEXT TO in time — the nearest run at-or-after the
    fieldmap (a fieldmap is normally run just before the block it covers, the
    same convention :func:`bids._assign_by_time` claims runs by), else the
    nearest one at all. Not the first one in scan order.
    The pair is only a pair to the extent the head did not move between them:
    the field is estimated from forward-vs-reverse difference, so any motion in
    that gap is written straight into the field, and everything the group
    corrects inherits it. Session order is task-then-run, which in a session that
    interleaves tasks put the borrowed image tens of minutes from the fieldmap
    (bug of record: sub-3001's LR-run1 borrowed a run 16 minutes later when one
    2.5 minutes later was in the same group).

    Falls back to the first intended run when the sidecars carry no
    ``AcquisitionTime`` — there is nothing better to order by then.
    """
    mine = [b for b in bold_runs if (b.task, b.run) in fg.intended_runs]
    if not mine:
        return None
    ft = fg.acq_time
    timed = [b for b in mine if b.acq_time is not None]
    if ft is not None and timed:
        following = [b for b in timed if b.acq_time >= ft]  # type: ignore[operator]
        pool = following or timed
        return str(min(pool, key=lambda b: abs(b.acq_time - ft)).rep)  # type: ignore[operator]
    return str(mine[0].rep)


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
    toks = [tok for tok in _CHAIN_ORDER if include.get(tok, False)]
    return _apply_in_source_swaps(toks, opt)


# Which option flips which (nonlinear, linear) pair. _CHAIN_ORDER is written
# leftmost-acts-first *on coordinates*, i.e. rightmost-acts-first on the data, so
# the default "affine then warp" reads as (nl, lin). Estimating the warp in the
# source frame makes it act on the data BEFORE the affine, which is the other order.
_IN_SOURCE_PAIRS = (
    ("xses_nonlin_in_source", "xses_nl", "xses_lin"),
    ("xfmap_nonlin_in_source", "xfmap_nl", "xfmap_lin"),
    ("xrun_nonlin_in_source", "wxrun_nl", "wxrun_lin"),
)


def _apply_in_source_swaps(tokens: list[str], opt) -> list[str]:
    """Swap each in-source stage's nonlinear link past its own affine.

    Only the pair moves; every other link keeps its place, because only that one
    warp changed the space it lives in.
    """
    out = list(tokens)
    for attr, nl, lin in _IN_SOURCE_PAIRS:
        if not getattr(opt, attr, False):
            continue
        if nl not in out or lin not in out:
            continue
        i, j = out.index(nl), out.index(lin)
        out[i], out[j] = out[j], out[i]
    return out


def build_plan(subject: Subject, opt: Options) -> Plan:
    """Resolve references and build every run's warp chain."""
    ref_session = _resolve_ref_session(subject, opt)
    multi_session = len([s for s in subject.sessions if s.session is not None]) > 1

    runs: list[PlanRun] = []
    for sess in subject.sessions:
        # Do this before anything reads sess.fmaps: the split can change fmap_ids,
        # and the reference group must be chosen from the post-split list.
        sess.fmaps = split_b0_by_polarity(sess.fmaps, sess.bold_runs)
        ref_fmap = _resolve_ref_fmap(sess.fmaps, opt)
        # -distortion off (e.g. bare_bones): treat as no fmaps → no blip/xfmap,
        # xrun falls back to first-run anchoring.
        has_fmaps = bool(sess.fmaps) and opt.distortion
        # First run of the session anchors xrun when there are no fmaps. Identity
        # is the BoldRun itself, not its `run` entity: a session whose runs are
        # distinguished by TASK (task-bar1/bar2/... with no run- entity) has
        # run=None everywhere, so comparing `run` made every run the reference and
        # dropped xrun alignment entirely (bug of record).
        session_first_run = sess.bold_runs[0] if sess.bold_runs else None
        # Each fmap group's DISTORTED forward = the image stage04 estimates the
        # field against, and hence the xrun base for that group: the fmap's own
        # matched-PE mate when it has one (self-contained AP/PA pair), else the
        # borrowed rep of one of its runs — which is always the case for a GRE
        # fieldmap, since a GRE acquisition is not an EPI and has no distorted
        # mate at all.
        forward_by_fmap: dict[str, str] = {}
        if has_fmaps:
            for fg in sess.fmaps:
                if fg.forward_path is not None:
                    forward_by_fmap[fg.fmap_id] = str(fg.forward_path)
                    continue
                borrowed = _borrowed_forward(fg, sess.bold_runs)
                if borrowed is not None:
                    forward_by_fmap[fg.fmap_id] = borrowed
        ref_fmap_id = ref_fmap.fmap_id if ref_fmap is not None else None
        for bold in sess.bold_runs:
            fmap, inherited = (
                _fmap_for_run(bold, sess.fmaps, ref_fmap) if has_fmaps else (None, False)
            )
            pr = PlanRun(
                bold=bold,
                fmap=fmap,
                is_ref_session=(sess.session == ref_session),
                is_ref_fmap=(
                    fmap is not None and ref_fmap is not None and fmap.fmap_id == ref_fmap.fmap_id
                ),
                is_ref_run=(not has_fmaps and bold is session_first_run),
                fmap_forward=(forward_by_fmap.get(fmap.fmap_id) if fmap else None),
                # Every run of a fieldmap session shares the session's common grid,
                # whether or not it has a fieldmap of its own.
                ref_fmap_id=(ref_fmap_id if has_fmaps else None),
                ref_fmap_is_b0=(has_fmaps and ref_fmap is not None and ref_fmap.is_b0),
                fmap_inherited=inherited,
                # Only ``-moco_ref sbref`` makes the SBRef the post-moco space; any
                # other base leaves it an unregistered image with no transform of
                # its own, so the lane is off.
                use_sbref=(bold.sbref_path is not None and opt.moco_ref == "sbref"),
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
