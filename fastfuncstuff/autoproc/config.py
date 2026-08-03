"""Default per-stage option strings for ffs_autoproc.

Each ``-<stage>_opts`` CLI flag *replaces* the corresponding default string
here; ``--help`` prints the default so the user knows what they're overriding
(afni_proc.py style). Keep these as the flags a hand-written script would use —
they are the pipeline's opinion, not a minimal set.

The nonlinear-edge-safety requirement (don't let a cut-off FOV drive phantom
stretching) is realized through the registration defaults: ``-source_automask
-autoweight`` on every linear step, which weights brain over the empty/near-zero
FOV. Callers wanting a harder zero-weight threshold override via ``-*_opts``.
"""

from __future__ import annotations

# Per-stage op strings. Every key here gets a ``-<key>_opts`` CLI flag (see
# STAGE_OPT_KEYS); structural flags (-input/-prefix/-device/paths the emitter
# computes) are NOT in these strings and cannot be overridden this way.
DEFAULT_OPTS: dict[str, str] = {
    # NORDIC denoise. -nordic = the NORDIC (not MPPCA) threshold rule.
    "nordic": "-nordic -verbose",
    # slice timing (-slicetiming_method first). tzero 0 = shift to the first
    # slice's acquisition instant, matching what the integrate path assumes.
    "tshift": "-tzero 0",
    # motion correction (per run). wls = batched Gauss-Newton, the fast default and
    # the right cost for within-run same-contrast rigid moco (lpa forces a slow
    # per-volume optimizer — do NOT use it here). -twopass = coarse-blur then fine
    # pass for robustness to larger motion; independent of the cost, stays batched.
    "moco": "-cost wls -twopass",
    # residual PE-axis nonlinear motion (locomoco).
    "locomoco": "-backend flow -superhard -no_movie -ref mean -want_pcs 3",
    # fieldmap distortion (blipflip); reproduces FSL b02b0 by default.
    "blip": "-workhard",
    # measured GRE fieldmap (b0fmap). Defaults are the tool's own — the field
    # conditioning was tuned there against real sinus gradients, so overriding it
    # here would silently diverge from what the tool was validated with.
    "b0fmap": "",
    # cross-run linear: rigid lpa to the fmap/first-run reference. -final wsinc5:
    # these aligned means feed the grandmean, so a sharper single resample avoids
    # accumulating interpolation blur down the chain (applies to every allineate).
    "xrun": "-rigid -cost lpa -source_automask -autoweight -final wsinc5",
    # cross-run nonlinear refinement (formwarp/SyN residual). -final_interp wsinc5
    # for the same reason (sharper warped mean).
    "xrun_nl": "-metric lpa -cc_radius 4 -update_var 2 -final_interp wsinc5 "
    "-iters 1000x1000x1000 -smooth 0x0x0 -conv_thresh 1e-07",
    # cross-fmap linear/nonlinear (align non-ref fmap groups to the ref fmap).
    "xfmap": "-rigid -cost lpa -source_automask -autoweight -smallrange -final wsinc5",
    "xfmap_nl": "-metric lpa -cc_radius 4 -update_var 2 -final_interp wsinc5 "
    "-iters 1000x1000x1000 -smooth 0x0x0 -conv_thresh 1e-05",
    # cross-session linear/nonlinear (session grandmean → ref session).
    "xses": "-rigid -cost lpa -source_automask -autoweight -smallrange -final wsinc5",
    "xses_nl": "-metric lpa -cc_radius 4 -update_var 2 -final_interp wsinc5 "
    "-iters 1000x1000x1000 -smooth 0x0x0 -conv_thresh 1e-05",
    # anat linear (cross-modal EPI→anat) — lpc with EPI masked to brain.
    "anat": "-rigid -cost lpc -source_automask -autoweight -interp cubic -cmass -fast -final wsinc5",
    # anat nonlinear (ffs_segment). Two tunings: the FS-derived TPM has 8
    # hard-edge classes, so it needs its own ngaus/cleanup; an SPM-style TPM
    # uses the plain one. Which applies is set by -suma/-tpm, not by the user.
    "segment": "-niter 2000 -samp 1.5",
    "segment_fstpm": "-ngaus 1 1 1 1 1 2 3 4 -cleanup 0 -samp 1.5",
    # final compose+resample (ffs_nwarp).
    "nwarp": "-interp wsinc5 -no_neg",
    # ROMEO temporal phase unwrapping (-phase_proc). "-t epi" is the single-echo
    # EPI timeseries mode (identical echo times); append a TE in ms if you want
    # ROMEO's B0 output. NOT an ffs tool — romeo must be on $PATH.
    "unwrap": "-t epi -v",
}

# Stages exposed as ``-<key>_opts`` on the CLI, in pipeline order. Anything in
# DEFAULT_OPTS should be here — the emitter reads DEFAULT_OPTS, so an unlisted
# key is a stage the user silently cannot tune. The grid utilities
# (autobox/resample/automask/3dmath) are deliberately absent: their flags define
# the output grid the rest of the chain assumes, so they are not free knobs.
STAGE_OPT_KEYS: tuple[str, ...] = tuple(DEFAULT_OPTS)

# Working / final / GLM output extensions. Intermediates default to zstd (read
# many times, big); final timeseries to gzip (portability); GLM to gzip.
# Compression suffix appended AFTER ".nii" in filenames (templates are "...nii$FMT"):
# ".zst" → .nii.zst, ".gz" → .nii.gz, "" → .nii.
DEFAULT_FMT = ".zst"
DEFAULT_FINAL_FMT = ".gz"
DEFAULT_GLM_FMT = ".gz"

DEFAULT_DEVICE = "cuda"
DEFAULT_FINAL_INTERP = "wsinc5"


# ---------------------------------------------------------------------------
# GLM nuisance regressors, by name. `-glm_ortvec motion motion_deriv locomoco`
# selects from here; each entry becomes one [[nuisance]] block in the design
# TOML. Adding a source (CSF PCs, physio, censor-derived spikes) is one entry
# plus whatever stage writes the .1D — nothing in the emitter changes.
#
#   pattern    glob over per-run .1D files, {task} substituted. The run index is
#              inferred from the filename (see cli_utils._RUN_INDEX_PATTERNS).
#   transform  per-run transform applied before the block enters the design
#              (see cli_utils.NUISANCE_TRANSFORMS); "deriv" = 1d_tool.py's.
#   requires   an Options field that must be truthy, else the entry is skipped
#              with a warning (asking for locomoco PCs without locomoco).
# ---------------------------------------------------------------------------
GLM_ORTVEC: dict[str, dict[str, str]] = {
    "motion": {
        "pattern": "stage02.moco.*task-{task}*.motion.1D",
        "transform": "none",
        "note": "6 rigid-body parameters from ffs_moco, per run",
    },
    "motion_deriv": {
        "pattern": "stage02.moco.*task-{task}*.motion.1D",
        "transform": "deriv",
        "note": "temporal derivative of the same file (1d_tool.py -derivative)",
    },
    "locomoco": {
        "pattern": "stage03.nlmoco.*task-{task}*_locomoco_pcs.1D",
        "transform": "none",
        "requires": "locomoco",
        "note": "top-N temporal PCs of the residual nonlinear motion warp",
    },
}
# What a bare `-glm_ortvec` (or a recipe asking for nuisance) selects. Entries
# whose `requires` is unmet are dropped silently here — this is the default set,
# not an explicit request.
DEFAULT_GLM_ORTVEC = ("motion", "motion_deriv", "locomoco")


# ---------------------------------------------------------------------------
# Recipes: named bundles of defaults so users hit the ground running. Each maps
# to a dict of Options-field overrides; explicit CLI flags still win over these.
# A recipe only sets the *baseline* — future versions may tune more params.
# ---------------------------------------------------------------------------
RECIPES: dict[str, dict] = {
    # moco (to first-run/sbref) + GLM, in EPI space. Fast, minimal inputs.
    "bare_bones": {
        "go_to_anat": False,
        "distortion": False,
        "slicetiming_method": "none",
        "run_glm": True,
    },
    # slice timing (if present) + distortion + moco + xrun + linear anat + GLM.
    "simple": {
        "go_to_anat": True,
        "distortion": True,
        "slicetiming_method": "integrate",
        "run_glm": True,
    },
    # + nonlinear cross-run refinement.
    "simple_nonlin": {
        "go_to_anat": True,
        "distortion": True,
        "slicetiming_method": "integrate",
        "xrun_nonlin": True,
        "run_glm": True,
    },
    # + nonlinear anat (ffs_segment) — needs an anat and a subject TPM.
    "complete": {
        "go_to_anat": True,
        "distortion": True,
        "slicetiming_method": "integrate",
        "xrun_nonlin": True,
        "anat_nonlin": True,
        "run_glm": True,
    },
    # everything + locomoco (residual NL motion); its warp-PCs join motion and
    # motion-derivative as GLM nuisance (see GLM_ORTVEC).
    "extreme": {
        "go_to_anat": True,
        "distortion": True,
        "slicetiming_method": "integrate",
        "xrun_nonlin": True,
        "anat_nonlin": True,
        "locomoco": True,
        "glm_ortvec": list(DEFAULT_GLM_ORTVEC),
        "run_glm": True,
    },
}
# simple_linear is a spelling alias of simple.
RECIPES["simple_linear"] = RECIPES["simple"]

# One-line, stage-wise summary of each recipe (shown in --help). Order matters.
RECIPE_SUMMARY: dict[str, str] = {
    "bare_bones": "moco (→sbref/first vol) + GLM, in EPI space. No slice-timing, distortion, or anat. Fast.",
    "simple": "slice-timing (if present) + distortion (if fmaps) + moco + cross-run align + LINEAR anat + GLM.",
    "simple_linear": "alias of simple.",
    "simple_nonlin": "= simple, plus nonlinear cross-run refinement.",
    "complete": "= simple_nonlin, plus NONLINEAR anat (ffs_segment; needs -anat/-suma and a TPM, or -suma).",
    "extreme": "= complete, plus locomoco (residual NL motion); GLM nuisance = motion + "
    "motion deriv + locomoco warp-PCs.",
}


def recipe_help() -> str:
    """Formatted recipe list for the CLI epilog."""
    lines = ["recipes (stage defaults; any explicit flag overrides the recipe):"]
    lines += [f"  {name:<14} {RECIPE_SUMMARY[name]}" for name in RECIPE_SUMMARY]
    return "\n".join(lines)
