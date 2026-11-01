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
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from fastfuncstuff.autoproc import config, optcheck
from fastfuncstuff.autoproc.bids import (
    BoldRun,
    find_events,
    pair_undetermined,
    parse_entities,
    scan_subject,
)
from fastfuncstuff.autoproc.emit import write_script
from fastfuncstuff.autoproc.family import family_dir
from fastfuncstuff.autoproc.glm import STIMULI_DIR, write_design_specs
from fastfuncstuff.autoproc.plan import Options, build_plan
from fastfuncstuff.cli_help import FfsArgumentParser, FfsHelpFormatter, suggest
from fastfuncstuff.design.spec import DEFAULT_DROP_TRIAL_TYPES, DEFAULT_EVENT_COLUMNS
from fastfuncstuff.processing import censor as censor_defaults

# The tuned-preset recipe -do_mni runs under. Named once: the op string says it
# (`-type MNI_T1a`) and the family→engine resolution has to agree with it.
MNI_RECIPE = "MNI_T1a"


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


def _device_spec(value: str) -> str:
    """Validate a device string for the emitted ``DEVICE`` variable.

    Only the spelling is checked, never availability: the script is routinely
    generated on one machine and run on another (a cluster node with the GPU).
    Accepts what every ffs CLI's ``parse_device_arg`` accepts — cuda | mps | cpu
    | auto, optionally ``,N`` (cuda device index / cpu thread count)."""
    v = value.strip().lower()
    backend, _, suffix = v.partition(",")
    if backend not in ("cuda", "mps", "cpu", "auto"):
        raise argparse.ArgumentTypeError(
            f"unrecognized device {value!r}; use cuda, mps, cpu, or auto (optionally cuda,N / cpu,N)"
        )
    if suffix and not suffix.isdigit():
        raise argparse.ArgumentTypeError(
            f"device {value!r}: the part after ',' must be a number "
            "(cuda,N = GPU index; cpu,N = threads)"
        )
    return v


def build_parser() -> argparse.ArgumentParser:
    p = FfsArgumentParser(
        prog="ffs_autoproc",
        description="Generate a readable, resumable ffs preprocessing script from a BIDS dataset.",
        epilog=config.recipe_help(),
        formatter_class=FfsHelpFormatter,
    )

    g = p.add_argument_group("inputs & scope")
    g.add_argument("-bids_dir", "-bids-dir", required=True, help="validated BIDS root")
    g.add_argument("-subject", required=True, help="subject id (e.g. 001 or sub-001)")
    g.add_argument("-session", nargs="+", help="restrict to these sessions (labels or ses-*)")
    g.add_argument("-task", nargs="+", help="restrict to these tasks")
    g.add_argument("-out", help="output script path (default: proc_sub-<id>.sh)")
    g.add_argument("-work_dir", "-work-dir", help="working dir baked into the script (OUT)")
    g.add_argument(
        "-exec",
        action="store_true",
        help="run the generated script immediately after writing it: "
        "bash OUT 2>&1 | tee OUT.log (both streamed live and saved). "
        "Exits with the script's own status.",
    )

    g = p.add_argument_group("recipe (preset defaults; see list below --help)")
    g.add_argument(
        "-recipe",
        choices=sorted(config.RECIPES),
        help="bare_bones | simple[_linear] | simple_nonlin | complete | extreme",
    )

    g = p.add_argument_group("anatomical & reference space")
    g.add_argument(
        "-anat",
        nargs="+",
        metavar="FILE",
        help="T1w to align to; it is skull-stripped in-script unless you add "
        "-anat_skull no (e.g. for a SUMA brain.nii.gz). Several files are aligned to the first and averaged "
        "(they must be the same contrast). Omit it and every T1w in the in-scope "
        "sessions is used the same way.",
    )
    g.add_argument("-suma", help="FreeSurfer SUMA dir; brain.nii.gz + builds a TPM from aseg.auto")
    g.add_argument(
        "-anat_skull",
        "-anat-skull",
        choices=["auto", "yes", "no"],
        default="auto",
        help="does the anat still have a skull? auto (default) = yes, unless it is "
        "-suma's brain.nii.gz. yes = strip it in-script with FreeSurfer's "
        "mri_synthstrip (FreeSurfer >= 8 required on $PATH), so a T1w straight off "
        "the scanner can be the alignment target with no recon-all. no = it is "
        "already stripped (e.g. -anat a SUMA brain.nii.gz). Both images are "
        "written: stage09.anat_head (as given) and stage09.anat_brain (stripped, "
        "what everything aligns to). This gives no tissue priors, so -anat_nonlin "
        "still needs -tpm.",
    )
    g.add_argument(
        "-presurf",
        choices=["auto", "no"],
        default="auto",
        help="auto (default): when the anat comes from the BIDS tree and is an MP2RAGE "
        "acq-uni T1w with its acq-inv2 (and acq-inv1) beside it, run ffs_util_presurf "
        "(MPRAGEise UNI by the bias-corrected INV2) and skull-strip THAT with "
        "mri_synthstrip, instead of aligning to the noisy-background UNI. Needs a "
        "tissue template (-presurf_tpm, -tpm or $FFS_TPM). no = UNI as found.",
    )
    g.add_argument(
        "-presurf_denoise",
        "-presurf-denoise",
        choices=["off", "reg", "mprage"],
        default="off",
        help="ffs_util_presurf -denoise (needs acq-inv1): jointly denoise INV1/INV2 "
        "first. reg = the O'Brien-regularised UNI, mprage = the rebuilt UNI "
        "MPRAGEised. off (default) = presurfer as published.",
    )
    g.add_argument(
        "-presurf_tpm",
        "-presurf-tpm",
        metavar="FILE",
        help="6-class tissue template for the presurf segmentations (SPM's tpm/TPM.nii); "
        "default -spm_tpm, else -tpm, else $FFS_TPM",
    )
    g.add_argument(
        "-spm_tpm",
        "-spm-tpm",
        metavar="FILE",
        help="SPM-style GROUP tissue template (SPM's tpm/TPM.nii), used to derive the "
        "subject priors -anat_nonlin needs when there is no -suma: the anat is "
        "segmented against it and its posteriors become the subject TPM. Default "
        "$FFS_TPM. (-tpm is the opposite end — a subject TPM already in anat space.)",
    )
    g.add_argument(
        "-anat_t2",
        "-anat-t2",
        metavar="FILE",
        help="T2w to segment alongside the T1w when building the subject TPM "
        "(-spm_tpm): it is rigidly aligned onto the T1 grid first, then both go in "
        "as channels, which is what separates bone from CSF and dura from grey. "
        "Default: the matching T2w in the BIDS anat/ dir, if there is one. Pass "
        "'none' to segment the T1w alone.",
    )
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
        help="nonlinear anat warp; how it is found is -anat_nonlin_method",
    )
    g.add_argument(
        "-anat_nonlin_method",
        "-anat-nonlin-method",
        choices=["segment", "register"],
        default="segment",
        help="segment (default): ffs_segment's PE-axis warp, driven by a tissue "
        "template (needs -tpm or -suma). register: the anat anchor registered to the "
        "anat by the -anat_nl_backend engine (default -nl_backend), in the anchor's "
        "own frame and along the PE axis only, so it can only undo residual "
        "distortion. For an anat with no usable tissue contrast for a template, "
        "e.g. a thin-slab multi-echo GRE.",
    )
    g.add_argument(
        "-ref_anat",
        "-ref-anat",
        metavar="FILE",
        help="an anat from the EPI's OWN scanner (same gradient coils, ideally the same "
        "session), e.g. the 7T MP2RAGE when -anat is a 3T T1w. The EPI aligns to it "
        "instead of to -anat (anat_lin, and anat_nl with the ref's own posteriors as "
        "priors), so gradient nonlinearity the two share needs no correcting. Then the "
        "ref is taken onto -anat, rigid + a full 3-D warp (-ref_anat_nonlin_method), "
        "and -anat stays the final space: grid, masks, tissue, surfaces and -do_mni are "
        "all unchanged. Both links ride the head of every chain, so the data still makes "
        "one trip through an interpolator. Stripped with mri_synthstrip when -anat is. "
        "The shared-gradient premise holds when both were (or were not) gradient-"
        "corrected alike -- check ImageType for DIS2D/DIS3D/ND -- and anat_nl, PE-axis "
        "only, cannot absorb a through-slice difference.",
    )
    g.add_argument(
        "-ref_anat_inv2",
        "-ref-anat-inv2",
        metavar="FILE",
        help="-ref_anat is an MP2RAGE UNI: its INV2, for ffs_util_presurf to MPRAGEise "
        "it (as -presurf does for -anat). Default: the acq-inv2 filed beside an acq-uni "
        "-ref_anat. -presurf no turns it off.",
    )
    g.add_argument(
        "-ref_anat_inv1",
        "-ref-anat-inv1",
        metavar="FILE",
        help="...and its INV1 (only read by -presurf_denoise). Default: the acq-inv1 beside it.",
    )
    g.add_argument(
        "-ref_anat_nonlin_method",
        "-ref-anat-nonlin-method",
        choices=["segment", "register"],
        default="segment",
        help="how the -ref_anat → -anat warp is found. segment (default): ffs_segment of "
        "the ref head against -anat's subject TPM, free on all three axes; tissue "
        "priors hold up where a 7T ref's intensities do not (ventral B1 dropout). "
        "register: the -ref_anat_nl_backend engine (default -nl_backend), ref brain to "
        "anat brain on intensity, in the ref's frame.",
    )
    g.add_argument(
        "-surface_target",
        "-surface-target",
        nargs="+",
        default=None,
        metavar="MESH",
        help="also sample every run onto a cortical mesh in the SAME single interpolation "
        "(ffs_nwarp -surf): 'native' for the subject's own vertices, or a FreeSurfer-format "
        "template on fsaverage's sphere by path or by name (beside the subject or in "
        "$SUBJECTS_DIR, e.g. "
        "onavg-ico64) for group-ready output; several (native onavg-ico64) share one pass. "
        "The chain is each run's own up to the anat (never the MNI links), read at "
        "footprint points on the equivolume mid-surface. Writes "
        "stage10s.surf.<run>.<SPACE>.?h.func.gii with its mask and placed geometry. "
        "Needs the FreeSurfer subject: -fs_subject, "
        "or the folder above -suma.",
    )
    g.add_argument(
        "-surface_qc",
        "-surface-qc",
        action="store_true",
        help="with -surface_target: also write ffs_nwarp -surf_qc's maps per run -- "
        "footprint reads per native EPI voxel (stage10s.surf.<run>.<SPACE>.samples), "
        "effective voxel volume and blur FWHM per vertex. About one extra pass over 64 "
        "frames per run.",
    )
    g.add_argument(
        "-surface_glm",
        "-surface-glm",
        action="store_true",
        help="with -surface_target: stage12s fits every task's model on every surface "
        "target and hemisphere (ffs_reml on the .func.gii series). Shares stage12's "
        "design spec, censoring, TR, -glm_drop_* and -glm_opts; inside the cross-run "
        "surface mask. Buckets: stage12s[.TAG].stats-reml.task-T.<SPACE>.?h.func.gii.",
    )
    g.add_argument(
        "-surface_glm_blur",
        "-surface-glm-blur",
        type=float,
        default=None,
        metavar="FWHM",
        help="surface GLM smoothing BY this FWHM (mm): heat diffusion along the cortex, "
        "the mask edge a wall (ffs_reml -do_blur on vertices). Independent of -glm_blur.",
    )
    g.add_argument(
        "-surface_glm_blur_to",
        "-surface-glm-blur-to",
        type=float,
        default=None,
        metavar="FWHM",
        help="surface GLM smoothing TO this FWHM (ffs_reml -blur_to_fwhm, SurfSmooth "
        "-target_fwhm). Exclusive with -surface_glm_blur.",
    )
    g.add_argument(
        "-surface_glm_blur_local",
        action="store_true",
        help="with -surface_glm_blur_to: spatially varying heat diffusion that stops "
        "wherever the local smoothness has reached the target (ffs_reml -blur_to_local). "
        "Buckets: stage12s.blurto<F>local.*. Pair with -clustsim_null resid.",
    )
    g.add_argument(
        "-surface_glm_blur_tsnr",
        type=float,
        default=None,
        metavar="TSNR",
        help="surface GLM smoothing per vertex until its residual tSNR reaches TSNR "
        "(ffs_reml -blur_to_tsnr), up to -surface_glm_blur_max_fwhm. Buckets: "
        "stage12s.tsnr<T>.*. Exclusive with -surface_glm_blur / -surface_glm_blur_to.",
    )
    g.add_argument(
        "-surface_glm_blur_max_fwhm",
        type=float,
        default=None,
        metavar="MM",
        help="-surface_glm_blur_tsnr: no vertex smoothed past this local FWHM (default 8).",
    )
    g.add_argument(
        "-surface_glm_blur_nbhd",
        type=float,
        default=None,
        metavar="MM",
        help="-surface_glm_blur_local / _tsnr: neighbourhood (mm along the cortex) of "
        "each local estimate (default 8).",
    )
    g.add_argument(
        "-surface_clustsim",
        "-surface-clustsim",
        action="store_true",
        help="SurfClustSim cluster-AREA tables (mm^2) for each surface bucket, from that "
        "fit's residual ACF on its own mesh and mask (ffs_reml -clustsim on vertices); "
        "the residual ACF is saved beside it (-save_acf). The viewer reads the tables.",
    )
    g.add_argument(
        "-surface_clustsim_niter",
        "-surface-clustsim-niter",
        type=int,
        default=None,
        metavar="N",
        help="SurfClustSim iterations (default ffs_reml's).",
    )
    g.add_argument(
        "-surface_glm_opts",
        "-surface-glm-opts",
        default="",
        metavar="OPTS",
        help="extra ffs_reml flags for the surface fits only (-glm_opts applies to both).",
    )
    g.add_argument(
        "-fs_subject",
        "-fs-subject",
        default=None,
        metavar="DIR",
        help="FreeSurfer subject (a folder with surf/, or a name in $SUBJECTS_DIR) for "
        "-surface_target; defaults to the parent of -suma. Baked into the script as an "
        "absolute path, as are the template meshes. Its surfaces must share scanner space with the anat.",
    )
    g.add_argument(
        "-do_mni",
        "-do-mni",
        action="store_true",
        help="also take the output to an MNI template: an affine + nonlinear anat→MNI "
        "pair estimated on the skull-stripped anat and applied at the head of every "
        "run's warp chain, so the data still makes ONE trip through an interpolator. "
        "The final grid, mask and underlay all move to MNI with it.",
    )
    g.add_argument(
        "-mni_use",
        "-mni-use",
        # Families first, then the individual engines — "qwarp" and "formwarp" are
        # both, so dict.fromkeys keeps each spelling once.
        choices=list(dict.fromkeys(["optiwarp", *config.nl_backends()])),
        default="optiwarp",
        help="engine for the nonlinear half of -do_mni. A family name picks the "
        "member the -type recipe prefers (optiwarp → optiwarp_hs for MNI_T1a). "
        "(default: optiwarp)",
    )
    g.add_argument(
        "-mni_template",
        "-mni-template",
        metavar="FILE",
        help="MNI template to align to; sub-brick 0 must be the skull-stripped "
        "volume. Default: MNI152_2009_template_SSW.nii.gz beside the afni on $PATH.",
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
        choices=["auto", "grandmean", "sbmean", "ref_fmap", "mean_fmap"],
        default=None,
        help="Override -ref_image for the anat step alone (same vocabulary). All "
        "choices share the reference-fmap grid: auto (sbmean when the dataset has "
        "SBRefs, else grandmean) | grandmean (best SNR, only option w/o fieldmaps "
        "or SBRefs) | sbmean (the same average built from "
        "every run's SBRef: single-band contrast, one interpolation instead of "
        "two) | ref_fmap (reference fieldmap's undistorted mean; sharpest, defines "
        "the space) | mean_fmap (ref_fmap averaged with the other groups' aligned "
        "means). Default: whatever -ref_image is, else auto.",
    )
    g.add_argument(
        "-anat_nonlin_input",
        "-anat-nonlin-input",
        choices=["auto", "grandmean", "sbmean", "ref_fmap", "mean_fmap", "blipfor", "blip_pair"],
        default="auto",
        help="ffs_segment input: auto (= -anat_source auto: sbmean with SBRefs, else "
        "grandmean) | grandmean (works w/o fieldmap) | sbmean | ref_fmap | "
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
        "-ref_file_anat",
        "-ref-file-anat",
        help="with -ref_file: the anat the reference's transforms land in (copied in; "
        "ffs_segment tpm-space anat + QC, and the output grid)",
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
        "-cut_task_vols",
        "-cut-task-vols",
        action="append",
        nargs=2,
        default=None,
        metavar=("TASK", "N"),
        help="keep only the first N volumes of every run of TASK; runs already at or "
        "below N are left alone. For runs that ended a few volumes apart (390, 395, "
        "402 -> -cut_task_vols TASK 390). The cut is a sub-brick selector on the "
        "source, so every stage (NORDIC included) sees the cut series, and the GLM "
        "drops events past the new end. Repeat for more tasks. A cut task is taken "
        "to have NO noise volumes (a run that needs cutting was stopped before the "
        "noise scans): -noise_vols still applies to every other task, and NORDIC "
        "sets the cut task's threshold from its g-factor normalisation alone, as "
        "MATLAB NORDIC does without noise scans.",
    )
    g.add_argument(
        "-family",
        action="append",
        nargs=2,
        default=None,
        metavar=("TASK", "LABEL"),
        help="put every run of TASK in the acquisition family LABEL. Families are "
        "detected from voxel size, matrix, PE direction, readout time and echo count, "
        "and that detection is printed; use this when it groups wrongly. Tasks given "
        "the same LABEL become one family even if their acquisitions differ. Repeat "
        "for more tasks.",
    )
    g.add_argument(
        "-family_ref",
        default=None,
        metavar="MODE|LABEL",
        help="with several acquisition families, how they reach the anat. anat "
        "(default): no root -- every family aligns to the anat on its own, with its own "
        "anat_lin, anat_nl and output grid, and they meet at the anat, whose chain "
        "(-ref_anat, -do_mni) they share. Or name a root the others align to EPI-to-EPI "
        "(xfam): highest_res (the low-res family moves toward the sharper target), "
        "lowest_res, coverage (largest field of view), least_distorted (smallest "
        "readout time x PE voxel size), or a family LABEL. A child links to the root "
        "only when their slabs overlap enough (-xfam_min_overlap); otherwise it too "
        "aligns to the anat on its own.",
    )
    g.add_argument(
        "-family_link",
        action="append",
        nargs=2,
        default=None,
        metavar=("LABEL", "root|anat"),
        help="override the overlap decision for one family: root = align it to the "
        "root family whatever the overlap; anat = align it to the anat on its own. "
        "Repeat for more families.",
    )
    g.add_argument(
        "-xfam_min_overlap",
        type=float,
        default=None,
        metavar="FRAC",
        help="least fraction of a family's field of view the root must also cover "
        "(from the headers) for the two to be aligned to each other directly "
        "(default 0.5). The overlap must also span 20 mm along the family's slab "
        "normal: a thinner shared sliver cannot pin rotation.",
    )
    g.add_argument(
        "-phase_proc",
        "-phase-proc",
        action="store_true",
        help="carry the phase timeseries through the pipeline: unwrap it up front "
        "(after NORDIC if NORDIC runs), then ride it along the magnitude's warp chain at "
        "the final resample → stage10.final.*.part-phase.*, unwrapped radians in the same "
        "space as the magnitude (ffs_phase_unwrap: ROMEO once, every frame anchored to it "
        "on -device). Requires a part-phase bold "
        "for every run. The unwrapped phase then has per-slice field wobble "
        "(breathing, B0 drift) removed in native space by ffs_phasedrift — see "
        "-no_phase_drift.",
    )
    g.add_argument(
        "-medic",
        action="store_true",
        help="with -phase_proc and multi-echo runs: frame-wise distortion correction "
        "(MEDIC) from the per-frame field map the unwrap stage already computes "
        "(ffs_medic -fieldmap, SVD-denoised) -> stage01.medic.*: the undistorted echoes "
        "moco reads, and a time-varying warp rightmost in every echo's chain, so stage10 "
        "still resamples once from the raw series. It IS the distortion correction: "
        "fieldmaps are not used, the SBRef is not the moco base (it has no per-frame "
        "field), and MEDIC owns the in-plane axis: 3-D locomoco solves the partition "
        "shift only, and 2-D skips locomoco (see -medic_inplane_locomoco). Needs "
        "TotalReadoutTime and PhaseEncodingDirection in every run's sidecar.",
    )
    g.add_argument(
        "-medic_inplane_locomoco",
        action="store_true",
        help="with -medic and -locomoco: keep locomoco's in-plane solve on top of MEDIC "
        "(3-D: in-plane + partition; 2-D: in-plane) instead of leaving in-plane to MEDIC.",
    )
    g.add_argument(
        "-no_phase_drift",
        "-no-phase-drift",
        action="store_true",
        help="with -phase_proc, skip ffs_phasedrift. By default each slice's "
        "dynamic field offset and in-plane gradient are removed per volume right "
        "after the unwrap, while the slices are still the acquired ones; on 7T 0.8 mm "
        "data that cut phase noise ~3x and is what lets phase regression see the "
        "vessel phase at all. Skip it only to reproduce older output.",
    )
    g.add_argument(
        "-phasereg",
        action="store_true",
        help="with -phase_proc, run ffs_phasereg on each task's runs that have phase, "
        "after stage10, into phasereg_NN/ (task-X_corrected_run-NN, one per run in run "
        "order; -keep_drift, so the GLM models drift as usual). Each setting "
        "(-phasereg_veins, -phasereg_opts, -tissue, the commands) gets its own numbered "
        "folder with a settings.txt saying what was run and a history.txt of when; an "
        "identical rerun resumes in its folder, other settings add the next number. "
        "With a GLM, each task's model is also fit on the corrected runs: "
        "phasereg_NN/stats-*.task-X. stage10 is untouched, so the fits compare like "
        "for like. Default fit: Deming damped by ODR's factor (-shrink slope), no cap: "
        "stable, an even ~10%% suppression.",
    )
    g.add_argument(
        "-phasereg_me",
        choices=["per_echo", "field"],
        default="per_echo",
        help="multi-echo phase regression. per_echo (default): each echo against its own\n"
        "phase, then the corrected echoes are combined. field (EXPERIMENTAL): the optimal\n"
        "combination against ONE frequency-shift series fitted from every echo's phase\n"
        "(phase is 2*pi*df*TE at every echo, so df is inverse-variance weighted across\n"
        "them: a much less noisy regressor, and one regression per voxel instead of one\n"
        "per echo). Part of the folder's settings.",
    )
    g.add_argument(
        "-phasereg_opts",
        default="",
        metavar="'FLAGS'",
        help='extra ffs_phasereg flags (quoted), e.g. "-shrink odr" (phaseprep\'s '
        'published output) or "-phase_filter sgf"; part of the folder\'s settings',
    )
    g.add_argument(
        "-phasereg_veins",
        choices=["off", "auto"],
        default="off",
        help="off (default): plain phase regression, each voxel its own phase, Deming "
        "damped by ODR's factor (-shrink slope): stable, an even ~10%% suppression. auto: "
        "when anat tissue maps exist (-tissue, -presurf's own, or -suma's aseg when the "
        "anat is SUMA's brain), ffs_veinmask's local vein map (dark ridges: sulcal "
        "plates, vessels, the pial edge) is where a voxel may borrow a neighbour's "
        "phase (-spr), with an OLS slope there (-spr_regression ols) and donors chosen "
        "on task-residual coupling when every run has events. Conservative outside "
        "plausible veins, firmer inside. On ds003427 (2026-10-14) it was the one "
        "literature rule that reduced the pial GE/SE excess, modestly, at little "
        "tissue cost; still to be checked on event designs.",
    )
    g.add_argument(
        "-tissue",
        nargs="+",
        default=None,
        metavar="FILE",
        help="GM, WM, CSF fractions of the anat (one 4-D file in SPM c1/c2/c3 order, or "
        "three files), in the anat's scanner space, for -phasereg_veins. Default with "
        "-presurf: presurfer's own UNI classes; with -suma: GM/WM/CSF from aseg.auto.",
    )
    g.add_argument(
        "-hrftdv",
        action="store_true",
        help="temporal deveining (EXPERIMENTAL), after stage10, into hrftdv_NN/: "
        "ffs_hrftdv finds an early and a late HRF per cortical column of the "
        "FreeSurfer subject (-fs_subject or -suma; its surfaces must sit on the anat) "
        "from each task's design TOML, then that model is fit with them "
        "(hrftdv_NN/stats-*.task-X, NAME_early / NAME_late per contrast), so the "
        "late column takes the delayed venous part. A few minutes per task on the "
        "GPU, plus one more GLM. Runs with the GLM (FFS_RUN_GLM). Each setting "
        "(-hrftdv_model, -hrftdv_opts, the design TOMLs) gets its own numbered folder, "
        "as -phasereg's do. Beat one HRF on held-out R2 in 69%% of voxels on an event "
        "design, 34%% on 10 s blocks.",
    )
    g.add_argument(
        "-hrftdv_model",
        choices=["pair", "pooled_late"],
        default="pair",
        help="pair (default): two columns per condition, {cond}_early and {cond}_late. "
        "pooled_late: one early column per condition and one late column per voxel "
        "(its column's late curve x the column's condition pattern, via -dsort): the "
        "early and late betas fight less (VIF ~16 -> ~4 on an event design), but the "
        "column's shared selectivity is called late.",
    )
    g.add_argument(
        "-hrftdv_opts",
        default="",
        metavar="'FLAGS'",
        help='extra ffs_hrftdv flags (quoted), e.g. "-patch_mm 3 -max_lag 3"; part of the '
        "folder's settings",
    )
    g.add_argument(
        "-nordic_save_resid",
        "-nordic-save-resid",
        action="store_true",
        help="save the NORDIC residual map",
    )
    g.add_argument(
        "-nordic_task_rescue",
        "-nordic-task-rescue",
        nargs="?",
        const="ica",
        default=None,
        metavar="MODE",
        help="ffs_nordic -task_rescue MODE (ica).\n"
        "Off by default. -nordic_task_qc writes the task-leak REPORT — how much of the\n"
        "design survives in what the denoising threw away, whether it lands on\n"
        "responding tissue, and which removed components follow the task. This ACTS on\n"
        "it: the flagged components go back into the output (the report is written\n"
        "too, since rescue runs on it). Note each restored component hands back a\n"
        "degree of freedom (ffs_reml -adjust_dof).",
    )
    g.add_argument(
        "-no_nordic_dof_adjust",
        "-no-nordic-dof-adjust",
        dest="nordic_dof_adjust",
        action="store_false",
        default=True,
        help="Do NOT correct the GLM's statistics for the degrees of freedom NORDIC\n"
        "removed. On by default when NORDIC runs: stage00 saves the per-voxel component\n"
        "count, stage10 carries each run's map through that run's own warp chain, and\n"
        "stage12 sums them per task and passes the sum to ffs_reml -adjust_dof, which\n"
        "converts every t/F to a z at the corrected dof.\n"
        "This is what makes a NORDIC run comparable to a non-NORDIC one. Without it the\n"
        "denoised run's p-values are labelled with dof the data no longer has, so it\n"
        "wins the comparison partly by accounting.",
    )
    g.add_argument(
        "-nordic_task_qc",
        action="store_true",
        help="Pass each run's events to ffs_nordic with -task_qc: the stage00\n"
        "task-leak diagnostic. Off by default; it measures and changes nothing, and\n"
        "costs a per-patch test, three task fits and an ICA scan per run.",
    )
    # Accepted so older commands still parse; the QC is now off unless asked for.
    g.add_argument("-no_nordic_events", action="store_true", help=argparse.SUPPRESS)

    g = p.add_argument_group("slice timing & motion")
    g.add_argument(
        "-acq",
        choices=["auto", "2d", "3d"],
        default="auto",
        help="2-D multi-slice or 3-D EPI. auto (default): the sidecars' "
        "MRAcquisitionType if present, else 2d when any run has SliceTiming (or "
        "-slicetiming is given), else 3d — no slice information means no acquired "
        "slices to rely on. The decision is printed at generation time; pass 2d "
        "if your 2-D data merely lost SliceTiming in conversion.",
    )
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
        "-tzero",
        type=float,
        default=None,
        metavar="SECONDS",
        help="within-TR time slice timing aligns every slice to (default 0, the first "
        "slice). Both -slicetiming_method paths use it, and the GLM stage gets it as "
        "-microtime_offset so the model is sampled where the data now are.",
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
        "-motsim",
        nargs="?",
        const="both,12",
        default=None,
        metavar="MODE[,N]",
        help="motion-simulation nuisance regressors, estimated inside stage02\n"
        "(ffs_moco -motsim) and written as {stem}.motsim.1D per run. Models the\n"
        "SIGNAL changes the motion caused rather than the parameters themselves.\n"
        "MODE is forward|backward|both; N is a PC count or a fraction in (0,1)\n"
        "asking for that much of the simulated series' variance. Bare -motsim is\n"
        "both,12 (Patriat 2017's 12Both); forward,12 skips the second registration\n"
        "pass and the paper found it no worse. Selecting it for the GLM is\n"
        "-glm_ortvec motsim; in a DEFAULT nuisance set it replaces motion and\n"
        "motion_deriv, which its PCs are derived from.",
    )
    g.add_argument(
        "-censor_motion",
        nargs="?",
        const=censor_defaults.DEFAULT_MOTION_LIMIT,
        type=float,
        default=None,
        metavar="L",
        help="censor TRs whose motion enorm exceeds L, plus the TR before each\n"
        "(afni_proc -regress_censor_motion, via ffs_moco in stage02). Bare flag =\n"
        f"{censor_defaults.DEFAULT_MOTION_LIMIT}. Off by default. Censored TRs are dropped from the GLM\n"
        "(ffs_reml -censor), with the noise model stepping across each gap.",
    )
    g.add_argument(
        "-censor_outliers",
        nargs="?",
        const=censor_defaults.DEFAULT_OUTLIER_LIMIT,
        type=float,
        default=None,
        metavar="F",
        help="censor TRs where more than a fraction F of automask voxels are\n"
        "outliers (afni_proc -regress_censor_outliers, 3dToutcount's rule). Bare\n"
        f"flag = {censor_defaults.DEFAULT_OUTLIER_LIMIT}. Off by default. Counted on the series moco reads,\n"
        "which is the NORDIC output when NORDIC runs (AFNI counts raw data).",
    )
    g.add_argument(
        "-locomoco",
        action="store_const",
        const=True,
        default=None,
        help="residual PE-axis nonlinear motion",
    )
    g.add_argument(
        "-locomoco_detask",
        "-locomoco-detask",
        nargs="?",
        const="field",
        default=None,
        metavar="MODE",
        help="ffs_locomoco -detask MODE (field | filter[:N] | fit[:D] | ica[:N]).\n"
        "Off by default. -locomoco_task_qc writes the task-coupling REPORT; this\n"
        "ACTS on it, so read that report first — see `ffs_locomoco -help` for which\n"
        "mode suits your design. filter/fit cut the task before the estimate and run\n"
        "no QC unless -locomoco_task_qc is given too.",
    )

    g.add_argument(
        "-locomoco_phase",
        "-locomoco-phase",
        action="store_true",
        help="EXPERIMENTAL. Give ffs_locomoco each echo's unwrapped phase as well\n"
        "(the joint magnitude+phase solve for the TE-scaled partition shift). Off by\n"
        "default: on the hardest test run (1.25 mm, 3 echoes) magnitude alone left\n"
        "2-4x LESS partition wiggle at the later echo, and phase put flow where the\n"
        "magnitude has none (the head's edge). Needs -locomoco, -phase_proc and 3-D\n"
        "multi-echo EPI; with -medic the phase is undistorted along with it.",
    )
    g.add_argument(
        "-locomoco_task_qc",
        action="store_true",
        help="Pass each run's events to ffs_locomoco with -task_qc: the stage03\n"
        "task-coupling diagnostic. Off by default; it measures and changes nothing,\n"
        "and costs design fits on the data and every field axis plus an ICA sweep\n"
        "per run.",
    )
    g.add_argument("-no_locomoco_events", action="store_true", help=argparse.SUPPRESS)

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
        "-fmap_kind",
        "-fmap-kind",
        choices=["auto", "pepolar", "b0"],
        default="auto",
        help="which fieldmap flavour to use from a fmap/ that offers both: "
        "pepolar (reverse-PE pair, ffs_blipflip) | b0 (dual-echo GRE, "
        "ffs_util_b0fmap) | auto = pepolar when present, else b0",
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
        "-xfam_nonlin",
        "-xfam-nonlin",
        action="store_const",
        const=True,
        default=None,
        help="after the rigid link of a family to the root family, also warp it onto "
        "the root along its OWN phase-encode axis. Fitted in the family's EPI frame "
        "(the root's corrected anchor pulled back onto the family's grid), where PE is "
        "a grid axis, and acting beside the family's own anat_nl. Only where the two "
        "slabs overlap drives it; outside, the warp is left to its regularisation. "
        "Off by default: it trusts the root's correction over this family's own.",
    )
    g.add_argument(
        "-xses_nonlin",
        "-xses-nonlin",
        action="store_const",
        const=True,
        default=None,
        help="nonlinear cross-session refinement",
    )
    suggest(
        g.add_argument(
            "-nl_backend",
            "-nl-backend",
            default=None,
            metavar="NAME",
            help="Engine for every nonlinear cross-run/fmap/session refinement "
            f"(default: {config.DEFAULT_NL_BACKEND}). One of: "
            + ", ".join(config.nl_backends())
            + ". These are the same backends ffs_tunewarp measures, under the same "
            "names, so a tuning run's conclusion names the thing you set here. The "
            "settings themselves are separate: pass them with -xrun_nl_opts and its "
            "siblings, whose defaults follow the backend you pick.",
        ),
        # The live list, not a copy: a new backend shows up at the shell the day
        # it is registered.
        config.nl_backends(),
    )
    for _stage, _what in (
        ("xrun", "cross-run"),
        ("xfmap", "cross-fmap-group"),
        ("xses", "cross-session"),
        ("anat", "EPI-to-anat (-anat_nonlin_method register)"),
        ("ref_anat", "ref-anat-to-anat (-ref_anat_nonlin_method register)"),
        ("xfam", "cross-family (-xfam_nonlin)"),
    ):
        suggest(
            g.add_argument(
                f"-{_stage}_nl_backend",
                f"-{_stage}-nl-backend",
                default=None,
                metavar="NAME",
                help=f"engine for the {_what} nonlinear step only, overriding "
                "-nl_backend. The stages are different problems -- cross-run is a "
                "small residual between two images of one head minutes apart, "
                "cross-session can be a different day and a different shim.",
            ),
            config.nl_backends(),
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
            "linear stage's matrix and its un-allineated input to the backend, which "
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
        "-glm_stim_vec",
        "-glm-stim-vec",
        action="append",
        nargs=2,
        default=None,
        metavar=("LABEL", "PATH"),
        help="continuous TR-locked stimulus vector for the GLM: a regressor given "
        "as one number per TR rather than as onsets (an oscillating background, a "
        "motion-energy trace). Becomes one [[stim_vec]] block in the design TOML — "
        "a STIMULUS, so it gets a beta and a t-stat, not a nuisance regressor. "
        "PATH may be a single full-length file or a glob matching one file per run, "
        "and may contain {task}. LABEL takes the same modifier suffix as the design "
        "TOML (LABEL:abs rectifies, LABEL:deriv differences). Repeatable. For a "
        "pre-convolved vector, add `convolve = false` to the block afterwards — the "
        "TOML is the model.",
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
        "-glm_drop_first",
        "-glm-drop-first",
        type=int,
        default=0,
        metavar="N",
        help="drop the first N TRs of every run at GLM time (steady-state "
        "volumes). Event timing is shifted back by N*TR and the design is "
        "compiled against the trimmed runs, so onsets stay aligned.",
    )
    g.add_argument(
        "-glm_drop_last",
        "-glm-drop-last",
        type=int,
        default=0,
        metavar="N",
        help="drop the last N TRs of every run at GLM time. Needs no timing "
        "shift, but events left past the new run end are dropped.",
    )
    g.add_argument(
        "-glm_blur",
        "-glm-blur",
        action="append",
        nargs="+",
        default=None,
        metavar="FWHM [TASK]",
        help="spatially smooth at GLM time with a Gaussian of FWHM mm "
        "(ffs_reml -do_blur). Preprocessing stays unsmoothed, so this is one "
        "parameter you can change and re-run stage12 alone. Blurred buckets are "
        "tagged (stage12.blur6.stats-reml.task-X.nii.gz), so an unsmoothed and a "
        "smoothed fit of the same task sit side by side. Add a TASK to smooth "
        "just that one, and repeat the flag: `-glm_blur 3 -glm_blur 6 ImgStim` "
        "gives ImgStim 6 mm and every other task 3 mm. With only task-specific "
        "entries, the tasks none of them names are not smoothed at all.",
    )
    g.add_argument(
        "-glm_mask",
        "-glm-mask",
        choices=("none", "epi", "anat", "brain"),
        default="none",
        help="which stage10b mask to fit the GLM inside. none (default) fits "
        "every voxel -- a mask is an analysis decision, and an over-tight one "
        "drops voxels from the stats with nothing left to show it happened. epi "
        "= the automask of the final-space data (what was acquired), anat = the "
        "anat brain automasked on the same grid (what is brain, with no eyes or "
        "neck), brain = their intersection (stage10.mask_brain). All three are "
        "written either way, so this is re-runnable against stage12 alone.",
    )
    g.add_argument(
        "-glm_label",
        "-glm-label",
        default=None,
        metavar="TAG",
        help="name this model variant. TAG lands in the bucket names next to the "
        "blur token (-glm_label fir -> stage12.fir.stats-reml.task-X.nii.gz), so "
        "two models of the same task -- a different nuisance set, a different "
        "HRF, a hand-edited design TOML -- do not overwrite each other. Letters, "
        "digits, - and _ only.",
    )
    g.add_argument(
        "-clustsim",
        action="store_true",
        help="attach Monte-Carlo cluster-size thresholds (3dClustSim) to each "
        "task's stat bucket, so the AFNI viewer reports cluster significance. "
        "Runs inside ffs_reml off that fit's own residual ACF and the pipeline's "
        "final-space mask, so it needs no extra input and writes no residual dataset. "
        "Adds roughly a minute or two per task.",
    )
    g.add_argument(
        "-clustsim_null",
        choices=("acf", "resid"),
        default="acf",
        help="the null fields behind -clustsim and -surface_clustsim (ffs_reml "
        "-clustsim_null). acf (default, 3dClustSim-compatible): noise with the residuals' "
        "mixed ACF, stationary by construction. resid -- RECOMMENDED: Gaussian sums of "
        "each fit's own residual time points, with their real spatial covariance. On a "
        "half-rough, half-smooth null field the ACF table's real false-positive rate was "
        "0.17 at a nominal 0.05 (resid: 0.050); on real cortex the tables differed ~3x at "
        "p=0.01. Above all use it with -glm_blur_local / -glm_blur_tsnr (and their "
        "-surface_ twins), which make smoothness vary on purpose.",
    )
    g.add_argument(
        "-glm_blur_to",
        type=float,
        default=None,
        metavar="FWHM",
        help="smooth at GLM time TO this FWHM instead of BY a kernel (ffs_reml "
        "-blur_to_fwhm; 3dBlurToFWHM): the in-mask Gaussian is chosen so the residuals of "
        "an OLS fit of the task's own model measure FWHM mm. Needs -glm_mask (the "
        "smoothness is measured inside it). Buckets: stage12.blurto<F>[local].*. "
        "Exclusive with -glm_blur and -glm_blur_tsnr.",
    )
    g.add_argument(
        "-glm_blur_local",
        action="store_true",
        help="with -glm_blur_to: spatially varying (ffs_reml -blur_to_local) -- diffuse "
        "and stop wherever the LOCAL smoothness reached the target, so rough places get "
        "more blur than smooth ones. Pair with -clustsim_null resid.",
    )
    g.add_argument(
        "-glm_blur_tsnr",
        type=float,
        default=None,
        metavar="TSNR",
        help="smooth each voxel only until its residual tSNR reaches TSNR (ffs_reml "
        "-blur_to_tsnr): noisy places get more blur, clean ones none, nothing past "
        "-glm_blur_max_fwhm, and a place stops once more blur stops buying tSNR "
        "(structured noise). Needs -glm_mask. Buckets: stage12.tsnr<T>.*. Pair with "
        "-clustsim_null resid. Exclusive with -glm_blur and -glm_blur_to.",
    )
    g.add_argument(
        "-glm_blur_max_fwhm",
        type=float,
        default=None,
        metavar="MM",
        help="-glm_blur_tsnr: no voxel smoothed past this local FWHM (default ffs_reml's, 8).",
    )
    g.add_argument(
        "-glm_blur_nbhd",
        type=float,
        default=None,
        metavar="MM",
        help="-glm_blur_local / -glm_blur_tsnr: the neighbourhood each local smoothness "
        "or tSNR estimate is pooled over (default ffs_reml's, 8).",
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
    g.add_argument(
        "-event_ignore",
        "-event-ignore",
        nargs="*",
        default=None,
        metavar="LABEL",
        help="trial_type values to leave out of the model entirely, for EVERY task "
        "(ffs_reml -event_ignore; written as drop_trial_types in each design TOML, and "
        "passed to the per-run stages that read the same events). This REPLACES the "
        "default drop list " + ", ".join(DEFAULT_DROP_TRIAL_TYPES) + ", so "
        "`-event_ignore gray` drops gray and MODELS rest; bare `-event_ignore` with no "
        "labels models every trial type. Matching is on the whole label, not a substring.",
    )
    g.add_argument(
        "-sep_event_ignore",
        "-sep-event-ignore",
        nargs="+",
        action="append",
        default=None,
        metavar="TASK LABEL",
        help="per-task version of -event_ignore; repeat once per task. Wins over "
        "-event_ignore for the task it names: `-sep_event_ignore IVYmetamer gray` "
        "leaves the other tasks on the default list. TASK alone models everything.",
    )

    g.add_argument(
        "-round_onsets",
        "-round-onsets",
        action="append",
        nargs="+",
        default=None,
        metavar="N [TASK ...]",
        help="round event onsets in the design TOML (round_onset in each [[events]] "
        "block). N is decimal places (0 = nearest second, 1 = nearest tenth) or TR "
        "(nearest multiple of the TR). No TASK = every task; name tasks to limit it, "
        "and repeat the flag for different N: `-round_onsets 1 -round_onsets TR "
        "loc08pos`. The other tools split this in two: -round_onsets PLACES, -round_onsets_tr. Only reaches a TOML "
        "that is (re)written — add -glm_spec_overwrite for an existing one.",
    )
    g.add_argument(
        "-round_durations",
        "-round-durations",
        action="append",
        nargs="+",
        default=None,
        metavar="N [TASK ...]",
        help="round event durations the same way (round_duration). This is the fix "
        "for frame-timing jitter: duration = from_events gives one column per EXACT "
        "duration, so 9.9915 s and 10.0083 s blocks become separate, often "
        "collinear, regressors. `-round_durations 0` merges them.",
    )

    for action, verb in (("in", "keep only"), ("out", "drop")):
        g.add_argument(
            f"-event_filter_{action}",
            f"-event-filter-{action}",
            action="append",
            nargs="+",
            default=None,
            metavar="TASK COLUMN VALUE",
            help=f"{verb} that task's events rows whose COLUMN (any TSV column) equals "
            "one of the VALUEs; written as [meta].event_filters in its design TOML "
            "(ffs_reml -event_filter_" + action + "). Repeatable; filters AND "
            "together: `-event_filter_out loc08pos hemifield right -event_filter_out "
            "loc08pos vertical_field upper`. A filtered design is a model variant, so "
            "-glm_label is required and names its TOML and buckets — run autoproc "
            "once per variant (e.g. -glm_label lefthemi, then -glm_label righthemi).",
        )

    g.add_argument(
        "-prebuilt_contrasts",
        "-prebuilt-contrasts",
        action="append",
        nargs=2,
        default=None,
        metavar=("TASK", "FILE"),
        help="a file of [[contrasts]] tables (plain TOML text) to paste into TASK's "
        "design TOML. Every label is checked against the regressors that design "
        "will actually build — after event filters, rounding and duration splits — "
        "and a typo stops ffs_autoproc instead of the GLM an hour later. The text "
        "goes in verbatim, comments and all. Repeatable, also per task. Only reaches "
        "a TOML that is (re)written — add -glm_spec_overwrite for an existing one.",
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

    g = p.add_argument_group("compute device")
    g.add_argument(
        "-device",
        type=_device_spec,
        default=None,
        metavar="DEV",
        help=f"compute device every ffs stage in the script runs on → DEVICE "
        f"(default: {config.DEFAULT_DEVICE}). cuda | mps | cpu | auto, or the "
        "indexed/threaded forms every ffs CLI accepts (cuda,0 / cpu,8). "
        "Edit DEVICE at the top of the emitted script to change it later. "
        "On a Mac, auto or mps uses Metal for the stages it helps and the CPU for the "
        "float64-only ones (ffs_blipflip, ffs_segment).",
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


def _resolve_anat(args) -> tuple[list[str], str | None]:
    """Resolve the anat file(s) and a default tpm_source from -anat/-suma.

    More than one path means "align these together and average them" (see
    :func:`anat_group` for how the auto-discovered set is chosen)."""
    if args.anat:
        return list(args.anat), args.tpm_source
    if args.suma:
        brain = str(Path(args.suma) / "brain.nii.gz")
        surfvol = sorted(Path(args.suma).glob("*SurfVol*.nii*"))
        tpm_src = args.tpm_source or (str(surfvol[0]) if surfvol else None)
        return [brain], tpm_src
    return [], args.tpm_source


def anat_candidates(subject, bids_dir: str | None = None) -> tuple[list[Path], list[Path]]:
    """(the in-scope sessions' T1w, every T1w this subject has).

    The second list leads with the first: a ses-anat holds no BOLD runs, so it is
    not a session of this Subject at all — but it is usually where the structural
    actually lives, and the sweep is the only way to see it."""
    from fastfuncstuff.autoproc.bids import scan_all_anats

    in_scope = [p for sess in subject.sessions for p in sess.anats]
    every = list(in_scope)
    if bids_dir:
        every += [p for p in scan_all_anats(bids_dir, subject.subject) if p not in in_scope]
    return in_scope, every


def anat_group(subject, bids_dir: str | None = None) -> list[Path]:
    """Every scanned T1w that may be averaged with the one we would have picked.

    An MP2RAGE session files inv1, inv2 and uni all as ``_T1w``; a multi-echo
    MPRAGE files four echoes and their combination the same way; and a "rec-norm"
    reconstruction is not the same image as the raw one. Averaging across any of
    those is averaging different contrasts. So the group is the picked anat's own
    (acq, rec, echo) entities, pooled across the in-scope sessions: repeats of ONE
    acquisition, which is exactly what averaging is for."""
    from fastfuncstuff.autoproc.bids import _pick_anat

    in_scope, every = anat_candidates(subject, bids_dir)
    # The in-scope session leads when it has one: same visit as the functional
    # data. Other sessions can still contribute repeats of that same acquisition.
    primary = _pick_anat(in_scope) or _pick_anat(every)
    if primary is None:
        return []

    def key(p: Path) -> tuple[str | None, ...]:
        ent = parse_entities(p.name)
        return tuple((ent.get(e) or "").lower() or None for e in ("acq", "rec", "echo"))

    want = key(primary)
    # The picked one leads: it is the base every other is aligned to.
    rest = [p for p in every if p != primary and key(p) == want]
    return [primary, *rest]


def _resolve_mni(args) -> tuple[str, str | None]:
    """(backend, template) for -do_mni.

    A family name (``optiwarp``) is resolved through the preset registry rather
    than by picking a member here: which engine of a family a recipe means is the
    recipe's fact, and MNI_T1a's answer is hs, not the family's own default."""
    from fastfuncstuff.processing.tunespec import preferred_backend

    backend = args.mni_use
    if backend not in config.nl_backends():
        backend = preferred_backend(MNI_RECIPE, backend, f"{backend}_demons")
    template = args.mni_template or find_mni_template()
    return backend, template


def find_mni_template() -> str | None:
    """AFNI's MNI152_2009_template_SSW.nii.gz, found the way a user would: next to
    the afni on $PATH. Sub-brick 0 is its skull-off volume."""
    afni = shutil.which("afni")
    if afni is None:
        return None
    cand = Path(afni).resolve().parent / "MNI152_2009_template_SSW.nii.gz"
    return str(cand) if cand.is_file() else None


def _resolve_nl_backends(args) -> dict[str, str]:
    """Which engine runs each nonlinear stage: per-stage flag, then -nl_backend.

    Validated here rather than by argparse ``choices`` so the error can name the
    tuner: these are ffs_tunewarp's backend names, and a typo is far more often a
    half-remembered name than an unsupported engine.
    """
    known = config.nl_backends()
    out: dict[str, str] = {}
    for stage in config.NL_STAGE_KEYS:
        chosen = (
            getattr(args, f"{stage}_backend", None) or args.nl_backend or config.DEFAULT_NL_BACKEND
        )
        if chosen not in known:
            raise SystemExit(
                f"unknown nonlinear backend {chosen!r}. Known: {', '.join(known)} "
                "(the same names ffs_tunewarp reports)."
            )
        out[f"{stage}_backend"] = chosen
    return out


def _multiecho_errors(opt: Options, subject) -> list[str]:
    """What the multi-echo path does not handle yet, refused at generation time.

    Multi-echo runs are estimated from echo 1, every echo is warped with its own
    chain at stage10 and the echoes are optimally combined into the run's series.
    The stages below were written for one series per run."""
    me = [r for s in subject.sessions for r in s.bold_runs if r.is_multiecho]
    if not me:
        return []
    errs = []
    no_te = [f"{r.task} run {r.run or '-'}" for r in me if len(r.echo_tes) != r.n_echoes]
    if no_te:
        errs.append(
            "multi-echo runs without an EchoTime in every echo's sidecar: " + ", ".join(no_te)
        )
    if opt.phasereg:
        # Each echo is regressed across the task's runs together, so they must agree.
        by_task: dict[str, set] = {}
        for s in subject.sessions:
            for r in s.bold_runs:
                if r.phase_path is not None:
                    by_task.setdefault(r.task, set()).add(
                        tuple(round(t, 3) for t in r.echo_tes) if r.is_multiecho else ()
                    )
        mixed = [t for t, v in by_task.items() if len(v) > 1]
        if mixed:
            errs.append(
                "-phasereg regresses each echo across a task's runs, so they need the same "
                "echoes; these tasks mix echo counts or TEs: " + ", ".join(mixed)
            )
    return errs


def _medic_errors(opt: Options, subject) -> list[str]:
    """-medic needs, for every run, what the frame-wise field and its warp are made of:
    multi-echo phase, the readout time and the PE direction. Nothing is guessed."""
    if not opt.medic:
        return []
    if not opt.phase_proc:
        return ["-medic needs -phase_proc: its field map comes from the unwrap stage."]
    errs = []
    for r in (r for s in subject.sessions for r in s.bold_runs):
        name = f"{r.session or '-'}/{r.task}/{r.run or '-'}"
        if not r.is_multiecho or any(p is None for p in r.echo_phases):
            errs.append(f"-medic: {name} is not multi-echo with a phase for every echo.")
            continue
        if r.json.get("TotalReadoutTime") is None:
            errs.append(f"-medic: {name} has no TotalReadoutTime in its sidecar.")
        if not r.pe_dir:
            errs.append(f"-medic: {name} has no PhaseEncodingDirection in its sidecar.")
    return errs


def _locomoco_phase_errors(opt: Options, subject) -> list[str]:
    """-locomoco_phase is the joint solve over 3-D multi-echo runs with phase; anything
    else has nothing to hand it, and is refused rather than silently ignored."""
    if not opt.locomoco_phase:
        return []
    if not opt.locomoco:
        return ["-locomoco_phase needs -locomoco."]
    if not opt.phase_proc:
        return ["-locomoco_phase needs -phase_proc: the phase comes from the unwrap stage."]
    if not opt.acq_3d:
        return ["-locomoco_phase is the 3-D partition solve; it needs -acq_3d EPI."]
    errs = []
    for r in (r for s in subject.sessions for r in s.bold_runs):
        if not r.is_multiecho or any(p is None for p in r.echo_phases):
            name = f"{r.session or '-'}/{r.task}/{r.run or '-'}"
            errs.append(f"-locomoco_phase: {name} is not multi-echo with a phase for every echo.")
    return errs


def _phasereg_me_errors(opt: Options) -> list[str]:
    """-phasereg_me field regresses the combined echoes on a field fitted from every
    echo's phase; without phase regression or the processed phase there is none."""
    if opt.phasereg_me != "field":
        return []
    if not opt.phasereg:
        return ["-phasereg_me field needs -phasereg."]
    if not opt.phase_proc:
        return [
            "-phasereg_me field needs -phase_proc: the field is fitted from the unwrapped phase."
        ]
    return []


def _hrftdv_errors(opt: Options) -> list[str]:
    """-hrftdv reads the cortical ribbon off the FreeSurfer surfaces in the anat's
    scanner space, so the final data must be in that space; and ffs_hrftdv has no
    -drop_first, so its fit would not see the window the GLM fits."""
    if not opt.hrftdv:
        return []
    errs = []
    if not opt.go_to_anat or opt.grand_reference or opt.ref_file or opt.do_mni:
        errs.append(
            "-hrftdv finds its columns on the FreeSurfer surfaces, which sit on THIS "
            "pipeline's anat: the final space must be that anat (not -no_anat, "
            "-grand_reference, -ref_file or -do_mni)."
        )
    if opt.fs_subject is None:
        errs.append("-hrftdv needs the FreeSurfer subject: -fs_subject DIR or -suma.")
    elif not (Path(opt.fs_subject) / "surf").is_dir():
        errs.append(f"-hrftdv: -fs_subject has no surf/ folder: {opt.fs_subject}")
    if opt.glm_drop_first or opt.glm_drop_last:
        errs.append(
            "-hrftdv cannot take -glm_drop_first / -glm_drop_last: ffs_hrftdv has no trim, "
            "so its pairs would come from other volumes than the GLM fits."
        )
    return errs


def preflight(args, opt: Options, anat_path: str | None, subject) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Non-empty errors ⇒ do not write the script."""
    errors: list[str] = []
    warnings: list[str] = []

    # Typo in an override: catch it now, not when the script reaches that stage.
    # A nonlinear stage is checked against the engine that will actually run it.
    nl_module = {
        key: config.nl_command(getattr(opt, f"{key}_backend")).split()[0].removeprefix("ffs_")
        for key in config.NL_STAGE_KEYS
    }
    nl_module["mni_nonlin"] = config.nl_command(opt.mni_backend).split()[0].removeprefix("ffs_")
    for key in (*config.STAGE_OPT_KEYS, "glm"):
        val = getattr(args, f"{key}_opts", None)
        if val:
            errors += optcheck.check_opts(key, val, module=nl_module.get(key))

    needs_anat = opt.go_to_anat and not opt.grand_reference and not opt.ref_file
    if needs_anat and anat_path is None:
        errors.append(
            "this recipe aligns to an anatomical, but no anat is available. Pass -anat FILE "
            "(skull-stripped T1w) or -suma DIR (FreeSurfer SUMA; uses brain.nii.gz). "
            "The in-scope session(s) had no T1w — e.g. The T1w might be in a different session, "
            "not the functional session."
        )
    if opt.anat_nonlin and opt.anat_nonlin_method == "register" and opt.has_grand_ref:
        errors.append(
            "-anat_nonlin_method register refines this data's own anat alignment; with "
            "-grand_reference / -ref_file the anat link is borrowed (a chain of matrices "
            "the backend's single -matrix cannot take). Use -grand_reference_nonlin."
        )
    if opt.anat_nonlin and opt.anat_nonlin_method == "segment" and not opt.tpm:
        why = (
            # The anat route is the one that needs no extra acquisition, so when it
            # is unavailable, say which half is missing rather than listing all three.
            " An unstripped anat plus an SPM group TPM would build one, but "
            + (
                "no group TPM was found: pass -spm_tpm FILE (SPM's tpm/TPM.nii) or set $FFS_TPM."
                if opt.anat_skull
                else "this anat is already skull-stripped (-anat_skull no), and the group "
                "TPM's bone/soft-tissue/air classes need the head they describe."
            )
        )
        errors.append(
            "-anat_nonlin (ffs_segment) needs a subject TPM: pass -tpm FILE, or -suma DIR "
            "to build one from FreeSurfer (aseg.auto + SurfVol)." + why
        )
    rooted = [label for label, mode in opt.family_links.items() if mode == "root"]
    if rooted and opt.family_ref == "anat":
        errors.append(
            f"-family_link {rooted[0]} root: with -family_ref anat (the default) there is "
            "no root to link to. Name one: -family_ref highest_res, or a family label."
        )
    if opt.ref_anat:
        if not opt.go_to_anat:
            errors.append("-ref_anat is a step on the way to -anat: it needs one (not -no_anat).")
        if opt.xfam_nonlin:
            errors.append(
                "-ref_anat with -xfam_nonlin: that warp is fitted by pulling the root's "
                "anchor back through the inverted anat chain, and the ref→anat link is a "
                "warp, which ffs_nwarp cannot invert. Drop -xfam_nonlin (xfam_lin is fine)."
            )
        if opt.ref_file or opt.grand_reference:
            errors.append(
                "-ref_anat with -ref_file / -grand_reference: the anat link is borrowed "
                "there, so there is no own anat step for the ref to stand in for."
            )
        if opt.ref_anat_nonlin_method == "segment" and not opt.tpm:
            errors.append(
                "-ref_anat_nonlin_method segment needs -anat's subject TPM: -tpm FILE, "
                "-suma DIR, or an unstripped -anat plus -spm_tpm / $FFS_TPM. Or use "
                "-ref_anat_nonlin_method register."
            )
    if opt.anat_extra and (opt.grand_reference or opt.ref_file):
        borrowed = "-grand_reference" if opt.grand_reference else "-ref_file"
        warnings.append(
            f"{len(opt.anat_extra) + 1} T1w were found, but {borrowed} supplies the "
            "anat link — they are not aligned or averaged, and this run never reads them."
        )
    if opt.anat_skull:
        if args.suma:
            errors.append(
                "-anat_skull yes with -suma: the SUMA brain.nii.gz is already stripped. "
                "Drop -anat_skull, or point -anat at the unstripped T1w instead."
            )
        # Only warned about when the script will actually run the strip.
        elif (
            opt.go_to_anat
            and opt.ref_file is None
            and opt.grand_reference is None
            and shutil.which("mri_synthstrip") is None
        ):
            warnings.append(
                "-anat_skull yes: 'mri_synthstrip' is not on $PATH (it ships with "
                "FreeSurfer >= 8). The script's preflight will refuse to run until it is."
            )
    if opt.do_mni:
        if not opt.go_to_anat:
            errors.append("-do_mni needs an anatomical: it is the anat that reaches MNI. ")
        if opt.ref_file:
            errors.append(
                "-do_mni with -ref_file: the reference's transforms are yours to supply, "
                "so add the anat→MNI pair to -ref_transforms instead."
            )
        if opt.mni_template is None:
            errors.append(
                "-do_mni found no MNI template: 'afni' is not on $PATH (its directory is "
                "where MNI152_2009_template_SSW.nii.gz lives). Pass -mni_template FILE."
            )
        elif not Path(opt.mni_template).exists():
            errors.append(f"-mni_template not found: {opt.mni_template}")
        if opt.grand_reference:
            # Borrowed rather than recomputed — but only if the reference has one.
            missing = [
                f"{opt.grand_reference.rstrip('/')}/stage09.mni{tail}"
                for tail in (".aff12.1D", "_nl_WARP.nii.gz")
            ]
            absent = [m for m in missing if not Path(m).exists()]
            if absent:
                errors.append(
                    "-do_mni with -grand_reference borrows the reference's anat→MNI warp, "
                    f"and it is not there: {', '.join(absent)}. Re-run the reference with "
                    "-do_mni first."
                )
    if opt.surface_qc and not opt.surface_target:
        errors.append("-surface_qc maps the surface projection: it needs -surface_target.")
    surf_glm_flags = [
        f
        for f, on in (
            ("-surface_glm_blur", opt.surface_glm_blur is not None),
            ("-surface_glm_blur_to", opt.surface_glm_blur_to is not None),
            ("-surface_glm_blur_local", opt.surface_glm_blur_local),
            ("-surface_glm_blur_tsnr", opt.surface_glm_blur_tsnr is not None),
            ("-surface_glm_blur_max_fwhm", opt.surface_glm_blur_max_fwhm is not None),
            ("-surface_glm_blur_nbhd", opt.surface_glm_blur_nbhd is not None),
            ("-surface_clustsim", opt.surface_clustsim),
            ("-surface_clustsim_niter", opt.surface_clustsim_niter is not None),
            ("-surface_glm_opts", bool(opt.surface_glm_opts)),
        )
        if on
    ]
    if opt.surface_glm and not opt.surface_target:
        errors.append("-surface_glm fits the surface projection: it needs -surface_target.")
    elif surf_glm_flags and not opt.surface_glm:
        errors.append(f"{', '.join(surf_glm_flags)} configure the surface GLM: add -surface_glm.")
    errors += _target_blur_errors(
        "-surface_glm_blur",
        by=opt.surface_glm_blur is not None,
        to=opt.surface_glm_blur_to,
        tsnr=opt.surface_glm_blur_tsnr,
        local=opt.surface_glm_blur_local,
        max_fwhm=opt.surface_glm_blur_max_fwhm,
        nbhd=opt.surface_glm_blur_nbhd,
    )
    errors += _target_blur_errors(
        "-glm_blur",
        by=opt.glm_blur is not None or bool(opt.sep_glm_blur),
        to=opt.glm_blur_to,
        tsnr=opt.glm_blur_tsnr,
        local=opt.glm_blur_local,
        max_fwhm=opt.glm_blur_max_fwhm,
        nbhd=opt.glm_blur_nbhd,
    )
    if (opt.glm_blur_to or opt.glm_blur_tsnr) and opt.glm_mask == "none":
        errors.append(
            "-glm_blur_to / -glm_blur_tsnr measure smoothness inside the GLM mask, and "
            "-glm_mask is none: pick -glm_mask brain (or epi / anat). Air would read as "
            "white noise and drag the measurement down."
        )
    varying_vol = opt.clustsim and (opt.glm_blur_local or opt.glm_blur_tsnr)
    varying_surf = opt.surface_clustsim and (
        opt.surface_glm_blur_local or opt.surface_glm_blur_tsnr
    )
    if opt.clustsim_null == "acf" and (varying_vol or varying_surf):
        warnings.append(
            "cluster tables with the ACF null on a blur that varies smoothness on purpose "
            "(-*_blur_local / -*_blur_tsnr): one ACF cannot describe it -- its real "
            "false-positive rate was 0.17 at a nominal 0.05 on a half-rough, half-smooth "
            "field. Strongly recommended: -clustsim_null resid (0.050 there)."
        )
    elif opt.clustsim_null == "acf" and (opt.clustsim or opt.surface_clustsim):
        warnings.append(
            "cluster tables use the ACF null (3dClustSim-compatible). -clustsim_null resid "
            "is recommended: it uses the residuals' real spatial covariance (on real cortex "
            "the tables differed ~3x at p=0.01)."
        )
    if opt.clustsim_null == "resid" and not (opt.clustsim or opt.surface_clustsim):
        warnings.append("-clustsim_null resid does nothing without -clustsim / -surface_clustsim.")
    if opt.surface_target:
        if not opt.go_to_anat or opt.grand_reference or opt.ref_file:
            errors.append(
                "-surface_target samples through the chain to THIS pipeline's own anat; it "
                "needs an anat alignment here (not -no_anat, -grand_reference or -ref_file)."
            )
        if opt.fs_subject is None:
            errors.append("-surface_target needs the FreeSurfer subject: -fs_subject DIR or -suma.")
        elif not (Path(opt.fs_subject) / "surf").is_dir():
            errors.append(
                f"-fs_subject has no surf/ folder (as a path, or a name in $SUBJECTS_DIR): "
                f"{opt.fs_subject}"
            )
        else:
            from fastfuncstuff.processing.surface_projection import resolve_mesh

            for mesh in opt.surface_target:
                try:
                    resolve_mesh(mesh, opt.fs_subject)
                except FileNotFoundError as exc:
                    errors.append(str(exc).replace("-surf_mesh", "-surface_target"))
    if opt.slicetiming_method != "none" and opt.tr is None:
        # Slice timing needs a TR per run; the sidecar is the only source here.
        no_tr = [r for s in subject.sessions for r in s.bold_runs if r.tr is None]
        if no_tr:
            errors.append(
                f"slice timing is on but {len(no_tr)} run(s) have no RepetitionTime in their "
                "sidecar. Pass -TR SECONDS (volume TR) or -slicetiming_method none."
            )
    errors += _multiecho_errors(opt, subject)
    errors += _medic_errors(opt, subject)
    errors += _locomoco_phase_errors(opt, subject)
    errors += _phasereg_me_errors(opt)
    if opt.medic and any(s.fmaps for s in subject.sessions):
        warnings.append(
            "-medic: the session fieldmaps are not used -- the per-frame field corrects "
            "the distortion, and a fieldmap would correct it twice."
        )
    if opt.medic and opt.locomoco and not opt.acq_3d and not opt.medic_inplane_locomoco:
        warnings.append(
            "-medic with 2-D EPI: locomoco is skipped -- MEDIC corrects in-plane frame by "
            "frame and 2-D has no partition axis (-medic_inplane_locomoco keeps it)."
        )
    if opt.phase_proc:
        # Phase is carried per run, wherever a run has one: a session can mix
        # gradient-echo runs (with phase) and spin-echo runs (without).
        runs = [r for s in subject.sessions for r in s.bold_runs]
        missing = [
            f"{r.session or '-'}/{r.task}/{r.run or '-'}" for r in runs if r.phase_path is None
        ]
        if len(missing) == len(runs):
            errors.append("-phase_proc: no run has a phase timeseries (part-phase bold)")
        elif missing:
            warnings.append(
                "-phase_proc: these runs have no phase and are processed magnitude-only: "
                + ", ".join(missing)
            )
    if opt.phasereg and not opt.phase_proc:
        errors.append("-phasereg needs -phase_proc: it regresses the phase that carries.")
    errors += _hrftdv_errors(opt)
    if opt.hrftdv and not opt.run_glm:
        warnings.append("-hrftdv runs with the GLM: set FFS_RUN_GLM=1 when running the script.")
    if opt.tissue:
        if len(opt.tissue) not in (1, 3):
            errors.append("-tissue takes one 4-D file (GM, WM, CSF) or three files.")
        errors += [f"-tissue not found: {t}" for t in opt.tissue if not Path(t).is_file()]
    if opt.phasereg and opt.phasereg_veins == "auto" and not (opt.tissue or opt.presurf):
        from fastfuncstuff.autoproc.emit import _aseg_tissue

        if _aseg_tissue(opt):
            aseg = Path(opt.suma_dir) / "aseg.auto.nii.gz"
            if not aseg.is_file():
                errors.append(f"-phasereg: -suma has no aseg.auto.nii.gz for tissue maps: {aseg}")
        else:
            warnings.append(
                "-phasereg: no anat tissue maps (-tissue, -presurf or -suma), so no vein mask: "
                "plain phase regression. Give -tissue to restrict -spr to vein candidates."
            )
    if opt.fs_tpm:  # in-script TPM build needs the FS label volume
        aseg = Path(opt.suma_dir) / "aseg.auto.nii.gz"
        if not aseg.is_file():
            errors.append(f"-suma has no aseg.auto.nii.gz to build a TPM from: {aseg}")
    # File/dir existence for everything the user pointed at.
    for extra in opt.anat_extra:
        if not Path(extra).exists():
            errors.append(f"-anat not found: {extra}")
    for label, val in (
        ("-anat", anat_path if args.anat else None),
        ("-tpm", args.tpm),
        ("-spm_tpm (or $FFS_TPM)", opt.spm_tpm if opt.anat_tpm else None),
        ("-anat_t2", opt.anat_t2),
        ("-presurf_tpm (or $FFS_TPM)", opt.presurf_tpm),
        ("-tpm_source", opt.tpm_source),
        ("-ref_file", opt.ref_file),
        ("-ref_file_anat", opt.ref_file_anat),
        ("-ref_anat", opt.ref_anat),
        ("-ref_anat_inv2", args.ref_anat_inv2),
        ("-ref_anat_inv1", args.ref_anat_inv1),
        ("-slicetiming", opt.slicetiming_file),
    ):
        if val and not Path(val).exists():
            errors.append(f"{label} not found: {val}")
    no_inv1 = [Path(uni).name for uni, _, inv1 in opt.presurf if inv1 is None]
    if opt.ref_presurf and opt.ref_presurf[2] is None:
        no_inv1.append(Path(opt.ref_presurf[0]).name)
    if opt.presurf_denoise != "off" and no_inv1:
        errors.append(
            f"-presurf_denoise {opt.presurf_denoise} denoises INV1 and INV2 jointly, but "
            f"there is no acq-inv1 beside {', '.join(no_inv1)}. Drop -presurf_denoise."
        )
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
    if opt.ref_file and not (opt.ref_transforms or opt.ref_file_anat):
        warnings.append(
            "-ref_file given without -ref_transforms/-ref_file_anat: the reference is assumed to "
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
        elif not any(s.fmaps for s in subject.sessions) or not opt.distortion or opt.medic:
            warnings.append(
                f"-anat_source {opt.anat_source} needs fieldmaps; falling back to grandmean "
                "(ffs_segment is what recovers the distortion in that case)."
            )
    # A bad -motsim spec is a typo, and the script it would write runs for hours
    # before ffs_moco rejects it.
    if opt.motsim is not None:
        from fastfuncstuff.processing.motsim import parse_motsim_spec

        try:
            parse_motsim_spec(opt.motsim)
        except ValueError as exc:
            errors.append(f"-motsim: {exc}")

        # Estimating them and not modelling with them is a legitimate thing to
        # want (the .1D is a diagnostic in its own right), but it is far more
        # often a forgotten flag, and the whole run is spent before you find out.
        if "motsim" not in opt.glm_ortvec:
            warnings.append(
                "-motsim is on but the GLM does not use it. The regressors are still "
                "written per run; add -glm_ortvec motsim (or a bare -glm_ortvec, which "
                "puts MotSim in place of motion/motion_deriv) to model with them."
            )

    if opt.censor_motion is not None and opt.censor_motion < 0:
        errors.append("-censor_motion: the limit must be >= 0.")
    if opt.censor_outliers is not None and not 0.0 <= opt.censor_outliers <= 1.0:
        errors.append("-censor_outliers: the fraction must be in [0, 1].")
    if (opt.censor_motion is not None or opt.censor_outliers is not None) and not opt.run_glm:
        warnings.append(
            "censoring is on but the GLM is not: the per-run censor files are written "
            "in stage02 and nothing in this script uses them."
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
    # A blur of 0 or less is not a no-op request, it is a typo: -do_blur would
    # reject it downstream, an hour of preprocessing later.
    for what, fwhm in [("-glm_blur", opt.glm_blur)] + [
        (f"-glm_blur ... {t}", f) for t, f in sorted(opt.sep_glm_blur.items())
    ]:
        if fwhm is not None and fwhm <= 0:
            errors.append(
                f"{what} must be a positive FWHM in mm (got {fwhm:g}); "
                "omit the flag for no smoothing."
            )
    # The label becomes a filename token in a dot-delimited scheme, and reaches a
    # generated bash script -- so anything but [A-Za-z0-9_-] is refused here
    # rather than producing an unparseable name or a shell surprise.
    if opt.glm_label is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", opt.glm_label):
        errors.append(
            f"-glm_label {opt.glm_label!r}: use letters, digits, - and _ only (it becomes a "
            "filename token, and '.' is what separates tokens)."
        )
    if opt.event_filters and opt.glm_label is None:
        errors.append(
            "-event_filter_in/-event_filter_out change the design, so the fit is a model "
            "variant: name it with -glm_label (e.g. -glm_label lefthemi). The label "
            "names the design TOML and the stage12 buckets, so variants cannot collide."
        )
    # The anat side of the mask needs an anat this pipeline segments and aligns
    # itself; borrowed/overridden reference geometry brings no local brain to mask
    # with, so say so rather than emitting a -mask for a file that is never written.
    if opt.glm_mask in ("anat", "brain") and not (
        opt.go_to_anat and opt.ref_file is None and opt.grand_reference is None
    ):
        warnings.append(
            f"-glm_mask {opt.glm_mask} needs this pipeline's own anat (-anat / BIDS T1w, "
            "no -grand_reference or -ref_file) — falling back to the EPI mask."
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
    for flag, per_task in (
        ("-round_onsets", opt.sep_round_onsets),
        ("-round_durations", opt.sep_round_durations),
        ("-event_filter_in/out", opt.event_filters),
        ("-sep_event_ignore", opt.sep_event_ignore),
        ("-prebuilt_contrasts", opt.prebuilt_contrasts),
    ):
        for task in sorted(per_task):
            if task not in all_tasks:
                warnings.append(
                    f"{flag} names task '{task}', which is not in this subject/scope "
                    f"({', '.join(sorted(all_tasks)) or 'none'}) — it does nothing."
                )
    for task in sorted(opt.family_overrides):
        if task not in all_tasks:
            warnings.append(
                f"-family names task '{task}', which is not in this subject/scope "
                f"({', '.join(sorted(all_tasks)) or 'none'}) — it does nothing."
            )
    for task in sorted(opt.cut_task_vols):
        if task not in all_tasks:
            warnings.append(
                f"-cut_task_vols names task '{task}', which is not in this subject/scope "
                f"({', '.join(sorted(all_tasks)) or 'none'}) — it does nothing."
            )
        elif not opt.events:
            from fastfuncstuff.autoproc.glm import cut_event_warnings

            warnings += cut_event_warnings(task, _events_by_run(args.bids_dir, task, subject), opt)
    for task in sorted(opt.sep_glm_blur):
        if task not in all_tasks:
            warnings.append(
                f"-glm_blur names task '{task}', which is not in this subject/scope "
                f"({', '.join(sorted(all_tasks)) or 'none'}) — that FWHM is never used."
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

    # BIDS lets an events file omit trial_type; every row is then one condition.
    # That is right for a single-condition block design and wrong if the condition
    # sits in a column under another name, so say which happened.
    if opt.run_glm:
        import csv

        from fastfuncstuff.autoproc.glm import requested_event_cols
        from fastfuncstuff.design.bids_events import SINGLE_TRIAL_TYPE

        for task in sorted(all_tasks):
            if requested_event_cols(task, opt) is not None:
                continue
            if opt.events:
                paths = [Path(e) for e in opt.events]
            else:
                paths = [ev for _, ev in _events_by_run(args.bids_dir, task, subject) if ev]
            lacking = []
            for path in dict.fromkeys(paths):
                try:
                    with open(path, newline="") as fh:
                        header = next(csv.reader(fh, delimiter="\t"), [])
                except OSError:
                    continue
                if "trial_type" not in header:
                    lacking.append((Path(path).name, header))
            if lacking:
                name, header = lacking[0]
                warnings.append(
                    f"task-{task}: {name} has no trial_type column (columns: "
                    f"{', '.join(header)}), so every row is modelled as ONE condition "
                    f"'{SINGLE_TRIAL_TYPE}'. If a column names the conditions, pass "
                    "-spec_event_cols ONSET DURATION TRIAL_TYPE."
                )

    # A filter on a column the TSVs lack would build the wrong model an hour from now.
    if opt.run_glm and opt.event_filters:
        import csv

        for task in sorted(set(opt.event_filters) & all_tasks):
            if opt.events:
                paths = [Path(e) for e in opt.events]
            else:
                paths = [ev for _, ev in _events_by_run(args.bids_dir, task, subject) if ev]
            for path in paths:
                try:
                    with open(path, newline="") as fh:
                        header = next(csv.reader(fh, delimiter="\t"), [])
                except OSError:
                    continue
                absent = sorted({f.column for f in opt.event_filters[task]} - set(header))
                if absent:
                    errors.append(
                        f"-event_filter_in/out task-{task}: column(s) {', '.join(absent)} "
                        f"not in {Path(path).name} (columns: {', '.join(header)})."
                    )
                    break

    # A -event_ignore label that matches nothing is a typo doing nothing, and the
    # condition the user meant to drop is still in the model. Whole-label match.
    if opt.run_glm and (opt.event_ignore or opt.sep_event_ignore):
        from fastfuncstuff.autoproc.glm import requested_event_ignore, resolve_event_cols
        from fastfuncstuff.design.bids_events import read_tsv_rows

        for task in sorted(all_tasks):
            want = requested_event_ignore(task, opt)
            if not want:
                continue
            if opt.events:
                paths = [Path(e) for e in opt.events]
            else:
                paths = [ev for _, ev in _events_by_run(args.bids_dir, task, subject) if ev]
            if not paths:
                continue
            cols, _warn = resolve_event_cols(task, paths, opt)
            onset_col, dur_col, tt_col = cols or DEFAULT_EVENT_COLUMNS
            present: set[str] = set()
            for path in dict.fromkeys(paths):
                try:
                    rows, _ = read_tsv_rows(path, onset_col, dur_col, tt_col)
                except (OSError, ValueError):
                    present = set()
                    break
                present.update(r["_trial_type"] for r in rows)
            unseen = [lab for lab in want if lab not in present] if present else []
            if unseen:
                warnings.append(
                    f"-event_ignore task-{task}: {', '.join(unseen)} match no trial_type "
                    f"(present: {', '.join(sorted(present))}) — nothing is dropped for them."
                )

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
    # A set nobody spelled out is a default, and the default with -motsim on is
    # MotSim in place of the parameters it was derived from.
    if args.motsim is not None and not args.glm_ortvec:
        names = config.apply_motsim_default(names)
    unknown = [n for n in names if n not in config.GLM_ORTVEC]
    if unknown:
        raise SystemExit(
            f"ffs_autoproc: unknown -glm_ortvec name(s): {', '.join(unknown)}\n"
            f"  known: {', '.join(config.GLM_ORTVEC)}"
        )
    return names


def _resolve_event_filters(args) -> dict[str, list]:
    """``{task: [EventFilter]}`` from ``-event_filter_in/out TASK COLUMN VALUE ...``."""
    from fastfuncstuff.design.bids_events import EventFilter

    per_task: dict[str, list] = {}
    for action in ("in", "out"):
        for entry in getattr(args, f"event_filter_{action}", None) or []:
            if len(entry) < 3:
                raise SystemExit(
                    f"ffs_autoproc: -event_filter_{action} takes TASK COLUMN VALUE [VALUE ...] "
                    f"(got {' '.join(entry)})."
                )
            task = entry[0][len("task-") :] if entry[0].startswith("task-") else entry[0]
            per_task.setdefault(task, []).append(
                EventFilter(column=entry[1], values=list(entry[2:]), action=action)
            )
    return per_task


def _resolve_prebuilt_contrasts(args) -> dict[str, list[str]]:
    """``{task: [absolute path]}``; each file must exist and parse as [[contrasts]] only.

    Label checks need the events, so they happen when the design TOML is written.
    """
    from fastfuncstuff.design.spec import parse_contrasts_text

    per_task: dict[str, list[str]] = {}
    for task, path in args.prebuilt_contrasts or []:
        task = task[len("task-") :] if task.startswith("task-") else task
        p = Path(path).expanduser()
        if not p.is_file():
            raise SystemExit(f"ffs_autoproc: -prebuilt_contrasts {task}: no such file {path}")
        try:
            parse_contrasts_text(p.read_text(), str(p))
        except ValueError as exc:
            raise SystemExit(f"ffs_autoproc: -prebuilt_contrasts {task}: {exc}") from None
        per_task.setdefault(task, []).append(str(p.resolve()))
    return per_task


def _resolve_round_modes(
    entries: list[list[str]] | None, flag: str
) -> tuple[int | str | None, dict[str, int | str]]:
    """``(dataset-wide mode, {task: mode})`` from repeated ``FLAG N [TASK ...]``.

    A bare N applies to every task no entry names; a task named twice keeps the
    last. A bad N is refused here rather than at GLM time an hour later.
    """
    from fastfuncstuff.design.spec import parse_round_mode

    wide: int | str | None = None
    per_task: dict[str, int | str] = {}
    for entry in entries or []:
        try:
            mode = parse_round_mode(entry[0], flag)
        except ValueError as exc:
            raise SystemExit(f"ffs_autoproc: {exc}. The mode comes first: {flag} 0 TASK.") from None
        assert mode is not None
        if len(entry) == 1:
            wide = mode
        for task in entry[1:]:
            per_task[task[len("task-") :] if task.startswith("task-") else task] = mode
    return wide, per_task


def _target_blur_errors(
    what: str,
    *,
    by: bool,
    to: float | None,
    tsnr: float | None,
    local: bool,
    max_fwhm: float | None,
    nbhd: float | None,
) -> list[str]:
    """Consistency of one GLM's smoothing flags (``what`` = -glm_blur or
    -surface_glm_blur): BY a kernel, TO a FWHM, or TO a tSNR -- one of them -- with
    the modifiers only where they mean something."""
    errors = []
    kinds = [f for f, on in ((what, by), (f"{what}_to", to is not None),
                             (f"{what}_tsnr", tsnr is not None)) if on]  # fmt: skip
    if len(kinds) > 1:
        errors.append(f"{' and '.join(kinds)} are alternatives: pick one.")
    if local and to is None:
        errors.append(f"{what}_local makes {what}_to spatially varying: it needs {what}_to.")
    if max_fwhm is not None and tsnr is None:
        errors.append(f"{what}_max_fwhm caps {what}_tsnr: it needs {what}_tsnr.")
    if nbhd is not None and not (local or tsnr is not None):
        errors.append(f"{what}_nbhd is for {what}_local / {what}_tsnr.")
    for flag, val in ((f"{what}_to", to), (f"{what}_tsnr", tsnr),
                      (f"{what}_max_fwhm", max_fwhm), (f"{what}_nbhd", nbhd)):  # fmt: skip
        if val is not None and val <= 0:
            errors.append(f"{flag} must be positive (got {val:g}).")
    return errors


def _resolve_glm_blur(args) -> tuple[float | None, dict[str, float]]:
    """``(dataset-wide FWHM, {task: FWHM})`` from repeated ``-glm_blur FWHM [TASK]``.

    A bare FWHM is the default for every task no entry names; a task named twice
    keeps the last. Shape errors are refused here rather than reaching a script.
    """
    wide: float | None = None
    per_task: dict[str, float] = {}
    for entry in args.glm_blur or []:
        if len(entry) > 2:
            raise SystemExit(
                "ffs_autoproc: -glm_blur takes a FWHM and at most one TASK "
                f"(got {' '.join(entry)}); repeat the flag for a second task."
            )
        try:
            fwhm = float(entry[0])
        except ValueError:
            raise SystemExit(
                f"ffs_autoproc: -glm_blur {entry[0]!r} is not a FWHM in mm. The "
                "number comes first: -glm_blur 6 ImgStim."
            ) from None
        if len(entry) == 1:
            wide = fwhm
        else:
            task = entry[1]
            per_task[task[len("task-") :] if task.startswith("task-") else task] = fwhm
    return wide, per_task


def _resolve_cut_task_vols(args) -> dict[str, int]:
    """``{task: N}`` from repeated ``-cut_task_vols TASK N``; a task named twice
    keeps the last."""
    out: dict[str, int] = {}
    for task, n in args.cut_task_vols or []:
        try:
            n_keep = int(n)
        except ValueError:
            raise SystemExit(
                f"ffs_autoproc: -cut_task_vols {task} {n!r}: N must be a whole number "
                "of volumes. The task comes first: -cut_task_vols TASK 390."
            ) from None
        if n_keep <= 0:
            raise SystemExit(f"ffs_autoproc: -cut_task_vols {task} {n}: N must be positive.")
        out[task[len("task-") :] if task.startswith("task-") else task] = n_keep
    return out


def _resolve_family_overrides(args) -> dict[str, str]:
    """``{task: label}`` from repeated ``-family TASK LABEL``; a task named twice
    keeps the last."""
    out: dict[str, str] = {}
    for task, label in args.family or []:
        # The label names the family's output folder (fam-<label>).
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", label):
            raise SystemExit(
                f"ffs_autoproc: -family {task} {label!r}: LABEL must be letters, digits, "
                "'-' or '_' (it becomes the folder fam-LABEL)."
            )
        out[task[len("task-") :] if task.startswith("task-") else task] = label
    return out


def _resolve_family_links(args) -> dict[str, str]:
    """``{label: root|anat}`` from repeated ``-family_link LABEL MODE``."""
    out: dict[str, str] = {}
    for label, mode in args.family_link or []:
        if mode not in ("root", "anat"):
            raise SystemExit(f"ffs_autoproc: -family_link {label} {mode!r}: MODE is root or anat.")
        out[label] = mode
    return out


def _report_families(plan) -> None:
    """Print the acquisition families — always, since a wrong grouping is cheap to
    see here and expensive to find in the outputs."""
    fams = plan.families
    if not fams:
        return
    print(f"== acquisition families: {len(fams)} ==", file=sys.stderr)
    for fam in fams:
        how = "  [-family]" if fam.forced else ""
        print(
            f"  {fam.label}: {fam.signature.describe()}  —  {len(fam.members)} run(s), "
            f"task {', '.join(fam.tasks)}{how}",
            file=sys.stderr,
        )
        if fam.link:
            to = {"root": "ROOT", "xfam": "→ root (rigid EPI-EPI)", "anat": "→ anat on its own"}
            print(f"    {to[fam.link]}: {fam.link_reason}", file=sys.stderr)
        if fam.mixed:
            print(
                f"    NOTE: -family put runs with different acquisitions in '{fam.label}';"
                " they share one correction and one output grid.",
                file=sys.stderr,
            )
    if plan.family_plans:
        print(
            "  Each family runs as its own pipeline in "
            + ", ".join(f"{family_dir(f)}/" for f in fams)
            + ".",
            file=sys.stderr,
        )
    elif len(fams) > 1:
        print(
            "  NOTE: no anat for the families to meet at — they run as ONE pipeline at"
            " the first run's voxel size.",
            file=sys.stderr,
        )


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


def _resolve_event_ignore(args) -> tuple[list[str] | None, dict[str, list[str]]]:
    """``(dataset-wide labels, {task: labels})`` from -event_ignore /
    -sep_event_ignore. A task named twice keeps the last list."""
    per_task: dict[str, list[str]] = {}
    for task, *labels in args.sep_event_ignore or []:
        per_task[task[len("task-") :] if task.startswith("task-") else task] = list(labels)
    wide = list(args.event_ignore) if args.event_ignore is not None else None
    return wide, per_task


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


def _report_multiecho(subject, acq_3d: bool) -> None:
    """Say which runs are multi-echo and what happens to them (printed once)."""
    me = [r for s in subject.sessions for r in s.bold_runs if r.is_multiecho]
    if not me:
        return
    tes = sorted({tuple(round(t, 2) for t in r.echo_tes) for r in me})
    how = (
        "per-echo partition shift+stretch in moco, joint locomoco per echo"
        if acq_3d
        else "one in-plane warp for every echo"
    )
    print(
        f"== multi-echo: {len(me)} run(s), TEs {', '.join('/'.join(f'{x:g}' for x in t) for t in tes)} ms — "
        f"estimated from echo 1, {how}, each echo warped at stage10, optimally combined ==",
        file=sys.stderr,
    )


def _resolve_acq_3d(args, subject) -> bool:
    """2-D multi-slice or 3-D, decided once and printed.

    Several stages care (slice timing, anything fit per acquired slice), and a
    wrong guess is silent downstream, so the decision and the evidence for it are
    shown at generation time, like the anat and fmap choices."""
    runs = [r for s in subject.sessions for r in s.bold_runs]
    types = {r.mr_acquisition_type for r in runs} - {None}
    n_st = sum(r.has_slice_timing for r in runs)
    if args.acq != "auto":
        is_3d = args.acq == "3d"
        why = f"-acq {args.acq}"
    elif len(types) == 1:
        is_3d = types == {"3D"}
        why = f"MRAcquisitionType={next(iter(types))} in the sidecars"
    elif len(types) > 1:
        is_3d = False
        why = f"sidecars disagree (MRAcquisitionType {sorted(types)}); treating as 2D"
    elif n_st or args.slicetiming:
        is_3d = False
        why = f"SliceTiming in {n_st} of {len(runs)} sidecars" if n_st else "-slicetiming given"
    else:
        is_3d = True
        why = "no MRAcquisitionType and no SliceTiming in any sidecar: assuming 3D"
    label = "3D (one field state per volume)" if is_3d else "2D multi-slice"
    print(f"== acquisition: {label} — {why} ==", file=sys.stderr)
    if args.acq == "auto" and is_3d and not types:
        print(
            "  Not 3D? The slice information was lost in conversion: pass -acq 2d "
            "(and -slicetiming FILE) and regenerate.",
            file=sys.stderr,
        )
    return is_3d


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
            # it's the difference between two blip inputs and one. A GRE fieldmap
            # is neither — it is measured, so it names the form it was measured in.
            if fg.is_b0:
                kind = (
                    "GRE phasediff"
                    if fg.phasediff_path is not None
                    else "GRE Hz"
                    if fg.fieldmap_path is not None
                    else "GRE phase1/2"
                )
            else:
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


def _resolve_anat_skull(args, presurf: bool) -> bool:
    """Strip the anat in-script? A T1w off the scanner has a skull far more often
    than not, and an unstripped target sends cross-modal lpc to the scalp; -suma's
    brain.nii.gz is the one input known to be stripped already. presurf's head
    always has one, whatever -anat_skull says."""
    if presurf or args.anat_skull == "yes":
        return True
    return args.anat_skull == "auto" and not args.suma


def _resolve_tpm(
    args, *, anat_nonlin: bool, anat_path: str | None, anat_skull: bool
) -> tuple[str | None, bool, bool, str | None, str | None]:
    """(subject tpm, build-from-FreeSurfer, build-from-anat, group tpm, T2 channel).

    Three ways to get the tissue priors ffs_segment's PE warp is driven by, in
    the order they are preferred:

    1. ``-tpm`` — a subject TPM in anat space, already built. Nothing to do.
    2. ``-suma`` — FreeSurfer ran, so aseg.auto's labels are the priors: eight
       hard-edge classes of this subject's own anatomy, the best of the three.
    3. the anat itself, segmented against an SPM-style group TPM. Softer and
       coarser than aseg's labels, but it needs nothing but the T1w that is
       already here — which is the common case, and used to be a hard error.

    (3) needs the head with its skull on, because the group TPM's non-brain
    classes are half of what the EPI fit uses to know where brain is not.
    """
    tpm, fs_tpm, anat_tpm = args.tpm, False, False
    spm_tpm = args.spm_tpm or os.environ.get("FFS_TPM")
    anat_t2 = None
    # -ref_anat segment reads the same subject TPM, on the ref head.
    ref_seg = bool(args.ref_anat) and args.ref_anat_nonlin_method == "segment"
    if not ((anat_nonlin and args.anat_nonlin_method == "segment") or ref_seg) or tpm:
        return tpm, fs_tpm, anat_tpm, spm_tpm, anat_t2
    if args.suma:
        fs_tpm = True
        return "tpm_work/sub_specific_fsTPM.nii.gz", fs_tpm, anat_tpm, spm_tpm, anat_t2
    # A borrowed anat link (-ref_file / -grand_reference) never builds this run's
    # head, so there is nothing here to segment.
    borrowed = bool(args.ref_file or args.grand_reference)
    if not (anat_path and anat_skull and spm_tpm) or borrowed:
        return tpm, fs_tpm, anat_tpm, spm_tpm, anat_t2  # preflight reports why
    anat_tpm = True
    anat_t2 = _resolve_anat_t2(args, anat_path)
    chan = f" (+ {Path(anat_t2).name})" if anat_t2 else ""
    print(
        f"== subject TPM: segmenting the anat{chan} against {spm_tpm} ==\n"
        "  (no -suma: the anat's own posteriors are the tissue priors -anat_nonlin "
        "needs. -suma's FreeSurfer labels are sharper when a recon exists.)",
        file=sys.stderr,
    )
    return "tpm_work/sub_specific_anatTPM.nii.gz", fs_tpm, anat_tpm, spm_tpm, anat_t2


def _resolve_anat_t2(args, anat_path: str) -> str | None:
    """The T2w channel for the anat segmentation: as given, else the one filed
    beside the T1w. ``-anat_t2 none`` says T1 alone.

    Looked for next to an explicit ``-anat FILE`` too, unlike presurf's MP2RAGE
    partners: a second channel only sharpens the priors the segmentation is being
    run for, and the pathological case — a T2w of some other state of the head —
    is what the printed line and ``-anat_t2 none`` are for."""
    from fastfuncstuff.autoproc.bids import t2w_partner

    if args.anat_t2:
        return None if args.anat_t2.lower() == "none" else str(Path(args.anat_t2).resolve())
    t2 = t2w_partner(Path(anat_path))
    return str(t2) if t2 else None


def _resolve_presurf(
    args, anat_paths: list[str], from_bids: bool
) -> tuple[list[tuple[str, str, str | None]], str | None, str]:
    """(one (uni, inv2, inv1) per anat input, the TPM, a line for the report).

    Only for anats the BIDS scan chose: -anat FILE means that file, as it is.
    All inputs or none — a mean of MPRAGEised and raw UNI mixes two contrasts."""
    from fastfuncstuff.autoproc.bids import mp2rage_partners

    if args.presurf == "no" or not from_bids or not anat_paths:
        return [], None, ""
    pairs = [(p, *mp2rage_partners(Path(p))) for p in anat_paths]
    if not any(inv2 for _, inv2, _ in pairs):
        return [], None, ""
    missing = [Path(p).name for p, inv2, _ in pairs if inv2 is None]
    if missing:
        return (
            [],
            None,
            (f"  presurfer skipped: no acq-inv2 beside {', '.join(missing)}; UNI used as found."),
        )
    tpm = args.presurf_tpm or args.spm_tpm or args.tpm or os.environ.get("FFS_TPM")
    if not tpm:
        return (
            [],
            None,
            (
                "  presurfer skipped: MP2RAGE inv2/uni found, but no tissue template. Pass "
                "-presurf_tpm (SPM's tpm/TPM.nii) or set $FFS_TPM; UNI used as found."
            ),
        )
    triples = [(p, str(inv2), str(inv1) if inv1 else None) for p, inv2, inv1 in pairs]
    dn = f", -denoise {args.presurf_denoise}" if args.presurf_denoise != "off" else ""
    return (
        triples,
        tpm,
        (
            f"  presurfer{dn}: MPRAGEise each UNI by its INV2, then mri_synthstrip "
            "(stage09.anat_head / stage09.anat_brain)"
        ),
    )


def _resolve_ref_presurf(args) -> tuple[tuple[str, str, str | None] | None, str | None, str]:
    """(uni, inv2, inv1) for an MP2RAGE -ref_anat, the TPM presurfer needs, a report line.

    Unlike -anat FILE, an explicit -ref_anat still gets its partners looked up:
    the ref is the 7T anat, which is the MP2RAGE case, and its UNI's noise
    background is exactly what the ref's alignment and segmentation must not see."""
    from fastfuncstuff.autoproc.bids import mp2rage_partners

    if not args.ref_anat or args.presurf == "no":
        return None, None, ""
    inv2, inv1 = mp2rage_partners(Path(args.ref_anat))
    inv2 = args.ref_anat_inv2 or (str(inv2) if inv2 else None)
    inv1 = args.ref_anat_inv1 or (str(inv1) if inv1 else None)
    if inv2 is None:
        return None, None, ""
    tpm = args.presurf_tpm or args.spm_tpm or args.tpm or os.environ.get("FFS_TPM")
    if not tpm:
        return (
            None,
            None,
            "  ref presurfer skipped: an INV2 was found for -ref_anat, but no tissue "
            "template. Pass -presurf_tpm (SPM's tpm/TPM.nii) or set $FFS_TPM.",
        )
    return (
        (args.ref_anat, inv2, inv1),
        tpm,
        f"  ref presurfer: MPRAGEise -ref_anat by {Path(inv2).name}",
    )


def _report_anat(anat_paths: list[str], from_bids: bool, skipped: list[Path] | None = None) -> None:
    """Print the T1w the anat step will use — all of them when several are being
    averaged, and the ones the scan saw but rejected.

    Printed for the same reason the events assignment is: "it found the wrong
    anatomical" has to be visible now, not after the last stage. The rejected
    list is what makes that judgeable, and -anat FILE [FILE ...] is the answer
    when the automatic choice is wrong — no code change needed."""
    if not anat_paths:
        return
    src = "scanned from BIDS" if from_bids else "given"
    head = f"== anat ({src})" + (
        f": {len(anat_paths)} T1w, aligned + averaged" if len(anat_paths) > 1 else ""
    )
    print(f"{head} ==", file=sys.stderr)
    for i, p in enumerate(anat_paths, start=1):
        lead = f"  {i}. " if len(anat_paths) > 1 else "  "
        print(
            f"{lead}{p}" + ("   (base)" if i == 1 and len(anat_paths) > 1 else ""), file=sys.stderr
        )
    if skipped:
        print(
            f"  {len(skipped)} other T1w not used (different acq/rec/echo — a repeat of a"
            " DIFFERENT image is not something to average in):",
            file=sys.stderr,
        )
        for p in skipped[:10]:
            print(f"    - {p.name}", file=sys.stderr)
        if len(skipped) > 10:
            print(f"    … and {len(skipped) - 10} more", file=sys.stderr)
        print("  Wrong choice? Pass -anat FILE [FILE ...] to say exactly which.", file=sys.stderr)


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
    "suma",
    "tpm",
    "tpm_source",
    "grand_reference",
    "ref_file",
    "ref_file_anat",
    "ref_anat",
    "ref_anat_inv2",
    "ref_anat_inv1",
)
_PATH_LIST_ARGS = ("anat", "ref_transforms", "events")


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


def _baked_subject(subject: str | None) -> str | None:
    """The FreeSurfer subject as an absolute path, resolved NOW: the script may run
    where $SUBJECTS_DIR is not set. Unresolvable names pass through for validation."""
    from fastfuncstuff.processing.surface_projection import resolve_subject

    if subject is None:
        return None
    try:
        return str(resolve_subject(subject).resolve())
    except FileNotFoundError:
        return subject


def _baked_meshes(meshes: list[str] | None, subject: str | None) -> list[str] | None:
    """Template meshes as absolute folders, for the same reason as the subject; the
    output names (the folder's own name) do not change."""
    from fastfuncstuff.processing.surface_projection import resolve_mesh

    if not meshes or subject is None:
        return meshes
    out = []
    for mesh in meshes:
        try:
            folder = resolve_mesh(mesh, subject)
        except FileNotFoundError:
            folder = None  # validation reports it
        out.append(str(folder.resolve()) if folder is not None else mesh)
    return out


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

    # Backend first, opt strings second: an unset -<stage>_nl_opts means "whatever
    # this backend's baseline is", which is not knowable until the backend is. An
    # explicit override still wins, so a user who passes both gets what they typed
    # even where the two disagree.
    nl_backends = _resolve_nl_backends(args)
    for key in config.NL_STAGE_KEYS:
        if getattr(args, f"{key}_opts", None) is None:
            config.DEFAULT_OPTS[key] = config.nl_stage_opts(nl_backends[f"{key}_backend"], key)

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
        fmap_kind=args.fmap_kind,
    )
    if not subject.sessions:
        print("ERROR: no matching BOLD runs found", file=sys.stderr)
        return 1

    _report_fmap_assignment(subject)
    acq_3d = _resolve_acq_3d(args, subject)
    _report_multiecho(subject, acq_3d)

    anat_paths, tpm_source = _resolve_anat(args)
    unused: list[Path] = []
    if not anat_paths:  # fall back to the scanned T1w(s)
        group = anat_group(subject, args.bids_dir)
        anat_paths = [str(p) for p in group]
        unused = [p for p in anat_candidates(subject, args.bids_dir)[1] if p not in group]
    anat_path = anat_paths[0] if anat_paths else None
    from_bids = not (args.anat or args.suma)
    presurf, presurf_tpm, presurf_note = _resolve_presurf(args, anat_paths, from_bids)
    used = {Path(p) for t in presurf for p in t[1:] if p}
    unused = [p for p in unused if p not in used]

    _report_anat(anat_paths, from_bids=from_bids, skipped=unused)
    if presurf_note:
        print(presurf_note, file=sys.stderr)
    ref_presurf, ref_presurf_tpm, ref_note = _resolve_ref_presurf(args)
    if ref_note:
        print(ref_note, file=sys.stderr)

    go_to_anat = False if args.no_anat else rget("go_to_anat", True)
    fs_subject = _baked_subject(
        args.fs_subject or (str(Path(args.suma).parent) if args.suma else None)
    )
    anat_nonlin = eff(args.anat_nonlin, "anat_nonlin")
    mni_backend, mni_template = _resolve_mni(args)
    if args.do_mni:
        # The template and the engine are both resolved rather than typed by the
        # user, so both get printed: the output space is not a detail.
        print(
            f"== MNI ==\n  template:  {mni_template}\n"
            f"  nonlinear: {config.nl_command(mni_backend)} "
            f"{config.DEFAULT_OPTS['mni_nonlin']}",
            file=sys.stderr,
        )
    event_cols, event_cols_by_task = _resolve_event_cols(args)
    event_ignore, event_ignore_by_task = _resolve_event_ignore(args)
    glm_blur, sep_glm_blur = _resolve_glm_blur(args)
    round_onsets, sep_round_onsets = _resolve_round_modes(args.round_onsets, "-round_onsets")
    round_durations, sep_round_durations = _resolve_round_modes(
        args.round_durations, "-round_durations"
    )

    anat_skull = _resolve_anat_skull(args, bool(presurf and go_to_anat))
    tpm, fs_tpm, anat_tpm, spm_tpm, anat_t2 = _resolve_tpm(
        args,
        anat_nonlin=anat_nonlin,
        anat_path=anat_path if go_to_anat else None,
        anat_skull=anat_skull,
    )

    opt = Options(
        recipe=args.recipe,
        want_nordic=eff(args.want_nordic, "want_nordic"),
        phase_proc=args.phase_proc,
        phase_drift=not args.no_phase_drift,
        medic=args.medic,
        medic_inplane_locomoco=args.medic_inplane_locomoco,
        locomoco_phase=args.locomoco_phase,
        phasereg=args.phasereg,
        phasereg_opts=args.phasereg_opts,
        phasereg_me=args.phasereg_me,
        phasereg_veins=args.phasereg_veins,
        hrftdv=args.hrftdv,
        hrftdv_model=args.hrftdv_model,
        hrftdv_opts=args.hrftdv_opts,
        tissue=[str(Path(t).resolve()) for t in args.tissue] if args.tissue else None,
        acq_3d=acq_3d,
        noise_vols=args.noise_vols,
        cut_task_vols=_resolve_cut_task_vols(args),
        family_overrides=_resolve_family_overrides(args),
        family_ref=args.family_ref or "anat",
        family_links=_resolve_family_links(args),
        xfam_min_overlap=(0.5 if args.xfam_min_overlap is None else args.xfam_min_overlap),
        slicetiming_method=_resolve_slicetiming(args, rget, subject),
        tzero=float(args.tzero if args.tzero is not None else rget("tzero", 0.0)),
        slicetiming_file=args.slicetiming,
        tr=args.tr,
        distortion=(False if args.no_distortion else rget("distortion", True)),
        run_glm=(False if args.no_glm else rget("run_glm", True)),
        glm_ortvec=_resolve_glm_ortvec(args, recipe),
        glm_stim_vec=[(lbl, path) for lbl, path in (args.glm_stim_vec or [])],
        glm_opts=args.glm_opts or "",
        glm_drop_first=args.glm_drop_first,
        glm_drop_last=args.glm_drop_last,
        glm_blur=glm_blur,
        sep_glm_blur=sep_glm_blur,
        glm_label=args.glm_label,
        glm_mask=args.glm_mask,
        clustsim=args.clustsim,
        clustsim_null=args.clustsim_null,
        glm_blur_to=args.glm_blur_to,
        glm_blur_local=args.glm_blur_local,
        glm_blur_tsnr=args.glm_blur_tsnr,
        glm_blur_max_fwhm=args.glm_blur_max_fwhm,
        glm_blur_nbhd=args.glm_blur_nbhd,
        glm_spec_overwrite=args.glm_spec_overwrite,
        spec_event_cols=event_cols,
        sep_spec_event_cols=event_cols_by_task,
        event_ignore=event_ignore,
        sep_event_ignore=event_ignore_by_task,
        round_onsets=round_onsets,
        sep_round_onsets=sep_round_onsets,
        round_durations=round_durations,
        sep_round_durations=sep_round_durations,
        event_filters=_resolve_event_filters(args),
        prebuilt_contrasts=_resolve_prebuilt_contrasts(args),
        motsim=eff(args.motsim, "motsim", None),
        censor_motion=eff(args.censor_motion, "censor_motion", None),
        censor_outliers=eff(args.censor_outliers, "censor_outliers", None),
        locomoco=eff(args.locomoco, "locomoco"),
        nordic_task_rescue=args.nordic_task_rescue,
        nordic_dof_adjust=args.nordic_dof_adjust,
        nordic_task_qc=args.nordic_task_qc,
        locomoco_task_qc=args.locomoco_task_qc,
        locomoco_detask=args.locomoco_detask,
        xrun_nonlin=eff(args.xrun_nonlin, "xrun_nonlin"),
        xfmap_nonlin=eff(args.xfmap_nonlin, "xfmap_nonlin"),
        xfam_nonlin=eff(args.xfam_nonlin, "xfam_nonlin"),
        xses_nonlin=eff(args.xses_nonlin, "xses_nonlin"),
        xrun_nonlin_in_source=eff(args.xrun_nonlin_in_source, "xrun_nonlin_in_source"),
        xfmap_nonlin_in_source=eff(args.xfmap_nonlin_in_source, "xfmap_nonlin_in_source"),
        xses_nonlin_in_source=eff(args.xses_nonlin_in_source, "xses_nonlin_in_source"),
        **nl_backends,
        ref_ses=args.ref_ses,
        fmap_ref=args.fmap_ref,
        ref_image=args.ref_image,
        go_to_anat=go_to_anat,
        final_dxyz=args.final_dxyz,
        anat_nonlin=anat_nonlin,
        # -ref_image is the answer for every level; -anat_source overrides it for
        # the anat step alone.
        anat_source=args.anat_source or args.ref_image or "auto",
        # presurf's head still has a skull; synthstrip is what makes the brain.
        anat_skull=anat_skull,
        presurf=presurf if go_to_anat else [],
        presurf_denoise=args.presurf_denoise,
        presurf_tpm=presurf_tpm or ref_presurf_tpm,
        do_mni=args.do_mni,
        mni_backend=mni_backend,
        mni_template=mni_template,
        anat_nonlin_input=args.anat_nonlin_input,
        anat_nonlin_method=args.anat_nonlin_method,
        ref_anat_nonlin_method=args.ref_anat_nonlin_method,
        ref_presurf=ref_presurf if go_to_anat else None,
        anat_path=anat_path if go_to_anat else None,
        anat_extra=anat_paths[1:] if go_to_anat else [],
        moco_ref=args.moco_ref,
        grand_reference=args.grand_reference,
        grand_reference_nonlin=args.grand_reference_nonlin,
        tpm=tpm,
        tpm_source=tpm_source,
        fs_tpm=fs_tpm,
        anat_tpm=anat_tpm,
        spm_tpm=spm_tpm,
        anat_t2=anat_t2,
        suma_dir=args.suma,
        surface_target=_baked_meshes(args.surface_target, fs_subject),
        fs_subject=fs_subject,
        surface_qc=args.surface_qc,
        surface_glm=args.surface_glm,
        surface_glm_blur=args.surface_glm_blur,
        surface_glm_blur_to=args.surface_glm_blur_to,
        surface_glm_blur_local=args.surface_glm_blur_local,
        surface_glm_blur_tsnr=args.surface_glm_blur_tsnr,
        surface_glm_blur_max_fwhm=args.surface_glm_blur_max_fwhm,
        surface_glm_blur_nbhd=args.surface_glm_blur_nbhd,
        surface_clustsim=args.surface_clustsim,
        surface_clustsim_niter=args.surface_clustsim_niter,
        surface_glm_opts=args.surface_glm_opts,
        ref_file=args.ref_file,
        ref_transforms=args.ref_transforms,
        ref_file_anat=args.ref_file_anat,
        ref_anat=args.ref_anat,
        events=args.events,
    )
    opt.nordic_save_resid = args.nordic_save_resid  # ty: ignore[unresolved-attribute]  # emitter reads via getattr
    opt.batch_overwrite = args.batch_overwrite
    opt.qc = not args.no_qc
    # Output formats: keep the Options (config) default unless the user set one.
    if args.fmt is not None:
        opt.fmt = args.fmt
    if args.final_fmt is not None:
        opt.final_fmt = args.final_fmt
    if args.glm_fmt is not None:
        opt.glm_fmt = args.glm_fmt
    if args.device is not None:
        opt.device = args.device

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

    try:
        plan = build_plan(subject, opt)
    except ValueError as exc:
        print(f"ffs_autoproc: {exc}\nnothing written.", file=sys.stderr)
        return 1
    _report_families(plan)

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
    if plan.family_plans:
        spec_rows = [
            row
            for fam, sub in plan.family_plans
            for row in write_design_specs(sub, args.bids_dir, str(Path(work_dir) / family_dir(fam)))
        ]
    else:
        spec_rows = write_design_specs(plan, args.bids_dir, work_dir)

    tag = f" [{args.recipe}]" if args.recipe else ""
    print(f"wrote {out_path}{tag}  ({len(plan.runs)} run(s), ref session {plan.ref_session})")
    print(f"  working dir baked in: {work_dir}")
    failed = [(t, st) for t, _, st in spec_rows if st.startswith("error")]
    for task, path, status in spec_rows:
        note = {
            "wrote": "design spec written — EDIT IT (contrasts, HRF) before the GLM stage",
            "kept": "design spec already exists, left untouched (-glm_spec_overwrite to replace)",
        }.get(status, status)
        print(f"  task-{task}: {Path(path).name}  [{note}]")
    if failed:
        for task, status in failed:
            print(f"ERROR task-{task}: {status[len('error: ') :]}", file=sys.stderr)
        print(
            "ffs_autoproc: design spec(s) not written — fix the above and re-run.",
            file=sys.stderr,
        )
        return 1
    n_stim = len(list((Path(work_dir) / STIMULI_DIR).glob("*.tsv")))
    if n_stim:
        print(f"  {n_stim} events TSV(s) copied to {STIMULI_DIR}/ (what the specs name)")
    log_path = out_path.with_suffix(".log")
    run_cmd = f"bash {shlex.quote(str(out_path))} 2>&1 | tee {shlex.quote(str(log_path))}"
    if not args.exec:
        print("  read it, edit it, then run it:")
        print(f"    {run_cmd}")
        return 0

    print(f"  running: {run_cmd}")
    # bash -c, not the fish/whatever login shell: the pipe and pipefail are bash
    # syntax, and this is the exact command just printed above -- a user who
    # copy-pastes it gets identical behaviour, log included, regardless of shell.
    proc = subprocess.run(
        ["bash", "-c", f"set -o pipefail; {run_cmd}"],
    )
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
