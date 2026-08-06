#!/usr/bin/env python3
"""ffs_pyrf - GPU population receptive field (pRF) mapping.

A pyre for receptive fields: the compressive spatial summation (CSS) pRF model,
fit on the GPU.

This is a reimplementation of the *method* published by Kendrick Kay and
colleagues, and the model, the super-grid seeding strategy, the staged fit, and
the parameter conventions are all theirs:

    Kay KN, Winawer J, Mezer A, Wandell BA (2013).
    Compressive spatial summation in human visual cortex.
    Journal of Neurophysiology 110(2), 481-494.
    https://doi.org/10.1152/jn.00105.2013

    analyzePRF (MATLAB), by Kendrick Kay - http://kendrickkay.net/analyzePRF/
    Copyright (c) 2014 Kendrick Kay. Licensed CC BY 3.0 Unported.

**If you publish results from this tool, cite Kay et al. 2013.** The reference
toolbox's licence asks for that citation, and it is the right thing regardless.

What is ours is the implementation, not the idea: batched analytic Gauss-Newton
with variable projection over the gain, per-voxel HRF selection, fold-local
leave-one-run-out cross-validation, and the chunking that keeps it on the GPU.
Numerical agreement with the reference is checked in ``tests/test_prf.py``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import torch

from fastfuncstuff.cli_utils import (
    add_load_threads_arg,
    add_ortvec_arguments,
    add_verbose_arg,
    auto_polort,
    build_nuisance_per_run,
    collect_nuisance_blocks,
    compute_run_lengths,
    get_average_run_duration,
    load_and_preprocess_runs,
    parse_device_arg,
    parse_input_files,
    parse_prefix,
)
from fastfuncstuff.denoise.sequential import extract_noise_pcs_per_run
from fastfuncstuff.design.hrf import (
    get_hrf_library,
    load_canonical_hrf_basic,
    load_canonical_hrf_library,
)
from fastfuncstuff.design.prf import (
    PRFRefinedFit,
    PRFRefinementConfig,
    downsample_aperture,
    fit_prf_loro,
    fit_prf_supergrid,
    grid_seeds_as_fit,
    make_analyzeprf_grid,
    refine_prf_all_hrfs,
    refine_prf_hrf_window,
    refine_prf_supergrid,
    screen_voxels_ridge,
    select_noise_pc_count,
    summarize_hrf_selection,
)
from fastfuncstuff.io.afni import load_nifti, save_nifti
from fastfuncstuff.memory import estimate_chunk_size
from fastfuncstuff.utils import configure_torch_backends

_EPILOG = """
OUTPUT
  Everything lands in one labeled 4D bucket at {prefix}. Spatial quantities are
  in degrees of visual angle when -screen_extent is given, and in aperture
  pixels otherwise -- pixels are not comparable across studies, or even across
  -stim_downsample settings, so pass -screen_extent for anything you will share.

    x, y            pRF center relative to the aperture center. y is positive
                    upward in the visual field.
    angle           polar angle in degrees; 0 = right, increasing toward y.
                    NaN where eccentricity is exactly 0.
    eccentricity    distance from the aperture center, hypot(x, y).
    sigma           Gaussian standard deviation of the pRF, before the CSS
                    exponent is applied.
    exponent        CSS compressive exponent n. Fixed at 1 for -model_mode 3.
    rfsize          sigma / sqrt(n) -- THE size measure to report. sigma and n
                    are individually unidentifiable (the objective has a flat
                    valley where growing sigma trades against shrinking n), so
                    validating against raw sigma will look broken when nothing
                    is.
    gain            amplitude of the fitted response, in the units of the input.
                    Divide by meanvol for percent signal change.
    meanvol         each voxel's mean over time, the scale gain refers to.
    correlation     correlation between prediction and data after nuisance
                    projection; r2 is the coefficient of determination.
    xval_r2         held-out R2 from leave-one-run-out (multi-run input only).
                    See -xval_hrf: by default the folds pick their own HRF at
                    grid resolution, so this scores a slightly different model
                    than the other sub-bricks report.
    hrf_index       selected HRF, ONE-BASED, into the library actually used.
    hrf_index_continuous, hrf_evidence
                    (-hrf_select refine only) parabolic sub-step interpolation
                    of the R2-vs-HRF curve, and how peaked that curve is.
                    hrf_evidence near zero means the library is indistinguishable
                    for that voxel -- threshold on it before believing hrf_index.
    grid_index      winning super-grid candidate, one-based.
    residual_ss, gn_iterations, gn_converged
                    fit diagnostics. A voxel that never converged has
                    gn_iterations equal to -maxiter; that is common and not by
                    itself a failure.

  Optional extra files: {prefix}_canonical (-save_canonical) holds the same
  bucket fit with the canonical HRF forced, for a like-for-like comparison
  against the selected-HRF fit; {prefix}_hrf_r2 (-save_hrf_r2) holds the raw
  per-voxel x per-HRF R2 matrix behind the HRF choice.

CHOOSING SETTINGS  (measured on 3 runs x 300 TRs, one subject -- confirm on yours)
  -hrf_select     'grid+1' is the sweet spot: it agrees with the full 20-HRF
                  search on 96% of well-fit voxels at equal R2, for well under
                  half the time. Plain 'grid' is cheaper again and still agrees
                  ~84% of the time; the grid choice is within one library step
                  of the refined one essentially always, which is exactly why
                  grid+1 works and grid+2 adds nothing.
  -grid_angles    32 costs no measurable time (the super-grid stage is
                  launch-bound, not candidate-bound) and clearly beats
                  analyzePRF's 16. Subdividing sigma or eccentricity instead
                  sharpens the seeds but does NOT change the refined fit.
  -stim_downsample
                  refinement cost and memory scale with aperture AREA. A native
                  1080x1080 aperture is ~136x the work of the ~100x100 the
                  reference resizes to, and is not usable as-is.
  -model_mode     3 (linear Gaussian, exponent fixed at 1) is the classic
                  Dumoulin-Wandell model and the natural comparison baseline,
                  but it is NOT meaningfully faster: cost is dominated by the
                  per-voxel Gaussian over the aperture, which every mode pays.
  -mask           cost is linear in voxels, so masking is a large lever.
  -denoise        data-derived noise components, GLMdenoise style, with the
                  noise pool taken from the screening pass rather than an
                  anatomical guess. How many components to keep is chosen by
                  cross-validation ON THE SUPER-GRID FIT -- seconds per
                  candidate count, where refining the full model at every count
                  and fold would be tens of fits -- and zero is a legitimate
                  answer when denoising does not earn its degrees of freedom.
                  Needs 2+ runs. -save_denoise writes the sweep figure.
  -screen_top     larger still, and functional rather than anatomical: a linear
                  ridge pRF is fitted to every voxel in seconds, and only the
                  best-screening ones get the CSS refinement. Whole brain,
                  496k voxels, canonical HRF: 5:48 -> 20 s at -screen_top 0.1,
                  retaining 98% of the voxels the full fit gives R2 > 0.4 and
                  99.8% of those above 0.5. -save_screen writes the map, which
                  is usable on its own as a functionally derived mask.

Method and conventions are from Kay, Winawer, Mezer & Wandell (2013),
'Compressive spatial summation in human visual cortex', J Neurophysiol
110(2):481-494, doi:10.1152/jn.00105.2013 -- and the analyzePRF MATLAB toolbox
by Kendrick Kay (http://kendrickkay.net/analyzePRF/, Copyright (c) 2014
Kendrick Kay, CC BY 3.0).
PLEASE CITE Kay et al. 2013 if you publish results from this tool.
"""


class _PyrfHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    """Show each flag's default, but leave the epilog's own layout alone."""


def create_parser() -> argparse.ArgumentParser:
    """Create the ffs_pyrf command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "ffs_pyrf - GPU compressive-spatial-summation (CSS) population receptive "
            "field mapping. A pyre for receptive fields."
        ),
        epilog=_EPILOG,
        formatter_class=_PyrfHelpFormatter,
    )
    required = parser.add_argument_group("required arguments")
    required.add_argument("-input", nargs="+", required=True, help="fMRI run(s), one file per run")
    required.add_argument(
        "-stimulus",
        nargs="+",
        default=None,
        help=(
            "Aperture source(s), one per input run: a row x column x time NIfTI/.npy movie "
            "or a directory of TR-aligned PNG frames"
        ),
    )
    required.add_argument(
        "-stim-pngs",
        "-stim_pngs",
        dest="stim_sources",
        action="append",
        default=[],
        metavar="DIR",
        help="PNG frame directory for one input run; repeat in -input order",
    )
    required.add_argument(
        "-stim-pngs-multi",
        "-stim_pngs_multi",
        dest="stim_sources",
        action="append",
        nargs=2,
        metavar=("DIR", "N_RUNS"),
        help=(
            "Naturally ordered PNG frames from DIR covering the next N_RUNS input runs, "
            "either concatenated and split between them or one run's worth reused for "
            "all of them. The frame count decides which"
        ),
    )
    required.add_argument(
        "-stim-nii",
        "-stim_nii",
        dest="stim_sources",
        action="append",
        metavar="FILE",
        help=(
            "Aperture NIfTI (row x column [x 1] x time) for one input run; repeat in "
            "-input order. Interchangeable with -stim-pngs and evaluated in the order given"
        ),
    )
    required.add_argument(
        "-stim-nii-multi",
        "-stim_nii_multi",
        dest="stim_sources",
        action="append",
        nargs=2,
        metavar=("FILE", "N_RUNS"),
        help=(
            "Aperture NIfTI covering the next N_RUNS input runs. Holding those runs' "
            "frames concatenated, it is split between them; holding exactly one run's "
            "worth, the same frames are reused for every one of them (a repeated "
            "identical sweep). The frame count decides which"
        ),
    )
    required.add_argument(
        "-stim-downsample",
        "-stim_downsample",
        type=int,
        default=100,
        metavar="PIXELS",
        help=(
            "Block-average each aperture axis down to about this many pixels, using the "
            "nearest factor that divides evenly (1080 -> 108). Refinement cost scales "
            "with aperture area, so full-resolution apertures are ruinously slow. "
            "Use 0 to keep the native resolution"
        ),
    )
    required.add_argument("-prefix", required=True, help="Output prefix")

    model = parser.add_argument_group("pRF model")
    model.add_argument(
        "-tr", type=float, default=None, help="TR in seconds; defaults to the input header"
    )
    model.add_argument(
        "-screen-extent",
        "-screen_extent",
        "-screen-deg",
        "-screen_deg",
        dest="screen_extent",
        type=float,
        default=None,
        metavar="DEGREES",
        help=(
            "Degrees of visual angle spanned by the FULL WIDTH of the stimulus "
            "aperture (so half of it is the largest eccentricity reachable along "
            "the horizontal). Reports x, y, sigma, eccentricity, and rfsize in "
            "degrees instead of aperture pixels"
        ),
    )
    model.add_argument(
        "-hrf",
        "-hrf-mode",
        "-hrf_mode",
        dest="hrf_mode",
        choices=["canonical", "library", "pighs"],
        default="library",
        help=(
            "HRF source. 'canonical' is a single fixed double-gamma, which skips "
            "per-HRF selection entirely and is by far the fastest. 'library' is the "
            "20-HRF double-gamma family (or -hrf_library); pair it with "
            "-hrf_select grid+1. 'pighs' generates a half-cosine family of "
            "-num_hrfs shapes"
        ),
    )
    model.add_argument(
        "-num-hrfs",
        "-num_hrfs",
        "-n_hrfs",
        dest="num_hrfs",
        type=int,
        default=None,
        help=(
            "How many HRFs to fit. For -hrf library this evenly subsamples the "
            "library (fewer, more widely spaced shapes); for -hrf pighs it is the "
            "number generated. Defaults to the whole library / 20 PIGHS shapes"
        ),
    )
    model.add_argument(
        "-hrf-library",
        "-hrf_library",
        dest="hrf_library",
        default=None,
        help="Custom column-wise HRF TSV, e.g. from ffs_librarian (-hrf library only)",
    )
    model.add_argument(
        "-save-canonical",
        "-save_canonical",
        action="store_true",
        help=(
            "Also fit the canonical HRF and write it to {prefix}_canonical. The "
            "canonical is appended to the library, so this costs one extra "
            "refinement pass and lets the HRF-selected and fixed-HRF fits be "
            "compared voxelwise on identical data"
        ),
    )
    model.add_argument(
        "-hrf-duration", "-hrf_duration", type=float, default=32.0, help="HRF duration in seconds"
    )
    model.add_argument(
        "-grid-angles",
        "-grid_angles",
        dest="grid_angles",
        type=int,
        default=32,
        help=(
            "Polar angles in the super-grid. The highest-value density knob: "
            "doubling it from analyzePRF's 16 costs no measurable time (the grid "
            "stage is launch-bound, not candidate-bound) and measurably improves "
            "both the HRF choice and the seed position"
        ),
    )
    model.add_argument(
        "-grid-angle-mode",
        "-grid_angle_mode",
        dest="grid_angle_mode",
        choices=["uniform", "arc"],
        default="uniform",
        help=(
            "How -grid_angles is spread over the eccentricity rings. 'uniform' is "
            "analyzePRF's: the same count on every ring. 'arc' scales the count with "
            "ring radius for constant pixel spacing, so -grid_angles applies to the "
            "OUTERMOST ring. 'arc' sounds better but measured WORSE (82.0%% vs 84.2%% "
            "HRF agreement at matched cost): pRF size grows with eccentricity, so "
            "uniform angles are already evenly spaced in units of pRF width, and arc "
            "leaves mid-eccentricity rings with only 2-5 angles"
        ),
    )
    model.add_argument(
        "-grid-sigma-mode",
        "-grid_sigma_mode",
        dest="grid_sigma_mode",
        choices=["absolute", "slope"],
        default="absolute",
        help=(
            "How super-grid pRF sizes are sampled. 'absolute' is analyzePRF's fixed "
            "sigma ladder at every eccentricity, which spends candidates on sizes "
            "anatomy rules out (a 64-px pRF at the fovea). 'slope' samples the slope "
            "of the linear size-vs-eccentricity relationship instead, so each ring "
            "gets the sizes plausible there (-grid_sigma_steps does not apply). "
            "Measured: HRF agreement 84.2%% -> 85.1%% for 22%% more candidates"
        ),
    )
    model.add_argument(
        "-grid-sigma-steps",
        "-grid_sigma_steps",
        dest="grid_sigma_steps",
        type=int,
        default=1,
        help="Super-grid sigma samples per octave (1 = analyzePRF's powers of two)",
    )
    model.add_argument(
        "-grid-ecc-steps",
        "-grid_ecc_steps",
        dest="grid_ecc_steps",
        type=int,
        default=1,
        help="Super-grid rings between each pair of analyzePRF's reference eccentricities",
    )
    model.add_argument(
        "-candidate-chunk",
        "-candidate_chunk",
        type=int,
        default=256,
        help="Spatial/CSS candidates evaluated per GPU prediction batch",
    )
    model.add_argument(
        "-batch-size",
        "-batch_size",
        type=int,
        default=None,
        help="Voxel chunk size; defaults to the shared device-aware memory estimate",
    )
    model.add_argument(
        "-polort",
        type=int,
        default=None,
        help=(
            "Per-run Legendre polynomial degree; defaults to AFNI-style automatic "
            "selection. Use -1 for no drift terms at all (analyzePRF's NaN case)"
        ),
    )
    model.add_argument("-maxiter", type=int, default=50, help="Maximum CSS Gauss-Newton iterations")
    model.add_argument(
        "-expt-lower-bound",
        "-expt_lower_bound",
        type=float,
        default=1e-3,
        help="Lower bound on the CSS exponent (analyzePRF exptlowerbound)",
    )
    model.add_argument(
        "-hrf-select",
        "-hrf_select",
        choices=["refine", "grid", "grid+1", "grid+2"],
        default="refine",
        help=(
            "How the per-voxel HRF is chosen. 'refine' refits the pRF under every "
            "HRF and keeps the best. 'grid' keeps the super-grid's choice, which "
            "matches the refined one ~84%% of the time on well-fit voxels. 'grid+N' "
            "refits only the HRFs within N steps of the grid choice: grid+1 reaches "
            "~96%% agreement at equal R2 for well under half the time, and is the "
            "recommended setting. grid+2 measured identical to grid+1, because the "
            "grid choice is within one library step essentially always. The window "
            "slides at the library edges, so a voxel that picked HRF 1 is still "
            "scored on three HRFs"
        ),
    )
    model.add_argument(
        "-save-hrf-r2",
        "-save_hrf_r2",
        action="store_true",
        help=(
            "Write the full per-voxel x per-HRF R2 map to {prefix}_hrf_r2, the raw "
            "input to any HRF-selection criterion (-hrf_select refine only)"
        ),
    )
    model.add_argument(
        "-quick",
        action="store_true",
        help=(
            "Skip Gauss-Newton refinement and return the super-grid seeds "
            "(analyzePRF seedmode -2); disables cross-validation"
        ),
    )
    model.add_argument(
        "-gn-damping",
        "-gn_damping",
        type=float,
        default=1e-3,
        help="Initial relative Levenberg damping",
    )
    model.add_argument(
        "-gn-step-tol",
        "-gn_step_tol",
        type=float,
        default=1e-4,
        help="CSS parameter convergence tolerance",
    )
    model.add_argument(
        "-xval",
        choices=["auto", "none", "loro"],
        default="auto",
        help="Cross-validation: auto=LORO for multiple runs, none, or explicit LORO",
    )
    model.add_argument(
        "-xval-hrf",
        "-xval_hrf",
        dest="xval_hrf",
        choices=["grid", "fixed", "refine"],
        default="grid",
        help=(
            "How each cross-validation fold picks its HRF. 'grid' lets the fold's "
            "own super-grid choose: leak-free, but grid-level HRF choice is near "
            "chance, so xval_r2 scores a worse model than the other sub-bricks "
            "report. 'fixed' reuses the full fit's per-voxel HRF, which scores the "
            "reported model at no extra cost but leaks one discrete choice from the "
            "held-out run. 'refine' refits every HRF inside each fold: exact and "
            "leak-free, at roughly n_hrfs times the cross-validation cost"
        ),
    )
    model.add_argument(
        "-model-mode",
        "-model_mode",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help=(
            "1=staged CSS (exponent frozen for the first half of the iterations), "
            "2=direct CSS, 3=fixed-exponent linear Gaussian pRF. Mode 3 is the "
            "classic Dumoulin-Wandell model and the natural comparison baseline; it "
            "is not meaningfully faster, since the per-voxel Gaussian over the "
            "aperture dominates every mode"
        ),
    )
    add_ortvec_arguments(model)

    screening = parser.add_argument_group("screening (fast linear pre-pass)")
    screening.add_argument(
        "-screen",
        type=float,
        default=None,
        metavar="R2",
        help=(
            "Fit a fast linear pRF (ridge onto a random hashed-Gaussian basis) to "
            "every voxel first, and refine only those whose cross-validated R2 "
            "exceeds this. Seconds for a whole brain, and CSS refinement cost is "
            "linear in voxels, so this is the big lever on unmasked data. Excluded "
            "voxels are written as NaN. Try 0.0 to keep anything the linear model "
            "predicts at all"
        ),
    )
    screening.add_argument(
        "-screen-top",
        "-screen_top",
        dest="screen_top",
        type=float,
        default=None,
        metavar="FRACTION",
        help="Instead of a threshold, keep this fraction of the best-screening voxels",
    )
    screening.add_argument(
        "-save-screen",
        "-save_screen",
        action="store_true",
        help="Write the screening R2 map to {prefix}_screen -- a functionally derived mask",
    )
    screening.add_argument(
        "-denoise",
        action="store_true",
        help=(
            "Project data-derived noise components out of the fit (GLMdenoise "
            "style). The noise pool is the voxels the screening pass says carry no "
            "stimulus response at all; components are taken per run, and how many "
            "to keep is chosen by cross-validation -- including the option of zero, "
            "when denoising does not earn its degrees of freedom. Needs 2+ runs"
        ),
    )
    screening.add_argument(
        "-max-pcs",
        "-max_pcs",
        dest="max_pcs",
        type=int,
        default=10,
        help="Largest number of noise components to consider",
    )
    screening.add_argument(
        "-noise-pool-r2",
        "-noise_pool_r2",
        dest="noise_pool_r2",
        type=float,
        default=0.0,
        help=(
            "Screening R2 below which a voxel joins the noise pool. The default 0 "
            "takes only voxels the linear pRF model fits WORSE than their own mean, "
            "which is a strong statement that there is no stimulus response to "
            "leak into the components"
        ),
    )
    screening.add_argument(
        "-denoise-tolerance",
        "-denoise_tolerance",
        dest="denoise_tolerance",
        type=float,
        default=0.05,
        help=(
            "Keep the fewest components within this fraction of the best "
            "cross-validated improvement, rather than the noisy argmax"
        ),
    )
    screening.add_argument(
        "-save-denoise",
        "-save_denoise",
        action="store_true",
        help=(
            "Write the noise-pool mask to {prefix}_noisepool, the components to "
            "{prefix}_noisepcs.1D, and the component-count sweep to {prefix}_denoise.png"
        ),
    )
    screening.add_argument(
        "-screen-tiles",
        "-screen_tiles",
        dest="screen_tiles",
        type=int,
        default=250,
        help="Random Gaussian tiles in the screening basis",
    )

    processing = parser.add_argument_group("processing")
    processing.add_argument("-mask", default=None, help="Optional brain mask")
    processing.add_argument(
        "-do-scale", "-do_scale", action="store_true", help="Scale to percent signal"
    )
    processing.add_argument("-device", default=None, help="Compute device, e.g. cuda or cpu")
    processing.add_argument(
        "-keep-on-cpu",
        "-keep_on_cpu",
        action="store_true",
        help="Stream CPU-held data to the compute device",
    )
    add_load_threads_arg(processing)
    add_verbose_arg(processing, default=1)
    return parser


def _natural_path_key(path: Path) -> list[int | str]:
    """Sort frame names numerically, so ``frame_2.png`` precedes ``frame_10.png``."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _load_png_directory(path: Path) -> np.ndarray:
    """Load a naturally ordered PNG frame directory as a row x column x time movie."""
    frame_paths = sorted(
        (frame for frame in path.iterdir() if frame.is_file() and frame.suffix.lower() == ".png"),
        key=_natural_path_key,
    )
    if not frame_paths:
        raise FileNotFoundError(f"No PNG frames found in stimulus directory {path}")

    frames: list[np.ndarray] = []
    frame_shape: tuple[int, int] | None = None
    for frame_path in frame_paths:
        frame = mpimg.imread(frame_path)
        if frame.ndim == 3 and frame.shape[-1] in (3, 4):
            frame = np.tensordot(
                frame[..., :3], np.array([0.2126, 0.7152, 0.0722]), axes=([-1], [0])
            )
        if frame.ndim != 2:
            raise ValueError(
                f"PNG stimulus frame {frame_path} must be grayscale or RGB/RGBA, got {frame.shape}"
            )
        if frame_shape is not None and frame.shape != frame_shape:
            raise ValueError(
                f"PNG stimulus frame {frame_path} has shape {frame.shape}; expected {frame_shape}"
            )
        frame_shape = frame.shape
        frames.append(np.asarray(frame, dtype=np.float32))

    return np.stack(frames, axis=-1)


def _load_stimulus_run(path: str, downsample: int = 0) -> tuple[torch.Tensor, tuple[int, int]]:
    """Load a TR-aligned aperture movie or PNG directory as time-by-pixel samples."""
    stimulus_path = Path(path)
    if stimulus_path.is_dir():
        movie = _load_png_directory(stimulus_path)
    elif stimulus_path.suffix.lower() == ".npy":
        movie = np.load(stimulus_path)
    else:
        movie = load_nifti(stimulus_path).get_fdata(dtype=np.float32)
    if movie.ndim == 4:
        # An aperture written as a NIfTI volume is row x column x 1 x time: the
        # slice axis only exists because NIfTI has no 2D-plus-time layout.
        singleton_axes = tuple(axis for axis in range(3) if movie.shape[axis] == 1)
        if len(singleton_axes) == 1:
            movie = movie.reshape(
                tuple(size for axis, size in enumerate(movie.shape) if axis != singleton_axes[0])
            )
    if movie.ndim != 3:
        raise ValueError(
            f"Stimulus {stimulus_path} must be a row x column x time movie, got {movie.shape}"
        )
    rows, columns, n_timepoints = movie.shape
    if min(rows, columns, n_timepoints) < 1:
        raise ValueError(f"Stimulus {stimulus_path} has an empty dimension: {movie.shape}")
    aperture = torch.from_numpy(np.ascontiguousarray(movie, dtype=np.float32))
    if downsample > 0:
        aperture, (rows, columns) = downsample_aperture(aperture, downsample)
    frames = aperture.permute(2, 0, 1).reshape(n_timepoints, rows * columns)
    return frames.contiguous(), (rows, columns)


def _load_stimulus_sources(
    sources: list[str | list[str]],
    run_lengths: list[int],
    downsample: int = 0,
    verbose: bool = False,
) -> tuple[list[torch.Tensor], tuple[int, int]]:
    """Load ordered PNG/NIfTI sources and map multi-run ones onto their input runs.

    A source claiming ``N_RUNS`` runs is either the runs' frames concatenated, or
    one run's frames to be reused by all of them -- the two cases retinotopy
    actually produces, and the frame count tells them apart unambiguously. A
    repeated identical sweep (the common design) is the reuse case.
    """
    runs: list[torch.Tensor] = []
    stimulus_shape: tuple[int, int] | None = None
    for source in sources:
        if isinstance(source, str):
            location, n_runs = source, 1
        else:
            location, n_runs_text = source
            try:
                n_runs = int(n_runs_text)
            except ValueError as error:
                raise ValueError(
                    f"Stimulus run count for {location} must be an integer, got {n_runs_text!r}"
                ) from error
        if n_runs < 1:
            raise ValueError(f"Stimulus run count for {location} must be positive, got {n_runs}")
        if len(runs) + n_runs > len(run_lengths):
            raise ValueError(
                f"Stimulus source {location} maps beyond the {len(run_lengths)} input runs"
            )

        frames, shape = _load_stimulus_run(location, downsample)
        if stimulus_shape is not None and shape != stimulus_shape:
            raise ValueError(
                f"Stimulus source {location} has shape {shape}; expected {stimulus_shape}"
            )
        stimulus_shape = shape
        mapped_lengths = run_lengths[len(runs) : len(runs) + n_runs]
        expected_frames = sum(mapped_lengths)
        n_frames = frames.shape[0]
        if n_frames == expected_frames:
            offset = 0
            for run_length in mapped_lengths:
                runs.append(frames[offset : offset + run_length].contiguous())
                offset += run_length
            if verbose and n_runs > 1:
                print(f"  {location}: splitting {n_frames} frames across {n_runs} runs")
        elif n_runs > 1 and all(length == n_frames for length in mapped_lengths):
            runs.extend(frames for _ in mapped_lengths)
            if verbose:
                print(f"  {location}: reusing the same {n_frames} frames for all {n_runs} runs")
        else:
            raise ValueError(
                f"Stimulus source {location} has {n_frames} frames, which is neither "
                f"{expected_frames} (the concatenated length of its {n_runs} mapped runs) "
                f"nor one run's length (runs are {mapped_lengths})"
            )

    if len(runs) != len(run_lengths):
        raise ValueError(
            f"Stimulus sources provide {len(runs)} runs but -input contains {len(run_lengths)} runs"
        )
    assert stimulus_shape is not None
    return runs, stimulus_shape


def _build_hrf_library(
    args: argparse.Namespace, tr: float, device: torch.device
) -> tuple[torch.Tensor, int | None]:
    """Build the HRF set to fit, and the index of the canonical HRF within it.

    The canonical is appended last rather than searched for in the library: the
    library's own entries are a parametric family that does not contain the
    canonical shape exactly, so "which library entry is the canonical" has no
    honest answer.
    """
    canonical = load_canonical_hrf_basic(
        microtime_dt=tr, hrf_duration=args.hrf_duration, device=device
    ).unsqueeze(0)
    if args.hrf_mode == "canonical":
        return canonical, 0

    if args.hrf_mode == "pighs":
        library = get_hrf_library(
            mode="pighs",
            microtime_dt=tr,
            hrf_duration=args.hrf_duration,
            n_hrfs=args.num_hrfs or 20,
            device=device,
        )
    else:
        library = load_canonical_hrf_library(
            microtime_dt=tr,
            hrf_duration=args.hrf_duration,
            device=device,
            library_path=args.hrf_library,
        )
        if args.num_hrfs is not None and args.num_hrfs < library.shape[0]:
            # The library is ordered by peak time, so an even stride keeps the
            # full timing range instead of truncating one end of it.
            keep = torch.linspace(0, library.shape[0] - 1, args.num_hrfs).round().long()
            library = library[keep.unique()]

    if not args.save_canonical:
        return library, None
    return torch.cat([library, canonical], dim=0), library.shape[0]


def _expand_to_all_voxels(values: torch.Tensor, keep: torch.Tensor, n_voxels: int) -> torch.Tensor:
    """Scatter a screened subset's results back into full voxel order."""
    full = torch.zeros((n_voxels, *values.shape[1:]), device=values.device, dtype=values.dtype)
    full[keep] = values
    return full


def _expand_fit(fit: PRFRefinedFit, keep: torch.Tensor, n_voxels: int) -> PRFRefinedFit:
    """Same, for every field of a refined fit."""
    return PRFRefinedFit(
        **{
            name: _expand_to_all_voxels(getattr(fit, name), keep, n_voxels)
            for name in (
                "candidate_index",
                "hrf_index",
                "parameters",
                "gain",
                "correlation",
                "r2",
                "residual_ss",
                "n_iters",
                "converged",
            )
        }
    )


def _add_noise_components(
    args: argparse.Namespace,
    data: torch.Tensor,
    screen_scores: torch.Tensor,
    invalid_voxels: torch.Tensor,
    stimulus_runs: list[torch.Tensor],
    stimulus_shape: tuple[int, int],
    grid,
    hrf_library: torch.Tensor,
    loaded,
    nuisance_per_run: list[torch.Tensor],
    device: torch.device,
) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
    """Pick a noise-component count by cross-validation and fold it into the nuisance.

    The screening pass has already said which voxels carry a stimulus response,
    so the pool is simply the ones that carry none -- by default the ones the
    linear pRF fits worse than their own mean. Components are extracted per run
    (``extract_noise_pcs_per_run``), which is what keeps this honest under
    cross-validation: a held-out run's regressors come from that run's own
    noise-pool voxels, never from the training runs and never from the voxel
    being scored.

    How many to keep is decided on the SUPER-GRID fit rather than the refined
    one. Refining the full model at every candidate count and every fold is tens
    of fits; the grid is about a second, and the question here is only which
    nuisance model predicts held-out data better, which does not need the
    refinement. The count is then used for the one real fit.
    """
    verbose = args.verb > 0
    pool = (screen_scores < args.noise_pool_r2) & torch.isfinite(screen_scores) & ~invalid_voxels
    if int(pool.sum()) < 100:
        raise ValueError(
            f"noise pool has only {int(pool.sum())} voxels at -noise_pool_r2 "
            f"{args.noise_pool_r2:g}; raise the threshold"
        )
    # Score the sweep on the strongest responders, EXCLUDING the noise pool. A
    # voxel that helped build the components cannot also judge them: with a tight
    # mask the two sets otherwise overlap, and denoising is then scored partly on
    # its own input. A bounded set also keeps the sweep cheap.
    rankable = torch.where(
        invalid_voxels | pool, torch.full_like(screen_scores, -torch.inf), screen_scores
    )
    n_criteria = min(int(torch.isfinite(rankable).sum()), 3000)
    if n_criteria < 10:
        raise ValueError(
            "fewer than 10 voxels sit outside the noise pool; -noise_pool_r2 is too high"
        )
    criteria = torch.topk(rankable, n_criteria).indices
    if verbose:
        print(
            f"Denoising: {int(pool.sum()):,} noise-pool voxels (screen R2 < "
            f"{args.noise_pool_r2:g}), scored on {n_criteria:,} responders"
        )

    components = extract_noise_pcs_per_run(
        data,
        list(loaded.run_starts),
        pool,
        max_components=args.max_pcs,
        nuisance_per_run=nuisance_per_run,
        device=device,
    )
    assert isinstance(components, list)  # return_loadings=False
    available = min(args.max_pcs, min(block.shape[1] for block in components))
    criteria_data = data[criteria]

    r2_by_count: list[torch.Tensor] = []
    counts = range(available + 1)
    if verbose:
        from tqdm.auto import tqdm

        counts = tqdm(counts, desc="pRF noise-PC sweep", leave=True)
    for n_components in counts:
        trial_nuisance = _append_components(nuisance_per_run, components, n_components)
        fit = fit_prf_loro(
            criteria_data,
            stimulus_runs,
            stimulus_shape,
            grid,
            hrf_library,
            loaded.run_starts,
            nuisance_per_run=trial_nuisance,
            candidate_chunk_size=args.candidate_chunk,
            voxel_chunk_size=n_criteria,
            refine=False,
            device=device,
        )
        r2_by_count.append(fit.r2.detach().cpu())

    median_r2 = [float(values.median()) for values in r2_by_count]
    chosen = select_noise_pc_count(median_r2, tolerance=args.denoise_tolerance)
    if verbose:
        curve = "  ".join(f"{n}:{value:.4f}" for n, value in enumerate(median_r2))
        print(f"  held-out median R2 by component count: {curve}")
        if chosen == 0:
            print("  keeping 0 components: denoising did not improve held-out R2")
        else:
            print(
                f"  keeping {chosen} components "
                f"(+{median_r2[chosen] - median_r2[0]:.4f} held-out median R2)"
            )
    if args.save_denoise:
        _plot_noise_sweep(
            r2_by_count, chosen, f"{parse_prefix(args.prefix).stem}_denoise.png", n_criteria
        )
    kept = [block[:, :chosen] for block in components]
    return _append_components(nuisance_per_run, components, chosen), pool, kept


def _plot_noise_sweep(
    r2_by_count: list[torch.Tensor], chosen: int, output_path: str, n_criteria: int
) -> None:
    """Plot held-out R2 against noise-component count, and the gain over none.

    Two panels for the same reason ffs_denoise uses two: the absolute curve says
    whether the fit is any good, and the paired difference from the undenoised
    model -- same voxels, same folds -- says whether denoising is what did it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = list(range(len(r2_by_count)))
    stacked = torch.stack(r2_by_count)
    median = stacked.median(dim=1).values.numpy()
    low = stacked.quantile(0.25, dim=1).numpy()
    high = stacked.quantile(0.75, dim=1).numpy()
    delta = stacked - stacked[0]

    figure, (axis, axis_delta) = plt.subplots(1, 2, figsize=(11, 4.2))
    line = axis.plot(counts, median, marker="o")[0]
    axis.fill_between(counts, low, high, alpha=0.15, color=line.get_color())
    axis.axvline(chosen, color="crimson", lw=1.2, ls="--", label=f"kept {chosen}")
    axis.set_xlabel("Noise components projected out")
    axis.set_ylabel(f"Median held-out R2 ({n_criteria:,} responders)")
    axis.set_title("Cross-validated fit vs component count")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)

    axis_delta.plot(counts, delta.median(dim=1).values.numpy(), marker="o", color="tab:green")
    axis_delta.fill_between(
        counts,
        delta.quantile(0.25, dim=1).numpy(),
        delta.quantile(0.75, dim=1).numpy(),
        alpha=0.15,
        color="tab:green",
    )
    axis_delta.axhline(0, color="0.5", lw=1.0)
    axis_delta.axvline(chosen, color="crimson", lw=1.2, ls="--")
    axis_delta.set_xlabel("Noise components projected out")
    axis_delta.set_ylabel("Change in held-out R2")
    axis_delta.set_title("Improvement over no denoising")
    axis_delta.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    plt.close(figure)


def _append_components(
    nuisance_per_run: list[torch.Tensor], components: list[torch.Tensor], n_components: int
) -> list[torch.Tensor]:
    """Concatenate the first ``n_components`` noise PCs onto each run's nuisance block."""
    if n_components < 1:
        return nuisance_per_run
    return [
        torch.cat([block, run_components[:, :n_components].to(block.device, block.dtype)], dim=1)
        for block, run_components in zip(nuisance_per_run, components, strict=True)
    ]


def _invalid_voxels(data: torch.Tensor, run_starts: list[int]) -> torch.Tensor:
    """Match analyzePRF: reject non-finite voxels and voxels zero in any run."""
    invalid = ~torch.isfinite(data).all(dim=1)
    run_ends = [*run_starts[1:], data.shape[1]]
    for start, end in zip(run_starts, run_ends, strict=True):
        invalid |= data[:, start:end].eq(0).all(dim=1)
    return invalid


def _save_voxel_matrix(
    values: torch.Tensor,
    output_path: str,
    loaded,
    labels: list[str],
    invalid_voxels: torch.Tensor | None,
) -> None:
    """Save an ``(n_voxels, n_columns)`` matrix as a labeled 4D bucket."""
    array = values.cpu().numpy()
    if invalid_voxels is not None:
        array[invalid_voxels.cpu().numpy()] = np.nan
    if loaded.mask_flat is not None:
        full = np.zeros((loaded.mask_flat.size, array.shape[1]), dtype=array.dtype)
        full[loaded.mask_flat] = array
        array = full
    save_nifti(
        array.reshape((*loaded.volume_shape, len(labels))),
        output_path=output_path,
        affine=loaded.affine,
        header=loaded.nifti_header,
        brick_labels=labels,
    )


def _save_results(
    results,
    output_path: str,
    loaded,
    stimulus_shape: tuple[int, int],
    xval_r2: torch.Tensor | None = None,
    invalid_voxels: torch.Tensor | None = None,
    mean_volume: torch.Tensor | None = None,
    hrf_r2_map: torch.Tensor | None = None,
    screen_extent: float | None = None,
) -> None:
    """Save all primary pRF parameters in one labeled 4D NIfTI bucket."""
    extent = float(max(stimulus_shape))
    center = (1.0 + extent) / 2.0
    row = results.parameters[:, 0]
    column = results.parameters[:, 1]
    exponent = results.parameters[:, 3].clamp_min(torch.finfo(results.parameters.dtype).eps)
    # Larger row index is the upper visual field for the aperture layouts these
    # tools are given, so vertical position is row - center, not center - row.
    # y and angle must share this sign or they describe different half-fields.
    vertical = row - center
    eccentricity = torch.hypot(vertical, column - center)
    angle = torch.rad2deg(torch.atan2(vertical, column - center)).remainder(360.0)
    angle = torch.where(eccentricity == 0, torch.full_like(angle, torch.nan), angle)
    rfsize = results.parameters[:, 2].abs() / torch.sqrt(exponent)
    # Positions are reported as x/y offsets from the aperture center (x right,
    # y up), matching the angle convention above, rather than as the raw
    # one-based row/column the optimizer works in. With -screen_extent every
    # spatial quantity is scaled to degrees of visual angle by the same factor;
    # pixel units are an artifact of the aperture resolution and change under
    # -stim_downsample, so they are not comparable across studies.
    scale = 1.0 if screen_extent is None else screen_extent / extent
    labels = [
        "x",
        "y",
        "sigma",
        "exponent",
        "gain",
        "angle",
        "eccentricity",
        "rfsize",
        "correlation",
        "r2",
        "hrf_index",
        "grid_index",
        "residual_ss",
        "gn_iterations",
        "gn_converged",
    ]
    columns = [
        (column - center) * scale,
        vertical * scale,
        results.parameters[:, 2].abs() * scale,
        results.parameters[:, 3],
        results.gain,
        angle,
        eccentricity * scale,
        rfsize * scale,
        results.correlation,
        results.r2,
        results.hrf_index + 1,
        results.candidate_index + 1,
        results.residual_ss,
        results.n_iters,
        results.converged,
    ]
    if xval_r2 is not None:
        labels.append("xval_r2")
        columns.append(xval_r2)
    if mean_volume is not None:
        # analyzePRF's results.meanvol - the scale the gain is expressed against,
        # and the reference for turning gain into a percent-signal-change.
        labels.append("meanvol")
        columns.append(mean_volume)
    if hrf_r2_map is not None:
        _, continuous_hrf, hrf_evidence = summarize_hrf_selection(hrf_r2_map)
        # hrf_evidence near zero means the library is indistinguishable for this
        # voxel: threshold on it before believing hrf_index.
        labels += ["hrf_index_continuous", "hrf_evidence"]
        columns += [continuous_hrf + 1, hrf_evidence]
    values = torch.column_stack(columns).cpu().numpy()
    if invalid_voxels is not None:
        values[invalid_voxels.cpu().numpy()] = np.nan
    if loaded.mask_flat is not None:
        full = np.zeros((loaded.mask_flat.size, len(labels)), dtype=values.dtype)
        full[loaded.mask_flat] = values
        data = full.reshape((*loaded.volume_shape, len(labels)))
    else:
        data = values.reshape((*loaded.volume_shape, len(labels)))
    save_nifti(
        data,
        output_path=output_path,
        affine=loaded.affine,
        header=loaded.nifti_header,
        brick_labels=labels,
    )


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    prefix_info = parse_prefix(args.prefix)
    input_files = parse_input_files(args.input)
    if bool(args.stimulus) == bool(args.stim_sources):
        parser.error(
            "Specify exactly one of -stimulus or -stim-pngs/-stim-pngs-multi/"
            "-stim-nii/-stim-nii-multi"
        )
    if args.stimulus is not None and len(args.stimulus) != len(input_files):
        parser.error("-stimulus must provide exactly one aperture source per -input run")
    if args.denoise and len(input_files) < 2:
        parser.error("-denoise needs at least two runs to cross-validate the component count")
    if args.max_pcs < 1:
        parser.error("-max_pcs must be positive")
    if not 0 <= args.denoise_tolerance < 1:
        parser.error("-denoise_tolerance must be in [0, 1)")
    if args.screen is not None and args.screen_top is not None:
        parser.error("Specify at most one of -screen and -screen_top")
    if args.screen_top is not None and not 0 < args.screen_top <= 1:
        parser.error("-screen_top must be a fraction in (0, 1]")
    if args.screen_tiles < 1:
        parser.error("-screen_tiles must be positive")
    if args.grid_angles < 1 or args.grid_sigma_steps < 1 or args.grid_ecc_steps < 1:
        parser.error("-grid_angles, -grid_sigma_steps, and -grid_ecc_steps must be positive")
    if args.screen_extent is not None and args.screen_extent <= 0:
        parser.error("-screen_extent must be positive")
    if args.num_hrfs is not None and args.num_hrfs < 1:
        parser.error("-num_hrfs must be positive")
    if args.hrf_library and args.hrf_mode != "library":
        parser.error("-hrf_library applies only to -hrf library")
    if args.stim_downsample < 0:
        parser.error("-stim-downsample must be non-negative (0 keeps the native resolution)")
    if args.candidate_chunk < 1 or (args.batch_size is not None and args.batch_size < 1):
        parser.error("-candidate-chunk and -batch-size must be positive")
    if args.maxiter < 1 or args.gn_damping <= 0 or args.gn_step_tol <= 0:
        parser.error("-maxiter, -gn-damping, and -gn-step-tol must be positive")
    if args.xval == "loro" and len(input_files) < 2:
        parser.error("-xval loro requires at least two input runs")
    # Check the aperture paths before the fMRI load, which can take minutes.
    for source in args.stimulus or []:
        if not Path(source).exists():
            parser.error(f"-stimulus source does not exist: {source}")
    for source in args.stim_sources:
        location = source if isinstance(source, str) else source[0]
        if not Path(location).exists():
            parser.error(f"Stimulus source does not exist: {location}")

    device, _, _ = parse_device_arg(args.device)
    configure_torch_backends(device)
    loaded = load_and_preprocess_runs(
        input_files,
        tr=args.tr,
        mask_file=args.mask,
        do_scale=args.do_scale,
        device=device,
        force_cpu=args.keep_on_cpu,
        verbose=args.verb > 0,
        load_threads=args.load_threads,
    )
    run_lengths = compute_run_lengths(loaded.run_starts, loaded.n_timepoints)
    invalid_voxels = _invalid_voxels(loaded.data, loaded.run_starts)
    analysis_data = torch.nan_to_num(loaded.data, nan=0.0, posinf=0.0, neginf=0.0)
    if args.verb and invalid_voxels.any():
        print(f"Excluding {invalid_voxels.sum().item():,} non-finite or all-zero voxels")
    if args.stim_sources:
        stimulus_runs, stimulus_shape = _load_stimulus_sources(
            args.stim_sources, run_lengths, args.stim_downsample, verbose=args.verb > 0
        )
    else:
        stimulus_runs = []
        stimulus_shape: tuple[int, int] | None = None
        for input_file, stimulus_file, run_length in zip(
            input_files, args.stimulus, run_lengths, strict=True
        ):
            stimulus, shape = _load_stimulus_run(stimulus_file, args.stim_downsample)
            if stimulus.shape[0] != run_length:
                raise ValueError(
                    f"Stimulus {stimulus_file} has {stimulus.shape[0]} frames but {input_file} has {run_length} volumes"
                )
            if stimulus_shape is not None and shape != stimulus_shape:
                raise ValueError(
                    f"Stimulus {stimulus_file} has shape {shape}; expected {stimulus_shape}"
                )
            stimulus_runs.append(stimulus)
            stimulus_shape = shape
        assert stimulus_shape is not None
    if args.verb:
        print(f"Aperture resolution: {stimulus_shape[0]} x {stimulus_shape[1]} pixels")
        if args.screen_extent:
            degrees_per_pixel = args.screen_extent / max(stimulus_shape)
            print(
                f"Visual angle: {args.screen_extent:g} deg full width "
                f"({degrees_per_pixel:.4f} deg/pixel, "
                f"{args.screen_extent / 2:g} deg maximum eccentricity)"
            )
    polort = args.polort
    if polort is None:
        polort = auto_polort(get_average_run_duration(run_lengths, loaded.tr))
    nuisance_blocks = collect_nuisance_blocks(
        args, loaded.run_starts, loaded.n_timepoints, verbose=args.verb > 1
    )
    nuisance_per_run = build_nuisance_per_run(
        loaded.run_starts,
        loaded.n_timepoints,
        polort,
        device,
        blocks=nuisance_blocks,
        verbose=args.verb > 1,
    )
    hrf_library, canonical_index = _build_hrf_library(args, loaded.tr, device)

    screen_scores = None
    keep_index = None
    noise_pool = None
    noise_components: list[torch.Tensor] = []
    if args.screen is not None or args.screen_top is not None or args.denoise:
        screen_scores = screen_voxels_ridge(
            analysis_data,
            stimulus_runs,
            stimulus_shape,
            load_canonical_hrf_basic(
                microtime_dt=loaded.tr, hrf_duration=args.hrf_duration, device=device
            ),
            loaded.run_starts,
            nuisance_per_run=nuisance_per_run,
            n_tiles=args.screen_tiles,
            voxel_chunk_size=estimate_chunk_size(
                n_voxels=loaded.n_voxels,
                n_timepoints=loaded.n_timepoints,
                n_regressors=args.screen_tiles,
                device=device,
                operation="xval",
            ),
            device=device,
            verbose=args.verb > 0,
        )
    if args.denoise:
        grid_for_sweep = make_analyzeprf_grid(
            stimulus_shape,
            exponents=(1.0,) if args.model_mode == 3 else (0.5, 0.25, 0.125),
            n_angles=args.grid_angles,
            angle_mode=args.grid_angle_mode,
            sigma_mode=args.grid_sigma_mode,
            sigma_steps_per_octave=args.grid_sigma_steps,
            eccentricity_steps=args.grid_ecc_steps,
            device=device,
        )
        nuisance_per_run, noise_pool, noise_components = _add_noise_components(
            args,
            analysis_data,
            screen_scores,
            invalid_voxels,
            stimulus_runs,
            stimulus_shape,
            grid_for_sweep,
            hrf_library,
            loaded,
            nuisance_per_run,
            device,
        )

    if args.screen is not None or args.screen_top is not None:
        assert screen_scores is not None
        # Rank only over voxels that could be fit at all, so the fraction is a
        # fraction of real data rather than of background.
        rankable = torch.where(
            invalid_voxels, torch.full_like(screen_scores, -torch.inf), screen_scores
        )
        if args.screen_top is not None:
            n_valid = int((~invalid_voxels).sum())
            n_keep = max(1, min(n_valid, int(round(args.screen_top * n_valid))))
            threshold = torch.topk(rankable, n_keep).values.min().item()
        else:
            threshold = args.screen
        keep = (rankable >= threshold) & torch.isfinite(rankable)
        if not bool(keep.any()):
            parser.error(f"screening kept no voxels at threshold {threshold:g}")
        keep_index = torch.nonzero(keep, as_tuple=False).squeeze(1)
        analysis_data = analysis_data[keep_index]
        if args.verb:
            print(
                f"Screening kept {keep_index.numel():,} of {loaded.n_voxels:,} voxels "
                f"({100 * keep_index.numel() / loaded.n_voxels:.1f}%) at R2 >= {threshold:g}"
            )

    n_fit_voxels = analysis_data.shape[0]
    grid = make_analyzeprf_grid(
        stimulus_shape,
        exponents=(1.0,) if args.model_mode == 3 else (0.5, 0.25, 0.125),
        n_angles=args.grid_angles,
        angle_mode=args.grid_angle_mode,
        sigma_mode=args.grid_sigma_mode,
        sigma_steps_per_octave=args.grid_sigma_steps,
        eccentricity_steps=args.grid_ecc_steps,
        device=device,
    )
    n_pixels = stimulus_shape[0] * stimulus_shape[1]
    voxel_chunk_size = args.batch_size or estimate_chunk_size(
        n_voxels=n_fit_voxels,
        n_timepoints=loaded.n_timepoints,
        n_regressors=args.candidate_chunk,
        device=device,
        operation="xval",
    )
    # Refinement is dominated by the per-voxel Gaussian over the aperture, not by
    # the design, so it needs its own (much smaller) chunk than the grid search.
    refine_chunk_size = args.batch_size or estimate_chunk_size(
        n_voxels=n_fit_voxels,
        n_timepoints=loaded.n_timepoints,
        n_regressors=n_pixels,
        device=device,
        operation="prf",
        min_chunk_size=1,
    )
    if args.verb:
        print(
            f"Fitting {n_fit_voxels:,} voxels against {grid.n_candidates:,} CSS candidates "
            f"and {hrf_library.shape[0]} {args.hrf_mode} "
            f"HRF{'s' if hrf_library.shape[0] > 1 else ''} on {device}."
            + (" (last is the canonical)" if canonical_index and args.save_canonical else "")
        )
        print(
            f"Chunk sizes: {voxel_chunk_size:,} voxels (grid), "
            f"{refine_chunk_size:,} voxels (refinement, {n_pixels:,}-pixel aperture)"
        )
    grid_results = fit_prf_supergrid(
        analysis_data,
        stimulus_runs,
        stimulus_shape,
        grid,
        hrf_library,
        loaded.run_starts,
        nuisance_per_run=nuisance_per_run,
        candidate_chunk_size=args.candidate_chunk,
        voxel_chunk_size=voxel_chunk_size,
        device=device,
        verbose=args.verb > 0,
    )
    refinement_config = PRFRefinementConfig(
        max_iter=args.maxiter,
        damping=args.gn_damping,
        step_tolerance=args.gn_step_tol,
        min_exponent=args.expt_lower_bound,
        fix_exponent=args.model_mode == 3,
        stagewise_exponent=args.model_mode == 1,
    )
    hrf_r2_map = None
    canonical_results = None
    if args.quick:
        results = grid_seeds_as_fit(grid_results)
    elif args.hrf_select == "refine" and hrf_library.shape[0] > 1:
        results, hrf_r2_map, canonical_results = refine_prf_all_hrfs(
            analysis_data,
            stimulus_runs,
            stimulus_shape,
            grid_results,
            hrf_library,
            loaded.run_starts,
            nuisance_per_run=nuisance_per_run,
            voxel_chunk_size=refine_chunk_size,
            device=device,
            config=refinement_config,
            keep_hrf_index=canonical_index,
            verbose=args.verb > 0,
        )
    elif args.hrf_select.startswith("grid+") and hrf_library.shape[0] > 1:
        results = refine_prf_hrf_window(
            analysis_data,
            stimulus_runs,
            stimulus_shape,
            grid_results,
            hrf_library,
            loaded.run_starts,
            window=int(args.hrf_select.removeprefix("grid+")),
            nuisance_per_run=nuisance_per_run,
            voxel_chunk_size=refine_chunk_size,
            device=device,
            config=refinement_config,
            verbose=args.verb > 0,
        )
    else:
        results = refine_prf_supergrid(
            analysis_data,
            stimulus_runs,
            stimulus_shape,
            grid_results,
            hrf_library,
            loaded.run_starts,
            nuisance_per_run=nuisance_per_run,
            voxel_chunk_size=refine_chunk_size,
            device=device,
            config=refinement_config,
            verbose=args.verb > 0,
        )
    do_loro = not args.quick and (
        args.xval == "loro" or (args.xval == "auto" and len(input_files) > 1)
    )
    xval_r2 = None
    if do_loro:
        xval_results = fit_prf_loro(
            analysis_data,
            stimulus_runs,
            stimulus_shape,
            grid,
            hrf_library,
            loaded.run_starts,
            nuisance_per_run=nuisance_per_run,
            candidate_chunk_size=args.candidate_chunk,
            voxel_chunk_size=voxel_chunk_size,
            refine_chunk_size=refine_chunk_size,
            device=device,
            refinement_config=refinement_config,
            hrf_mode=args.xval_hrf,
            fixed_hrf_index=results.hrf_index if args.xval_hrf == "fixed" else None,
            verbose=args.verb > 0,
        )
        xval_r2 = xval_results.r2
    if keep_index is not None:
        results = _expand_fit(results, keep_index, loaded.n_voxels)
        if hrf_r2_map is not None:
            hrf_r2_map = _expand_to_all_voxels(hrf_r2_map, keep_index, loaded.n_voxels)
        if xval_r2 is not None:
            xval_r2 = _expand_to_all_voxels(xval_r2, keep_index, loaded.n_voxels)
        if canonical_results is not None:
            canonical_results = _expand_fit(canonical_results, keep_index, loaded.n_voxels)
        # Screened-out voxels were never fit, so they are reported as NaN rather
        # than as zeros that look like a fit that failed.
        screened_out = torch.ones_like(invalid_voxels)
        screened_out[keep_index] = False
        invalid_voxels = invalid_voxels | screened_out
    _save_results(
        results,
        f"{prefix_info.stem}{prefix_info.nifti_ext}",
        loaded,
        stimulus_shape,
        xval_r2=xval_r2,
        invalid_voxels=invalid_voxels,
        mean_volume=loaded.data.mean(dim=1),
        hrf_r2_map=hrf_r2_map,
        screen_extent=args.screen_extent,
    )
    if args.verb:
        print(f"Wrote pRF results: {prefix_info.stem}{prefix_info.nifti_ext}")
    if canonical_results is not None:
        canonical_path = f"{prefix_info.stem}_canonical{prefix_info.nifti_ext}"
        _save_results(
            canonical_results,
            canonical_path,
            loaded,
            stimulus_shape,
            invalid_voxels=invalid_voxels,
            mean_volume=loaded.data.mean(dim=1),
            screen_extent=args.screen_extent,
        )
        if args.verb:
            print(f"Wrote fixed-canonical-HRF results: {canonical_path}")
    elif args.save_canonical and args.verb and args.hrf_mode != "canonical":
        print("-save_canonical ignored: needs -hrf_select refine")
    if args.save_denoise and noise_pool is not None:
        pool_path = f"{prefix_info.stem}_noisepool{prefix_info.nifti_ext}"
        _save_voxel_matrix(
            noise_pool.to(torch.float32).unsqueeze(1), pool_path, loaded, ["noise_pool"], None
        )
        if args.verb:
            print(f"Wrote noise pool: {pool_path}")
        if noise_components and noise_components[0].shape[1] > 0:
            components_path = f"{prefix_info.stem}_noisepcs.1D"
            np.savetxt(
                components_path,
                torch.cat(noise_components, dim=0).cpu().numpy(),
                fmt="%.8g",
            )
            if args.verb:
                print(f"Wrote noise components: {components_path}")
    elif args.save_denoise and args.verb:
        print("-save_denoise ignored: needs -denoise")
    if args.save_screen and screen_scores is not None:
        _save_voxel_matrix(
            screen_scores.unsqueeze(1),
            f"{prefix_info.stem}_screen{prefix_info.nifti_ext}",
            loaded,
            ["screen_r2"],
            None,
        )
        if args.verb:
            print(f"Wrote screening map: {prefix_info.stem}_screen{prefix_info.nifti_ext}")
    elif args.save_screen and args.verb:
        print("-save_screen ignored: needs -screen or -screen_top")
    if args.save_hrf_r2 and hrf_r2_map is not None:
        _save_voxel_matrix(
            hrf_r2_map,
            f"{prefix_info.stem}_hrf_r2{prefix_info.nifti_ext}",
            loaded,
            [f"hrf{index + 1:02d}" for index in range(hrf_r2_map.shape[1])],
            invalid_voxels,
        )
        if args.verb:
            print(f"Wrote per-HRF R2 map: {prefix_info.stem}_hrf_r2{prefix_info.nifti_ext}")
    elif args.save_hrf_r2 and args.verb:
        print("-save_hrf_r2 ignored: needs -hrf_select refine with more than one HRF")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
