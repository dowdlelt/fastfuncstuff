#!/usr/bin/env python

"""

Command-line tool for building fMRI design matrices

Simplified syntax compared to AFNI's 3dDeconvolve:
- No numbering of stimuli or GLTs
- Auto-padding for per-run regressors
- Clean, intuitive interface

Examples
--------
Basic usage with two stimuli:
    build_design_matrix.py \\
        -input data_r01.nii.gz data_r02.nii.gz \\
        -polort 3 \\
        -stim times.movie.txt 'SPMG1(5)' movie \\
        -stim times.prompt.txt 'SPMG1(5)' prompt \\
        -gltsym 'SYM: +1*movie -1*prompt' movieVprompt \\
        -xmat X.xmat.1D

With nuisance regressors:
    build_design_matrix.py \\
        -input data.nii.gz \\
        -polort 2 \\
        -ortvec motion.1D motion \\
        -padortvec motion_r01.1D motion_r01 1 \\
        -stim times.task.txt 'SPMG1(10)' task \\
        -xmat X.xmat.1D

Individual modulation (IM):
    build_design_matrix.py \\
        -input data.nii.gz \\
        -stim_IM times.events.txt 'SPMG1(0)' events \\
        -xmat X.xmat.1D
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from fastfuncsim.afni_io import get_tr_from_file, load_nifti
from fastfuncsim.design_builder import (
    build_design_matrix,
    parse_glt_string,
    write_afni_xmat,
)

try:
    import nibabel as nib
except ImportError:
    print("Error: nibabel is required. Install with: pip install nibabel", file=sys.stderr)
    sys.exit(1)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Build fMRI design matrix with simplified syntax',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input data (for metadata only)
    parser.add_argument(
        '-input',
        nargs='+',
        metavar='FILE',
        required=True,
        help='Input fMRI data files (one per run). Used only for TR and run length metadata.',
    )

    # Polynomial drift
    parser.add_argument(
        '-polort',
        type=int,
        default=3,
        metavar='N',
        help='Polynomial drift order (default: 3). Use -1 for no polynomials.',
    )

    # Stimulus regressors
    parser.add_argument(
        '-stim',
        action='append',
        nargs=3,
        metavar=('TIMING_FILE', 'HRF_MODEL', 'LABEL'),
        dest='stims',
        help="Add stimulus regressor. HRF_MODEL examples: 'SPMG1(5)', 'BLOCK(10)'. "
             "Can be used multiple times.",
    )

    # Individual modulation stimuli
    parser.add_argument(
        '-stim_IM',
        action='append',
        nargs=3,
        metavar=('TIMING_FILE', 'HRF_MODEL', 'LABEL'),
        dest='stims_im',
        help="Add stimulus with Individual Modulation (one column per event). "
             "Can be used multiple times.",
    )

    # Nuisance regressors (full length)
    parser.add_argument(
        '-ortvec',
        action='append',
        nargs=2,
        metavar=('FILE', 'LABEL'),
        dest='ortvecs',
        help='Add nuisance regressor (full concatenated length). Can be used multiple times.',
    )

    # Nuisance regressors (per-run with auto-padding)
    parser.add_argument(
        '-padortvec',
        action='append',
        nargs=3,
        metavar=('FILE', 'LABEL', 'RUN'),
        dest='padortvecs',
        help='Add per-run nuisance regressor with auto zero-padding. '
             'RUN is 1-indexed. Can be used multiple times.',
    )

    # GLT contrasts
    parser.add_argument(
        '-gltsym',
        action='append',
        nargs=2,
        metavar=('CONTRAST', 'LABEL'),
        dest='glts',
        help="Add GLT contrast. CONTRAST format: 'SYM: +1*labelA -1*labelB'. "
             "Can be used multiple times.",
    )

    # Output
    parser.add_argument(
        '-xmat',
        required=True,
        metavar='FILE',
        help='Output design matrix file (.xmat.1D format)',
    )

    # Optional flags
    parser.add_argument(
        '-TR',
        type=float,
        metavar='SECONDS',
        help='Override TR from input files',
    )

    parser.add_argument(
        '-verbose',
        action='store_true',
        help='Print detailed progress information',
    )

    return parser.parse_args()


def get_input_metadata(input_files: List[str], tr_override: Optional[float] = None) -> Tuple[float, List[int]]:
    """
    Extract TR and run lengths from input files

    Parameters
    ----------
    input_files : list of str
        Paths to input fMRI files
    tr_override : float, optional
        Override TR value

    Returns
    -------
    tr : float
        Repetition time in seconds
    n_timepoints_per_run : list of int
        Number of timepoints in each run
    """
    n_timepoints_per_run = []
    tr_values = []

    for input_file in input_files:
        path = Path(input_file)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Load image
        img = load_nifti(path)

        # Get number of timepoints (4th dimension)
        n_timepoints = img.shape[3] if len(img.shape) > 3 else 1
        n_timepoints_per_run.append(n_timepoints)

        # Get TR
        if tr_override is None:
            tr_values.append(get_tr_from_file(str(path)))

    # Use override or check consistency
    if tr_override is not None:
        tr = tr_override
    else:
        # Check all TRs match
        if len(set(tr_values)) > 1:
            raise ValueError(f"Inconsistent TRs across runs: {tr_values}. Use -TR to override.")
        tr = tr_values[0]

    return tr, n_timepoints_per_run


def main():
    """Main CLI entry point"""
    args = parse_args()

    # Get input metadata
    if args.verbose:
        print("Reading input file metadata...")

    tr, n_timepoints_per_run = get_input_metadata(args.input, args.TR)

    if args.verbose:
        print(f"  TR: {tr}s")
        print(f"  Runs: {len(n_timepoints_per_run)}")
        print(f"  Timepoints per run: {n_timepoints_per_run}")
        print(f"  Total timepoints: {sum(n_timepoints_per_run)}")

    # Prepare stimulus arguments
    timing_files = []
    stim_labels = []
    hrf_models = []
    im_modes = []

    # Add standard stimuli
    if args.stims:
        for timing_file, hrf_model, label in args.stims:
            timing_files.append(timing_file)
            stim_labels.append(label)
            hrf_models.append(hrf_model)
            im_modes.append(False)

    # Add IM stimuli
    if args.stims_im:
        for timing_file, hrf_model, label in args.stims_im:
            timing_files.append(timing_file)
            stim_labels.append(label)
            hrf_models.append(hrf_model)
            im_modes.append(True)

    if args.verbose and len(timing_files) > 0:
        print(f"\nStimuli: {len(timing_files)}")
        for i, (label, hrf, im) in enumerate(zip(stim_labels, hrf_models, im_modes)):
            mode_str = " (IM)" if im else ""
            print(f"  {i+1}. {label}: {hrf}{mode_str}")

    # Prepare nuisance regressors
    padortvec_files = []
    if args.padortvecs:
        for filepath, label, run_str in args.padortvecs:
            run_num = int(run_str)
            padortvec_files.append((filepath, label, run_num))

    ortvec_files = []
    if args.ortvecs:
        for filepath, label in args.ortvecs:
            ortvec_files.append((filepath, label))

    if args.verbose and (padortvec_files or ortvec_files):
        print("\nNuisance regressors:")
        if padortvec_files:
            print(f"  Padded (per-run): {len(padortvec_files)}")
            for filepath, label, run_num in padortvec_files:
                print(f"    - {label} (run {run_num})")
        if ortvec_files:
            print(f"  Full-length: {len(ortvec_files)}")
            for filepath, label in ortvec_files:
                print(f"    - {label}")

    # Build design matrix
    if args.verbose:
        print("\nBuilding design matrix...")

    try:
        design, labels, run_starts, metadata = build_design_matrix(
            timing_files=timing_files if timing_files else None,
            stim_labels=stim_labels if stim_labels else None,
            n_timepoints_per_run=n_timepoints_per_run,
            tr=tr,
            polort=args.polort,
            hrf_models=hrf_models if hrf_models else None,
            im_mode=im_modes if im_modes else None,
            padortvec_files=padortvec_files if padortvec_files else None,
            ortvec_files=ortvec_files if ortvec_files else None,
        )
    except Exception as e:
        print(f"Error building design matrix: {e}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"  Design matrix shape: {design.shape}")
        print(f"  Columns: {len(labels)}")
        print(f"  Stimulus columns: {len(metadata['stim_indices'])}")
        print(f"  Nuisance columns: {len(metadata['nuisance_indices'])}")

    # Parse GLT contrasts
    glt_contrasts = []
    if args.glts:
        if args.verbose:
            print(f"\nGLT contrasts: {len(args.glts)}")

        for contrast_str, label in args.glts:
            # Validate
            weights, valid = parse_glt_string(contrast_str)

            if args.verbose:
                print(f"  {label}: {contrast_str}")
                if not valid:
                    print(f"    WARNING: Weights sum to {sum(weights.values()):.3f} (expected 0 or 1)")

            glt_contrasts.append((contrast_str, label))

    # Build command line string for metadata
    command_line = ' '.join(sys.argv)

    # Write output
    if args.verbose:
        print(f"\nWriting design matrix to: {args.xmat}")

    try:
        write_afni_xmat(
            filepath=args.xmat,
            design_matrix=design,
            regressor_labels=labels,
            run_starts=run_starts,
            metadata=metadata,
            glt_contrasts=glt_contrasts if glt_contrasts else None,
            command_line=command_line,
        )
    except Exception as e:
        print(f"Error writing design matrix: {e}", file=sys.stderr)
        return 1

    if args.verbose:
        print("✓ Done!")
    else:
        print(f"Wrote design matrix to {args.xmat}")

    return 0


if __name__ == '__main__':
    sys.exit(main())

