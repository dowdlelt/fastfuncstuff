"""Simulate a task response, fit it with OLS, and save a NIfTI stats bucket."""

import torch

import fastfuncstuff as ffs
from fastfuncstuff.cli_help import FfsArgumentParser
from fastfuncstuff.cli_utils import add_device_arg, setup_device


def main():
    parser = FfsArgumentParser(description=__doc__)
    parser.add_argument("-output", default="synthetic_glm.nii.gz", help="Output NIfTI stats bucket")
    add_device_arg(parser, default="cpu")
    args = parser.parse_args()
    device = setup_device(args.device)

    torch.manual_seed(7)

    tr = 1.0
    n_timepoints = 96
    shape = (4, 4, 2)
    onsets = torch.zeros((n_timepoints, 2), device=device)
    onsets[[8, 32, 60, 80], 0] = 1
    onsets[[19, 47, 70], 1] = 1
    hrf = ffs.get_canonical_hrf(stim_duration=1.0, tr=tr, device=device)
    data = ffs.simulate_fmri_run(
        onsets,
        betas=[2.0, 1.0],
        hrf=hrf,
        tr=tr,
        n_timepoints=n_timepoints,
        matrix_size=shape,
        noise_level=0.4,
        add_scanner_drift=False,
        device=device,
    )
    design = ffs.build_glm_design(onsets, hrf, n_timepoints, device=device)
    results = ffs.fit_glm(data, design, tr=tr, max_poly_degree=0, device=device, verbose=False)
    output = ffs.write_glm_bucket_as_nifti(
        results,
        args.output,
        volume_shape=shape,
        condition_names=["condition_a", "condition_b"],
        apply_afni_metadata=False,
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
