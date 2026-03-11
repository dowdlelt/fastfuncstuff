"""MotSim: Motion-simulation regressors (Patriat, Reynolds & Birn 2017).

Implements the MotSim approach from PMC5533292. Given a reference EPI and
motion parameters, creates simulated 4D datasets that capture the voxel-wise
signal changes caused by rigid-body motion, then extracts temporal PCs as
nuisance regressors.

Three variants:
  - forward:  Apply inverse motion to reference → simulated 4D.
  - backward: Re-register the forward sim back to reference → residual
              interpolation artifacts.
  - both:     Concatenate forward + backward spatially, then PCA (best).

Usage:
    ffs_motsim -base mean_epi.nii.gz -aff12 moco.aff12.1D -prefix motsim \\
               -n_pcs 12 -variant both
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch
from torch import Tensor

from .affine import apply_affine_interp, dicom_matrix_to_voxel, params_to_matrix
from .io import load_image, save_image
from .nwarpforge import load_affine_1D


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ffs_motsim",
        description=(
            "Generate motion-simulation nuisance regressors (Patriat et al. 2017). "
            "Applies motion parameters to a reference EPI to simulate motion-induced "
            "signal changes, then extracts PCs as regressors of no interest."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── I/O ──
    g_io = p.add_argument_group("Input / Output")
    g_io.add_argument("-base", required=True, metavar="MEAN.nii.gz",
                      help="Reference EPI volume (3D). Typically the mean or "
                           "base volume from motion correction")
    g_mot = p.add_mutually_exclusive_group(required=True)
    g_mot.add_argument("-aff12", metavar="MOCO.aff12.1D",
                       help="AFNI-format .aff12.1D matrix file from ffs_moco "
                            "(-1Dmatrix_save output, one 3x4 matrix per volume)")
    g_mot.add_argument("-1Dfile", dest="onedfile", metavar="MOTION.1D",
                       help="6-column motion parameter file from ffs_moco "
                            "(-1Dfile output: roll pitch yaw dS dL dP)")
    g_mot.add_argument("-dfile", metavar="DFILE.1D",
                       help="9-column diagnostic file from ffs_moco "
                            "(-dfile output: vol# roll pitch yaw dI dS dL rms_bef rms_aft)")
    g_io.add_argument("-prefix", required=True, metavar="PREFIX",
                      help="Output prefix. Produces PREFIX_motsim.1D (regressors)")
    g_io.add_argument("-verb", type=int, default=1, metavar="LEVEL",
                      help="Verbosity: 0=quiet, 1=normal, 2=debug [default: %(default)s]")

    # ── PCA options ──
    g_pca = p.add_argument_group("PCA Options")
    g_pca.add_argument("-n_pcs", type=int, default=12, metavar="N",
                       help="Number of PCs to extract [default: %(default)s]")
    g_pca.add_argument("-variant", choices=["forward", "backward", "both"],
                       default="both",
                       help="Which simulation(s) to use: 'forward' = inverse-motion "
                            "simulation only, 'backward' = re-registered simulation only, "
                            "'both' = spatial concatenation (recommended, Patriat et al.) "
                            "[default: %(default)s]")

    # ── Mask ──
    g_mask = p.add_argument_group("Masking")
    g_mask.add_argument("-mask", default=None, metavar="MASK.nii.gz",
                        help="Brain mask. If not provided, auto-generated from "
                             "the reference via intensity thresholding")
    g_mask.add_argument("-dilate", type=int, default=2, metavar="N",
                        help="Dilate mask by N voxels to capture edge effects "
                             "[default: %(default)s]")

    # ── Processing ──
    g_proc = p.add_argument_group("Processing")
    g_proc.add_argument("-interp", default="cubic",
                        choices=["linear", "cubic", "quintic", "heptic", "wsinc5"],
                        help="Interpolation method for resampling "
                             "[default: %(default)s]")
    g_proc.add_argument("-save_sim", action="store_true",
                        help="Also save the simulated 4D volumes as NIfTI "
                             "(PREFIX_forward.nii.gz, PREFIX_backward.nii.gz)")
    g_proc.add_argument("-device", type=str, default=None,
                        help="Force device: 'cuda', 'cpu', etc.")

    return p.parse_args(argv)


def _load_motion_1d(path: str) -> np.ndarray:
    """Load 6-column .1D motion file (roll pitch yaw dS dL dP).

    Returns (nt, 6) DICOM params [dx, dy, dz, rz, rx, ry] matching
    the internal convention used by ffs_moco.
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) < 6:
                raise ValueError(f"Expected 6 columns in {path}, got {len(vals)}")
            roll, pitch, yaw, dS, dL, dP = vals[:6]
            # Reverse AFNI mapping: rz=-roll, rx=pitch, ry=yaw, dz=-dS, dx=dL, dy=dP
            rows.append([dL, dP, -dS, -roll, pitch, yaw])
    return np.array(rows, dtype=np.float64)


def _load_dfile(path: str) -> np.ndarray:
    """Load 9-column dfile (vol# roll pitch yaw dS dL dP rms_bef rms_aft).

    Returns (nt, 6) DICOM params [dx, dy, dz, rz, rx, ry].
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) < 7:
                raise ValueError(f"Expected >= 7 columns in dfile {path}, got {len(vals)}")
            # Columns: vol# roll pitch yaw dS dL dP [rms_bef rms_aft]
            _, roll, pitch, yaw, dS, dL, dP = vals[:7]
            # Reverse AFNI mapping (same as .1D)
            rows.append([dL, dP, -dS, -roll, pitch, yaw])
    return np.array(rows, dtype=np.float64)


def _params_to_voxel_matrices(
    params_dicom: np.ndarray,
    nifti_affine: np.ndarray,
) -> Tensor:
    """Convert (nt, 6) DICOM rigid params to (nt, 4, 4) voxel-space matrices.

    Builds 4×4 DICOM-space matrices from [dx, dy, dz, rz, rx, ry],
    then converts to voxel index space using the NIfTI affine.
    """
    nt = params_dicom.shape[0]
    matrices_vox = torch.zeros(nt, 4, 4, dtype=torch.float32)

    for t in range(nt):
        # Build full 12-param vector: [dx,dy,dz, rz,rx,ry, sx,sy,sz, shyx,shzx,shzy]
        p12 = torch.zeros(12, dtype=torch.float32)
        p12[:6] = torch.from_numpy(params_dicom[t].astype(np.float32))
        p12[6:9] = 1.0  # scales = identity

        M_dicom = params_to_matrix(p12)
        M_vox = dicom_matrix_to_voxel(M_dicom, nifti_affine, nifti_affine)
        matrices_vox[t] = M_vox

    return matrices_vox


def _automask_dilate(vol: Tensor, dilate_voxels: int = 2) -> Tensor:
    """Create a brain mask via intensity thresholding + dilation.

    Uses a simple approach: threshold at median of nonzero voxels,
    then binary dilation via max-pool.
    """
    nonzero = vol[vol > 0]
    if nonzero.numel() == 0:
        return torch.ones_like(vol, dtype=torch.bool)
    thresh = nonzero.median().item() * 0.2
    mask = vol > thresh

    if dilate_voxels > 0:
        # Binary dilation via 3D max pool
        m = mask.float()[None, None]  # (1,1,D,H,W)
        kernel = 2 * dilate_voxels + 1
        m = torch.nn.functional.max_pool3d(
            m, kernel_size=kernel, stride=1,
            padding=dilate_voxels,
        )
        mask = m[0, 0] > 0.5

    return mask


def _run_forward_sim(
    reference: Tensor,
    matrices_vox: Tensor,
    device: torch.device,
    interp: str = "cubic",
    verb: int = 1,
) -> Tensor:
    """Create forward simulation: apply inverse motion to reference.

    For each timepoint, inverts the registration matrix and applies it
    to the reference, simulating what the scanner would have acquired
    if the head was at that position.

    Args:
        reference: (nz, ny, nx) reference EPI.
        matrices_vox: (nt, 4, 4) voxel-space registration matrices
            (output→source mapping, as loaded from .aff12.1D).
        device: torch device.
        interp: interpolation method.
        verb: verbosity.

    Returns:
        (nt, nz, ny, nx) forward simulation.
    """
    nt = matrices_vox.shape[0]
    nz, ny, nx = reference.shape
    forward_sim = torch.zeros(nt, nz, ny, nx, dtype=torch.float32)

    ref_gpu = reference.to(device)

    for t in range(nt):
        # The loaded matrix M_t maps base → source_t (pull mapping for registration).
        # To simulate "reference as seen from position t", we invert it:
        # M_t_inv maps source_t → base, so apply_affine_interp(ref, M_t_inv)
        # resamples the reference into the source_t coordinate frame.
        M_inv = torch.linalg.inv(matrices_vox[t]).to(device)
        forward_sim[t] = apply_affine_interp(ref_gpu, M_inv, interp=interp).cpu()

    if verb >= 1:
        print(f"Forward simulation: {nt} volumes ({interp} interp)")

    return forward_sim


def _run_backward_sim(
    forward_sim: Tensor,
    reference: Tensor,
    device: torch.device,
    interp: str = "cubic",
    verb: int = 1,
) -> Tensor:
    """Create backward simulation: re-register forward sim to reference.

    Runs rigid-body registration on the forward-simulated data back to
    the reference. The result captures residual interpolation artifacts
    that standard motion correction cannot remove.

    Args:
        forward_sim: (nt, nz, ny, nx) forward simulation.
        reference: (nz, ny, nx) reference EPI.
        device: torch device.
        interp: interpolation method for estimation and final output.
        verb: verbosity.

    Returns:
        (nt, nz, ny, nx) backward simulation.
    """
    from .ffs_moco import MocoConfig, moco

    # moco's internal resampler doesn't support "linear" — use "cubic" as minimum
    moco_interp = interp if interp != "linear" else "cubic"

    config = MocoConfig(
        base_index=0,
        cost="wls",
        interp=moco_interp,
        final_interp=moco_interp,
        max_iter=5,
        chain_init=True,
        compile=False,           # small dataset, not worth compile overhead
        device=str(device),
    )

    # Prepend reference as volume 0 so moco uses it as the base
    sim_with_ref = torch.cat([reference.unsqueeze(0), forward_sim], dim=0)

    if verb >= 1:
        print(f"Backward simulation: re-registering {forward_sim.shape[0]} volumes...")

    result = moco(sim_with_ref, config)

    # Drop the reference volume (index 0) from the aligned output
    backward_sim = result.aligned[1:]
    return backward_sim


def _extract_pcs(
    data_4d: Tensor,
    mask: Tensor,
    n_pcs: int,
    verb: int = 1,
) -> tuple[Tensor, Tensor]:
    """Extract temporal PCs from masked 4D data.

    Uses the project's PCA class (covariance-trick SVD, efficient for
    n_voxels >> n_timepoints which is always the case for fMRI).

    Args:
        data_4d: (nt, nz, ny, nx) simulation data.
        mask: (nz, ny, nx) boolean mask.
        n_pcs: number of PCs.

    Returns:
        (pcs, var_explained): pcs is (nt, n_pcs), var_explained is (n_pcs,).
    """
    from ..pca import PCA

    nt = data_4d.shape[0]
    n_pcs = min(n_pcs, nt - 1)

    # Flatten to (nt, n_voxels_in_mask)
    mask_flat = mask.reshape(-1)
    mat = data_4d.reshape(nt, -1)[:, mask_flat].float()

    # PCA class handles centering + covariance trick (n_voxels >> n_timepoints)
    pca = PCA(n_components=n_pcs)
    scores = pca.fit_transform(mat)  # (nt, n_pcs)

    # Normalize scores to unit variance for use as regressors
    sc_std = scores.std(dim=0, keepdim=True).clamp(min=1e-10)
    pcs = scores / sc_std

    var_explained = pca.explained_variance_ratio_[:n_pcs]

    if verb >= 2:
        cumvar = var_explained.cumsum(0)
        print(f"  PC variance explained (cumulative): "
              f"{', '.join(f'{v*100:.1f}%' for v in cumvar.tolist())}")

    return pcs, var_explained


def _save_1d(pcs: Tensor, var_explained: Tensor, path: str,
             variant: str, n_vols: int) -> None:
    """Write PCs as AFNI-style .1D file."""
    n_pcs = pcs.shape[1]
    with open(path, "w") as f:
        f.write("# MotSim regressors (Patriat et al. 2017, PMC5533292)\n")
        f.write(f"# Variant: {variant}, {n_vols} volumes, {n_pcs} PCs\n")
        f.write(f"# Variance explained: "
                f"{' '.join(f'{v*100:.2f}%' for v in var_explained.tolist())}\n")
        for row in pcs.cpu().numpy():
            f.write("  ".join(f"{v: .6f}" for v in row) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.time()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if args.verb >= 1:
        print(f"ffs_motsim: device={device}")

    # Load reference
    ref_data, header_info = load_image(args.base, device=torch.device("cpu"))
    if ref_data.ndim == 4:
        if args.verb >= 1:
            print(f"Reference is 4D ({ref_data.shape[0]} vols), using mean")
        ref_data = ref_data.float().mean(dim=0)
    reference = ref_data.float()
    nz, ny, nx = reference.shape
    if args.verb >= 1:
        print(f"Reference: {nx}x{ny}x{nz}")

    # Load motion matrices (from whichever format was provided)
    nifti_affine = header_info["affine"]
    if args.aff12:
        aff_xform = load_affine_1D(
            args.aff12, output_affine=nifti_affine,
            device=torch.device("cpu"), debug=(args.verb >= 2),
        )
        matrices_vox = aff_xform.matrices  # (nt, 4, 4) in voxel space
        src_label = args.aff12
    elif args.onedfile:
        params_dicom = _load_motion_1d(args.onedfile)
        matrices_vox = _params_to_voxel_matrices(params_dicom, nifti_affine)
        src_label = args.onedfile
    else:
        params_dicom = _load_dfile(args.dfile)
        matrices_vox = _params_to_voxel_matrices(params_dicom, nifti_affine)
        src_label = args.dfile
    nt = matrices_vox.shape[0]
    if args.verb >= 1:
        print(f"Motion matrices: {nt} timepoints (from {src_label})")

    # Mask
    if args.mask:
        mask_data, _ = load_image(args.mask, device=torch.device("cpu"))
        mask = mask_data > 0.5
    else:
        mask = _automask_dilate(reference, dilate_voxels=args.dilate)
    n_vox = mask.sum().item()
    if args.verb >= 1:
        print(f"Mask: {n_vox} voxels ({n_vox / mask.numel() * 100:.1f}%)")

    # Prefix
    prefix = args.prefix
    for ext in (".nii.gz", ".nii"):
        if prefix.endswith(ext):
            prefix = prefix[:-len(ext)]

    # --- Forward simulation ---
    forward_sim = _run_forward_sim(reference, matrices_vox, device,
                                    interp=args.interp, verb=args.verb)

    if args.save_sim:
        save_image(forward_sim, f"{prefix}_forward.nii.gz", header_info=header_info)
        if args.verb >= 1:
            print(f"Saved: {prefix}_forward.nii.gz")

    # --- Backward simulation (if needed) ---
    backward_sim = None
    if args.variant in ("backward", "both"):
        backward_sim = _run_backward_sim(
            forward_sim, reference, device,
            interp=args.interp, verb=args.verb,
        )
        if args.save_sim:
            save_image(backward_sim, f"{prefix}_backward.nii.gz",
                       header_info=header_info)
            if args.verb >= 1:
                print(f"Saved: {prefix}_backward.nii.gz")

    # --- Extract PCs ---
    if args.variant == "forward":
        pca_input = forward_sim
    elif args.variant == "backward":
        pca_input = backward_sim
    else:  # "both"
        # Concatenate spatially: (nt, 2*nz, ny, nx) — doubles the voxel dimension
        # This is the recommended approach from the paper (12Both)
        pca_input = torch.cat([forward_sim, backward_sim], dim=1)

    pcs, var_explained = _extract_pcs(
        pca_input, mask if args.variant != "both" else _expand_mask_both(mask),
        args.n_pcs, args.verb,
    )

    # Save
    out_path = f"{prefix}_motsim.1D"
    _save_1d(pcs, var_explained, out_path, args.variant, nt)

    elapsed = time.time() - t0
    if args.verb >= 1:
        var_pct = [f"{v*100:.1f}%" for v in var_explained.tolist()]
        print(f"Extracted {args.n_pcs} MotSim PCs ({args.variant}), "
              f"var explained: {', '.join(var_pct)}")
        print(f"Saved: {out_path}")
        print(f"Done. ({elapsed:.1f}s)")

    return 0


def _expand_mask_both(mask: Tensor) -> Tensor:
    """Expand mask for 'both' variant: repeat along z (dim 0) for concat."""
    return torch.cat([mask, mask], dim=0)


if __name__ == "__main__":
    sys.exit(main())
