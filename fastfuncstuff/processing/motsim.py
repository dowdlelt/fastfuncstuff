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
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .affine import apply_affine_interp, dicom_matrix_to_voxel, params_to_matrix


def load_motion_1d(path: str) -> np.ndarray:
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


def load_dfile(path: str) -> np.ndarray:
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


def params_to_voxel_matrices(
    params_dicom: np.ndarray,
    nifti_affine: np.ndarray,
) -> Tensor:
    """Convert (nt, 6) DICOM rigid params to (nt, 4, 4) voxel-space matrices.

    Builds 4x4 DICOM-space matrices from [dx, dy, dz, rz, rx, ry],
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


def automask_dilate(vol: Tensor, dilate_voxels: int = 2) -> Tensor:
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
            m,
            kernel_size=kernel,
            stride=1,
            padding=dilate_voxels,
        )
        mask = m[0, 0] > 0.5

    return mask


def run_forward_sim(
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
            (output->source mapping, as loaded from .aff12.1D).
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
        M_inv = torch.linalg.inv(matrices_vox[t]).to(device)
        forward_sim[t] = apply_affine_interp(ref_gpu, M_inv, interp=interp).cpu()

    if verb >= 1:
        print(f"Forward simulation: {nt} volumes ({interp} interp)")

    return forward_sim


def run_backward_sim(
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
        compile=False,
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


def extract_pcs(
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
    from fastfuncstuff.decomposition.pca import PCA

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
        print(
            f"  PC variance explained (cumulative): "
            f"{', '.join(f'{v * 100:.1f}%' for v in cumvar.tolist())}"
        )

    return pcs, var_explained


def save_1d(pcs: Tensor, var_explained: Tensor, path: str, variant: str, n_vols: int) -> None:
    """Write PCs as AFNI-style .1D file."""
    n_pcs = pcs.shape[1]
    with open(path, "w") as f:
        f.write("# MotSim regressors (Patriat et al. 2017, PMC5533292)\n")
        f.write(f"# Variant: {variant}, {n_vols} volumes, {n_pcs} PCs\n")
        f.write(
            f"# Variance explained: {' '.join(f'{v * 100:.2f}%' for v in var_explained.tolist())}\n"
        )
        for row in pcs.cpu().numpy():
            f.write("  ".join(f"{v: .6f}" for v in row) + "\n")


def expand_mask_both(mask: Tensor) -> Tensor:
    """Expand mask for 'both' variant: repeat along z (dim 0) for concat."""
    return torch.cat([mask, mask], dim=0)
