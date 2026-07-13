"""Export GLMsingle .mat results to NIfTI files for benchmark comparison.

Reads the glmsingle_comparison.mat (HDF5/v7.3) and exports 3D/4D NIfTI
files to glmsingle/. Handles MATLAB's Fortran-order
flattening by reshaping back to 3D volume before saving.

This is equivalent to Phase 2 of run_glmsingle_comparison.m but runs in
Python so the user doesn't need MATLAB just to re-export.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def export_glmsingle_niftis(
    mat_file: str | Path,
    template_nifti: str | Path,
    output_dir: str | Path,
    force: bool = False,
) -> bool:
    """Export GLMsingle .mat results to NIfTI files.

    Parameters
    ----------
    mat_file : path
        Path to glmsingle_comparison.mat (HDF5/v7.3 format).
    template_nifti : path
        Path to a NIfTI file to use as spatial template (affine, header).
    output_dir : path
        Directory to write NIfTI outputs.
    force : bool
        If True, overwrite existing files. If False, skip if outputs exist.

    Returns
    -------
    bool
        True if export was performed, False if skipped (files exist).
    """
    import h5py
    import nibabel as nib

    output_dir = Path(output_dir)
    check_file = output_dir / "glmsingle_hrf_index.nii.gz"
    if check_file.exists() and not force:
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load template for affine and header
    template = nib.load(str(template_nifti))
    affine = template.affine

    # Load .mat file
    with h5py.File(str(mat_file), "r") as f:
        vol_size = tuple(np.array(f["vol_size"]).flatten().astype(int))
        nx, ny, nz = vol_size

        def load_flat(key: str) -> np.ndarray:
            """Load a flattened MATLAB array and reorder from Fortran to C."""
            data = np.array(f[key]).flatten().astype(np.float32)
            return data.reshape(vol_size, order="F")

        def load_2d(key: str) -> np.ndarray:
            """Load a 2D MATLAB array (n_voxels x n_cols) → 4D NIfTI."""
            raw = np.array(f[key]).astype(np.float32)
            n_vox = np.prod(vol_size)
            # h5py transposes MATLAB arrays: (n_cols, n_vox) or (n_vox, n_cols)
            if raw.shape[0] == n_vox:
                data_2d = raw  # (n_vox, n_cols)
            elif raw.shape[1] == n_vox:
                data_2d = raw.T  # transpose to (n_vox, n_cols)
            else:
                raise ValueError(
                    f"Shape mismatch for {key}: {raw.shape}, expected one dim = {n_vox}"
                )
            n_cols = data_2d.shape[1]
            # Reorder each column from Fortran to C
            vol4d = np.zeros((nx, ny, nz, n_cols), dtype=np.float32)
            for c in range(n_cols):
                vol4d[:, :, :, c] = data_2d[:, c].reshape(vol_size, order="F")
            return vol4d

        def save_3d(data_3d: np.ndarray, name: str) -> None:
            img = nib.Nifti1Image(data_3d.astype(np.float32), affine)
            path = output_dir / name
            nib.save(img, str(path))
            print(f"  Saved: {path}")

        def save_4d(data_4d: np.ndarray, name: str) -> None:
            img = nib.Nifti1Image(data_4d.astype(np.float32), affine)
            path = output_dir / name
            nib.save(img, str(path))
            print(f"  Saved: {path}  ({data_4d.shape[3]} volumes)")

        print("Exporting GLMsingle results to NIfTI...")

        # --- Type B ---
        print("  Type B (HRF selection):")
        save_3d(load_flat("HRFindex"), "glmsingle_hrf_index.nii.gz")
        save_3d(load_flat("R2_B"), "glmsingle_r2_B.nii.gz")
        save_4d(load_2d("FitHRFR2"), "glmsingle_fithrf_r2.nii.gz")
        save_4d(load_2d("modelmd_B"), "glmsingle_betas_B.nii.gz")

        # --- Type C ---
        print("  Type C (PC denoising):")
        save_3d(load_flat("noisepool"), "glmsingle_noisepool.nii.gz")
        save_3d(load_flat("R2_C"), "glmsingle_r2_C.nii.gz")
        save_4d(load_2d("modelmd_C"), "glmsingle_betas_C.nii.gz")

        pcnum = int(np.array(f["pcnum"]).item())
        pcnum_path = output_dir / "glmsingle_pcnum.txt"
        pcnum_path.write_text(f"{pcnum}\n")
        print(f"  Saved: {pcnum_path}  (pcnum={pcnum})")

        xvaltrend = np.array(f["xvaltrend"]).flatten()
        xval_path = output_dir / "glmsingle_xvaltrend.txt"
        np.savetxt(str(xval_path), xvaltrend.reshape(1, -1), fmt="%.6f")
        print(f"  Saved: {xval_path}  ({len(xvaltrend)} values)")

        # --- Type D ---
        print("  Type D (fracridge):")
        save_3d(load_flat("FRACvalue"), "glmsingle_fracvalue.nii.gz")
        save_3d(load_flat("R2_D"), "glmsingle_r2_D.nii.gz")
        save_4d(load_2d("modelmd_D"), "glmsingle_betas_D.nii.gz")
        save_4d(load_2d("scaleoffset"), "glmsingle_scaleoffset.nii.gz")

        # --- Shared ---
        save_3d(load_flat("mask"), "glmsingle_mask.nii.gz")

    print(f"  Export complete: {output_dir}")
    return True
