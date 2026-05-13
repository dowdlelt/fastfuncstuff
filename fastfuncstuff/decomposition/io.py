"""
I/O utilities for PCA and ICA decomposition results

Save and load PCA/ICA spatial maps and timeseries in NIfTI format,
compatible with AFNI and other neuroimaging tools.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.io.afni import get_tr_from_file, load_nifti, save_nifti


def save_masked_component_maps_4d(
    components_kv: np.ndarray,
    mask3d: np.ndarray | None,
    shape3d: tuple[int, int, int],
    affine: np.ndarray,
    out_file: str | Path,
) -> Path:
    """Save (n_components, n_voxels) maps into a 4D image with optional masking."""
    k, n_vox = components_kv.shape
    out = np.zeros((*shape3d, k), dtype=np.float32)
    if mask3d is None:
        if np.prod(shape3d) != n_vox:
            raise ValueError("Component size does not match full volume size")
        for i in range(k):
            out[..., i] = components_kv[i].reshape(shape3d)
    else:
        flat_mask = mask3d.reshape(-1)
        for i in range(k):
            vol = np.zeros(flat_mask.shape[0], dtype=np.float32)
            vol[flat_mask] = components_kv[i]
            out[..., i] = vol.reshape(shape3d)

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_nifti(out, output_path=out_path, affine=affine)
    return out_path


def save_masked_component_map_3d(
    component_v: np.ndarray,
    mask3d: np.ndarray | None,
    shape3d: tuple[int, int, int],
    affine: np.ndarray,
    out_file: str | Path,
) -> Path:
    """Save a single (n_voxels,) map into a 3D image with optional masking."""
    n_vox = component_v.shape[0]
    if mask3d is None:
        if np.prod(shape3d) != n_vox:
            raise ValueError("Component size does not match full volume size")
        out = component_v.reshape(shape3d).astype(np.float32)
    else:
        flat_mask = mask3d.reshape(-1)
        out = np.zeros(flat_mask.shape[0], dtype=np.float32)
        out[flat_mask] = component_v.astype(np.float32)
        out = out.reshape(shape3d)

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_nifti(out, output_path=out_path, affine=affine)
    return out_path


def _compute_psc(
    raw_components_kv: np.ndarray,
    mixing_tk: np.ndarray,
    masked_mean_v: np.ndarray,
    varnorm_std_v: np.ndarray | None = None,
    mixing_amplitude_k: np.ndarray | None = None,
    psc_clip: float = 50.0,
) -> np.ndarray:
    """Compute PSC maps (K, V).

    PSC[k, v] = 100 * amplitude_k * raw_comp[k, v] * varnorm_std[v] / mean[v]

    ICA operates on varnorm-divided data, so the reconstruction is:
        mixing[t,k] * comp_raw[k,v] ≈ data_varnorm[t,v] = data_original[t,v] / varnorm_std[v]

    mixing_amplitude_k must be the per-component std of the mixing matrix BEFORE
    any var_norm step.  var_norm divides each column by its std, so
    std(mixing_varnormed[:,k]) = 1 regardless of true amplitude, inflating PSC
    ~7–10×.  Pass the pre-var_norm std (shape (K,)) to get correct amplitudes.

    varnorm_std_v must be in the same units as masked_mean_v (both raw-scanner
    or both PSC-scaled).  When provided, PSC values are in the correct range
    (~0.1–5 % for BOLD).  Without it, values are off by the CoV (~50–100×).

    Uses the pre-noise-normalisation ("raw oIC") spatial maps so that
    mixing @ raw_comp ≈ preprocessed_data and the amplitude scale is preserved.
    Noise-normalised maps are z-stat-scaled and must NOT be used here.

    Returns float32 (K, V) array, clipped to ±psc_clip.
    """
    if mixing_amplitude_k is not None:
        mix_std = np.asarray(mixing_amplitude_k, dtype=np.float64).ravel()  # (K,)
    else:
        mix_std = np.asarray(mixing_tk, dtype=np.float64).std(axis=0)  # (K,)
    comp = np.asarray(raw_components_kv, dtype=np.float64)          # (K, V)
    mean_v = np.asarray(masked_mean_v, dtype=np.float64)

    pos = mean_v[mean_v > 0]
    eps = max(1e-6, 1e-3 * float(np.median(pos) if pos.size > 0 else 1.0))
    safe_mean = np.where(np.abs(mean_v) < eps, 1.0, mean_v)

    if varnorm_std_v is not None:
        std_v = np.asarray(varnorm_std_v, dtype=np.float64).ravel()
        psc = (100.0 * mix_std[:, None] * comp * std_v[None, :] / safe_mean[None, :]).astype(np.float32)
    else:
        psc = (100.0 * mix_std[:, None] * comp / safe_mean[None, :]).astype(np.float32)

    psc[:, np.abs(mean_v) < eps] = 0.0
    if psc_clip > 0:
        np.clip(psc, -float(psc_clip), float(psc_clip), out=psc)
    return psc


def _write_interleaved_stat_bucket(
    vol1_kv: np.ndarray,
    vol2_kv: np.ndarray,
    label1: str,
    label2: str,
    out_file: Path,
    mask3d: np.ndarray | None,
    shape3d: tuple[int, int, int],
    affine: np.ndarray,
    stat1_type: str | None = None,
    stat2_type: str | None = None,
) -> Path:
    """Write interleaved [vol1_0, vol2_0, vol1_1, vol2_1, ...] 4D NIfTI.

    Applies 3drefit sub-brick labels (and stat types where given) when
    3drefit is on PATH; otherwise writes a companion _labels.txt file.
    """
    import shutil
    import subprocess

    k = vol1_kv.shape[0]
    n_vox = vol1_kv.shape[1]
    n_digits = len(str(k))

    out_4d = np.zeros((*shape3d, 2 * k), dtype=np.float32)
    if mask3d is None:
        for i in range(k):
            out_4d[..., 2 * i] = vol1_kv[i].astype(np.float32).reshape(shape3d)
            out_4d[..., 2 * i + 1] = vol2_kv[i].astype(np.float32).reshape(shape3d)
    else:
        flat_mask = mask3d.reshape(-1).astype(bool)
        for i in range(k):
            v1 = np.zeros(flat_mask.shape[0], dtype=np.float32)
            v2 = np.zeros(flat_mask.shape[0], dtype=np.float32)
            v1[flat_mask] = vol1_kv[i].astype(np.float32)
            v2[flat_mask] = vol2_kv[i].astype(np.float32)
            out_4d[..., 2 * i] = v1.reshape(shape3d)
            out_4d[..., 2 * i + 1] = v2.reshape(shape3d)

    out_file.parent.mkdir(parents=True, exist_ok=True)

    out_str = str(out_file)
    if out_str.endswith(".nii.gz"):
        tmp_path = Path(out_str[:-7] + ".nii")
    else:
        tmp_path = out_file
    save_nifti(out_4d, output_path=tmp_path, affine=affine)

    sub_labels = []
    for i in range(k):
        tag = str(i + 1).zfill(n_digits)
        sub_labels.append(f"IC{tag}_{label1}")
        sub_labels.append(f"IC{tag}_{label2}")

    has_3drefit = shutil.which("3drefit") is not None
    if has_3drefit:
        cmd = ["3drefit"]
        for i, lab in enumerate(sub_labels):
            cmd.extend(["-sublabel", str(i), lab])
        for i in range(k):
            if stat1_type:
                cmd.extend(["-substatpar", str(2 * i), stat1_type])
            if stat2_type:
                cmd.extend(["-substatpar", str(2 * i + 1), stat2_type])
        cmd.append(str(tmp_path))
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"  WARN: 3drefit failed: {e.stderr.decode(errors='replace')[-300:]}")
    else:
        labels_txt = Path(str(tmp_path.with_suffix("")) + "_labels.txt")
        with labels_txt.open("w") as f:
            for i, lab in enumerate(sub_labels):
                f.write(f"{i}\t{lab}\n")

    if tmp_path != out_file:
        from fastfuncstuff.io.afni import compress_nifti
        compress_nifti(str(tmp_path), str(out_file), remove_original=True)

    return out_file


def save_psc_zstat_bucket(
    components_kv: np.ndarray,
    z_maps_kv: np.ndarray,
    mixing_tk: np.ndarray,
    mean3d: np.ndarray,
    mask3d: np.ndarray | None,
    shape3d: tuple[int, int, int],
    affine: np.ndarray,
    out_file: str | Path,
    psc_clip: float = 50.0,
) -> Path:
    """Save AFNI-style interleaved PSC + Z-stat bucket (legacy convenience wrapper).

    For each component k: sub-brick 2k = PSC, sub-brick 2k+1 = Z (FIZT).

    ``components_kv`` must be the PRE-noise-normalisation ("raw oIC") maps so
    that mixing @ components ≈ preprocessed_data and the PSC scale is correct.
    Passing noise-normalised z-stat maps will produce meaningless amplitudes.
    """
    flat_mean = np.asarray(mean3d, dtype=np.float64).reshape(-1)
    if mask3d is not None:
        masked_mean = flat_mean[mask3d.reshape(-1).astype(bool)]
    else:
        masked_mean = flat_mean
    psc_kv = _compute_psc(components_kv, mixing_tk, masked_mean, psc_clip=psc_clip)
    return _write_interleaved_stat_bucket(
        vol1_kv=psc_kv,
        vol2_kv=z_maps_kv.astype(np.float32),
        label1="PSC",
        label2="Z",
        out_file=Path(out_file),
        mask3d=mask3d,
        shape3d=shape3d,
        affine=affine,
        stat2_type="fizt",
    )


def safe_relative_symlink(target: str | Path, link_path: str | Path) -> Path:
    """Create/replace symlink at link_path pointing to target via relative path."""
    target_path = Path(target)
    link = Path(link_path)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    rel_target = os.path.relpath(str(target_path), start=str(link.parent))
    link.symlink_to(rel_target)
    return link


def write_melodic_compat_outputs(
    compat_dir: str | Path,
    maps_file: str | Path,
    zmaps_file: str | Path | None,
    timecourse_file: str | Path,
    pca_scree_ratio: np.ndarray,
    component_explained_share_pct: np.ndarray,
    component_total_share_pct: np.ndarray,
    mixing_np: np.ndarray,
    mask3d: np.ndarray | None,
    mean3d: np.ndarray,
    shape3d: tuple[int, int, int],
    affine: np.ndarray,
    z_maps: np.ndarray | None = None,
    p_maps: np.ndarray | None = None,
    thresh_z_maps: np.ndarray | None = None,
    comp_kv_for_stats: np.ndarray | None = None,
    oic_components_kv: np.ndarray | None = None,
    varnorm_std_v: np.ndarray | None = None,
    mixing_amplitude_k: np.ndarray | None = None,
    write_per_comp_stats: bool = False,
    write_psc_prob: bool = True,
    write_zp: bool = True,
    psc_clip: float = 50.0,
) -> Path:
    """Write MELODIC-style compatibility files for ICA outputs.

    Optional parameters extend MELODIC parity:
      comp_kv_for_stats    : (K, V) post-noise-norm spatial IC maps; used to
                             compute the spatial-kurtosis column of melodic_ICstats.
      oic_components_kv    : (K, V) raw pre-noise-norm IC maps; saved as
                             melodic_oIC.nii.gz and used for PSC computation.
      varnorm_std_v        : (V,) per-voxel temporal std used for varnorm, in the
                             same units as mean3d.  Required for correct PSC values:
                             without it, PSC is off by the CoV (~50-100×).
      mixing_amplitude_k   : (K,) per-component std of mixing BEFORE var_norm.
                             var_norm sets std(mixing[:,k])=1, inflating PSC ~7-10×.
                             Pass this to use the true pre-var_norm amplitude.
      write_per_comp_stats : Write per-component 3D probmap_NNN.nii.gz and
                             thresh_zstatNNN.nii.gz (zero-padded).  Off by
                             default — the 4D bucket files cover the same data
                             more compactly.
      write_psc_prob       : Write stats/psc_prob.nii.gz — interleaved
                             [PSC_k, Prob_k] for AFNI viewing (default on).
                             PSC uses oic_components_kv (pre-noise-norm) so
                             mixing @ oic ≈ preprocessed data; requires both
                             oic_components_kv and p_maps.
      write_zp             : Write stats/z_prob.nii.gz — interleaved [Z_k,
                             Prob_k] for AFNI viewing (default on); requires
                             z_maps and p_maps.
      psc_clip             : Clip PSC to ±psc_clip % (default 50).
    """
    compat = Path(compat_dir)
    maps = Path(maps_file)
    zmap_path = Path(zmaps_file) if zmaps_file is not None else None
    tcs = Path(timecourse_file)
    compat.mkdir(parents=True, exist_ok=True)

    # MELODIC's melodic_IC.nii.gz contains the mixture-model z-stat maps (range
    # ±30–50, std≈1), not the raw noise-normed IC maps.  When GGM zmaps are
    # available, symlink to those; otherwise fall back to noise-normed maps.
    ic_target = zmap_path if (zmap_path is not None and zmap_path.exists()) else maps
    safe_relative_symlink(ic_target, compat / "melodic_IC.nii.gz")
    safe_relative_symlink(tcs, compat / "melodic_mix")
    safe_relative_symlink(tcs, compat / "melodic_Tmodes")

    save_nifti(mean3d.astype(np.float32), output_path=compat / "mean.nii.gz", affine=affine)
    if mask3d is None:
        mask_out = np.ones(shape3d, dtype=np.float32)
    else:
        mask_out = mask3d.astype(np.float32)
    save_nifti(mask_out, output_path=compat / "mask.nii.gz", affine=affine)

    ftmix = np.abs(np.fft.rfft(mixing_np, axis=0)) ** 2
    if ftmix.shape[0] > 1:
        ftmix = ftmix[1:, :]
    np.savetxt(compat / "melodic_FTmix", ftmix, fmt="%.8f")

    # melodic_unmix: K × T, the full unmixing W = pinv(mixing). MELODIC's
    # melodic_unmix folds in the whitening, so this is the source-extraction
    # operator that maps voxel timecourses back to component timecourses.
    unmix = np.linalg.pinv(np.asarray(mixing_np, dtype=np.float64))
    np.savetxt(compat / "melodic_unmix", unmix, fmt="%.8f")

    # melodic_ICstats: K × 4 — explained%, total%, spatial kurtosis, signal-fraction.
    n_k = component_explained_share_pct.shape[0]
    if comp_kv_for_stats is not None and comp_kv_for_stats.shape[0] >= n_k:
        from scipy.stats import kurtosis as _kurt
        ic_kurt = np.asarray([_kurt(comp_kv_for_stats[i]) for i in range(n_k)], dtype=np.float64)
    else:
        ic_kurt = np.zeros(n_k, dtype=np.float64)
    if p_maps is not None and p_maps.shape[0] >= n_k:
        ic_signal = np.asarray([float(np.mean(p_maps[i])) for i in range(n_k)], dtype=np.float64)
    else:
        ic_signal = np.zeros(n_k, dtype=np.float64)
    icstats = np.column_stack([
        component_explained_share_pct,
        component_total_share_pct,
        ic_kurt,
        ic_signal,
    ])
    np.savetxt(compat / "melodic_ICstats", icstats, fmt="%.8f")
    np.savetxt(compat / "eigenvalues_percent", pca_scree_ratio * 100.0, fmt="%.8f")

    # melodic_oIC: raw IC spatial maps before noise-normalization (FFS analog
    # of MELODIC's "original" ICs).  Only written if the caller saved a copy
    # of components prior to apply_melodic_noise_normalization.
    if oic_components_kv is not None:
        save_masked_component_maps_4d(
            components_kv=np.asarray(oic_components_kv, dtype=np.float32),
            mask3d=mask3d,
            shape3d=shape3d,
            affine=affine,
            out_file=compat / "melodic_oIC.nii.gz",
        )

    # stats/ sub-folder: bucket files + optional per-component 3D outputs
    need_stats_dir = (
        (write_per_comp_stats and z_maps is not None and p_maps is not None)
        or (write_psc_prob and oic_components_kv is not None and p_maps is not None)
        or (write_zp and z_maps is not None and p_maps is not None)
    )
    if need_stats_dir:
        stats_dir = compat / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)

        # Pre-compute masked mean for PSC (needed for both psc_prob bucket and
        # per-component stats when oic is available).
        flat_mean = np.asarray(mean3d, dtype=np.float64).reshape(-1)
        masked_mean_v = flat_mean[mask3d.reshape(-1).astype(bool)] if mask3d is not None else flat_mean

        if write_per_comp_stats and z_maps is not None and p_maps is not None:
            n_comp = z_maps.shape[0]
            n_digits = len(str(n_comp))
            for i in range(n_comp):
                tag = str(i + 1).zfill(n_digits)
                save_masked_component_map_3d(
                    component_v=p_maps[i],
                    mask3d=mask3d, shape3d=shape3d, affine=affine,
                    out_file=stats_dir / f"probmap_{tag}.nii.gz",
                )
                if thresh_z_maps is not None:
                    save_masked_component_map_3d(
                        component_v=thresh_z_maps[i],
                        mask3d=mask3d, shape3d=shape3d, affine=affine,
                        out_file=stats_dir / f"thresh_zstat{tag}.nii.gz",
                    )

        if write_psc_prob and oic_components_kv is not None and p_maps is not None:
            n_k = min(oic_components_kv.shape[0], p_maps.shape[0])
            amp_k = mixing_amplitude_k[:n_k] if mixing_amplitude_k is not None else None
            psc_kv = _compute_psc(
                oic_components_kv[:n_k], mixing_np[:, :n_k], masked_mean_v,
                varnorm_std_v=varnorm_std_v,
                mixing_amplitude_k=amp_k,
                psc_clip=psc_clip,
            )
            _write_interleaved_stat_bucket(
                vol1_kv=psc_kv,
                vol2_kv=p_maps[:n_k].astype(np.float32),
                label1="PSC",
                label2="Prob",
                out_file=stats_dir / "psc_prob.nii.gz",
                mask3d=mask3d, shape3d=shape3d, affine=affine,
            )

        if write_zp and z_maps is not None and p_maps is not None:
            n_k = min(z_maps.shape[0], p_maps.shape[0])
            _write_interleaved_stat_bucket(
                vol1_kv=z_maps[:n_k].astype(np.float32),
                vol2_kv=p_maps[:n_k].astype(np.float32),
                label1="Z",
                label2="Prob",
                out_file=stats_dir / "z_prob.nii.gz",
                mask3d=mask3d, shape3d=shape3d, affine=affine,
                stat1_type="fizt",
            )

    return compat


def save_component_maps(
    components: np.ndarray | torch.Tensor,
    mask_file: str | Path,
    output_file: str | Path,
    labels: list | None = None,
) -> None:
    """
    Save spatial components as 4D NIfTI file

    Parameters
    ----------
    components : array-like, shape (n_components, n_voxels)
        Spatial components (e.g., ICA maps or PCA eigenvectors)
    mask_file : str or Path
        Path to mask file defining brain voxels (used for geometry)
    output_file : str or Path
        Output path for 4D NIfTI file
    labels : list of str, optional
        Component labels (saved in AFNI sub-brick labels)

    Notes
    -----
    Output is a 4D NIfTI file with dimensions (x, y, z, n_components).
    Each volume represents one spatial component.

    Examples
    --------
    >>> # Save ICA spatial maps
    >>> save_component_maps(
    ...     ica.components_,
    ...     mask_file='mask.nii.gz',
    ...     output_file='ica_maps.nii.gz',
    ...     labels=[f'IC_{i}' for i in range(ica.n_components_)]
    ... )
    """
    # Convert to numpy
    if isinstance(components, torch.Tensor):
        components = components.cpu().numpy()

    n_components, n_voxels = components.shape

    # Load mask to get geometry
    mask_img = load_nifti(mask_file)
    mask_data = mask_img.get_fdata()
    mask_bool = mask_data > 0

    if mask_bool.sum() != n_voxels:
        raise ValueError(
            f"Number of voxels in components ({n_voxels}) does not match "
            f"number of voxels in mask ({mask_bool.sum()})"
        )

    # Create 4D volume
    output_shape = (*mask_data.shape, n_components)
    output_data = np.zeros(output_shape, dtype=np.float32)

    # Fill in components
    for i in range(n_components):
        output_data[..., i][mask_bool] = components[i]

    # Validate labels if provided
    if labels is not None:
        if len(labels) != n_components:
            raise ValueError(f"Number of labels ({len(labels)}) != n_components ({n_components})")

    # Save
    save_nifti(output_data, output_path=output_file, affine=mask_img.affine)


def load_component_maps(
    map_file: str | Path,
    mask_file: str | Path,
) -> tuple[np.ndarray, list]:
    """
    Load spatial components from 4D NIfTI file

    Parameters
    ----------
    map_file : str or Path
        Path to 4D NIfTI file with component maps
    mask_file : str or Path
        Path to mask file defining brain voxels

    Returns
    -------
    components : np.ndarray, shape (n_components, n_voxels)
        Spatial components
    labels : list of str
        Component labels (if available in AFNI sub-brick labels)

    Examples
    --------
    >>> components, labels = load_component_maps('ica_maps.nii.gz', 'mask.nii.gz')
    >>> print(f"Loaded {components.shape[0]} components")
    """
    # Load mask
    mask_img = load_nifti(mask_file)
    mask_data = mask_img.get_fdata()
    mask_bool = mask_data > 0
    n_voxels = mask_bool.sum()

    # Load component maps
    map_img = load_nifti(map_file)
    map_data = map_img.get_fdata()

    if map_data.ndim != 4:
        raise ValueError(f"Component map file must be 4D, got {map_data.ndim}D")

    n_components = map_data.shape[3]

    # Extract components
    components = np.zeros((n_components, n_voxels), dtype=np.float32)
    for i in range(n_components):
        components[i] = map_data[..., i][mask_bool]

    # Try to extract labels from AFNI extensions
    labels = None
    for ext in map_img.header.extensions:
        if ext.get_code() == 4:  # AFNI extension
            content = ext.get_content().decode("utf-8")
            if "BRICK_LABS=" in content:
                label_str = content.split("BRICK_LABS=")[1].split("\x00")[0]
                labels = label_str.split("~")
                break

    if labels is None:
        labels = [f"Component_{i}" for i in range(n_components)]

    return components, labels


def save_timeseries(
    timeseries: np.ndarray | torch.Tensor,
    output_file: str | Path,
    tr: float | None = None,
    reference_file: str | Path | None = None,
    labels: list | None = None,
) -> None:
    """
    Save component timeseries as .1D file or 2D NIfTI

    Parameters
    ----------
    timeseries : array-like, shape (n_timepoints, n_components)
        Component timeseries (e.g., ICA mixing matrix)
    output_file : str or Path
        Output path (.1D for AFNI format, .nii.gz for NIfTI)
    tr : float, optional
        TR in seconds (for NIfTI header)
    reference_file : str or Path, optional
        Reference fMRI file to extract TR from
    labels : list of str, optional
        Component labels (written as column headers in .1D)

    Examples
    --------
    >>> # Save as AFNI .1D file
    >>> save_timeseries(
    ...     ica.mixing_,
    ...     output_file='ica_timeseries.1D',
    ...     labels=[f'IC_{i}' for i in range(ica.n_components_)]
    ... )
    >>>
    >>> # Save as NIfTI (for FSL compatibility)
    >>> save_timeseries(
    ...     ica.mixing_,
    ...     output_file='ica_timeseries.nii.gz',
    ...     reference_file='func.nii.gz'
    ... )
    """
    # Convert to numpy
    if isinstance(timeseries, torch.Tensor):
        timeseries = timeseries.cpu().numpy()

    n_timepoints, n_components = timeseries.shape

    output_path = Path(output_file)

    if output_path.suffix == ".1D" or output_path.name.endswith(".1D"):
        # Save as AFNI .1D format
        with open(output_path, "w") as f:
            # Write header with column labels if provided
            if labels is not None:
                if len(labels) != n_components:
                    raise ValueError(
                        f"Number of labels ({len(labels)}) != n_components ({n_components})"
                    )
                f.write("# " + " ".join(labels) + "\n")

            # Write data
            np.savetxt(f, timeseries, fmt="%.6f")

    elif output_path.suffix == ".gz" or output_path.suffixes == [".nii", ".gz"]:
        # Save as NIfTI (1 x 1 x n_timepoints x n_components)

        # Get TR
        if tr is None:
            if reference_file is None:
                raise ValueError("Must provide either 'tr' or 'reference_file'")
            tr = get_tr_from_file(str(reference_file))

        # Create 4D volume (1 x 1 x n_timepoints x n_components)
        data_4d = timeseries.T.reshape(1, 1, n_timepoints, n_components)

        # Save with identity affine and TR
        save_nifti(data_4d, output_path=output_path, tr=tr)

    else:
        raise ValueError(f"Unsupported file extension: {output_path.suffix}. Use .1D or .nii.gz")


def load_timeseries(
    timeseries_file: str | Path,
) -> tuple[np.ndarray, list | None]:
    """
    Load component timeseries from file

    Parameters
    ----------
    timeseries_file : str or Path
        Path to timeseries file (.1D or .nii.gz)

    Returns
    -------
    timeseries : np.ndarray, shape (n_timepoints, n_components)
        Component timeseries
    labels : list of str or None
        Component labels (if available)

    Examples
    --------
    >>> timeseries, labels = load_timeseries('ica_timeseries.1D')
    >>> print(f"Shape: {timeseries.shape}")
    >>> print(f"Labels: {labels}")
    """
    path = Path(timeseries_file)

    if path.suffix == ".1D" or path.name.endswith(".1D"):
        # Load AFNI .1D format
        labels = None

        # Try to read labels from header
        with open(path) as f:
            first_line = f.readline()
            if first_line.startswith("#"):
                labels = first_line.strip("# \n").split()

        # Load data
        timeseries = np.loadtxt(path)

        # Ensure 2D
        if timeseries.ndim == 1:
            timeseries = timeseries.reshape(-1, 1)

        return timeseries, labels

    elif path.suffix == ".gz" or path.suffixes == [".nii", ".gz"]:
        # Load NIfTI
        img = load_nifti(path)
        data = img.get_fdata()

        # Reshape from (1, 1, n_timepoints, n_components) to (n_timepoints, n_components)
        if data.ndim == 4:
            n_timepoints = data.shape[2]
            n_components = data.shape[3]
            timeseries = data.reshape(n_timepoints, n_components)
        elif data.ndim == 2:
            timeseries = data
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")

        # No labels in standard NIfTI
        labels = None

        return timeseries, labels

    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")


def save_decomposition_results(
    components: np.ndarray | torch.Tensor,
    timeseries: np.ndarray | torch.Tensor,
    mask_file: str | Path,
    output_prefix: str | Path,
    tr: float | None = None,
    reference_file: str | Path | None = None,
    labels: list | None = None,
    method: str = "ICA",
    nii_ext: str = ".nii.gz",
) -> dict[str, Path]:
    """
    Save complete decomposition results (maps + timeseries)

    Convenience function to save both spatial maps and timeseries with
    consistent naming.

    Parameters
    ----------
    components : array-like, shape (n_components, n_voxels)
        Spatial components
    timeseries : array-like, shape (n_timepoints, n_components)
        Component timeseries
    mask_file : str or Path
        Path to mask file
    output_prefix : str or Path
        Output prefix for files (e.g., 'results/ica_' → 'results/ica_maps.nii.gz')
    tr : float, optional
        TR in seconds
    reference_file : str or Path, optional
        Reference fMRI file
    labels : list of str, optional
        Component labels
    method : str, default='ICA'
        Method name (for default labels)

    Returns
    -------
    output_files : dict
        Dictionary with paths to created files:
        - 'maps': path to spatial maps (4D NIfTI)
        - 'timeseries_1D': path to timeseries (.1D)

    Examples
    --------
    >>> files = save_decomposition_results(
    ...     ica.components_,
    ...     ica.mixing_,
    ...     mask_file='mask.nii.gz',
    ...     output_prefix='results/ica',
    ...     reference_file='func.nii.gz',
    ...     method='ICA'
    ... )
    >>> print(f"Saved: {files['maps']}")
    """
    output_prefix = Path(output_prefix)
    output_dir = output_prefix.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate labels if not provided
    if labels is None:
        n_components = (
            components.shape[0] if isinstance(components, np.ndarray) else components.shape[0]
        )
        labels = [f"{method}_{i:03d}" for i in range(n_components)]

    # Save spatial maps
    maps_file = output_dir / f"{output_prefix.name}_maps{nii_ext}"
    save_component_maps(components, mask_file, maps_file, labels=labels)

    # Save timeseries (.1D format only)
    ts_1d_file = output_dir / f"{output_prefix.name}_timeseries.1D"
    save_timeseries(timeseries, ts_1d_file, labels=labels)

    return {
        "maps": maps_file,
        "timeseries_1D": ts_1d_file,
    }
