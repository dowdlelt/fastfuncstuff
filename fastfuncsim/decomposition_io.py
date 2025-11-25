"""
I/O utilities for PCA and ICA decomposition results

Save and load PCA/ICA spatial maps and timeseries in NIfTI format,
compatible with AFNI and other neuroimaging tools.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import torch

from .afni_io import get_tr_from_file, replace_afni_extension


def save_component_maps(
    components: Union[np.ndarray, torch.Tensor],
    mask_file: Union[str, Path],
    output_file: Union[str, Path],
    labels: Optional[list] = None,
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
    mask_img = nib.load(str(mask_file))
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

    # Create NIfTI image
    output_img = nib.Nifti1Image(output_data, mask_img.affine, mask_img.header)

    # Add AFNI sub-brick labels if provided
    if labels is not None:
        if len(labels) != n_components:
            raise ValueError(f"Number of labels ({len(labels)}) != n_components ({n_components})")

        # AFNI-style sub-brick labels
        label_str = '~'.join(labels)
        output_img.header.extensions.append(
            nib.nifti1.Nifti1Extension(
                'afni',
                f'BRICK_LABS={label_str}'.encode('utf-8')
            )
        )

    # Save
    nib.save(output_img, str(output_file))


def load_component_maps(
    map_file: Union[str, Path],
    mask_file: Union[str, Path],
) -> Tuple[np.ndarray, list]:
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
    mask_img = nib.load(str(mask_file))
    mask_data = mask_img.get_fdata()
    mask_bool = mask_data > 0
    n_voxels = mask_bool.sum()

    # Load component maps
    map_img = nib.load(str(map_file))
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
            content = ext.get_content().decode('utf-8')
            if 'BRICK_LABS=' in content:
                label_str = content.split('BRICK_LABS=')[1].split('\x00')[0]
                labels = label_str.split('~')
                break

    if labels is None:
        labels = [f'Component_{i}' for i in range(n_components)]

    return components, labels


def save_timeseries(
    timeseries: Union[np.ndarray, torch.Tensor],
    output_file: Union[str, Path],
    tr: Optional[float] = None,
    reference_file: Optional[Union[str, Path]] = None,
    labels: Optional[list] = None,
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

    if output_path.suffix == '.1D' or output_path.name.endswith('.1D'):
        # Save as AFNI .1D format
        with open(output_path, 'w') as f:
            # Write header with column labels if provided
            if labels is not None:
                if len(labels) != n_components:
                    raise ValueError(
                        f"Number of labels ({len(labels)}) != n_components ({n_components})"
                    )
                f.write('# ' + ' '.join(labels) + '\n')

            # Write data
            np.savetxt(f, timeseries, fmt='%.6f')

    elif output_path.suffix == '.gz' or output_path.suffixes == ['.nii', '.gz']:
        # Save as NIfTI (1 x 1 x n_timepoints x n_components)

        # Get TR
        if tr is None:
            if reference_file is None:
                raise ValueError("Must provide either 'tr' or 'reference_file'")
            tr = get_tr_from_file(str(reference_file))

        # Create 4D volume (1 x 1 x n_timepoints x n_components)
        data_4d = timeseries.T.reshape(1, 1, n_timepoints, n_components)

        # Create affine (identity with TR in pixdim)
        affine = np.eye(4)

        # Create header with TR
        header = nib.Nifti1Header()
        header['pixdim'][4] = tr

        # Create and save image
        img = nib.Nifti1Image(data_4d, affine, header)
        nib.save(img, str(output_path))

    else:
        raise ValueError(
            f"Unsupported file extension: {output_path.suffix}. "
            f"Use .1D or .nii.gz"
        )


def load_timeseries(
    timeseries_file: Union[str, Path],
) -> Tuple[np.ndarray, Optional[list]]:
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

    if path.suffix == '.1D' or path.name.endswith('.1D'):
        # Load AFNI .1D format
        labels = None

        # Try to read labels from header
        with open(path, 'r') as f:
            first_line = f.readline()
            if first_line.startswith('#'):
                labels = first_line.strip('# \n').split()

        # Load data
        timeseries = np.loadtxt(path)

        # Ensure 2D
        if timeseries.ndim == 1:
            timeseries = timeseries.reshape(-1, 1)

        return timeseries, labels

    elif path.suffix == '.gz' or path.suffixes == ['.nii', '.gz']:
        # Load NIfTI
        img = nib.load(str(path))
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
    components: Union[np.ndarray, torch.Tensor],
    timeseries: Union[np.ndarray, torch.Tensor],
    mask_file: Union[str, Path],
    output_prefix: Union[str, Path],
    tr: Optional[float] = None,
    reference_file: Optional[Union[str, Path]] = None,
    labels: Optional[list] = None,
    method: str = 'ICA',
) -> Dict[str, Path]:
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
        - 'timeseries_nii': path to timeseries (NIfTI)

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
        n_components = components.shape[0] if isinstance(components, np.ndarray) else components.shape[0]
        labels = [f'{method}_{i:03d}' for i in range(n_components)]

    # Save spatial maps
    maps_file = output_dir / f"{output_prefix.name}_maps.nii.gz"
    save_component_maps(components, mask_file, maps_file, labels=labels)

    # Save timeseries (both formats)
    ts_1d_file = output_dir / f"{output_prefix.name}_timeseries.1D"
    save_timeseries(timeseries, ts_1d_file, labels=labels)

    ts_nii_file = output_dir / f"{output_prefix.name}_timeseries.nii.gz"
    save_timeseries(timeseries, ts_nii_file, tr=tr, reference_file=reference_file)

    return {
        'maps': maps_file,
        'timeseries_1D': ts_1d_file,
        'timeseries_nii': ts_nii_file,
    }
