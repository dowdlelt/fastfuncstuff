"""
HDF5 data caching for fast loading of fMRI data.

Provides ~10x speedup by caching preprocessed (scaled) data in HDF5 format
instead of loading multiple compressed NIfTI files on each run.
"""

import hashlib
import h5py
import numpy as np
import torch
from pathlib import Path
from typing import List, Union, Tuple, Optional
import time


def _compute_file_hash(files: List[Union[str, Path]]) -> str:
    """Compute hash of file paths and modification times for cache validation."""
    hash_obj = hashlib.md5()
    for f in sorted(files):
        p = Path(f)
        # Hash filename and mtime
        hash_obj.update(str(p.absolute()).encode())
        if p.exists():
            hash_obj.update(str(p.stat().st_mtime).encode())
    return hash_obj.hexdigest()


def save_cache(
    cache_file: Union[str, Path],
    data: np.ndarray,
    input_files: List[Union[str, Path]],
    run_starts: Optional[List[int]] = None,
    affine: Optional[np.ndarray] = None,
    volume_shape: Optional[Tuple[int, ...]] = None,
    was_scaled: bool = False,
    original_mean: Optional[float] = None,
    nifti_header: Optional[object] = None,
):
    """
    Save preprocessed fMRI data to HDF5 cache.

    Parameters
    ----------
    cache_file : str or Path
        Output HDF5 file path
    data : ndarray, shape (n_voxels, n_timepoints)
        Preprocessed (possibly scaled) fMRI data
    input_files : list of str/Path
        Original input NIfTI file paths (for validation)
    run_starts : list of int, optional
        Starting timepoint indices for each run
    affine : ndarray (4, 4), optional
        Affine transformation matrix
    volume_shape : tuple, optional
        Original volume shape (x, y, z)
    was_scaled : bool
        Whether data was auto-scaled to mean=100
    original_mean : float, optional
        Mean of data before scaling
    nifti_header : nibabel header object, optional
        Full NIfTI header to preserve all metadata (pixdim, units, orientation, etc.)
    """
    cache_path = Path(cache_file)

    print(f"\n💾 Creating HDF5 cache: {cache_path.name}")

    start_time = time.time()

    with h5py.File(cache_path, 'w') as f:
        # Store data with compression
        f.create_dataset(
            'data',
            data=data,
            compression='gzip',
            compression_opts=4,  # Balance speed vs compression
            chunks=(min(1000, data.shape[0]), data.shape[1])  # Chunk by voxels
        )

        # Store metadata
        meta = f.create_group('metadata')
        meta.attrs['file_hash'] = _compute_file_hash(input_files)
        meta.attrs['n_voxels'] = data.shape[0]
        meta.attrs['n_timepoints'] = data.shape[1]
        meta.attrs['was_scaled'] = was_scaled

        if original_mean is not None:
            meta.attrs['original_mean'] = original_mean

        # Store input file list
        meta.create_dataset('input_files', data=np.array([str(Path(f).absolute()) for f in input_files], dtype=h5py.string_dtype()))

        if run_starts is not None:
            meta.create_dataset('run_starts', data=np.array(run_starts, dtype=np.int32))

        if affine is not None:
            meta.create_dataset('affine', data=affine)

        if volume_shape is not None:
            meta.attrs['volume_shape'] = volume_shape

    # Save NIfTI header as separate pickle file (simpler and cleaner!)
    if nifti_header is not None:
        import pickle
        header_pickle_path = Path(str(cache_path) + '.header.pkl')
        with open(header_pickle_path, 'wb') as f:
            pickle.dump(nifti_header, f)

    elapsed = time.time() - start_time
    size_mb = cache_path.stat().st_size / (1024 ** 2)

    print(f"   Saved: {size_mb:.1f} MB in {elapsed:.1f}s")
    if was_scaled:
        print(f"   ⚠️  Data was auto-scaled: {original_mean:.1f} → 100.0")


def load_cache(
    cache_file: Union[str, Path],
    input_files: Optional[List[Union[str, Path]]] = None,
    validate: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    Load preprocessed fMRI data from HDF5 cache.

    Parameters
    ----------
    cache_file : str or Path
        HDF5 cache file path
    input_files : list of str/Path, optional
        Expected input files (for validation)
    validate : bool
        If True, validate cache matches input files

    Returns
    -------
    data : ndarray, shape (n_voxels, n_timepoints)
        Cached fMRI data
    metadata : dict
        Cache metadata (run_starts, affine, volume_shape, etc.)

    Raises
    ------
    ValueError
        If cache is invalid or doesn't match input files
    """
    cache_path = Path(cache_file)

    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    print(f"\n📦 Loading from HDF5 cache: {cache_path.name}")
    start_time = time.time()

    with h5py.File(cache_path, 'r') as f:
        # Validate cache if requested
        if validate and input_files is not None:
            expected_hash = _compute_file_hash(input_files)
            cached_hash = f['metadata'].attrs.get('file_hash', '')

            if expected_hash != cached_hash:
                raise ValueError(
                    f"Cache is stale! Input files have changed.\n"
                    f"Delete {cache_path} or use different cache file."
                )

        # Load data (this is fast due to HDF5's efficient chunking)
        data = f['data'][:]

        # Load metadata
        meta = f['metadata']
        metadata = {
            'n_voxels': meta.attrs['n_voxels'],
            'n_timepoints': meta.attrs['n_timepoints'],
            'was_scaled': meta.attrs.get('was_scaled', False),
            'original_mean': meta.attrs.get('original_mean', None),
        }

        # Load optional metadata
        if 'run_starts' in meta:
            metadata['run_starts'] = meta['run_starts'][:]

        if 'affine' in meta:
            metadata['affine'] = meta['affine'][:]

        if 'volume_shape' in meta.attrs:
            metadata['volume_shape'] = tuple(meta.attrs['volume_shape'])

        if 'input_files' in meta:
            metadata['input_files'] = [s.decode() if isinstance(s, bytes) else s
                                      for s in meta['input_files'][:]]

    # Load NIfTI header from separate pickle file if it exists
    header_pickle_path = Path(str(cache_path) + '.header.pkl')
    if header_pickle_path.exists():
        import pickle
        with open(header_pickle_path, 'rb') as f:
            metadata['nifti_header'] = pickle.load(f)

    elapsed = time.time() - start_time

    print(f"   ✓ Loaded: {metadata['n_voxels']:,} voxels × {metadata['n_timepoints']:,} TPs in {elapsed:.1f}s")
    if metadata['was_scaled']:
        print(f"   ℹ️  Data was pre-scaled to mean=100")

    return data, metadata


def check_cache_valid(
    cache_file: Union[str, Path],
    input_files: List[Union[str, Path]],
) -> bool:
    """
    Check if cache file exists and is valid for given input files.

    Parameters
    ----------
    cache_file : str or Path
        HDF5 cache file path
    input_files : list of str/Path
        Input files to validate against

    Returns
    -------
    valid : bool
        True if cache exists and matches input files
    """
    cache_path = Path(cache_file)

    if not cache_path.exists():
        return False

    try:
        with h5py.File(cache_path, 'r') as f:
            expected_hash = _compute_file_hash(input_files)
            cached_hash = f['metadata'].attrs.get('file_hash', '')
            return expected_hash == cached_hash
    except (OSError, KeyError):
        return False
