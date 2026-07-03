"""
AFNI file I/O utilities for reading and processing AFNI design matrices

This module provides utilities for:
- Reading AFNI onset timing files
- Reading AFNI design matrices (X.xmat.1D format)
- Working with design matrix metadata (ColumnGroups, RunStart, GoodList)
- Selecting regressors by type (stimulus, baseline, polynomial)
- Handling censored timepoints

Key Functions
-------------
read_afni_design_matrix:
    Read AFNI design matrix with full metadata parsing

get_regressor_groups:
    Organize regressors by ColumnGroups (polort, baseline, stimuli)

select_regressors_by_group:
    Extract specific regressor types from design matrix

get_censored_mask:
    Get boolean mask of censored timepoints from GoodList

select_uncensored_timepoints:
    Filter design matrix and data to uncensored timepoints

Examples
--------
# Read design matrix and inspect metadata
>>> design = read_afni_design_matrix('X.xmat.1D')
>>> print(f"TR: {design['tr']}s")
>>> print(f"Runs: {len(design['run_starts'])} runs")
>>> print(f"Stimuli: {design['stim_labels']}")

# Select only stimulus regressors
>>> groups = get_regressor_groups(design)
>>> X_stim = design['matrix'][:, groups['all_stimuli']]

# Handle censored data
>>> censored = get_censored_mask(design)
>>> X_clean, Y_clean = select_uncensored_timepoints(design, fmri_data)
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

try:
    import nibabel as nib
except ImportError as exc:  # pragma: no cover - nibabel is required for AFNI interoperability
    raise ImportError(
        "nibabel is required for AFNI mask and imaging utilities. Install it with `pip install nibabel`."
    ) from exc

from fastfuncstuff.utils import get_device, to_tensor


def parse_subbrick_selector(path: str | Path) -> tuple[str, list[int] | None]:
    """Parse AFNI-style sub-brick selectors from a file path.

    Supports the following selector syntax appended to a file path:

    - ``file.nii.gz[0]``            — single volume
    - ``file.nii.gz[1,3,5]``        — specific volumes
    - ``file.nii.gz[0..10]``        — range (inclusive)
    - ``file.nii.gz[0..$]``         — range to last volume (resolved later)
    - ``file.nii.gz[0..$(2)]``      — every 2nd volume (step)
    - ``file.nii.gz[0..10(3)]``     — range with step

    Quoting with single quotes around the selector (shell-style) is stripped
    automatically: ``file.nii.gz'[0..5]'`` works the same as ``file.nii.gz[0..5]``.

    Parameters
    ----------
    path : str or Path
        File path, optionally with a ``[selector]`` suffix.

    Returns
    -------
    clean_path : str
        The file path with the selector removed.
    indices : list[int] or None
        Resolved volume indices, or *None* if no selector was present.
        A ``$`` end-point is stored as -1 and resolved at load time.
    """
    path = str(path).strip().rstrip("'")
    # Find the bracket selector — scan from the right to avoid matching
    # brackets that might be in directory names.
    bracket_start = path.rfind("[")
    if bracket_start == -1:
        return path, None

    bracket_end = path.rfind("]")
    if bracket_end == -1 or bracket_end < bracket_start:
        return path, None

    # Strip optional leading quote before '['
    clean_path = path[:bracket_start].rstrip("'")
    selector = path[bracket_start + 1 : bracket_end]

    return clean_path, _parse_selector(selector)


def _parse_selector(selector: str) -> list[int]:
    """Parse the content inside brackets into a list of volume indices.

    ``$`` is stored as -1 to be resolved once the number of volumes is known.
    """
    selector = selector.strip()

    # Comma-separated list: 0,1,3,5
    if "," in selector:
        return [int(s.strip()) for s in selector.split(",")]

    # Range: start..end  or  start..end(step)  or  start..$(step)
    if ".." in selector:
        range_part, _, step_part = selector.partition("(")
        step = 1
        if step_part:
            step = int(step_part.rstrip(")"))

        start_str, end_str = range_part.split("..", 1)
        start = int(start_str.strip())

        end_str = end_str.strip()
        if end_str == "$" or end_str == "":
            # -1 sentinel → resolve at load time
            return _range_with_sentinel(start, -1, step)

        end = int(end_str)
        return list(range(start, end + 1, step))  # inclusive end, like AFNI

    # Single index: 0
    return [int(selector)]


def _range_with_sentinel(start: int, end: int, step: int) -> list[int]:
    """Build a range list; if *end* is -1, return a marker for deferred resolution."""
    if end == -1:
        # Store as (start, sentinel, step) encoded in a list with a negative marker.
        # Convention: [-1, start, step] — the -1 first element flags deferred.
        return [-1, start, step]
    return list(range(start, end + 1, step))


def _resolve_indices(indices: list[int], n_volumes: int) -> list[int]:
    """Resolve deferred ``$`` selectors once the volume count is known."""
    if indices and indices[0] == -1 and len(indices) == 3:
        _, start, step = indices
        return list(range(start, n_volumes, step))
    # Validate explicit indices
    for i in indices:
        if i < 0:
            raise ValueError(f"Negative volume index {i} is not valid")
        if i >= n_volumes:
            raise ValueError(f"Volume index {i} out of range for image with {n_volumes} volumes")
    return indices


def load_nifti(filepath: str | Path) -> nib.Nifti1Image:
    """
    Load NIfTI files with support for .nii, .nii.gz, and .nii.zst formats.

    Supports AFNI-style sub-brick selectors appended to the path::

        load_nifti("func.nii.gz[0]")          # first volume
        load_nifti("func.nii.gz[1..$]")       # 2nd through last
        load_nifti("func.nii.gz[0..250]")     # first 251 volumes
        load_nifti("func.nii.gz[0..$(2)]")    # every 2nd volume
        load_nifti("func.nii.gz[1,2,3,5]")    # specific volumes

    Parameters
    ----------
    filepath : str or Path
        Path to NIfTI file, optionally with ``[selector]`` suffix. Supports:
        - .nii (uncompressed)
        - .nii.gz (gzip compressed)
        - .nii.zst (zstandard compressed)

    Returns
    -------
    nib.Nifti1Image
        Loaded NIfTI image (sub-selected if a selector was given)

    Examples
    --------
    >>> img = load_nifti('func.nii.gz')
    >>> img = load_nifti('func.nii.gz[0]')
    >>> data = img.get_fdata()

    Notes
    -----
    For .nii.zst files, requires zstd to be installed and available in PATH.
    The file is decompressed to a temporary file before loading with nibabel.
    """
    # Parse sub-brick selector before resolving the path
    clean_path, indices = parse_subbrick_selector(str(filepath))
    filepath = Path(clean_path)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Handle .nii.zst files by decompressing through zstd
    if str(filepath).endswith(".nii.zst"):
        # Check if zstd is available
        try:
            subprocess.run(["zstd", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as err:
            raise RuntimeError(
                "zstd is not installed or not available in PATH. "
                "Install zstd to load .nii.zst files."
            ) from err

        # Decompress to temporary file
        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Decompress: zstd -dc input.nii.zst > output.nii
            with open(tmp_path, "wb") as out_file:
                subprocess.run(
                    ["zstd", "-dc", str(filepath)],
                    stdout=out_file,
                    check=True,
                    stderr=subprocess.PIPE,
                )

            # Load the decompressed file
            img = nib.load(tmp_path)

            # Load data into memory and create new image to avoid lazy loading issues
            # This ensures the temp file can be safely deleted
            data = img.get_fdata()
            affine = img.affine
            header = img.header.copy()

            # Create new image from in-memory data
            img_inmem = nib.Nifti1Image(data, affine, header)

        finally:
            # Clean up temporary file
            Path(tmp_path).unlink(missing_ok=True)

        img_out = img_inmem

    else:
        # Standard nibabel loading for .nii and .nii.gz
        img_out = nib.load(str(filepath))

    # Apply sub-brick selection if requested
    if indices is not None:
        data = np.asarray(img_out.dataobj)
        if data.ndim < 4:
            raise ValueError(f"Sub-brick selector requires a 4D image, got {data.ndim}D")
        n_volumes = data.shape[3]
        resolved = _resolve_indices(indices, n_volumes)
        data = data[:, :, :, resolved]
        header = img_out.header.copy()
        # Update dim[4] for the new volume count
        if data.ndim == 4:
            header["dim"][4] = data.shape[3]
        elif data.ndim == 3:
            header["dim"][4] = 1
        img_out = nib.Nifti1Image(data, img_out.affine, header)

    return img_out


def read_afni_onset_file(filepath: str | Path) -> list[np.ndarray]:
    """
    Read AFNI onset timing file

    AFNI onset files contain one row per run, with space-separated onset times in seconds.

    Parameters
    ----------
    filepath : str or Path
        Path to AFNI onset timing file

    Returns
    -------
    onsets_per_run : list of np.ndarray
        List of onset time arrays, one per run
        Each array contains onset times in seconds for that run

    Examples
    --------
    >>> onsets = read_afni_onset_file('onsets_condition1.txt')
    >>> print(f"Run 1 onsets: {onsets[0]}")
    >>> print(f"Run 2 onsets: {onsets[1]}")
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Onset file not found: {filepath}")

    with open(filepath) as f:
        lines = f.readlines()

    onsets_per_run = []
    for line in lines:
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue

        # Parse space-separated onset times
        onsets = np.array([float(x) for x in line.split()])
        onsets_per_run.append(onsets)

    return onsets_per_run


def read_afni_onset_files(filepaths: list[str | Path]) -> list[list[np.ndarray]]:
    """
    Read multiple AFNI onset timing files (one per condition)

    Parameters
    ----------
    filepaths : list of str or Path
        Paths to AFNI onset timing files, one per condition

    Returns
    -------
    onsets_per_condition : list of list of np.ndarray
        onsets_per_condition[cond][run] = onset times for condition cond, run run

    Examples
    --------
    >>> files = ['onsets_condition1.txt', 'onsets_condition2.txt']
    >>> onsets = read_afni_onset_files(files)
    >>> print(f"Condition 1, Run 1: {onsets[0][0]}")
    """
    return [read_afni_onset_file(fp) for fp in filepaths]


def onsets_to_tr_matrix(
    onsets_per_condition: list[list[np.ndarray]],
    n_timepoints: int,
    tr: float,
) -> np.ndarray:
    """
    Convert AFNI onset times to binary onset matrix at TR resolution.

    This is a simpler version of onsets_to_binary_matrix for use with
    FIR/TENT models that work at TR resolution (no convolution).

    Parameters
    ----------
    onsets_per_condition : list of list of np.ndarray
        onsets_per_condition[cond][run] = onset times in seconds
        For single run: [[cond1_onsets, cond2_onsets, ...]]
    n_timepoints : int
        Total number of timepoints (TRs)
    tr : float
        TR in seconds

    Returns
    -------
    onset_matrix : np.ndarray
        Binary onset matrix at TR resolution
        Shape: (n_timepoints, n_conditions)
        Values are 1.0 at onset TRs, 0.0 elsewhere

    Examples
    --------
    >>> onsets = [[np.array([2.0, 6.0, 10.0]), np.array([4.0, 8.0])]]
    >>> matrix = onsets_to_tr_matrix(onsets, n_timepoints=20, tr=1.0)
    >>> matrix.shape
    (20, 2)
    >>> matrix[2, 0]  # Onset at 2.0s for condition 0
    1.0
    """
    n_conditions = len(onsets_per_condition)
    n_runs = len(onsets_per_condition[0])

    # Initialize matrix
    onset_matrix = np.zeros((n_timepoints, n_conditions), dtype=np.float32)

    # Convert onset times to TR indices
    for cond_idx in range(n_conditions):
        for run_idx in range(n_runs):
            onset_times = onsets_per_condition[cond_idx][run_idx]

            for onset_time in onset_times:
                # Convert time to TR index (round to nearest TR)
                tr_idx = int(np.round(onset_time / tr))

                if 0 <= tr_idx < n_timepoints:
                    onset_matrix[tr_idx, cond_idx] = 1.0

    return onset_matrix


def onsets_to_binary_matrix(
    onsets_per_condition: list[list[np.ndarray]],
    n_timepoints: int,
    tr: float,
    run_starts: list[int] | None = None,
    device: torch.device | None = None,
    microtime_dt: float = 0.1,
) -> torch.Tensor:
    """
    Convert AFNI onset times to binary onset matrix at microtime resolution.

    Parameters
    ----------
    onsets_per_condition : list of list of np.ndarray
        onsets_per_condition[cond][run] = onset times in seconds
    n_timepoints : int
        Total number of timepoints across all runs (at TR resolution)
    tr : float
        TR in seconds
    run_starts : list of int, optional
        Starting timepoint index for each run (from AFNI RunStart parameter)
        If None, assumes equal-length runs
        Example: [0, 300, 600, 900] for 4 runs of 300 TRs each
    device : torch.device, optional
        Device for output tensor
    microtime_dt : float, default=0.1
        Microtime resolution in seconds. Default 0.1s is the standard throughout
        the pipeline. Output matrix has shape (n_microtime_points, n_conditions)
        where n_microtime_points = n_timepoints * (tr / microtime_dt).

    Returns
    -------
    onset_matrix : torch.Tensor
        Binary onset matrix at microtime_dt resolution.
        Shape: (n_microtime_points, n_conditions)

    Examples
    --------
    >>> files = ['onsets_cond1.txt', 'onsets_cond2.txt']
    >>> onsets = read_afni_onset_files(files)
    >>> # Standard 0.1s microtime resolution
    >>> onset_mat = onsets_to_binary_matrix(onsets, n_timepoints=200, tr=2.0)
    >>> onset_mat.shape  # (4000, n_conditions) for TR=2s, dt=0.1s
    """
    if device is None:
        device = get_device()

    if microtime_dt <= 0:
        raise ValueError(f"microtime_dt must be > 0, got {microtime_dt}")

    n_conditions = len(onsets_per_condition)
    n_runs = len(onsets_per_condition[0])

    # Validate all conditions have same number of runs
    for cond_idx, cond_onsets in enumerate(onsets_per_condition):
        if len(cond_onsets) != n_runs:
            raise ValueError(f"Condition {cond_idx} has {len(cond_onsets)} runs, expected {n_runs}")

    # Calculate bins per TR and total microtime points
    bins_per_tr = int(round(tr / microtime_dt))
    n_microtime_points = n_timepoints * bins_per_tr

    # Create binary onset matrix at microtime resolution
    onset_matrix = np.zeros((n_microtime_points, n_conditions))

    # Determine run boundaries (scale to microtime resolution)
    if run_starts is not None:
        # Use provided run starts (in TR units, scale to microtime)
        if len(run_starts) != n_runs:
            raise ValueError(
                f"run_starts has {len(run_starts)} entries, but data has {n_runs} runs"
            )
        timepoint_offsets = [rs * bins_per_tr for rs in run_starts]
    else:
        # Assume equal-length runs
        microtime_per_run = n_microtime_points // n_runs
        timepoint_offsets = [run_idx * microtime_per_run for run_idx in range(n_runs)]

    for cond_idx, cond_onsets in enumerate(onsets_per_condition):
        for run_idx, run_onsets in enumerate(cond_onsets):
            # Convert onset times to microtime indices
            timepoint_offset = timepoint_offsets[run_idx]

            for onset_time in run_onsets:
                # Convert onset time to microtime bin (precise placement)
                microtime_bin = int(np.round(onset_time / microtime_dt))
                global_microtime = timepoint_offset + microtime_bin

                if 0 <= global_microtime < n_microtime_points:
                    onset_matrix[global_microtime, cond_idx] = 1

    return to_tensor(onset_matrix, device=device, dtype=torch.float32)


def parse_afni_matrix_notation(notation: str) -> np.ndarray:
    """
    Parse AFNI matrix notation (e.g., "1,48,20@0,2@0.5,26@0")

    AFNI uses compact notation like "N@value" meaning N repetitions of value.
    Format: "n_rows,n_cols,values" where values can be "N@value" or just "value"

    Parameters
    ----------
    notation : str
        AFNI matrix notation string

    Returns
    -------
    matrix : np.ndarray
        Parsed matrix

    Examples
    --------
    >>> mat = parse_afni_matrix_notation("1,48,20@0,2@0.5,26@0")
    >>> print(mat.shape)  # (1, 48)
    """
    parts = notation.split(",")
    n_rows = int(parts[0])
    n_cols = int(parts[1])

    # Parse values
    values = []
    for part in parts[2:]:
        if "@" in part:
            # Repeat notation: "N@value"
            count, value = part.split("@")
            values.extend([float(value)] * int(count))
        else:
            # Single value
            values.append(float(part))

    # Total expected value count is n_rows * n_cols (F-tests have r > 1).
    expected = n_rows * n_cols
    if len(values) != expected:
        raise ValueError(
            f"Expected {expected} values ({n_rows} rows × {n_cols} cols), got {len(values)}"
        )

    matrix = np.array(values).reshape(n_rows, n_cols)
    return matrix


def read_afni_design_matrix(filepath: str | Path) -> dict:
    """
    Read AFNI design matrix file (X.xmat.1D format)

    Parses both the header metadata and the data matrix from AFNI's 3dDeconvolve output.

    Parameters
    ----------
    filepath : str or Path
        Path to X.xmat.1D file

    Returns
    -------
    design_info : dict
        Dictionary containing:
        - 'matrix': np.ndarray (n_timepoints, n_regressors) - design matrix
        - 'tr': float - repetition time
        - 'n_timepoints': int - number of timepoints
        - 'n_regressors': int - number of regressors
        - 'column_labels': list of str - regressor names
        - 'column_groups': list of int - group assignments for regressors
        - 'run_starts': list of int - starting indices for each run (for ARMA modeling)
        - 'good_list': list of int - uncensored TR indices (for REML with censoring)
        - 'stim_labels': list of str - stimulus/condition names
        - 'stim_bots': list of int - bottom indices for stimulus columns
        - 'stim_tops': list of int - top indices for stimulus columns
        - 'n_glt': int - number of GLT contrasts (0 if none)
        - 'glt_labels': list of str - contrast names
        - 'glt_matrices': list of np.ndarray - contrast matrices
        - 'basis_info': dict - HRF basis function information per stimulus
        - 'command_line': str - original 3dDeconvolve command

    Examples
    --------
    >>> design = read_afni_design_matrix('X.xmat.1D')
    >>> print(f"Design shape: {design['matrix'].shape}")
    >>> print(f"TR: {design['tr']}s")
    >>> print(f"Stimulus labels: {design['stim_labels']}")
    >>> print(f"Contrast labels: {design['glt_labels']}")
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Design matrix file not found: {filepath}")

    # Initialize metadata
    metadata = {
        "column_labels": None,
        "column_groups": None,
        "tr": None,
        "run_starts": None,
        "good_list": None,  # List of uncensored TR indices (for REML with censoring)
        "stim_labels": None,
        "stim_bots": None,
        "stim_tops": None,
        "n_glt": 0,  # Number of GLT contrasts (0 if none)
        "glt_labels": None,
        "glt_matrices": [],
        "basis_info": {},
        "command_line": None,
        "n_timepoints": None,
        "n_regressors": None,
    }

    data_lines = []

    with open(filepath) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # Parse header lines
            if line.startswith("#"):
                # Remove leading '# ' or '#'
                line = line.lstrip("#").strip()

                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"')

                    # Parse specific fields
                    if key == "ColumnLabels":
                        metadata["column_labels"] = [x.strip() for x in value.split(";")]

                    elif key == "ColumnGroups":
                        # Parse format like "20@-1,1..4,24@0"
                        groups = []
                        for part in value.split(","):
                            if "@" in part:
                                count, val = part.split("@")
                                groups.extend([int(val)] * int(count))
                            elif ".." in part:
                                start, end = map(int, part.split(".."))
                                groups.extend(range(start, end + 1))
                            else:
                                groups.append(int(part))
                        metadata["column_groups"] = groups

                    elif key == "RowTR":
                        metadata["tr"] = float(value)

                    elif key == "NRowFull":
                        metadata["n_timepoints"] = int(value)

                    elif key == "RunStart":
                        metadata["run_starts"] = [int(x) for x in value.split(",")]

                    elif key == "GoodList":
                        # Parse formats: "0..719", "0..100,102..719", etc.
                        # This lists TR indices that were NOT censored
                        indices = []
                        for part in value.split(","):
                            if ".." in part:
                                start, end = map(int, part.split(".."))
                                indices.extend(range(start, end + 1))
                            else:
                                indices.append(int(part))
                        metadata["good_list"] = indices

                    elif key == "StimLabels":
                        metadata["stim_labels"] = [x.strip() for x in value.split(";")]

                    elif key == "StimBots":
                        # Parse formats: "20..23", "20,21,22", or mixed "20..23,25,26"
                        indices = []
                        for part in value.split(","):
                            if ".." in part:
                                start, end = map(int, part.split(".."))
                                indices.extend(range(start, end + 1))
                            else:
                                indices.append(int(part))
                        metadata["stim_bots"] = indices

                    elif key == "StimTops":
                        # Parse formats: "20..23", "20,21,22", or mixed "20..23,25,26"
                        indices = []
                        for part in value.split(","):
                            if ".." in part:
                                start, end = map(int, part.split(".."))
                                indices.extend(range(start, end + 1))
                            else:
                                indices.append(int(part))
                        metadata["stim_tops"] = indices

                    elif key == "Nglt":
                        # Number of GLT contrasts - used to determine if var_betas needed
                        metadata["n_glt"] = int(value)

                    elif key == "GltLabels":
                        metadata["glt_labels"] = [x.strip() for x in value.split(";")]

                    elif key.startswith("GltMatrix_"):
                        # Parse contrast matrix
                        glt_matrix = parse_afni_matrix_notation(value)
                        metadata["glt_matrices"].append(glt_matrix)

                    elif key.startswith("BasisName_"):
                        stim_idx = key.split("_")[1]
                        if stim_idx not in metadata["basis_info"]:
                            metadata["basis_info"][stim_idx] = {}
                        metadata["basis_info"][stim_idx]["name"] = value

                    elif key.startswith("BasisFormula_"):
                        stim_idx = key.split("_")[1]
                        if stim_idx not in metadata["basis_info"]:
                            metadata["basis_info"][stim_idx] = {}
                        metadata["basis_info"][stim_idx]["formula"] = value

                    elif key.startswith("BasisColumns_"):
                        stim_idx = key.split("_")[1]
                        if stim_idx not in metadata["basis_info"]:
                            metadata["basis_info"][stim_idx] = {}
                        # Parse "20:20" format
                        start, end = map(int, value.split(":"))
                        metadata["basis_info"][stim_idx]["columns"] = (start, end)

                    elif key == "CommandLine":
                        metadata["command_line"] = value

                    elif key == "ni_dimen":
                        metadata["n_timepoints"] = int(value)

                    elif key == "ni_type":
                        # Parse "48*double" format
                        n_cols = int(value.split("*")[0])
                        metadata["n_regressors"] = n_cols

            else:
                # Data line
                data_lines.append(line)

    # Parse data matrix
    data_matrix = []
    for line in data_lines:
        values = [float(x) for x in line.split()]
        data_matrix.append(values)

    data_matrix = np.array(data_matrix)

    # Add matrix to metadata
    metadata["matrix"] = data_matrix

    # Infer n_regressors if not set
    if metadata["n_regressors"] is None:
        metadata["n_regressors"] = data_matrix.shape[1]

    # Infer n_timepoints if not set
    if metadata["n_timepoints"] is None:
        metadata["n_timepoints"] = data_matrix.shape[0]

    return metadata


def extract_stimulus_columns(
    design_info: dict,
    stim_indices: list[int] | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Extract stimulus/task regressor columns from AFNI design matrix

    Parameters
    ----------
    design_info : dict
        Output from read_afni_design_matrix()
    stim_indices : list of int, optional
        Indices of stimulus columns to extract (0-indexed into stim_labels)
        If None, extracts all stimulus columns
    device : torch.device, optional
        Device for output tensor

    Returns
    -------
    stim_design : torch.Tensor
        Stimulus design matrix (n_timepoints, n_stim)

    Examples
    --------
    >>> design = read_afni_design_matrix('X.xmat.1D')
    >>> stim_design = extract_stimulus_columns(design)
    >>> print(f"Stimulus design shape: {stim_design.shape}")
    """
    if device is None:
        device = get_device()

    matrix = design_info["matrix"]
    stim_bots = design_info["stim_bots"]
    stim_tops = design_info["stim_tops"]

    if stim_indices is None:
        # Extract all stimulus columns
        stim_cols = []
        for bot, top in zip(stim_bots, stim_tops, strict=False):
            stim_cols.extend(range(bot, top + 1))
    else:
        # Extract specified stimulus columns
        stim_cols = []
        for idx in stim_indices:
            bot = stim_bots[idx]
            top = stim_tops[idx]
            stim_cols.extend(range(bot, top + 1))

    stim_design = matrix[:, stim_cols]

    return to_tensor(stim_design, device=device, dtype=torch.float32)


def extract_nuisance_columns(design_info: dict, device: torch.device | None = None) -> torch.Tensor:
    """
    Extract nuisance regressor columns (polynomials + motion) from AFNI design matrix

    Parameters
    ----------
    design_info : dict
        Output from read_afni_design_matrix()
    device : torch.device, optional
        Device for output tensor

    Returns
    -------
    nuisance_design : torch.Tensor
        Nuisance design matrix (n_timepoints, n_nuisance)

    Examples
    --------
    >>> design = read_afni_design_matrix('X.xmat.1D')
    >>> nuisance = extract_nuisance_columns(design)
    >>> print(f"Nuisance design shape: {nuisance.shape}")
    """
    if device is None:
        device = get_device()

    matrix = design_info["matrix"]
    stim_bots = design_info["stim_bots"]
    stim_tops = design_info["stim_tops"]

    # Find nuisance columns (everything except stimulus columns)
    stim_cols = set()
    for bot, top in zip(stim_bots, stim_tops, strict=False):
        stim_cols.update(range(bot, top + 1))

    nuisance_cols = [i for i in range(matrix.shape[1]) if i not in stim_cols]

    nuisance_design = matrix[:, nuisance_cols]

    return to_tensor(nuisance_design, device=device, dtype=torch.float32)


def load_and_concatenate_runs(
    run_files: list[str | Path],
    device: torch.device | None = None,
    keep_on_cpu: bool = False,
    mask_flat: np.ndarray | None = None,
) -> tuple[torch.Tensor, list[int]]:
    """
    Load multiple fMRI run files and concatenate them (memory-efficient)

    Parameters
    ----------
    run_files : list of str or Path
        Paths to neuroimaging files, one per run
        Supports: NIfTI (.nii, .nii.gz, .nii.zst) and AFNI (.HEAD, .BRIK, .BRIK.gz)
        For AFNI files, provide either .HEAD or .BRIK path
        Note: .nii.zst requires zstd to be installed
    device : torch.device, optional
        Device for output tensor (ignored if keep_on_cpu=True)
    keep_on_cpu : bool, default=False
        If True, load data to CPU regardless of device.
        Use this for large datasets where GPU memory is limited,
        and you'll process data in chunks.

    Returns
    -------
    concatenated_data : torch.Tensor
        Concatenated data (n_voxels, total_timepoints)
        On CPU if keep_on_cpu=True, else on specified device
    run_starts : list of int
        Starting timepoint index for each run

    Examples
    --------
    >>> # Standard GPU loading (for smaller datasets)
    >>> run_files = ['run01.nii.gz', 'run02.nii.gz', 'run03.nii.gz']
    >>> data, run_starts = load_and_concatenate_runs(run_files)
    >>> print(f"Total timepoints: {data.shape[1]}")

    >>> # CPU loading for large datasets
    >>> data_cpu, run_starts = load_and_concatenate_runs(run_files, keep_on_cpu=True)
    >>> print(f"Data device: {data_cpu.device}")  # cpu

    Notes
    -----
    Memory-efficient implementation that converts each run to torch immediately
    and concatenates to avoid holding multiple copies in memory.

    For large datasets with GPU acceleration, use keep_on_cpu=True and process
    voxel chunks on GPU in your analysis code.
    """
    import gc

    if device is None and not keep_on_cpu:
        device = get_device()

    # Force CPU if requested
    if keep_on_cpu:
        device = torch.device("cpu")

    run_starts = [0]
    current_start = 0
    torch_runs = []

    # Progress bar for loading runs
    run_iterator = enumerate(run_files)
    if len(run_files) > 1:
        run_iterator = enumerate(tqdm(run_files, desc="Loading fMRI runs", unit="run"))

    for i, run_file in run_iterator:
        # Load run
        img = load_nifti(run_file)
        data_np = img.get_fdata(dtype=np.float32)

        # Reshape to (n_voxels, n_timepoints)
        if data_np.ndim == 4:
            n_voxels = data_np.shape[0] * data_np.shape[1] * data_np.shape[2]
            n_timepoints = data_np.shape[3]
            data_np = data_np.reshape(n_voxels, n_timepoints)
        elif data_np.ndim != 2:
            raise ValueError(f"Data must be 2D or 4D, got shape {data_np.shape}")

        # Apply mask before moving to device (saves GPU memory)
        if mask_flat is not None:
            if mask_flat.shape[0] != data_np.shape[0]:
                raise ValueError(
                    f"mask_flat length {mask_flat.shape[0]} does not match "
                    f"n_voxels {data_np.shape[0]} for run {i}"
                )
            data_np = data_np[mask_flat, :]

        # Convert to torch immediately and move to device
        # This avoids keeping numpy copy around
        data_torch = torch.from_numpy(data_np.astype(np.float32, copy=False)).to(device)

        # Delete numpy array immediately to free memory
        del data_np, img
        gc.collect()

        # All runs must share the voxel dimension to concatenate along time.
        # A mismatch means the runs are on different spatial grids (e.g. resampled
        # to a template without a shared -master), which torch.cat only reports as
        # an opaque size error -- name the offending run and grids instead.
        if torch_runs and data_torch.shape[0] != torch_runs[0].shape[0]:
            raise ValueError(
                f"Run {i} ({run_file}) has {data_torch.shape[0]} voxels but run 0 "
                f"({run_files[0]}) has {torch_runs[0].shape[0]}. All runs must be on "
                f"the same spatial grid; resample every run to a shared -master."
            )

        torch_runs.append(data_torch)

        # Track run start for next run
        current_start += data_torch.shape[1]
        run_starts.append(current_start)

    # Remove last entry (it's the end, not a start)
    run_starts = run_starts[:-1]

    # Concatenate on device (no numpy intermediate)
    concatenated = torch.cat(torch_runs, dim=1)

    # Clean up
    del torch_runs
    gc.collect()

    return concatenated, run_starts


def load_afni_mask(
    mask_file: str | Path,
    threshold: float = 0.0,
) -> np.ndarray:
    """Load an AFNI-style mask and return a boolean volume.

    Parameters
    ----------
    mask_file : str or Path
        Path to the mask file (NIfTI .nii/.nii.gz/.nii.zst or AFNI BRIK/HEAD format).
    threshold : float, optional
        Values strictly greater than this threshold are treated as inside the mask.

    Returns
    -------
    mask : np.ndarray
        Boolean mask array with shape matching the NIfTI volume (nx, ny, nz).

    Notes
    -----
    Handles singleton dimensions automatically. For example, AFNI masks with
    shape (64, 64, 35, 1) are automatically squeezed to (64, 64, 35).
    """

    mask_path = Path(mask_file)
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    img = load_nifti(mask_path)
    data = img.get_fdata(dtype=np.float32)

    # Squeeze out singleton dimensions (common in AFNI masks)
    # e.g., (64, 64, 35, 1) → (64, 64, 35)
    data = np.squeeze(data)

    if data.ndim != 3:
        raise ValueError(
            f"Mask file must be 3D after squeezing singleton dimensions "
            f"(received shape {data.shape} after squeeze). "
            f"Ensure a volumetric mask is provided."
        )

    mask = data > float(threshold)
    if not np.any(mask):
        raise ValueError(
            f"Mask '{mask_path}' is empty after thresholding at {threshold}. Lower the threshold or verify the file."
        )

    return mask


def get_run_lengths(run_starts: list[int], n_timepoints: int) -> list[int]:
    """
    Calculate run lengths from run starts

    Parameters
    ----------
    run_starts : list of int
        Starting timepoint indices for each run
    n_timepoints : int
        Total number of timepoints

    Returns
    -------
    run_lengths : list of int
        Number of timepoints in each run

    Examples
    --------
    >>> run_starts = [0, 300, 600, 900]
    >>> run_lengths = get_run_lengths(run_starts, 1200)
    >>> print(run_lengths)  # [300, 300, 300, 300]
    """
    run_lengths = []
    for i in range(len(run_starts)):
        if i < len(run_starts) - 1:
            length = run_starts[i + 1] - run_starts[i]
        else:
            length = n_timepoints - run_starts[i]
        run_lengths.append(length)

    return run_lengths


def get_contrast_matrix(
    design_info: dict, contrast_index: int, device: torch.device | None = None
) -> torch.Tensor:
    """
    Get a specific contrast matrix from AFNI design info

    Parameters
    ----------
    design_info : dict
        Output from read_afni_design_matrix()
    contrast_index : int
        Index of contrast to retrieve (0-indexed)
    device : torch.device, optional
        Device for output tensor

    Returns
    -------
    contrast : torch.Tensor
        Contrast matrix (n_contrasts, n_regressors)

    Examples
    --------
    >>> design = read_afni_design_matrix('X.xmat.1D')
    >>> contrast = get_contrast_matrix(design, 0)
    >>> print(f"Contrast '{design['glt_labels'][0]}': {contrast}")
    """
    if device is None:
        device = get_device()

    if contrast_index >= len(design_info["glt_matrices"]):
        raise ValueError(
            f"Contrast index {contrast_index} out of range "
            f"(only {len(design_info['glt_matrices'])} contrasts)"
        )

    contrast = design_info["glt_matrices"][contrast_index]

    return to_tensor(contrast, device=device, dtype=torch.float32)


def extract_design_metadata(
    design_info: dict,
) -> tuple[list[str], list[str], list[int]]:
    """
    Extract metadata from design_info dictionary with clear, unambiguous names.

    This helper consolidates the common pattern of extracting stimulus indices
    and labels from AFNI design matrices. Use this instead of duplicating the
    extraction logic across multiple files.

    Parameters
    ----------
    design_info : dict
        Dictionary returned by read_afni_design_matrix()

    Returns
    -------
    full_labels : list of str
        All column labels from the design matrix (length = n_regressors_full)
    stim_labels : list of str
        Labels for stimulus columns only (length = n_stim)
    stim_column_indices : list of int
        Column indices for stimulus regressors in the full design matrix

    Examples
    --------
    >>> design_info = read_afni_design_matrix('X.xmat.1D')
    >>> full_labels, stim_labels, stim_indices = extract_design_metadata(design_info)
    >>> print(f"Full design: {len(full_labels)} columns")
    >>> print(f"Stimulus only: {len(stim_labels)} columns")
    >>> print(f"Stimulus indices: {stim_indices[0]}..{stim_indices[-1]}")

    Notes
    -----
    This function implements the clean naming convention:
    - `full_labels`: ALL labels from design matrix
    - `stim_labels`: ONLY stimulus labels (filtered subset)
    - `stim_column_indices`: Indices to extract stim columns from full design
    """
    # Extract full labels (all columns)
    full_labels = design_info.get("column_labels", [])

    # Extract stimulus indices from StimBots/StimTops
    stim_bots = design_info.get("stim_bots", [])
    stim_tops = design_info.get("stim_tops", [])
    stim_column_indices = []

    if stim_bots and stim_tops:
        for bot, top in zip(stim_bots, stim_tops, strict=False):
            stim_column_indices.extend(range(bot, top + 1))

    # Extract stimulus labels (subset of full labels)
    if stim_column_indices and full_labels:
        stim_labels = [full_labels[i] for i in stim_column_indices]
    else:
        stim_labels = []

    return full_labels, stim_labels, stim_column_indices


def get_regressor_groups(design_info: dict) -> dict[str, list[int]]:
    """
    Get regressor indices organized by ColumnGroups

    AFNI ColumnGroups convention:
      -1 = polynomial drift regressors
       0 = motion and other baseline regressors (non-polort)
       1,2,3,... = stimuli of interest (experimental conditions)

    Parameters
    ----------
    design_info : dict
        Output from read_afni_design_matrix()

    Returns
    -------
    groups : dict
        Dictionary mapping group names to column indices:
        - 'polort': list of polynomial regressor indices
        - 'baseline': list of motion/baseline regressor indices (group 0)
        - 'stimulus_1', 'stimulus_2', ...: lists of stimulus regressor indices
        - 'all_stimuli': combined list of all stimulus indices
        - 'all_nuisance': combined list of all nuisance indices (polort + baseline)

    Examples
    --------
    >>> design = read_afni_design_matrix('X.xmat.1D')
    >>> groups = get_regressor_groups(design)
    >>> print(f"Polynomial regressors: {groups['polort']}")
    >>> print(f"Stimulus 1 regressors: {groups['stimulus_1']}")
    >>> print(f"All stimuli: {groups['all_stimuli']}")
    """
    column_groups = design_info.get("column_groups")
    if column_groups is None:
        raise ValueError("Design matrix does not have ColumnGroups information")

    groups = {
        "polort": [],
        "baseline": [],
        "all_stimuli": [],
        "all_nuisance": [],
    }

    # Organize columns by group
    for col_idx, group_id in enumerate(column_groups):
        if group_id == -1:
            groups["polort"].append(col_idx)
            groups["all_nuisance"].append(col_idx)
        elif group_id == 0:
            groups["baseline"].append(col_idx)
            groups["all_nuisance"].append(col_idx)
        elif group_id > 0:
            # Stimulus group
            stim_key = f"stimulus_{group_id}"
            if stim_key not in groups:
                groups[stim_key] = []
            groups[stim_key].append(col_idx)
            groups["all_stimuli"].append(col_idx)

    return groups


def select_regressors_by_group(
    design_info: dict,
    include_groups: list[str] | None = None,
    exclude_groups: list[str] | None = None,
) -> np.ndarray:
    """
    Select regressor columns by group type

    Parameters
    ----------
    design_info : dict
        Output from read_afni_design_matrix()
    include_groups : list of str, optional
        Groups to include. Options:
        - 'polort': polynomial drift
        - 'baseline': motion/baseline (group 0)
        - 'stimulus_1', 'stimulus_2', ...: specific stimuli
        - 'all_stimuli': all stimulus regressors
        - 'all_nuisance': all nuisance regressors
        If None, includes all regressors
    exclude_groups : list of str, optional
        Groups to exclude (applied after include)

    Returns
    -------
    design_matrix : np.ndarray
        Subset of design matrix (n_timepoints, n_selected)

    Examples
    --------
    >>> design = read_afni_design_matrix('X.xmat.1D')
    >>> # Get only stimulus regressors
    >>> stim_design = select_regressors_by_group(design, include_groups=['all_stimuli'])
    >>> # Get design without polynomials
    >>> X = select_regressors_by_group(design, exclude_groups=['polort'])
    """
    groups = get_regressor_groups(design_info)
    matrix = design_info["matrix"]

    # Start with all columns if no include specified
    if include_groups is None:
        selected_cols = list(range(matrix.shape[1]))
    else:
        selected_cols = []
        for group_name in include_groups:
            if group_name in groups:
                selected_cols.extend(groups[group_name])
            else:
                available = [k for k in groups.keys() if groups[k]]
                raise ValueError(f"Unknown group '{group_name}'. Available groups: {available}")
        selected_cols = sorted(set(selected_cols))

    # Apply exclusions
    if exclude_groups is not None:
        exclude_cols = []
        for group_name in exclude_groups:
            if group_name in groups:
                exclude_cols.extend(groups[group_name])
        selected_cols = [c for c in selected_cols if c not in exclude_cols]

    return matrix[:, selected_cols]


def read_censor_1d(filepath: str | Path, n_expected: int | None = None) -> list[int]:
    """Read an AFNI-style censor ``.1D`` file into a GoodList.

    The file is one value per concatenated timepoint (length = total TRs across
    all runs, in order): ``1`` = keep this TR, ``0`` = censor it. This is the
    format ``3dDeconvolve``/``1d_tool.py`` write (e.g. ``-censor_motion`` output).

    Parameters
    ----------
    filepath : str or Path
        Path to the censor ``.1D`` (single column of 0/1).
    n_expected : int, optional
        Expected number of timepoints (total concatenated length). If given and
        the file length differs, a ValueError is raised — a length mismatch
        almost always means the censor file does not match the data/design.

    Returns
    -------
    good_list : list[int]
        Indices (0-based, into the concatenated timeline) of the kept TRs — the
        same convention as the xmat ``GoodList`` header, so it can be dropped
        straight into ``design_info["good_list"]``.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Censor file not found: {filepath}")

    values = np.loadtxt(filepath).reshape(-1)
    if values.ndim != 1:
        raise ValueError(
            f"Censor file {filepath} must be a single column; got shape {values.shape}"
        )

    uniq = set(np.unique(values).tolist())
    if not uniq <= {0.0, 1.0}:
        raise ValueError(
            f"Censor file {filepath} must contain only 0 and 1; found values {sorted(uniq)}"
        )

    if n_expected is not None and len(values) != n_expected:
        raise ValueError(
            f"Censor file {filepath} has {len(values)} rows but the data/design "
            f"has {n_expected} timepoints. The censor file must have one row per "
            f"concatenated TR."
        )

    return np.nonzero(values > 0.5)[0].tolist()


def get_censored_mask(design_info: dict) -> np.ndarray:
    """
    Get boolean mask indicating which timepoints were censored

    Uses the GoodList attribute to determine which TRs were kept vs. censored.

    Parameters
    ----------
    design_info : dict
        Design matrix info from read_afni_design_matrix()

    Returns
    -------
    censored_mask : np.ndarray of bool, shape (n_timepoints,)
        True for censored timepoints, False for uncensored (good) timepoints

    Notes
    -----
    AFNI's GoodList contains indices of timepoints that were NOT censored.
    This is used by 3dREMLfit to properly handle temporal autocorrelation
    when some timepoints have been removed due to motion or other artifacts.

    Examples
    --------
    >>> design = read_afni_design_matrix('X.xmat.1D')
    >>> censored = get_censored_mask(design)
    >>> n_censored = censored.sum()
    >>> print(f"Censored {n_censored} of {len(censored)} timepoints")

    >>> # Extract only uncensored data
    >>> data_uncensored = data[~censored]
    >>> design_uncensored = design['matrix'][~censored]
    """
    good_list = design_info.get("good_list")
    n_timepoints = design_info.get("n_timepoints")

    if good_list is None:
        raise ValueError("Design matrix does not have GoodList information")

    if n_timepoints is None:
        raise ValueError("Design matrix does not have n_timepoints information")

    # Create mask: True = censored, False = good
    censored_mask = np.ones(n_timepoints, dtype=bool)
    censored_mask[good_list] = False

    return censored_mask


def select_uncensored_timepoints(
    design_info: dict,
    data: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Select only uncensored timepoints from design matrix and/or data

    Parameters
    ----------
    design_info : dict
        Design matrix info from read_afni_design_matrix()
    data : np.ndarray, optional
        fMRI data array with time as first dimension
        If provided, will also filter data to uncensored timepoints

    Returns
    -------
    design_uncensored : np.ndarray
        Design matrix with only uncensored timepoints
    data_uncensored : np.ndarray (only if data provided)
        Data array with only uncensored timepoints

    Examples
    --------
    >>> design = read_afni_design_matrix('X.xmat.1D')
    >>> # Filter design matrix only
    >>> X_uncensored = select_uncensored_timepoints(design)
    >>>
    >>> # Filter both design and data
    >>> X_uncensored, Y_uncensored = select_uncensored_timepoints(design, fmri_data)
    """
    good_list = design_info.get("good_list")

    if good_list is None:
        raise ValueError("Design matrix does not have GoodList information")

    matrix = design_info["matrix"]
    design_uncensored = matrix[good_list, :]

    if data is not None:
        data_uncensored = data[good_list]
        return design_uncensored, data_uncensored

    return design_uncensored


def replace_afni_extension(filepath: str, new_extension: str = ".nii.gz") -> str:
    """
    Replace any AFNI or NIfTI extension with a new extension.

    Preserves periods in base filename (e.g., "stats.blur.2mm" stays intact).
    By default outputs .nii.gz format (we only write NIfTI now).

    Parameters
    ----------
    filepath : str
        Input file path
    new_extension : str, default=".nii.gz"
        New extension to add (should start with ".")

    Returns
    -------
    str
        Filepath with new extension

    Examples
    --------
    >>> replace_afni_extension("stats", ".nii.gz")
    'stats.nii.gz'
    >>> replace_afni_extension("stats.nii", ".nii.gz")
    'stats.nii.gz'
    >>> replace_afni_extension("stats+orig.HEAD", ".nii.gz")
    'stats.nii.gz'
    >>> replace_afni_extension("stats.blur.2mm", ".nii.gz")
    'stats.blur.2mm.nii.gz'
    >>> replace_afni_extension("stats.nii.zst", ".nii.gz")
    'stats.nii.gz'
    """
    # Define known extensions (order matters - check longest first!)
    EXTENSIONS = [
        "+orig.BRIK.gz",
        "+tlrc.BRIK.gz",  # AFNI compressed
        "+orig.BRIK",
        "+tlrc.BRIK",  # AFNI uncompressed
        "+orig.HEAD",
        "+tlrc.HEAD",  # AFNI headers
        ".nii.zst",  # NIfTI zstandard compressed
        ".nii.gz",  # NIfTI gzip compressed
        ".nii",  # NIfTI uncompressed
    ]

    # Strip any existing extension
    for ext in EXTENSIONS:
        if filepath.endswith(ext):
            filepath = filepath[: -len(ext)]
            break

    # Add new extension
    return filepath + new_extension


def get_tr_from_file(filepath: str | Path) -> float:
    """
    Extract TR (repetition time) from NIfTI header.

    Parameters
    ----------
    filepath : str or Path
        Path to NIfTI file

    Returns
    -------
    float
        TR in seconds, or 1.0 if not found

    Examples
    --------
    >>> tr = get_tr_from_file("func.nii.gz")
    >>> print(f"TR = {tr}s")
    """
    try:
        img = load_nifti(filepath)
        tr = img.header.get_zooms()[-1]  # Last dimension is time
        if tr == 0 or tr is None:
            print(f"WARNING: TR not found in {filepath} header, using 1.0")
            return 1.0
        return float(tr)
    except Exception as e:
        print(f"WARNING: Could not read TR from {filepath}: {e}")
        print("Using TR=1.0 as fallback")
        return 1.0


def load_fmri_data(
    fmri_file: str | Path,
    mask_file: str | Path,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Load fMRI data and mask to 2D array format (timepoints × voxels)

    Parameters
    ----------
    fmri_file : str or Path
        Path to 4D fMRI file (supports .nii, .nii.gz, .nii.zst)
    mask_file : str or Path
        Path to 3D mask file (values > 0 define brain voxels, supports .nii, .nii.gz, .nii.zst)
    dtype : np.dtype, default=np.float32
        Data type for output array

    Returns
    -------
    data : np.ndarray, shape (n_timepoints, n_voxels)
        fMRI data in 2D format

    Examples
    --------
    >>> data = load_fmri_data('func.nii.gz', 'mask.nii.gz')
    >>> print(f"Shape: {data.shape}")  # (n_timepoints, n_voxels)
    """
    # Load fMRI data
    fmri_img = load_nifti(fmri_file)
    fmri_data = fmri_img.get_fdata()

    if fmri_data.ndim != 4:
        raise ValueError(f"fMRI data must be 4D, got {fmri_data.ndim}D")

    # Load mask
    mask_img = load_nifti(mask_file)
    mask_data = mask_img.get_fdata()

    if mask_data.ndim != 3:
        raise ValueError(f"Mask must be 3D, got {mask_data.ndim}D")

    # Check spatial dimensions match
    if fmri_data.shape[:3] != mask_data.shape:
        raise ValueError(
            f"Spatial dimensions mismatch: fMRI {fmri_data.shape[:3]} vs mask {mask_data.shape}"
        )

    # Create boolean mask
    mask_bool = mask_data > 0
    n_voxels = mask_bool.sum()
    n_timepoints = fmri_data.shape[3]

    # Extract masked data
    data = np.zeros((n_timepoints, n_voxels), dtype=dtype)
    for t in range(n_timepoints):
        data[t] = fmri_data[..., t][mask_bool]

    return data


def compress_nifti(
    uncompressed_path: str | Path,
    output_path: str | Path | None = None,
    *,
    remove_original: bool = True,
) -> Path:
    """Compress an uncompressed .nii file to .nii.gz (pigz) or .nii.zst (zstd).

    The target format is determined by the extension of *output_path*.
    If *output_path* is None, the extension of *uncompressed_path* is used
    (which must be .nii.gz or .nii.zst).

    Parameters
    ----------
    uncompressed_path : str or Path
        Path to the uncompressed .nii file on disk.
    output_path : str or Path, optional
        Desired compressed output path.  Must end with ``.nii.gz`` or
        ``.nii.zst``.  When *None*, compression format is inferred from
        *uncompressed_path* (e.g. if it already ends with ``.nii.gz``
        nibabel wrote it uncompressed inside a .gz name – we recompress).
    remove_original : bool
        Delete *uncompressed_path* after successful compression (default True).

    Returns
    -------
    Path
        Final path of the compressed file.
    """
    src = Path(uncompressed_path)
    if not src.exists():
        raise FileNotFoundError(f"compress_nifti: source not found: {src}")

    if output_path is not None:
        dst = Path(output_path)
    else:
        dst = src  # in-place: same name, compress based on extension

    dst_str = str(dst)

    if dst_str.endswith(".nii.zst"):
        return _compress_zst(src, dst, remove_original)
    elif dst_str.endswith(".nii.gz"):
        return _compress_gz(src, dst, remove_original)
    else:
        raise ValueError(
            f"compress_nifti: output_path must end with .nii.gz or .nii.zst, got '{dst}'"
        )


def _compress_gz(src: Path, dst: Path, remove_original: bool) -> Path:
    """Compress *src* (.nii) → *dst* (.nii.gz) using pigz or gzip fallback."""
    if shutil.which("pigz"):
        # pigz appends .gz to the input filename.  Work with that.
        subprocess.run(
            ["pigz", "-f", str(src)],
            check=True,
            capture_output=True,
        )
        pigz_out = Path(str(src) + ".gz")
        if pigz_out != dst:
            pigz_out.rename(dst)
        return dst

    # Fallback: Python gzip (single-threaded)
    import gzip

    with open(src, "rb") as f_in, gzip.open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    if remove_original and src != dst:
        src.unlink()
    return dst


def _compress_zst(src: Path, dst: Path, remove_original: bool) -> Path:
    """Compress *src* (.nii) → *dst* (.nii.zst) using zstd with all cores."""
    zstd = shutil.which("zstd")
    if zstd is None:
        raise RuntimeError(
            "zstd is not installed or not in PATH.  "
            "Install zstd to write .nii.zst files (e.g. apt install zstd)."
        )
    # -T0 = use all physical cores, -f = overwrite, --rm = remove source
    cmd = ["zstd", "-T0", "-f"]
    if remove_original:
        cmd.append("--rm")
    cmd.extend([str(src), "-o", str(dst)])
    subprocess.run(cmd, check=True, capture_output=True)
    return dst


_NIFTI_ECODE_AFNI = 4
# NIfTI datatype codes used by AFNI's NIfTI_nums consistency string
_NP_DTYPE_TO_NIFTI_CODE: dict[str, int] = {
    "float32": 16,
    "float64": 64,
    "int16": 4,
    "int32": 8,
    "uint8": 2,
    "int8": 256,
    "uint16": 512,
}


def _generate_afni_idcode() -> str:
    """Generate an AFNI-style unique ID: AFN_<22 base64 chars>."""
    import base64
    import os

    raw = base64.b64encode(os.urandom(16)).decode("ascii")
    # AFNI IDs use a URL-safe-ish alphabet; replace +/ with safe chars
    raw = raw.replace("+", "x").replace("/", "X").rstrip("=")
    return f"AFN_{raw[:22]}"


def get_afni_space_info(header: object) -> dict[str, str | int]:
    """Extract AFNI view code and template space from a NIfTI header.

    Returns a dict with:
        - ``view``: 0 (orig), 1 (acpc), or 2 (tlrc)
        - ``space``: e.g. "ORIG", "MNI_2009c_asym", "TLRC"

    If the header has no AFNI extension, returns ``{"view": 0, "space": "ORIG"}``.
    """
    import re

    result: dict[str, str | int] = {"view": 0, "space": "ORIG"}

    try:
        extensions = header.extensions  # type: ignore[union-attr]
    except AttributeError:
        return result

    for ext in extensions:
        if ext.get_code() == _NIFTI_ECODE_AFNI:
            xml = ext.content.decode("utf-8", errors="replace")

            # SCENE_DATA — first value is the view code
            m = re.search(
                r'atr_name="SCENE_DATA"\s*>\s*\n\s*(\d+)',
                xml,
            )
            if m:
                result["view"] = int(m.group(1))

            # TEMPLATE_SPACE
            m = re.search(
                r'atr_name="TEMPLATE_SPACE"\s*>\s*\n\s*"([^"]*)"',
                xml,
            )
            if m:
                result["space"] = m.group(1)

            break

    return result


def set_afni_space_info(
    header: object,
    view: int,
    space: str,
) -> None:
    """Set AFNI view code and template space in a NIfTI header's extension.

    Args:
        header: nibabel NIfTI header (must already have an AFNI extension)
        view: 0 (orig), 1 (acpc), or 2 (tlrc)
        space: Template space string, e.g. "ORIG", "MNI_2009c_asym"
    """
    import re

    try:
        import nibabel as nib

        extensions = header.extensions  # type: ignore[union-attr]
    except AttributeError:
        return

    for i, ext in enumerate(extensions):
        if ext.get_code() == _NIFTI_ECODE_AFNI:
            xml = ext.content.decode("utf-8", errors="replace")

            # Update SCENE_DATA — replace first integer
            xml = re.sub(
                r'(atr_name="SCENE_DATA"\s*>\s*\n\s*)\d+',
                rf"\g<1>{view}",
                xml,
            )

            # Update TEMPLATE_SPACE
            xml = re.sub(
                r'(atr_name="TEMPLATE_SPACE"\s*>\s*\n\s*)"[^"]*"',
                rf'\1"{space}"',
                xml,
            )

            new_ext = nib.nifti1.Nifti1Extension(_NIFTI_ECODE_AFNI, xml.encode("utf-8"))
            extensions[i] = new_ext
            break


def set_afni_func_type(header: object, func_code: int = 11) -> None:
    """Set AFNI SCENE_DATA[1] (func type) and TYPESTRING in a NIfTI AFNI extension.

    Call this after inheriting a header from an EPI source when the output is a
    stats/parameter bucket — the EPI source has epan/3DIM_HEAD_ANAT, but GLM
    outputs should be fbuc/3DIM_HEAD_FUNC.

    Parameters
    ----------
    header : nibabel NIfTI header with an AFNI extension (ecode=4)
    func_code : int
        11 = FUNC_BUCK_TYPE (fbuc) — stat buckets (Rbuck, Rvar, etc.)
         2 = ANAT_EPI_TYPE  (epan) — EPI timeseries (restore if needed)
    """
    import re

    type_str = "3DIM_HEAD_FUNC" if func_code >= 10 else "3DIM_HEAD_ANAT"

    try:
        import nibabel as nib

        extensions = header.extensions  # type: ignore[union-attr]
    except AttributeError:
        return

    for i, ext in enumerate(extensions):
        if ext.get_code() == _NIFTI_ECODE_AFNI:
            xml = ext.content.decode("utf-8", errors="replace")

            # Update SCENE_DATA[1] (second integer, one per line in NIfTI XML)
            xml = re.sub(
                r'(atr_name="SCENE_DATA"[^>]*>\s*\n\s*\d+\s*\n\s*)\d+',
                rf"\g<1>{func_code}",
                xml,
            )

            # Update TYPESTRING
            xml = re.sub(
                r'(atr_name="TYPESTRING"\s*>\s*\n\s*)"[^"]*"',
                rf'\1"{type_str}"',
                xml,
            )

            extensions[i] = nib.nifti1.Nifti1Extension(_NIFTI_ECODE_AFNI, xml.encode("utf-8"))
            break


def _update_afni_extension(
    header: object,
    data_shape: tuple[int, ...],
    data_dtype: np.dtype,
) -> None:
    """Update the AFNI NIfTI extension (ecode=4) in a header, if present.

    Updates:
        - self_idcode / IDCODE_STRING → new unique ID
        - IDCODE_DATE → current timestamp
        - NIfTI_nums → matches actual data shape/dtype
        - HISTORY_NOTE → appends fastfuncstuff provenance
        - BRICK_STATS / BRICK_STATSYM / BRICK_LABS → trimmed if volume count changed
    """
    import re
    import sys
    from datetime import datetime

    try:
        extensions = header.extensions  # type: ignore[union-attr]
    except AttributeError:
        return

    # Find AFNI extension
    afni_idx = None
    afni_xml = None
    for i, ext in enumerate(extensions):
        if ext.get_code() == _NIFTI_ECODE_AFNI:
            afni_idx = i
            afni_xml = ext.content.decode("utf-8", errors="replace")
            break

    if afni_idx is None or afni_xml is None:
        return  # No AFNI extension to update

    new_id = _generate_afni_idcode()
    now_str = datetime.now().strftime("%a %b %d %H:%M:%S %Y")

    # Build NIfTI_nums: "nx,ny,nz,nt,nu,datatype"
    nx = data_shape[0] if len(data_shape) >= 1 else 1
    ny = data_shape[1] if len(data_shape) >= 2 else 1
    nz = data_shape[2] if len(data_shape) >= 3 else 1
    nt = data_shape[3] if len(data_shape) >= 4 else 1
    nu = data_shape[4] if len(data_shape) >= 5 else 1
    dtype_code = _NP_DTYPE_TO_NIFTI_CODE.get(str(data_dtype), 16)
    new_nums = f"{nx},{ny},{nz},{nt},{nu},{dtype_code}"

    # --- Update self_idcode in the group header ---
    afni_xml = re.sub(
        r'self_idcode="[^"]*"',
        f'self_idcode="{new_id}"',
        afni_xml,
    )

    # --- Update NIfTI_nums ---
    afni_xml = re.sub(
        r'NIfTI_nums="[^"]*"',
        f'NIfTI_nums="{new_nums}"',
        afni_xml,
    )

    # --- Update IDCODE_STRING ---
    afni_xml = re.sub(
        r'(atr_name="IDCODE_STRING"\s*>\s*\n)\s*"[^"]*"',
        rf'\1 "{new_id}"',
        afni_xml,
    )

    # --- Update IDCODE_DATE ---
    afni_xml = re.sub(
        r'(atr_name="IDCODE_DATE"\s*>\s*\n)\s*"[^"]*"',
        rf'\1 "{now_str}"',
        afni_xml,
    )

    # --- Append to HISTORY_NOTE ---
    cmd_line = " ".join(sys.argv)
    history_entry = f"[fastfuncstuff: {now_str}] {cmd_line}"

    history_match = re.search(
        r'(atr_name="HISTORY_NOTE"\s*>\s*\n)\s*"([^"]*)"',
        afni_xml,
        re.DOTALL,
    )
    if history_match:
        old_history = history_match.group(2)
        new_history = old_history + r"\n" + history_entry
        afni_xml = (
            afni_xml[: history_match.start(2)] + new_history + afni_xml[history_match.end(2) :]
        )
    # If no HISTORY_NOTE found, that's fine — don't add one

    # --- Update DATASET_RANK[1] (sub-brick count) ---
    # DATASET_RANK is stored as 8 ints: [3, n_sub_briks, 0, 0, 0, 0, 0, 0].
    # Index 1 must match nt (the new volume count); mismatches cause AFNI's
    # "NBL does not match nim" error when 3drefit tries to relabel a bucket
    # whose header was copied from a shorter time-series file.
    n_bricks = nt if nt > 1 else nu
    afni_xml = re.sub(
        r'(atr_name="DATASET_RANK"[^>]*>\s*\n\s*\d+\s+)(\d+)',
        rf"\g<1>{n_bricks}",
        afni_xml,
    )

    # --- Update TAXIS_NUMS[0] (number of time steps / sub-bricks) ---
    # TAXIS_NUMS[0] is the field AFNI's nifti_image_write_engine uses to validate
    # brick count during write ("NBL == TAXIS_NUMS[0]").  When the header is
    # copied from an fMRI run file this will be the run length (e.g. 268); it
    # must be updated to the new brick count or the write fails with
    # "NBL does not match nim".  The remaining TAXIS_NUMS fields (nt_type,
    # datum type, offsets) are left unchanged.
    afni_xml = re.sub(
        r'(atr_name="TAXIS_NUMS"[^>]*>\s*\n\s*)(\d+)',
        rf"\g<1>{n_bricks}",
        afni_xml,
    )

    # --- Trim per-brick attributes when volume count changed ---
    # BRICK_STATS/STATSYM/LABS/TYPES all have one entry per sub-brick and are
    # stale after a volume-count change.  Remove them; AFNI/3drefit regenerates
    # them from -substatpar / -relabel_all.
    for atr_name in (
        "BRICK_STATS",
        "BRICK_STATSYM",
        "BRICK_LABS",
        "BRICK_TYPES",
        "BRICK_FLOAT_FACS",
    ):
        afni_xml = re.sub(
            rf'<AFNI_atr\s[^>]*atr_name="{atr_name}"[^>]*>.*?</AFNI_atr>\s*',
            "",
            afni_xml,
            flags=re.DOTALL,
        )

    # Write back the updated extension
    import nibabel as nib

    new_ext = nib.nifti1.Nifti1Extension(_NIFTI_ECODE_AFNI, afni_xml.encode("utf-8"))
    extensions[afni_idx] = new_ext


def _set_afni_brick_labels(header: object, labels: list[str]) -> None:
    """Set per-sub-brick labels (BRICK_LABS) so AFNI viewers show them.

    NIfTI has no native sub-brick labels; AFNI stores them in its NIfTI
    extension (ecode=4) as a ``~``-separated ``BRICK_LABS`` attribute. This
    injects/replaces that attribute, reusing an existing AFNI extension when the
    input had one, or creating a minimal one (matching what AFNI itself writes)
    for a plain NIfTI input. Call *after* :func:`_update_afni_extension`, which
    strips any stale BRICK_LABS first.
    """
    import re

    import nibabel as nib

    try:
        extensions = header.extensions  # type: ignore[union-attr]
    except AttributeError:
        return

    labs = "~".join(labels)
    atr = (
        "<AFNI_atr\n"
        '  ni_type="String"\n'
        '  ni_dimen="1"\n'
        '  atr_name="BRICK_LABS" >\n'
        f' "{labs}"\n'
        "</AFNI_atr>\n"
    )

    afni_idx = None
    for i, ext in enumerate(extensions):
        if ext.get_code() == _NIFTI_ECODE_AFNI:
            afni_idx = i
            break

    if afni_idx is not None:
        xml = extensions[afni_idx].content.decode("utf-8", errors="replace")
        xml = re.sub(
            r'<AFNI_atr\s[^>]*atr_name="BRICK_LABS"[^>]*>.*?</AFNI_atr>\s*',
            "",
            xml,
            flags=re.DOTALL,
        )
        if "</AFNI_attributes>" in xml:
            xml = xml.replace("</AFNI_attributes>", atr + "</AFNI_attributes>", 1)
        else:
            xml = xml + atr
        extensions[afni_idx] = nib.nifti1.Nifti1Extension(_NIFTI_ECODE_AFNI, xml.encode("utf-8"))
    else:
        idc = _generate_afni_idcode()
        payload = (
            "<?xml version='1.0' ?>\n"
            "<AFNI_attributes\n"
            f'  self_idcode="{idc}"\n'
            '  ni_form="ni_group" >\n'
            f"{atr}"
            "</AFNI_attributes>\n\x00"
        )
        extensions.append(nib.nifti1.Nifti1Extension(_NIFTI_ECODE_AFNI, payload.encode("utf-8")))


def _statsym_for(code: int, params: tuple[float, ...]) -> str:
    """Symbolic AFNI stat label for one sub-brick (``Ttest(dof)`` / ``Ftest(n,d)`` /
    ``Zscore()``). ``"none"`` for non-stat sub-bricks. Mirrors the parser in
    ``cli/util_concalc.py`` so round-trips agree."""
    if code == 3 and len(params) == 1:  # fitt
        return f"Ttest({int(params[0])})"
    if code == 4 and len(params) == 2:  # fift
        return f"Ftest({int(params[0])},{int(params[1])})"
    if code == 5:  # fizt
        return "Zscore()"
    return "none"


def _set_afni_brick_stataux(
    header: object,
    stataux: dict[int, tuple[int, tuple[float, ...]]],
    n_sub: int,
) -> None:
    """Tag sub-bricks as statistics (``BRICK_STATAUX`` + ``BRICK_STATSYM``) so AFNI
    can threshold them and compute FDR. ``stataux`` maps ``sub_brick_index ->
    (afni_stat_code, params)`` (e.g. ``{1: (3, (dof,))}`` for a t-stat). Mirrors
    :func:`_set_afni_brick_labels`: reuses an existing AFNI extension or creates a
    minimal one, replacing any stale STATAUX/STATSYM."""
    import re

    import nibabel as nib

    try:
        extensions = header.extensions  # type: ignore[union-attr]
    except AttributeError:
        return

    stataux_floats: list[float] = []
    for idx in sorted(stataux):
        code, params = stataux[idx]
        stataux_floats.extend([float(idx), float(code), float(len(params))])
        stataux_floats.extend(float(p) for p in params)
    syms = ";".join(_statsym_for(*stataux[i]) if i in stataux else "none" for i in range(n_sub))
    aux_body = " " + "\n ".join(f"{v:g}" for v in stataux_floats)
    atr = (
        "<AFNI_atr\n"
        '  ni_type="float"\n'
        f'  ni_dimen="{len(stataux_floats)}"\n'
        '  atr_name="BRICK_STATAUX" >\n'
        f"{aux_body}\n"
        "</AFNI_atr>\n"
        "<AFNI_atr\n"
        '  ni_type="String"\n'
        '  ni_dimen="1"\n'
        '  atr_name="BRICK_STATSYM" >\n'
        f' "{syms}"\n'
        "</AFNI_atr>\n"
    )

    afni_idx = None
    for i, ext in enumerate(extensions):
        if ext.get_code() == _NIFTI_ECODE_AFNI:
            afni_idx = i
            break
    if afni_idx is not None:
        xml = extensions[afni_idx].content.decode("utf-8", errors="replace")
        for name in ("BRICK_STATAUX", "BRICK_STATSYM"):
            xml = re.sub(
                rf'<AFNI_atr\s[^>]*atr_name="{name}"[^>]*>.*?</AFNI_atr>\s*',
                "",
                xml,
                flags=re.DOTALL,
            )
        if "</AFNI_attributes>" in xml:
            xml = xml.replace("</AFNI_attributes>", atr + "</AFNI_attributes>", 1)
        else:
            xml = xml + atr
        extensions[afni_idx] = nib.nifti1.Nifti1Extension(_NIFTI_ECODE_AFNI, xml.encode("utf-8"))
    else:
        idc = _generate_afni_idcode()
        payload = (
            "<?xml version='1.0' ?>\n"
            "<AFNI_attributes\n"
            f'  self_idcode="{idc}"\n'
            '  ni_form="ni_group" >\n'
            f"{atr}"
            "</AFNI_attributes>\n\x00"
        )
        extensions.append(nib.nifti1.Nifti1Extension(_NIFTI_ECODE_AFNI, payload.encode("utf-8")))


def save_nifti(
    data: np.ndarray,
    output_path: str | Path,
    reference_img: str | Path | None = None,
    affine: np.ndarray | None = None,
    tr: float | None = None,
    header: object | None = None,
    brick_labels: list[str] | None = None,
    brick_stataux: dict[int, tuple[int, tuple[float, ...]]] | None = None,
):
    """Save data as a NIfTI file with efficient compression.

    Compression is chosen automatically from the *output_path* extension:

    * ``.nii``     — uncompressed (nibabel direct write)
    * ``.nii.gz``  — gzip via **pigz** (multicore) with gzip fallback
    * ``.nii.zst`` — zstandard via **zstd -T0** (multicore, fastest)

    Parameters
    ----------
    data : np.ndarray
        Data to save (3D or 4D array).
    output_path : str or Path
        Output file path (extension determines compression).
    reference_img : str or Path, optional
        Reference NIfTI file to copy affine/header from.
    affine : np.ndarray, optional
        4×4 affine transformation matrix.
        If None and no *reference_img*, uses identity.
    tr : float, optional
        Repetition time in seconds (for 4D data).
    header : nibabel header, optional
        NIfTI header to copy (preserves TR, xyzt_units, etc.).
        Overridden by *reference_img* if both are given.
    """
    try:
        import nibabel as nib
    except ImportError as err:
        raise ImportError(
            "nibabel is required to save NIfTI files. Install with: pip install nibabel"
        ) from err

    # Get affine and header info
    if reference_img is not None:
        ref_img = load_nifti(reference_img)
        affine = ref_img.affine
        header = ref_img.header.copy()
    elif header is not None:
        header = header.copy()
        if affine is None:
            affine = np.eye(4)
    elif affine is None:
        affine = np.eye(4)
        header = nib.Nifti1Header()
    else:
        header = nib.Nifti1Header()

    # Sync header dims to actual data shape (handles sub-brick selection,
    # partial loads, or any processing that changed the volume count)
    if header is not None:
        header.set_data_shape(data.shape)
        # Sync on-disk dtype to the data. A header copied from a short/int input
        # otherwise forces nibabel to quantize our float32 results to int16 on
        # write (precision loss), and leaves the AFNI extension's NIfTI_nums
        # datatype mismatched -- which makes AFNI warn that the file's
        # "dimensions altered since AFNI extension was added".
        header.set_data_dtype(data.dtype)

    # Update AFNI NIfTI extension if present (ecode=4):
    #   - new IDCODE so AFNI treats this as a distinct dataset
    #   - NIfTI_nums consistency string matching actual data
    #   - IDCODE_DATE with current timestamp
    #   - HISTORY_NOTE appended with our command
    _update_afni_extension(header, data.shape, data.dtype)

    # Per-sub-brick labels for AFNI viewers (NIfTI has no native equivalent).
    if brick_labels is not None:
        _set_afni_brick_labels(header, brick_labels)

    # Per-sub-brick stat tagging so AFNI can threshold + FDR the sub-bricks.
    if brick_stataux is not None:
        n_sub = data.shape[3] if data.ndim == 4 else 1
        _set_afni_brick_stataux(header, brick_stataux, n_sub)

    # Create NIfTI image
    img = nib.Nifti1Image(data, affine, header=header)

    # Set TR if provided
    if tr is not None:
        img.header.set_xyzt_units(xyz="mm", t="sec")
        img.header["pixdim"][4] = tr

    out = Path(output_path)
    out_str = str(out)

    if out_str.endswith(".nii.zst") or (out_str.endswith(".nii.gz") and shutil.which("pigz")):
        # Write uncompressed to a temp file first, then compress externally.
        # This avoids nibabel's single-threaded gzip and enables pigz / zstd.
        stem = out.name
        for ext in (".nii.zst", ".nii.gz"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        tmp_nii = out.parent / (stem + ".nii")
        nib.save(img, str(tmp_nii))
        compress_nifti(tmp_nii, out, remove_original=True)
    else:
        # .nii or .nii.gz without pigz → let nibabel handle it directly
        nib.save(img, str(out))
