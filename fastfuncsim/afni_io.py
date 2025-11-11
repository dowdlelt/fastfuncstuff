"""
AFNI file I/O utilities
Read AFNI onset timing files and design matrices (X.xmat.1D format)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm.auto import tqdm

try:
    import nibabel as nib
except (
    ImportError
) as exc:  # pragma: no cover - nibabel is required for AFNI interoperability
    raise ImportError(
        "nibabel is required for AFNI mask and imaging utilities. Install it with `pip install nibabel`."
    ) from exc

from .utils import get_device, to_tensor


def read_afni_onset_file(filepath: Union[str, Path]) -> List[np.ndarray]:
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

    with open(filepath, "r") as f:
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


def read_afni_onset_files(filepaths: List[Union[str, Path]]) -> List[List[np.ndarray]]:
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


def onsets_to_binary_matrix(
    onsets_per_condition: List[List[np.ndarray]],
    n_timepoints: int,
    tr: float,
    run_starts: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Convert AFNI onset times to binary onset matrix

    Parameters
    ----------
    onsets_per_condition : list of list of np.ndarray
        onsets_per_condition[cond][run] = onset times in seconds
    n_timepoints : int
        Total number of timepoints across all runs
    tr : float
        TR in seconds
    run_starts : list of int, optional
        Starting timepoint index for each run (from AFNI RunStart parameter)
        If None, assumes equal-length runs
        Example: [0, 300, 600, 900] for 4 runs of 300 TRs each
    device : torch.device, optional
        Device for output tensor

    Returns
    -------
    onset_matrix : torch.Tensor
        Binary onset matrix (n_timepoints, n_conditions)

    Examples
    --------
    >>> files = ['onsets_cond1.txt', 'onsets_cond2.txt']
    >>> onsets = read_afni_onset_files(files)
    >>> # Equal-length runs
    >>> onset_mat = onsets_to_binary_matrix(onsets, n_timepoints=200, tr=2.0)
    >>> # Unequal runs (from AFNI RunStart)
    >>> onset_mat = onsets_to_binary_matrix(onsets, n_timepoints=1200, tr=1.5,
    ...                                     run_starts=[0, 300, 600, 900])
    """
    if device is None:
        device = get_device()

    n_conditions = len(onsets_per_condition)
    n_runs = len(onsets_per_condition[0])

    # Validate all conditions have same number of runs
    for cond_idx, cond_onsets in enumerate(onsets_per_condition):
        if len(cond_onsets) != n_runs:
            raise ValueError(
                f"Condition {cond_idx} has {len(cond_onsets)} runs, expected {n_runs}"
            )

    # Create binary onset matrix
    onset_matrix = np.zeros((n_timepoints, n_conditions))

    # Determine run boundaries
    if run_starts is not None:
        # Use provided run starts
        if len(run_starts) != n_runs:
            raise ValueError(
                f"run_starts has {len(run_starts)} entries, but data has {n_runs} runs"
            )
        timepoint_offsets = run_starts
    else:
        # Assume equal-length runs
        timepoints_per_run = n_timepoints // n_runs
        timepoint_offsets = [run_idx * timepoints_per_run for run_idx in range(n_runs)]

    for cond_idx, cond_onsets in enumerate(onsets_per_condition):
        for run_idx, run_onsets in enumerate(cond_onsets):
            # Convert onset times to timepoint indices
            timepoint_offset = timepoint_offsets[run_idx]

            for onset_time in run_onsets:
                timepoint = int(np.round(onset_time / tr))
                global_timepoint = timepoint_offset + timepoint

                if 0 <= global_timepoint < n_timepoints:
                    onset_matrix[global_timepoint, cond_idx] = 1

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

    # Create matrix
    if len(values) != n_cols:
        raise ValueError(f"Expected {n_cols} values, got {len(values)}")

    matrix = np.array(values).reshape(n_rows, n_cols)
    return matrix


def read_afni_design_matrix(filepath: Union[str, Path]) -> Dict:
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
        - 'run_starts': list of int - starting indices for each run
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

    with open(filepath, "r") as f:
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
                        metadata["column_labels"] = [
                            x.strip() for x in value.split(";")
                        ]

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

                    elif key == "StimLabels":
                        metadata["stim_labels"] = [x.strip() for x in value.split(";")]

                    elif key == "StimBots":
                        # Parse "20..23" format
                        if ".." in value:
                            start, end = map(int, value.split(".."))
                            metadata["stim_bots"] = list(range(start, end + 1))
                        else:
                            metadata["stim_bots"] = [int(x) for x in value.split(",")]

                    elif key == "StimTops":
                        if ".." in value:
                            start, end = map(int, value.split(".."))
                            metadata["stim_tops"] = list(range(start, end + 1))
                        else:
                            metadata["stim_tops"] = [int(x) for x in value.split(",")]

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
    design_info: Dict,
    stim_indices: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
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
        for bot, top in zip(stim_bots, stim_tops):
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


def extract_nuisance_columns(
    design_info: Dict, device: Optional[torch.device] = None
) -> torch.Tensor:
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
    for bot, top in zip(stim_bots, stim_tops):
        stim_cols.update(range(bot, top + 1))

    nuisance_cols = [i for i in range(matrix.shape[1]) if i not in stim_cols]

    nuisance_design = matrix[:, nuisance_cols]

    return to_tensor(nuisance_design, device=device, dtype=torch.float32)


def load_and_concatenate_runs(
    run_files: List[Union[str, Path]], device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, List[int]]:
    """
    Load multiple fMRI run files and concatenate them (memory-efficient)

    Parameters
    ----------
    run_files : list of str or Path
        Paths to neuroimaging files, one per run
        Supports: NIfTI (.nii, .nii.gz) and AFNI (.HEAD, .BRIK, .BRIK.gz)
        For AFNI files, provide either .HEAD or .BRIK path
    device : torch.device, optional
        Device for output tensor

    Returns
    -------
    concatenated_data : torch.Tensor
        Concatenated data (n_voxels, total_timepoints)
    run_starts : list of int
        Starting timepoint index for each run

    Examples
    --------
    >>> run_files = ['run01.nii.gz', 'run02.nii.gz', 'run03.nii.gz']
    >>> data, run_starts = load_and_concatenate_runs(run_files)
    >>> print(f"Total timepoints: {data.shape[1]}")
    >>> print(f"Run starts: {run_starts}")

    Notes
    -----
    Memory-efficient implementation that converts each run to torch immediately
    and concatenates on-device to avoid holding multiple copies in memory.
    """
    import gc

    if device is None:
        device = get_device()

    run_starts = [0]
    current_start = 0
    torch_runs = []

    # Progress bar for loading runs
    run_iterator = enumerate(run_files)
    if len(run_files) > 1:
        run_iterator = enumerate(tqdm(run_files, desc="Loading fMRI runs", unit="run"))

    for i, run_file in run_iterator:
        # Load run
        img = nib.load(str(run_file))
        data_np = img.get_fdata(dtype=np.float32)

        # Reshape to (n_voxels, n_timepoints)
        if data_np.ndim == 4:
            n_voxels = data_np.shape[0] * data_np.shape[1] * data_np.shape[2]
            n_timepoints = data_np.shape[3]
            data_np = data_np.reshape(n_voxels, n_timepoints)
        elif data_np.ndim != 2:
            raise ValueError(f"Data must be 2D or 4D, got shape {data_np.shape}")

        # Convert to torch immediately and move to device
        # This avoids keeping numpy copy around
        data_torch = torch.from_numpy(data_np.astype(np.float32, copy=False)).to(device)

        # Delete numpy array immediately to free memory
        del data_np, img
        gc.collect()

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
    mask_file: Union[str, Path],
    threshold: float = 0.0,
) -> np.ndarray:
    """Load an AFNI-style mask and return a boolean volume.

    Parameters
    ----------
    mask_file : str or Path
        Path to the mask file (NIfTI or AFNI BRIK/HEAD format).
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

    img = nib.load(str(mask_path))
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


def get_run_lengths(run_starts: List[int], n_timepoints: int) -> List[int]:
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
    design_info: Dict, contrast_index: int, device: Optional[torch.device] = None
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


def extract_design_metadata(design_info: Dict) -> Tuple[List[str], List[str], List[int]]:
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
        for bot, top in zip(stim_bots, stim_tops):
            stim_column_indices.extend(range(bot, top + 1))

    # Extract stimulus labels (subset of full labels)
    if stim_column_indices and full_labels:
        stim_labels = [full_labels[i] for i in stim_column_indices]
    else:
        stim_labels = []

    return full_labels, stim_labels, stim_column_indices
