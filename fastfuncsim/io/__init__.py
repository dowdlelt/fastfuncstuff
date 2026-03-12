"""File I/O: AFNI format support, NIfTI loading, onset file parsing."""

from fastfuncsim.io.afni import (
    extract_nuisance_columns,
    extract_stimulus_columns,
    get_contrast_matrix,
    get_run_lengths,
    load_afni_mask,
    load_and_concatenate_runs,
    onsets_to_binary_matrix,
    parse_afni_matrix_notation,
    read_afni_design_matrix,
    read_afni_onset_file,
    read_afni_onset_files,
)

__all__ = [
    "read_afni_onset_file",
    "read_afni_onset_files",
    "onsets_to_binary_matrix",
    "read_afni_design_matrix",
    "extract_stimulus_columns",
    "extract_nuisance_columns",
    "get_contrast_matrix",
    "parse_afni_matrix_notation",
    "load_and_concatenate_runs",
    "get_run_lengths",
    "load_afni_mask",
]
