"""File I/O: AFNI format support, NIfTI loading, onset file parsing.

Re-exports resolve lazily (PEP 562) so that importing a header-only module —
``fastfuncstuff.io.headers`` / ``fastfuncstuff.io.dsetinfo`` — does not drag in
``io.afni`` and, behind it, torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

_LAZY: dict[str, str] = {
    "read_afni_onset_file": "fastfuncstuff.io.afni",
    "read_afni_onset_files": "fastfuncstuff.io.afni",
    "onsets_to_binary_matrix": "fastfuncstuff.io.afni",
    "read_afni_design_matrix": "fastfuncstuff.io.afni",
    "extract_stimulus_columns": "fastfuncstuff.io.afni",
    "extract_nuisance_columns": "fastfuncstuff.io.afni",
    "get_contrast_matrix": "fastfuncstuff.io.afni",
    "parse_afni_matrix_notation": "fastfuncstuff.io.afni",
    "load_and_concatenate_runs": "fastfuncstuff.io.afni",
    "get_run_lengths": "fastfuncstuff.io.afni",
    "load_afni_mask": "fastfuncstuff.io.afni",
    # Header-only (torch-free) entry points.
    "read_nifti_header": "fastfuncstuff.io.headers",
    "nifti_shape": "fastfuncstuff.io.headers",
    "parse_subbrick_selector": "fastfuncstuff.io.headers",
    "read_info": "fastfuncstuff.io.dsetinfo",
    "DatasetInfo": "fastfuncstuff.io.dsetinfo",
}


def __getattr__(name: str):
    try:
        module = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    from importlib import import_module

    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


if TYPE_CHECKING:
    from fastfuncstuff.io.afni import (
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
    from fastfuncstuff.io.dsetinfo import DatasetInfo, read_info
    from fastfuncstuff.io.headers import nifti_shape, parse_subbrick_selector, read_nifti_header

__all__ = sorted(_LAZY)
