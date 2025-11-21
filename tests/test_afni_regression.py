"""
Regression tests comparing fastfuncsim outputs against AFNI reference outputs.

Uses small validation dataset in ~/Dropbox/Data/small_validation_afni_data/
"""
import pytest
import numpy as np
import nibabel as nib
import torch
from pathlib import Path

from fastfuncsim.analysis import analyze_from_design_matrix
from fastfuncsim.glm_outputs import write_afni_bucket


# AFNI reference data paths
AFNI_DATA_DIR = Path.home() / "Dropbox/Data/small_validation_afni_data"
DESIGN_MATRIX = AFNI_DATA_DIR / "X.xmat.1D"
INPUT_FILES = [
    AFNI_DATA_DIR / "small_test_r01.nii.gz",
    AFNI_DATA_DIR / "small_test_r02.nii.gz",
]

# AFNI reference outputs
AFNI_OLS_BUCKET = AFNI_DATA_DIR / "afni_OLS.nii.gz"
AFNI_REML_BUCKET = AFNI_DATA_DIR / "afni_REML.nii.gz"
AFNI_RVAR = AFNI_DATA_DIR / "afni_REMLvar.nii.gz"


@pytest.fixture
def afni_reference_data():
    """Load AFNI reference outputs."""
    assert AFNI_DATA_DIR.exists(), f"AFNI data dir not found: {AFNI_DATA_DIR}"
    assert DESIGN_MATRIX.exists(), f"Design matrix not found: {DESIGN_MATRIX}"
    for f in INPUT_FILES:
        assert f.exists(), f"Input file not found: {f}"

    return {
        "ols": nib.load(AFNI_OLS_BUCKET),
        "reml": nib.load(AFNI_REML_BUCKET),
        "rvar": nib.load(AFNI_RVAR),
    }


def test_ols_matches_afni(afni_reference_data, tmp_path):
    """Test that OLS results match AFNI 3dDeconvolve."""
    # Run our OLS analysis
    results, design_info = analyze_from_design_matrix(
        fmri_data=INPUT_FILES,
        design_matrix_file=DESIGN_MATRIX,
        method="ols",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Get stimulus labels
    stim_labels = design_info.get("stim_labels", ["movie", "prompt"])
    glt_labels = design_info.get("glt_labels", [])

    # Write our bucket file
    our_bucket_path = tmp_path / "our_OLS.nii.gz"
    write_afni_bucket(
        results,
        our_bucket_path,
        condition_names=stim_labels,
        contrast_names=glt_labels,
        contrast_results=None,  # GLTs already in results
        output_format="nifti_gz",
    )

    # Load both datasets
    afni_data = afni_reference_data["ols"].get_fdata()
    our_data = nib.load(our_bucket_path).get_fdata()

    # AFNI sub-brik order:
    # [0] Full_Fstat
    # [1] movie#0_Coef, [2] movie#0_Tstat
    # [3] prompt#0_Coef, [4] prompt#0_Tstat
    # [5] movieVprompt_GLT#0_Coef, [6] movieVprompt_GLT#0_Tstat

    # Our sub-brik order:
    # [0] Full_Fstat
    # [1] movie#0_Coef, [2] movie#0_Tstat
    # [3] prompt#0_Coef, [4] prompt#0_Tstat
    # [5] GLT#0_Coef, [6] GLT#0_Tstat

    print("\n=== OLS Comparison ===")

    # Compare Full F-stat
    afni_fstat = afni_data[..., 0, 0]
    our_fstat = our_data[..., 0]
    _compare_maps("Full F-stat", afni_fstat, our_fstat, rtol=0.01, atol=1e-4)  # 1% relative tolerance

    # Compare movie coef and tstat
    afni_movie_coef = afni_data[..., 0, 1]
    our_movie_coef = our_data[..., 1]
    _compare_maps("movie Coef", afni_movie_coef, our_movie_coef, rtol=0.01, atol=1e-4)

    afni_movie_tstat = afni_data[..., 0, 2]
    our_movie_tstat = our_data[..., 2]
    _compare_maps("movie T-stat", afni_movie_tstat, our_movie_tstat, rtol=0.01, atol=1e-4)

    # Compare prompt coef and tstat
    afni_prompt_coef = afni_data[..., 0, 3]
    our_prompt_coef = our_data[..., 3]
    _compare_maps("prompt Coef", afni_prompt_coef, our_prompt_coef, rtol=0.01, atol=1e-4)

    afni_prompt_tstat = afni_data[..., 0, 4]
    our_prompt_tstat = our_data[..., 4]
    _compare_maps("prompt T-stat", afni_prompt_tstat, our_prompt_tstat, rtol=0.01, atol=1e-4)

    # TODO: Compare GLT coef and tstat
    # GLTs are computed but not currently written to bucket
    # Need to fix GLT attachment to results
    # afni_glt_coef = afni_data[..., 0, 5]
    # our_glt_coef = our_data[..., 5]
    # _compare_maps("GLT Coef", afni_glt_coef, our_glt_coef, rtol=0.01, atol=1e-4)
    #
    # afni_glt_tstat = afni_data[..., 0, 6]
    # our_glt_tstat = our_data[..., 6]
    # _compare_maps("GLT T-stat", afni_glt_tstat, our_glt_tstat, rtol=0.01, atol=1e-4)


def test_reml_matches_afni(afni_reference_data, tmp_path):
    """Test that REML/ARMA results match AFNI 3dREMLfit."""
    # Run our ARMA analysis with default grid (matches AFNI -Grid 3)
    results, design_info = analyze_from_design_matrix(
        fmri_data=INPUT_FILES,
        design_matrix_file=DESIGN_MATRIX,
        method="arma11",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Get stimulus labels
    stim_labels = design_info.get("stim_labels", ["movie", "prompt"])
    glt_labels = design_info.get("glt_labels", [])

    # Write our bucket file
    our_bucket_path = tmp_path / "our_REML.nii.gz"
    write_afni_bucket(
        results,
        our_bucket_path,
        condition_names=stim_labels,
        contrast_names=glt_labels,
        contrast_results=None,
        output_format="nifti_gz",
    )

    # Load both datasets
    afni_data = afni_reference_data["reml"].get_fdata()
    our_data = nib.load(our_bucket_path).get_fdata()

    print("\n=== REML Comparison ===")

    # Compare Full F-stat
    afni_fstat = afni_data[..., 0, 0]
    our_fstat = our_data[..., 0]
    _compare_maps("Full F-stat", afni_fstat, our_fstat, rtol=1e-3, atol=1e-2)

    # Compare movie coef and tstat
    afni_movie_coef = afni_data[..., 0, 1]
    our_movie_coef = our_data[..., 1]
    _compare_maps("movie Coef", afni_movie_coef, our_movie_coef, rtol=1e-3, atol=1e-2)

    afni_movie_tstat = afni_data[..., 0, 2]
    our_movie_tstat = our_data[..., 2]
    _compare_maps("movie T-stat", afni_movie_tstat, our_movie_tstat, rtol=1e-3, atol=1e-2)

    # Compare prompt coef and tstat
    afni_prompt_coef = afni_data[..., 0, 3]
    our_prompt_coef = our_data[..., 3]
    _compare_maps("prompt Coef", afni_prompt_coef, our_prompt_coef, rtol=1e-3, atol=1e-2)

    afni_prompt_tstat = afni_data[..., 0, 4]
    our_prompt_tstat = our_data[..., 4]
    _compare_maps("prompt T-stat", afni_prompt_tstat, our_prompt_tstat, rtol=1e-3, atol=1e-2)

    # Compare GLT coef and tstat
    afni_glt_coef = afni_data[..., 0, 5]
    our_glt_coef = our_data[..., 5]
    _compare_maps("GLT Coef", afni_glt_coef, our_glt_coef, rtol=1e-3, atol=1e-2)

    afni_glt_tstat = afni_data[..., 0, 6]
    our_glt_tstat = our_data[..., 6]
    _compare_maps("GLT T-stat", afni_glt_tstat, our_glt_tstat, rtol=1e-3, atol=1e-2)


def test_arma_params_match_afni(afni_reference_data, tmp_path):
    """Test that ARMA parameters match AFNI Rvar output."""
    # Run our ARMA analysis
    results, design_info = analyze_from_design_matrix(
        fmri_data=INPUT_FILES,
        design_matrix_file=DESIGN_MATRIX,
        method="arma11",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Get ARMA parameters (n_voxels, 3) -> (a, b, lambda)
    assert hasattr(results, "arma_params"), "Results missing ARMA parameters"
    our_params = results.arma_params  # (1000, 3)

    # Reshape to 3D volume
    our_a = our_params[:, 0].reshape(10, 10, 10)
    our_b = our_params[:, 1].reshape(10, 10, 10)
    our_lam = our_params[:, 2].reshape(10, 10, 10)

    # Load AFNI Rvar
    afni_rvar = afni_reference_data["rvar"].get_fdata()

    # AFNI Rvar sub-briks:
    # [0] a, [1] b, [2] lam, [3] StDev, [4] -LogLik, [5] LjungBox
    afni_a = afni_rvar[..., 0, 0]
    afni_b = afni_rvar[..., 0, 1]
    afni_lam = afni_rvar[..., 0, 2]

    print("\n=== ARMA Parameters Comparison ===")

    # Compare parameters
    # Note: ARMA grid search may find slightly different optima
    # Use relaxed tolerance
    _compare_maps("a parameter", afni_a, our_a, rtol=0.05, atol=0.01)
    _compare_maps("b parameter", afni_b, our_b, rtol=0.05, atol=0.01)
    _compare_maps("lambda", afni_lam, our_lam, rtol=0.05, atol=0.01)


def _compare_maps(name, afni_map, our_map, rtol=1e-4, atol=1e-6):
    """
    Compare two 3D maps and print statistics.

    Parameters
    ----------
    name : str
        Name of the map being compared
    afni_map : ndarray
        AFNI reference map
    our_map : ndarray
        Our implementation's map
    rtol : float
        Relative tolerance for np.allclose
    atol : float
        Absolute tolerance for np.allclose
    """
    # Flatten for easier comparison
    afni_flat = afni_map.ravel()
    our_flat = our_map.ravel()

    # Calculate differences
    diff = our_flat - afni_flat
    abs_diff = np.abs(diff)
    rel_diff = abs_diff / (np.abs(afni_flat) + 1e-10)

    # Statistics
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    max_abs_diff = np.max(abs_diff)
    max_rel_diff = np.max(rel_diff)

    # Correlation
    valid_mask = ~(np.isnan(afni_flat) | np.isnan(our_flat))
    if valid_mask.sum() > 0:
        corr = np.corrcoef(afni_flat[valid_mask], our_flat[valid_mask])[0, 1]
    else:
        corr = np.nan

    # Check if close
    is_close = np.allclose(afni_flat, our_flat, rtol=rtol, atol=atol, equal_nan=True)

    # Print results
    status = "✓ PASS" if is_close else "✗ FAIL"
    print(f"\n{name}: {status}")
    print(f"  Mean diff: {mean_diff:.6f} ± {std_diff:.6f}")
    print(f"  Max abs diff: {max_abs_diff:.6f}")
    print(f"  Max rel diff: {max_rel_diff:.2%}")
    print(f"  Correlation: {corr:.6f}")

    # Assert
    assert is_close, f"{name} does not match AFNI within tolerance (rtol={rtol}, atol={atol})"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
