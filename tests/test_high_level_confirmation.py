"""
Confirmation tests for high-level fMRI analysis pipelines.
Verifies AFNI onset parsing, HRF library fitting, and cross-validation.
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from fastfuncsim.analysis import (
    analyze_from_onsets,
    analyze_with_cross_validation,
)
from fastfuncsim.hrf import get_hrf_library
from fastfuncsim.glm_core import fit_glm_hrf_library
from fastfuncsim.simulation import create_parametric_voxels

# Test data directory
TEST_DATA_DIR = (
    Path(__file__).parent.parent / "test_data" / "small_validation_afni_data"
)


@pytest.fixture
def simulated_data():
    """Create simple simulated data for testing."""
    n_voxels = 100
    n_timepoints = 200
    tr = 2.0

    # Create 2 simple stimulus regressors (spikes)
    stim_onsets = [
        np.array([10.0, 50.0, 90.0, 130.0, 170.0]),
        np.array([30.0, 70.0, 110.0, 150.0, 190.0]),
    ]

    # Create simple binary onset matrix (n_timepoints, n_conditions)
    onsets_matrix = torch.zeros((n_timepoints, 2))
    for i, onsets in enumerate(stim_onsets):
        for onset in onsets:
            onsets_matrix[int(onset / tr), i] = 1.0

    # Generate ground truth data using a standard HRF
    # Important: simulate_fmri_run expects (n_timepoints, n_conditions) onsets
    from fastfuncsim.hrf import get_canonical_hrf

    hrf = get_canonical_hrf(stim_duration=0, tr=tr)

    # Create random betas
    betas = torch.randn(n_voxels, 2)

    from fastfuncsim.simulation import simulate_fmri_run

    # matrix_size=(10, 10, 1) = 100 voxels
    data_4d = simulate_fmri_run(
        onsets=onsets_matrix,
        betas=betas,
        hrf=hrf,
        tr=tr,
        n_timepoints=n_timepoints,
        matrix_size=(10, 10, 1),
        noise_level=0.5,
        baseline=100.0,
        add_scanner_drift=False,
        device=onsets_matrix.device,
    )
    # data_4d is (10, 10, 1, 200) -> reshape to (100, 200)
    data = data_4d.view(-1, n_timepoints)

    return {
        "data": data,  # (n_voxels, n_timepoints)
        "onsets_matrix": onsets_matrix,
        "tr": tr,
        "n_timepoints": n_timepoints,
    }


def test_hrf_library_creation():
    """Confirm HRF library creation is functional and sane."""
    tr = 2.0
    stim_duration = 0.0
    n_hrfs = 10

    # Test library mode - always returns 20 HRFs (fixed library from file)
    lib_library = get_hrf_library(
        mode="library", stim_duration=stim_duration, tr=tr, n_hrfs=n_hrfs
    )
    assert lib_library.shape[0] == 20  # Library has fixed 20 HRFs
    assert lib_library.shape[1] > 0
    # At TR resolution, peaks may not hit exactly 1.0 due to sampling
    # Use lenient check - all should have reasonable positive values
    assert torch.all(lib_library.max(dim=1)[0] > 0.5), "HRFs should have positive peaks"

    # Test at microtime resolution - peaks should be closer to 1.0
    lib_microtime = get_hrf_library(
        mode="library",
        stim_duration=stim_duration,
        tr=tr,
        n_hrfs=n_hrfs,
        microtime_resolution=16,
    )
    # At higher resolution, peaks should be very close to 1.0
    assert torch.all(lib_microtime.max(dim=1)[0] > 0.95), (
        "Microtime HRFs should peak near 1.0"
    )

    # Test PIGHS library - respects n_hrfs (tests backwards compatibility with 'flobs' mode too)
    lib_pighs = get_hrf_library(
        mode="pighs", stim_duration=stim_duration, tr=tr, n_hrfs=n_hrfs
    )
    assert lib_pighs.shape[0] == n_hrfs

    # Also test that 'flobs' mode still works for backwards compatibility
    lib_flobs = get_hrf_library(
        mode="flobs", stim_duration=stim_duration, tr=tr, n_hrfs=n_hrfs
    )
    assert lib_flobs.shape[0] == n_hrfs

    # Check that they are different
    assert not torch.allclose(lib_library[0], lib_library[1]), (
        "HRFs in library should vary"
    )


def test_fit_glm_hrf_library_logic(simulated_data):
    """Confirm that fitting with an HRF library works and selects 'best' HRF."""
    data = simulated_data["data"]
    tr = simulated_data["tr"]
    onsets_matrix = simulated_data["onsets_matrix"]

    # Create HRF library - library always returns 20 HRFs from file
    hrf_library = get_hrf_library(mode="library", tr=tr)
    n_hrfs = hrf_library.shape[0]  # 20

    # Fit
    results, hrf_idx, r2_all = fit_glm_hrf_library(
        data=data, design=onsets_matrix, hrf_library=hrf_library, tr=tr
    )

    assert results.betas.shape == (data.shape[0], 2)
    assert r2_all.shape == (data.shape[0], n_hrfs)
    assert results.r2.shape == (data.shape[0],)

    # Check that it actually chose the best HRF (highest R2)
    # The final R2 in results should be the max of r2_all
    max_r2_all = r2_all.max(dim=1)[0]
    assert torch.allclose(results.r2, max_r2_all, atol=1e-5)


def test_analyze_from_onsets_sanity():
    """Confirm the high-level AFNI onset pipeline is sane."""
    if not TEST_DATA_DIR.exists():
        pytest.skip("Test data not found")

    movie_timing = TEST_DATA_DIR / "ses01_times.movie.txt"
    prompt_timing = TEST_DATA_DIR / "ses01_times.prompt.txt"
    fmri_file = TEST_DATA_DIR / "small_test_r01.nii.gz"

    # Run high-level analysis
    results = analyze_from_onsets(
        fmri_data=fmri_file,
        onset_files=[movie_timing, prompt_timing],
        stim_labels=["movie", "prompt"],
        tr=1.0,
        polort=2,
        stim_duration=5.0,  # Matches AFNI 'SPMG1(5)'
        hrf_mode="library",  # This tests library mode through analyze_from_onsets
        test_n_voxels=50,  # Fast mode
        verbose=False,
    )

    assert results.betas.shape[0] >= 50
    # Results should only contain task betas (separated from nuisance/polort)
    assert results.betas.shape[1] == 2
    assert results.r2.shape == (results.betas.shape[0],)
    assert torch.isfinite(results.betas).all()


def test_analyze_with_cross_validation_sanity():
    """Confirm cross-validation pipeline is sane."""
    if not TEST_DATA_DIR.exists():
        pytest.skip("Test data not found")

    fmri_files = [
        TEST_DATA_DIR / "small_test_r01.nii.gz",
        TEST_DATA_DIR / "small_test_r02.nii.gz",
    ]
    design_file = TEST_DATA_DIR / "X.xmat.1D"

    # Run CV
    results, design_info = analyze_with_cross_validation(
        fmri_data=fmri_files,
        design_matrix_file=design_file,
        cv_strategy=1,  # LORO
        test_n_voxels=50,
        verbose=False,
    )

    assert "r2_median" in results
    assert results["r2_median"].shape == (50,)
    assert results["n_splits"] == 2  # 2 runs LORO = 2 splits
    assert "run_starts" in design_info


def test_microtime_resolution(simulated_data):
    """Confirm sub-TR timing (microtime resolution) works correctly."""
    from fastfuncsim.afni_io import onsets_to_binary_matrix
    from fastfuncsim.design import convolve_hrf_microtime
    from fastfuncsim.hrf import get_canonical_hrf

    tr = simulated_data["tr"]
    n_timepoints = simulated_data["n_timepoints"]

    # Create onsets at sub-TR times (e.g., 0.5 seconds into a 2s TR)
    # These should be placed in different microtime bins
    onsets_per_condition = [
        [np.array([0.5, 10.5, 20.5])],  # Condition 1: mid-TR onsets
        [np.array([1.0, 11.0, 21.0])],  # Condition 2: different offset
    ]

    # Test with TR-locked (legacy) - onsets get rounded
    onsets_tr = onsets_to_binary_matrix(
        onsets_per_condition, n_timepoints, tr, microtime_resolution=1
    )
    assert onsets_tr.shape == (n_timepoints, 2)
    # At TR=2.0, onset at 0.5s rounds to TR 0, onset at 10.5s rounds to TR 5
    assert onsets_tr[0, 0] == 1  # 0.5s -> TR 0
    assert onsets_tr[5, 0] == 1  # 10.5s -> TR 5

    # Test with microtime resolution (default 16x)
    microtime_res = 16
    onsets_micro = onsets_to_binary_matrix(
        onsets_per_condition, n_timepoints, tr, microtime_resolution=microtime_res
    )
    assert onsets_micro.shape == (n_timepoints * microtime_res, 2)

    # Check that onsets are placed at correct microtime bins
    # 0.5s / (2.0s / 16) = 0.5 / 0.125 = 4 -> bin 4
    # 10.5s / 0.125 = 84 -> bin 84
    assert onsets_micro[4, 0] == 1, "Onset at 0.5s should be in bin 4"
    assert onsets_micro[84, 0] == 1, "Onset at 10.5s should be in bin 84"

    # Test microtime convolution produces correct output shape
    hrf = get_canonical_hrf(stim_duration=0, tr=tr)
    design = convolve_hrf_microtime(
        onsets_micro,
        hrf,
        n_timepoints,
        microtime_res,
        microtime_onset=microtime_res // 2 + 1,  # Middle of TR
    )
    assert design.shape == (n_timepoints, 2), (
        f"Expected ({n_timepoints}, 2), got {design.shape}"
    )

    # Verify convolved design has reasonable values (not all zeros)
    assert design.abs().sum() > 0, "Convolved design should have non-zero values"


def test_microtime_vs_tr_locked():
    """Confirm microtime produces different results than TR-locked for sub-TR onsets."""
    from fastfuncsim.afni_io import onsets_to_binary_matrix
    from fastfuncsim.design import convolve_hrf, convolve_hrf_microtime
    from fastfuncsim.hrf import get_canonical_hrf

    tr = 2.0
    n_timepoints = 100

    # Two onsets within the same TR: 0.2s and 0.8s (both round to TR 0 with TR-locking)
    # With microtime, they should produce subtly different temporal profiles
    onsets_early = [[np.array([0.2])]]  # Very early in TR 0
    onsets_late = [[np.array([0.8])]]  # Later in TR 0 (still rounds to TR 0)

    hrf = get_canonical_hrf(stim_duration=0, tr=tr)
    microtime_res = 16

    # Microtime designs should differ (different sub-TR placement)
    onsets_early_micro = onsets_to_binary_matrix(
        onsets_early, n_timepoints, tr, microtime_resolution=microtime_res
    )
    onsets_late_micro = onsets_to_binary_matrix(
        onsets_late, n_timepoints, tr, microtime_resolution=microtime_res
    )

    design_early = convolve_hrf_microtime(
        onsets_early_micro,
        hrf,
        n_timepoints,
        microtime_res,
        microtime_onset=microtime_res // 2 + 1,
    )
    design_late = convolve_hrf_microtime(
        onsets_late_micro,
        hrf,
        n_timepoints,
        microtime_res,
        microtime_onset=microtime_res // 2 + 1,
    )

    # Early and late onsets should produce different designs
    # (the temporal shift should be visible in the convolved signal)
    assert not torch.allclose(design_early, design_late, atol=1e-3), (
        "Microtime should differentiate early vs late sub-TR onsets"
    )

    # TR-locked designs would be identical (both round to TR 0)
    onsets_early_tr = onsets_to_binary_matrix(
        onsets_early, n_timepoints, tr, microtime_resolution=1
    )
    onsets_late_tr = onsets_to_binary_matrix(
        onsets_late, n_timepoints, tr, microtime_resolution=1
    )

    # Verify both round to TR 0
    assert onsets_early_tr[0, 0] == 1, "0.2s should round to TR 0"
    assert onsets_late_tr[0, 0] == 1, "0.8s should round to TR 0"

    design_early_tr = convolve_hrf(onsets_early_tr, hrf, n_timepoints)
    design_late_tr = convolve_hrf(onsets_late_tr, hrf, n_timepoints)

    # Both round to TR 0, so TR-locked designs should be identical
    assert torch.allclose(design_early_tr, design_late_tr, atol=1e-6), (
        "TR-locked should produce identical designs for sub-TR offsets that round to same TR"
    )
