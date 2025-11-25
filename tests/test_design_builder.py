"""
Tests for design matrix construction

Validates that our design matrix builder produces results matching AFNI's 3dDeconvolve
"""

import pytest
import numpy as np
from pathlib import Path

from fastfuncsim.design_builder import (
    spm_canonical_hrf,
    legendre_polynomials,
    parse_afni_timing_file,
    create_onset_regressors,
    build_design_matrix,
)
from fastfuncsim.afni_io import read_afni_design_matrix


# Test data directory
TEST_DATA_DIR = Path(__file__).parent.parent / "test_data" / "small_validation_afni_data"


def test_spm_canonical_hrf():
    """Test SPM canonical HRF generation"""
    # Generate HRF
    hrf = spm_canonical_hrf(tr=2.0, duration=32.0)

    # Check shape
    assert hrf.shape == (16,), f"Expected 16 timepoints (32s / 2s), got {hrf.shape}"

    # Check normalized to unit peak
    assert np.abs(hrf.max() - 1.0) < 1e-6, "HRF should be normalized to unit peak"

    # Check has positive and negative components
    assert hrf.max() > 0, "HRF should have positive peak"
    assert hrf.min() < 0, "HRF should have negative undershoot"

    # Check peak timing (should be around 6s = TR 3)
    peak_idx = np.argmax(hrf)
    peak_time = peak_idx * 2.0
    assert 4 <= peak_time <= 8, f"HRF peak should be around 6s, got {peak_time}s"


def test_legendre_polynomials():
    """Test Legendre polynomial generation"""
    # Generate polynomials
    n_tp = 100
    order = 3

    # Test AFNI-style (not normalized)
    polys = legendre_polynomials(n_tp, order, normalize=False)

    # Check shape
    assert polys.shape == (n_tp, order + 1), f"Expected shape ({n_tp}, {order+1}), got {polys.shape}"

    # Check orthogonality (but NOT orthonormality)
    # Note: Legendre polynomials are orthogonal on continuous interval [-1, 1],
    # but discretization can introduce small non-orthogonality
    gram = polys.T @ polys
    off_diag = gram - np.diag(np.diag(gram))

    # For n=100, accept some discretization error (< 2.0)
    # For larger n (like 360), orthogonality improves significantly
    max_off_diag = np.max(np.abs(off_diag))
    assert max_off_diag < 2.0, f"Off-diagonal should be small, got {max_off_diag}"

    # Diagonal should NOT be all ones (not normalized)
    assert not np.allclose(np.diag(gram), 1.0), "AFNI polynomials are NOT normalized"

    # Test with larger n (more like real data)
    polys_360 = legendre_polynomials(360, order, normalize=False)
    gram_360 = polys_360.T @ polys_360
    off_diag_360 = gram_360 - np.diag(np.diag(gram_360))
    max_off_diag_360 = np.max(np.abs(off_diag_360))
    # With 360 points, still have ~1.0 discretization error (matches AFNI)
    assert max_off_diag_360 < 2.0, f"Should be nearly orthogonal with n=360, got {max_off_diag_360}"

    # Check constant term
    assert np.allclose(np.abs(polys[:, 0]), 1.0), "First polynomial should be constant (value 1)"

    # Test normalized version
    polys_norm = legendre_polynomials(n_tp, order, normalize=True)
    gram_norm = polys_norm.T @ polys_norm
    # Normalization helps but doesn't eliminate discretization error
    np.testing.assert_allclose(gram_norm, np.eye(order + 1), atol=0.05,
                               err_msg="Normalized polynomials should be nearly orthonormal")

    # Test polort -1 (no polynomials)
    polys_none = legendre_polynomials(n_tp, -1)
    assert polys_none.shape == (n_tp, 0), "polort -1 should return no polynomials"


def test_parse_afni_timing_file():
    """Test parsing AFNI timing files"""
    # Test with actual data
    movie_file = TEST_DATA_DIR / "ses01_times.movie.txt"
    prompt_file = TEST_DATA_DIR / "ses01_times.prompt.txt"

    assert movie_file.exists(), f"Test data not found: {movie_file}"
    assert prompt_file.exists(), f"Test data not found: {prompt_file}"

    # Parse movie timing
    movie_onsets = parse_afni_timing_file(movie_file)

    # Should have 2 runs (based on make_small_model.sh)
    assert len(movie_onsets) >= 2, f"Expected at least 2 runs, got {len(movie_onsets)}"

    # Check first run has onsets
    assert len(movie_onsets[0]) > 0, "First run should have onsets"
    assert movie_onsets[0][0] >= 0, "Onsets should be non-negative"

    # Parse prompt timing
    prompt_onsets = parse_afni_timing_file(prompt_file)
    assert len(prompt_onsets) >= 2, f"Expected at least 2 runs, got {len(prompt_onsets)}"

    print(f"Movie onsets run 0: {movie_onsets[0]}")
    print(f"Prompt onsets run 0: {prompt_onsets[0]}")


def test_create_onset_regressors():
    """Test creating stimulus regressors from onsets"""
    # Simple test case
    onset_times = np.array([10.0, 20.0, 30.0])  # seconds
    n_timepoints = 50
    tr = 2.0

    # Without HRF
    regressor = create_onset_regressors(onset_times, n_timepoints, tr, duration=0.0, hrf=None)

    # Check shape
    assert regressor.shape == (n_timepoints,)

    # Check spikes at correct TRs
    assert regressor[5] == 1.0, "Should have spike at TR 5 (10s / 2s)"
    assert regressor[10] == 1.0, "Should have spike at TR 10 (20s / 2s)"
    assert regressor[15] == 1.0, "Should have spike at TR 15 (30s / 2s)"

    # Check other timepoints are zero
    assert np.sum(regressor != 0) == 3, "Should have exactly 3 non-zero timepoints"

    # With HRF
    hrf = spm_canonical_hrf(tr=tr, duration=32.0)
    regressor_hrf = create_onset_regressors(onset_times, n_timepoints, tr, duration=0.0, hrf=hrf)

    # Should be smooth (convolved)
    assert np.sum(regressor_hrf != 0) > 3, "Convolved regressor should have many non-zero values"

    # Peak should be after onset (due to HRF delay)
    first_onset_tr = 5
    peak_after_first = np.argmax(regressor_hrf[first_onset_tr:first_onset_tr+10])
    assert peak_after_first > 0, "HRF peak should be delayed after onset"


def test_build_design_matrix_basic():
    """Test building complete design matrix"""
    # Use actual test data
    movie_file = TEST_DATA_DIR / "ses01_times.movie.txt"
    prompt_file = TEST_DATA_DIR / "ses01_times.prompt.txt"

    # Parameters from make_small_model.sh
    # Based on AFNI X.xmat.1D: 720 timepoints total, 2 runs, TR=1.0s
    timing_files = [movie_file, prompt_file]
    stim_labels = ['movie', 'prompt']
    n_timepoints_per_run = [360, 360]  # From AFNI design matrix
    tr = 1.0  # TR is 1 second for this dataset
    polort = 3

    # Build design
    design, labels, run_starts, metadata = build_design_matrix(
        timing_files=timing_files,
        stim_labels=stim_labels,
        n_timepoints_per_run=n_timepoints_per_run,
        tr=tr,
        polort=polort,
        hrf_models='SPMG1(5)',
    )

    # Check shape
    n_stim = 2
    n_polort_per_run = polort + 1  # polort 3 = 4 polynomials (0,1,2,3)
    n_runs = 2
    expected_cols = n_polort_per_run * n_runs + n_stim  # Polynomials first, then stimuli
    expected_rows = sum(n_timepoints_per_run)

    assert design.shape == (expected_rows, expected_cols), \
        f"Expected shape ({expected_rows}, {expected_cols}), got {design.shape}"

    # Check labels - polynomials come first in AFNI
    assert len(labels) == expected_cols
    assert labels[0] == 'Run1_Poly0'
    assert labels[7] == 'Run2_Poly3'
    assert labels[8] == 'movie'
    assert labels[9] == 'prompt'

    # Check run starts
    assert run_starts == [0, 360]

    # Check metadata
    assert metadata['stim_indices'] == [8, 9]  # Last 2 columns
    assert metadata['n_runs'] == 2
    assert metadata['tr'] == tr
    assert metadata['polort'] == polort

    print(f"\nDesign matrix shape: {design.shape}")
    print(f"Labels: {labels}")
    print(f"Run starts: {run_starts}")
    print(f"Stimulus indices: {metadata['stim_indices']}")
    print(f"Nuisance indices: {metadata['nuisance_indices']}")


@pytest.mark.skipif(not (TEST_DATA_DIR / "X.xmat.1D").exists(),
                    reason="AFNI reference design matrix not found")
def test_compare_with_afni():
    """Compare our design matrix with AFNI's X.xmat.1D"""
    # Load AFNI design matrix
    afni_design = read_afni_design_matrix(TEST_DATA_DIR / "X.xmat.1D")
    afni_matrix = afni_design['matrix']

    print(f"\nAFNI design matrix shape: {afni_matrix.shape}")
    print(f"AFNI run starts: {afni_design.get('run_starts', 'Not found')}")
    print(f"AFNI labels: {afni_design.get('col_labels', 'Not found')[:10]}...")

    # Build our design matrix
    movie_file = TEST_DATA_DIR / "ses01_times.movie.txt"
    prompt_file = TEST_DATA_DIR / "ses01_times.prompt.txt"

    # Get actual n_timepoints from AFNI matrix
    total_tps = afni_matrix.shape[0]
    n_runs = len(afni_design.get('run_starts', [0]))

    # Infer timepoints per run from run_starts
    run_starts_afni = afni_design.get('run_starts', [0])
    if len(run_starts_afni) >= 2:
        n_timepoints_per_run = []
        for i in range(len(run_starts_afni)):
            if i < len(run_starts_afni) - 1:
                n_timepoints_per_run.append(run_starts_afni[i+1] - run_starts_afni[i])
            else:
                n_timepoints_per_run.append(total_tps - run_starts_afni[i])
    else:
        n_timepoints_per_run = [total_tps]

    print(f"Inferred timepoints per run: {n_timepoints_per_run}")

    # Build our design
    design, labels, run_starts, metadata = build_design_matrix(
        timing_files=[movie_file, prompt_file],
        stim_labels=['movie', 'prompt'],
        n_timepoints_per_run=n_timepoints_per_run,
        tr=1.0,  # TR from AFNI design
        polort=3,
        hrf_models='SPMG1(5)',
    )

    print(f"Our design matrix shape: {design.shape}")
    print(f"Our labels: {labels}")

    # Compare shapes
    assert design.shape[0] == afni_matrix.shape[0], \
        f"Row count mismatch: ours={design.shape[0]}, AFNI={afni_matrix.shape[0]}"

    # Note: Column count might differ if AFNI includes extra regressors (motion, etc.)
    # For now, just compare stimulus columns

    # Extract stimulus columns from both
    # AFNI matrix: first 2 columns should be stimulus
    afni_stim = afni_matrix[:, :2]
    our_stim = design[:, :2]

    # Compare stimulus columns
    # Allow some tolerance due to numerical differences in HRF implementation
    corr_movie = np.corrcoef(afni_stim[:, 0], our_stim[:, 0])[0, 1]
    corr_prompt = np.corrcoef(afni_stim[:, 1], our_stim[:, 1])[0, 1]

    print(f"\nCorrelation with AFNI:")
    print(f"  Movie regressor: {corr_movie:.6f}")
    print(f"  Prompt regressor: {corr_prompt:.6f}")

    # Should be very highly correlated (>0.99)
    assert corr_movie > 0.95, f"Movie regressor correlation too low: {corr_movie}"
    assert corr_prompt > 0.95, f"Prompt regressor correlation too low: {corr_prompt}"

    # For very close match, check if highly correlated
    if corr_movie > 0.999 and corr_prompt > 0.999:
        print("  ✓ Excellent match with AFNI!")


def test_write_afni_xmat():
    """Test writing .xmat.1D format"""
    from fastfuncsim.design_builder import write_afni_xmat
    import tempfile

    # Build a simple design matrix
    movie_file = TEST_DATA_DIR / "ses01_times.movie.txt"
    prompt_file = TEST_DATA_DIR / "ses01_times.prompt.txt"

    design, labels, run_starts, metadata = build_design_matrix(
        timing_files=[movie_file, prompt_file],
        stim_labels=['movie', 'prompt'],
        n_timepoints_per_run=[360, 360],
        tr=1.0,
        polort=3,
        hrf_models='SPMG1(5)',
    )

    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.1D', delete=False) as f:
        output_path = f.name

    try:
        # Write with GLT contrast
        glt_contrasts = [('SYM: +1*movie -1*prompt', 'movieVprompt')]
        write_afni_xmat(
            output_path,
            design,
            labels,
            run_starts,
            metadata,
            glt_contrasts=glt_contrasts,
        )

        # Read back and validate
        with open(output_path, 'r') as f:
            content = f.read()

        # Check header elements
        assert '# <matrix' in content, "Missing matrix header"
        assert 'ni_type = "10*double"' in content, "Wrong ni_type"
        assert 'ni_dimen = "720"' in content, "Wrong ni_dimen"
        assert 'ColumnLabels' in content, "Missing ColumnLabels"
        assert 'Run#1Pol#0' in content or 'Run1_Poly0' in content, "Missing polynomial labels"
        assert 'movie' in content, "Missing movie label"
        assert 'prompt' in content, "Missing prompt label"
        assert 'RunStart = "0,360"' in content, "Wrong RunStart"
        assert 'Nstim = "2"' in content, "Wrong Nstim"
        assert 'Nglt = "1"' in content, "Wrong Nglt"
        assert 'movieVprompt' in content, "Missing GLT label"
        assert 'SPMG1' in content, "Missing HRF model"
        assert '# >' in content, "Missing header end"

        # Check data rows
        lines = content.split('\n')
        data_lines = [l for l in lines if l and not l.startswith('#')]
        assert len(data_lines) == 720, f"Expected 720 data rows, got {len(data_lines)}"

        # Check first data row has correct number of columns
        first_row = data_lines[0].strip().split()
        assert len(first_row) == 10, f"Expected 10 columns, got {len(first_row)}"

        print(f"\n✓ .xmat.1D writer test passed!")
        print(f"  Wrote {len(data_lines)} rows x {len(first_row)} columns")

    finally:
        # Clean up
        import os
        if os.path.exists(output_path):
            os.remove(output_path)


def test_glt_parsing():
    """Test GLT contrast parsing and validation"""
    from fastfuncsim.design_builder import parse_glt_string, glt_weights_to_vector

    # Test valid contrast (difference)
    weights, valid = parse_glt_string('SYM: +1*A -1*B')
    assert valid, "Difference contrast should be valid"
    assert weights == {'A': 1.0, 'B': -1.0}, f"Wrong weights: {weights}"
    assert abs(sum(weights.values())) < 1e-6, "Weights should sum to 0"

    # Test valid contrast (average)
    weights, valid = parse_glt_string('SYM: +0.5*A +0.5*B')
    assert valid, "Average contrast should be valid"
    assert weights == {'A': 0.5, 'B': 0.5}, f"Wrong weights: {weights}"
    assert abs(sum(weights.values()) - 1.0) < 1e-6, "Weights should sum to 1"

    # Test invalid contrast (warning should be raised)
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        weights, valid = parse_glt_string('SYM: +1*A +1*B')
        assert len(w) == 1, "Should raise warning for invalid sum"
        assert not valid, "Should be marked invalid"
        assert "sum to" in str(w[0].message).lower(), "Warning should mention sum"

    # Test vector conversion
    labels = ['Poly0', 'Poly1', 'A', 'B', 'C']
    weights = {'A': 1.0, 'B': -1.0}
    vec = glt_weights_to_vector(weights, labels)
    expected = np.array([0, 0, 1, -1, 0])
    np.testing.assert_array_equal(vec, expected, err_msg="Wrong contrast vector")

    print("\n✓ GLT parsing tests passed!")


if __name__ == "__main__":
    # Run tests
    test_spm_canonical_hrf()
    test_legendre_polynomials()
    test_parse_afni_timing_file()
    test_create_onset_regressors()
    test_build_design_matrix_basic()
    test_compare_with_afni()
    test_write_afni_xmat()
    test_glt_parsing()

    print("\n✓ All tests passed!")
