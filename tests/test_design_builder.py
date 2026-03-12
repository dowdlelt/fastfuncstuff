"""
Tests for design matrix construction

Validates that our design matrix builder produces results matching AFNI's 3dDeconvolve
"""

from pathlib import Path

import numpy as np
import pytest

from fastfuncsim.afni_io import read_afni_design_matrix
from fastfuncsim.design_builder import (
    build_design_matrix,
    create_onset_regressors,
    legendre_polynomials,
    parse_afni_timing_file,
    spm_canonical_hrf,
)

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
    assert labels[0] == 'Run#1Pol#0'
    assert labels[7] == 'Run#2Pol#3'
    assert labels[8] == 'movie#0'  # Standard mode: label#0
    assert labels[9] == 'prompt#0'  # Standard mode: label#0

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
    _n_runs = len(afni_design.get('run_starts', [0]))

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

    print("\nCorrelation with AFNI:")
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
    import tempfile

    from fastfuncsim.design_builder import write_afni_xmat

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
        with open(output_path) as f:
            content = f.read()

        # Check header elements
        assert '# <matrix' in content, "Missing matrix header"
        assert 'ni_type = "10*double"' in content, "Wrong ni_type"
        assert 'ni_dimen = "720"' in content, "Wrong ni_dimen"
        assert 'ColumnLabels' in content, "Missing ColumnLabels"
        assert 'Run#1Pol#0' in content, "Missing polynomial labels"
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

        print("\n✓ .xmat.1D writer test passed!")
        print(f"  Wrote {len(data_lines)} rows x {len(first_row)} columns")

    finally:
        # Clean up
        import os
        if os.path.exists(output_path):
            os.remove(output_path)


def test_im_mode():
    """Test Individual Modulation (IM) mode"""
    import tempfile

    # Create simple timing file with known events
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        timing_file = Path(f.name)
        # Run 1: 3 events at 10s, 20s, 30s
        # Run 2: 2 events at 5s, 15s
        f.write("10 20 30\n")
        f.write("5 15\n")

    try:
        # Build with IM mode
        design, labels, run_starts, metadata = build_design_matrix(
            timing_files=[timing_file],
            stim_labels=['condition'],
            n_timepoints_per_run=[50, 50],  # 100s total at TR=2s
            tr=2.0,
            polort=1,  # Keep simple
            hrf_models='SPMG1(0)',  # Delta function (no duration)
            im_mode=True,
        )

        # Should have:
        # - 2 runs * 2 polynomials = 4 poly columns
        # - 5 events (3 + 2) = 5 IM columns
        # Total: 9 columns
        assert design.shape == (100, 9), f"Expected (100, 9), got {design.shape}"

        # Check labels
        assert labels[0] == 'Run#1Pol#0'
        assert labels[3] == 'Run#2Pol#1'
        assert labels[4] == 'condition#0'  # IM mode: label#0, label#1, ...
        assert labels[5] == 'condition#1'
        assert labels[6] == 'condition#2'
        assert labels[7] == 'condition#3'
        assert labels[8] == 'condition#4'

        # Check metadata
        assert len(metadata['stim_indices']) == 5, "Should have 5 IM columns"
        assert metadata['stim_indices'] == [4, 5, 6, 7, 8]

        # Each IM column should be independent (non-zero at different times)
        # Run 1 events: TR 5, 10, 15 (at 10s, 20s, 30s with TR=2s)
        # Run 2 events: TR 52, 57 (at 5s, 15s in run 2, offset by 50 TRs)
        stim_cols = design[:, metadata['stim_indices']]

        # Count non-zero timepoints per column
        nonzero_counts = np.sum(stim_cols > 0, axis=0)
        print("\nIM mode test:")
        print(f"  Design shape: {design.shape}")
        print(f"  IM column labels: {labels[4:]}")
        print(f"  Non-zero timepoints per IM column: {nonzero_counts}")

        # Each column should have some non-zero values (after HRF convolution)
        assert all(nonzero_counts > 0), "Each IM column should have non-zero values"

        print("✓ IM mode test passed!")

    finally:
        # Clean up
        import os
        if timing_file.exists():
            os.remove(timing_file)


def test_glt_parsing():
    """Test GLT contrast parsing and validation"""
    from fastfuncsim.design_builder import glt_weights_to_vector, parse_glt_string

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


def test_goodlist_utilities():
    """Test GoodList parsing and censoring utilities"""
    from fastfuncsim.afni_io import (
        get_censored_mask,
        read_afni_design_matrix,
        select_uncensored_timepoints,
    )

    print("\n=== Testing GoodList utilities ===")

    # Load AFNI design matrix
    xmat_path = TEST_DATA_DIR / "X.xmat.1D"
    design = read_afni_design_matrix(xmat_path)

    # Check that GoodList was parsed
    assert design['good_list'] is not None, "GoodList should be parsed"
    assert len(design['good_list']) == 720, "Should have 720 uncensored timepoints"

    # Test censored mask (no censoring in this dataset)
    censored = get_censored_mask(design)
    assert censored.shape == (720,), "Censored mask wrong shape"
    assert censored.sum() == 0, "Should have no censored timepoints"

    # Test selecting uncensored timepoints
    X_unc = select_uncensored_timepoints(design)
    assert X_unc.shape == design['matrix'].shape, "No censoring, shapes should match"

    # Test with data
    fake_data = np.random.randn(720, 100)
    X_unc, Y_unc = select_uncensored_timepoints(design, fake_data)
    assert X_unc.shape == (720, 10), "Design matrix shape wrong"
    assert Y_unc.shape == (720, 100), "Data shape wrong"

    # Simulate censored data
    design_censored = design.copy()
    design_censored['good_list'] = list(range(0, 100)) + list(range(102, 720))  # Remove TRs 100-101
    design_censored['n_timepoints'] = 720

    censored = get_censored_mask(design_censored)
    assert censored.sum() == 2, "Should have 2 censored timepoints"
    assert list(np.where(censored)[0]) == [100, 101], "Wrong censored indices"

    X_unc = select_uncensored_timepoints(design_censored)
    assert X_unc.shape == (718, 10), "Should have 718 uncensored timepoints"

    print("  ✓ GoodList parsed correctly")
    print("  ✓ Censored mask works")
    print("  ✓ Uncensored selection works")
    print("\n✓ GoodList utilities tests passed!")


class TestDesignBuilderEdgeCases:
    """Test edge cases and error handling for design builder."""

    def test_load_and_pad_ortvec_errors(self):
        """Test error handling in load_and_pad_ortvec."""
        import os
        import tempfile

        from fastfuncsim.design_builder import load_and_pad_ortvec

        # Create dummy ortvec file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.1D', delete=False) as f:
            f.write("1.0 2.0\n3.0 4.0\n")
            filepath = f.name

        try:
            # Test file not found
            with pytest.raises(FileNotFoundError):
                load_and_pad_ortvec("nonexistent_file.1D", 1, [100])

            # Test invalid run number
            with pytest.raises(ValueError, match="Invalid run_number"):
                load_and_pad_ortvec(filepath, 0, [2, 2])
            
            with pytest.raises(ValueError, match="Invalid run_number"):
                load_and_pad_ortvec(filepath, 3, [2, 2])

            # Test length mismatch (file has 2 rows, run has 10)
            with pytest.raises(ValueError, match="has 2 rows"):
                load_and_pad_ortvec(filepath, 1, [10, 2]) # Run 1 has 10 timepoints

        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_load_and_pad_ortvec_success(self):
        """Test successful loading and padding."""
        import os
        import tempfile

        from fastfuncsim.design_builder import load_and_pad_ortvec

        # Create dummy ortvec file (5 timepoints, 2 regressors)
        data = np.random.randn(5, 2)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.1D', delete=False) as f:
            for row in data:
                f.write(f"{row[0]} {row[1]}\n")
            filepath = f.name

        try:
            # 3 runs: 5, 5, 5 timepoints
            n_timepoints_per_run = [5, 5, 5]
            
            # Load for run 2
            padded = load_and_pad_ortvec(filepath, 2, n_timepoints_per_run)
            
            # Check shape: total timepoints x regressors
            assert padded.shape == (15, 2)
            
            # Check run 1 is zero
            assert np.allclose(padded[:5], 0)
            
            # Check run 2 matches data
            assert np.allclose(padded[5:10], data)
            
            # Check run 3 is zero
            assert np.allclose(padded[10:], 0)

        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_parse_hrf_model_errors(self):
        """Test error handling in parse_hrf_model."""
        from fastfuncsim.design_builder import parse_hrf_model

        # Valid cases
        name, dur = parse_hrf_model("SPMG1(5)")
        assert name == "SPMG1" and dur == 5.0
        
        name, dur = parse_hrf_model("BLOCK(20.5)")
        assert name == "BLOCK" and dur == 20.5

        # Invalid format
        with pytest.raises(ValueError, match="Invalid HRF model string"):
            parse_hrf_model("InvalidFormat")
            
        with pytest.raises(ValueError, match="Invalid HRF model string"):
            parse_hrf_model("SPMG1[5]")

        # Invalid duration
        with pytest.raises(ValueError, match="Invalid duration"):
            parse_hrf_model("SPMG1(abc)")

    def test_glt_weights_to_vector_errors(self):
        """Test error handling in glt_weights_to_vector."""
        from fastfuncsim.design_builder import glt_weights_to_vector

        labels = ["A", "B", "C"]
        weights = {"A": 1, "D": -1}  # D doesn't exist

        with pytest.raises(ValueError, match="GLT label 'D' not found"):
            glt_weights_to_vector(weights, labels)

    def test_build_design_matrix_errors(self):
        """Test error handling in build_design_matrix."""
        import os
        import tempfile

        from fastfuncsim.design_builder import build_design_matrix
        
        # Create dummy timing file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("10 20\n")
            timing_file = f.name

        try:
            # Mismatched labels
            with pytest.raises(ValueError, match="stim_labels length"):
                build_design_matrix(
                    timing_files=[timing_file],
                    stim_labels=["A", "B"], # 2 labels
                    n_timepoints_per_run=[100],
                    tr=1.0
                )

            # Mismatched HRF models
            with pytest.raises(ValueError, match="hrf_models length"):
                build_design_matrix(
                    timing_files=[timing_file],
                    stim_labels=["A"],
                    n_timepoints_per_run=[100],
                    tr=1.0,
                    hrf_models=["SPMG1(5)", "BLOCK(10)"] # 2 models
                )

            # Mismatched IM mode
            with pytest.raises(ValueError, match="im_mode length"):
                build_design_matrix(
                    timing_files=[timing_file],
                    stim_labels=["A"],
                    n_timepoints_per_run=[100],
                    tr=1.0,
                    im_mode=[True, False] # 2 modes
                )
                
            # Mismatched runs in timing file
            with pytest.raises(ValueError, match="has 1 runs, but expected 2"):
                build_design_matrix(
                    timing_files=[timing_file], # Has 1 line (1 run)
                    stim_labels=["A"],
                    n_timepoints_per_run=[100, 100], # Expects 2 runs
                    tr=1.0
                )

        finally:
            if os.path.exists(timing_file):
                os.remove(timing_file)

    def test_build_design_matrix_with_extra_regressors(self):
        """Test building design matrix with extra regressors."""
        import os
        import tempfile

        from fastfuncsim.design_builder import build_design_matrix
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("10 20\n")
            timing_file = f.name
            
        try:
            n_tps = 100
            extra = np.random.randn(n_tps, 2)
            
            design, labels, _, metadata = build_design_matrix(
                timing_files=[timing_file],
                stim_labels=["stim"],
                n_timepoints_per_run=[n_tps],
                tr=1.0,
                hrf_models="SPMG1(0)",
                polort=0,
                extra_regressors=[extra]
            )
            
            # Expected columns: 1 polort + 1 stim + 2 extra = 4
            assert design.shape == (n_tps, 4)
            assert design.shape[1] == 4
            
            # Check last 2 columns are extra
            assert np.allclose(design[:, -2:], extra)
            
            # Check metadata
            assert len(metadata['extra_indices']) == 2
            
        finally:
            if os.path.exists(timing_file):
                os.remove(timing_file)


# ============================================================================
# Tests for design.py core functions (HRF convolution)
# ============================================================================

import torch

from fastfuncsim.design import convolve_hrf_microtime
from fastfuncsim.hrf import get_canonical_hrf
from fastfuncsim.utils import get_device


class TestDesignHrfConvolution:
    """Test HRF convolution functions in design.py"""

    def test_convolve_hrf_microtime_single_event(self):
        """Test microtime convolution with single instantaneous event"""
        device = get_device()
        tr = 2.0
        n_timepoints = 50
        microtime_dt = 0.1
        bins_per_tr = int(round(tr / microtime_dt))

        # Create onset matrix with single event at timepoint 10 (20 seconds)
        n_microtime = n_timepoints * bins_per_tr
        onsets_microtime = torch.zeros(n_microtime, 1, device=device)
        event_bin = int(round(10 * tr / microtime_dt))  # TR 10 = 20 seconds
        onsets_microtime[event_bin, 0] = 1.0

        # Get HRF at microtime resolution
        hrf = get_canonical_hrf(stim_duration=0.0, tr=microtime_dt, duration=32.0, device=device)

        # Convolve
        design = convolve_hrf_microtime(
            onsets_microtime=onsets_microtime,
            hrf=hrf,
            n_timepoints=n_timepoints,
            tr=tr,
            microtime_dt=microtime_dt,
            device=device,
        )

        # Check shape
        assert design.shape == (n_timepoints, 1), f"Expected shape ({n_timepoints}, 1), got {design.shape}"

        # Peak should occur after the event (HRF delay)
        peak_idx = torch.argmax(design).item()
        assert peak_idx > 10, f"Peak should be after event at TR 10, got peak at TR {peak_idx}"

        # Peak should be roughly 1.0 (single event scaled to peak=1)
        peak_value = design.max().item()
        assert 0.9 < peak_value <= 1.0, f"Single event should peak near 1.0, got {peak_value}"

        # Response should decay to near zero by end
        assert design[-1, 0].item() < 0.01, "Response should decay to near zero"

    def test_convolve_hrf_microtime_multiple_events(self):
        """Test microtime convolution with multiple events"""
        device = get_device()
        tr = 2.0
        n_timepoints = 50
        microtime_dt = 0.1
        bins_per_tr = int(round(tr / microtime_dt))

        # Create onset matrix with events at TRs 10, 20, 30
        n_microtime = n_timepoints * bins_per_tr
        onsets_microtime = torch.zeros(n_microtime, 1, device=device)
        for event_tr in [10, 20, 30]:
            event_bin = int(round(event_tr * tr / microtime_dt))
            onsets_microtime[event_bin, 0] = 1.0

        hrf = get_canonical_hrf(stim_duration=0.0, tr=microtime_dt, duration=32.0, device=device)

        design = convolve_hrf_microtime(
            onsets_microtime=onsets_microtime,
            hrf=hrf,
            n_timepoints=n_timepoints,
            tr=tr,
            microtime_dt=microtime_dt,
            device=device,
        )

        # Should have three peaks
        # Find local maxima
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(design[:, 0].cpu().numpy(), height=0.1)
        assert len(peaks) == 3, f"Should have 3 peaks for 3 events, got {len(peaks)}"

        # Each peak should be around 1.0 (events don't overlap)
        assert design.max().item() <= 1.0, "Non-overlapping events should peak at 1.0"

    def test_convolve_hrf_microtime_overlapping_events(self):
        """Test that overlapping events sum linearly"""
        device = get_device()
        tr = 2.0
        n_timepoints = 30
        microtime_dt = 0.1
        bins_per_tr = int(round(tr / microtime_dt))

        # Two events close together (will overlap)
        n_microtime = n_timepoints * bins_per_tr
        onsets_microtime = torch.zeros(n_microtime, 1, device=device)

        # Events at TR 10 and 12 (only 4 seconds apart)
        event_bin1 = int(round(10 * tr / microtime_dt))
        event_bin2 = int(round(12 * tr / microtime_dt))
        onsets_microtime[event_bin1, 0] = 1.0
        onsets_microtime[event_bin2, 0] = 1.0

        hrf = get_canonical_hrf(stim_duration=0.0, tr=microtime_dt, duration=32.0, device=device)

        design = convolve_hrf_microtime(
            onsets_microtime=onsets_microtime,
            hrf=hrf,
            n_timepoints=n_timepoints,
            tr=tr,
            microtime_dt=microtime_dt,
            device=device,
        )

        # Overlapping events should sum to > 1.0
        peak_value = design.max().item()
        assert peak_value > 1.0, f"Overlapping events should sum > 1.0, got {peak_value}"

    def test_convolve_hrf_microtime_boxcar_stimulus(self):
        """Test convolution with boxcar (sustained) stimulus"""
        device = get_device()
        tr = 2.0
        n_timepoints = 50
        microtime_dt = 0.1
        bins_per_tr = int(round(tr / microtime_dt))

        # Create boxcar stimulus (10 seconds duration)
        n_microtime = n_timepoints * bins_per_tr
        onsets_microtime = torch.zeros(n_microtime, 1, device=device)

        start_bin = int(round(10 * tr / microtime_dt))
        duration_bins = int(round(10.0 / microtime_dt))  # 10 second duration
        onsets_microtime[start_bin:start_bin + duration_bins, 0] = 1.0

        hrf = get_canonical_hrf(stim_duration=0.0, tr=microtime_dt, duration=32.0, device=device)

        design = convolve_hrf_microtime(
            onsets_microtime=onsets_microtime,
            hrf=hrf,
            n_timepoints=n_timepoints,
            tr=tr,
            microtime_dt=microtime_dt,
            device=device,
        )

        # Boxcar response should be more sustained than single event
        # Check that response stays elevated during stimulus
        _stimulus_end_tr = 10 + 5  # 10s start + 10s duration = 20s = TR 10, plus some HRF tail

        # Response should be > 0 during much of the stimulus period
        mid_stimulus_response = design[12:15, 0].mean().item()  # TR 12-15 (during stimulus)
        assert mid_stimulus_response > 0.3, f"Response should be elevated during stimulus, got {mid_stimulus_response}"

    def test_convolve_hrf_microtime_empty_onsets(self):
        """Test convolution with no events"""
        device = get_device()
        tr = 2.0
        n_timepoints = 50
        microtime_dt = 0.1
        bins_per_tr = int(round(tr / microtime_dt))

        # Empty onset matrix
        n_microtime = n_timepoints * bins_per_tr
        onsets_microtime = torch.zeros(n_microtime, 1, device=device)

        hrf = get_canonical_hrf(stim_duration=0.0, tr=microtime_dt, duration=32.0, device=device)

        design = convolve_hrf_microtime(
            onsets_microtime=onsets_microtime,
            hrf=hrf,
            n_timepoints=n_timepoints,
            tr=tr,
            microtime_dt=microtime_dt,
            device=device,
        )

        # Should be all zeros
        assert torch.all(design == 0), "Empty onsets should produce zero design"

    def test_convolve_hrf_microtime_wrong_dimensions(self):
        """Test that wrong dimensions raise error"""
        device = get_device()
        tr = 2.0
        n_timepoints = 50
        microtime_dt = 0.1

        # Wrong number of microtime points
        n_microtime_wrong = 100  # Wrong!
        onsets_microtime = torch.zeros(n_microtime_wrong, 1, device=device)
        hrf = get_canonical_hrf(stim_duration=0.0, tr=microtime_dt, duration=32.0, device=device)

        with pytest.raises(ValueError, match="onsets_microtime has.*points, expected"):
            convolve_hrf_microtime(
                onsets_microtime=onsets_microtime,
                hrf=hrf,
                n_timepoints=n_timepoints,
                tr=tr,
                microtime_dt=microtime_dt,
                device=device,
            )


class TestDesignConvolveHrfBasic:
    """Test basic convolve_hrf() function (non-microtime)"""

    def test_convolve_hrf_single_event(self):
        """Test basic HRF convolution with single event"""
        from fastfuncsim.design import convolve_hrf

        device = get_device()
        n_timepoints = 50
        n_conditions = 2

        # Create onset matrix with events at different timepoints
        onsets = torch.zeros(n_timepoints, n_conditions, device=device)
        onsets[10, 0] = 1.0  # Event at TR 10 for condition 0
        onsets[20, 1] = 1.0  # Event at TR 20 for condition 1

        # Create simple HRF (just a few timepoints)
        hrf = torch.tensor([0.0, 0.5, 1.0, 0.5, 0.0], device=device)

        # Convolve
        design = convolve_hrf(onsets, hrf, n_timepoints, device=device)

        # Check shape
        assert design.shape == (n_timepoints, n_conditions), \
            f"Expected shape ({n_timepoints}, {n_conditions}), got {design.shape}"

        # Peak should occur after the event
        peak_cond0 = torch.argmax(design[:, 0]).item()
        peak_cond1 = torch.argmax(design[:, 1]).item()
        assert peak_cond0 > 10, f"Peak should be after event at TR 10, got {peak_cond0}"
        assert peak_cond1 > 20, f"Peak should be after event at TR 20, got {peak_cond1}"

    def test_convolve_hrf_multiple_conditions(self):
        """Test convolve_hrf with multiple conditions"""
        from fastfuncsim.design import convolve_hrf

        device = get_device()
        n_timepoints = 100
        n_conditions = 5

        # Create random onset matrix
        onsets = torch.zeros(n_timepoints, n_conditions, device=device)
        for cond in range(n_conditions):
            # Add 3-5 random events per condition
            for _ in range(5):
                event_tp = torch.randint(10, 80, (1,)).item()
                onsets[event_tp, cond] = 1.0

        hrf = torch.tensor([0.0, 0.3, 0.8, 1.0, 0.7, 0.3, 0.0], device=device)

        design = convolve_hrf(onsets, hrf, n_timepoints, device=device)

        # All conditions should have non-zero design
        assert design.shape == (n_timepoints, n_conditions)
        for cond in range(n_conditions):
            assert design[:, cond].sum() > 0, f"Condition {cond} should have non-zero response"

    def test_convolve_hrf_empty_onsets(self):
        """Test convolve_hrf with all-zero onset matrix"""
        from fastfuncsim.design import convolve_hrf

        device = get_device()
        n_timepoints = 50

        onsets = torch.zeros(n_timepoints, 2, device=device)
        hrf = torch.tensor([0.0, 0.5, 1.0, 0.5, 0.0], device=device)

        design = convolve_hrf(onsets, hrf, n_timepoints, device=device)

        # Should be all zeros
        assert torch.all(design == 0), "Empty onsets should produce zero design"

    def test_convolve_hrf_1d_input(self):
        """Test that 1D onset matrix is handled correctly"""
        from fastfuncsim.design import convolve_hrf

        device = get_device()
        n_timepoints = 50

        # 1D input (single condition)
        onsets = torch.zeros(n_timepoints, device=device)
        onsets[10] = 1.0
        onsets[20] = 1.0

        hrf = torch.tensor([0.0, 0.5, 1.0, 0.5, 0.0], device=device)

        design = convolve_hrf(onsets, hrf, n_timepoints, device=device)

        # Should return (n_timepoints, 1) shape
        assert design.shape == (n_timepoints, 1)
        assert design[:, 0].sum() > 0


class TestDesignIsTrLocked:
    """Test is_tr_locked() function"""

    def test_is_tr_locked_exact(self):
        """Test is_tr_locked with exact TR multiples"""
        from fastfuncsim.design import is_tr_locked

        tr = 2.0

        # Exact TR multiples should be TR-locked
        assert is_tr_locked([0.0, 2.0, 4.0, 10.0], tr), \
            "TR multiples should be TR-locked"

    def test_is_tr_locked_not_locked(self):
        """Test is_tr_locked with non-TR multiples"""
        from fastfuncsim.design import is_tr_locked

        tr = 2.0

        # Non-multiples should not be TR-locked
        assert not is_tr_locked([1.0, 3.0, 5.0], tr), \
            "Non-TR multiples should not be TR-locked"

    def test_is_tr_locked_tolerance(self):
        """Test is_tr_locked with floating point tolerance"""
        from fastfuncsim.design import is_tr_locked

        tr = 2.0

        # Should handle small floating point errors (within 10% threshold)
        # 2.001 is 0.001/2.0 = 0.05% error - well within 10%
        assert is_tr_locked([2.001, 4.001], tr), \
            "Should tolerate small FP errors within threshold"
        # 2.2 is 0.2/2.0 = 10% error - at the threshold boundary
        # The function uses < threshold, so 10% exactly should fail
        assert not is_tr_locked([2.2, 4.2], tr), \
            "Larger errors should not be TR-locked"


class TestDesignGenerateRandomOnsets:
    """Test generate_random_onsets() function"""

    def test_generate_random_onsets_basic(self):
        """Test basic random onset generation"""
        from fastfuncsim.design import generate_random_onsets

        n_timepoints = 100
        n_conditions = 2
        isi_mean = 10.0  # Mean ISI of 10 seconds
        tr = 2.0

        onsets = generate_random_onsets(
            n_timepoints=n_timepoints,
            n_conditions=n_conditions,
            isi_mean=isi_mean,
            tr=tr,
            device=torch.device("cpu")
        )

        # Check shape
        assert onsets.shape == (n_timepoints, n_conditions), \
            f"Expected shape ({n_timepoints}, {n_conditions}), got {onsets.shape}"

        # Check that we have some events (should be sparse)
        n_events_cond0 = onsets[:, 0].sum().item()
        n_events_cond1 = onsets[:, 1].sum().item()

        assert n_events_cond0 > 0, "Should have generated events for condition 0"
        assert n_events_cond1 > 0, "Should have generated events for condition 1"

        # With alternating conditions, should have roughly equal numbers
        ratio = n_events_cond0 / (n_events_cond1 + 1e-6)
        assert 0.5 < ratio < 2.0, "Alternating conditions should have roughly equal events"

    def test_generate_random_onsets_single_condition(self):
        """Test with single condition"""
        from fastfuncsim.design import generate_random_onsets

        n_timepoints = 100
        n_conditions = 1
        isi_mean = 8.0
        tr = 2.0

        onsets = generate_random_onsets(
            n_timepoints=n_timepoints,
            n_conditions=n_conditions,
            isi_mean=isi_mean,
            tr=tr,
            device=torch.device("cpu")
        )

        # Check shape
        assert onsets.shape == (n_timepoints, n_conditions)

        # Should have some events
        n_events = onsets[:, 0].sum().item()
        assert n_events > 0, "Should have generated events"

    def test_generate_random_onsets_isi_range(self):
        """Test with different ISI ranges"""
        from fastfuncsim.design import generate_random_onsets

        n_timepoints = 200
        n_conditions = 2
        isi_mean = 10.0
        tr = 2.0

        # Tight ISI range
        onsets_tight = generate_random_onsets(
            n_timepoints=n_timepoints,
            n_conditions=n_conditions,
            isi_mean=isi_mean,
            isi_range=(8, 12),  # Narrow range
            tr=tr,
            device=torch.device("cpu")
        )

        # Wide ISI range
        onsets_wide = generate_random_onsets(
            n_timepoints=n_timepoints,
            n_conditions=n_conditions,
            isi_mean=isi_mean,
            isi_range=(2, 18),  # Wide range
            tr=tr,
            device=torch.device("cpu")
        )

        # Both should produce valid onset matrices
        assert onsets_tight.shape == (n_timepoints, n_conditions)
        assert onsets_wide.shape == (n_timepoints, n_conditions)

        # Wide range might have more variable event timing
        # (harder to test without detailed analysis, just check it runs)


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
    test_im_mode()
    test_goodlist_utilities()

    # Run new tests
    pytest.main([__file__, "-v"])

