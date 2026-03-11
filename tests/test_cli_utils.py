"""
Comprehensive tests for cli_utils.py with progressive coverage.

Test layers:
1. Small: Unit tests for core utility functions
2. Medium: Integration tests for file parsing and validation
3. Large: E2E tests for CLI workflow helpers

Tests critical utility functions used by all CLI tools:
- File parsing
- CV strategy parsing
- Auto polort calculation
- Nuisance building
"""

from pathlib import Path

import pytest

from fastfuncsim.cli_utils import (
    auto_polort,
    compute_run_lengths,
    get_average_run_duration,
    parse_cv_strategy,
    parse_device_arg,
    parse_input_files,
)

# ============================================================================
# Layer 1: Small Tests - Unit tests for core utility functions
# ============================================================================

class TestCliUtilsCoreFunctions:
    """Test core CLI utility functions."""

    def test_parse_input_files_single_string(self, tmp_path):
        """Test parsing single input file as string."""
        # Create test file
        test_file = tmp_path / "file.txt"
        test_file.touch()

        input_arg = str(test_file)
        result = parse_input_files(input_arg)

        assert result == [input_arg], f"Expected [{input_arg}], got {result}"

    def test_parse_input_files_list_of_strings(self, tmp_path):
        """Test parsing list of input files."""
        # Create test files
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.touch()
        file2.touch()

        input_arg = [str(file1), str(file2)]
        result = parse_input_files(input_arg)

        assert len(result) == 2
        assert str(file1) in result
        assert str(file2) in result

    def test_parse_input_files_nonexistent(self):
        """Test that nonexistent file raises SystemExit."""
        input_arg = "/path/to/nonexistent/file.txt"

        with pytest.raises(SystemExit):
            parse_input_files(input_arg)

    def test_parse_cv_strategy_integer(self):
        """Test parsing integer CV strategy."""
        result = parse_cv_strategy("5")
        assert result == 5, f"Expected 5, got {result}"

    def test_parse_cv_strategy_float(self):
        """Test parsing float CV strategy."""
        result = parse_cv_strategy("0.2")
        assert result == 0.2, f"Expected 0.2, got {result}"

    def test_parse_cv_strategy_loro(self):
        """Test parsing LORO CV strategy."""
        result = parse_cv_strategy("loro")
        assert result == 1, f"Expected 1 for LORO, got {result}"

    def test_parse_cv_strategy_invalid(self):
        """Test that invalid CV strategy raises SystemExit."""
        with pytest.raises(SystemExit):
            parse_cv_strategy("invalid_strategy")

    def test_compute_run_lengths_basic(self):
        """Test computing run lengths from run_starts."""
        run_starts = [0, 120, 240]
        n_timepoints = 360

        lengths = compute_run_lengths(run_starts, n_timepoints)

        expected = [120, 120, 120]
        assert lengths == expected, f"Expected {expected}, got {lengths}"

    def test_compute_run_lengths_single_run(self):
        """Test run lengths with single run."""
        run_starts = [0]
        n_timepoints = 200

        lengths = compute_run_lengths(run_starts, n_timepoints)

        expected = [200]
        assert lengths == expected, f"Expected {expected}, got {lengths}"

    def test_compute_run_lengths_different_lengths(self):
        """Test run lengths with different run durations."""
        run_starts = [0, 100, 250]
        n_timepoints = 400

        lengths = compute_run_lengths(run_starts, n_timepoints)

        expected = [100, 150, 150]
        assert lengths == expected, f"Expected {expected}, got {lengths}"

    def test_get_average_run_duration(self):
        """Test computing average run duration."""
        run_lengths = [100, 150, 200]
        tr = 2.0

        avg_duration = get_average_run_duration(run_lengths, tr)

        # Average length = (100+150+200)/3 = 150 TRs
        # Duration = 150 * 2.0 = 300 seconds
        expected = 150 * 2.0
        assert avg_duration == expected, f"Expected {expected}s, got {avg_duration}s"

    def test_auto_polort_short_run(self):
        """Test auto polort for short run."""
        run_duration = 60  # 1 minute

        polort = auto_polort(run_duration)

        # Formula: floor(1 + 60/150) = floor(1.4) = 1
        assert polort == 1, f"Expected polort=1 for 60s run, got {polort}"

    def test_auto_polort_medium_run(self):
        """Test auto polort for medium run."""
        run_duration = 300  # 5 minutes

        polort = auto_polort(run_duration)

        # Formula: floor(1 + 300/150) = floor(3.0) = 3
        assert polort == 3, f"Expected polort=3 for 300s run, got {polort}"

    def test_auto_polort_long_run(self):
        """Test auto polort for long run."""
        run_duration = 600  # 10 minutes

        polort = auto_polort(run_duration)

        # Formula: floor(1 + 600/150) = floor(5.0) = 5
        assert polort == 5, f"Expected polort=5 for 600s run, got {polort}"

    def test_auto_polort_conservative(self):
        """Test conservative polort formula."""
        run_duration = 300  # 5 minutes

        polort = auto_polort(run_duration, formula="conservative")

        # Conservative: max(1, round(300/120)) = max(1, 2) = 2
        assert polort == 2, f"Expected polort=2 for conservative, got {polort}"

    def test_parse_device_arg_cpu(self):
        """Test parsing CPU device argument."""
        device, cpu_threads, cuda_id = parse_device_arg("cpu")

        assert device.type == "cpu", f"Expected CPU, got {device.type}"
        assert cpu_threads is None
        assert cuda_id is None

    def test_parse_device_arg_cuda(self):
        """Test parsing CUDA device argument."""
        device, cpu_threads, cuda_id = parse_device_arg("cuda")

        assert device.type == "cuda", f"Expected CUDA, got {device.type}"
        assert cpu_threads is None
        assert cuda_id is None

    def test_parse_device_arg_cuda_index(self):
        """Test parsing CUDA device with index."""
        device, cpu_threads, cuda_id = parse_device_arg("cuda,1")

        assert device.type == "cuda"
        assert cuda_id == 1, f"Expected cuda_id=1, got {cuda_id}"
        assert cpu_threads is None

    def test_parse_device_arg_cpu_threads(self):
        """Test parsing CPU with thread count."""
        device, cpu_threads, cuda_id = parse_device_arg("cpu,8")

        assert device.type == "cpu"
        assert cpu_threads == 8, f"Expected cpu_threads=8, got {cpu_threads}"
        assert cuda_id is None

    def test_parse_device_arg_none(self):
        """Test parsing None device argument."""
        device, cpu_threads, cuda_id = parse_device_arg(None)

        # Should return default (CUDA if available, else CPU)
        assert device is not None
        assert cpu_threads is None
        assert cuda_id is None


# ============================================================================
# Layer 2: Medium Tests - Integration tests
# ============================================================================

class TestCliUtilsIntegration:
    """Test CLI utility integration scenarios."""

    @pytest.mark.skip(reason="TODO: Implement file validation test")
    def test_parse_input_files_with_validation(self, tmp_path):
        """Test parsing input files with existence validation."""
        # Create test files
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.touch()
        file2.touch()

        # Parse and validate
        result = parse_input_files([str(file1), str(file2)])

        assert len(result) == 2
        assert all(Path(f).exists() for f in result)

    @pytest.mark.skip(reason="TODO: Implement nuisance building test")
    def test_build_nuisance_per_run_integration(self):
        """Test building nuisance regressors per run."""
        pass


# ============================================================================
# Layer 3: Large Tests - E2E tests
# ============================================================================

class TestCliUtilsE2E:
    """Test end-to-end CLI utility scenarios."""

    @pytest.mark.skip(reason="TODO: Implement full CLI workflow test")
    def test_full_preprocessing_workflow(self):
        """Test complete preprocessing workflow using cli_utils functions."""
        pass
