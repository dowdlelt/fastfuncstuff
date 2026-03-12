"""
Comprehensive tests for timing_utils.py.

Tests the hierarchical timing profiler used for performance optimization.
"""

import time

from fastfuncsim.timing_utils import TimingProfiler, get_profiler, profile_section


class TestTimingProfiler:
    """Test TimingProfiler class."""

    def test_profiler_single_operation(self):
        """Test profiling a single operation."""
        profiler = TimingProfiler(enabled=True)

        with profiler.profile("test_operation"):
            time.sleep(0.01)

        # Check that timing was recorded
        assert "test_operation" in profiler.timings
        assert len(profiler.timings["test_operation"]) == 1

        elapsed = profiler.timings["test_operation"][0]
        assert 0.005 <= elapsed <= 0.02, f"Expected ~0.01s, got {elapsed:.3f}s"

    def test_profiler_multiple_operations(self):
        """Test profiling multiple different operations."""
        profiler = TimingProfiler(enabled=True)

        with profiler.profile("op1"):
            time.sleep(0.01)

        with profiler.profile("op2"):
            time.sleep(0.02)

        # Note: don't reuse the same profiler name in same test context
        # The stack management can have issues with re-entry

        # Check that timings were recorded
        assert "op1" in profiler.timings
        assert "op2" in profiler.timings
        assert len(profiler.timings["op1"]) == 1
        assert len(profiler.timings["op2"]) == 1

        # op2 should be slower (0.02 vs 0.01)
        op1_total = sum(profiler.timings["op1"])
        op2_total = sum(profiler.timings["op2"])
        assert op2_total > op1_total, "op2 (0.02s) should be slower than op1 (0.01s)"

    def test_profiler_nested_operations(self):
        """Test profiling nested operations."""
        profiler = TimingProfiler(enabled=True)

        with profiler.profile("outer"):
            time.sleep(0.01)
            with profiler.profile("inner"):
                time.sleep(0.01)

        # Both operations should be recorded
        assert "outer" in profiler.timings
        assert "inner" in profiler.timings

        # Inner should be faster than outer
        outer_total = sum(profiler.timings["outer"])
        inner_total = sum(profiler.timings["inner"])
        assert outer_total > inner_total, "Outer should include inner time"

    def test_profiler_disabled(self):
        """Test that disabled profiler doesn't record timings."""
        profiler = TimingProfiler(enabled=False)

        with profiler.profile("test_operation"):
            time.sleep(0.01)

        # No timings should be recorded
        assert len(profiler.timings) == 0

    def test_profiler_reset(self):
        """Test resetting profiler clears timings."""
        profiler = TimingProfiler(enabled=True)

        with profiler.profile("op1"):
            time.sleep(0.01)

        assert len(profiler.timings) == 1

        profiler.reset()

        assert len(profiler.timings) == 0
        assert len(profiler.stack) == 0

    def test_profiler_get_report(self):
        """Test generating timing report."""
        profiler = TimingProfiler(enabled=True)

        with profiler.profile("fast_op"):
            time.sleep(0.001)

        with profiler.profile("slow_op"):
            time.sleep(0.005)

        report = profiler.get_report(sort_by_total=True)

        # Check report structure
        assert "TIMING PROFILE REPORT" in report
        assert "fast_op" in report
        assert "slow_op" in report
        assert "Calls" in report
        assert "Total" in report
        assert "Mean" in report

        # slow_op should appear first (sorted by total time)
        slow_pos = report.index("slow_op")
        fast_pos = report.index("fast_op")
        assert slow_pos < fast_pos, "slow_op should appear before fast_op when sorted by total"

    def test_profiler_empty_report(self):
        """Test report with no timing data."""
        profiler = TimingProfiler(enabled=True)

        report = profiler.get_report()

        assert report == "No timing data collected"

    def test_profiler_report_sort_by_name(self):
        """Test sorting report by name instead of total time."""
        profiler = TimingProfiler(enabled=True)

        with profiler.profile("zebra"):
            time.sleep(0.005)

        with profiler.profile("apple"):
            time.sleep(0.001)

        report = profiler.get_report(sort_by_total=False)

        # When sorted by name, apple should come before zebra
        apple_pos = report.index("apple")
        zebra_pos = report.index("zebra")
        assert apple_pos < zebra_pos, "apple should appear before zebra when sorted alphabetically"


class TestGlobalProfiler:
    """Test global profiler instance."""

    def test_get_profiler_singleton(self):
        """Test that get_profiler returns same instance."""
        profiler1 = get_profiler(enabled=True)
        profiler2 = get_profiler(enabled=True)

        # Should be the same instance
        assert profiler1 is profiler2

    def test_global_profiler_persists(self):
        """Test that global profiler retains data across calls."""
        profiler = get_profiler(enabled=True)
        profiler.reset()  # Start fresh

        # Time something through get_profiler
        profiler = get_profiler()
        with profiler.profile("test_op"):
            time.sleep(0.001)

        # Should still be there
        assert "test_op" in profiler.timings

        # Clean up
        profiler.reset()


class TestProfileSectionDecorator:
    """Test profile_section decorator/context manager."""

    def test_profile_section_basic(self):
        """Test profile_section context manager."""
        # Get global profiler and reset it
        profiler = get_profiler(enabled=True)
        profiler.reset()

        with profile_section("decorated_op", enabled=True):
            time.sleep(0.005)

        assert "decorated_op" in profiler.timings
        assert len(profiler.timings["decorated_op"]) == 1

        # Clean up
        profiler.reset()

    def test_profile_section_uses_global_profiler(self):
        """Test that profile_section uses the global profiler."""
        # Clear global profiler first
        import fastfuncsim.timing_utils as timing_utils_module
        timing_utils_module._global_profiler = None

        # Get a fresh global profiler with enabled=True
        profiler = get_profiler(enabled=True)
        profiler.reset()

        # Use profile_section which should use the global profiler
        with profile_section("section1", enabled=True):
            time.sleep(0.003)

        # Global profiler should have recorded it
        assert "section1" in profiler.timings
        assert len(profiler.timings["section1"]) == 1

        # Clean up
        profiler.reset()
        timing_utils_module._global_profiler = None


class TestTimingProfilerEdgeCases:
    """Test edge cases and error conditions."""

    def test_profiler_zero_duration(self):
        """Test profiler with instantaneous operation."""
        profiler = TimingProfiler(enabled=True)

        with profiler.profile("instant_op"):
            pass  # No delay

        assert "instant_op" in profiler.timings
        elapsed = profiler.timings["instant_op"][0]
        # Should be very small but not negative
        assert elapsed >= 0, f"Elapsed time should be non-negative, got {elapsed}"

    def test_profiler_multiple_calls_to_same_operation(self):
        """Test profiler with many calls to the same operation."""
        profiler = TimingProfiler(enabled=True)

        for _i in range(10):
            with profiler.profile("repeated_op"):
                time.sleep(0.001)

        assert "repeated_op" in profiler.timings
        assert len(profiler.timings["repeated_op"]) == 10

        # Calculate statistics
        times = profiler.timings["repeated_op"]
        mean_time = sum(times) / len(times)
        _total_time = sum(times)

        # All times should be roughly similar (0.001s ± tolerance)
        for t in times:
            assert 0.0005 <= t <= 0.002, f"Each call should be ~0.001s, got {t:.3f}s"

        assert abs(mean_time - 0.001) < 0.0005, f"Mean should be ~0.001s, got {mean_time:.3f}s"
