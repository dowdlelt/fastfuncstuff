"""Timing utilities for performance profiling."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, List, Optional

import torch


class TimingProfiler:
    """Context manager for hierarchical timing profiling."""

    def __init__(self, enabled: bool = True):
        """Create a timing profiler.

        Parameters
        ----------
        enabled : bool, default=True
            Whether profiling instrumentation is active.

        Returns
        -------
        None
            Initializes timing buffers and stack state.
        """
        self.enabled = enabled
        self.timings: Dict[str, List[float]] = {}
        self.stack: List[tuple] = []  # (name, start_time)

    @contextmanager
    def profile(self, name: str):
        """Time a code block."""
        if not self.enabled:
            yield
            return

        # Sync CUDA if available
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.time()
        self.stack.append((name, start))

        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = time.time() - start
            popped_name, _ = self.stack.pop()

            if popped_name not in self.timings:
                self.timings[popped_name] = []
            self.timings[popped_name].append(elapsed)

    def get_report(self, sort_by_total: bool = True) -> str:
        """Generate timing report."""
        if not self.timings:
            return "No timing data collected"

        lines = []
        lines.append("=" * 80)
        lines.append("TIMING PROFILE REPORT")
        lines.append("=" * 80)
        lines.append(f"{'Operation':<50} {'Calls':>8} {'Total':>10} {'Mean':>10} {'%':>6}")
        lines.append("-" * 80)

        # Calculate totals
        total_time = sum(sum(times) for times in self.timings.values())

        # Sort by total time or name
        items = list(self.timings.items())
        if sort_by_total:
            items.sort(key=lambda x: sum(x[1]), reverse=True)
        else:
            items.sort(key=lambda x: x[0])

        for name, times in items:
            n_calls = len(times)
            total = sum(times)
            mean = total / n_calls
            pct = 100 * total / total_time if total_time > 0 else 0

            lines.append(
                f"{name:<50} {n_calls:>8} {total:>9.3f}s {mean:>9.3f}s {pct:>5.1f}%"
            )

        lines.append("-" * 80)
        lines.append(f"{'TOTAL':<50} {'':<8} {total_time:>9.3f}s")
        lines.append("=" * 80)

        return "\n".join(lines)

    def reset(self):
        """Clear all timing data."""
        self.timings.clear()
        self.stack.clear()


# Global profiler instance
_global_profiler: Optional[TimingProfiler] = None


def get_profiler(enabled: bool = True) -> TimingProfiler:
    """Get or create global profiler.

    If the global profiler already exists, its ``enabled`` flag is updated
    to match the *enabled* argument so callers can re-enable profiling
    after it was previously disabled.
    """
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = TimingProfiler(enabled=enabled)
    else:
        _global_profiler.enabled = enabled
    return _global_profiler


@contextmanager
def profile_section(name: str, enabled: bool = True):
    """Context manager for profiling a named code section using global profiler.

    This is a convenience wrapper around TimingProfiler.profile() for the
    common case of using a global profiler instance.

    Parameters
    ----------
    name : str
        Name of the code section to profile
    enabled : bool
        Whether profiling is enabled

    Examples
    --------
    >>> with profile_section("section_name", enabled=True):
    ...     # code to time
    ...     pass
    """
    profiler = get_profiler(enabled=enabled)
    with profiler.profile(name):
        yield
