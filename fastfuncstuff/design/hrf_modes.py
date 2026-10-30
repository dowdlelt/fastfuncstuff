"""-hrf_mode values shared by the HRF-library tools; torch-free, for parse time.

``library`` / ``pighs`` / ``blend`` / ``blend:N``.  ``blend`` always takes the whole
20-curve library and adds N PIGHS shapes (default :data:`BLEND_DEFAULT_EXTRA`), so
``-n_hrfs`` keeps meaning the pighs library's size and the two never interact.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

BLEND_DEFAULT_EXTRA = 20
HRF_LIBRARY_MODES = ("library", "pighs", "blend")


def parse_hrf_mode(value: str) -> tuple[str, int | None]:
    """``library`` / ``pighs`` / ``blend`` / ``blend:N`` -> (mode, N or None)."""
    mode, _, extra = str(value).strip().lower().partition(":")
    if not extra:
        return mode, None
    if mode != "blend":
        raise ValueError(f"only blend takes a count ({value!r}); use -n_hrfs for pighs")
    try:
        n = int(extra)
    except ValueError:
        raise ValueError(f"blend:N needs an integer N, not {extra!r}") from None
    if n < 1:
        raise ValueError("blend:N needs N >= 1")
    return mode, n


def hrf_mode_type(extra: tuple[str, ...] = ()) -> Callable[[str], str]:
    """argparse ``type`` for ``-hrf_mode``: library, pighs, blend, blend:N, plus ``extra``."""
    allowed = (*HRF_LIBRARY_MODES, *extra)

    def check(value: str) -> str:
        try:
            mode, _ = parse_hrf_mode(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from None
        if mode not in allowed:
            names = ", ".join((*allowed, "blend:N"))
            raise argparse.ArgumentTypeError(f"{value!r}: choose {names}")
        return str(value).strip().lower()

    return check
