"""Enumerate cardiac alias hypotheses without a TR-dependent step truncation."""

import math

import numpy as np


def complete_alias_offsets(times, phase, tr, avoid=(), band=(0.6, 2.0)):
    """Return every in-band sign/step pair, preserving distinct phase trajectories.

    A frequency-only dictionary key loses mirrored clocks with the same mean rate
    but opposite rate fluctuations. Keys therefore identify the transformation.
    """
    span = times[-1].mean() - times[0].mean()
    frequency = (phase[-1].mean() - phase[0].mean()) / (2 * np.pi * span)
    out = {}
    for sign in (1, -1):
        first = math.ceil((band[0] - sign * frequency) * tr - 1e-10)
        last = math.floor((band[1] - sign * frequency) * tr + 1e-10)
        for step in range(first, last + 1):
            rate = sign * frequency + step / tr
            if all(abs(rate - centre) >= width for centre, width in avoid):
                out[(sign, step)] = (sign, step)
    return out
