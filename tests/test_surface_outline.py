"""Filling a shape traced on a surface."""

from __future__ import annotations

import numpy as np

from fastfuncstuff.surface.outline import fill_outline, is_closed, trace_path
from tests.test_surface_ribbon import _icosphere


def _ring(d, z0, n=24):
    """Picks around a circle of latitude, sparse like a fast drag."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = np.sqrt(1 - z0**2)
    pts = np.c_[r * np.cos(ang), r * np.sin(ang), np.full(n, z0)]
    return [int(np.argmax(d @ p)) for p in pts]


def test_a_traced_ring_fills_the_cap_inside_it():
    d, f = _icosphere(4)
    assert is_closed(f)
    inside = fill_outline(100 * d, f, _ring(d, 0.7))
    cap = d[:, 2] > 0.75
    rim = (d[:, 2] > 0.6) & ~cap
    assert inside[cap].all()  # all of the cap
    assert not inside[d[:, 2] < 0.6].any()  # nothing of the rest
    assert inside[rim].mean() > 0.3  # and the traced band itself


def test_the_path_is_connected_along_edges():
    d, f = _icosphere(3)
    path = trace_path(100 * d, f, _ring(d, 0.0, n=8))
    edges = {tuple(sorted(e)) for e in np.r_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]].tolist()}
    steps = [tuple(sorted((int(a), int(b)))) for a, b in zip(path[:-1], path[1:], strict=True)]
    assert all(s in edges for s in steps) and path[0] == path[-1]


def test_a_flat_sheet_is_not_closed():
    f = np.array([[0, 1, 2], [1, 3, 2]])
    assert not is_closed(f)
