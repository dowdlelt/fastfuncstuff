"""Re-anchoring: refit with the sampling footprint rebuilt at the fitted pose.

The cost samples match points only where the source sits at the start pose, so a
thin slab started a few mm off cannot see its true peak (ds004539 VASO slab: 0.37 vs
1.07 on a common yardstick). These tests pin the decision and the loop; the
real-data behaviour was measured on that slab (0.105 -> 0.57 -> 1.10 -> 1.20).
"""

import importlib

import numpy as np
import pytest
import torch

A = importlib.import_module("fastfuncstuff.processing.allineate")
CPU = torch.device("cpu")


def _shift(dx=0.0, dy=0.0, dz=0.0):
    m = torch.eye(4)
    m[:3, 3] = torch.tensor([dx, dy, dz])
    return m


def test_corner_motion_is_zero_for_identical_fits():
    assert A._corner_motion_mm(_shift(), _shift(), (4, 5, 6), (1.0, 1.0, 1.0)) == 0.0


def test_corner_motion_scales_with_voxel_size():
    # A base-voxel shift of 2 along x at 0.5 mm voxels is 1 mm, on every corner.
    moved = A._corner_motion_mm(_shift(), _shift(dx=2.0), (4, 5, 6), (0.5, 0.7, 0.9))
    assert moved == pytest.approx(1.0)


def test_corner_motion_sees_rotation_that_a_centroid_would_miss():
    # Rotation about the volume centre leaves its centroid still but swings corners.
    th = np.radians(10.0)
    c, s = np.cos(th), np.sin(th)
    rot = torch.eye(4)
    rot[:3, :3] = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    centre = torch.tensor([4.5, 4.5, 4.5])
    rot[:3, 3] = centre - rot[:3, :3] @ centre
    assert A._corner_motion_mm(_shift(), rot, (10, 10, 10), (1.0, 1.0, 1.0)) > 1.0


HDR = {"affine": np.eye(4)}


@pytest.mark.parametrize(
    ("mode", "coverage", "expected"),
    [
        ("off", 0.01, False),
        ("on", 0.9, True),
        ("auto", 0.02, True),
        ("auto", 0.9, False),
    ],
)
def test_reanchor_decision(mode, coverage, expected):
    cfg = A.AffineAlignConfig(reanchor=mode)
    assert A._reanchor_wanted(cfg, {"source_coverage": coverage}, HDR, HDR) is expected


def test_reanchor_needs_headers_to_hand_the_pose_on():
    cfg = A.AffineAlignConfig(reanchor="on")
    assert not A._reanchor_wanted(cfg, {"source_coverage": 0.01}, None, HDR)


def _run(monkeypatch, shifts, **cfg_kw):
    """Run allineate with _align mocked to return a scripted sequence of poses."""
    calls = []

    def fake_align(base, source, config, bh, sh, *a):
        i = len(calls)
        calls.append(config.init_matrix)
        a[-1]["source_coverage"] = 0.02  # the info dict
        return _shift(dx=shifts[min(i, len(shifts) - 1)]), source

    monkeypatch.setattr(A, "_align", fake_align)
    cfg = A.AffineAlignConfig(device="cpu", verb=0, **cfg_kw)
    base, src = torch.zeros(4, 4, 4), torch.zeros(4, 4, 4)
    matrix, _ = A.allineate(base, src, cfg, base_header=HDR, source_header=HDR)
    return calls, matrix


def test_loop_stops_once_the_source_stops_moving(monkeypatch):
    calls, matrix = _run(monkeypatch, [0.0, 8.0, 8.1, 8.1, 8.1])
    # first fit + pass 1 (moved 8) + pass 2 (moved 0.1 < 0.25 tolerance) -> stop
    assert len(calls) == 3
    assert calls[0] is None and calls[1] is not None  # later passes start from the fit
    assert float(matrix[0, 3]) == pytest.approx(8.1)


def test_loop_is_capped_when_it_never_settles(monkeypatch):
    calls, _ = _run(monkeypatch, [0, 10, 20, 30, 40, 50], reanchor_passes=2)
    assert len(calls) == 1 + 2


def test_loop_off_runs_once(monkeypatch):
    calls, _ = _run(monkeypatch, [0.0, 5.0], reanchor="off")
    assert len(calls) == 1
