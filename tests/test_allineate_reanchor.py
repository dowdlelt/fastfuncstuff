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
    ("mode", "overlap", "expected"),
    [
        ("off", 0.1, False),
        ("on", 1.0, True),
        ("auto", 0.5, True),  # the fit landed mostly outside what was sampled
        ("auto", 0.95, False),
    ],
)
def test_reanchor_decision(mode, overlap, expected):
    cfg = A.AffineAlignConfig(reanchor=mode)
    assert A._reanchor_wanted(cfg, {"footprint_overlap": overlap}, HDR, HDR) is expected


def test_reanchor_needs_headers_to_hand_the_pose_on():
    cfg = A.AffineAlignConfig(reanchor="on")
    assert not A._reanchor_wanted(cfg, {"footprint_overlap": 0.01}, None, HDR)


def _setup(footprint):
    """Just the fields _footprint_overlap reads, on a 10x6x6 base of unit weight."""
    from types import SimpleNamespace

    base = torch.zeros(10, 6, 6)
    return SimpleNamespace(base=base, start_footprint=footprint, brain_weight=torch.ones_like(base))


def _slab_matrix(z0):
    """Base->source voxel map putting a 3-slice source over base slices z0..z0+2."""
    return _shift(dz=-float(z0))


def test_overlap_is_one_when_the_fit_stays_on_what_was_sampled():
    start = A._compute_source_validity_mask((3, 6, 6), (10, 6, 6), _slab_matrix(2))
    assert A._footprint_overlap(_setup(start), (3, 6, 6), _slab_matrix(2)) == pytest.approx(1.0)


def test_overlap_drops_when_the_fit_moves_onto_unsampled_data():
    # Slab started at slices 2-4 and fitted to 4-6: one of its three slices was sampled.
    start = A._compute_source_validity_mask((3, 6, 6), (10, 6, 6), _slab_matrix(2))
    got = A._footprint_overlap(_setup(start), (3, 6, 6), _slab_matrix(4))
    assert got == pytest.approx(1.0 / 3.0)


def test_overlap_is_one_without_a_footprint():
    # A source covering the whole grid has nothing cut away, so nothing to re-anchor.
    assert A._footprint_overlap(_setup(None), (10, 6, 6), _shift()) == 1.0


def _run(monkeypatch, shifts, scores=None, **cfg_kw):
    """Run allineate with _align mocked to return a scripted sequence of poses.

    ``scores``: what the common-yardstick closure returns, as a function of the
    matrices it is asked about (None -> the closure is absent, as for non-blok costs).
    """
    calls, ranges = [], []

    def fake_align(base, source, config, bh, sh, *a):
        i = len(calls)
        calls.append(config.init_matrix)
        ranges.append(config.range_scale)
        a[-1]["footprint_overlap"] = 0.5  # the info dict
        if scores is not None:
            a[-1]["score"] = scores
        return _shift(dx=shifts[min(i, len(shifts) - 1)]), source

    monkeypatch.setattr(A, "_align", fake_align)
    cfg = A.AffineAlignConfig(device="cpu", verb=0, **cfg_kw)
    base, src = torch.zeros(4, 4, 4), torch.zeros(4, 4, 4)
    matrix, _ = A.allineate(base, src, cfg, base_header=HDR, source_header=HDR)
    return (calls, matrix, ranges) if scores is not None else (calls, matrix)


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


def _better_by_x(mats):
    """A yardstick that likes larger x-shifts, so every pass looks like a gain."""
    return [float(m[0, 3]) + 1.0 for m in mats]


def test_later_passes_search_a_shrinking_range(monkeypatch):
    # The user's range must not compound: each pass is centred on the last fit.
    _, _, ranges = _run(monkeypatch, [0, 10, 20, 30], scores=_better_by_x, range_scale=0.25)
    assert ranges == [0.25, 0.125, 0.0625, 0.03125]


def test_pass_is_dropped_when_the_common_yardstick_says_no_gain(monkeypatch):
    # Pose 8 scores worse than pose 0 on the common yardstick: keep the first fit.
    calls, matrix, _ = _run(
        monkeypatch, [0.0, 8.0, 16.0], scores=lambda mats: [1.0, 0.5], reanchor_passes=3
    )
    assert len(calls) == 2  # the first fit and one rejected pass, then it stops
    assert float(matrix[0, 3]) == pytest.approx(0.0)


def test_noise_gain_from_a_weak_fit_does_not_count(monkeypatch):
    # 0.017 -> 0.0175 is what a wrong-cost fit does between passes: still weak, drop it.
    calls, matrix, _ = _run(monkeypatch, [0.0, 8.0, 16.0], scores=lambda mats: [0.017, 0.0175])
    assert len(calls) == 2
    assert float(matrix[0, 3]) == pytest.approx(0.0)


def test_a_weak_fit_that_finds_the_peak_is_followed(monkeypatch):
    # A start far from the data: weak at first, then the footprint finds the real peak.
    seq = iter([[0.04, 0.5], [0.5, 0.9], [0.9, 0.95]])
    calls, matrix, _ = _run(
        monkeypatch, [0.0, 8.0, 12.0, 12.1], scores=lambda mats: next(seq), reanchor_passes=3
    )
    assert len(calls) == 4
    assert float(matrix[0, 3]) == pytest.approx(12.1)


def test_weak_fit_warning_names_the_lpc_contrast_trap():
    assert A._weak_fit_warning("lpa", 0.5) is None
    msg = A._weak_fit_warning("lpc", 0.02)
    assert "INVERSE contrast" in msg and "-cost lpa" in msg
    assert "INVERSE" not in A._weak_fit_warning("lpa", 0.02)
