"""Tests for source-localized phase regression (Vu & Gallant 2015)."""

from __future__ import annotations

import torch

from fastfuncstuff.phasereg import build_neighbor_index, phase_regress, select_phase_donor


def test_neighbor_index_face_adjacency():
    """Interior voxels see 6 distinct neighbours; faces fall back to self."""
    shape = (3, 3, 3)
    neigh = build_neighbor_index(shape, None, connectivity=6)
    assert neigh.shape == (27, 7)

    # Column 0 is always the voxel itself.
    assert torch.equal(neigh[:, 0], torch.arange(27))

    centre = 13  # (1,1,1) in C order
    assert len(set(neigh[centre].tolist())) == 7, "centre voxel has 6 real neighbours"

    corner = 0  # (0,0,0) — only 3 in-bounds neighbours
    assert len(set(neigh[corner].tolist())) == 4


def test_neighbor_index_respects_mask():
    """Masked-out neighbours are replaced by self, never by a wrong voxel."""
    shape = (3, 1, 1)
    mask = torch.tensor([True, False, True])
    neigh = build_neighbor_index(shape, mask, connectivity=6)
    assert neigh.shape == (2, 7)
    # Voxel 0 and voxel 1 (grid x=0 and x=2) are not adjacent to each other,
    # and the voxel between them is masked out, so neither may donate.
    assert set(neigh[0].tolist()) == {0}
    assert set(neigh[1].tolist()) == {1}


def test_donor_selection_finds_the_phase_carrying_neighbour():
    """A voxel whose own phase is noise borrows from the neighbour that isn't."""
    torch.manual_seed(0)
    n_tp = 120
    shape = (3, 1, 1)
    task = torch.sin(torch.linspace(0, 8 * torch.pi, n_tp))

    # Voxel 1 (the "vein"): strong task magnitude, pure-noise phase.
    # Voxel 2 (vein-adjacent): weak magnitude, clean task phase.
    mag = torch.randn(n_tp, 3) * 0.1
    pha = torch.randn(n_tp, 3) * 0.1
    mag[:, 1] += 5.0 * task
    pha[:, 2] += 5.0 * task

    neigh = build_neighbor_index(shape, None, connectivity=6)
    donor, donor_corr = select_phase_donor(mag, pha, neigh)

    assert donor[1].item() == 2, "vein voxel should borrow phase from voxel 2"
    assert abs(donor_corr[1].item()) > 0.9


def test_spr_suppresses_a_vein_that_standard_pr_misses():
    """End-to-end: sPR removes task variance standard PR leaves behind."""
    torch.manual_seed(1)
    n_tp = 150
    shape = (2, 1, 1)
    task = torch.sin(torch.linspace(0, 10 * torch.pi, n_tp))

    # Voxel 0 = vein: big task magnitude, phase carries none of it.
    # Voxel 1 = neighbour: phase carries the task.
    mag = 1000.0 + torch.randn(n_tp, 2) * 2.0
    pha = torch.randn(n_tp, 2) * 0.01
    mag[:, 0] += 50.0 * task
    pha[:, 1] += 0.5 * task

    mag_t = mag.T.contiguous()
    pha_t = pha.T.contiguous()
    common = dict(
        tr=2.0,
        task_removal="none",
        max_poly_degree=1,
        signal_thresh=0.0,
        regression="ols",
        shrink_mode="none",
        device=torch.device("cpu"),
    )

    res_pr = phase_regress(mag_t.clone(), pha_t.clone(), **common)
    res_spr = phase_regress(mag_t.clone(), pha_t.clone(), spr=True, volume_shape=shape, **common)

    assert res_spr.spr_donor is not None
    assert res_spr.spr_donor[0].item() == 1, "vein voxel must borrow neighbour phase"

    # How much task-locked variance survives in the vein voxel?
    task_c = task - task.mean()

    def task_amplitude(corrected: torch.Tensor) -> float:
        ts = corrected[0] - corrected[0].mean()
        return abs(float((ts * task_c).sum() / (task_c * task_c).sum()))

    left_by_pr = task_amplitude(res_pr.magnitude_corrected)
    left_by_spr = task_amplitude(res_spr.magnitude_corrected)

    assert left_by_pr > 40.0, "standard PR should be near-blind to this vein"
    assert left_by_spr < 0.2 * left_by_pr, (
        f"sPR should suppress the vein: PR left {left_by_pr:.1f}, sPR left {left_by_spr:.1f}"
    )


def test_spr_is_a_noop_when_own_phase_is_best():
    """sPR must reduce to standard PR when no neighbour helps."""
    torch.manual_seed(2)
    n_tp = 100
    shape = (2, 1, 1)
    task = torch.sin(torch.linspace(0, 6 * torch.pi, n_tp))

    mag = 1000.0 + torch.randn(n_tp, 2) * 2.0
    pha = torch.randn(n_tp, 2) * 0.01
    # Both voxels carry their own coupled magnitude+phase signal.
    mag += 30.0 * task.unsqueeze(1)
    pha += 0.4 * task.unsqueeze(1)

    common = dict(
        tr=2.0,
        task_removal="none",
        max_poly_degree=1,
        signal_thresh=0.0,
        regression="deming",
        device=torch.device("cpu"),
    )
    res = phase_regress(
        mag.T.contiguous(), pha.T.contiguous(), spr=True, volume_shape=shape, **common
    )
    assert res.spr_donor is not None
    assert torch.equal(res.spr_donor, torch.arange(2)), "each voxel should keep its own phase"
    assert res.spr_donor_offset is not None
    assert float(res.spr_donor_offset.max()) == 0.0


def test_coupling_pvalue_matches_scipy():
    """Our on-device t survival function must match scipy.stats.t."""
    pytest = __import__("pytest")
    stats = pytest.importorskip("scipy.stats")
    import numpy as np

    from fastfuncstuff.phasereg.veinmask import coupling_pvalue

    r = torch.tensor([0.0, 0.05, 0.1, 0.2, 0.35, 0.6, 0.9])
    for df in (20, 97, 300):
        got = coupling_pvalue(r, df).numpy()
        t = np.abs(r.numpy()) * np.sqrt(df) / np.sqrt(1 - r.numpy() ** 2)
        assert np.abs(got - 2 * stats.t.sf(t, df)).max() < 1e-6


def test_coupling_pvalue_sidak_correction():
    """Searching K donors must inflate p by the Sidak factor."""
    from fastfuncstuff.phasereg.veinmask import coupling_pvalue

    r = torch.tensor([0.2])
    p1 = coupling_pvalue(r, 100, n_candidates=1).item()
    p7 = coupling_pvalue(r, 100, n_candidates=7).item()
    assert p7 > p1
    # Tolerance is float32 round-trip, not algebra: p is returned in r.dtype.
    assert abs(p7 - (1 - (1 - p1) ** 7)) < 1e-6


def test_vein_mask_flags_vessel_not_grey_matter():
    """A phase-coupled voxel is excluded; plain task-responsive tissue is not."""
    torch.manual_seed(3)
    n_tp = 200
    shape = (3, 1, 1)
    task = (torch.sin(torch.linspace(0, 10 * torch.pi, n_tp)) > 0).float()

    mag = 1000.0 + torch.randn(n_tp, 3) * 5.0
    pha = torch.randn(n_tp, 3) * 0.02
    mag += 10.0 * task.unsqueeze(1)  # microvascular response everywhere
    # Voxel 1 is a vessel: magnitude AND phase track the task.
    mag[:, 1] += 60.0 * task
    pha[:, 1] += 0.30 * task

    res = phase_regress(
        mag.T.contiguous(),
        pha.T.contiguous(),
        tr=2.0,
        task_removal="none",
        max_poly_degree=1,
        signal_thresh=0.0,
        vein_mask=True,
        volume_shape=shape,
        device=torch.device("cpu"),
    )
    assert res.vein_exclude is not None
    assert res.coupling_r is not None
    assert bool(res.vein_exclude[1]), "phase-coupled vessel voxel must be flagged"
    assert not bool(res.vein_exclude[0]), "grey matter must not be flagged"
    assert not bool(res.vein_exclude[2]), "grey matter must not be flagged"
    # The corrected series must be untouched by asking for a mask.
    res_no = phase_regress(
        mag.T.contiguous(),
        pha.T.contiguous(),
        tr=2.0,
        task_removal="none",
        max_poly_degree=1,
        signal_thresh=0.0,
        device=torch.device("cpu"),
    )
    assert torch.allclose(res.magnitude_corrected, res_no.magnitude_corrected)
