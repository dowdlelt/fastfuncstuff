"""Autobox must find the same box as 3dAutobox, and crop without moving anything.

The reference bounds below come from ``3dAutobox`` (AFNI 25.3.03) on the volume
built by :func:`_volume` — an ellipsoid plus one bright speck in the corner that
the clustering is supposed to reject.

The failure this guards against is subtle: a box that is one voxel tight on every
side still looks like a plausible crop, but it shaves the outer shell off every
dataset the pipeline touches afterwards.
"""

import numpy as np
import torch

from fastfuncstuff.processing.autobox import (
    autobox_bounds,
    crop_to_bounds,
    largest_cluster_6conn,
    pad_bounds,
)
from fastfuncstuff.processing.mask import _peel, erode_many

AFFINE = np.array(
    [
        [-2.5, 0.0, 0.0, 75.0],
        [0.0, 2.0, 0.0, -80.0],
        [0.0, 0.0, 3.0, -60.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# 3dAutobox on _volume(): default, and with -noclust.
AFNI_DEFAULT = (18, 48, 11, 49, 14, 38)
AFNI_NOCLUST = (2, 48, 4, 49, 3, 38)


def _volume() -> torch.Tensor:
    """(nz, ny, nx) ellipsoid + an isolated bright speck near the corner."""
    rng = np.random.default_rng(0)
    nx, ny, nz = 61, 73, 55
    zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    r = ((xx - 33.0) / 16.0) ** 2 + ((yy - 30.0) / 20.0) ** 2 + ((zz - 26.0) / 13.0) ** 2
    vol = np.where(r < 1.0, 900.0 * (1.0 - 0.4 * r), 0.0)
    vol += rng.normal(0, 3.0, vol.shape) * (vol > 0)
    vol[3:6, 4:7, 2:5] = 800.0  # the speck clustering must drop
    return torch.from_numpy(vol.astype(np.float32))


def test_default_bounds_match_3dautobox():
    assert autobox_bounds(_volume()) == AFNI_DEFAULT


def test_noclust_bounds_match_3dautobox():
    # -noclust turns off clipping too, so the speck (and the noise floor) count.
    assert autobox_bounds(_volume(), clust=False, clip=False) == AFNI_NOCLUST


def test_clustering_is_what_rejects_the_speck():
    clustered = autobox_bounds(_volume())
    raw = autobox_bounds(_volume(), clust=False, clip=False)
    assert raw[0] < clustered[0] and raw[2] < clustered[2] and raw[4] < clustered[4]


def test_4d_input_collapses_by_max_abs():
    vol = _volume()
    # A voxel active in any sub-brick must keep the box open, so a stack whose
    # frames are scaled copies gives the same box as one frame.
    stacked = torch.stack([vol * 0.5, vol, vol * 0.8])
    assert autobox_bounds(stacked) == autobox_bounds(vol)


def test_negative_values_count_toward_the_box():
    vol = _volume()
    assert autobox_bounds(-vol) == autobox_bounds(vol)


def test_npad_grows_the_box_symmetrically():
    b = autobox_bounds(_volume())
    padded = pad_bounds(b, 4)
    assert padded == (b[0] - 4, b[1] + 4, b[2] - 4, b[3] + 4, b[4] - 4, b[5] + 4)


def test_npad_safety_clamps_inside_the_matrix():
    vol = _volume()
    shape = tuple(vol.shape)
    padded = pad_bounds(autobox_bounds(vol), 40, shape)
    assert padded == (0, shape[2] - 1, 0, shape[1] - 1, 0, shape[0] - 1)


def test_crop_keeps_every_voxel_in_the_same_place():
    vol = _volume()
    b = autobox_bounds(vol)
    cropped, affine = crop_to_bounds(vol, AFFINE, b)
    assert cropped.shape == (b[5] - b[4] + 1, b[3] - b[2] + 1, b[1] - b[0] + 1)
    assert torch.equal(cropped, vol[b[4] : b[5] + 1, b[2] : b[3] + 1, b[0] : b[1] + 1])
    # The new origin is the world position of the old voxel (imin, jmin, kmin):
    # same point in space, different index.
    expected = AFFINE[:3, :3] @ np.array([b[0], b[2], b[4]]) + AFFINE[:3, 3]
    assert np.allclose(affine[:3, 3], expected)
    assert np.allclose(affine[:3, :3], AFFINE[:3, :3])  # orientation untouched


def test_crop_with_positive_pad_zero_fills():
    vol = _volume()
    b = pad_bounds(autobox_bounds(vol), 60)  # far outside the matrix
    cropped, _ = crop_to_bounds(vol, AFFINE, b)
    assert float(cropped[0, 0, 0]) == 0.0
    kept = vol[max(0, b[4]) : b[5] + 1, max(0, b[2]) : b[3] + 1, max(0, b[0]) : b[1] + 1]
    assert torch.isclose(cropped.sum(), kept.sum(), rtol=1e-6)


def test_largest_cluster_beats_seeded_flood_fill_on_a_brighter_decoy():
    # Two blobs: a big dim one and a small bright one. THD_mask_clust keeps the
    # big one; a flood fill seeded at the brightest voxel would keep the decoy.
    vol = torch.zeros(20, 20, 20)
    vol[2:12, 2:12, 2:12] = 10.0
    vol[16:19, 16:19, 16:19] = 1000.0
    mask = largest_cluster_6conn(vol > 0)
    assert bool(mask[5, 5, 5]) and not bool(mask[17, 17, 17])


def test_erode_many_redilates_a_solid_boundary():
    # THD_mask_erodemany peels *and re-dilates*. Erode-only takes a full shell
    # off a solid block (1728 -> 1000 here); the round trip gives the faces back
    # and keeps only the 8 corners, which have no surviving 18-neighbour.
    mask = torch.zeros(20, 20, 20, dtype=torch.bool)
    mask[4:16, 4:16, 4:16] = True
    peeled = _peel(mask, peelcount=1)
    restored = erode_many(mask, npeel=1)
    assert int(restored.sum()) == int(mask.sum()) - 8
    assert int(peeled.sum()) == 1000
    assert bool(restored[4, 10, 10]) and not bool(peeled[4, 10, 10])  # face comes back
    assert not bool(restored[4, 4, 4])  # corner does not
    assert not bool((peeled & ~restored).any())  # re-dilate only ever adds


def test_erode_many_still_removes_a_thin_filament():
    # The re-dilate must not restore what the peel is there to remove: a
    # one-voxel bridge has too few neighbours to come back.
    mask = torch.zeros(20, 20, 20, dtype=torch.bool)
    mask[4:16, 4:16, 4:16] = True
    mask[10, 10, 16:19] = True  # filament sticking out of the block
    out = erode_many(mask, npeel=1)
    assert not bool(out[10, 10, 18])


def test_erode_many_does_not_shave_a_mask_at_the_matrix_edge():
    # AFNI clamps neighbour indices at the volume face; zero-padding instead
    # makes edge voxels look under-connected and erodes a shell off any mask
    # that reaches the matrix boundary.
    mask = torch.ones(20, 20, 20, dtype=torch.bool)
    assert torch.equal(erode_many(mask, npeel=1), mask)
