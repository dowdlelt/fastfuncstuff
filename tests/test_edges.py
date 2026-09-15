"""Tests for the thin 3-D edge map (processing/edges.py)."""

from __future__ import annotations

import torch

from fastfuncstuff.processing.edges import edge_map


def _sphere(n=33, radius=10.0, spacing=(1.0, 1.0, 1.0)) -> torch.Tensor:
    axes = [(torch.arange(n) - n // 2).float() * s for s in spacing]
    kk, jj, ii = torch.meshgrid(*axes, indexing="ij")
    return (torch.sqrt(kk**2 + jj**2 + ii**2) <= radius).float() * 100.0


def test_sphere_edge_is_one_voxel_thin_at_the_boundary():
    """A thick band hides a one-voxel misalignment; the overlay is useless without thinning."""
    vol = _sphere()
    edges = edge_map(vol, sigma=1.0)
    c = 33 // 2
    # Along +x from the centre, exactly one retained voxel, on the boundary.
    ray = edges[c, c, c:]
    hits = torch.nonzero(ray > 0).flatten()
    assert hits.numel() == 1
    assert abs(int(hits[0]) - 10) <= 1
    # No edges inside the solid or far outside it.
    assert edges[c - 5 : c + 6, c - 5 : c + 6, c - 5 : c + 6].abs().sum() == 0
    assert edges[:3].abs().sum() == 0

    thick = edge_map(vol, sigma=1.0, thin=False)
    assert int((thick[c, c, c:] > 0).sum()) > 1


def test_threshold_is_relative_to_image_units():
    vol = _sphere()
    a = edge_map(vol, threshold=0.2) > 0
    b = edge_map(vol * 1e-3, threshold=0.2) > 0
    # A symmetric phantom makes exact ties along the gradient, and rescaling moves a
    # handful of them by float rounding; anything beyond that is a units dependence.
    assert int((a ^ b).sum()) <= 0.01 * int(a.sum())


def test_mask_zeroes_edges_outside():
    vol = _sphere()
    mask = torch.zeros_like(vol)
    mask[:, :, : 33 // 2] = 1
    edges = edge_map(vol, mask=mask)
    assert edges[:, :, 33 // 2 :].abs().sum() == 0
    assert edges[:, :, : 33 // 2].abs().sum() > 0
