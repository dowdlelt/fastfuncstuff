"""Weight image (AFNI mri_weightize) tests."""

import torch

from fastfuncstuff.processing.weight import _thd_cliplevel, compute_weight_image


def _brain_with_background(seed=0):
    """Bright brain blob + a faded blob + low nonzero background filling the FOV."""
    torch.manual_seed(seed)
    nz, ny, nx = 40, 48, 48
    kk, jj, ii = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32),
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    brain = torch.clamp(1 - ((ii - 24) ** 2 + (jj - 24) ** 2 + (kk - 20) ** 2) / 100.0, min=0) * 1000
    brain += torch.exp(-((ii - 24) ** 2 + (jj - 34) ** 2 + (kk - 20) ** 2) / (2 * 25)) * 150
    return brain + torch.rand(nz, ny, nx) * 40  # background everywhere


def test_clusterize_drops_background_keeps_brain():
    """The AFNI cleanup (bottom-clip + largest cluster) removes the background
    'square' while keeping the brain and a smooth halo."""
    img = _brain_with_background()
    w = compute_weight_image(
        img, edge_fraction=0.05, median_radius=2.25, clusterize=True, hist_cliplevel=True
    )
    # background corner is fully zeroed
    assert float((w[2:8, 2:10, 2:10] > 0.05).float().mean()) == 0.0
    # brain centre carries full weight
    assert w[20, 24, 24].item() > 0.9
    # a soft halo survives (graded, not a hard binary mask)
    graded = ((w > 0.01) & (w < 0.99)).sum().item()
    assert graded > 100
    # far fewer nonzero voxels than the whole FOV (background gone)
    assert int((w > 0).sum()) < 0.3 * w.numel()


def test_default_weight_unchanged_fills_fov():
    """Default path (moco/qwarp) stays Gaussian-only: background not clustered out."""
    img = _brain_with_background()
    w = compute_weight_image(img)  # defaults: no median, no clusterize
    # Without clusterize the smoothed weight still covers most of the interior
    # (far broader than the <0.3 clusterize path); the border band is zeroed
    # (AFNI -edging), so it no longer fills the entire FOV.
    assert int((w > 0).sum()) > 0.5 * w.numel()
    # The edge band is exactly zero (post-smooth -edging).
    assert float(w[:3].abs().sum()) == 0.0  # first z-faces zeroed


def test_thd_cliplevel_separates_brain_from_background():
    """The histogram clip level sits between background and brain intensity."""
    img = _brain_with_background()
    cl = _thd_cliplevel(img, 0.5)
    assert 40.0 < cl < 1000.0  # above background (~40), below brain peak (1000)
