"""looks_skull_stripped: tell a masked brain from a clipped field of view."""

from __future__ import annotations

import torch

from fastfuncstuff.processing.mask import looks_skull_stripped


def _grid(shape=(40, 48, 40)):
    return torch.meshgrid(*[torch.linspace(-1, 1, n) for n in shape], indexing="ij")


def test_stripped_brain_is_flagged():
    z, y, x = _grid()
    brain = (z / 0.7) ** 2 + (y / 0.8) ** 2 + (x / 0.7) ** 2 <= 1
    assert looks_skull_stripped(brain)


def test_clipped_fov_and_slab_are_not():
    z, y, x = _grid()
    wedge_clipped = (z + 0.5 * y) < 0.6  # most of the grid, one corner missing
    assert not looks_skull_stripped(wedge_clipped)
    slab = (z.abs() < 0.3) & (y.abs() < 0.9) & (x.abs() < 0.9)  # partial-coverage box
    assert not looks_skull_stripped(slab)
    assert not looks_skull_stripped(torch.ones(10, 10, 10, dtype=torch.bool))


def _args(**kw):
    from types import SimpleNamespace

    return SimpleNamespace(
        **{"nocoverage": False, "coverage_erode": 0, "force_coverage": False, **kw}
    )


def test_image_support_drops_coverage_on_a_stripped_brain():
    """Coverage on a stripped T1 -> MNI pair lost up to 10 labels and folded the warp."""
    from fastfuncstuff.cli_utils import frame_coverage, image_support

    z, y, x = _grid()
    vol = ((z / 0.7) ** 2 + (y / 0.8) ** 2 + (x / 0.7) ** 2 <= 1).float() * 100.0
    cpu = torch.device("cpu")
    assert image_support(vol, _args(), cpu, False, "base", 0)[1] is None
    assert frame_coverage(vol, _args(), cpu) is None
    assert image_support(vol, _args(force_coverage=True), cpu, False, "base", 0)[1] is not None
    assert frame_coverage(vol, _args(force_coverage=True), cpu) is not None

    clipped = ((z + 0.5 * y) < 0.6).float() * 100.0  # a clipped FoV keeps its coverage
    assert image_support(clipped, _args(), cpu, False, "base", 0)[1] is not None


def test_cross_fill_lands_on_the_receiving_scale():
    """A fill on the other image's raw scale is a cliff of its own (3e5 into ~1.6e3)."""
    from fastfuncstuff.processing.mask import cross_fill_no_data

    z, y, x = _grid()
    fixed = 1600.0 + 100.0 * x
    moving = 200.0 * fixed  # same anatomy, 200x the intensity
    fcov, mcov = (z < 0.5), (z > -0.5)
    f_m, m_m, cover = cross_fill_no_data(fixed * fcov, moving * mcov, fcov, mcov)
    torch.testing.assert_close(f_m, fixed, rtol=1e-5, atol=1e-2)
    torch.testing.assert_close(m_m, moving, rtol=1e-5, atol=1.0)
    assert bool((cover == (fcov & mcov)).all())
