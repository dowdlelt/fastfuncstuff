"""Slab projection in the image windows: the overlays reduced across slices."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.viewer.compose import render_viewport, slab_reduce
from fastfuncstuff.viewer.layers import SignMode
from fastfuncstuff.viewer.session import ViewerSession
from fastfuncstuff.viewer.vocab import OpenView, SetOverlay, SetUnderlay, SetViewSlab

nib = pytest.importorskip("nibabel")
CPU = torch.device("cpu")
SHAPE = (24, 24, 20)
AFFINE = np.diag([2.0, 2.0, 2.0, 1.0])


def test_mip_takes_value_and_stat_from_the_slice_with_the_largest_statistic():
    values = torch.tensor([[[1.0]], [[5.0]], [[3.0]]])
    stat = torch.tensor([[[2.0]], [[-9.0]], [[4.0]]])
    v, s = slab_reduce(values, stat, "mip", SignMode.BOTH)
    assert (float(v), float(s)) == (5.0, -9.0)
    # Positive only: the negative peak is not a peak.
    v, s = slab_reduce(values, stat, "mip", SignMode.POS)
    assert (float(v), float(s)) == (3.0, 4.0)


def test_minip_skips_zeros_and_mean_ignores_nan():
    values = torch.tensor([[[0.0]], [[5.0]], [[3.0]]])
    v, _ = slab_reduce(values, values, "minip", SignMode.BOTH)
    assert float(v) == 3.0
    values = torch.tensor([[[float("nan")]], [[4.0]], [[2.0]]])
    v, _ = slab_reduce(values, values, "mean", SignMode.BOTH)
    assert float(v) == 3.0


@pytest.fixture
def session(tmp_path):
    rng = np.random.default_rng(0)
    anat = rng.normal(100, 2, SHAPE).astype(np.float32)
    over = np.zeros(SHAPE, np.float32)
    # Not integers, or the load guesses a label layer, which a slab leaves alone.
    over[10:13, 10:13, 13] = 47.3  # three slices (6 mm) above the middle
    nib.save(nib.Nifti1Image(anat, AFFINE), str(tmp_path / "anat.nii.gz"))
    nib.save(nib.Nifti1Image(over, AFFINE), str(tmp_path / "over.nii.gz"))
    s = ViewerSession(device=CPU)
    s.do(SetUnderlay(str(tmp_path / "anat.nii.gz")))
    s.do(SetOverlay(str(tmp_path / "over.nii.gz")))
    top = s.state.layers.layers[-1]
    s.state.layers.update(
        top.key, colormap="hot", range_lo=0.0, range_hi=50.0, threshold=10.0, sign_mode=SignMode.POS
    )
    s.state.crosshair = (12, 12, 10)
    yield s
    s.close()


def _coloured(img) -> int:
    rgb = np.asarray(img.rgba)[..., :3].astype(int)
    return int(np.sum(np.abs(rgb[..., 0] - rgb[..., 2]) > 30))


def test_a_slab_draws_an_overlay_that_sits_off_the_slice(session):
    s = session
    s.do(OpenView("V9", "image", "axial"))
    plain = render_viewport(s, s.state.viewports.get("V9"))
    assert _coloured(plain) == 0
    s.do(SetViewSlab("V9", 12.0, "mip"))  # +-6 mm: three 2 mm slices each side
    vp = s.state.viewports.get("V9")
    assert vp.slab_mm == 12.0 and "mip 12 mm" in vp.title
    slab = render_viewport(s, vp)
    assert _coloured(slab) >= 9
    s.do(SetViewSlab("V9", 10.0, "mip"))  # +-5 mm: two slices, one short
    assert _coloured(render_viewport(s, s.state.viewports.get("V9"))) == 0


def test_an_unknown_slab_mode_is_refused(session):
    session.do(OpenView("V9", "image", "axial"))
    with pytest.raises(ValueError, match="slab mode"):
        session.do(SetViewSlab("V9", 10.0, "median"))
