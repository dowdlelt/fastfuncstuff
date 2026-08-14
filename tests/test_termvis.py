"""Terminal orthoview rendering (ffs_info -vis)."""

import numpy as np
import pytest

from fastfuncstuff.termvis import orthoview, to_ras


def _blob(shape=(20, 24, 16)):
    """A bright off-centre sphere, so orientation errors are visible in the output."""
    zz, yy, xx = np.indices(shape)
    center = np.array(shape) / 2 + np.array([3, -2, 1])
    r = np.sqrt((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2)
    return np.exp(-(r**2) / 20.0).astype(np.float32)


def _visible_width(line: str) -> int:
    import re

    return len(re.sub(r"\x1b\[[0-9;]*m", "", line))


class TestOrthoview:
    @pytest.mark.parametrize("color", [False, True])
    def test_panels_are_rectangular_and_fit_the_requested_width(self, color):
        out = orthoview(_blob(), (1.0, 1.0, 2.0), width=90, color=color)
        lines = out.splitlines()
        widths = {_visible_width(line) for line in lines}
        assert len(widths) == 1, f"ragged panel rows: {sorted(widths)}"
        assert widths.pop() <= 90

    def test_anisotropic_voxels_stretch_the_through_plane_axis(self):
        """A 1×1×4 mm volume must not be drawn as if it were isotropic."""
        iso = orthoview(_blob(), (1.0, 1.0, 1.0), width=80, color=False)
        thick = orthoview(_blob(), (1.0, 1.0, 4.0), width=80, color=False)
        assert len(thick.splitlines()) > len(iso.splitlines())

    def test_all_zero_volume_does_not_divide_by_zero(self):
        out = orthoview(np.zeros((8, 8, 8), dtype=np.float32), (1.0, 1.0, 1.0), color=False)
        assert out.splitlines()

    def test_nonfinite_only_is_reported_not_raised(self):
        vol = np.full((8, 8, 8), np.nan, dtype=np.float32)
        assert "no finite voxels" in orthoview(vol, (1.0, 1.0, 1.0), color=False)

    def test_rejects_4d(self):
        with pytest.raises(ValueError, match="3D"):
            orthoview(np.zeros((4, 4, 4, 2), dtype=np.float32), (1.0, 1.0, 1.0))

    def test_captions_name_the_slice_and_the_up_direction(self):
        out = orthoview(_blob(), (1.0, 1.0, 1.0), width=120, color=False)
        caption = out.splitlines()[-1]
        assert "sag" in caption and "cor" in caption and "axi" in caption
        assert "↑S" in caption and "↑A" in caption


class TestToRAS:
    def test_reorients_and_permutes_zooms_together(self):
        # LAS-ish storage with the axes shuffled: k is the R/L axis.
        data = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        affine = np.zeros((4, 4))
        affine[0, 2] = 1.5  # k → +x (Right)
        affine[1, 0] = -2.0  # i → -y
        affine[2, 1] = 3.0  # j → +z
        affine[3, 3] = 1.0
        out, zooms = to_ras(data, affine)
        assert out.shape == (4, 2, 3)
        assert zooms == pytest.approx((1.5, 2.0, 3.0))

    def test_ras_input_is_unchanged(self):
        data = _blob()
        out, zooms = to_ras(data, np.diag([2.0, 2.0, 3.0, 1.0]))
        assert np.array_equal(out, data)
        assert zooms == pytest.approx((2.0, 2.0, 3.0))
