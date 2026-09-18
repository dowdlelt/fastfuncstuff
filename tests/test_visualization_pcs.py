"""Denoising component figures: orientation and colour scale.

The panels are read as anatomy, so the only thing that may decide which way they
face is the affine. Two datasets that hold the same brain in different storage
orders have to draw the same picture.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from fastfuncstuff.visualization import plot_denoising_pcs  # noqa: E402

SHAPE = (10, 12, 8)  # (i, j, k) as stored


def _marked_volume() -> np.ndarray:
    """RAS-ordered volume with a single bright voxel right-anterior-superior."""
    vol = np.zeros(SHAPE, dtype=np.float32)
    vol[8, 10, 6] = 1.0
    return vol


def _run_figure(vol_ijk: np.ndarray, affine: np.ndarray, n_runs: int = 2):
    n_vox = int(np.prod(SHAPE))
    weights = vol_ijk.reshape(-1, 1)
    figs = plot_denoising_pcs(
        noise_pcs_per_run=[np.zeros((20, 1), dtype=np.float32) for _ in range(n_runs)],
        run_starts=[0, 20][:n_runs],
        pc_weights_per_run=[weights for _ in range(n_runs)],
        volume_shape=SHAPE,
        voxel_mask=np.ones(n_vox, dtype=bool),
        noise_pool_mask=None,
        n_pcs_to_show=1,
        n_slices=1,
        tr=2.0,
        affine=affine,
        return_figs=True,
    )
    return figs[0]


def _panel_images(fig):
    return [
        ax.get_images()[0].get_array()
        for ax in fig.axes
        if ax.get_images() and ax.get_images()[0].get_array().ndim == 2
    ]


class TestOrientation:
    def test_storage_order_does_not_change_the_picture(self):
        """The same brain written RAS and written LPI must draw identically."""
        ras = _marked_volume()
        ras_affine = np.diag([2.0, 2.0, 3.0, 1.0])

        # Same anatomy, stored flipped on every axis, with the affine to match.
        lpi = ras[::-1, ::-1, ::-1].copy()
        lpi_affine = np.diag([-2.0, -2.0, -3.0, 1.0])
        lpi_affine[:3, 3] = [
            (SHAPE[0] - 1) * 2.0,
            (SHAPE[1] - 1) * 2.0,
            (SHAPE[2] - 1) * 3.0,
        ]

        a = _panel_images(_run_figure(ras, ras_affine))
        b = _panel_images(_run_figure(lpi, lpi_affine))

        assert len(a) == len(b) == 6  # 3 planes x 2 runs
        for pa, pb in zip(a, b, strict=True):
            np.testing.assert_allclose(np.nan_to_num(pa), np.nan_to_num(pb))

    def test_axial_panel_is_radiological_with_anterior_up(self):
        """Subject right on the image left, anterior at the top."""
        fig = _run_figure(_marked_volume(), np.diag([2.0, 2.0, 3.0, 1.0]), n_runs=1)
        # plane_specs lead with sagittal by default; axial is the third row and
        # the third image drawn for the single run.
        axial = np.nan_to_num(_panel_images(fig)[2])
        row, col = np.unravel_index(int(np.argmax(axial)), axial.shape)
        assert row < axial.shape[0] / 2, "anterior must be in the top half"
        assert col < axial.shape[1] / 2, "subject right must be on the image left"


class TestColourScale:
    def test_each_run_gets_its_own_symmetric_scale(self):
        """A run's loadings carry that run's amplitude, so each column scales itself."""
        # Loadings everywhere, not one marked voxel: a 98th percentile needs a
        # distribution to sit in.
        vol_a = np.random.default_rng(0).normal(size=SHAPE).astype(np.float32)
        vol_b = vol_a * 0.25
        n_vox = int(np.prod(SHAPE))
        figs = plot_denoising_pcs(
            noise_pcs_per_run=[np.zeros((20, 1), dtype=np.float32) for _ in range(2)],
            run_starts=[0, 20],
            pc_weights_per_run=[vol_a.reshape(-1, 1), vol_b.reshape(-1, 1)],
            volume_shape=SHAPE,
            voxel_mask=np.ones(n_vox, dtype=bool),
            noise_pool_mask=None,
            n_pcs_to_show=1,
            n_slices=1,
            affine=np.diag([2.0, 2.0, 3.0, 1.0]),
            return_figs=True,
        )
        clims = [ax.get_images()[0].get_clim() for ax in figs[0].axes if ax.get_images()]
        assert len(clims) == 6  # 3 planes x 2 runs
        # Three planes of one run share that run's limits; the two runs differ.
        run_a, run_b = set(clims[:3]), set(clims[3:])
        assert len(run_a) == len(run_b) == 1
        assert run_a != run_b, "a quarter-amplitude run must not reuse the other's scale"
        for lo, hi in run_a | run_b:
            assert lo == pytest.approx(-hi)

    def test_no_colorbar_is_drawn(self):
        """The scale is per panel and arbitrary; a colourbar would imply otherwise."""
        fig = _run_figure(_marked_volume(), np.diag([2.0, 2.0, 3.0, 1.0]), n_runs=1)
        assert all(getattr(ax, "_colorbar", None) is None for ax in fig.axes)
        assert len(fig.axes) == 4  # timecourse + 3 planes
