"""Tests for processing/b0fmap.py (B0 GRE fieldmap -> undistortion warp)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from fastfuncstuff.processing.b0fmap import (
    condition_field,
    extrapolate_outward,
    field_to_pe_warp,
    read_echo_times,
    read_epi_geometry,
    romeo_available,
    run_romeo,
    synthesize_distorted,
)
from fastfuncstuff.processing.topup import _jacobian_pe


def _smooth_field(shape=(12, 20, 20), amp=60.0):
    """A smooth, sign-changing Hz field — a caricature of a sinus dropout."""
    nz, ny, nx = shape
    z, y, x = torch.meshgrid(
        torch.linspace(-1, 1, nz),
        torch.linspace(-1, 1, ny),
        torch.linspace(-1, 1, nx),
        indexing="ij",
    )
    return amp * torch.exp(-((y + 0.4) ** 2 + x**2) / 0.3) - 0.3 * amp * z


class TestExtrapolate:
    def test_fills_outward_without_touching_the_interior(self):
        field = _smooth_field()
        mask = torch.zeros_like(field, dtype=torch.bool)
        mask[3:9, 5:15, 5:15] = True
        out, grown = extrapolate_outward(field * mask, mask, n_iter=3)

        assert torch.allclose(out[mask], field[mask]), "interior must be untouched"
        assert grown.sum() > mask.sum(), "support must grow"
        # Everything the original mask touched plus its 3-voxel shell is now defined.
        assert bool(grown[3:9, 5:15, 5:15].all())
        assert bool(grown[2, 5, 5])

    def test_no_step_at_the_mask_edge(self):
        """The point of extrapolating: the gradient across the boundary must stay
        bounded instead of falling off a cliff to zero."""
        field = _smooth_field()
        mask = torch.zeros_like(field, dtype=torch.bool)
        mask[:, 5:15, 5:15] = True

        zeroed = field * mask
        extended, _ = extrapolate_outward(zeroed, mask, n_iter=4)

        # Jump across the boundary column, along y.
        jump_zeroed = (zeroed[:, 4, 8] - zeroed[:, 5, 8]).abs().max()
        jump_ext = (extended[:, 4, 8] - extended[:, 5, 8]).abs().max()
        assert jump_ext < 0.25 * jump_zeroed

    def test_zero_iterations_is_identity(self):
        field = _smooth_field()
        mask = torch.ones_like(field, dtype=torch.bool)
        out, grown = extrapolate_outward(field, mask, n_iter=0)
        assert torch.equal(out, field)
        assert bool(grown.all())


class TestConditionField:
    def test_taper_reaches_zero_far_from_the_object(self):
        field = _smooth_field(shape=(16, 32, 32))
        mask = torch.zeros_like(field, dtype=torch.bool)
        mask[6:10, 13:19, 13:19] = True
        out, support = condition_field(
            field, mask, (2.0, 2.0, 2.0), fwhm_mm=0.0, extend_mm=4.0, rolloff_mm=4.0
        )
        assert float(out[0, 0, 0].abs()) < 1e-3, "far air must taper to zero"
        assert float(out[mask].abs().mean()) > 1.0, "the object's field must survive"
        assert support.sum() >= mask.sum()

    def test_smoothing_is_weighted_not_diluted_at_the_edge(self):
        """Normalised convolution must not drag the mask-edge field toward the zeros
        outside it — that would be a systematic underestimate exactly at the brain edge."""
        field = torch.full((8, 16, 16), 50.0)
        mask = torch.zeros_like(field, dtype=torch.bool)
        mask[:, 4:12, 4:12] = True
        out, _ = condition_field(
            field, mask, (2.0, 2.0, 2.0), fwhm_mm=6.0, extend_mm=0.0, rolloff_mm=0.0
        )
        # A constant field stays constant inside the mask, edge included.
        assert torch.allclose(out[mask], torch.full_like(out[mask], 50.0), atol=1e-2)

    def test_anisotropic_voxels_use_per_axis_sigma(self):
        """A physical FWHM on an anisotropic grid must smooth fewer voxels along the
        thick axis; a scalar sigma would over-smooth it."""
        field = torch.zeros((16, 16, 16))
        field[8, 8, 8] = 100.0
        mask = torch.ones_like(field, dtype=torch.bool)
        out, _ = condition_field(
            field, mask, (8.0, 1.0, 1.0), fwhm_mm=6.0, extend_mm=0.0, rolloff_mm=0.0
        )
        spread_z = float(out[:, 8, 8].abs().sum() - out[8, 8, 8].abs())
        spread_x = float(out[8, 8, :].abs().sum() - out[8, 8, 8].abs())
        assert spread_x > 3 * spread_z


class TestFieldToWarp:
    @pytest.mark.parametrize("pe_dir", ["i", "j", "k", "j-"])
    def test_pull_and_inverse_compose_to_identity(self, pe_dir):
        field = _smooth_field(shape=(16, 24, 24), amp=25.0)
        pull, inv, pe_tdim = field_to_pe_warp(field, readout_s=0.03, pe_dir=pe_dir)

        # Applying the pull warp then the inverse must return each voxel to itself.
        g = pull.movedim(pe_tdim, -1)
        h = inv.movedim(pe_tdim, -1)
        n = g.shape[-1]
        idx = torch.arange(n, dtype=g.dtype).expand_as(g)
        from fastfuncstuff.processing.medic import _interp_along_last_axis

        residual = (h + _interp_along_last_axis(g, idx + h)).abs()
        # Ignore the two edge voxels, where the 1-D inverse clamps by construction.
        assert float(residual[..., 2:-2].max()) < 0.05

    def test_sign_flips_with_pe_polarity(self):
        field = _smooth_field()
        pos, _, _ = field_to_pe_warp(field, 0.03, "j")
        neg, _, _ = field_to_pe_warp(field, 0.03, "j-")
        assert torch.allclose(pos, -neg)

    def test_displacement_scales_with_readout(self):
        field = _smooth_field()
        a, _, _ = field_to_pe_warp(field, 0.02, "j")
        b, _, _ = field_to_pe_warp(field, 0.04, "j")
        assert torch.allclose(2 * a, b)

    def test_bad_pe_dir_rejected(self):
        with pytest.raises(ValueError, match="phase-encode"):
            field_to_pe_warp(_smooth_field(), 0.03, "q")


class TestSynthesizeDistorted:
    def test_round_trip_recovers_the_undistorted_image(self):
        """Forward-warp an image through the field, pull it back with the warp, and the
        original must return. This is the whole correctness claim of the warp convention:
        the GRE field IS the pull warp, no inversion."""
        torch.manual_seed(0)
        nz, ny, nx = 12, 32, 32
        # A smooth image — an interpolation round trip cannot preserve sharp edges.
        z, y, x = torch.meshgrid(
            torch.linspace(-1, 1, nz),
            torch.linspace(-1, 1, ny),
            torch.linspace(-1, 1, nx),
            indexing="ij",
        )
        img = torch.exp(-(y**2 + x**2 + z**2) / 0.5) + 0.3 * torch.cos(3 * y)

        field = _smooth_field(shape=(nz, ny, nx), amp=20.0)
        pull, inv, pe_tdim = field_to_pe_warp(field, readout_s=0.03, pe_dir="j")
        assert float(pull.abs().max()) > 0.5, "test needs a non-trivial displacement"

        distorted = synthesize_distorted(img, inv, pe_tdim)
        from fastfuncstuff.processing.topup import _resample_pe

        back = _resample_pe(distorted, pull, pe_tdim)

        interior = (slice(None), slice(6, -6), slice(6, -6))
        err = (back[interior] - img[interior]).abs().max()
        assert float(err) < 0.05, f"round-trip error {float(err):.4f}"

    def test_identity_field_is_a_no_op(self):
        img = _smooth_field()
        zero = torch.zeros_like(img)
        _, inv, pe_tdim = field_to_pe_warp(zero, 0.03, "j")
        assert torch.allclose(synthesize_distorted(img, inv, pe_tdim), img, atol=1e-5)

    def test_jacobian_is_positive_for_a_mild_field(self):
        field = _smooth_field(amp=15.0)
        pull, _, pe_tdim = field_to_pe_warp(field, 0.03, "j")
        assert float(_jacobian_pe(pull, pe_tdim).min()) > 0


class TestSidecars:
    def test_echo_times_from_multi_echo_sidecars(self, tmp_path):
        paths = []
        for i, te in enumerate([0.004, 0.00711], start=1):
            nii = tmp_path / f"sub-1_phase{i}.nii.gz"
            nii.touch()
            (tmp_path / f"sub-1_phase{i}.json").write_text(json.dumps({"EchoTime": te}))
            paths.append(str(nii))
        assert read_echo_times(paths) == pytest.approx([4.0, 7.11])

    def test_phasediff_sidecar_collapses_to_delta_te(self, tmp_path):
        nii = tmp_path / "sub-1_phasediff.nii.gz"
        nii.touch()
        (tmp_path / "sub-1_phasediff.json").write_text(
            json.dumps({"EchoTime1": 0.004, "EchoTime2": 0.00646})
        )
        assert read_echo_times([str(nii)], phasediff=True) == pytest.approx([2.46])

    def test_missing_sidecar_returns_none(self, tmp_path):
        nii = tmp_path / "sub-1_phase1.nii.gz"
        nii.touch()
        assert read_echo_times([str(nii)]) is None

    def test_epi_geometry(self, tmp_path):
        nii = tmp_path / "sub-1_bold.nii.gz"
        nii.touch()
        (tmp_path / "sub-1_bold.json").write_text(
            json.dumps({"PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.03465})
        )
        assert read_epi_geometry(str(nii)) == ("j-", pytest.approx(0.03465))

    def test_epi_geometry_falls_back_to_echo_spacing(self, tmp_path):
        nii = tmp_path / "sub-1_bold.nii.gz"
        nii.touch()
        (tmp_path / "sub-1_bold.json").write_text(
            json.dumps(
                {
                    "PhaseEncodingDirection": "j",
                    "EffectiveEchoSpacing": 0.00055,
                    "ReconMatrixPE": 64,
                }
            )
        )
        pe, trt = read_epi_geometry(str(nii))
        assert pe == "j"
        assert trt == pytest.approx(0.00055 * 63)


def _synthetic_echoes(amp: float, tes_ms: list[float]):
    """A smooth Hz field plus the wrapped phase it would produce at each TE."""
    nx, ny, nz = 32, 32, 16
    a, b, c = np.meshgrid(
        np.linspace(-1, 1, nx),
        np.linspace(-1, 1, ny),
        np.linspace(-1, 1, nz),
        indexing="ij",
    )
    truth = (amp * np.exp(-(b**2 + c**2) / 0.5) - (amp / 3.0) * a).astype(np.float32)
    mag = (np.exp(-(a**2 + b**2 + c**2) / 0.8) * 1000.0).astype(np.float32)
    phase = np.stack(
        [np.angle(np.exp(1j * 2 * np.pi * truth * (te / 1000.0))) for te in tes_ms], axis=-1
    ).astype(np.float32)
    mag4d = np.stack([mag] + [mag * 0.8] * (len(tes_ms) - 1), axis=-1)
    return truth, phase, mag4d, mag > 100.0


@pytest.mark.skipif(not romeo_available(), reason="romeo binary not on PATH")
class TestRomeo:
    def test_recovers_a_known_field_from_synthetic_phase(self, tmp_path):
        """Simulate two echoes of a known field, wrap the phase, and check ROMEO
        unwraps back to it. At this amplitude the phase wraps twice at the longer TE,
        so the unwrapping is genuinely exercised."""
        tes_ms = [4.0, 7.11]
        truth, phase, mag4d, obj = _synthetic_echoes(120.0, tes_ms)
        assert 2 * np.pi * np.abs(truth).max() * tes_ms[1] / 1000.0 > 2 * np.pi

        out = run_romeo(
            phase, mag4d, tes_ms, np.eye(4), outdir=tmp_path / "r", mask="nomask", verbose=False
        )
        assert out.b0_hz.shape == truth.shape
        # ROMEO's global offset correction can leave a whole-volume n*2pi/dTE shift;
        # what must match is the spatial structure.
        err = (out.b0_hz - truth)[obj]
        assert np.abs(err - np.median(err)).max() < 15.0, "unwrap did not recover the field"

    def test_radian_phase_is_not_rescaled(self, tmp_path):
        """Bug of record: ROMEO maps its input's [min,max] onto [-pi,pi] by default. Phase
        already in radians that does not span the full circle is therefore *stretched*,
        and the field comes back scaled by 2*pi/observed_range — a 4.3x error here, with
        no warning from anywhere. ``phase_units="auto"`` must detect radians and pass
        --no-phase-rescale.

        The amplitude is deliberately small enough that no wrapping occurs at either
        echo, so a correct run must reproduce the field essentially exactly and any
        scale error is unambiguous.
        """
        tes_ms = [4.0, 7.11]
        truth, phase, mag4d, obj = _synthetic_echoes(20.0, tes_ms)
        assert np.abs(phase).max() < np.pi, "test needs an unwrapped input"

        good = run_romeo(
            phase, mag4d, tes_ms, np.eye(4), outdir=tmp_path / "a", mask="nomask", verbose=False
        )
        ratio = np.median(good.b0_hz[obj] / truth[obj])
        assert ratio == pytest.approx(1.0, abs=0.05), f"field scaled by {ratio:.3f}"

        # And forcing "radians" explicitly must give exactly what auto chose.
        forced = run_romeo(
            phase,
            mag4d,
            tes_ms,
            np.eye(4),
            outdir=tmp_path / "b",
            mask="nomask",
            phase_units="radians",
            verbose=False,
        )
        assert np.allclose(forced.b0_hz, good.b0_hz, atol=1e-4)

    def test_phasediff_single_echo_form(self, tmp_path):
        """A Siemens phasediff volume is already the inter-echo difference, so passing
        the single delta-TE must give the same Hz field as the two-echo form."""
        dte = 3.11
        truth, phase, mag4d, obj = _synthetic_echoes(20.0, [dte])
        out = run_romeo(
            phase[..., 0],
            mag4d[..., 0],
            [dte],
            np.eye(4),
            outdir=tmp_path / "pd",
            mask="nomask",
            verbose=False,
        )
        err = (out.b0_hz - truth)[obj]
        assert np.abs(err - np.median(err)).max() < 2.0


class TestPhaseScaling:
    """Raw scanner phase is converted to radians by us, never by ROMEO, and every echo
    is scaled from its OWN min/max. Measured: ROMEO pools the whole 4-D, so widening one
    echo's range halves another echo's scaling (6.2832 -> 3.1424 rad). SPM does it per
    image (``pm_scale_phase.m``), and so do we.
    """

    @pytest.mark.skipif(not romeo_available(), reason="romeo binary not on PATH")
    def test_each_echo_scaled_from_its_own_range(self, tmp_path):
        """Same field, two echoes written at DIFFERENT quantisation depths — each one
        filling its own range, as the reconstruction does. Scaling each from its own
        min/max recovers one consistent field; pooling the 4-D would stretch the
        shallower echo and corrupt the phase difference.
        """
        tes_ms = [4.0, 7.11]
        truth, phase, mag4d, obj = _synthetic_echoes(120.0, tes_ms)  # large: phase wraps
        levels = (4096, 1024)  # 12-bit and 10-bit
        quant = np.stack(
            [np.round((phase[..., i] / (2 * np.pi) + 0.5) * (n - 1)) for i, n in enumerate(levels)],
            axis=-1,
        ).astype(np.float32)
        for i, n in enumerate(levels):  # each echo fills its own range
            span = quant[..., i].max() - quant[..., i].min()
            assert quant[..., i].min() == 0 and span > 0.99 * (n - 1)

        out = run_romeo(
            quant, mag4d, tes_ms, np.eye(4), outdir=tmp_path / "s", mask="nomask", verbose=False
        )
        err = out.b0_hz[obj] - truth[obj]
        assert np.abs(err - np.median(err)).max() < 15.0

    def test_scaling_is_independent_across_echoes(self):
        """Widening one echo's range must not change what another echo maps to — the
        property ROMEO's pooled rescale violates."""
        from fastfuncstuff.processing import b0fmap as B

        base = np.linspace(1024, 3072, 64, dtype=np.float32).reshape(1, 1, 64, 1)

        def scaled(second):
            ph = np.concatenate([base, second], axis=-1)
            captured = {}
            orig = B.subprocess.run

            def fake(cmd, **kw):  # stop before ROMEO; we only want the converted phase
                import nibabel as nib

                captured["phase"] = np.asarray(
                    nib.load(str(cmd[cmd.index("-p") + 1])).dataobj, dtype=np.float32
                )
                raise RuntimeError("stop")

            B.subprocess.run = fake
            try:
                B.run_romeo(ph, np.ones_like(ph), [4.0, 8.0], np.eye(4), verbose=False)
            except RuntimeError:
                pass
            finally:
                B.subprocess.run = orig
            return captured["phase"][..., 0]

        narrow_mate = scaled(base.copy())
        wide_mate = scaled(np.linspace(0, 4095, 64, dtype=np.float32).reshape(1, 1, 64, 1))
        assert np.allclose(narrow_mate, wide_mate, atol=1e-5)
        assert narrow_mate.min() == pytest.approx(-np.pi, abs=1e-5)
        assert narrow_mate.max() == pytest.approx(np.pi, abs=1e-5)

    @pytest.mark.skipif(not romeo_available(), reason="romeo binary not on PATH")
    def test_explicit_phase_range_applies_to_every_echo(self, tmp_path):
        """-phase_range pins the conversion for inputs that do not fill their range."""
        tes_ms = [3.11]
        truth, phase, mag4d, obj = _synthetic_echoes(20.0, tes_ms)
        narrow = np.round((phase[..., 0] / (2 * np.pi) + 0.5) * 2048 + 1024).astype(np.float32)
        out = run_romeo(
            narrow,
            mag4d[..., 0],
            tes_ms,
            np.eye(4),
            outdir=tmp_path / "p",
            mask="nomask",
            phase_units="scanner",
            phase_range=(0.0, 4095.0),
            verbose=False,
        )
        ratio = np.median(out.b0_hz[obj] / truth[obj])
        assert ratio == pytest.approx(0.5, abs=0.03), f"got {ratio:.3f}"
