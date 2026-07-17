"""Tests for ffs_util_complex polar/Cartesian NIfTI conversions."""

from __future__ import annotations

import nibabel as nib
import numpy as np

from fastfuncstuff.cli.util_complex import main


def _write(path, data):
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine), path)
    return str(path)


def test_magnitude_phase_to_real_imag_preserves_affine(tmp_path):
    magnitude = _write(tmp_path / "mag.nii.gz", [[[2.0, 3.0]]])
    phase = _write(tmp_path / "phase.nii.gz", [[[0.0, np.pi / 2]]])
    prefix = tmp_path / "complex.nii.gz"

    assert main(["-mag", magnitude, "-phase", phase, "-prefix", str(prefix), "-device", "cpu"]) == 0

    real = nib.load(tmp_path / "complex_real.nii.gz")
    imag = nib.load(tmp_path / "complex_imag.nii.gz")
    np.testing.assert_allclose(real.get_fdata(), [[[2.0, 0.0]]], atol=1e-6)
    np.testing.assert_allclose(imag.get_fdata(), [[[0.0, 3.0]]], atol=1e-6)
    np.testing.assert_allclose(real.affine, np.diag([2.0, 3.0, 4.0, 1.0]))


def test_scale_phase_and_allow_suppressing_magnitude(tmp_path):
    magnitude = _write(tmp_path / "mag.nii.gz", [[[1.0, 1.0]]])
    phase = _write(tmp_path / "phase_raw.nii.gz", [[[-4096.0, 4095.0]]])
    cartesian = tmp_path / "cartesian"
    assert (
        main(
            [
                "-mag",
                magnitude,
                "-phase",
                phase,
                "-phase_units",
                "scale",
                "-prefix",
                str(cartesian),
                "-device",
                "cpu",
            ]
        )
        == 0
    )
    np.testing.assert_allclose(
        nib.load(tmp_path / "cartesian_real.nii.gz").get_fdata(), [[[-1.0, -1.0]]]
    )
    np.testing.assert_allclose(
        nib.load(tmp_path / "cartesian_imag.nii.gz").get_fdata(), [[[0.0, 0.0]]], atol=1e-6
    )

    polar = tmp_path / "polar.nii.gz"
    assert (
        main(
            [
                "-real",
                str(tmp_path / "cartesian_real.nii.gz"),
                "-imag",
                str(tmp_path / "cartesian_imag.nii.gz"),
                "-nomag",
                "-prefix",
                str(polar),
                "-device",
                "cpu",
            ]
        )
        == 0
    )
    assert not (tmp_path / "polar_mag.nii.gz").exists()
    np.testing.assert_allclose(
        np.abs(nib.load(tmp_path / "polar_phase.nii.gz").get_fdata()), [[[np.pi, np.pi]]]
    )


def test_input_pair_and_output_validation(tmp_path):
    magnitude = _write(tmp_path / "mag.nii.gz", np.ones((2, 2, 2)))
    phase = _write(tmp_path / "phase.nii.gz", np.zeros((2, 2, 2)))
    assert main(["-mag", magnitude, "-prefix", str(tmp_path / "out")]) == 1
    assert (
        main(
            [
                "-mag",
                magnitude,
                "-phase",
                phase,
                "-no_real",
                "-no_imag",
                "-prefix",
                str(tmp_path / "out"),
            ]
        )
        == 1
    )
