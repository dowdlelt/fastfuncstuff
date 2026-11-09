"""ffs_moco -stack: cross-run alignment as one short series, one allineate-form matrix per run."""

import nibabel as nib
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift as ndshift

from fastfuncstuff.cli.moco import main


def _slab_runs(tmp_path):
    """A 16-slice slab cut from a taller volume, as three runs a few slices apart."""
    rng = np.random.default_rng(5)
    tall = gaussian_filter(rng.normal(size=(40, 40, 28)), 1.5)  # (x, y, z) as NIfTI
    tall = (tall - tall.min()) / np.ptp(tall) + 0.5
    affine = np.diag([2.0, 2.0, 1.5, 1.0])
    paths, last = [], None
    for i, dz in enumerate((0.0, 2.0, 3.5)):
        last = ndshift(tall, (0.3, -0.2, -dz), order=3, mode="nearest")[:, :, 2:18]
        p = tmp_path / f"run{i}.nii.gz"  # the runs see x 2..37 of the volume
        nib.save(nib.Nifti1Image(last[2:-2].astype(np.float32), affine), p)
        paths.append(p)
    # The last run again on a WIDER grid (all 40 x voxels, header moved to
    # match): the same image in scanner space, covering the base fully.
    wide_aff = affine.copy()
    wide_aff[0, 3] -= 2 * affine[0, 0]
    wide = tmp_path / "run2_wide.nii.gz"
    nib.save(nib.Nifti1Image(last.astype(np.float32), wide_aff), wide)
    return paths, wide


@pytest.mark.parametrize("twopass", [False, True])
def test_stack_writes_one_allineate_matrix_per_source(tmp_path, twopass):
    (base, r1, r2), wide = _slab_runs(tmp_path)
    mats = [tmp_path / f"m{i}.aff12.1D" for i in range(3)]
    aligned = [tmp_path / f"a{i}.nii.gz" for i in range(3)]
    argv = ["-stack", str(base), str(r1), str(r2), str(wide),
            "-stack_matrices", *map(str, mats), "-stack_aligned", *map(str, aligned),
            "-device", "cpu", "-verb", "0"] + (["-twopass"] if twopass else [])  # fmt: skip
    main(argv)
    rows = [np.loadtxt(m) for m in mats]
    assert all(r.shape == (12,) for r in rows)  # allineate's single-row form, no header
    # How far each matrix moves the slab centre along z (scanner mm): 2 and 3.5
    # slices of 1.5 mm. The matrix's own translation column also carries any small
    # rotation times the distance to the origin, so it is not the shift itself.
    img = nib.load(base)
    centre = img.affine @ np.r_[(np.array(img.shape) - 1) / 2, 1.0]
    centre[:2] *= -1  # .aff12 matrices are DICOM: x and y negated from NIfTI RAS
    moved = np.array([r.reshape(3, 4) @ centre - centre[:3] for r in rows[:2]])
    np.testing.assert_allclose(np.abs(moved[:, 2]), [3.0, 5.25], atol=0.15)
    assert np.abs(moved[:, :2]).max() < 0.1  # every run shares the in-plane offset
    # a source on another grid gets the matrix of the same image on the base grid
    np.testing.assert_allclose(rows[2], rows[1], atol=0.05)
    assert nib.load(aligned[0]).shape == nib.load(base).shape


def test_stack_refuses_mismatched_outputs(tmp_path):
    (base, r1, r2), _ = _slab_runs(tmp_path)
    with pytest.raises(SystemExit):
        main(["-stack", str(base), str(r1), str(r2), "-stack_matrices", str(tmp_path / "m.1D"),
              "-device", "cpu"])  # fmt: skip
