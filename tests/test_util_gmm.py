"""Tests for ffs_util_gmm (MELODIC-style mixture modelling of a stat image)."""

import nibabel as nib
import numpy as np
import pytest

from fastfuncstuff.cli.util_gmm import main


def _write(path, arr, affine=None):
    nib.save(nib.Nifti1Image(arr.astype(np.float32), affine or np.eye(4)), str(path))


def _synth(rng, shape3d, n_k, signal_frac=0.05):
    v = int(np.prod(shape3d))
    out = np.zeros((*shape3d, n_k), dtype=np.float32)
    for k in range(n_k):
        x = rng.standard_normal(v)
        idx = rng.choice(v, int(v * signal_frac), replace=False)
        x[idx] += 5.0
        out[..., k] = x.reshape(shape3d)
    return out


def test_util_gmm_recovers_null_and_signal(tmp_path):
    rng = np.random.default_rng(3)
    shape3d = (20, 20, 20)
    data = _synth(rng, shape3d, 3)
    inp = tmp_path / "zstat.nii.gz"
    _write(inp, data)

    main(["-input", str(inp), "-prefix", str(tmp_path / "out"), "-mmthresh", "0", "-verb", "0"])

    z = nib.load(str(tmp_path / "out_zmaps.nii.gz")).get_fdata()
    p = nib.load(str(tmp_path / "out_probmap.nii.gz")).get_fdata()
    assert z.shape == (*shape3d, 3)
    assert p.shape == (*shape3d, 3)
    assert np.isfinite(z).all() and np.isfinite(p).all()
    assert 0.0 <= p.min() and p.max() <= 1.0

    stats = np.loadtxt(tmp_path / "out_mmstats.txt")
    assert stats.shape == (3, 9)
    # pi_noise / pi_pos / pi_neg must be a valid mixture
    for row in stats:
        assert row[2] + row[5] + row[8] == pytest.approx(1.0, abs=1e-4)
        assert row[2] > 0.8  # 5% signal -> the null still dominates

    # The null should end up standardised: most voxels within a few sigma.
    assert np.percentile(np.abs(z), 50) < 1.5
    # ~5% signal planted -> mean P(signal) in the same ballpark, not 0 or 1.
    assert 0.01 < p.mean() < 0.30


def test_util_gmm_drops_constant_voxels(tmp_path):
    """Constant voxels must leave the written mask, not merely be zeroed."""
    rng = np.random.default_rng(4)
    shape3d = (20, 20, 20)
    data = _synth(rng, shape3d, 2)
    data[:, :, 0, :] = 0.0  # one constant slab, 400 voxels
    inp = tmp_path / "zstat.nii.gz"
    _write(inp, data)
    mask = np.ones(shape3d, dtype=np.float32)
    _write(tmp_path / "mask.nii.gz", mask)

    main(
        [
            "-input",
            str(inp),
            "-mask",
            str(tmp_path / "mask.nii.gz"),
            "-prefix",
            str(tmp_path / "out"),
            "-verb",
            "0",
        ]
    )

    out_mask = nib.load(str(tmp_path / "out_mask.nii.gz")).get_fdata() > 0
    assert out_mask.sum() == np.prod(shape3d) - 400
    assert not out_mask[:, :, 0].any()

    # Keeping them must change the answer — that is the whole point of the flag.
    main(
        [
            "-input",
            str(inp),
            "-mask",
            str(tmp_path / "mask.nii.gz"),
            "-prefix",
            str(tmp_path / "keep"),
            "-no_drop_constant",
            "-verb",
            "0",
        ]
    )
    kept_mask = nib.load(str(tmp_path / "keep_mask.nii.gz")).get_fdata() > 0
    assert kept_mask.sum() == np.prod(shape3d)


def test_util_gmm_accepts_3d_input(tmp_path):
    rng = np.random.default_rng(5)
    shape3d = (16, 16, 16)
    data = _synth(rng, shape3d, 1)[..., 0]
    inp = tmp_path / "one.nii.gz"
    _write(inp, data)

    main(["-input", str(inp), "-prefix", str(tmp_path / "o"), "-verb", "0"])
    z = nib.load(str(tmp_path / "o_zmaps.nii.gz")).get_fdata()
    assert z.shape[:3] == shape3d
    assert np.isfinite(z).all()
