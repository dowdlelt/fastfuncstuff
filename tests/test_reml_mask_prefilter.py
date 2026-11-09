"""ffs_reml loads only the -mask voxels when nothing before the fit needs others.

The guard and the scaling are voxel-wise, so loading the mask alone must give the
same in-mask bucket as the full-grid path (still taken with whole-volume
diagnostics such as -save_tsnr). The data carries what a resampled slab does:
zeros outside the field of view, and an in-mask voxel that one run lost.
"""

import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest

TR = 2.0
SHAPE = (5, 4, 3)
DEAD = (1, 1, 1)  # in the mask, all zeros in run 2


def _write(tmp_path):
    rng = np.random.default_rng(3)
    paths, events = [], []
    for r in range(2):
        n = 80
        t = np.arange(n) * TR
        arr = 100 + rng.normal(0, 1, (*SHAPE, n))
        onsets = np.arange(8.0, n * TR - 20, 20.0)
        for o in onsets:
            arr += 2 * np.exp(-0.5 * ((t - o - 5) / 2) ** 2)
        arr[3:, :, :] = 0  # outside the slab
        if r == 1:
            arr[DEAD] = 0
        p = tmp_path / f"run{r}.nii.gz"
        img = nib.Nifti1Image(arr.astype(np.float32), np.eye(4))
        img.header.set_xyzt_units(xyz="mm", t="sec")
        img.header.set_zooms((1, 1, 1, TR))
        nib.save(img, p)
        ev = tmp_path / f"ev{r}.tsv"
        ev.write_text("onset\tduration\ttrial_type\n" + "".join(f"{o}\t1\ta\n" for o in onsets))
        paths.append(str(p))
        events.append(str(ev))
    mask = np.zeros(SHAPE, np.uint8)
    mask[:3, :, :2] = 1
    nib.save(nib.Nifti1Image(mask, np.eye(4)), tmp_path / "mask.nii.gz")
    return paths, events, mask.astype(bool)


def _reml(tmp_path, paths, events, prefix, *extra):
    res = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "fastfuncstuff.cli.reml", "-input", *paths, "-events", *events]
        + ["-hrf_model", "SPMG1", "-device", "cpu", "-mask", "mask.nii.gz", "-do_scale"]
        + ["-tout", "-Obuck", prefix, *extra],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        cwd=tmp_path,
    )
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    return res.stdout, np.asanyarray(nib.load(tmp_path / f"{prefix}.nii.gz").dataobj)


@pytest.mark.parametrize("missing", [[], ["-handle_missing", "-guard_min_family", "1"]])
def test_mask_only_load_matches_the_full_grid_path(tmp_path, missing):
    paths, events, mask = _write(tmp_path)
    out, fast = _reml(tmp_path, paths, events, "fast", *missing)
    full_out, full = _reml(tmp_path, paths, events, "full", *missing, "-save_tsnr", "tsnr")
    assert "-mask voxels only" in out and "-mask voxels only" not in full_out
    np.testing.assert_allclose(fast[mask], full[mask], rtol=1e-5, atol=1e-6)
    assert not fast[~mask].any() and not full[~mask].any()
    # The guard still sees the in-mask voxel run 2 lost: dropped, or kept to refit.
    assert bool(fast[DEAD].any()) == bool(missing)
