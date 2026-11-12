"""ffs_reml time-series outputs: every flag that is accepted must write its file.

Each of these was accepted, listed in the startup plan, and then silently wrote
nothing: -Rfitts (fitted values never requested from the fit), the -save_*
family under -events (the fit keeps only task betas), -Oerrts/-Ofitts on an
OLS-only run, the OLS outputs beside a REML fit (an internal writer that knew
only -Obuck), -Rerrts/-Rfitts under a per-voxel HRF, and the Ljung-Box brick of
-Rvar under a per-voxel HRF. The referee is the
exact partition data = taskfit + nuisance + residuals.
"""

from __future__ import annotations

import sys

import nibabel as nib
import numpy as np
import pytest
import torch
from scipy.stats import gamma

from fastfuncstuff.glm.outputs import recover_full_betas

SHAPE, T, TR = (6, 5, 4), 80, 2.0


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("remlts")
    rng = np.random.default_rng(0)
    t = np.arange(T) * TR
    box = ((t % 40) < 20).astype(float)
    h = np.exp(-((np.arange(0, 20, TR) - 6) ** 2) / 8)
    x = np.convolve(box, h)[:T]
    x /= x.max()
    n = int(np.prod(SHAPE))
    amp = np.where(np.arange(n) < n // 2, 3.0, 0.0)
    data = 100 + amp[:, None] * x[None] + 0.02 * t[None] + rng.normal(0, 1, (n, T))
    img = nib.Nifti1Image(data.reshape(*SHAPE, T).astype(np.float32), np.eye(4))
    img.header.set_xyzt_units("mm", "sec")
    img.header["pixdim"][4] = TR
    img.to_filename(str(d / "vol.nii"))
    (d / "events.tsv").write_text(
        "onset\tduration\ttrial_type\n"
        + "".join(f"{o}\t20\ttask\n" for o in np.arange(0, T * TR, 40))
    )
    return d


def _reml(monkeypatch, d, *argv):
    from fastfuncstuff.cli import reml

    monkeypatch.chdir(d)
    monkeypatch.setattr(
        sys,
        "argv",
        ["ffs_reml", "-input", "vol.nii", "-events", "events.tsv", "-polort", "2",
         *argv, "-device", "cpu", "-verb", "0"],
    )  # fmt: skip
    reml.main()


def _load(path):
    return np.asarray(nib.load(str(path)).dataobj, dtype=np.float64)


def test_reml_series_partition_the_data(world, monkeypatch):
    out = world / "reml"
    out.mkdir()
    _reml(
        monkeypatch, world,
        "-Rbuck", "reml/rb.nii", "-Rfitts", "reml/fitts.nii", "-Rerrts", "reml/errts.nii",
        "-save_clean", "reml/clean.nii", "-save_nuisance", "reml/nuis.nii",
        "-save_taskfit", "reml/task.nii",
    )  # fmt: skip
    data = _load(world / "vol.nii")
    fitts, errts = _load(out / "fitts.nii"), _load(out / "errts.nii")
    task, nuis, clean = _load(out / "task.nii"), _load(out / "nuis.nii"), _load(out / "clean.nii")
    # float32 storage on ~100-valued data: 1e-6 relative.
    np.testing.assert_allclose(fitts + errts, data, atol=1e-3)
    np.testing.assert_allclose(task + nuis, fitts, atol=1e-3)
    np.testing.assert_allclose(clean, data - nuis + data.mean(-1, keepdims=True), atol=1e-3)
    # The task fit carries the task: active half far above the null half.
    flat = task.reshape(-1, T)
    assert flat[: len(flat) // 2].std() > 5 * flat[len(flat) // 2 :].std()


def test_ols_outputs_written_alone_and_beside_reml(world, monkeypatch):
    (world / "ols").mkdir()
    (world / "both").mkdir()
    _reml(
        monkeypatch, world, "-Obuck", "ols/ob.nii", "-Oerrts", "ols/e.nii", "-Ofitts", "ols/f.nii"
    )
    _reml(
        monkeypatch, world,
        "-Rbuck", "both/rb.nii", "-Obuck", "both/ob.nii", "-Obeta", "both/obeta.nii",
        "-Oerrts", "both/e.nii", "-Ofitts", "both/f.nii",
    )  # fmt: skip
    data = _load(world / "vol.nii")
    np.testing.assert_allclose(
        _load(world / "ols/e.nii") + _load(world / "ols/f.nii"), data, atol=1e-3
    )
    for name in ("ob.nii", "e.nii", "f.nii"):
        np.testing.assert_array_equal(_load(world / "both" / name), _load(world / "ols" / name))
    assert (world / "both/obeta.nii").exists()


def test_nuisance_buckets_refused_before_the_fit(world, monkeypatch):
    with pytest.raises(SystemExit, match="-Onuisance needs a fit of the full design"):
        _reml(monkeypatch, world, "-Obuck", "x.nii", "-Onuisance", "xn.nii")
    assert not (world / "x.nii").exists()


def _hrf(t, k):
    return np.where(t > 0, gamma.pdf(t, k) - gamma.pdf(t, k * 2.6) / 6, 0)


def test_per_voxel_hrf_reml_writes_errts_and_fitts(tmp_path, monkeypatch):
    tt = np.arange(320) * 0.1
    lib = np.stack([_hrf(tt, 3.5), _hrf(tt, 6.0)])
    lib /= lib.max(1, keepdims=True)
    torch.save(
        {"hrf_library": torch.tensor(lib, dtype=torch.float32)}, tmp_path / "lib_hrf_library.pt"
    )
    shape, nt = (4, 3, 3), 160
    idx = (np.indices(shape).sum(0) % 2) + 1
    nib.save(nib.Nifti1Image(idx.astype(np.float32), np.eye(4)), tmp_path / "lib_hrf_index.nii.gz")
    onsets = np.arange(10, nt - 25, 7.3)
    rng = np.random.default_rng(1)
    data = 100 + rng.normal(0, 1, (*shape, nt))
    img = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
    img.header["pixdim"][4] = 1.0
    nib.save(img, tmp_path / "run.nii.gz")
    (tmp_path / "ev.tsv").write_text(
        "onset\tduration\ttrial_type\n" + "".join(f"{o}\t0.3\ta\n" for o in onsets)
    )
    from fastfuncstuff.cli import reml

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["ffs_reml", "-input", "run.nii.gz", "-events", "ev.tsv", "-TR", "1",
         "-hrfopt_prefix", "lib", "-round_durations", "1", "-Rbuck", "rb",
         "-Rerrts", "e.nii", "-Rfitts", "f.nii", "-Rvar", "v.nii",
         "-device", "cpu", "-verb", "0"],
    )  # fmt: skip
    reml.main()
    np.testing.assert_allclose(
        _load(tmp_path / "e.nii") + _load(tmp_path / "f.nii"),
        _load(tmp_path / "run.nii.gz"),
        atol=1e-3,
    )
    # Brick 5 is AFNI's Ljung-Box; the per-HRF merge used to drop it.
    rvar = _load(tmp_path / "v.nii")
    assert rvar.shape[-1] >= 6 and np.any(rvar[..., 5] > 0)


@pytest.mark.parametrize("rank_deficient", [False, True])
def test_recover_full_betas_reproduces_the_fit(rank_deficient):
    rng = np.random.default_rng(3)
    t, n_vox = 60, 7
    x = rng.normal(size=(t, 5))
    if rank_deficient:
        x[:, 4] = x[:, 2] + x[:, 3]  # a nuisance column the others already span
    beta = rng.normal(size=(n_vox, 5))
    pred = (beta @ x.T).astype(np.float32)
    task = [0, 1]
    full = recover_full_betas(beta[:, task], pred, x, task)
    np.testing.assert_allclose(full[:, task], beta[:, task], rtol=1e-6)
    # Exact betas when identifiable, the same fitted signal either way.
    np.testing.assert_allclose(full @ x.T, pred, atol=1e-4)
    if not rank_deficient:
        np.testing.assert_allclose(full, beta, atol=1e-4)
