"""Depth sampling and voxel mapping must preserve residuals and data units."""

import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from fastfuncstuff.laminar.params import ModelSpec
from fastfuncstuff.processing.io import load_image, save_image
from fastfuncstuff.processing.laminar_tdv import (
    depth_interpolation,
    depth_patches,
    devein_volume_runs,
)

CPU = torch.device("cpu")


def test_patches_never_jump_between_disconnected_banks():
    mask = np.zeros((9, 9, 9), bool)
    mask[:, :, 2:4] = True
    mask[:, :, 5:7] = True
    depth = np.indices(mask.shape)[0] / 8
    labels = depth_patches(mask, depth, np.ones(3), 5)
    z = np.indices(mask.shape)[2][mask]
    assert not set(labels[z < 4]) & set(labels[z > 4])
    assert (labels >= 0).all()


def _runs():
    from fastfuncstuff.laminar.timeseries import linearized_vasculature, vascular_transfer

    T, K = 90, 3
    spec = ModelSpec(N=K, K=K, TR=2, device=CPU)
    freq = torch.fft.rfftfreq(T, spec.TR)
    h, _ = vascular_transfer(linearized_vasculature(spec), freq)
    torch.manual_seed(20)
    a = torch.randn(T, K, dtype=torch.float64) * 0.002
    a = torch.fft.irfft(torch.fft.rfft(a, dim=0) / (1 + (freq[:, None] / 0.04) ** 4), n=T, dim=0)
    y = torch.fft.irfft(torch.einsum("fkn,fn->fk", h, torch.fft.rfft(a, dim=0)), n=T, dim=0).float()
    depth = np.tile(np.array([1 / 6, 0.5, 5 / 6]), 20)
    labels = np.zeros(len(depth), int)
    read = torch.from_numpy(depth_interpolation(depth, K))
    mean = torch.linspace(100, 500, len(depth))[:, None]
    return (
        [mean * (1 + ((read @ y.T) + 0.005 * torch.randn(len(depth), T)) / 100) for _ in range(3)],
        depth,
        labels,
        spec,
    )


@pytest.mark.parametrize("transfer_mode", ["dynamic", "static"])
def test_voxel_mapping_preserves_baseline_units_and_deepest_series(transfer_mode):
    runs, depth, labels, spec = _runs()
    res = devein_volume_runs(
        runs,
        depth,
        labels,
        spec,
        strengths=(0.01,),
        pad_seconds=0,
        transfer_mode=transfer_mode,
        verbose=False,
    )
    assert res.valid.all()
    for raw, out, rem in zip(runs, res.corrected, res.removed_psc, strict=True):
        torch.testing.assert_close(out, raw - rem * raw.mean(-1, keepdim=True) / 100)
        torch.testing.assert_close(out[depth > 0.8], raw[depth > 0.8])
        torch.testing.assert_close(out.mean(-1), raw.mean(-1), rtol=1e-6, atol=1e-5)
        assert rem[depth < 0.3].abs().max() > 0.001
    assert np.isfinite(res.noise_gain).all()


def test_cli_writes_full_grid_and_keeps_outside_voxels(tmp_path):
    runs, depth, labels, spec = _runs()
    shape = (5, 4, 3)
    info = {"affine": np.diag([1.0, 1.0, 1.0, 1.0])}
    # All 60 voxels belong to one supplied patch; one outside it passes through.
    depth = depth.reshape(shape)
    gm = np.ones(shape, np.float32)
    gm[0, 0, 0] = 0
    for i, r in enumerate(runs[:2]):
        save_image(r.T.reshape(r.shape[1], *shape), tmp_path / f"r{i}.nii.gz", info)
    for name, arr in [("depth", depth), ("gm", gm), ("columns", np.ones(shape))]:
        save_image(torch.as_tensor(arr), tmp_path / f"{name}.nii.gz", info)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fastfuncstuff.cli.laminar_tdv",
            "-input",
            "r0.nii.gz",
            "r1.nii.gz",
            "-depth",
            "depth.nii.gz",
            "-gmfrac",
            "gm.nii.gz",
            "-depth_order",
            "pial_to_white",
            "-columns",
            "columns.nii.gz",
            "-depths",
            "3",
            "-extend_mm",
            "0",  # the outside voxel would otherwise be shell
            "-tr",
            "2",
            "-ridge",
            ".01",
            "-te",
            ".025",
            "-b0",
            "7",
            "-device",
            "cpu",
            "-prefix",
            "out",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    raw, _ = load_image(tmp_path / "r0.nii.gz")
    out, _ = load_image(tmp_path / "out_corrected_run-01.nii.gz")
    torch.testing.assert_close(out[:, 0, 0, 0], raw[:, 0, 0, 0])
    assert out.shape == raw.shape
    meta = json.loads((tmp_path / "out_metadata.json").read_text())
    assert meta["n_valid"] == 59

    # Several -ridge values without -repeated_runs: chosen across voxel halves.
    from fastfuncstuff.cli.util_concalc import _read_brick_labels

    res = subprocess.run(
        [sys.executable, "-m", "fastfuncstuff.cli.laminar_tdv", "-input", "r0.nii.gz",
         "r1.nii.gz", "-depth", "depth.nii.gz", "-gmfrac", "gm.nii.gz", "-depth_order",
         "pial_to_white", "-columns", "columns.nii.gz", "-depths", "3", "-extend_mm", "0",
         "-tr", "2", "-ridge", ".01", ".1", "1", "-ridge_cv_mm", "1", "-te", ".025", "-b0",
         "7", "-device", "cpu", "-prefix", "split"],
        cwd=tmp_path, capture_output=True, text=True,
    )  # fmt: skip
    assert res.returncode == 0, res.stdout + res.stderr
    assert json.loads((tmp_path / "split_metadata.json").read_text())["ridge_cv"] == "split"
    import nibabel as nib

    assert "split_leak" in _read_brick_labels(nib.load(tmp_path / "split_ltdv.nii.gz"))


def test_cli_refuses_to_guess_echo_time(tmp_path):
    runs, *_ = _runs()
    info = {"affine": np.eye(4)}
    save_image(runs[0].T.reshape(-1, 5, 4, 3), tmp_path / "r0.nii.gz", info)
    (tmp_path / "r0.json").write_text(json.dumps({"MagneticFieldStrength": 7}))
    from fastfuncstuff.cli.laminar_tdv import main

    with pytest.raises(SystemExit, match="-te"):
        main(["-input", str(tmp_path / "r0.nii.gz"), "-fs_subj", str(tmp_path), "-ridge",
              "0.1", "-removal", "column", "-prefix", str(tmp_path / "o")])  # fmt: skip


@pytest.mark.slow  # 17s
def test_cli_surfaces_remove_deep_drainage_and_leave_the_rest(tmp_path):
    """Sphere-shell ribbon, one deep source: removal grows toward pial, off-ribbon is untouched."""
    import nibabel as nib

    from fastfuncstuff.laminar.timeseries import linearized_vasculature, vascular_transfer
    from tests.viewer.test_surfaces import RADIUS, _subject

    vox, n, T, K, tr = 1.0, 60, 90, 3, 2.0
    aff = np.diag([vox, vox, vox, 1.0])
    aff[:3, 3] = -(n - 1) * vox / 2
    ijk = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1)
    r = np.linalg.norm(ijk * vox + aff[:3, 3], axis=-1)
    frac = np.clip((r - RADIUS) / (0.2 * RADIUS), 0, 1)  # 0 white .. 1 pial
    ribbon = (r > RADIUS) & (r < 1.2 * RADIUS)
    spec = ModelSpec(N=K, K=K, TR=tr, device=CPU)
    freq = torch.fft.rfftfreq(T, tr)
    h, _ = vascular_transfer(linearized_vasculature(spec), freq)
    torch.manual_seed(3)
    a = torch.zeros(T, K, dtype=torch.float64)
    a[:, -1] = torch.randn(T, dtype=torch.float64) * 0.2  # deepest model depth only
    a = torch.fft.irfft(torch.fft.rfft(a, dim=0) / (1 + (freq[:, None] / 0.04) ** 4), n=T, dim=0)
    y = torch.fft.irfft(torch.einsum("fkn,fn->fk", h, torch.fft.rfft(a, dim=0)), n=T, dim=0)
    read = depth_interpolation(1 - frac[ribbon], K)  # model order: 0 = pial
    signal = (read @ y.numpy().T).astype(np.float32)
    rng = np.random.default_rng(0)
    for run in (1, 2):
        data = 100 + rng.normal(0, 0.05, (n, n, n, T)).astype(np.float32)
        data[ribbon] += signal
        nib.save(nib.Nifti1Image(data, aff), str(tmp_path / f"run{run}.nii.gz"))
    res = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "fastfuncstuff.cli.laminar_tdv", "-input", "run1.nii.gz",
         "run2.nii.gz", "-fs_subj", str(_subject(tmp_path)), "-hemi", "lh", "-depths", str(K),
         "-ridge", "0.001", "-te", "0.025", "-b0", "7", "-tr", str(tr), "-pad_seconds", "0",
         "-device", "cpu", "-prefix", "ltdv"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, cwd=tmp_path,
    )  # fmt: skip
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-2000:]
    from fastfuncstuff.cli.util_concalc import _read_brick_labels

    img = nib.load(tmp_path / "ltdv_ltdv.nii.gz")
    names, b = _read_brick_labels(img), np.asanyarray(img.dataobj)
    valid = b[..., names.index("valid")] > 0
    assert valid[ribbon].mean() > 0.8 and not valid[r < 0.8 * RADIUS].any()
    shell = b[..., names.index("shell")] > 0
    assert shell.sum() > 100 and not shell[(r < RADIUS - 3) | (r > 1.2 * RADIUS + 3)].any()
    depth = b[..., names.index("depth")]
    assert np.corrcoef(depth[valid], frac[valid])[0, 1] > 0.8
    rms = b[..., names.index("removed_rms")]
    deep, sup = valid & (frac < 0.25), valid & (frac > 0.75)
    assert np.median(rms[sup]) > 5 * np.median(rms[deep])
    raw = np.asanyarray(nib.load(tmp_path / "run1.nii.gz").dataobj)
    out = np.asanyarray(nib.load(tmp_path / "ltdv_corrected_run-01.nii.gz").dataobj)
    np.testing.assert_array_equal(out[~valid], raw[~valid])
    leak = signal[(frac[ribbon] > 0.75)].std(1).mean()
    assert np.abs(out[sup] - 100).std(-1).mean() < 0.5 * leak


def test_cli_events_choose_the_generating_physiology(tmp_path):
    """Three runs, three different schedules: leave-one-run-out must pick the true t0v."""
    from dataclasses import replace

    from fastfuncstuff.laminar.pipeline import reference_p0
    from fastfuncstuff.laminar.timeseries import (
        cbf_kernel,
        linearized_vasculature,
        vascular_transfer,
    )

    T, K, tr, shape = 150, 3, 2.0, (5, 4, 3)
    # Truth on the CLI's own baselines, so only t0v differs between candidates.
    spec = ModelSpec(N=K, K=K, TR=tr, p0=replace(reference_p0(), t0v=1.0), device=CPU)
    n_fft = T + 30
    h, _ = vascular_transfer(linearized_vasculature(spec), torch.fft.rfftfreq(n_fft, tr))
    kernel = cbf_kernel(spec, dt=0.1).double().numpy()
    depth = np.tile(np.array([1 / 6, 0.5, 5 / 6]), 20)
    read = torch.from_numpy(depth_interpolation(depth, K)).double()
    beta = torch.tensor([[2.0, 1.0], [1.5, 0.5], [1.0, 1.0]], dtype=torch.float64)  # depth x cond
    rng = np.random.default_rng(7)
    info = {"affine": np.eye(4)}
    for run in range(3):
        on = np.round(np.cumsum(rng.uniform(5, 11, 40)), 1)
        on = on[on < T * tr - 20]
        cond = rng.integers(0, 2, on.size)
        (tmp_path / f"ev{run}.tsv").write_text(
            "onset\tduration\ttrial_type\n"
            + "".join(f"{o:.1f}\t2\t{'ab'[c]}\n" for o, c in zip(on, cond, strict=True))
        )
        stim = np.zeros((2, int(T * tr / 0.1)))
        for o, c in zip(on, cond, strict=True):
            stim[c, int(round(o / 0.1)) : int(round((o + 2) / 0.1))] = 1
        X = np.stack([np.convolve(s, kernel)[: s.size][::20][:T] * 0.1 for s in stim], -1)
        Xf = torch.fft.rfft(torch.from_numpy(X), n=n_fft, dim=0)
        Z = torch.fft.irfft(h[..., None] * Xf[:, None, None, :], n=n_fft, dim=0)[:T]
        y = torch.einsum("tknc,nc->tk", Z, beta)  # PSC at model depths
        psc = read @ y.T + 0.05 * torch.randn(len(depth), T, dtype=torch.float64)
        data = (300 * (1 + psc / 100)).float()
        save_image(data.T.reshape(T, *shape), tmp_path / f"r{run}.nii.gz", info)
    save_image(torch.as_tensor(depth.reshape(shape)), tmp_path / "depth.nii.gz", info)
    save_image(torch.ones(shape), tmp_path / "gm.nii.gz", info)
    res = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "fastfuncstuff.cli.laminar_tdv", "-input", "r0.nii.gz",
         "r1.nii.gz", "r2.nii.gz", "-events", "ev0.tsv", "ev1.tsv", "ev2.tsv",
         "-cv_p0", "t0v", "0.5", "1", "2", "-depth", "depth.nii.gz", "-gmfrac", "gm.nii.gz",
         "-depth_order", "pial_to_white", "-patch_mm", "50", "-depths", str(K), "-ridge", "0.1",
         "-te", "0.025", "-b0", "7", "-tr", str(tr), "-polort", "1", "-device", "cpu",
         "-prefix", "out"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, cwd=tmp_path,
    )  # fmt: skip
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-2000:]
    rows = (tmp_path / "out_physiology_cv.tsv").read_text().splitlines()
    assert rows[0].split("\t")[:2] == ["candidate", "t0v"]
    r2 = [float(r.split("\t")[2]) for r in rows[1:]]
    assert int(np.argmax(r2)) == 1  # t0v = 1
    meta = json.loads((tmp_path / "out_metadata.json").read_text())
    assert meta["p0"]["t0v"] == 1.0 and len(meta["physiology_cv"]) == 3


def _vein_runs(n_runs=3, T=160, K=3, vein_gain=5.0, seed=5):
    """One column; three pial-side voxels carry vein_gain x the transported signal."""
    from fastfuncstuff.laminar.timeseries import linearized_vasculature, vascular_transfer

    spec = ModelSpec(N=K, K=K, TR=2, device=CPU)
    freq = torch.fft.rfftfreq(T, spec.TR)
    h, g = vascular_transfer(linearized_vasculature(spec), freq)
    depth = np.tile(np.array([1 / 6, 0.5, 5 / 6]), 20)
    vein = np.zeros(depth.size, bool)
    vein[[0, 3, 6]] = True  # superficial voxels (depth 1/6, model order 0 = pial)
    read = torch.from_numpy(depth_interpolation(depth, K)).double()
    torch.manual_seed(seed)
    runs, local_truth = [], []
    for _ in range(n_runs):
        a = torch.randn(T, K, dtype=torch.float64) * 0.003
        a = torch.fft.irfft(
            torch.fft.rfft(a, dim=0) / (1 + (freq[:, None] / 0.04) ** 4), n=T, dim=0
        )
        af = torch.fft.rfft(a, dim=0)
        loc = torch.fft.irfft(torch.einsum("fkn,fn->fk", g, af), n=T, dim=0)
        tr_ = torch.fft.irfft(torch.einsum("fkn,fn->fk", h - g, af), n=T, dim=0)
        lv, tv = read @ loc.T, read @ tr_.T
        gain = torch.where(torch.from_numpy(vein)[:, None], vein_gain, 1.0)
        psc = lv + gain * tv + 0.002 * torch.randn(depth.size, T, dtype=torch.float64)
        runs.append((200 * (1 + psc / 100)).float())
        local_truth.append(lv)
    return runs, local_truth, depth, vein, spec


def test_voxel_gains_remove_a_veins_excess_and_keep_local_signal():
    from fastfuncstuff.processing.laminar_tdv import devein_volume_runs

    runs, truth, depth, vein, spec = _vein_runs()
    labels = np.zeros(depth.size, int)
    kw = dict(strengths=(0.001,), pad_seconds=0, polort=1, verbose=False)
    col = devein_volume_runs(runs, depth, labels, spec, removal="column", **kw)
    vox = devein_volume_runs(runs, depth, labels, spec, removal="voxel", **kw)

    def err(res, sel):  # corrected PSC vs the local truth, all runs
        out = []
        for raw, cor, lt in zip(runs, res.corrected, truth, strict=True):
            psc = 100 * (cor / raw.mean(-1, keepdim=True) - 1)
            psc = psc - psc.mean(-1, keepdim=True)
            lt = lt - lt.mean(-1, keepdim=True)
            out.append(((psc[sel] - lt[sel].float()) ** 2).mean())
        return float(torch.stack(out).mean())

    superficial = depth < 0.3
    assert err(vox, vein) < 0.2 * err(col, vein)
    normal = superficial & ~vein
    assert err(vox, normal) < 2 * err(col, normal) + 1e-6
    b = vox.gains.beta
    assert np.median(b[vein, 1]) > 3 and abs(np.median(b[normal, 1]) - 1) < 0.5
    np.testing.assert_array_equal(vox.corrected[0][depth > 0.8], runs[0][depth > 0.8])


def test_split_half_ridge_shrinks_the_noisier_patch_without_repeated_runs():
    """Two patches, same signal, 10x different noise, every run a different source
    (nothing repeats, so leave-one-run-out has nothing to predict): split-half picks
    more shrinkage for the noisy patch, and its halves' noise does not leak."""
    from fastfuncstuff.laminar.timeseries import linearized_vasculature, vascular_transfer

    T, K, n = 120, 3, 90
    spec = ModelSpec(N=K, K=K, TR=2, device=CPU)
    freq = torch.fft.rfftfreq(T, spec.TR)
    h, _ = vascular_transfer(linearized_vasculature(spec), freq)
    depth = np.tile(np.array([1 / 6, 0.5, 5 / 6]), 2 * n // 3)
    labels = np.repeat([0, 1], n)
    read = torch.from_numpy(depth_interpolation(depth, K)).double()
    noise = torch.from_numpy(np.repeat([0.002, 0.02], n))[:, None]
    torch.manual_seed(3)
    runs = []
    for _ in range(2):
        a = torch.randn(T, K, dtype=torch.float64) * 0.003
        a = torch.fft.irfft(
            torch.fft.rfft(a, dim=0) / (1 + (freq[:, None] / 0.04) ** 4), n=T, dim=0
        )
        y = torch.fft.irfft(torch.einsum("fkn,fn->fk", h, torch.fft.rfft(a, dim=0)), n=T, dim=0)
        psc = read @ y.T + noise * torch.randn(depth.size, T, dtype=torch.float64)
        runs.append((200 * (1 + psc / 100)).float())
    res = devein_volume_runs(
        runs, depth, labels, spec, strengths=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
        pad_seconds=0, polort=1, ridge_cv="split", split_blocks=np.arange(depth.size),
        split_repeats=3, verbose=False,
    )  # fmt: skip
    quiet, loud = res.regularization[labels == 0][0], res.regularization[labels == 1][0]
    assert loud >= 10 * quiet, (quiet, loud)
    assert np.isfinite(res.cv_r2).all()
    # The leak index is an upper bound: in the quiet patch the little real signal above
    # the model band dominates it (~0.27 measured); where noise dominates it is ~0.
    assert np.abs(res.split_leak[labels == 1]).max() < 0.1


def test_runs_of_different_length_are_each_inverted_on_their_own():
    """At one ridge strength a run's correction depends on that run alone, so a run
    mixed with a shorter one must come out exactly as if it were processed by itself."""
    runs, depth, labels, spec = _runs()
    runs = [runs[0], runs[1][:, :70]]
    kw = dict(strengths=(0.01,), pad_seconds=20, polort=1, removal="column", verbose=False)
    both = devein_volume_runs(runs, depth, labels, spec, **kw)
    for r, run in enumerate(runs):
        alone = devein_volume_runs([run], depth, labels, spec, **kw)
        assert both.corrected[r].shape == run.shape
        torch.testing.assert_close(both.corrected[r], alone.corrected[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(both.removed_psc[r], alone.removed_psc[0], rtol=1e-5, atol=1e-5)
    assert np.isnan(both.cv_r2).all()  # held-out prediction needs a shared schedule


def test_voxel_gains_handle_runs_of_different_length():
    runs, truth, depth, vein, spec = _vein_runs()
    runs[1], truth[1] = runs[1][:, :120], truth[1][:, :120]
    labels = np.zeros(depth.size, int)
    kw = dict(strengths=(0.001,), pad_seconds=0, polort=1, verbose=False)
    col = devein_volume_runs(runs, depth, labels, spec, removal="column", **kw)
    vox = devein_volume_runs(runs, depth, labels, spec, removal="voxel", **kw)

    def err(res, sel):
        out = []
        for raw, cor, lt in zip(runs, res.corrected, truth, strict=True):
            psc = 100 * (cor / raw.mean(-1, keepdim=True) - 1)
            psc = psc - psc.mean(-1, keepdim=True)
            lt = lt - lt.mean(-1, keepdim=True)
            out.append(((psc[sel] - lt[sel].float()) ** 2).mean())
        return float(torch.stack(out).mean())

    assert [c.shape[1] for c in vox.corrected] == [160, 120, 160]
    assert err(vox, vein) < 0.2 * err(col, vein)
    assert np.median(vox.gains.beta[vein, 1]) > 3
    for raw, cor in zip(runs, vox.corrected, strict=True):
        torch.testing.assert_close(cor[depth > 0.8], raw[depth > 0.8], rtol=0, atol=0)


def test_runs_of_different_length_refuse_a_ridge_choice():
    runs, depth, labels, spec = _runs()
    with pytest.raises(ValueError, match="single ridge strength"):
        devein_volume_runs(
            [runs[0], runs[1][:, :70]], depth, labels, spec,
            strengths=(0.01, 0.1), removal="column", verbose=False,
        )  # fmt: skip
