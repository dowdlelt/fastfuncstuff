"""Functional connectivity primitives and the ffs_fc CLI."""

from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pytest
import torch

from fastfuncstuff.connectivity import edge
from fastfuncstuff.connectivity.maps import seed_maps
from fastfuncstuff.connectivity.matrix import (
    FZ_CAP,
    correlation_matrix,
    fisher_z,
    ledoit_wolf,
    partial_correlation,
)
from fastfuncstuff.connectivity.netcc import read_netcc, write_netcc
from fastfuncstuff.connectivity.prep import build_cleaner, stopband_regressors

CPU = torch.device("cpu")


def _legendre(n, deg):
    from fastfuncstuff.glm.core import construct_polynomial_matrix

    return construct_polynomial_matrix(n, deg, CPU, dtype=torch.float64).numpy()


# -- cleaning -------------------------------------------------------------------------


def test_stopband_keeps_the_band_and_removes_the_rest():
    n, tr = 200, 2.0
    sb = stopband_regressors(n, tr, 0.01, 0.1)
    t = np.arange(n)
    inside = np.cos(2 * np.pi * 20 * t / n)  # 20 / (200 * 2) = 0.05 Hz
    outside = np.sin(2 * np.pi * 60 * t / n)  # 0.15 Hz
    q, _ = np.linalg.qr(sb)
    assert np.linalg.norm(q.T @ inside) < 1e-8
    assert np.linalg.norm(outside - q @ (q.T @ outside)) < 1e-8
    # 3dTproject's edge rule removes the edge bins themselves (0.01 Hz is j=4, 0.1 Hz is
    # j=40): j = 1..4 and 40..100, Nyquist (j=100) cosine-only. 3dTproject -verb agrees.
    assert sb.shape[1] == 2 * 4 + 2 * 61 - 1


def test_cleaner_matches_per_run_least_squares_on_kept_rows():
    rng = np.random.default_rng(0)
    lens = [60, 80]
    n = sum(lens)
    y = rng.standard_normal((3, n))
    y[:, :60] += 50.0  # different run means
    keep = np.ones(n, bool)
    keep[[5, 6, 70]] = False
    y[:, 5] += 1e4  # a spike that must not reach the fit
    nuis = [_legendre(L, 2) for L in lens]
    cl = build_cleaner(nuis, [0, 60], n, keep=keep)
    out = cl.apply(torch.as_tensor(y, dtype=torch.float32)).numpy()
    assert out.shape == (3, n - 3)
    expect = []
    for (a, L), x in zip([(0, 60), (60, 80)], nuis, strict=True):
        k = keep[a : a + L]
        xx = x[k]
        yy = y[:, a : a + L][:, k]
        beta = np.linalg.lstsq(xx, yy.T, rcond=None)[0]
        expect.append(yy - (xx @ beta).T)
    np.testing.assert_allclose(out, np.concatenate(expect, 1), atol=2e-3)
    assert cl.n_removed == 6 and cl.dof == n - 3 - 6


def test_cleaner_drops_collinear_columns():
    n = 100
    poly = _legendre(n, 1)
    dup = np.concatenate([poly, 3.0 * poly[:, 1:2]], axis=1)  # a column the drift spans
    cl = build_cleaner([dup], [0], n)
    assert cl.ranks == [2]


# -- matrices -------------------------------------------------------------------------


def test_correlation_matrix_matches_corrcoef_and_zeros_constant_rows():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((5, 90))
    x[3] = 7.0
    r = correlation_matrix(torch.as_tensor(x)).numpy()
    ref = np.corrcoef(np.delete(x, 3, 0))
    np.testing.assert_allclose(np.delete(np.delete(r, 3, 0), 3, 1), ref, atol=1e-12)
    assert np.all(r[3] == 0) and np.all(r[:, 3] == 0)


def test_weighted_correlation_matches_the_formula():
    rng = np.random.default_rng(2)
    a, b = rng.standard_normal((2, 70))
    w = rng.uniform(0, 2, 70)
    ma, mb = (w * a).sum() / w.sum(), (w * b).sum() / w.sum()
    ref = (w * (a - ma) * (b - mb)).sum() / np.sqrt(
        (w * (a - ma) ** 2).sum() * (w * (b - mb) ** 2).sum()
    )
    r = correlation_matrix(torch.as_tensor(np.stack([a, b])), weights=w).numpy()
    assert r[0, 1] == pytest.approx(ref, abs=1e-12)
    # 0/1 weights are the correlation of the selected TRs.
    sel = w > 1
    r01 = correlation_matrix(torch.as_tensor(np.stack([a, b])), weights=sel.astype(float))
    assert float(r01[0, 1]) == pytest.approx(np.corrcoef(a[sel], b[sel])[0, 1], abs=1e-12)


def test_partial_correlation_three_variables():
    rng = np.random.default_rng(3)
    c = rng.standard_normal(500)
    x = np.stack([c + rng.standard_normal(500), c + rng.standard_normal(500), c])
    r = np.corrcoef(x)
    pc, pcb = partial_correlation(r)
    r12, r13, r23 = r[0, 1], r[0, 2], r[1, 2]
    assert pc[0, 1] == pytest.approx((r12 - r13 * r23) / np.sqrt((1 - r13**2) * (1 - r23**2)))
    assert np.allclose(np.diag(pc), 1) and np.allclose(np.diag(pcb), 1)
    # PCB is the regression coefficient of row on column given the rest (unit variances).
    beta = np.linalg.lstsq(np.stack([x[1], x[2]], 1) / x[1:].std(1), x[0] / x[0].std(),
                           rcond=None)[0]  # fmt: skip
    assert pcb[0, 1] == pytest.approx(beta[0], abs=0.02)


def test_partial_correlation_refuses_a_singular_matrix():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((30, 10))  # more ROIs than samples
    with pytest.raises(ValueError, match="shrunk"):
        partial_correlation(np.corrcoef(x))
    lw, s = ledoit_wolf(torch.as_tensor(x))
    assert 0 < s <= 1
    partial_correlation(lw)  # invertible once shrunk


def test_ledoit_wolf_matches_sklearn():
    sk = pytest.importorskip("sklearn.covariance")
    rng = np.random.default_rng(5)
    x = rng.standard_normal((12, 40)) + rng.standard_normal(40)
    lw, s = ledoit_wolf(torch.as_tensor(x))
    z = (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)
    cov, s_ref = sk.ledoit_wolf(z.T)
    assert s == pytest.approx(s_ref, rel=1e-10)
    np.testing.assert_allclose(lw.numpy(), cov, atol=1e-10)


def test_fisher_z_is_capped_like_afni():
    assert float(fisher_z(np.array(1.0))) == pytest.approx(FZ_CAP, abs=1e-6)
    assert float(fisher_z(np.array(0.5))) == pytest.approx(np.arctanh(0.5))


# -- maps and edges ------------------------------------------------------------------


def test_seed_maps_chunked_equal_corrcoef():
    rng = np.random.default_rng(6)
    v = rng.standard_normal((53, 40))
    s = rng.standard_normal((2, 40))
    r = seed_maps(torch.as_tensor(v, dtype=torch.float32),
                  torch.as_tensor(s, dtype=torch.float32), device=CPU, chunk=7).numpy()  # fmt: skip
    ref = np.corrcoef(np.concatenate([v, s]))[:53, 53:]
    np.testing.assert_allclose(r, ref, atol=1e-5)


def test_edge_series_average_to_r_and_rss_identity(tmp_path):
    rng = np.random.default_rng(7)
    x = rng.standard_normal((6, 50)) + rng.standard_normal(50)
    z = edge.zscore_rows(torch.as_tensor(x, dtype=torch.float32))
    iu, ju = edge.edge_index(6)
    e = (z[iu] * z[ju]).numpy()
    np.testing.assert_allclose(e.mean(1), np.corrcoef(x)[iu, ju], atol=1e-5)
    np.testing.assert_allclose(edge.rss(z), np.sqrt((e**2).sum(0)), rtol=1e-5)
    path = tmp_path / "e.npy"
    edge.write_edge_timeseries(z, str(path), max_bytes=4 * 50 * 4)  # force several blocks
    np.testing.assert_allclose(np.load(path), e, atol=1e-6)


def test_netcc_roundtrip(tmp_path):
    r = np.array([[1.0, 0.25], [0.25, 1.0]])
    path = write_netcc(tmp_path / "a.netcc", {"CC": r, "FZ": fisher_z(r)}, [3, 7],
                       ["left V1", "B"])  # fmt: skip
    mats, ids, names = read_netcc(path)
    assert ids == [3, 7] and names == ["left_V1", "B"]
    np.testing.assert_allclose(mats["CC"], r, atol=1e-4)
    assert mats["FZ"][0, 0] == pytest.approx(4.0)


# -- CLI ------------------------------------------------------------------------------


def _write(path, arr, tr=None):
    img = nib.Nifti1Image(np.asarray(arr), np.diag([2.0, 2.0, 2.0, 1.0]))
    if tr is not None:
        img.header.set_xyzt_units("mm", "sec")
        img.header["pixdim"][4] = tr
    nib.save(img, str(path))
    return str(path)


@pytest.fixture
def toy(tmp_path):
    """Two runs, three box ROIs sharing a common signal plus independent noise."""
    rng = np.random.default_rng(8)
    shape, lens = (8, 6, 4), [40, 50]
    labels = np.zeros(shape, np.int16)
    labels[0:3, :, :2] = 1
    labels[4:7, :3, :2] = 2
    labels[4:7, 3:, 2:] = 5
    common = rng.standard_normal(sum(lens))
    runs = []
    for k, L in enumerate(lens):
        a, b = sum(lens[:k]), sum(lens[:k]) + L
        d = 100 + 3 * rng.standard_normal((*shape, L))
        for lab, gain in ((1, 2.0), (2, 1.0), (5, 0.0)):
            d[labels == lab] += gain * common[a:b]
        d += 5.0 * k  # run offset that per-run detrending must remove
        d[7, 5, 3] = 0.0  # an out-of-FOV voxel
        runs.append(_write(tmp_path / f"run{k + 1}.nii.gz", d.astype(np.float32), tr=2.0))
    atlas = _write(tmp_path / "atlas.nii.gz", labels)
    return tmp_path, runs, atlas, labels, lens


def test_cli_matrix_matches_manual_pipeline(toy):
    from fastfuncstuff.cli.fc import main

    tmp, runs, atlas, labels, lens = toy
    prefix = str(tmp / "out" / "fc")
    rc = main(["-input", *runs, "-atlas", atlas, "-polort", "1", "-no_scale", "-part_corr", "-edge",
               "-ts_out", "-graph", "-device", "cpu", "-quiet", "-prefix", prefix])  # fmt: skip
    assert rc == 0
    data = np.concatenate([np.asarray(nib.load(r).dataobj) for r in runs], axis=3)
    means = np.stack([data[labels == lab].mean(0) for lab in (1, 2, 5)])
    resid = []
    for k, L in enumerate(lens):
        a = sum(lens[:k])
        x = _legendre(L, 1)
        y = means[:, a : a + L]
        resid.append(y - (x @ np.linalg.lstsq(x, y.T, rcond=None)[0]).T)
    ref = np.corrcoef(np.concatenate(resid, 1))
    fc = np.load(prefix + "_fc.npz")
    np.testing.assert_allclose(fc["CC"], ref, atol=1e-5)
    assert list(fc["ids"]) == [1, 2, 5]
    mats, ids, _ = read_netcc(prefix + ".netcc")
    assert set(mats) == {"CC", "FZ", "PC", "PCB"} and ids == [1, 2, 5]
    meta = json.loads(open(prefix + ".json").read())
    assert meta["dof"] == sum(lens) - 4 and meta["n_kept"] == sum(lens)
    assert open(prefix + "_rss.tsv").read().count("\n") == sum(lens) + 1
    assert "strength" in open(prefix + "_rois.tsv").readline()


def test_cli_maps_seed_censor_and_weights(toy):
    from fastfuncstuff.cli.fc import main

    tmp, runs, atlas, labels, lens = toy
    n = sum(lens)
    rng = np.random.default_rng(9)
    censor = np.ones(n, int)
    censor[[3, 44]] = 0
    np.savetxt(tmp / "censor.1D", censor, fmt="%d")
    np.savetxt(tmp / "seed.1D", rng.standard_normal(n))
    w = np.zeros((n, 2))
    w[:45, 0], w[45:, 1] = 1, 1
    np.savetxt(tmp / "w.1D", w)
    prefix = str(tmp / "m")
    rc = main(["-input", *runs, "-atlas", atlas, "-seed_1D", str(tmp / "seed.1D"),
               "-censor", str(tmp / "censor.1D"), "-weight_corr", str(tmp / "w.1D"),
               "-wb_corr", "-device", "cpu", "-quiet", "-prefix", prefix])  # fmt: skip
    assert rc == 0
    fc = np.load(prefix + "_fc.npz")
    assert {"CC_W1", "FZ_W1", "CC_W2"} <= set(fc.files)
    assert list(fc["ids"]) == [1, 2, 5, -1]
    meta = json.loads(open(prefix + ".json").read())
    assert meta["n_kept"] == n - 2
    wb = np.asarray(nib.load(prefix + "_wb_r_W1.nii.gz").dataobj)
    assert wb.shape == (*labels.shape, 4)
    assert wb[7, 5, 3].tolist() == [0, 0, 0, 0]  # constant voxel left out
    # ROI 1 is driven by the common signal: its voxels correlate with its own mean.
    assert wb[labels == 1][:, 0].mean() > 0.5


def test_cli_rejects_seed_without_maps(toy, capsys):
    from fastfuncstuff.cli.fc import main

    tmp, runs, *_ = toy
    np.savetxt(tmp / "s.1D", np.zeros(90))
    assert main(["-input", *runs, "-seed_1D", str(tmp / "s.1D"), "-quiet",
                 "-prefix", str(tmp / "x")]) == 2  # fmt: skip
    assert "needs -wb_corr" in capsys.readouterr().err


def test_cli_default_scales_voxels_before_averaging(toy):
    """A bright voxel must not outweigh its ROI: means are of per-run percent change."""
    from fastfuncstuff.cli.fc import main

    tmp, runs, atlas, labels, lens = toy
    rng = np.random.default_rng(10)
    own = rng.standard_normal(sum(lens))
    bright = []
    for k, path in enumerate(runs):
        a = sum(lens[:k])
        d = np.asarray(nib.load(path).dataobj).copy()
        d[0, 0, 0] = 2000 + 40 * own[a : a + lens[k]]  # 20x brighter, its own signal
        bright.append(_write(tmp / f"b{k}.nii.gz", d, tr=2.0))
    prefix = str(tmp / "s")
    assert main(["-input", *bright, "-atlas", atlas, "-polort", "1",
                 "-device", "cpu", "-quiet", "-prefix", prefix]) == 0  # fmt: skip
    data = [np.asarray(nib.load(r).dataobj, dtype=np.float64) for r in bright]
    with np.errstate(invalid="ignore"):  # the out-of-FOV voxel, in no ROI
        psc = np.concatenate(
            [np.minimum(100 * d / d.mean(-1, keepdims=True), 200) for d in data], 3
        )
    resid = []
    for k, L in enumerate(lens):
        a = sum(lens[:k])
        x = _legendre(L, 1)
        y = np.stack([psc[labels == lab][:, a : a + L].mean(0) for lab in (1, 2, 5)])
        resid.append(y - (x @ np.linalg.lstsq(x, y.T, rcond=None)[0]).T)
    ref = np.corrcoef(np.concatenate(resid, 1))
    np.testing.assert_allclose(np.load(prefix + "_fc.npz")["CC"], ref, atol=1e-4)
    assert json.loads(open(prefix + ".json").read())["scaled"] is True


def test_partial_correlation_rank_test_uses_dof():
    rng = np.random.default_rng(11)
    r = np.corrcoef(rng.standard_normal((5, 200)))  # well conditioned on its own
    partial_correlation(r, dof=50)
    with pytest.raises(ValueError, match="rank at most"):
        partial_correlation(r, dof=5)


def test_cli_too_many_rois_falls_back_to_shrunk_pc(toy):
    """One ROI per voxel (191 ROIs) from 86 residual dof: no PC, but PC_LW is written."""
    from fastfuncstuff.cli.fc import main

    tmp, runs, _, labels, lens = toy
    per_voxel = np.arange(1, labels.size + 1, dtype=np.int32).reshape(labels.shape)
    atlas = _write(tmp / "voxels.nii.gz", per_voxel)
    prefix = str(tmp / "lw")
    assert main(["-input", *runs, "-atlas", atlas, "-polort", "1", "-part_corr",
                 "-shrink", "lw", "-device", "cpu", "-quiet", "-prefix", prefix]) == 0  # fmt: skip
    fc = np.load(prefix + "_fc.npz")
    assert "PC_LW" in fc.files and "PC" not in fc.files
    assert "rank at most" in json.loads(open(prefix + ".json").read())["pc_skipped"]
    # Without -shrink the same request is an error, not a silent garbage matrix.
    assert main(["-input", *runs, "-atlas", atlas, "-polort", "1", "-part_corr",
                 "-device", "cpu", "-quiet", "-prefix", prefix + "2"]) == 2  # fmt: skip


@pytest.mark.gpu
def test_cleaner_is_exact_with_tf32_switched_on():
    """TF32 left on by another tool must not leak nuisance through the projection."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    rng = np.random.default_rng(12)
    n, tr = 360, 1.0
    y = 100 + rng.standard_normal((300, n))
    cl = build_cleaner([_legendre(n, 2)], [0], n, tr=tr, passband=(0.01, 0.1))
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        out = cl.apply(torch.as_tensor(y, dtype=torch.float32), device=torch.device("cuda"))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    s = np.linalg.svd(out.cpu().numpy().astype(np.float64), compute_uv=False)
    assert (s > 1e-4 * s[0]).sum() == cl.dof


def test_ledoit_wolf_shrinks_more_with_fewer_effective_samples():
    rng = np.random.default_rng(13)
    x = torch.as_tensor(rng.standard_normal((40, 300)) + rng.standard_normal(300))
    _, s_all = ledoit_wolf(x)
    _, s_eff = ledoit_wolf(x, n_effective=60)
    assert s_eff == pytest.approx(min(1.0, s_all * 5), rel=1e-6) or s_eff == 1.0


def test_passband_cost_counts_only_what_drift_does_not_already_remove():
    n, tr = 360, 1.0
    poly = _legendre(n, 3)
    cl = build_cleaner([poly], [0], n, tr=tr, passband=(0.01, 0.1))
    assert cl.band_free_ranks == [4]
    # 0.01-0.1 Hz at TR 1, 360 TRs: 295 stopband columns, as 3dTproject -verb reports;
    # some of their span the Legendre drift already covers.
    assert cl.passband_cost <= 295 and cl.ranks[0] == cl.passband_cost + 4
    assert build_cleaner([poly], [0], n).passband_cost == 0


@pytest.mark.parametrize("band", [(0.01, 0.1), (0.01, 0.0), (0.0, 0.2)])
def test_fft_and_regression_cleaning_agree(band):
    """The two algorithms are the same projection: same residual, same dof."""
    rng = np.random.default_rng(14)
    lens, tr = [120, 97], 1.5
    n = sum(lens)
    y = 100 + rng.standard_normal((20, n)).cumsum(1) * 0.1 + rng.standard_normal((20, n))
    nuis = [np.concatenate([_legendre(L, 2), rng.standard_normal((L, 3))], 1) for L in lens]
    kw = dict(tr=tr, passband=band)
    reg = build_cleaner(nuis, [0, 120], n, method="regression", **kw)
    fft = build_cleaner(nuis, [0, 120], n, method="fft", **kw)
    assert all(m is not None for m in fft.band_masks)
    assert reg.ranks == fft.ranks and reg.dof == fft.dof
    yt = torch.as_tensor(y, dtype=torch.float32)
    np.testing.assert_allclose(fft.apply(yt).numpy(), reg.apply(yt).numpy(), atol=2e-3)


def test_auto_cleaning_uses_regression_for_a_censored_run():
    n = 100
    keep = np.ones(2 * n, bool)
    keep[n + 5] = False
    cl = build_cleaner([_legendre(n, 1)] * 2, [0, n], 2 * n, keep=keep, tr=1.0,
                       passband=(0.01, 0.1))  # fmt: skip
    assert cl.band_masks[0] is not None and cl.band_masks[1] is None
    with pytest.raises(ValueError, match="censored"):
        build_cleaner([_legendre(n, 1)] * 2, [0, n], 2 * n, keep=keep, tr=1.0,
                      passband=(0.01, 0.1), method="fft")  # fmt: skip


def test_cli_refuses_percent_change_without_a_baseline(toy, capsys):
    from fastfuncstuff.cli.fc import main

    tmp, runs, atlas, labels, lens = toy
    demeaned = []
    for k, path in enumerate(runs):
        d = np.asarray(nib.load(path).dataobj)
        demeaned.append(_write(tmp / f"r{k}.nii.gz", d - d.mean(-1, keepdims=True), tr=2.0))
    args = ["-input", *demeaned, "-atlas", atlas, "-device", "cpu", "-quiet"]
    assert main([*args, "-prefix", str(tmp / "nb")]) == 2
    assert "-no_scale" in capsys.readouterr().err
    assert main([*args, "-no_scale", "-prefix", str(tmp / "nb2")]) == 0


def test_cli_plot(toy):
    from fastfuncstuff.cli.fc import main

    tmp, runs, atlas, *_ = toy
    prefix = str(tmp / "p")
    assert main(["-input", *runs, "-atlas", atlas, "-edge", "-plot", "-device", "cpu",
                 "-quiet", "-prefix", prefix]) == 0  # fmt: skip
    assert open(prefix + "_fc.png", "rb").read(8) == b"\x89PNG\r\n\x1a\n"


def test_plot_draws_only_major_module_boundaries():
    from fastfuncstuff.connectivity.plots import _major_boundaries

    # modules 1, 40, 1, 1, 50, 2: the singletons fold into the 40; one line, at the 50
    assert _major_boundaries([1, 41, 42, 43, 93], 95, 3) == [43]
    assert _major_boundaries([1, 41, 93], 95, 3) == [41]
    assert _major_boundaries([], 10, 3) == []


def test_cli_lw_samples_tr_is_the_textbook_shrinkage(toy):
    from fastfuncstuff.cli.fc import main

    tmp, runs, atlas, *_ = toy
    out = {}
    for mode in ("dof", "tr"):
        prefix = str(tmp / f"lw_{mode}")
        assert main(["-input", *runs, "-atlas", atlas, "-shrink", "lw", "-lw_samples", mode,
                     "-passband", "0.01", "0.1", "-device", "cpu", "-quiet",
                     "-prefix", prefix]) == 0  # fmt: skip
        out[mode] = json.loads(open(prefix + ".json").read())
    assert out["tr"]["lw_samples"] == "tr"
    assert out["dof"]["lw_shrinkage"] > out["tr"]["lw_shrinkage"]


def test_cleaner_keep_mean_restores_each_rows_level():
    rng = np.random.default_rng(15)
    n = 80
    y = 500 + rng.standard_normal((4, n)) + 0.1 * np.arange(n)
    cl = build_cleaner([_legendre(n, 1)], [0], n)
    yt = torch.as_tensor(y, dtype=torch.float32)
    kept = cl.apply(yt, keep_mean=True).numpy()
    plain = cl.apply(yt).numpy()
    np.testing.assert_allclose(kept.mean(1), y.mean(1), rtol=1e-5)
    np.testing.assert_allclose(kept - kept.mean(1, keepdims=True), plain, atol=1e-3)


# -- cross-validation across runs ----------------------------------------------------


def _three_runs(rng, n=6, lens=(50, 70, 60)):
    x = rng.standard_normal((n, sum(lens))) + rng.standard_normal(sum(lens))
    starts = list(np.cumsum([0, *lens[:-1]]))
    return torch.as_tensor(x), starts, lens


def test_cross_products_are_exact_for_any_subset_of_runs():
    from fastfuncstuff.connectivity.xval import cross_products

    rng = np.random.default_rng(16)
    x, starts, lens = _three_runs(rng)
    cp = cross_products(x, starts)
    np.testing.assert_allclose(cp.correlation(), correlation_matrix(x), atol=1e-12)
    cols = np.r_[0 : lens[0], lens[0] + lens[1] : sum(lens)]  # runs 1 and 3
    np.testing.assert_allclose(cp.correlation([0, 2]), correlation_matrix(x[:, cols]), atol=1e-12)
    w = rng.uniform(0, 2, sum(lens))
    cpw = cross_products(x, starts, weights=w)
    np.testing.assert_allclose(cpw.correlation(), correlation_matrix(x, weights=w), atol=1e-12)


def test_loro_compares_each_run_with_the_others():
    from fastfuncstuff.connectivity.xval import cross_products, loro_reproducibility

    rng = np.random.default_rng(17)
    x, starts, lens = _three_runs(rng)
    res = loro_reproducibility(cross_products(x, starts))
    a = lens[0]
    np.testing.assert_allclose(res.per_run[0], correlation_matrix(x[:, :a]), atol=1e-12)
    assert res.reproducibility.shape == (3,) and np.all(np.abs(res.reproducibility) <= 1)


def test_shrinkage_cv_shrinks_more_when_runs_are_short():
    """Held-out likelihood wants more shrinkage when each run carries little information."""
    from fastfuncstuff.connectivity.xval import cross_products, shrinkage_cv

    rng = np.random.default_rng(18)
    n = 40
    true = np.linalg.cholesky(0.7 * np.eye(n) + 0.3)  # one shared factor
    best = {}
    for t in (30, 600):
        x = true @ rng.standard_normal((n, 4 * t))
        best[t] = shrinkage_cv(cross_products(torch.as_tensor(x), [0, t, 2 * t, 3 * t])).best
    assert best[30] > best[600]
    # Not ~0 even at 1800 training samples: with N/T = 0.022 the smallest sample
    # eigenvalues sit ~30% low (Marchenko-Pastur), and the identity target corrects
    # that. With the TRUE matrix as the training estimate the optimum is 0.02
    # (checked over 20 repeats); with a sample estimate it is 0.15.
    assert 0.0 < best[600] < 0.3


def test_cli_per_run_loro_and_cv_shrinkage(toy):
    from fastfuncstuff.cli.fc import main

    tmp, runs, atlas, labels, lens = toy
    prefix = str(tmp / "xv")
    assert main(["-input", *runs, "-atlas", atlas, "-per_run", "-loro", "-shrink", "cv",
                 "-part_corr", "-device", "cpu", "-quiet", "-prefix", prefix]) == 0  # fmt: skip
    fc = np.load(prefix + "_fc.npz")
    assert fc["CC_runs"].shape == (2, 3, 3) and {"CV", "PC_CV"} <= set(fc.files)
    meta = json.loads(open(prefix + ".json").read())
    assert -1 <= meta["loro_mean"] <= 1
    assert {"best", "lw_dof", "lw_tr"} <= set(meta["shrink_cv"])
    assert open(prefix + "_loro.tsv").read().count("\n") == 3
