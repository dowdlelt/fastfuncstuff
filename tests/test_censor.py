"""Timepoint censoring primitives (processing/censor.py) and the AFNI-faithful automask.

The motion and outlier rules were cross-checked TR for TR against 1d_tool.py
-censor_motion and 3dToutcount on real and synthetic data; these tests pin the
semantics that check depended on, so a refactor cannot drift from them silently.
"""

import numpy as np
import pytest
import torch
from scipy.optimize import linprog

from fastfuncstuff.processing import censor as C
from fastfuncstuff.processing.mask import _afni_fillin_once, automask

CPU = torch.device("cpu")


def test_backward_diff_is_zero_at_every_run_start():
    x = np.arange(10, dtype=float)[:, None] ** 2
    d = C.backward_diff_per_run(x, [4, 6])
    assert d[0, 0] == 0 and d[4, 0] == 0  # the between-run jump never appears
    assert d[5, 0] == 25 - 16


def test_enorm_and_fd_units():
    p = np.zeros((3, 6))
    p[1] = [1.0, 0, 0, 0, 0, 2.0]  # 1 degree roll, 2 mm dP
    e = C.motion_enorm(p)
    fd = C.framewise_displacement(p)
    assert e[1] == pytest.approx(np.sqrt(1 + 4))  # degrees and mm summed raw (AFNI)
    assert fd[1] == pytest.approx(2.0 + np.deg2rad(1.0) * 50.0)
    assert e[0] == 0 and fd[0] == 0


def test_censor_from_trace_limit_is_inclusive_and_prev_is_default():
    trace = np.array([0, 0.1, 0.5, 0.51, 0.2, 0.0])
    keep = C.censor_from_trace(trace, 0.5)
    # 0.5 == limit survives; 0.51 is censored together with the TR before it.
    np.testing.assert_array_equal(keep, [1, 1, 0, 0, 1, 1])
    np.testing.assert_array_equal(
        C.censor_from_trace(trace, 0.5, censor_prev=False), [1, 1, 1, 0, 1, 1]
    )
    np.testing.assert_array_equal(
        C.censor_from_trace(trace, 0.5, censor_next=True), [1, 1, 0, 0, 0, 1]
    )


def test_censor_extensions_stay_inside_a_run_and_first_trs_does_not_spread():
    trace = np.array([0, 0, 0, 0.0, 9, 0, 0, 0])  # flagged at run 2's first TR
    keep = C.censor_from_trace(trace, 0.5, run_lengths=[4, 4], first_trs=1)
    np.testing.assert_array_equal(keep, [0, 1, 1, 1, 0, 1, 1, 1])


def test_outlier_censor_is_strict_and_skip_first_is_per_run():
    frac = np.array([0.5, 0.1, 0.2, 0.5, 0.1, 0.2])
    keep = C.censor_from_outliers(frac, 0.1, run_lengths=[3, 3], skip_first=1)
    np.testing.assert_array_equal(keep, [1, 1, 0, 1, 1, 0])


def test_spikes_one_column_per_censored_tr_and_none_when_clean():
    s = C.censor_to_spikes(np.array([1, 0, 1, 0]))
    np.testing.assert_array_equal(s, [[0, 0], [1, 0], [0, 0], [0, 1]])
    assert C.censor_to_spikes(np.ones(5)).shape == (5, 0)


def test_combine_is_a_product():
    np.testing.assert_array_equal(C.combine_censor([1, 0, 1, 1], [1, 1, 0, 1]), [1, 0, 0, 1])


def test_afni_median_averages_the_middle_pair():
    x = torch.tensor([[1.0], [2.0], [3.0], [10.0]])
    assert float(C.afni_median(x, 0)) == 2.5  # torch.median would say 2.0


def test_default_polort_matches_afni_proc():
    assert C.default_outlier_polort(2.0, 152) == 3  # 1 + floor(304/150)
    assert C.default_outlier_polort(1.0, 100) == 1


def test_l1_detrend_reaches_the_lp_optimum():
    rng = np.random.default_rng(0)
    T, V, p = 60, 5, 3
    X = np.polynomial.legendre.legvander(np.linspace(-1, 1, T), p - 1)
    Y = X @ rng.normal(size=(p, V)) + rng.standard_t(1.5, size=(T, V))  # heavy tails
    r = C.l1_detrend(torch.from_numpy(Y), torch.from_numpy(X)).numpy()
    for v in range(V):
        # min 1'(u+w)  s.t.  X b + u - w = y
        res = linprog(
            np.r_[np.zeros(p), np.ones(2 * T)],
            A_eq=np.hstack([X, np.eye(T), -np.eye(T)]),
            b_eq=Y[:, v],
            bounds=[(None, None)] * p + [(0, None)] * (2 * T),
        )
        assert np.abs(r[:, v]).sum() == pytest.approx(res.fun, rel=1e-8)


def test_outlier_counts_polort0_matches_the_definition():
    rng = np.random.default_rng(1)
    T, V = 40, 200
    y = rng.normal(size=(T, V))
    y[7, :50] += 30.0  # 50 of 200 voxels spike at TR 7
    counts = C.outlier_counts(torch.from_numpy(y).to(CPU), polort=0).numpy()
    med = np.median(y, axis=0)  # T even: numpy averages the middle pair, as AFNI
    r = y - med
    mad = np.median(np.abs(r), axis=0)
    expect = (np.abs(r) > C.outlier_alpha(T) * mad).sum(axis=1)
    np.testing.assert_array_equal(counts, expect)
    assert counts[7] >= 50


def test_outlier_fraction_4d_uses_each_runs_own_polort_and_mask():
    rng = np.random.default_rng(2)
    vol = torch.zeros(12, 12, 12)
    zz, yy, xx = torch.meshgrid(*(torch.arange(12),) * 3, indexing="ij")
    vol[((zz - 6) ** 2 + (yy - 6) ** 2 + (xx - 6) ** 2) < 20] = 100.0
    data = vol[None] + torch.from_numpy(rng.normal(0, 1, (30, 12, 12, 12))).float()
    data[25] += 50.0 * (vol > 0)  # every brain voxel spikes in run 2
    frac, nvox = C.outlier_fraction_4d(data, run_lengths=[15, 15], tr=2.0)
    assert len(nvox) == 2 and frac.shape == (30,)
    assert frac[25] > 0.9 and np.median(frac) < 0.05


def test_fillin_accepts_set_voxels_at_different_distances():
    # Set voxel at +1 on one side and -2 on the other: AFNI fills at nside=2
    # (any within 1..n on each side); a same-distance rule would not.
    m = np.zeros((7, 7, 9), dtype=bool)
    m[3, 3, 5] = True
    m[3, 3, 2] = True
    out, n = _afni_fillin_once(m, 2)
    assert out[3, 3, 4] and n >= 1
    out1, _ = _afni_fillin_once(m, 1)
    assert not out1[3, 3, 4]


def test_automask_keeps_a_solid_head_whole():
    zz, yy, xx = torch.meshgrid(*(torch.arange(32),) * 3, indexing="ij")
    head = (((zz - 16) ** 2 + (yy - 16) ** 2 + (xx - 16) ** 2) < 100).float() * 1000.0
    head += torch.rand(32, 32, 32) * 5.0
    m = automask(head)
    inside = head > 500
    # The peel re-dilates: the boundary shell must come back.
    assert int((m & inside).sum()) == int(inside.sum())


def test_util_outcount_cli_writes_fractions_and_censor(tmp_path):
    import nibabel as nib

    from fastfuncstuff.cli import util_outcount

    rng = np.random.default_rng(4)
    zz, yy, xx = np.meshgrid(*(np.arange(12),) * 3, indexing="ij")
    brain = ((zz - 6) ** 2 + (yy - 6) ** 2 + (xx - 6) ** 2) < 20
    data = 100.0 * brain[..., None] + rng.normal(0, 1, (12, 12, 12, 20))
    data[..., 9] += 50.0 * brain
    img = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
    img.header.set_xyzt_units("mm", "sec")
    img.header["pixdim"][4] = 2.0
    nib.save(img, tmp_path / "run.nii.gz")
    out, cen = tmp_path / "out.1D", tmp_path / "cen.1D"
    util_outcount.main(
        [
            "-input",
            str(tmp_path / "run.nii.gz"),
            "-prefix",
            str(out),
            "-censor_outliers",
            "-censor",
            str(cen),
            "-device",
            "cpu",
            "-verb",
            "0",
        ]
    )
    frac = np.loadtxt(out)
    keep = np.loadtxt(cen)
    assert frac.shape == (20,) and frac[9] > 0.9
    np.testing.assert_array_equal(keep, (frac <= C.DEFAULT_OUTLIER_LIMIT).astype(int))


def _moco_args(*extra):
    from fastfuncstuff.cli import moco

    return moco.parse_args(["-input", "epi.nii.gz", "-1Dfile", "m.1D", *extra])


def test_moco_bare_censor_flags_take_the_ffs_defaults():
    a = _moco_args("-censor_motion", "-censor_outliers", "-censor", "c.1D")
    assert a.censor_motion == C.DEFAULT_MOTION_LIMIT == 0.5
    assert a.censor_outliers == C.DEFAULT_OUTLIER_LIMIT == 0.1
    b = _moco_args("-censor_motion", "0.3", "-censor", "c.1D")
    assert b.censor_motion == 0.3 and b.censor_outliers is None


@pytest.mark.parametrize(
    "extra",
    [("-censor_motion",), ("-censor", "c.1D"), ("-censor_outliers", "1.5", "-censor", "c.1D")],
)
def test_moco_rejects_censor_requests_that_write_nothing_or_are_invalid(extra):
    from fastfuncstuff.cli import moco

    with pytest.raises(SystemExit):
        moco._validate_run_args(_moco_args(*extra))


def test_moco_batch_skip_sees_the_censor_outputs():
    from fastfuncstuff.cli import moco

    a = _moco_args(
        "-enorm", "e.1D", "-fd", "f.1D", "-outcount", "o.1D", "-censor_motion", "-censor", "c.1D"
    )
    moco._validate_run_args(a)
    assert {"e.1D", "f.1D", "o.1D", "c.1D"} <= set(moco._expected_outputs(a))


# --- the :spikes ortvec transform -------------------------------------------


def test_spikes_transform_rejects_anything_but_a_keep_mask():
    from fastfuncstuff.cli_utils import apply_nuisance_transform

    np.testing.assert_array_equal(
        apply_nuisance_transform(np.array([[1], [0], [1]]), "spikes"), [[0], [1], [0]]
    )
    with pytest.raises(ValueError, match="keep mask"):
        apply_nuisance_transform(np.ones((4, 6)), "spikes")  # a motion file by mistake


def _censor_file(tmp_path, keep):
    path = tmp_path / "censor.1D"
    np.savetxt(path, np.asarray(keep), fmt="%d")
    return path


def test_full_length_spikes_get_one_column_per_censored_tr(tmp_path):
    from fastfuncstuff.cli_utils import (
        build_nuisance_block_diag,
        make_nuisance_block_from_full_length,
    )

    keep = np.ones(12, dtype=int)
    keep[[1, 2, 9]] = 0  # two spikes in run 1, one in run 3, none in run 2
    blk = make_nuisance_block_from_full_length(
        _censor_file(tmp_path, keep), "cen", [0, 4, 8], 12, transform="spikes"
    )
    # A full-length file is normally SHARED across runs; spikes must not be, or
    # run 1's first spike and run 3's would collapse into one regressor.
    assert blk.block_diagonal and blk.n_columns == 2
    X = build_nuisance_block_diag(
        blocks=[blk], run_starts=[0, 4, 8], n_timepoints=12, polort=0, device=CPU, verbose=False
    ).numpy()
    spikes = X[:, 3:]  # after the three per-run constants
    assert spikes.shape[1] == 3
    for col, tr in zip(spikes.T, (1, 2, 9), strict=True):
        assert np.argmax(np.abs(col)) == tr  # demeaned per run, peak at the censored TR


def test_spikes_fit_equals_dropping_the_rows():
    from fastfuncstuff.cli_utils import NuisanceBlock, build_nuisance_block_diag

    rng = np.random.default_rng(5)
    T = 40
    keep = np.ones(T, dtype=int)
    keep[[3, 17, 18, 30]] = 0
    task = rng.normal(size=(T, 2))
    y = task @ [1.5, -0.7] + rng.normal(size=T)
    y[keep == 0] += 25.0  # the junk censoring is meant to remove
    blk = NuisanceBlock("cen", [keep[:20, None], keep[20:, None]], transform="spikes")
    N = build_nuisance_block_diag(
        blocks=[blk], run_starts=[0, 20], n_timepoints=T, polort=1, device=CPU, verbose=False
    ).numpy()
    b_spike = np.linalg.lstsq(np.hstack([task, N]), y, rcond=None)[0][:2]
    poly = N[:, :4]  # the two runs' constant + linear
    k = keep == 1
    b_drop = np.linalg.lstsq(np.hstack([task, poly])[k], y[k], rcond=None)[0][:2]
    np.testing.assert_allclose(b_spike, b_drop, atol=1e-5)


def test_design_spec_skips_a_spike_block_with_nothing_censored(tmp_path):
    from fastfuncstuff.cli.design_spec import _materialize_nuisance
    from fastfuncstuff.design.spec import NuisanceSpec

    clean = _censor_file(tmp_path, np.ones(10, dtype=int))
    spec = NuisanceSpec(file=str(clean), label="cen", transform="spikes")
    assert _materialize_nuisance(clean, spec, tmp_path, run_lengths=[5, 5]) is None
    keep = np.ones(10, dtype=int)
    keep[[2, 7]] = 0
    out = _materialize_nuisance(_censor_file(tmp_path, keep), spec, tmp_path, run_lengths=[5, 5])
    np.testing.assert_array_equal(np.flatnonzero(np.loadtxt(out, ndmin=2).sum(axis=1)), [2, 7])


def test_automask_dilate_grows_one_face_layer_like_3dautomask():
    """3dAutomask -dilate adds unset voxels with >= 3 of 18 neighbours set: on a
    box that is exactly one face layer per step (edge-diagonal voxels see only one
    in-mask edge neighbour), so it must not grow 18- or 26-connected corners."""
    head = torch.zeros(24, 30, 30)
    head[6:18, 8:22, 8:22] = 100.0
    base = automask(head)
    grown = automask(head, dilate_extra=1)
    expected = base.clone()
    for ax in range(3):
        expected |= torch.roll(base, 1, ax) | torch.roll(base, -1, ax)
    assert torch.equal(grown, expected)
