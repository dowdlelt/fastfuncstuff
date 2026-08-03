"""Correctness for fastfuncstuff.stats.dof_adjust (post-NORDIC dof adjustment).

Two layers:
* self-contained: t/F -> z conversions (known values, sign, clamp) and the
  bucket-rewrite logic (insert / update / invalid) with no AFNI needed.
* parity: when the real ``3dcalc`` is present, ``t_to_z`` / ``f_to_z`` must match
  ``fitt_t2z`` / ``fift_t2z`` to float precision.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

from fastfuncstuff.stats.dof_adjust import (
    STAT_FTEST,
    STAT_TTEST,
    STAT_ZSCORE,
    adjust_stats_dof,
    f_to_z,
    t_to_z,
    update_dof_in_file,
)

# ---------------------------------------------------------------------------
# z conversions
# ---------------------------------------------------------------------------


def test_t_to_z_sign_and_zero():
    # z has the sign of t; t=0 -> z=0; symmetric.
    assert t_to_z(0.0, 30) == pytest.approx(0.0, abs=1e-9)
    assert t_to_z(2.5, 30) > 0
    assert t_to_z(-2.5, 30) == pytest.approx(-t_to_z(2.5, 30), rel=1e-6)
    # smaller dof -> heavier tails -> a given t maps to a smaller z.
    assert t_to_z(3.0, 5) < t_to_z(3.0, 100)


def test_z_clamp_at_13_sigma():
    # AFNI cuts qginv off at 13 sigma; huge stats saturate, not inf.
    assert t_to_z(1e6, 50) == pytest.approx(13.0, abs=1e-6)
    assert f_to_z(1e9, 4, 200) == pytest.approx(13.0, abs=1e-6)
    assert np.isfinite(t_to_z(1e6, 50))


def test_f_to_z_monotonic_positive():
    # F -> z is increasing; small F gives small/negative z, large F large z.
    assert f_to_z(0.01, 3, 30) < f_to_z(1.0, 3, 30) < f_to_z(20.0, 3, 30)


@pytest.mark.parametrize("dof", [30, 8, 3])
def test_parity_t_to_z_with_3dcalc(dof):
    calc = shutil.which("3dcalc") or "/opt/mrisoftware/abin/3dcalc"
    if not os.path.exists(calc):
        pytest.skip("3dcalc not found")
    nib = pytest.importorskip("nibabel")
    d = tempfile.mkdtemp()
    tv = np.array([-8, -4, -2, -1, 0, 1, 2, 3, 6, 25], np.float64)
    g = np.zeros((8, 8, 8), np.float32)
    g.ravel()[: len(tv)] = tv
    ip, op = f"{d}/t.nii", f"{d}/tz.nii"
    nib.save(nib.Nifti1Image(g, np.eye(4, dtype=np.float32)), ip)
    subprocess.run(
        [calc, "-a", ip, "-expr", f"fitt_t2z(a,{dof})", "-prefix", op], capture_output=True
    )
    az = nib.load(op).get_fdata().ravel()[: len(tv)]
    assert np.max(np.abs(az - t_to_z(tv, dof))) < 1e-5


@pytest.mark.parametrize("dofnum,dofden", [(3, 30), (1, 8)])
def test_parity_f_to_z_with_3dcalc(dofnum, dofden):
    calc = shutil.which("3dcalc") or "/opt/mrisoftware/abin/3dcalc"
    if not os.path.exists(calc):
        pytest.skip("3dcalc not found")
    nib = pytest.importorskip("nibabel")
    d = tempfile.mkdtemp()
    fv = np.array([0.01, 0.1, 0.5, 1, 2, 3, 5, 8, 15, 40], np.float64)
    g = np.zeros((8, 8, 8), np.float32)
    g.ravel()[: len(fv)] = fv
    ip, op = f"{d}/f.nii", f"{d}/fz.nii"
    nib.save(nib.Nifti1Image(g, np.eye(4, dtype=np.float32)), ip)
    subprocess.run(
        [calc, "-a", ip, "-expr", f"fift_t2z(a,{dofnum},{dofden})", "-prefix", op],
        capture_output=True,
    )
    az = nib.load(op).get_fdata().ravel()[: len(fv)]
    assert np.max(np.abs(az - f_to_z(fv, dofnum, dofden))) < 1e-5


# ---------------------------------------------------------------------------
# bucket rewrite
# ---------------------------------------------------------------------------


def _make_bucket(dof=100.0):
    # Fstat, faces_Coef, faces_Tstat  (AFNI-order)
    x = np.zeros((4, 4, 4, 3), np.float32)
    x[..., 0] = 3.0  # F
    x[..., 1] = 1.5  # coef
    x[..., 2] = 2.5  # t
    labels = ["Full_Fstat", "faces#0_Coef", "faces#0_Tstat"]
    stataux = {0: (STAT_FTEST, (2.0, dof)), 2: (STAT_TTEST, (dof,))}
    return x, stataux, labels


def test_insert_layout_and_stataux():
    x, stataux, labels = _make_bucket(dof=100.0)
    res = adjust_stats_dof(x, stataux, labels, 20.0, verbose=False)
    # Fstat, Zstat, Coef, Tstat, Zstat
    assert res.data.shape[3] == 5
    assert res.labels == [
        "Full_Fstat",
        "Full_Zstat",
        "faces#0_Coef",
        "faces#0_Tstat",
        "faces#0_Zstat",
    ]
    assert res.stataux == {
        0: (STAT_FTEST, (2.0, 100.0)),
        1: (STAT_ZSCORE, ()),
        3: (STAT_TTEST, (100.0,)),
        4: (STAT_ZSCORE, ()),
    }
    # inserted t-z brick equals t_to_z at the reduced dof.
    assert res.data[..., 4][0, 0, 0] == pytest.approx(t_to_z(2.5, 80), rel=1e-5)
    # F-z uses the reduced *denominator* dof, unchanged numerator.
    assert res.data[..., 1][0, 0, 0] == pytest.approx(f_to_z(3.0, 2.0, 80), rel=1e-5)
    assert not res.updated_in_place


def test_update_mode_uses_model_dof_not_prior():
    # First adjust by 20, then re-run by 10: the second run must use the T-stat's
    # model dof (100), not the already-reduced 80, so z = t_to_z(2.5, 90).
    x, stataux, labels = _make_bucket(dof=100.0)
    first = adjust_stats_dof(x, stataux, labels, 20.0, verbose=False)
    second = adjust_stats_dof(first.data, first.stataux, first.labels, 10.0, verbose=False)
    assert second.updated_in_place
    assert second.data.shape[3] == 5  # no new bricks
    assert second.data[..., 4][0, 0, 0] == pytest.approx(t_to_z(2.5, 90), rel=1e-5)


def test_invalid_dof_flagged_and_clamped():
    x, stataux, labels = _make_bucket(dof=10.0)
    res = adjust_stats_dof(x, stataux, labels, 10.0, verbose=False)  # 10-10 = 0
    assert res.invalid.all()  # every voxel invalid
    # clamped to dof=1, so the z is finite (not nan/inf).
    assert np.all(np.isfinite(res.data))


def test_per_voxel_map_partial_invalid():
    x, stataux, labels = _make_bucket(dof=50.0)
    adj = np.full((4, 4, 4), 5.0, np.float32)
    adj[0, 0, 0] = 100.0  # this voxel over-subtracts
    res = adjust_stats_dof(x, stataux, labels, adj, verbose=False)
    assert int(res.invalid.sum()) == 1
    assert res.invalid[0, 0, 0]
    # a valid voxel uses dof 45; the invalid one is clamped but finite.
    assert res.data[..., 4][1, 1, 1] == pytest.approx(t_to_z(2.5, 45), rel=1e-5)
    assert np.isfinite(res.data[..., 4][0, 0, 0])


def test_file_roundtrip_writes_afni_metadata():
    nib = pytest.importorskip("nibabel")  # noqa: F841
    from fastfuncstuff.io.afni import load_nifti, read_brick_labels, read_brick_stataux, save_nifti

    d = tempfile.mkdtemp()
    x, stataux, labels = _make_bucket(dof=120.0)
    src = f"{d}/stats.nii.gz"
    save_nifti(
        x, src, affine=np.eye(4, dtype=np.float32), brick_labels=labels, brick_stataux=stataux
    )

    out = f"{d}/adj.nii.gz"
    res = update_dof_in_file(src, 15.0, out, verbose=False)
    assert res.data.shape[3] == 5

    img = load_nifti(out)
    assert read_brick_labels(img)[4] == "faces#0_Zstat"
    aux = read_brick_stataux(img)
    assert aux[4] == (STAT_ZSCORE, ())
    assert aux[3] == (STAT_TTEST, (120.0,))  # original t brick + dof preserved


def test_file_missing_stataux_errors():
    from fastfuncstuff.io.afni import save_nifti

    d = tempfile.mkdtemp()
    x = np.zeros((4, 4, 4, 2), np.float32)
    src = f"{d}/plain.nii.gz"
    save_nifti(x, src, affine=np.eye(4, dtype=np.float32))  # no stataux
    with pytest.raises(ValueError, match="BRICK_STATAUX"):
        update_dof_in_file(src, 5.0, f"{d}/o.nii.gz", verbose=False)


def test_end_to_end_from_real_reml_bucket():
    # The reml -adjust_dof handoff: a bucket written by write_glm_bucket_as_nifti
    # (stataux via 3drefit) must parse back and adjust cleanly. Needs AFNI.
    if not (shutil.which("3drefit") or os.path.exists("/opt/mrisoftware/abin/3drefit")):
        pytest.skip("3drefit not found")
    pytest.importorskip("nibabel")
    torch = pytest.importorskip("torch")
    from fastfuncstuff.glm.core import GLMResults
    from fastfuncstuff.glm.outputs import write_glm_bucket_as_nifti
    from fastfuncstuff.io.afni import load_nifti, read_brick_labels, read_brick_stataux

    os.environ["PATH"] = "/opt/mrisoftware/abin:" + os.environ.get("PATH", "")
    d = tempfile.mkdtemp()
    r = GLMResults()
    r.betas = torch.randn(64, 2)
    r.tstats = torch.randn(64, 2)
    r.fstats = torch.rand(64) * 5
    r.dof = 100
    r.tr = 2.0
    r.original_shape = (4, 4, 4)
    r.affine = np.eye(4)
    p = str(
        write_glm_bucket_as_nifti(
            r, f"{d}/stats.nii.gz", condition_names=["faces", "places"], apply_afni_metadata=True
        )
    )
    if not read_brick_stataux(load_nifti(p)):
        pytest.skip("3drefit did not embed stataux in this environment")

    update_dof_in_file(p, 12.0, p, verbose=False)  # in place, like ffs_reml -adjust_dof
    img = load_nifti(p)
    labels = read_brick_labels(img)
    assert labels == [
        "Full_Fstat",
        "Full_Zstat",
        "faces#0_Coef",
        "faces#0_Tstat",
        "faces#0_Zstat",
        "places#0_Coef",
        "places#0_Tstat",
        "places#0_Zstat",
    ]
    aux = read_brick_stataux(img)
    assert {i: c for i, (c, _) in aux.items()} == {0: 4, 1: 5, 3: 3, 4: 5, 6: 3, 7: 5}


# ---------------------------------------------------------------------------
# composing multiple adjustments (-adjust_dof repeated, -adjust_dof_set)
# ---------------------------------------------------------------------------


def _save_vol(path, arr):
    from fastfuncstuff.io.afni import save_nifti

    save_nifti(np.asarray(arr, np.float32), path, affine=np.eye(4, dtype=np.float32))


def test_combine_scalars_and_maps():
    from fastfuncstuff.stats.dof_adjust import combine_dof_adjustments

    assert combine_dof_adjustments([]) == 0.0
    assert combine_dof_adjustments([3.0, 4.0]) == 7.0

    m = np.full((2, 2, 2), 5.0, np.float32)
    out = combine_dof_adjustments([2.0, m])
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, 7.0)


def test_combine_rejects_mismatched_grid():
    from fastfuncstuff.stats.dof_adjust import combine_dof_adjustments

    with pytest.raises(ValueError, match="!="):
        combine_dof_adjustments([np.zeros((2, 2, 2), np.float32)], expected_shape=(4, 4, 4))


def test_adjust_dof_set_charges_only_surviving_runs():
    """A voxel that lost run 2 must not be charged run 2's dof cost."""
    from fastfuncstuff.stats.dof_adjust import resolve_dof_adjust_set

    d = tempfile.mkdtemp()
    # 3 runs costing 10, 20, 40 dof; uniform across space.
    per_run = np.zeros((2, 2, 2, 3), np.float32)
    per_run[..., 0], per_run[..., 1], per_run[..., 2] = 10.0, 20.0, 40.0
    inc = np.ones((2, 2, 2, 3), np.float32)
    inc[0, 0, 0, 2] = 0.0  # this voxel lost run 2
    inc[1, 1, 1, :] = [1.0, 0.0, 0.0]  # this one survived only run 0

    _save_vol(f"{d}/perrun.nii.gz", per_run)
    _save_vol(f"{d}/inc.nii.gz", inc)

    adj = resolve_dof_adjust_set(f"{d}/perrun.nii.gz", f"{d}/inc.nii.gz")
    assert adj.shape == (2, 2, 2)
    assert adj[0, 0, 0] == pytest.approx(30.0)  # 10 + 20, not 70
    assert adj[1, 1, 1] == pytest.approx(10.0)
    assert adj[0, 1, 0] == pytest.approx(70.0)  # full survivor pays everything


def test_adjust_dof_set_scalar_per_run():
    from fastfuncstuff.stats.dof_adjust import resolve_dof_adjust_set

    d = tempfile.mkdtemp()
    inc = np.ones((2, 2, 2, 4), np.float32)
    inc[0, 0, 0, :] = [1.0, 1.0, 0.0, 0.0]
    _save_vol(f"{d}/inc.nii.gz", inc)

    adj = resolve_dof_adjust_set(7.0, f"{d}/inc.nii.gz")
    assert adj[0, 0, 0] == pytest.approx(14.0)  # 7 x 2 surviving runs
    assert adj[1, 1, 1] == pytest.approx(28.0)  # 7 x 4


def test_adjust_dof_set_run_count_mismatch_errors():
    from fastfuncstuff.stats.dof_adjust import resolve_dof_adjust_set

    d = tempfile.mkdtemp()
    _save_vol(f"{d}/perrun.nii.gz", np.zeros((2, 2, 2, 3), np.float32))
    _save_vol(f"{d}/inc.nii.gz", np.ones((2, 2, 2, 5), np.float32))
    with pytest.raises(ValueError, match="3 runs but"):
        resolve_dof_adjust_set(f"{d}/perrun.nii.gz", f"{d}/inc.nii.gz")


def test_updatedof_cli_sums_repeated_adjustments():
    """Two -adjust_dof plus an -adjust_dof_set must land at the summed dof."""
    from fastfuncstuff.cli.util_updatedof import main as updatedof_main
    from fastfuncstuff.io.afni import save_nifti

    d = tempfile.mkdtemp()
    x, stataux, labels = _make_bucket(dof=200.0)
    src = f"{d}/stats.nii.gz"
    save_nifti(
        x, src, affine=np.eye(4, dtype=np.float32), brick_labels=labels, brick_stataux=stataux
    )

    vol_shape = x.shape[:3]
    inc = np.ones(vol_shape + (2,), np.float32)
    inc[..., 1] = 0.0  # everyone lost run 1
    _save_vol(f"{d}/inc.nii.gz", inc)

    out = f"{d}/adj.nii.gz"
    rc = updatedof_main(
        [
            "-input",
            src,
            "-adjust_dof",
            "5",
            "-adjust_dof",
            "3",
            "-adjust_dof_set",
            "10",
            f"{d}/inc.nii.gz",
            "-prefix",
            out,
            "-quiet",
        ]
    )
    assert rc == 0

    # total adjustment = 5 + 3 + (10 x 1 surviving run) = 18 -> dof 182
    expected = t_to_z(x[..., 2], 200.0 - 18.0)
    from fastfuncstuff.io.afni import load_nifti

    got = np.asarray(load_nifti(out).get_fdata(dtype=np.float32))[..., 4]
    assert np.allclose(got, expected, atol=1e-4)


def test_updatedof_cli_requires_an_adjustment():
    from fastfuncstuff.cli.util_updatedof import main as updatedof_main

    d = tempfile.mkdtemp()
    assert updatedof_main(["-input", "x.nii.gz", "-prefix", f"{d}/o.nii.gz"]) == 1
