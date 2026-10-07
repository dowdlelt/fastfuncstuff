"""ffs_reml on surface (GIfTI) data: the GLM through AFNI's 1-D-volume convention.

The referee is the volume path: the same numbers as a (V, 1, 1, T) NIfTI and as a
.func.gii must give the same statistics, with the GIfTI outputs carrying the input's
mesh fingerprint and stat codes. V is past NIfTI-1's 32,767 limit on purpose.
"""

from __future__ import annotations

import sys

import nibabel as nib
import numpy as np
import pytest

from fastfuncstuff.io.gifti import load_gifti_data, save_gifti_data

V, T, TR = 33000, 80, 2.0


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("remlsurf")
    rng = np.random.default_rng(0)
    t = np.arange(T) * TR
    box = ((t % 40) < 20).astype(float)
    h = np.exp(-((np.arange(0, 20, TR) - 6) ** 2) / 8)
    x = np.convolve(box, h)[:T]
    x /= x.max()
    amp = np.where(np.arange(V) < V // 2, 3.0, 0.0)
    data = (100 + amp[:, None] * x[None] + rng.normal(0, 1, (V, T))).astype(np.float32)
    img = nib.Nifti2Image(data[:, None, None, :], np.eye(4))
    img.header.set_xyzt_units("mm", "sec")
    img.header["pixdim"][4] = TR
    img.to_filename(str(d / "vol.nii"))
    save_gifti_data(d / "surf.func.gii", data, {"TR_seconds": "2", "mesh_fingerprint": "V_test"})
    mask = np.ones(V, np.float32)
    mask[:100] = 0
    save_gifti_data(d / "mask.shape.gii", mask, {"mesh_fingerprint": "V_test"}, time_series=False)
    with open(d / "events.tsv", "w") as f:
        f.write("onset\tduration\ttrial_type\n")
        for o in np.arange(0, T * TR, 40):
            f.write(f"{o}\t20\ttask\n")
    return d


def _reml(monkeypatch, *argv):
    from fastfuncstuff.cli import reml

    monkeypatch.setattr(sys, "argv", ["ffs_reml", *argv, "-device", "cpu", "-verb", "0"])
    reml.main()


def test_surface_glm_matches_the_volume_glm_and_keeps_the_mesh(world, monkeypatch):
    common = ["-events", str(world / "events.tsv"), "-polort", "2", "-tout"]
    _reml(monkeypatch, "-input", str(world / "vol.nii"), *common, "-Rbuck", str(world / "v.nii"))
    _reml(
        monkeypatch,
        "-input", str(world / "surf.func.gii"), *common,
        "-Rbuck", str(world / "s.func.gii"),
    )  # fmt: skip
    vol = np.asarray(nib.load(str(world / "v.nii")).dataobj).reshape(V, -1)
    surf, meta = load_gifti_data(world / "s.func.gii")
    np.testing.assert_array_equal(surf, vol)
    assert meta["mesh_fingerprint"] == "V_test"
    arrays = nib.load(str(world / "s.func.gii")).darrays
    t_arr = next(a for a in arrays if dict(a.meta).get("Name") == "task#0_Tstat")
    assert t_arr.intent == 3 and dict(t_arr.meta)["StatCode"] == "3"  # NIFTI_INTENT_TTEST
    assert (world / "s_ffsremlvar.func.gii").exists()


def test_surface_mask_and_diagnostics_stay_on_the_mesh(world, monkeypatch):
    _reml(
        monkeypatch,
        "-input", str(world / "surf.func.gii"), "-events", str(world / "events.tsv"),
        "-Rbuck", str(world / "m.func.gii"), "-mask", str(world / "mask.shape.gii"),
        "-save_tsnr", str(world / "tsnr"),
    )  # fmt: skip
    stats, _ = load_gifti_data(world / "m.func.gii")
    assert (stats[:100] == 0).all() and (stats[100:, 0] != 0).all()
    tsnr, meta = load_gifti_data(world / "tsnr.raw_tsnr.shape.gii")
    assert tsnr.shape == (V,) and meta["mesh_fingerprint"] == "V_test"
    assert 50 < np.median(tsnr[100:]) < 150  # mean 100, sd ~1-2


@pytest.mark.parametrize("flag", [["-do_blur", "4"], ["-save_tsnr", "x"]])
def test_grid_only_options_refuse_surface_input(world, monkeypatch, flag):
    with pytest.raises(SystemExit):
        _reml(
            monkeypatch,
            "-input", str(world / "surf.func.gii"), "-events", str(world / "events.tsv"),
            "-Rbuck", str(world / "x.func.gii"), *flag,
        )  # fmt: skip
