"""Multi-echo registration and timepoint trimming for ffs_moco.

Covers the Pass-2 split (resample_timeseries) and the CLI orchestration that
estimates motion from one echo (or the cross-echo mean) and applies the shared
transforms to every echo, plus -skip_first/-skip_last trimming.
"""

import numpy as np
import torch

from fastfuncstuff.cli.moco import main
from fastfuncstuff.processing.affine import (
    _build_homo_coords,
    identity_params,
    params_to_matrix,
    resample_affine_fast,
)
from fastfuncstuff.processing.ffs_moco import MocoConfig, moco, resample_timeseries
from fastfuncstuff.processing.io import load_image, save_image

DEV = torch.device("cpu")


def _blob(shape=(10, 12, 12)):
    nz, ny, nx = shape
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32),
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    cx, cy, cz = (nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
    return torch.exp(-r2 / (2 * 3.0**2))


def _shifted_series(base, n_vols=4, max_shift=0.6):
    shape = base.shape
    ts = torch.zeros(n_vols, *shape)
    ts[0] = base
    coords = _build_homo_coords(shape, DEV, torch.float32)
    for t in range(1, n_vols):
        p = identity_params(device=DEV, dtype=torch.float32)
        p[0] = max_shift * (t / n_vols)
        p[1] = -max_shift * (t / n_vols) * 0.5
        ts[t] = resample_affine_fast(base, params_to_matrix(p), coords, "heptic", shape)
    return ts


# ---------------------------------------------------------------------------
# resample_timeseries matches moco's internal Pass 2
# ---------------------------------------------------------------------------


def test_resample_timeseries_matches_moco():
    """Splitting Pass 2 out must reproduce moco's own aligned output exactly."""
    ts = _shifted_series(_blob())
    cfg = MocoConfig(device="cpu", verb=0, compile=False)

    full = moco(ts, cfg)  # estimate + resample in one call

    # Re-estimate only, then resample with the standalone helper.
    cfg_est = MocoConfig(device="cpu", verb=0, compile=False, skip_resample=True)
    est = moco(ts, cfg_est)
    aligned, _ = resample_timeseries(ts, est.matrices_vox, cfg, DEV, base_copy_idx=cfg.base_index)

    assert torch.allclose(aligned, full.aligned, atol=1e-5)


# ---------------------------------------------------------------------------
# Multi-echo CLI: shared transforms applied to every echo
# ---------------------------------------------------------------------------


def _write(path, tensor):
    save_image(tensor, str(path), header_info=None)


def test_multiecho_applies_shared_transforms(tmp_path):
    """Estimate from echo 1, apply to both echoes. Echo 2 = 2x echo 1, so the
    aligned outputs must keep that factor (resampling is linear)."""
    base = _blob()
    echo1 = _shifted_series(base)
    echo2 = echo1 * 2.0  # same motion, different intensity (like a later echo)

    e1_in = tmp_path / "e1.nii.gz"
    e2_in = tmp_path / "e2.nii.gz"
    _write(e1_in, echo1)
    _write(e2_in, echo2)

    prefix = tmp_path / "mc.nii.gz"
    main(
        [
            "-input",
            str(e1_in),
            str(e2_in),
            "-reg_echo",
            "1",
            "-prefix",
            str(prefix),
            "-device",
            "cpu",
            "-verb",
            "0",
        ]
    )

    out1 = tmp_path / "e1_mc.nii.gz"
    out2 = tmp_path / "e2_mc.nii.gz"
    assert out1.exists() and out2.exists()

    a1, _ = load_image(str(out1))
    a2, _ = load_image(str(out2))
    assert a1.shape == echo1.shape
    # Same transforms applied to 2x data -> 2x aligned output.
    assert torch.allclose(a2, a1 * 2.0, atol=1e-4)


def test_multiecho_mean_and_params(tmp_path):
    """-reg_echo mean writes both echoes and a single shared motion file."""
    base = _blob()
    echo1 = _shifted_series(base)
    echo2 = echo1 * 1.5

    e1_in = tmp_path / "e1.nii.gz"
    e2_in = tmp_path / "e2.nii.gz"
    _write(e1_in, echo1)
    _write(e2_in, echo2)

    prefix = tmp_path / "mc.nii.gz"
    oned = tmp_path / "motion.1D"
    main(
        [
            "-input",
            str(e1_in),
            str(e2_in),
            "-reg_echo",
            "mean",
            "-prefix",
            str(prefix),
            "-save_mean",
            "-1Dfile",
            str(oned),
            "-device",
            "cpu",
            "-verb",
            "0",
        ]
    )

    assert (tmp_path / "e1_mc.nii.gz").exists()
    assert (tmp_path / "e2_mc.nii.gz").exists()
    # -save_mean (no value) derives mean_{eN_prefix}.
    assert (tmp_path / "mean_e1_mc.nii.gz").exists()
    assert (tmp_path / "mean_e2_mc.nii.gz").exists()
    # One shared motion file for the whole multi-echo run.
    params = np.loadtxt(oned)
    assert params.shape == (echo1.shape[0], 6)


def test_multiecho_requires_reg_echo(tmp_path):
    """Multiple inputs without -reg_echo is an error."""
    import pytest

    e1_in = tmp_path / "e1.nii.gz"
    e2_in = tmp_path / "e2.nii.gz"
    _write(e1_in, _shifted_series(_blob()))
    _write(e2_in, _shifted_series(_blob()))

    with pytest.raises(SystemExit):
        main(
            [
                "-input",
                str(e1_in),
                str(e2_in),
                "-prefix",
                str(tmp_path / "mc.nii.gz"),
                "-device",
                "cpu",
                "-verb",
                "0",
            ]
        )


# ---------------------------------------------------------------------------
# -skip_first / -skip_last trimming
# ---------------------------------------------------------------------------


def test_skip_first_last_trims_volumes(tmp_path):
    """Trimming drops the requested volumes from each end before registration."""
    ts = _shifted_series(_blob(), n_vols=6)
    in_path = tmp_path / "epi.nii.gz"
    _write(in_path, ts)

    out = tmp_path / "epi_mc.nii.gz"
    main(
        [
            "-input",
            str(in_path),
            "-prefix",
            str(out),
            "-skip_first",
            "1",
            "-skip_last",
            "2",
            "-device",
            "cpu",
            "-verb",
            "0",
        ]
    )

    aligned, _ = load_image(str(out))
    assert aligned.shape[0] == 6 - 1 - 2  # 3 volumes survive


# ---------------------------------------------------------------------------
# -me_3depi: TE-dependent partition-axis shift, folded into the moco matrices
# ---------------------------------------------------------------------------


def test_me_3depi_removes_the_te_dependent_shift(tmp_path):
    """Echoes displaced by m*TE along z must come out co-registered in z.

    The regression test for the whole -me_3depi chain: estimate, TE fit through
    the origin, fold into each echo's matrix, one resample. A sign error anywhere
    would double the inter-echo shift instead of removing it.
    """
    from fastfuncstuff.processing.shiftcorr import apply_shift, estimate_pair_shift

    tes = [7.6, 21.7, 35.8]
    slope = -0.012  # voxels per ms of TE
    base = _shifted_series(_blob(shape=(24, 20, 20)), n_vols=3)

    paths = []
    for i, te in enumerate(tes):
        p = tmp_path / f"e{i + 1}.nii.gz"
        _write(p, apply_shift(base, np.full(base.shape[0], slope * te), axis=2))
        paths.append(str(p))

    prefix = tmp_path / "mc.nii.gz"
    main(
        [
            "-input",
            *paths,
            "-reg_echo",
            "1",
            "-prefix",
            str(prefix),
            "-me_3depi",
            "-axis",
            "z",
            "-echo_times",
            *[str(t) for t in tes],
            "-save_shifts",
            str(tmp_path / "qc"),
            "-1Dfile_shiftcorr",
            str(tmp_path / "motion_sc.1D"),
            "-1Dmatrix_shiftcorr",
            str(tmp_path / "mat_sc.aff12.1D"),
            "-device",
            "cpu",
            "-verb",
            "0",
        ]
    )

    # The fitted slope is the correction, i.e. minus the displacement we applied.
    te_fit = np.loadtxt(tmp_path / "qc_te_fit.1D")
    assert np.allclose(te_fit[:, 0], -slope, atol=2e-3)

    # Every echo now sits at the same partition offset.
    outs = [load_image(str(tmp_path / f"e{i + 1}_mc.nii.gz"))[0] for i in range(3)]
    for i in range(1, 3):
        resid, _ = estimate_pair_shift(outs[i - 1], outs[i], axis=2, max_shift=2.0)
        assert np.abs(resid).max() < 0.05, f"echo {i + 1} still offset by {resid}"

    # Per-echo total transforms exist and genuinely differ across echoes.
    p1 = np.loadtxt(tmp_path / "e1_motion_sc.1D")
    p3 = np.loadtxt(tmp_path / "e3_motion_sc.1D")
    assert np.abs(p1 - p3).max() > 1e-3
    for i in range(3):
        assert (tmp_path / f"e{i + 1}_mat_sc.aff12.1D").exists()


def test_me_3depi_rejects_a_single_echo(tmp_path):
    """The shift is measured BETWEEN echoes; one input cannot supply it."""
    import pytest

    e1 = tmp_path / "e1.nii.gz"
    _write(e1, _shifted_series(_blob()))
    with pytest.raises(SystemExit):
        main(
            [
                "-input",
                str(e1),
                "-prefix",
                str(tmp_path / "mc.nii.gz"),
                "-me_3depi",
                "-axis",
                "z",
                "-device",
                "cpu",
                "-verb",
                "0",
            ]
        )
