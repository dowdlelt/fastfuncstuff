"""Round-trip tests for the time-varying warp series I/O (5D file <-> 4D folder).

save_warp_series must write both formats from the same displacement fields, and
load_warp_series must read them back to identical values — the invariant that
keeps ffs_qwarp / ffs_locomoco / ffs_util_pcwarp interchangeable on either format.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

nib = pytest.importorskip("nibabel")

from fastfuncstuff.processing.io import load_warp_series, save_warp_field, save_warp_series


def _synth_series(T=6, nz=5, ny=6, nx=7, seed=0):
    """(T, nz, ny, nx) displacement fields with distinct per-axis structure."""
    g = torch.Generator().manual_seed(seed)
    xd = torch.rand(T, nz, ny, nx, generator=g)
    yd = torch.rand(T, nz, ny, nx, generator=g) * 2.0
    zd = torch.zeros(T, nz, ny, nx)  # inactive axis
    return xd, yd, zd


def test_asymmetric_padding_shifts_origin_by_lower_faces(tmp_path):
    affine = np.array(
        [[2.0, 0.1, 0.0, 10.0], [0.0, 3.0, 0.2, 20.0], [0.0, 0.0, 4.0, 30.0], [0, 0, 0, 1]]
    )
    padding = (2, 9, 3, 8, 4, 7)
    expected = affine.copy()
    expected[:3, 3] += affine[:3, :3] @ np.array([-2.0, -3.0, -4.0])
    xd, yd, zd = _synth_series(T=2)

    single = tmp_path / "single.nii.gz"
    save_warp_field(xd[0], yd[0], zd[0], single, affine=affine, padding=padding)
    np.testing.assert_allclose(nib.load(single).affine, expected)

    series = tmp_path / "series.nii.gz"
    save_warp_series(xd, yd, zd, series, as_5d=True, affine=affine, units="voxels", padding=padding)
    np.testing.assert_allclose(nib.load(series).affine, expected)


def test_warpqc_converts_saved_afni_mm_back_to_voxels(tmp_path):
    from fastfuncstuff.cli.util_warpqc import main as warpqc_main

    nz, ny, nx = 12, 13, 14
    xx = torch.arange(nx, dtype=torch.float32).view(1, 1, nx).expand(nz, ny, nx)
    xd = 0.2 * xx
    zero = torch.zeros_like(xd)
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    warp = tmp_path / "ramp_WARP.nii.gz"
    report = tmp_path / "qc.json"
    save_warp_field(xd, zero, zero, warp, affine=affine, units="mm")

    assert (
        warpqc_main(["-warp", str(warp), "-json", str(report), "-device", "cpu", "-verb", "0"]) == 0
    )
    qc = json.loads(report.read_text())[warp.name]
    assert qc["jac_neg_count"] == 0
    assert qc["jac_p50"] == pytest.approx(1.2, abs=1e-5)


def test_5d_and_folder_roundtrip_match(tmp_path):
    """Both formats reload to the same values, and to each other."""
    xd, yd, zd = _synth_series()
    affine = np.diag([2.0, 2.0, 2.0, 1.0])

    f5 = tmp_path / "warp5d.nii.gz"
    save_warp_series(xd, yd, zd, f5, as_5d=True, affine=affine, units="mm")
    folder = tmp_path / "warp_folder"
    glob = save_warp_series(xd, yd, zd, folder, as_5d=False, affine=affine, units="mm")

    assert f5.exists()
    assert glob.endswith("warp_*.nii.gz")

    x5, y5, z5, _, n5 = load_warp_series(f5)
    xf, yf, zf, _, nf = load_warp_series(str(folder))

    assert n5 == nf == xd.shape[0]
    # 5D vs folder must be bit-for-bit consistent (same on-disk convention).
    torch.testing.assert_close(x5, xf)
    torch.testing.assert_close(y5, yf)
    torch.testing.assert_close(z5, zf)


def test_voxels_roundtrip_is_identity(tmp_path):
    """units='voxels' with an axis-aligned affine round-trips the raw values."""
    xd, yd, zd = _synth_series()
    affine = np.diag([3.0, 3.0, 3.0, 1.0])
    f5 = tmp_path / "w.nii.gz"
    save_warp_series(xd, yd, zd, f5, as_5d=True, affine=affine, units="voxels")
    x5, y5, z5, _, _ = load_warp_series(f5)
    torch.testing.assert_close(x5, xd)
    torch.testing.assert_close(y5, yd)
    torch.testing.assert_close(z5, zd)


def test_single_4d_file_reads_as_one_frame(tmp_path):
    """A plain 4D (nx,ny,nz,3) warp loads as a T=1 series."""
    from fastfuncstuff.processing.io import save_warp_field

    xd, yd, zd = _synth_series(T=1)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    f = tmp_path / "one.nii.gz"
    save_warp_field(xd[0], yd[0], zd[0], f, affine=affine, units="voxels")
    x, y, z, _, n = load_warp_series(f)
    assert n == 1
    torch.testing.assert_close(x, xd)


def test_pcwarp_cli_on_both_formats(tmp_path):
    """ffs_util_pcwarp produces the same PCs from a folder and from a 5D file."""
    from fastfuncstuff.cli.pcwarp import main

    xd, yd, zd = _synth_series(T=10)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    f5 = tmp_path / "warp5d.nii.gz"
    save_warp_series(xd, yd, zd, f5, as_5d=True, affine=affine, units="mm")
    folder = tmp_path / "wf"
    save_warp_series(xd, yd, zd, folder, as_5d=False, affine=affine, units="mm")

    out5 = tmp_path / "pc5.1D"
    outf = tmp_path / "pcf.1D"
    assert main(["-warp", str(f5), "-n_pcs", "3", "-output", str(out5), "-verb", "0"]) == 0
    assert (
        main(
            [
                "-warp_dir",
                str(folder),
                "-pattern",
                "warp_*.nii.gz",
                "-n_pcs",
                "3",
                "-output",
                str(outf),
                "-verb",
                "0",
            ]
        )
        == 0
    )

    def _rows(p):
        return np.array(
            [
                [float(v) for v in ln.split()]
                for ln in p.read_text().splitlines()
                if not ln.startswith("#")
            ]
        )

    a, b = _rows(out5), _rows(outf)
    assert a.shape == b.shape == (10, 3)
    # PCs are sign-ambiguous; compare |correlation| per component.
    for k in range(3):
        r = abs(np.corrcoef(a[:, k], b[:, k])[0, 1])
        assert r > 0.999, f"PC{k} differs between formats (|r|={r:.4f})"


def test_pcwarp_requires_exactly_one_source(tmp_path):
    from fastfuncstuff.cli.pcwarp import main

    assert main(["-n_pcs", "2", "-verb", "0"]) == 1  # neither
