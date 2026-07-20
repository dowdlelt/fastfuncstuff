"""Smoke tests for the ffs_segment CLI (end-to-end on a tiny phantom)."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import torch

from fastfuncstuff.cli.segment import main


def _write_nii(path, arr, affine=np.eye(4)):
    # arr is (nz, ny, nx[, t]); NIfTI wants (nx, ny, nz[, t])
    arr = np.asarray(arr.cpu() if hasattr(arr, "cpu") else arr, np.float32)
    data = arr.transpose(2, 1, 0) if arr.ndim == 3 else arr.transpose(2, 1, 0, 3)
    nib.save(nib.Nifti1Image(data, affine), str(path))


def _phantom(n=20):
    zz, yy, xx = torch.meshgrid(torch.arange(n), torch.arange(n), torch.arange(n), indexing="ij")
    r = ((xx - n / 2) ** 2 + (yy - n / 2) ** 2 + (zz - n / 2) ** 2).sqrt()
    tissue = torch.stack(
        [(r < 4).float(), ((r >= 4) & (r < 8)).float(), ((r >= 8) & (r < 10)).float()]
    )
    means = torch.tensor([150.0, 80.0, 40.0])
    vol = sum(means[k] * tissue[k] for k in range(3))
    return vol, tissue


def test_segment_cli_writes_all_outputs(tmp_path):
    vol, tissue = _phantom()
    prob = tissue / tissue.sum(0, keepdim=True).clamp_min(1e-6)
    tpm = prob.permute(1, 2, 3, 0)  # (nz,ny,nx,3)

    inp = tmp_path / "input.nii.gz"
    tpmp = tmp_path / "tpm.nii.gz"
    _write_nii(inp, vol)
    _write_nii(tpmp, tpm)

    rc = main(
        [
            "-input",
            str(inp),
            "-tpm",
            str(tpmp),
            "-prefix",
            str(tmp_path / "seg"),
            "-ngaus",
            "1",
            "1",
            "1",
            "-biasfwhm",
            "10",
            "-samp",
            "1",
            "-niter",
            "5",
            "-no_warp",
            "-device",
            "cpu",
            "-quiet",
        ]
    )
    assert rc == 0
    for t in range(3):
        assert (tmp_path / f"seg_c{t + 1}.nii.gz").exists()
    assert (tmp_path / "seg_biascorrected.nii.gz").exists()
    assert (tmp_path / "seg_biasfield.nii.gz").exists()

    # c1 (core) should light up the core region
    c1 = nib.load(str(tmp_path / "seg_c1.nii.gz")).get_fdata().transpose(2, 1, 0)
    core = tissue[0].numpy() > 0.5
    assert c1[core].mean() > 0.8


def test_segment_cli_writes_invwarp_that_inverts_warp(tmp_path):
    """With the deformation on, _invwarp.nii.gz is written and is the fixed-point
    inverse of _warp: composing them returns near-identity on the interior."""
    from fastfuncstuff.processing.interp import trilinear_interpolate

    vol, tissue = _phantom()
    prob = tissue / tissue.sum(0, keepdim=True).clamp_min(1e-6)
    tpm = prob.permute(1, 2, 3, 0)
    inp = tmp_path / "input.nii.gz"
    tpmp = tmp_path / "tpm.nii.gz"
    _write_nii(inp, vol)
    _write_nii(tpmp, tpm)

    rc = main(
        [
            "-input", str(inp), "-tpm", str(tpmp), "-prefix", str(tmp_path / "seg"),
            "-ngaus", "1", "1", "1", "-biasfwhm", "10", "-samp", "1", "-niter", "5",
            "-device", "cpu", "-quiet",
        ]
    )  # fmt: skip
    assert rc == 0
    assert (tmp_path / "seg_warp.nii.gz").exists()
    assert (tmp_path / "seg_invwarp.nii.gz").exists()

    # Load both fields (NIfTI (nx,ny,nz,1,3) -> voxel disp per component). Only the
    # relative round-trip matters, so read them back the same way and compose.
    def _load_field(p):
        d = nib.load(str(p)).get_fdata()  # (nx,ny,nz,[1,]3)
        if d.ndim == 5:
            d = d[:, :, :, 0, :]
        d = d.transpose(2, 1, 0, 3)  # (nz,ny,nx,3)
        return torch.from_numpy(np.ascontiguousarray(d)).float()

    fwd = _load_field(tmp_path / "seg_warp.nii.gz")
    inv = _load_field(tmp_path / "seg_invwarp.nii.gz")
    nz, ny, nx, _ = fwd.shape
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz).float(), torch.arange(ny).float(), torch.arange(nx).float(), indexing="ij"
    )
    xf, yf, zf = xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)
    qx, qy, qz = (
        xf + inv[..., 0].reshape(-1),
        yf + inv[..., 1].reshape(-1),
        zf + inv[..., 2].reshape(-1),
    )
    rx = qx + trilinear_interpolate(fwd[..., 0].contiguous(), qx, qy, qz)
    ry = qy + trilinear_interpolate(fwd[..., 1].contiguous(), qx, qy, qz)
    rz = qz + trilinear_interpolate(fwd[..., 2].contiguous(), qx, qy, qz)
    resid = torch.sqrt((rx - xf) ** 2 + (ry - yf) ** 2 + (rz - zf) ** 2)
    m = (xf > 3) & (xf < nx - 4) & (yf > 3) & (yf < ny - 4) & (zf > 3) & (zf < nz - 4)
    assert resid[m].max() < 0.05, f"warp∘invwarp not identity: max resid {resid[m].max():.4f} vox"


def test_segment_cli_no_invwarp_flag(tmp_path):
    vol, tissue = _phantom()
    prob = tissue / tissue.sum(0, keepdim=True).clamp_min(1e-6)
    tpm = prob.permute(1, 2, 3, 0)
    inp = tmp_path / "input.nii.gz"
    tpmp = tmp_path / "tpm.nii.gz"
    _write_nii(inp, vol)
    _write_nii(tpmp, tpm)
    rc = main(
        [
            "-input", str(inp), "-tpm", str(tpmp), "-prefix", str(tmp_path / "seg"),
            "-ngaus", "1", "1", "1", "-biasfwhm", "10", "-samp", "1", "-niter", "5",
            "-no_invwarp", "-device", "cpu", "-quiet",
        ]
    )  # fmt: skip
    assert rc == 0
    assert (tmp_path / "seg_warp.nii.gz").exists()
    assert not (tmp_path / "seg_invwarp.nii.gz").exists()


def test_segment_cli_ngaus_length_checked(tmp_path):
    vol, tissue = _phantom()
    tpm = (tissue / tissue.sum(0, keepdim=True).clamp_min(1e-6)).permute(1, 2, 3, 0)
    inp, tpmp = tmp_path / "i.nii.gz", tmp_path / "t.nii.gz"
    _write_nii(inp, vol)
    _write_nii(tpmp, tpm)
    try:
        main(
            [
                "-input",
                str(inp),
                "-tpm",
                str(tpmp),
                "-prefix",
                str(tmp_path / "s"),
                "-ngaus",
                "1",
                "1",
                "-device",
                "cpu",
                "-quiet",
            ]
        )  # 2 != 3 classes
        raise AssertionError("expected SystemExit for mismatched -ngaus")
    except SystemExit:
        pass


def test_segment_cli_save_histogram(tmp_path):
    vol, tissue = _phantom()
    prob = tissue / tissue.sum(0, keepdim=True).clamp_min(1e-6)
    tpm = prob.permute(1, 2, 3, 0)
    inp, tpmp = tmp_path / "input.nii.gz", tmp_path / "tpm.nii.gz"
    _write_nii(inp, vol)
    _write_nii(tpmp, tpm)

    rc = main(
        [
            "-input", str(inp),
            "-tpm", str(tpmp),
            "-prefix", str(tmp_path / "seg"),
            "-ngaus", "1", "1", "1",
            "-biasfwhm", "10", "-samp", "1", "-niter", "5", "-no_warp",
            "-save_histogram",  # bare → prefix_histogram.png
            "-tissue_names", "GM", "WM", "CSF",
            "-device", "cpu", "-quiet",
        ]
    )  # fmt: skip
    assert rc == 0
    hist = tmp_path / "seg_histogram.png"
    assert hist.exists() and hist.stat().st_size > 0
