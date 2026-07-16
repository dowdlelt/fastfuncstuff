"""Tests for ffs_util_pcwarp -warp_pc_recon (low-rank warp reconstruction)."""

import numpy as np
import torch

from fastfuncstuff.cli.pcwarp import main
from fastfuncstuff.processing.io import load_warp_series, save_warp_series


def _make_lowrank_y_warp(tmp_path, nz=4, ny=8, nx=6, T=15, noise=0.0, seed=0, name="warps5d"):
    """A y-only warp series that is (mean + rank-2) in time, optionally + noise."""
    rng = np.random.default_rng(seed)
    S1 = rng.standard_normal((nz, ny, nx)).astype(np.float32)
    S2 = rng.standard_normal((nz, ny, nx)).astype(np.float32)
    M = 0.5 * rng.standard_normal((nz, ny, nx)).astype(np.float32)  # static mean field
    a1 = np.sin(np.linspace(0, 3, T)).astype(np.float32)
    a2 = np.cos(np.linspace(0, 5, T)).astype(np.float32)
    yd = M[None] + a1[:, None, None, None] * S1[None] + a2[:, None, None, None] * S2[None]
    if noise:
        yd = yd + noise * rng.standard_normal(yd.shape).astype(np.float32)
    zero = torch.zeros(T, nz, ny, nx)
    path = tmp_path / f"{name}.nii.gz"
    save_warp_series(zero, torch.from_numpy(yd), zero, str(path), as_5d=True, units="voxels")
    return str(path), yd  # yd is the on-disk y displacement (voxels)


def test_full_rank_recon_reproduces_input(tmp_path):
    path, yd = _make_lowrank_y_warp(tmp_path, T=12, noise=0.1)
    # Fit/recon at full rank (T-1 PCs) must reproduce the input warp.
    rc = main(["-warp", path, "-n_pcs", "11", "-warp_pc_recon", "11", "-verb", "0"])
    assert rc == 0
    out = path[:-7] + "_pcrecon11.nii.gz"
    xr, yr, zr, _, n = load_warp_series(out)
    assert n == 12
    assert torch.allclose(yr, torch.from_numpy(yd), atol=1e-3)
    assert xr.abs().max() == 0 and zr.abs().max() == 0  # inactive axes stay zero


def test_rank2_recon_recovers_lowrank_signal(tmp_path):
    # Signal is (mean + rank-2); 2 PCs should recover it almost exactly despite noise.
    path, _ = _make_lowrank_y_warp(tmp_path, T=15, noise=0.05, name="noisy")
    _, clean_yd = _make_lowrank_y_warp(tmp_path, T=15, noise=0.0, seed=0, name="clean")

    main(["-warp", path, "-n_pcs", "6", "-warp_pc_recon", "2", "-verb", "0"])
    _, yr, _, _, _ = load_warp_series(path[:-7] + "_pcrecon2.nii.gz")

    # Rank-2 reconstruction of the noisy warp is closer to the clean signal than the
    # noisy input is — the point of the denoising.
    clean = torch.from_numpy(clean_yd)
    _, y_noisy, _, _, _ = load_warp_series(path)
    err_recon = (yr - clean).abs().mean().item()
    err_noisy = (y_noisy - clean).abs().mean().item()
    assert err_recon < err_noisy
    assert err_recon < 0.02  # near-perfect recovery of the rank-2 structure


def test_diag_frames_written_4d(tmp_path):
    path, yd = _make_lowrank_y_warp(tmp_path, T=15, noise=0.1)
    # Full-rank recon so the recon diag frame equals the original diag frame.
    main(["-warp", path, "-n_pcs", "14", "-warp_pc_recon", "14", "-diag_frame", "10", "-verb", "0"])
    base = path[:-7] + "_pcrecon14"
    xo, yo, zo, _, no = load_warp_series(base + "_frame10_orig.nii.gz")
    xr, yr, zr, _, nr = load_warp_series(base + "_frame10_recon.nii.gz")
    assert no == 1 and nr == 1  # single frame each
    # orig diag frame == input frame 9 (0-based); full-rank recon frame matches it.
    assert torch.allclose(yo[0], torch.from_numpy(yd[9]), atol=1e-4)
    assert torch.allclose(yr[0], yo[0], atol=1e-3)


def test_diag_frame_clamped_and_disable(tmp_path):
    path, _ = _make_lowrank_y_warp(tmp_path, T=6)
    # diag_frame 10 > T: clamps to the last frame (index 5 -> filename frame6).
    main(["-warp", path, "-n_pcs", "4", "-warp_pc_recon", "2", "-diag_frame", "10", "-verb", "0"])
    assert (tmp_path / "warps5d_pcrecon2_frame6_orig.nii.gz").exists()
    # diag_frame 0 disables.
    main(["-warp", path, "-n_pcs", "4", "-warp_pc_recon", "2", "-diag_frame", "0",
          "-recon_prefix", str(tmp_path / "nodiag.nii.gz"), "-verb", "0"])
    assert not list(tmp_path.glob("nodiag_frame*"))


def test_recon_prefix_and_default_path(tmp_path):
    path, _ = _make_lowrank_y_warp(tmp_path, T=10)
    custom = str(tmp_path / "denoised.nii.gz")
    rc = main(
        ["-warp", path, "-n_pcs", "3", "-warp_pc_recon", "3",
         "-recon_prefix", custom, "-verb", "0"]
    )
    assert rc == 0
    _, yr, _, _, _ = load_warp_series(custom)
    assert yr.shape[0] == 10
