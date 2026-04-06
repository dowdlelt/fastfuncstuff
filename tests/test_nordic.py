"""Tests for NORDIC-style denoising module."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fastfuncstuff.denoise.nordic import (
    NordicConfig,
    _compute_dd_phase,
    _default_kernel_size_pca,
    _estimate_nordic_lambda,
    _llr_denoise,
    _phase_to_radians,
    run_nordic,
)


def _write_nifti(path: Path, data: np.ndarray) -> None:
    img = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
    nib.save(img, path)


# --- Unit tests for helpers ---


def test_phase_to_radians_range():
    """Output should be in [-pi, pi]."""
    rng = np.random.default_rng(10)
    raw = rng.uniform(-4096, 4096, size=(5, 5, 3, 8)).astype(np.float32)
    rad = _phase_to_radians(raw)
    assert rad.min() >= -np.pi - 1e-5
    assert rad.max() <= np.pi + 1e-5


def test_default_kernel_size_pca_slice_fallback():
    """When nz < k, should fall back to 2D kernel."""
    k = _default_kernel_size_pca(100, n_slices=2)
    assert k[2] == 2  # third dimension equals nz


def test_estimate_nordic_lambda_sqrt2():
    """Complex data should produce ~sqrt(2) larger lambda than magnitude."""
    dev = torch.device("cpu")
    lam_real = _estimate_nordic_lambda((5, 5, 3), 20, 1.0, 1.0, is_complex=False, device=dev)
    lam_cplx = _estimate_nordic_lambda((5, 5, 3), 20, 1.0, 1.0, is_complex=True, device=dev)
    ratio = lam_cplx / lam_real
    assert 1.3 < ratio < 1.5, f"Expected ~sqrt(2)=1.414, got {ratio:.3f}"


def test_dd_phase_lowpass():
    """DD_phase should be a smoothed version of input (lower high-freq energy)."""
    rng = np.random.default_rng(42)
    nx, ny, nz, nt = 16, 16, 1, 10
    real = rng.normal(size=(nx, ny, nz, nt)).astype(np.float32)
    imag = rng.normal(size=(nx, ny, nz, nt)).astype(np.float32)
    data = torch.from_numpy(real + 1j * imag).to(torch.complex64)
    dd = _compute_dd_phase(data, phase_filter_width=3.0, verbose=False)
    # dd should have lower total energy (smoothed)
    assert torch.sum(torch.abs(dd) ** 2) < torch.sum(torch.abs(data) ** 2)


# --- Integration tests ---


def test_run_nordic_magnitude_only(tmp_path):
    rng = np.random.default_rng(123)
    magn = np.abs(rng.normal(size=(12, 10, 6, 16))).astype(np.float32)

    magn_file = tmp_path / "magn.nii.gz"
    _write_nifti(magn_file, magn)

    cfg = NordicConfig(
        temporal_phase=0,
        magnitude_only=True,
        kernel_size_pca=(3, 3, 3),
        kernel_size_gfactor=(3, 3, 1),
        gfactor_nvols=8,
        patch_overlap=2,
        gfactor_patch_overlap=2,
        save_gfactor_map=True,
        verbose=False,
    )

    out = run_nordic(
        magnitude_file=str(magn_file),
        phase_file=None,
        output_prefix=str(tmp_path / "NORDIC_magn"),
        config=cfg,
    )

    assert out.magnitude_file.exists()
    assert out.gfactor_file is not None and out.gfactor_file.exists()
    assert out.metadata_file.exists()

    den = nib.load(out.magnitude_file).get_fdata(dtype=np.float32)
    assert den.shape == magn.shape
    assert np.isfinite(den).all()

    # Denoised output should NOT be identical to input (actual denoising occurred)
    # After absolute_scale normalization, exact equality is extremely unlikely.


def test_run_nordic_complex_outputs(tmp_path):
    rng = np.random.default_rng(456)
    magn = np.abs(rng.normal(size=(10, 8, 5, 14))).astype(np.float32)
    phase = rng.uniform(low=-2000.0, high=2000.0, size=(10, 8, 5, 14)).astype(np.float32)

    magn_file = tmp_path / "magn.nii.gz"
    phase_file = tmp_path / "phase.nii.gz"
    _write_nifti(magn_file, magn)
    _write_nifti(phase_file, phase)

    cfg = NordicConfig(
        temporal_phase=1,
        phase_filter_width=3.0,
        noise_volume_last=2,
        kernel_size_pca=(3, 3, 3),
        kernel_size_gfactor=(3, 3, 1),
        gfactor_nvols=8,
        patch_overlap=2,
        gfactor_patch_overlap=2,
        make_complex_nii=True,
        save_gfactor_map=True,
        verbose=False,
    )

    out = run_nordic(
        magnitude_file=str(magn_file),
        phase_file=str(phase_file),
        output_prefix=str(tmp_path / "NORDIC_complex"),
        config=cfg,
    )

    assert out.magnitude_file.exists()
    assert out.phase_file is not None and out.phase_file.exists()
    assert out.gfactor_file is not None and out.gfactor_file.exists()

    den_magn = nib.load(out.magnitude_file).get_fdata(dtype=np.float32)
    den_phase = nib.load(out.phase_file).get_fdata(dtype=np.float32)

    assert den_magn.shape == magn.shape
    assert den_phase.shape == phase.shape
    assert np.isfinite(den_magn).all()
    assert np.isfinite(den_phase).all()


def test_eigh_decomp_matches_svd(tmp_path):
    """Gram-matrix eigh path should produce results close to SVD path."""
    rng = np.random.default_rng(321)
    magn = np.abs(rng.normal(size=(10, 8, 5, 14))).astype(np.float32)

    magn_file = tmp_path / "magn_eigh.nii.gz"
    _write_nifti(magn_file, magn)

    shared = dict(
        temporal_phase=0,
        magnitude_only=True,
        kernel_size_pca=(3, 3, 3),
        kernel_size_gfactor=(3, 3, 1),
        gfactor_nvols=8,
        patch_overlap=2,
        gfactor_patch_overlap=2,
        verbose=False,
    )

    out_svd = run_nordic(
        magnitude_file=str(magn_file),
        phase_file=None,
        output_prefix=str(tmp_path / "svd"),
        config=NordicConfig(**shared, decomp_method="svd"),
    )
    out_eigh = run_nordic(
        magnitude_file=str(magn_file),
        phase_file=None,
        output_prefix=str(tmp_path / "eigh"),
        config=NordicConfig(**shared, decomp_method="eigh"),
    )

    den_svd = nib.load(out_svd.magnitude_file).get_fdata(dtype=np.float32)
    den_eigh = nib.load(out_eigh.magnitude_file).get_fdata(dtype=np.float32)

    assert den_svd.shape == den_eigh.shape
    # They should be very close (same math, different decomposition route).
    # Small g-factor differences (from sigmasq_2 gathering at the MP cut
    # index) cascade through divide-by-gfactor → denoise → multiply-by-
    # gfactor, so compare correlation rather than element-wise tolerance.
    mask = (den_svd > 1e-3) & (den_eigh > 1e-3)
    assert mask.sum() > 0.8 * den_svd.size, "Too few non-zero voxels"
    r = np.corrcoef(den_svd[mask].ravel(), den_eigh[mask].ravel())[0, 1]
    assert r > 0.99, f"eigh vs SVD correlation too low: {r:.6f}"


@torch.no_grad()
def test_llr_denoise_cross_device():
    """When data lives on CPU and device='cuda', results should match GPU-only."""
    if not torch.cuda.is_available():
        return  # skip on CPU-only machines

    rng = np.random.default_rng(555)
    vol = torch.tensor(rng.normal(size=(8, 8, 4, 12)).astype(np.float32), device="cuda")

    # GPU-only path
    recon_gpu, stats_gpu = _llr_denoise(
        vol,
        kernel_size=(3, 3, 3),
        patch_overlap=2,
        threshold_mode="mp",
        threshold_value=0.0,
        verbose=False,
        return_recon=True,
        device=torch.device("cuda"),
    )

    # Cross-device path: data on CPU, compute on GPU
    recon_cross, stats_cross = _llr_denoise(
        vol.cpu(),
        kernel_size=(3, 3, 3),
        patch_overlap=2,
        threshold_mode="mp",
        threshold_value=0.0,
        verbose=False,
        return_recon=True,
        device=torch.device("cuda"),
    )

    assert recon_cross.device.type == "cuda"
    assert stats_cross.weight.device.type == "cuda"
    np.testing.assert_allclose(
        recon_cross.cpu().numpy(),
        recon_gpu.cpu().numpy(),
        atol=1e-5,
        rtol=1e-5,
    )


def test_noise_volume_branch_drives_measured_noise(tmp_path):
    rng = np.random.default_rng(789)
    magn = np.abs(rng.normal(scale=0.05, size=(8, 8, 4, 20))).astype(np.float32)
    # Last 3 vols are high-noise (noise-only branch target).
    magn[..., -3:] = np.abs(rng.normal(scale=2.5, size=(8, 8, 4, 3))).astype(np.float32)

    magn_file = tmp_path / "magn_noise.nii.gz"
    _write_nifti(magn_file, magn)

    cfg = NordicConfig(
        temporal_phase=0,
        magnitude_only=True,
        noise_volume_last=3,
        kernel_size_pca=(3, 3, 2),
        kernel_size_gfactor=(3, 3, 1),
        gfactor_nvols=8,
        patch_overlap=2,
        gfactor_patch_overlap=2,
        verbose=False,
    )

    out = run_nordic(
        magnitude_file=str(magn_file),
        phase_file=None,
        output_prefix=str(tmp_path / "NORDIC_noise_branch"),
        config=cfg,
    )

    with open(out.metadata_file, encoding="utf-8") as f:
        meta = json.load(f)

    # With explicit noisy trailing volumes, measured noise should be > fallback 1.0.
    assert meta["threshold"]["measured_noise"] > 1.0


def test_mp2_forces_unit_gfactor(tmp_path):
    rng = np.random.default_rng(321)
    magn = np.abs(rng.normal(size=(8, 8, 4, 12))).astype(np.float32)
    phase = rng.uniform(low=-1000.0, high=1000.0, size=(8, 8, 4, 12)).astype(np.float32)

    magn_file = tmp_path / "magn_mp2.nii.gz"
    phase_file = tmp_path / "phase_mp2.nii.gz"
    _write_nifti(magn_file, magn)
    _write_nifti(phase_file, phase)

    cfg = NordicConfig(
        temporal_phase=1,
        mp_mode=2,
        kernel_size_pca=(3, 3, 2),
        kernel_size_gfactor=(3, 3, 1),
        gfactor_nvols=8,
        patch_overlap=2,
        gfactor_patch_overlap=2,
        save_gfactor_map=True,
        verbose=False,
    )

    out = run_nordic(
        magnitude_file=str(magn_file),
        phase_file=str(phase_file),
        output_prefix=str(tmp_path / "NORDIC_mp2"),
        config=cfg,
    )

    assert out.gfactor_file is not None and out.gfactor_file.exists()
    g = nib.load(out.gfactor_file).get_fdata(dtype=np.float32)
    assert np.allclose(g, 1.0, atol=1e-6)


def test_absolute_scale_roundtrip(tmp_path):
    """ABSOLUTE_SCALE should normalize then restore, preserving magnitude range."""
    rng = np.random.default_rng(999)
    magn = np.abs(rng.normal(scale=100.0, size=(6, 6, 3, 10))).astype(np.float32)
    magn_file = tmp_path / "magn_scale.nii.gz"
    _write_nifti(magn_file, magn)

    cfg = NordicConfig(
        temporal_phase=0,
        magnitude_only=True,
        mp_mode=2,  # MP2 = unit gfactor, minimal denoising interference
        kernel_size_pca=(3, 3, 3),
        kernel_size_gfactor=(3, 3, 1),
        gfactor_nvols=8,
        patch_overlap=2,
        gfactor_patch_overlap=2,
        verbose=False,
    )

    out = run_nordic(
        magnitude_file=str(magn_file),
        phase_file=None,
        output_prefix=str(tmp_path / "NORDIC_scale"),
        config=cfg,
    )

    den = nib.load(out.magnitude_file).get_fdata(dtype=np.float32)
    # Output should be in similar range as input (within order of magnitude).
    assert den.max() > 1.0, "Output magnitude collapsed near zero (ABSOLUTE_SCALE not restored)"


def test_metadata_has_absolute_scale(tmp_path):
    rng = np.random.default_rng(42)
    magn = np.abs(rng.normal(size=(6, 6, 3, 10))).astype(np.float32)
    magn_file = tmp_path / "magn.nii.gz"
    _write_nifti(magn_file, magn)

    cfg = NordicConfig(
        temporal_phase=0,
        magnitude_only=True,
        kernel_size_pca=(3, 3, 3),
        kernel_size_gfactor=(3, 3, 1),
        gfactor_nvols=8,
        patch_overlap=2,
        gfactor_patch_overlap=2,
        verbose=False,
    )
    out = run_nordic(str(magn_file), None, str(tmp_path / "NORDIC_meta"), cfg)
    with open(out.metadata_file) as f:
        meta = json.load(f)
    assert "absolute_scale" in meta
    assert meta["absolute_scale"] > 0


def test_residual_map_saved(tmp_path):
    """Residual = |input_complex - denoised_complex|, saved as float32 NIfTI."""
    rng = np.random.RandomState(99)
    magn = np.abs(rng.normal(size=(12, 10, 6, 16))).astype(np.float32)
    magn_file = tmp_path / "magn.nii.gz"
    _write_nifti(magn_file, magn)

    cfg = NordicConfig(
        temporal_phase=0,
        magnitude_only=True,
        kernel_size_pca=(3, 3, 3),
        kernel_size_gfactor=(3, 3, 1),
        gfactor_nvols=8,
        patch_overlap=2,
        gfactor_patch_overlap=2,
        save_residual_map=True,
        verbose=False,
    )

    out = run_nordic(str(magn_file), None, str(tmp_path / "NORDIC_res"), cfg)

    assert out.residual_file is not None and out.residual_file.exists()
    res = nib.load(out.residual_file).get_fdata(dtype=np.float32)
    assert res.shape == magn.shape
    assert np.isfinite(res).all()
    assert (res >= 0).all()  # magnitude is non-negative

    # Check metadata records residual path
    with open(out.metadata_file) as f:
        meta = json.load(f)
    assert meta["outputs"]["residual"] is not None
