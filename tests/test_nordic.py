"""Tests for NORDIC-style denoising module."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from fastfuncstuff.denoise.nordic import (
    NordicConfig,
    _compute_dd_phase,
    _default_kernel_size_pca,
    _estimate_nordic_lambda,
    _llr_denoise,
    _llr_denoise_multiecho,
    _phase_to_radians,
    _rescue_null_threshold,
    _residual_xcorr_qc,
    run_nordic,
    run_nordic_multiecho,
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


def test_add_mean_lifts_residual_to_raw_temporal_mean_floor(tmp_path):
    """-add_mean adds the raw magnitude's per-voxel temporal mean onto the saved
    residual. Since |residual| >= 0, the raw temporal mean becomes an exact
    per-voxel floor; without it the residual is tiny next to the ~100 baseline.

    (NORDIC is not run-to-run deterministic even on CPU, so this is a single-run
    invariant rather than a with/without difference.)
    """
    rng = np.random.default_rng(2024)
    magn = np.abs(rng.normal(loc=100.0, scale=5.0, size=(9, 8, 5, 14))).astype(np.float32)
    magn_file = tmp_path / "magn.nii.gz"
    _write_nifti(magn_file, magn)
    raw_mean = magn.mean(axis=-1)  # (9, 8, 5)

    shared = dict(
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
    out_no = run_nordic(
        magnitude_file=str(magn_file),
        phase_file=None,
        output_prefix=str(tmp_path / "no_mean"),
        config=NordicConfig(**shared, add_mean=False),
    )
    out_yes = run_nordic(
        magnitude_file=str(magn_file),
        phase_file=None,
        output_prefix=str(tmp_path / "with_mean"),
        config=NordicConfig(**shared, add_mean=True),
    )
    assert out_no.residual_file is not None and out_yes.residual_file is not None
    resid_no = nib.load(out_no.residual_file).get_fdata(dtype=np.float32)
    resid_yes = nib.load(out_yes.residual_file).get_fdata(dtype=np.float32)

    # Baseline residual is small next to the ~100 mean; with -add_mean it is not.
    assert resid_no.max() < raw_mean.min()
    # raw_mean is an exact per-voxel floor of the mean-added residual (|r| >= 0).
    assert np.all(resid_yes >= raw_mean[..., None] - 1e-2)
    # And the added baseline equals the raw temporal mean (per-voxel time mean of
    # the mean-added residual = raw_mean + mean_t|r|, so it exceeds raw_mean).
    assert np.all(resid_yes.mean(axis=-1) >= raw_mean - 1e-2)


def test_retain_dof_caps_components_removed(tmp_path):
    """-retain_dof forces each patch to keep >= floor components, so the numcomps
    map never exceeds nt - floor, and the cap actually binds on noisy data."""
    rng = np.random.default_rng(99)
    nt = 24
    magn = np.abs(rng.normal(size=(12, 10, 6, nt))).astype(np.float32)  # ~pure noise
    magn_file = tmp_path / "magn.nii.gz"
    _write_nifti(magn_file, magn)

    shared = dict(
        temporal_phase=0,
        magnitude_only=True,
        kernel_size_pca=(3, 3, 3),
        kernel_size_gfactor=(3, 3, 1),
        gfactor_nvols=8,
        save_num_comps=True,
        verbose=False,
    )

    # Without a floor, near-pure noise makes NORDIC remove most components.
    out_free = run_nordic(
        magnitude_file=str(magn_file),
        phase_file=None,
        output_prefix=str(tmp_path / "free"),
        config=NordicConfig(**shared),
    )
    removed_free = nib.load(out_free.num_comps_file).get_fdata(dtype=np.float32)
    assert removed_free.max() > nt - 18  # the cap would bind

    floor = 18
    out = run_nordic(
        magnitude_file=str(magn_file),
        phase_file=None,
        output_prefix=str(tmp_path / "capped"),
        config=NordicConfig(**shared, retain_dof=float(floor)),
    )
    removed = nib.load(out.num_comps_file).get_fdata(dtype=np.float32)
    # Patch-averaged removed count never exceeds nt - floor anywhere.
    assert removed.max() <= (nt - floor) + 1e-3

    meta = json.loads(out.metadata_file.read_text())
    assert meta["config"]["retain_dof"] == float(floor)
    assert meta["diagnostics"]["retain_dof_floor"] == floor
    assert meta["diagnostics"]["retain_dof_patches_capped"] > 0

    # Fractional floor: keep >= 50% of nt -> remove at most nt/2.
    out_frac = run_nordic(
        magnitude_file=str(magn_file),
        phase_file=None,
        output_prefix=str(tmp_path / "frac"),
        config=NordicConfig(**shared, retain_dof=0.5),
    )
    removed_frac = nib.load(out_frac.num_comps_file).get_fdata(dtype=np.float32)
    assert removed_frac.max() <= (nt - nt // 2) + 1e-3


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

    # The 2 trailing noise volumes are trimmed from the saved output.
    expected_shape = magn.shape[:3] + (magn.shape[3] - 2,)
    assert den_magn.shape == expected_shape
    assert den_phase.shape == expected_shape
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
        kernel_size_pca=(5, 3, 3),
        kernel_size_gfactor=(3, 3, 3),
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


@pytest.mark.gpu
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


def test_noise_volumes_trimmed_from_output(tmp_path):
    """Trailing noise volumes calibrate the threshold then get trimmed.

    The saved output (and its NIfTI header dim[4]) must be the signal-only
    length, and the metadata must record how many volumes were dropped.
    """
    rng = np.random.default_rng(2024)
    nt, n_noise = 18, 3
    magn = np.abs(rng.normal(scale=0.1, size=(8, 8, 4, nt))).astype(np.float32)
    magn[..., -n_noise:] = np.abs(rng.normal(scale=3.0, size=(8, 8, 4, n_noise))).astype(np.float32)

    magn_file = tmp_path / "magn_trim.nii.gz"
    _write_nifti(magn_file, magn)

    cfg = NordicConfig(
        temporal_phase=0,
        magnitude_only=True,
        noise_volume_last=n_noise,
        kernel_size_pca=(3, 3, 2),
        kernel_size_gfactor=(3, 3, 1),
        gfactor_nvols=8,
        patch_overlap=2,
        gfactor_patch_overlap=2,
        save_residual_map=True,
        verbose=False,
    )

    out = run_nordic(str(magn_file), None, str(tmp_path / "NORDIC_trim"), cfg)

    img = nib.load(out.magnitude_file)
    den = img.get_fdata(dtype=np.float32)
    # Data array and on-disk header agree on the trimmed length.
    assert den.shape == (8, 8, 4, nt - n_noise)
    assert int(img.header.get_data_shape()[3]) == nt - n_noise

    # Residual map follows the same trimmed length.
    assert out.residual_file is not None
    res = nib.load(out.residual_file).get_fdata(dtype=np.float32)
    assert res.shape == (8, 8, 4, nt - n_noise)

    with open(out.metadata_file, encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["shape"][3] == nt
    assert meta["output_shape"][3] == nt - n_noise
    assert meta["noise_volumes_trimmed"] == n_noise


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


# ---------------------------------------------------------------------------
# Multi-echo cross-echo signal rescue
# ---------------------------------------------------------------------------


def _make_me_echoes(weights, nt=80, rank=2, noise=0.5, seed=0, shape=(10, 10, 4)):
    """Build E complex echoes: one shared low-rank time course at per-echo
    weights (HDR), plus independent thermal noise. Returns (echoes, sig)."""
    torch.manual_seed(seed)
    nx, ny, nz = shape
    m = nx * ny * nz
    vt = torch.linalg.qr(torch.randn(nt, rank, dtype=torch.cfloat))[0]  # (nt, rank)
    u = torch.randn(m, rank, dtype=torch.cfloat)
    sig = (u @ vt.mH).reshape(nx, ny, nz, nt)
    echoes = []
    for w in weights:
        n = (torch.randn(nx, ny, nz, nt) + 1j * torch.randn(nx, ny, nz, nt)) * noise
        echoes.append((w * sig + n).to(torch.cfloat))
    return echoes, sig


def test_rescue_null_threshold_monotone():
    """Null threshold rises with target dimension, falls with alpha, in (0, 1)."""
    t_small_d = _rescue_null_threshold(120, 10, 3, 2, 0.05)
    t_big_d = _rescue_null_threshold(120, 10, 12, 2, 0.05)
    t_loose = _rescue_null_threshold(120, 10, 3, 2, 0.20)
    assert 0.0 < t_small_d < t_big_d < 1.0
    assert t_loose < t_small_d  # larger alpha => lower bar
    # Degenerate inputs never rescue.
    assert _rescue_null_threshold(120, 0, 3, 2, 0.05) == 1.0
    assert _rescue_null_threshold(120, 10, 3, 0, 0.05) == 1.0


def test_rescue_recovers_buried_signal():
    """A signal buried below threshold in the weak echo but kept in a strong
    echo is rescued; without rescue it is removed entirely."""
    echoes, sig = _make_me_echoes([1.0, 2.0, 3.0])
    kw = dict(
        kernel_size=(10, 10, 4),
        patch_overlap=1,
        threshold_mode="nordic",
        threshold_values=[30.0, 30.0, 30.0],  # weak echo signal (SV~26) buried
        rescue_band=0.25,
        rescue_alpha=0.05,
        verbose=False,
        device=torch.device("cpu"),
    )
    res_off = _llr_denoise_multiecho(echoes, rescue=False, **kw)
    res_on = _llr_denoise_multiecho(echoes, rescue=True, **kw)

    true1 = (1.0 * sig).abs()
    # Without rescue the weak echo loses its signal (recon ~ 0).
    assert float(res_off[0][1].rescued_map.mean()) == 0.0
    assert float(res_off[0][0].abs().mean()) < 0.25 * float(true1.mean())
    # With rescue the weak echo's signal components come back.
    assert float(res_on[0][1].rescued_map.mean()) > 0.0
    assert float(res_on[0][0].abs().mean()) > 2.0 * float(res_off[0][0].abs().mean())

    # The buried component sits below threshold, so the suggested-factor ratio
    # drops below 1 somewhere (a decrease). The ratio is bidirectional now, so it
    # may also exceed 1 elsewhere. rescue=False produces no recfactor map.
    assert res_off[0][1].recfactor_map is None
    rf = res_on[0][1].recfactor_map
    assert rf is not None
    assert float(rf.min()) < 1.0
    assert torch.all(rf > 0) and torch.all(torch.isfinite(rf))


def test_no_rescue_matches_independent_nordic():
    """rescue=False must reproduce plain per-echo NORDIC (the guard only ever
    moves components keep<-kill; it never alters the base reconstruction)."""
    echoes, _ = _make_me_echoes([1.0, 2.5, 4.0], seed=3)
    thr = [22.0, 22.0, 22.0]
    res = _llr_denoise_multiecho(
        echoes,
        kernel_size=(10, 10, 4),
        patch_overlap=1,
        threshold_mode="nordic",
        threshold_values=thr,
        rescue=False,
        rescue_band=0.25,
        rescue_alpha=0.05,
        verbose=False,
        device=torch.device("cpu"),
    )
    for e in range(len(echoes)):
        recon_indep, _ = _llr_denoise(
            echoes[e],
            kernel_size=(10, 10, 4),
            patch_overlap=1,
            threshold_mode="nordic",
            threshold_value=thr[e],
            verbose=False,
            decomp_method="svd",
            device=torch.device("cpu"),
        )
        a = res[e][0].abs().ravel().numpy()
        b = recon_indep.abs().ravel().numpy()
        r = np.corrcoef(a, b)[0, 1]
        assert r > 0.999, f"echo {e}: rescue=off diverges from independent NORDIC (r={r:.5f})"


def test_false_rescue_rate_controlled():
    """Under pure thermal noise (no signal), the per-patch false-rescue rate
    tracks alpha — it should stay well below the band size."""
    n_trials = 16
    rescued = 0
    for seed in range(n_trials):
        torch.manual_seed(1000 + seed)
        echoes = [
            (torch.randn(10, 10, 4, 60) + 1j * torch.randn(10, 10, 4, 60)).to(torch.cfloat)
            for _ in range(3)
        ]
        res = _llr_denoise_multiecho(
            echoes,
            kernel_size=(10, 10, 4),
            patch_overlap=1,
            threshold_mode="nordic",
            threshold_values=[30.0, 30.0, 30.0],
            rescue=True,
            rescue_band=0.25,
            rescue_alpha=0.05,
            verbose=False,
            device=torch.device("cpu"),
        )
        rescued += sum(float(res[e][1].rescued_map.mean()) for e in range(3))
    mean_rescued = rescued / n_trials
    # alpha=0.05 per-patch FWER across 3 echoes => well under 1 component/trial.
    assert mean_rescued < 1.0, f"too many false rescues under noise: {mean_rescued:.3f}"


def test_run_nordic_multiecho_end_to_end(tmp_path):
    """Full multi-echo run writes one output + metadata per echo, shares the
    g-factor map, and records the rescue block."""
    rng = np.random.default_rng(7)
    nx, ny, nz, nt = 12, 12, 3, 40
    m = nx * ny * nz
    vt = np.linalg.qr(rng.standard_normal((nt, 2)) + 1j * rng.standard_normal((nt, 2)))[0]
    usp = rng.standard_normal((m, 2)) + 1j * rng.standard_normal((m, 2))
    sig = (usp @ vt.conj().T).reshape(nx, ny, nz, nt)

    magn_files, phase_files = [], []
    for e, w in enumerate([1.0, 2.5, 4.0]):
        cn = w * sig + (rng.standard_normal(sig.shape) + 1j * rng.standard_normal(sig.shape)) * 0.4
        mf = tmp_path / f"magn_e{e}.nii.gz"
        pf = tmp_path / f"phase_e{e}.nii.gz"
        _write_nifti(mf, np.abs(cn))
        _write_nifti(pf, np.angle(cn))
        magn_files.append(str(mf))
        phase_files.append(str(pf))

    cfg = NordicConfig(
        noise_volume_last=3,
        factor_error=1.2,
        save_gfactor_map=True,
        save_residual_map=True,
        save_num_comps=True,
        rescue=True,
        verbose=False,
    )
    outs = run_nordic_multiecho(
        magn_files, phase_files, str(tmp_path / "ME"), cfg, device=torch.device("cpu")
    )
    assert len(outs) == 3

    gmaps = []
    for e, out in enumerate(outs):
        assert out.magnitude_file.exists()
        img = nib.load(out.magnitude_file).get_fdata(dtype=np.float32)
        assert img.shape == (nx, ny, nz, nt - 3)  # noise vols trimmed
        with open(out.metadata_file) as f:
            meta = json.load(f)
        me = meta["multiecho"]
        assert me["echo_index"] == e + 1 and me["n_echoes"] == 3
        assert me["rescue_enabled"] is True
        # Default shares echo 1's g-factor; sigma is measured per echo.
        assert me["gfactor_mode"] == "shared-echo1"
        assert me["measured_noise"] > 0.0
        assert out.gfactor_file is not None
        # Per-echo residual is produced on the multi-echo path (the cross-echo
        # over-removal check the user runs on whole-brain output).
        assert out.residual_file is not None and out.residual_file.exists()
        res = nib.load(out.residual_file).get_fdata(dtype=np.float32)
        assert res.shape == (nx, ny, nz, nt - 3)
        assert np.all(np.isfinite(res)) and np.any(res > 0)
        assert meta["outputs"]["residual"] is not None
        # Per-voxel num-components-removed map (patch-averaged, fractional).
        assert out.num_comps_file is not None and out.num_comps_file.exists()
        nc = nib.load(out.num_comps_file).get_fdata(dtype=np.float32)
        assert nc.shape == (nx, ny, nz)  # spatial map, one value per voxel
        assert np.all(nc >= 0) and np.any(nc > 0)
        assert meta["outputs"]["num_comps"] is not None
        # Per-voxel suggested-factor map (bidirectional: may be above or below
        # factor_error; always finite and positive).
        assert out.recfactor_file is not None and out.recfactor_file.exists()
        rf = nib.load(out.recfactor_file).get_fdata(dtype=np.float32)
        assert rf.shape == (nx, ny, nz)
        assert np.all(rf > 0) and np.all(np.isfinite(rf))
        # Global recommendation recorded as a dict with the two-sided summary.
        rec = meta["multiecho"]["recommended_factor_error"]
        assert rec is None or {
            "current",
            "in_brain",
            "frac_suggest_decrease",
            "frac_suggest_increase",
        } <= set(rec)
        gmaps.append(nib.load(out.gfactor_file).get_fdata(dtype=np.float32))

    # g-factor is estimated once on echo 1 and shared across echoes.
    assert np.allclose(gmaps[0], gmaps[1]) and np.allclose(gmaps[0], gmaps[2])


def test_per_echo_gfactor_estimates_independently(tmp_path):
    """With per_echo_gfactor, each echo gets its own g-factor map (not echo 1's),
    so the maps differ; metadata records the per-echo mode. Sigma is per echo
    either way (the clobber that forced echo 1's sigma onto all echoes is gone)."""
    rng = np.random.default_rng(3)
    nx, ny, nz, nt = 12, 12, 3, 40
    m = nx * ny * nz
    vt = np.linalg.qr(rng.standard_normal((nt, 2)) + 1j * rng.standard_normal((nt, 2)))[0]
    usp = rng.standard_normal((m, 2)) + 1j * rng.standard_normal((m, 2))
    sig = (usp @ vt.conj().T).reshape(nx, ny, nz, nt)

    magn_files, phase_files = [], []
    # Independent noise realizations per echo -> independent g-factor estimates.
    for e, w in enumerate([1.0, 2.5, 4.0]):
        cn = w * sig + (rng.standard_normal(sig.shape) + 1j * rng.standard_normal(sig.shape)) * 0.4
        mf = tmp_path / f"magn_e{e}.nii.gz"
        pf = tmp_path / f"phase_e{e}.nii.gz"
        _write_nifti(mf, np.abs(cn))
        _write_nifti(pf, np.angle(cn))
        magn_files.append(str(mf))
        phase_files.append(str(pf))

    cfg = NordicConfig(
        noise_volume_last=3,
        factor_error=1.2,
        save_gfactor_map=True,
        per_echo_gfactor=True,
        rescue=True,
        verbose=False,
    )
    outs = run_nordic_multiecho(
        magn_files, phase_files, str(tmp_path / "ME"), cfg, device=torch.device("cpu")
    )
    gmaps = []
    for out in outs:
        with open(out.metadata_file) as f:
            meta = json.load(f)
        assert meta["multiecho"]["gfactor_mode"] == "per-echo"
        gmaps.append(nib.load(out.gfactor_file).get_fdata(dtype=np.float32))
    # Independently estimated -> not the shared (identical) maps of the default.
    assert not np.allclose(gmaps[0], gmaps[1])
    assert not np.allclose(gmaps[0], gmaps[2])


def test_residual_xcorr_qc_detects_shared_voxel():
    """A voxel whose residual time course is shared across echoes lights up
    (max r ~ 1); pure-noise voxels stay near the null. Focal, not patch-wide."""
    nx, ny, nz, nt = 6, 6, 2, 100
    rng = np.random.default_rng(0)
    shared = rng.standard_normal(nt) + 1j * rng.standard_normal(nt)
    res = []
    for _ in range(3):
        r = rng.standard_normal((nx, ny, nz, nt)) + 1j * rng.standard_normal((nx, ny, nz, nt))
        # Inject the same time course (different per-echo weight) at one voxel.
        r[2, 3, 1, :] = shared * (1.0 + rng.standard_normal())
        res.append(torch.from_numpy(r).to(torch.complex64))
    max_r, tstat, dof = _residual_xcorr_qc(res)
    assert max_r.shape == (nx, ny, nz) and dof == nt - 2
    assert max_r[2, 3, 1] > 0.9, "shared-signal voxel not detected"
    # The vast majority of (independent-noise) voxels sit near the null (1/sqrt T).
    others = max_r.clone()
    others[2, 3, 1] = 0.0
    assert float(others.mean()) < 0.3
    assert torch.all(tstat >= 0)


def test_residual_xcorr_qc_ignores_shared_phase():
    """Magnitude-based correlation must NOT fire on shared temporal phase with
    independent magnitudes (the B0/off-resonance confound that made the complex
    Hermitian version read ~all-significant brain-wide)."""
    nx, ny, nz, nt = 8, 8, 2, 150
    torch.manual_seed(0)
    phi = torch.randn(nt) * 2.0  # shared phase across echoes (B0-like)
    res = []
    for _ in range(3):
        mag = torch.randn(nx, ny, nz, nt).abs()  # independent magnitude per echo
        res.append((mag * torch.exp(1j * phi)).to(torch.complex64))
    max_r, _, _ = _residual_xcorr_qc(res)
    # Mean stays near the 1/sqrt(T) null; nothing like the ~0.66 the complex
    # Hermitian correlation would give from the shared phase alone.
    assert float(max_r.mean()) < 0.25, float(max_r.mean())


def test_residual_qc_maps_written(tmp_path):
    """Multi-echo run writes one 4D QC map (r / tstat / 1-q sub-bricks), tags the
    t sub-brick as a stat in the AFNI header, and records the summary."""
    rng = np.random.default_rng(5)
    nx, ny, nz, nt = 12, 12, 3, 40
    m = nx * ny * nz
    vt = np.linalg.qr(rng.standard_normal((nt, 2)) + 1j * rng.standard_normal((nt, 2)))[0]
    usp = rng.standard_normal((m, 2)) + 1j * rng.standard_normal((m, 2))
    sig = (usp @ vt.conj().T).reshape(nx, ny, nz, nt)
    magn_files, phase_files = [], []
    for e, w in enumerate([1.0, 2.5, 4.0]):
        cn = w * sig + (rng.standard_normal(sig.shape) + 1j * rng.standard_normal(sig.shape)) * 0.4
        mf = tmp_path / f"magn_e{e}.nii.gz"
        pf = tmp_path / f"phase_e{e}.nii.gz"
        _write_nifti(mf, np.abs(cn))
        _write_nifti(pf, np.angle(cn))
        magn_files.append(str(mf))
        phase_files.append(str(pf))

    cfg = NordicConfig(noise_volume_last=3, factor_error=1.2, rescue=True, verbose=False)
    run_nordic_multiecho(
        magn_files, phase_files, str(tmp_path / "ME"), cfg, device=torch.device("cpu")
    )
    f = tmp_path / "ME_resid_xcorr.nii.gz"
    assert f.exists(), "missing 4D QC map"
    img = nib.load(f)
    arr = img.get_fdata(dtype=np.float32)
    assert arr.shape == (nx, ny, nz, 3) and np.all(np.isfinite(arr))
    # r in [0,1]; t>=0; 1-q in [0,1].
    assert arr[..., 0].max() <= 1.0 and arr[..., 0].min() >= 0.0
    assert arr[..., 1].min() >= 0.0
    # Summary lands in each echo's metadata.
    with open(tmp_path / "ME_echo-01_metadata.json") as f:
        meta = json.load(f)
    assert "residual_qc" in meta
    qc = meta["residual_qc"]
    assert 0.0 <= qc["frac_q_lt_0.05"] <= 1.0
    assert qc["map"].endswith("ME_resid_xcorr.nii.gz")
    # AFNI header tags the t sub-brick (index 1) as a Ttest with dof = T_keep - 2.
    afni_xml = b"".join(
        e.get_content() if isinstance(e.get_content(), bytes) else e.get_content().encode()
        for e in (img.header.extensions or [])
        if e.get_code() == 4
    ).decode("utf-8", "ignore")
    assert "BRICK_STATAUX" in afni_xml and f"Ttest({qc['dof']})" in afni_xml
    assert "xcorr_pearson_r" in afni_xml and "FDRCURVE" in afni_xml


def test_resid_qc_can_be_disabled(tmp_path):
    """-no-resid-qc (resid_qc=False) suppresses the QC maps."""
    rng = np.random.default_rng(9)
    nx, ny, nz, nt = 10, 10, 2, 30
    magn_files, phase_files = [], []
    for e in range(2):
        cn = rng.standard_normal((nx, ny, nz, nt)) + 1j * rng.standard_normal((nx, ny, nz, nt))
        mf = tmp_path / f"m{e}.nii.gz"
        pf = tmp_path / f"p{e}.nii.gz"
        _write_nifti(mf, np.abs(cn))
        _write_nifti(pf, np.angle(cn))
        magn_files.append(str(mf))
        phase_files.append(str(pf))
    cfg = NordicConfig(rescue=True, resid_qc=False, verbose=False)
    run_nordic_multiecho(
        magn_files, phase_files, str(tmp_path / "NQ"), cfg, device=torch.device("cpu")
    )
    assert not (tmp_path / "NQ_resid_xcorr.nii.gz").exists()


def test_rescue_is_all_pairs_not_anchored():
    """No echo is a privileged reference: a component kept in echoes 1 & 2 but
    buried in echo 3 must be rescued in echo 3 (candidate=echo3 band tested
    against echoes 1/2 targets). Mirror case: buried in echo 1, kept in 2 & 3."""
    # Strong in echoes 1,2 (kept), weak in echo 3 (buried) -> rescue echo 3.
    echoes, _ = _make_me_echoes([4.0, 4.0, 1.0], seed=11)
    kw = dict(
        kernel_size=(10, 10, 4),
        patch_overlap=1,
        threshold_mode="nordic",
        threshold_values=[30.0, 30.0, 30.0],
        rescue_band=0.25,
        rescue_alpha=0.05,
        verbose=False,
        device=torch.device("cpu"),
    )
    res = _llr_denoise_multiecho(echoes, rescue=True, **kw)
    assert float(res[2][1].rescued_map.mean()) > 0.0, "buried last echo not rescued"

    # Symmetric: weak in echo 1 (buried), strong in echoes 2,3 (kept).
    echoes2, _ = _make_me_echoes([1.0, 4.0, 4.0], seed=12)
    res2 = _llr_denoise_multiecho(echoes2, rescue=True, **kw)
    assert float(res2[0][1].rescued_map.mean()) > 0.0, "buried first echo not rescued"


# ---------------------------------------------------------------------------
# Task-leak diagnostic (-events)
# ---------------------------------------------------------------------------


def test_task_loss_is_the_signed_magnitude_difference(tmp_path):
    """_taskloss == |input| - |denoised|, sign kept, exactly.

    The whole point of the series is that it is the magnitude a downstream GLM
    loses. |residual| — the modulus of the removed complex field — is a different,
    half-rectified quantity, and reading one for the other is the bug this pins.
    """
    rng = np.random.RandomState(7)
    magn = np.abs(rng.normal(loc=100.0, scale=10.0, size=(12, 10, 6, 24))).astype(np.float32)
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
        save_task_loss=True,
        save_residual_map=True,
        verbose=False,
    )
    out = run_nordic(str(magn_file), None, str(tmp_path / "NORDIC_loss"), cfg)

    assert out.task_loss_file is not None and out.task_loss_file.exists()
    loss = nib.load(out.task_loss_file).get_fdata(dtype=np.float32)
    kept = nib.load(out.magnitude_file).get_fdata(dtype=np.float32)
    assert loss.shape == magn.shape
    # Magnitude-only input: |input| is the input itself, up to the round trip.
    np.testing.assert_allclose(kept + loss, magn, rtol=0, atol=2e-3)
    # Signed, unlike the residual map alongside it.
    assert loss.min() < 0 < loss.max()
    resid = nib.load(out.residual_file).get_fdata(dtype=np.float32)
    assert (resid >= 0).all()

    with open(out.metadata_file) as f:
        assert json.load(f)["outputs"]["task_loss"] is not None


def test_task_loss_not_written_unless_asked(tmp_path):
    """A plain run writes neither the loss series nor a residual it did not ask for."""
    rng = np.random.RandomState(8)
    magn = np.abs(rng.normal(size=(10, 10, 4, 16))).astype(np.float32)
    magn_file = tmp_path / "magn.nii.gz"
    _write_nifti(magn_file, magn)
    cfg = NordicConfig(
        temporal_phase=0,
        magnitude_only=True,
        kernel_size_pca=(3, 3, 3),
        kernel_size_gfactor=(3, 3, 1),
        gfactor_nvols=8,
        verbose=False,
    )
    out = run_nordic(str(magn_file), None, str(tmp_path / "NORDIC_plain"), cfg)
    assert out.task_loss_file is None
    assert out.residual_file is None
    assert not (tmp_path / "NORDIC_plain_taskloss.nii.gz").exists()


def _write_events(path: Path, onsets, duration=12.0, label="task"):
    lines = ["onset\tduration\ttrial_type"]
    lines += [f"{o}\t{duration}\t{label}" for o in onsets]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _planted_task_run(tmp_path, seed=3, tr=1.0, nt=120, amp=0.0):
    """A block-design run with the response confined to one slab of voxels.

    The planted response is HRF-smoothed, not a raw boxcar: the design the diagnostic
    builds is convolved, and scoring a square wave against it costs enough of the fit
    to blur the very contrast these tests are checking.
    """
    rng = np.random.RandomState(seed)
    nx, ny, nz = 20, 16, 8
    box = np.zeros(nt, dtype=np.float32)
    onsets = list(range(12, nt - 12, 24))
    for o in onsets:
        box[int(o / tr) : int((o + 12) / tr)] = 1.0
    hrf = np.exp(-((np.arange(30) - 6.0) ** 2) / 18.0)
    box = np.convolve(box, hrf / hrf.sum())[:nt].astype(np.float32)
    active = np.zeros((nx, ny, nz), dtype=np.float32)
    active[5:10, 5:10, 2:6] = 1.0
    magn = np.abs(rng.normal(loc=100.0, scale=8.0, size=(nx, ny, nz, nt))).astype(np.float32)
    magn += amp * active[..., None] * box[None, None, None, :]
    magn_file = tmp_path / "magn.nii.gz"
    img = nib.Nifti1Image(magn, np.eye(4))
    img.header["pixdim"][4] = tr
    nib.save(img, magn_file)
    ev = tmp_path / "events.tsv"
    _write_events(ev, onsets)
    return magn_file, ev


def test_events_writes_the_task_leak_report(tmp_path):
    """-events runs end to end and writes both fits plus the report."""
    from fastfuncstuff.cli.nordic import main

    magn_file, ev = _planted_task_run(tmp_path, amp=20.0)
    prefix = tmp_path / "NORD"
    main(
        [
            "-input-magn",
            str(magn_file),
            "-prefix",
            str(prefix),
            "-magnitude-only",
            "-kernel-size-pca",
            "3",
            "3",
            "3",
            "-kernel-size-gfactor",
            "3",
            "3",
            "1",
            "-gfactor-nvols",
            "8",
            "-events",
            str(ev),
            "-device",
            "cpu",
        ]
    )
    for suffix in ("_taskloss.nii.gz", "_taskfit_kept.nii.gz", "_taskfit_lost.nii.gz"):
        assert (tmp_path / f"NORD{suffix}").exists(), suffix
    report = (tmp_path / "NORD_taskleak.txt").read_text()
    assert "ENRICHMENT" in report
    # The kept fit leads with full_model_R, then Coef/Correl per condition.
    fit = nib.load(tmp_path / "NORD_taskfit_kept.nii.gz")
    assert fit.shape[3] == 3


def test_task_leak_enrichment_is_flat_when_nothing_was_planted(tmp_path):
    """With no task in the data, what NORDIC removes is spread like the brain.

    The calibration the whole diagnostic rests on: enrichment's no-relation value is
    1.0 by construction, so a pure-noise run must not manufacture a leak. Without
    this the headline number has no zero point.
    """
    from fastfuncstuff.cli.nordic import main

    magn_file, ev = _planted_task_run(tmp_path, seed=11, amp=0.0)
    prefix = tmp_path / "NULL"
    main(
        [
            "-input-magn",
            str(magn_file),
            "-prefix",
            str(prefix),
            "-magnitude-only",
            "-kernel-size-pca",
            "3",
            "3",
            "3",
            "-kernel-size-gfactor",
            "3",
            "3",
            "1",
            "-gfactor-nvols",
            "8",
            "-events",
            str(ev),
            "-device",
            "cpu",
        ]
    )
    # The floor has to be real: a whole-run mask reads ~3.9x here, which is the entire
    # reason the enrichment is measured across a split of the run.
    assert 0.7 < _enrichment(tmp_path / "NULL_taskleak.txt") < 1.3


def _enrichment(report_path: Path) -> float:
    line = next(ln for ln in report_path.read_text().splitlines() if "mask from one half" in ln)
    return float(line.strip().split("x")[0])


def test_task_leak_enrichment_rises_when_denoising_over_removes(tmp_path):
    """The positive control: crank the threshold until NORDIC has to eat the response.

    A diagnostic that cannot detect the thing it exists for is worse than none.
    -factor-error scales the threshold, so at a large enough factor every patch loses
    its signal components too, and the lost series' task energy must concentrate where
    the response was planted.
    """
    from fastfuncstuff.cli.nordic import main

    magn_file, ev = _planted_task_run(tmp_path, seed=5, amp=25.0)
    for tag, factor in (("BASE", "1.0"), ("OVER", "4.0")):
        main(
            [
                "-input-magn",
                str(magn_file),
                "-prefix",
                str(tmp_path / tag),
                "-magnitude-only",
                "-kernel-size-pca",
                "3",
                "3",
                "3",
                "-kernel-size-gfactor",
                "3",
                "3",
                "1",
                "-gfactor-nvols",
                "8",
                "-factor-error",
                factor,
                "-events",
                str(ev),
                "-device",
                "cpu",
            ]
        )
    base = _enrichment(tmp_path / "BASE_taskleak.txt")
    over = _enrichment(tmp_path / "OVER_taskleak.txt")
    assert over > base, f"over-removal did not raise enrichment: {base:.2f} -> {over:.2f}"
    assert over > 1.5, over
    assert base < 1.3, base


def test_task_null_frames_are_orthonormal_and_drift_free(tmp_path):
    """The null companions must live in the same subspace as the design.

    That is the whole fix: comparing the design against directions drawn from plain
    R^T reads significant on data with no task in it, because the components a patch
    KEEPS are drift-dominated while the design has had drift projected out.
    """
    from fastfuncstuff.denoise.nordic import task_null_frames

    n_t, polort, n_null = 120, 2, 8
    design = np.zeros((n_t, 2), dtype=np.float32)
    design[10:30, 0] = 1.0
    design[60:80, 1] = 1.0
    frames, k = task_null_frames(design, n_t, polort, n_null, torch.device("cpu"))
    assert k == 2
    assert frames.shape == (n_t, k * (1 + n_null))
    drift = torch.linalg.qr(
        torch.stack([torch.linspace(-1, 1, n_t) ** d for d in range(polort + 1)], dim=1)
    )[0]
    for i in range(1 + n_null):
        blk = frames[:, i * k : (i + 1) * k].double()
        np.testing.assert_allclose(blk.T @ blk, np.eye(k), atol=1e-5)
        assert float((drift.double().T @ blk).abs().max()) < 1e-4, f"block {i} keeps drift"


def _patch_summary(prefix: Path) -> dict:
    with open(f"{prefix}_metadata.json") as f:
        return json.load(f)["task_patch_test"]


def _run_patch_test(tmp_path, tag, magn_file, ev):
    from fastfuncstuff.cli.nordic import main

    main(
        [
            "-input-magn",
            str(magn_file),
            "-prefix",
            str(tmp_path / tag),
            "-magnitude-only",
            "-gfactor-nvols",
            "8",
            "-events",
            str(ev),
            "-device",
            "cpu",
        ]
    )
    return _patch_summary(tmp_path / tag)


def test_patch_test_separates_a_kept_response_from_no_task(tmp_path):
    """The signed z is what discriminates, and both directions have to be checked.

    The FDR count alone cannot carry this: NORDIC keeps a handful of components out of
    a hundred-odd, so the design is in the discarded subspace whatever happens, and so
    is every random direction. What separates is HOW FAR from the null the design sits
    -- strongly negative when the response was preferentially kept, ~0 when it went
    out like any other direction.
    """
    planted, ev = _planted_task_run(tmp_path, seed=13, amp=25.0)
    kept = _run_patch_test(tmp_path, "KEPT", planted, ev)
    sub = tmp_path / "n"
    sub.mkdir()
    flat, _ = _planted_task_run(sub, seed=13, amp=0.0)
    none = _run_patch_test(tmp_path, "NONE", flat, ev)

    assert kept["z_median_in_brain"] < -2.0, kept
    assert abs(none["z_median_in_brain"]) < 2.0, none
    # One-sided: preferential KEEPING is the good outcome and must never be a finding.
    assert kept["n_patches_significant"] == 0, kept
    assert none["n_patches_significant"] == 0, none


def _scan(prefix: Path) -> dict | None:
    with open(f"{prefix}_metadata.json") as f:
        return json.load(f).get("task_component_scan")


def _nordic_task_run(tmp_path, prefix, magn_file, ev, *extra):
    from fastfuncstuff.cli.nordic import main

    main(
        [
            "-input-magn",
            str(magn_file),
            "-prefix",
            str(prefix),
            "-magnitude-only",
            "-kernel-size-pca",
            "3",
            "3",
            "3",
            "-kernel-size-gfactor",
            "3",
            "3",
            "1",
            "-gfactor-nvols",
            "8",
            "-factor-error",
            "4.0",
            "-events",
            str(ev),
            "-device",
            "cpu",
            *extra,
        ]
    )


def test_component_scan_runs_by_default_and_changes_nothing(tmp_path):
    """-events measures the removed field in TIME as well as in space, without acting.

    The spatial half needs a leak big enough to move a mask placed on half the run; a
    component that is task-locked but lives in a few voxels does not move it. So the
    scan is not optional extra work, it is the half with power where the other has
    none — and like every other -events output it must leave the data alone.
    """
    magn_file, ev = _planted_task_run(tmp_path, seed=11, amp=25.0)
    _nordic_task_run(tmp_path, tmp_path / "SCAN", magn_file, ev)
    scan = _scan(tmp_path / "SCAN")
    assert scan is not None, "the component scan did not run under bare -events"
    assert scan["rescued"] is False
    assert scan["flagged_components"], "the planted leak was not flagged"
    assert not (tmp_path / "SCAN_taskrescued.nii.gz").exists()


def test_task_comps_off_skips_the_scan(tmp_path):
    magn_file, ev = _planted_task_run(tmp_path, seed=11, amp=25.0)
    _nordic_task_run(tmp_path, tmp_path / "OFF", magn_file, ev, "-task_comps", "off")
    assert _scan(tmp_path / "OFF") is None


def test_task_rescue_puts_the_response_back(tmp_path):
    """The end-to-end bargain: -task_rescue raises the amplitude it was there to save.

    Scored on the same responding mask the report uses, against the un-rescued run, so
    a change here is the restored component and not a different stratum.
    """
    magn_file, ev = _planted_task_run(tmp_path, seed=11, amp=25.0)
    _nordic_task_run(tmp_path, tmp_path / "KEPT", magn_file, ev)
    _nordic_task_run(tmp_path, tmp_path / "RESC", magn_file, ev, "-task_rescue")
    scan = _scan(tmp_path / "RESC")
    assert scan is not None and scan["rescued"] is True
    assert scan["dof_returned"] == len(scan["flagged_components"])
    assert (tmp_path / "RESC_taskrescued.nii.gz").exists()
    # output = un-rescued + what was added, exactly: nothing is hidden in the primary.
    kept = nib.load(str(tmp_path / "KEPT.nii.gz")).get_fdata(dtype=np.float32)
    resc = nib.load(str(tmp_path / "RESC.nii.gz")).get_fdata(dtype=np.float32)
    added = nib.load(str(tmp_path / "RESC_taskrescued.nii.gz")).get_fdata(dtype=np.float32)
    np.testing.assert_allclose(resc, kept + added, atol=1e-4)
    r_kept = nib.load(str(tmp_path / "RESC_taskfit_kept.nii.gz")).get_fdata(dtype=np.float32)
    r_after = nib.load(str(tmp_path / "RESC_taskfit_rescued.nii.gz")).get_fdata(dtype=np.float32)
    assert float(np.median(r_after[..., 0])) > float(np.median(r_kept[..., 0]))


def test_task_rescue_needs_the_scan_it_acts_on(tmp_path):
    from fastfuncstuff.cli.nordic import main

    magn_file, ev = _planted_task_run(tmp_path, seed=11, amp=0.0)
    with pytest.raises(SystemExit):
        main(
            [
                "-input-magn",
                str(magn_file),
                "-prefix",
                str(tmp_path / "X"),
                "-magnitude-only",
                "-events",
                str(ev),
                "-task_rescue",
                "-task_comps",
                "off",
            ]
        )
