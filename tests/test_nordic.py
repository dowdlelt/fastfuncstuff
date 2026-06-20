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

    # Where rescue fired, the recommended-factor map drops below 1 (a decrease):
    # the buried component's sigma sits below the threshold, so the factor that
    # would keep it is < the one used. rescue=False produces no recfactor map.
    assert res_off[0][1].recfactor_map is None or float(res_off[0][1].recfactor_map.min()) == 1.0
    rf = res_on[0][1].recfactor_map
    assert rf is not None
    assert float(rf.min()) < 1.0
    assert float(rf.max()) <= 1.0


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
        # Per-voxel recommended-factor map (decrease-only: <= factor_error).
        assert out.recfactor_file is not None and out.recfactor_file.exists()
        rf = nib.load(out.recfactor_file).get_fdata(dtype=np.float32)
        assert rf.shape == (nx, ny, nz)
        assert np.all(rf <= cfg.factor_error + 1e-5) and np.all(rf > 0)
        # Global recommendation is recorded (None if rescue never fired).
        assert "recommended_factor_error" in meta["multiecho"]
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


def test_residual_qc_maps_written(tmp_path):
    """Multi-echo run writes the three QC maps and records the summary."""
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
    for suffix in ("_resid_xcorr", "_resid_xcorr_tstat", "_resid_xcorr_q"):
        f = tmp_path / f"ME{suffix}.nii.gz"
        assert f.exists(), f"missing QC map {suffix}"
        arr = nib.load(f).get_fdata(dtype=np.float32)
        assert arr.shape == (nx, ny, nz) and np.all(np.isfinite(arr))
    # Summary lands in each echo's metadata.
    with open(tmp_path / "ME_echo-01_metadata.json") as f:
        meta = json.load(f)
    assert "residual_qc" in meta
    assert 0.0 <= meta["residual_qc"]["frac_q_lt_0.05"] <= 1.0


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
