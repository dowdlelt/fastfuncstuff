"""Tests for patch-based residual-PC projection (ffs_pcpatch)."""

from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fastfuncstuff.denoise.pcpatch import PCPatchConfig, build_nuisance_basis, run_pcpatch


def _write_nifti(path: Path, data: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), path)


def _orthonormal_zeromean(n_t: int, k: int, rng) -> np.ndarray:
    """(T, k) columns: orthonormal AND zero-mean over time (so span ⊥ constant)."""
    g = rng.normal(size=(n_t, k)).astype(np.float64)
    g -= g.mean(axis=0, keepdims=True)
    q, _ = np.linalg.qr(g)
    return q[:, :k]


def test_pcpatch_recovers_and_removes_known_noise_subspace(tmp_path):
    """Data = mean + signal + noise, residual spans the same noise temporal
    directions (⊥ the signal). pcpatch should recover them and project the noise
    out of the data, leaving mean + signal — and preserving the temporal mean."""
    rng = np.random.default_rng(7)
    nx, ny, nz, nt = 12, 12, 6, 48
    nvox = nx * ny * nz
    r, k = 3, 4  # signal rank, noise rank

    basis = _orthonormal_zeromean(nt, r + k, rng)  # (T, r+k), mutually orthonormal
    v_sig, v_noise = basis[:, :r], basis[:, r:]  # (T, r), (T, k)

    mean_img = rng.uniform(80.0, 120.0, size=(nvox, 1)).astype(np.float64)
    a_sig = rng.normal(scale=5.0, size=(nvox, r))
    a_noise = rng.normal(scale=3.0, size=(nvox, k))
    a_resid = rng.normal(scale=3.0, size=(nvox, k))  # different noise realization

    data = mean_img + a_sig @ v_sig.T + a_noise @ v_noise.T  # (nvox, T)
    resid = a_resid @ v_noise.T  # residual spans the SAME noise directions
    signal_only = mean_img + a_sig @ v_sig.T

    data_v = data.reshape(nx, ny, nz, nt).astype(np.float32)
    resid_v = resid.reshape(nx, ny, nz, nt).astype(np.float32)
    target = signal_only.reshape(nx, ny, nz, nt).astype(np.float32)

    data_f = tmp_path / "data.nii.gz"
    resid_f = tmp_path / "resid.nii.gz"
    _write_nifti(data_f, data_v)
    _write_nifti(resid_f, resid_v)

    out = run_pcpatch(
        data_file=str(data_f),
        residual_file=str(resid_f),
        output_prefix=str(tmp_path / "clean"),
        config=PCPatchConfig(var_frac=0.99, patch_overlap=2, verbose=False),
    )
    cleaned = nib.load(out.data_file).get_fdata(dtype=np.float32)
    ncomps = nib.load(out.num_comps_file).get_fdata(dtype=np.float32)

    # Noise removed, signal + mean preserved.
    assert np.abs(cleaned - target).max() < 1e-2
    # Original data was genuinely different (noise present).
    assert np.abs(data_v - target).max() > 1.0
    # Temporal mean preserved.
    assert np.allclose(cleaned.mean(axis=-1), data_v.mean(axis=-1), atol=1e-3)
    # It removed ~k components per voxel.
    assert abs(float(ncomps.mean()) - k) < 0.5


def test_pcpatch_mask_leaves_outside_untouched(tmp_path):
    """Voxels with an all-zero residual (outside the data) keep their original
    values and report 0 components."""
    rng = np.random.default_rng(11)
    nx, ny, nz, nt = 10, 10, 4, 40
    nvox = nx * ny * nz
    k = 3
    v_noise = _orthonormal_zeromean(nt, k, rng)
    data = (rng.uniform(50, 60, size=(nvox, 1)) + rng.normal(size=(nvox, k)) @ v_noise.T).reshape(
        nx, ny, nz, nt
    )
    resid = (rng.normal(size=(nvox, k)) @ v_noise.T).reshape(nx, ny, nz, nt)
    # Zero the residual in a slab → "no data" there.
    resid[:, :, :2, :] = 0.0

    data_f = tmp_path / "d.nii.gz"
    resid_f = tmp_path / "r.nii.gz"
    _write_nifti(data_f, data.astype(np.float32))
    _write_nifti(resid_f, resid.astype(np.float32))

    out = run_pcpatch(
        data_file=str(data_f),
        residual_file=str(resid_f),
        output_prefix=str(tmp_path / "clean"),
        config=PCPatchConfig(var_frac=0.99, verbose=False),
    )
    cleaned = nib.load(out.data_file).get_fdata(dtype=np.float32)
    ncomps = nib.load(out.num_comps_file).get_fdata(dtype=np.float32)

    # Untouched where there is no residual data.
    assert np.allclose(cleaned[:, :, :2, :], data[:, :, :2, :].astype(np.float32), atol=1e-4)
    assert np.all(ncomps[:, :, :2] == 0)
    # And it did something where there is data.
    assert float(ncomps[:, :, 2:].mean()) > 0


def test_pcpatch_nuisance_projection_suppresses_removal(tmp_path):
    """If the residual is pure drift and -polort removes it, nothing is left to
    project out — cleaned ≈ data, ~0 components removed."""
    rng = np.random.default_rng(3)
    nx, ny, nz, nt = 8, 8, 4, 40
    nvox = nx * ny * nz
    t = np.linspace(-1, 1, nt)
    drift = np.stack([t, t**2], axis=1)  # (T, 2) low-order drift
    resid = (rng.normal(size=(nvox, 2)) @ drift.T).reshape(nx, ny, nz, nt)
    data = (rng.uniform(50, 60, size=(nvox, 1)) + rng.normal(scale=2.0, size=(nvox, nt))).reshape(
        nx, ny, nz, nt
    )

    data_f = tmp_path / "d.nii.gz"
    resid_f = tmp_path / "r.nii.gz"
    _write_nifti(data_f, data.astype(np.float32))
    _write_nifti(resid_f, resid.astype(np.float32))

    out = run_pcpatch(
        data_file=str(data_f),
        residual_file=str(resid_f),
        output_prefix=str(tmp_path / "clean"),
        config=PCPatchConfig(var_frac=0.99, polort=3, verbose=False),
    )
    cleaned = nib.load(out.data_file).get_fdata(dtype=np.float32)
    ncomps = nib.load(out.num_comps_file).get_fdata(dtype=np.float32)

    assert float(ncomps.mean()) < 0.2  # residual explained away by the drift basis
    assert np.abs(cleaned - data.astype(np.float32)).max() < 1e-2


def test_build_nuisance_basis_orthonormal_and_dedup():
    dev = torch.device("cpu")
    # polort 2 (3 cols) + a duplicate of the linear trend -> rank stays 3.
    t = torch.linspace(-1, 1, 30).numpy()
    ort = np.stack([t, 2.0 * t], axis=1)  # both collinear with the poly linear term
    q = build_nuisance_basis(30, 2, ort, dev)
    assert q is not None
    assert q.shape == (30, 3)
    # Orthonormal columns.
    gram = q.mT @ q
    assert torch.allclose(gram, torch.eye(3), atol=1e-5)
    # None when nothing requested.
    assert build_nuisance_basis(30, -1, None, dev) is None
