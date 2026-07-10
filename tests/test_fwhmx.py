"""Correctness for fastfuncstuff.stats.fwhmx (3dFWHMx-faithful classic + ACF).

Two layers:
* self-contained: smoothed-noise fields with a known FWHM, checking the classic
  Forman estimate recovers the per-axis widths (and ordering) and the ACF FWHM
  lands near the Gaussian truth. No AFNI needed.
* parity: when the real ``3dFWHMx`` binary is present, the ACF a/b/c/FWHM must
  match it to a couple percent on the same field.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest
import torch

from fastfuncstuff.stats.fwhmx import estimate_fwhmx_run

pytest.importorskip("scipy")
from scipy.ndimage import gaussian_filter  # noqa: E402


def _smoothed_field(nx, ny, nz, nt, sigma_vox, seed=0):
    """White noise smoothed per sub-brick by ``sigma_vox`` (scalar or per-axis)."""
    rng = np.random.default_rng(seed)
    vol = rng.standard_normal((nx, ny, nz, nt)).astype(np.float32)
    for t in range(nt):
        vol[..., t] = gaussian_filter(vol[..., t], sigma=sigma_vox, mode="nearest")
    return vol


def _to_ffs(vol, mask):
    """(X,Y,Z,T) numpy -> ((n_masked, T) tensor, (X,Y,Z) mask, shape).

    Axis-aligned like the CLI: grid stays in NIfTI (X,Y,Z) order and voxdims
    line up per-axis, so ``classic_fwhm`` comes back as (x, y, z).
    """
    nx, ny, nz, nt = vol.shape
    vol_t = np.transpose(vol, (3, 0, 1, 2))  # (T, X, Y, Z)
    mask_c = np.ascontiguousarray(mask)
    flat = vol_t.reshape(nt, -1)[:, mask_c.reshape(-1)].T
    return (
        torch.from_numpy(np.ascontiguousarray(flat)),
        torch.from_numpy(mask_c),
        (nx, ny, nz),
    )


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_classic_fwhm_recovers_isotropic_width():
    # A 2-voxel-sigma Gaussian smoothing -> FWHM = 2.3548 * 2 * vox.
    vox = 1.0
    sigma = 2.0
    vol = _smoothed_field(44, 44, 44, 40, sigma, seed=3)
    mask = np.ones(vol.shape[:3], bool)
    flat, m, shape = _to_ffs(vol, mask)
    res = estimate_fwhmx_run(flat, m, shape, (vox, vox, vox), device=_device(), progress=False)

    truth = 2.3548 * sigma * vox
    for f in res.classic_fwhm:
        assert f == pytest.approx(truth, rel=0.15)
    # ACF effective FWHM is a mix but should sit in the same neighbourhood.
    assert res.fwhm == pytest.approx(truth, rel=0.35)
    # centre of the ACF is 1 by construction -> a in (0,1), b,c > 0.
    assert 0.0 < res.a < 1.0 and res.b > 0 and res.c > 0


def test_classic_fwhm_axis_ordering_anisotropic():
    # Smooth x < y < z; the recovered classic FWHM must preserve that ordering.
    vox = 1.0
    vol = _smoothed_field(48, 44, 40, 45, (1.0, 2.0, 3.0), seed=5)
    mask = np.ones(vol.shape[:3], bool)
    flat, m, shape = _to_ffs(vol, mask)
    res = estimate_fwhmx_run(flat, m, shape, (vox, vox, vox), device=_device(), progress=False)

    fx, fy, fz = res.classic_fwhm
    assert fx < fy < fz
    assert fx == pytest.approx(2.3548 * 1.0, rel=0.2)
    assert fy == pytest.approx(2.3548 * 2.0, rel=0.2)


def test_radius_default_is_data_driven():
    # AFNI default radius = max(2.999*combined, 3.999*cbrt(voxvol)); grows with
    # smoothness, so a smoother field must use a larger ACF radius.
    mask = np.ones((40, 40, 40), bool)
    r_small = estimate_fwhmx_run(
        *_to_ffs(_smoothed_field(40, 40, 40, 40, 1.0, seed=7), mask),
        (1.0, 1.0, 1.0),
        device=_device(),
        progress=False,
    ).radius
    r_big = estimate_fwhmx_run(
        *_to_ffs(_smoothed_field(40, 40, 40, 40, 3.0, seed=7), mask),
        (1.0, 1.0, 1.0),
        device=_device(),
        progress=False,
    ).radius
    assert r_big > r_small


# ---------------------------------------------------------------------------
# Parity against the real 3dFWHMx binary (skipped when unavailable)
# ---------------------------------------------------------------------------


def _find_3dfwhmx():
    for cand in ("3dFWHMx", "/opt/mrisoftware/abin/3dFWHMx"):
        p = shutil.which(cand) or (cand if os.path.exists(cand) else None)
        if p:
            return p
    return None


@pytest.mark.parametrize("sigma,vox", [(2.0, 1.0), (1.5, 2.0)])
def test_parity_with_3dfwhmx_acf(sigma, vox):
    afni = _find_3dfwhmx()
    if afni is None:
        pytest.skip("3dFWHMx binary not found")
    try:
        import nibabel as nib
    except ImportError:
        pytest.skip("nibabel not available")

    nx = ny = nz = 40
    nt = 60
    vol = _smoothed_field(nx, ny, nz, nt, sigma, seed=0)
    mask = np.ones((nx, ny, nz), bool)

    d = tempfile.mkdtemp()
    aff = np.diag([vox, vox, vox, 1.0]).astype(np.float32)
    fp, mp = os.path.join(d, "x.nii.gz"), os.path.join(d, "m.nii.gz")
    nib.save(nib.Nifti1Image(vol, aff), fp)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), aff), mp)
    out = subprocess.run(
        [afni, "-acf", "NULL", "-mask", mp, "-input", fp],
        capture_output=True,
        text=True,
    )
    lines = [ln for ln in out.stdout.splitlines() if ln.strip() and not ln.startswith("#")]
    a_af, b_af, c_af, f_af = (float(x) for x in lines[-1].split())

    res = estimate_fwhmx_run(*_to_ffs(vol, mask), (vox, vox, vox), device=_device(), progress=False)
    # FWHM is what feeds 3dClustSim; tightest tolerance there.
    assert res.fwhm == pytest.approx(f_af, rel=0.03)
    assert res.a == pytest.approx(a_af, abs=0.03)
    assert res.b == pytest.approx(b_af, rel=0.08)
