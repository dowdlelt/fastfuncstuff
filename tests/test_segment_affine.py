"""spm_maff8 port: TPM-driven affine registration (processing/segment_affine.py)."""

from __future__ import annotations

import math

import numpy as np
import torch

from fastfuncstuff.processing.segment_affine import (
    affine_to_params,
    params_to_affine,
    register_to_tpm,
)

CPU = torch.device("cpu")


def test_polar_params_round_trip():
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.normal(0, 10, 3), rng.normal(0, 0.2, 3), rng.normal(0, 0.05, 6)])
    np.testing.assert_allclose(affine_to_params(params_to_affine(p)), p, atol=1e-9)


def _rotation(deg: float) -> np.ndarray:
    a = math.radians(deg)
    m = np.eye(4)
    m[1:3, 1:3] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    return m


def _synthetic_tpm(n: int = 40, vox: float = 3.0):
    """Nested soft shells (GM-like rim, WM-like core, CSF-like edge) + background."""
    aff = np.diag([vox, vox, vox, 1.0])
    aff[:3, 3] = -vox * (n - 1) / 2
    idx = torch.arange(n, dtype=torch.float64)
    zz, yy, xx = torch.meshgrid(idx, idx, idx, indexing="ij")
    c = (n - 1) / 2
    # an ellipsoid, so rotation about x is observable
    r = vox * torch.sqrt((xx - c) ** 2 + ((yy - c) / 1.3) ** 2 + ((zz - c) / 0.8) ** 2)

    def shell(lo, hi):
        return torch.sigmoid((r - lo) / 2.0) * torch.sigmoid((hi - r) / 2.0)

    # an off-centre core-tissue lobe in the rim breaks the remaining symmetry
    lobe = torch.sigmoid(
        (8.0 - vox * torch.sqrt((xx - c - 7) ** 2 + (yy - c - 4) ** 2 + (zz - c + 3) ** 2)) / 2.0
    )
    rim = shell(18, 30) * (1 - lobe)
    core = shell(-99, 18) + shell(18, 30) * lobe
    p = torch.stack([core, rim, shell(30, 38)])
    p = torch.cat([p, (1 - p.sum(0, keepdim=True)).clamp_min(0)])
    p = p / p.sum(0, keepdim=True)
    log_prior = torch.log(p + 1e-4).float()
    return log_prior, aff, p[:, 0, 0, 0].float(), p[:, -1, 0, 0].float()


def test_register_to_tpm_recovers_known_affine():
    """Subject = the TPM's tissue seen through a known affine; registration undoes it."""
    log_prior, tpm_aff, bg_lo, bg_hi = _synthetic_tpm()
    true = _rotation(10.0)
    true[:3, 3] = [5.0, -4.0, 3.0]  # subject world -> TPM world

    n, vox = 48, 2.5
    subj_aff = np.diag([vox, vox, vox, 1.0])
    subj_aff[:3, 3] = -vox * (n - 1) / 2
    idx = torch.arange(n, dtype=torch.float64)
    zz, yy, xx = torch.meshgrid(idx, idx, idx, indexing="ij")
    pts = torch.stack([xx, yy, zz], -1).reshape(-1, 3)
    vox2vox = np.linalg.inv(tpm_aff) @ true @ subj_aff
    from fastfuncstuff.processing.segment import apply_affine_pts, sample_tpm_prior

    prior = sample_tpm_prior(
        log_prior, apply_affine_pts(pts, torch.as_tensor(vox2vox)).float(), bg_lo, bg_hi
    )
    means = torch.tensor([60.0, 110.0, 30.0, 5.0])
    gen = torch.Generator().manual_seed(0)
    # each voxel is ONE class drawn from the prior — the model's own generative story. A
    # partial-volume blend (prior @ means) has a genuinely shifted MI optimum, and argmax
    # labels leave a local optimum on the way in from identity.
    label = torch.multinomial(prior.double(), 1, generator=gen)[:, 0]
    img = means[label].reshape(n, n, n) + 3.0 * torch.randn(n, n, n, generator=gen)

    est, _ = register_to_tpm(
        img, subj_aff, log_prior, tpm_aff, bg_lo, bg_hi, samp=2.5, regtype="rigid"
    )
    # compare where it matters: displacement over the head
    corners = np.array([[x, y, z, 1.0] for x in (-40, 40) for y in (-40, 40) for z in (-40, 40)])
    err = np.abs(corners @ (est - true).T)[:, :3].max()
    start = np.abs(corners @ (np.eye(4) - true).T)[:, :3].max()
    assert start > 8.0
    assert err < 1.0, f"max displacement error {err:.2f} mm (started at {start:.1f})"
