"""Tests for the ffs_moco -reweight data-driven weight refinement pre-pass."""

from __future__ import annotations

import torch

from fastfuncstuff.processing.affine import (
    apply_affine_interp,
    params_to_matrix,
    params_to_matrix_batched,
)
from fastfuncstuff.processing.cost import _separable_smooth_3d
from fastfuncstuff.processing.ffs_moco import MocoConfig, compute_derivative_images
from fastfuncstuff.processing.moco_reweight import compute_residual_reweight, compute_reweight
from fastfuncstuff.processing.weight import compute_weight_image

DEVICE = torch.device("cpu")


def _structured_base(shape, seed=0):
    """Smooth random volume — has gradients in every direction so the per-patch
    Gauss-Newton motion estimate is well posed."""
    g = torch.Generator().manual_seed(seed)
    vol = torch.randn(*shape, generator=g)
    return _separable_smooth_3d(vol.abs(), sigma=1.5)


def _rigid(base, params6, interp="cubic"):
    p = torch.zeros(12, dtype=base.dtype)
    p[6] = p[7] = p[8] = 1.0
    p[:6] = torch.tensor(params6, dtype=base.dtype)
    mat = params_to_matrix(p)
    return apply_affine_interp(base, mat, interp, tuple(base.shape), zero_outside=True)


def _make_series(base, nt=24, artifact_region=None):
    """Global rigid motion over time; if artifact_region given, that box moves
    with the *opposite* motion (an anti-correlated artifact patch)."""
    import math

    vols = []
    motions = []
    for t in range(nt):
        ph = 2 * math.pi * t / nt
        m = [
            1.0 * math.sin(ph),  # dx
            0.8 * math.sin(ph + 0.7),  # dy
            0.6 * math.cos(ph),  # dz
            1.4 * math.cos(ph + 0.3),  # rz (deg)
            1.1 * math.sin(ph + 1.1),  # rx
            0.9 * math.cos(ph + 2.0),  # ry
        ]
        motions.append(m)
        src = _rigid(base, m)
        if artifact_region is not None:
            opp = _rigid(base, [-v for v in m])
            zs, ys, xs = artifact_region
            src = src.clone()
            src[zs, ys, xs] = opp[zs, ys, xs]
        vols.append(src)
    return torch.stack(vols), torch.tensor(motions, dtype=base.dtype)


def _global_matrices(motions):
    """Ground-truth voxel-space alignment transforms from the known motion.

    The series warps the base by ``M_motion`` (``src(p) = base(M_motion @ p)``);
    the registration fit that aligns source back to base is therefore the inverse,
    matching what ``moco`` returns in ``matrices_vox`` (base->source pull that
    undoes the motion). The reweight prediction must use that same fit direction.
    """
    nt = motions.shape[0]
    p12 = torch.zeros(nt, 12, dtype=motions.dtype)
    p12[:, :6] = motions
    p12[:, 6:9] = 1.0
    return torch.linalg.inv(params_to_matrix_batched(p12))


def test_reweight_rejects_anticorrelated_patch():
    shape = (20, 32, 32)
    base = _structured_base(shape).to(DEVICE)
    weight0 = compute_weight_image(base).to(DEVICE)
    derivs = compute_derivative_images(base, DEVICE)

    # A large block whose content moves opposite to the global head motion —
    # its local displacement contradicts the global prediction, so it should be
    # dropped while the coherent rest is kept.
    artifact = (slice(3, 17), slice(3, 16), slice(3, 16))
    series, motions = _make_series(base, nt=24, artifact_region=artifact)
    global_matrices = _global_matrices(motions)

    res = compute_reweight(
        base,
        series,
        weight0,
        derivs,
        global_matrices,
        voxdims=(1.0, 1.0, 1.0),
        tr=1.0,
        max_iter=8,
        device=DEVICE,
        verb=0,
    )

    assert res.applied
    assert res.n_kept < res.n_patches  # something was dropped

    w_orig = weight0.cpu()
    w_new = res.weight.cpu()

    # The artifact box should be largely zeroed where it had weight.
    art_mask = torch.zeros(shape, dtype=torch.bool)
    art_mask[artifact] = True
    had_weight = art_mask & (w_orig > 0)
    dropped = had_weight & (w_new == 0)
    assert dropped.sum() >= 0.5 * had_weight.sum()

    # A coherent region far from the artifact should mostly keep its weight.
    far = torch.zeros(shape, dtype=torch.bool)
    far[10:18, 20:30, 20:30] = True
    far_had = far & (w_orig > 0)
    far_kept = far_had & (w_new > 0)
    assert far_kept.sum() >= 0.5 * far_had.sum()

    # Refined weight only ever removes weight, never adds it.
    assert torch.all(w_new <= w_orig + 1e-6)

    # A patch label map with distinct ids for kept patches.
    assert int(res.patch_labels.max()) == res.n_kept


def test_reweight_low_motion_guard():
    shape = (18, 30, 30)
    base = _structured_base(shape, seed=3).to(DEVICE)
    weight0 = compute_weight_image(base).to(DEVICE)
    derivs = compute_derivative_images(base, DEVICE)

    # No motion: every volume is the base, global transforms are identity.
    # Nothing to learn -> guard skips, weight unchanged.
    nt = 20
    series = base[None].repeat(nt, 1, 1, 1)
    global_matrices = torch.eye(4, dtype=base.dtype)[None].repeat(nt, 1, 1)

    res = compute_reweight(
        base,
        series,
        weight0,
        derivs,
        global_matrices,
        voxdims=(1.0, 1.0, 1.0),
        tr=2.0,
        max_iter=6,
        device=DEVICE,
        verb=0,
    )

    assert not res.applied
    assert torch.equal(res.weight.cpu(), weight0.cpu())


def test_residual_reweight_softly_penalizes_discordant_region():
    shape = (20, 32, 32)
    base = _structured_base(shape, seed=8).to(DEVICE)
    weight0 = compute_weight_image(base).to(DEVICE)
    artifact = (slice(3, 17), slice(3, 16), slice(3, 16))
    series, motions = _make_series(base, nt=24, artifact_region=artifact)

    res = compute_residual_reweight(
        base,
        series,
        weight0,
        _global_matrices(motions),
        config=MocoConfig(
            base_index=0,
            final_interp="cubic",
            use_shear=False,
            device="cpu",
            verb=0,
        ),
        voxdims=(1.0, 1.0, 1.0),
        smooth_fwhm=3.0,
        device=DEVICE,
        verb=0,
    )

    assert res.applied
    multiplier = res.weight / weight0.clamp(min=1e-20)
    art_mask = torch.zeros(shape, dtype=torch.bool)
    art_mask[artifact] = True
    far_mask = torch.zeros(shape, dtype=torch.bool)
    far_mask[10:18, 20:30, 20:30] = True
    art = multiplier[art_mask & (weight0 > 0)]
    far = multiplier[far_mask & (weight0 > 0)]

    assert 0.1 <= float(art.median()) < 1.0
    assert float(art.median()) + 0.2 < float(far.median())
    assert torch.all(res.weight <= weight0 + 1e-6)
    assert torch.any((res.weight > 0) & (res.weight < weight0))
