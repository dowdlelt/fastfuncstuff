"""The fused Gauss-Newton normal equations must match the portable accumulator.

The kernel exists because assembling the steepest-descent columns was the slowest
kernel at every pyramid level; it earns that by never building them, which means
nothing but a comparison against the path it replaces can tell us it is right.
"""

import pytest
import torch

import fastfuncstuff.processing.warp as warp_mod
from fastfuncstuff.processing.warp import _gn_accumulate

triton_mod = pytest.importorskip("fastfuncstuff.processing.gn_triton")
gn_normal_eqs_triton = triton_mod.gn_normal_eqs_triton


@pytest.mark.gpu
@pytest.mark.parametrize(
    "b,v,d,nb,has_mean,per_voxel_scale",
    [
        (3, 210, 3, 8, True, False),  # smallest sane patch batch
        (3, 210, 3, 8, False, True),  # the local path: no centring, per-voxel scale
        (27, 4096, 3, 10, True, False),  # a coarse level
        (7, 999, 1, 8, False, False),  # a single active direction (-noYdis -noZdis)
        (53, 1024, 2, 10, True, False),  # two directions, padded column tile
        (4096, 729, 3, 10, True, False),  # a fine level: many patches, few voxels
        (1, 65536, 3, 10, True, False),  # level 0: one patch, the whole grid
    ],
)
def test_fused_normal_eqs_matches_accumulator(monkeypatch, b, v, d, nb, has_mean, per_voxel_scale):
    if not torch.cuda.is_available():
        pytest.skip("fused normal equations are CUDA-only")
    dev = torch.device("cuda")
    torch.manual_seed(3)
    g = torch.randn(d, b, v, device=dev)
    hw = torch.rand(d, 1, 1, device=dev) + 0.5
    bt = torch.randn(v, nb, device=dev)
    omega = torch.rand(b, v, device=dev)
    res = torch.randn(b, v, device=dev)
    if per_voxel_scale:
        scale = torch.rand(b, v, device=dev) + 0.5
    else:
        scale = (torch.rand(b, 1, device=dev) + 0.5).expand(b, v)
    mean = torch.randn(b, 1, d * nb, device=dev) if has_mean else None

    # _gn_accumulate now takes the fused path itself on CUDA, so the reference has
    # to be the chunked one it replaced -- otherwise this compares a kernel to itself.
    monkeypatch.setattr(warp_mod, "gn_normal_eqs_triton", None)
    expected_h, expected_g = _gn_accumulate(g, hw, bt, scale, omega, res, mean, v)
    monkeypatch.undo()
    actual_h, actual_g = gn_normal_eqs_triton(g, hw, bt, scale, omega, res, mean)

    assert actual_h.shape == (b, d * nb, d * nb)
    assert actual_g.shape == (b, d * nb)
    scale_h = expected_h.abs().max().clamp_min(1e-9)
    scale_g = expected_g.abs().max().clamp_min(1e-9)
    assert float((actual_h - expected_h).abs().max() / scale_h) < 2e-5
    assert float((actual_g - expected_g).abs().max() / scale_g) < 2e-5
