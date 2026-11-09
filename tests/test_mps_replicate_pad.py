"""Metal's depth-axis replicate pad is wrong for large planes; the blurs must not inherit it.

torch 2.14 on MPS: F.pad(mode="replicate") along depth returns garbage once the
in-plane size H*W reaches 2**16 (255^2 is fine, 256^2 is not; 20x300x300 fails
too). conv3d itself is exact. `_separable_smooth_3d` hit it on any base of 256+
voxels per side, which silently deleted most of a 320^3 brain from
ffs_allineate's work grid, and zeroed the autoweight.
"""

import pytest
import torch
import torch.nn.functional as F

from fastfuncstuff.processing.cost import _separable_smooth_3d
from fastfuncstuff.utils import replicate_pad_3d

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS-only bug")


@pytest.mark.parametrize("shape", [(20, 300, 300), (64, 256, 256), (32, 255, 255)])
@pytest.mark.parametrize("sigma", [(1.0, 0.0, 0.0), (0.5, 0.5, 0.5)])
def test_mps_blur_matches_cpu_across_plane_limit(shape, sigma):
    g = torch.Generator().manual_seed(0)
    x = torch.rand(*shape, generator=g)
    ref = _separable_smooth_3d(x, sigma)
    got = _separable_smooth_3d(x.to("mps"), sigma).cpu()
    # Wrong output differs by ~1; a correct one by float32 rounding.
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(20, 300, 300), (8, 256, 256), (8, 255, 255)])
def test_replicate_pad_3d_matches_cpu(shape):
    x = torch.rand(1, 2, *shape, generator=torch.Generator().manual_seed(1))
    pad = (2, 1, 0, 3, 4, 2)
    ref = F.pad(x, pad, mode="replicate")
    torch.testing.assert_close(replicate_pad_3d(x.to("mps"), pad).cpu(), ref, rtol=0, atol=0)


def _large_plane_cases():
    from fastfuncstuff.processing import locomoco, segment

    kern = lambda v: torch.tensor([0.25, 0.5, 0.25], device=v.device)  # noqa: E731
    return {
        "locomoco._gaussian_blur3d": lambda v: locomoco._gaussian_blur3d(v, 1.5),
        "locomoco._blur3d_b": lambda v: locomoco._blur3d_b(v[None], 1.5),
        "segment.blur_log_prior": lambda v: segment.blur_log_prior(
            torch.log(v[None].repeat(2, 1, 1, 1) + 0.1), 1.5
        ),
        "segment._smooth_field": lambda v: segment._smooth_field(
            v[..., None].repeat(1, 1, 1, 3), 1.5
        ),
        "segment._smooth3": lambda v: segment._smooth3(v, kern(v)),
    }


@pytest.mark.parametrize("name", list(_large_plane_cases()))
def test_depth_only_blurs_match_cpu_on_large_planes(name):
    # Every one pads depth alone on a 5-D volume: the pattern the Metal bug needs.
    fn = _large_plane_cases()[name]
    x = torch.rand(24, 300, 300, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(fn(x.to("mps")).cpu(), fn(x), rtol=1e-4, atol=1e-4)
