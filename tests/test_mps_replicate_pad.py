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
