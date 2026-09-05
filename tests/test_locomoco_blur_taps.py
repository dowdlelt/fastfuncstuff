"""Short conv2d kernels are zero-padded past oneDNN's algorithm cliff.

Below ~17 taps oneDNN picks a conv2d algorithm that is far slower on CPU: at
the (268, 76, 76) shape a locomoco slice-movie blur has, 15 taps measured
52.7 ms against 7.9 ms for 17 -- more arithmetic, less time. Padding a short
kernel with zeros cannot change the result, and these tests hold that line.
"""

from __future__ import annotations

import pytest
import torch

from fastfuncstuff.processing.locomoco import (
    _CPU_CONV2D_MIN_TAPS,
    _blur2d,
    _blur_inplane,
    _gaussian_blur3d,
    _gaussian_kernel1d,
    _pad_kernel_for_cpu_conv2d,
)

CPU = torch.device("cpu")


@pytest.mark.parametrize("sigma", [0.5, 1.0, 1.5, 2.0])
def test_short_kernels_are_padded_to_the_threshold(sigma):
    k = _gaussian_kernel1d(sigma, CPU, torch.float32)
    padded = _pad_kernel_for_cpu_conv2d(k)
    assert k.numel() < _CPU_CONV2D_MIN_TAPS
    assert padded.numel() == _CPU_CONV2D_MIN_TAPS
    # Padding must be zeros and must keep the kernel centred, or the blur shifts.
    pad = (padded.numel() - k.numel()) // 2
    assert torch.equal(padded[pad : pad + k.numel()], k)
    assert float(padded[:pad].abs().sum()) == 0.0
    assert float(padded[padded.numel() - pad :].abs().sum()) == 0.0


def test_long_kernels_are_left_alone():
    k = _gaussian_kernel1d(6.0, CPU, torch.float32)
    assert k.numel() >= _CPU_CONV2D_MIN_TAPS
    assert _pad_kernel_for_cpu_conv2d(k) is k


def test_padding_does_not_move_the_blur():
    """float64 isolates the answer from float32 accumulation order.

    In float32 the padded and unpadded forms differ by ~3e-07 relative, because
    the two oneDNN algorithms accumulate in different orders -- not because they
    compute different things, which is what float64 shows.
    """
    torch.manual_seed(0)
    img = torch.randn(8, 40, 40, dtype=torch.float64)
    for sigma in (0.5, 1.0, 2.0):
        k = _gaussian_kernel1d(sigma, CPU, torch.float64)
        assert k.numel() < _CPU_CONV2D_MIN_TAPS  # the case the padding applies to
        blurred = _blur2d(img, sigma)
        # Reference: the same blur with the unpadded kernel, done by hand.
        import torch.nn.functional as F

        r = (k.numel() - 1) // 2
        x = F.pad(img.unsqueeze(1), (r, r, r, r), mode="replicate")
        x = F.conv2d(x, k.view(1, 1, -1, 1))
        expected = F.conv2d(x, k.view(1, 1, 1, -1)).squeeze(1)
        assert torch.allclose(blurred, expected, rtol=0, atol=1e-12)


def test_a_uniform_image_survives_the_wider_replicate_pad():
    """A wider radius pulls in more replicate-padded border; a constant image
    must still come back constant, or the padding is being mishandled."""
    img = torch.full((4, 12, 12), 3.5)
    assert torch.allclose(_blur2d(img, 1.0), img, atol=1e-5)
    assert torch.allclose(_blur_inplane(img, 1.0), img, atol=1e-5)


def test_blur_inplane_is_padded_too():
    torch.manual_seed(0)
    stack = torch.randn(3, 32, 32, dtype=torch.float64)
    import torch.nn.functional as F

    k = _gaussian_kernel1d(1.0, CPU, torch.float64)
    r = (k.numel() - 1) // 2
    x = F.pad(stack[:, None], (0, 0, r, r), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, -1, 1))
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    assert torch.allclose(_blur_inplane(stack, 1.0), x[:, 0], rtol=0, atol=1e-12)


def test_the_3d_blur_is_deliberately_not_padded():
    """Plain conv3d has no cliff -- it scales monotonically, and padding a 7-tap
    kernel to 17 measured 4x SLOWER there. Guards against 'fixing' it too."""
    torch.manual_seed(0)
    vol = torch.randn(8, 12, 12, dtype=torch.float64)
    import torch.nn.functional as F

    k = _gaussian_kernel1d(1.0, CPU, torch.float64)
    r = (k.numel() - 1) // 2
    x = vol[None, None]
    for pad, view in (
        ((0, 0, 0, 0, r, r), (1, 1, -1, 1, 1)),
        ((0, 0, r, r, 0, 0), (1, 1, 1, -1, 1)),
        ((r, r, 0, 0, 0, 0), (1, 1, 1, 1, -1)),
    ):
        x = F.conv3d(F.pad(x, pad, mode="replicate"), k.view(*view))
    assert torch.equal(_gaussian_blur3d(vol, 1.0), x[0, 0])
