"""Conversions between polar and Cartesian MRI complex-data components."""

from __future__ import annotations

import torch
from torch import Tensor


def scale_phase_to_radians(phase: Tensor) -> Tensor:
    """Linearly map the input phase range to the closed interval ``[-pi, pi]``.

    This supports scanner phase encodings such as Siemens' ``-4096..4095``.
    Phase that is already in radians must be passed through unchanged instead.
    """
    phase_min = phase.amin()
    phase_max = phase.amax()
    phase_range = phase_max - phase_min
    if not torch.isfinite(phase_range) or phase_range.item() == 0.0:
        raise ValueError("Phase data must have a finite, non-zero range to be scaled.")
    return ((phase - phase_min) / phase_range) * (2.0 * torch.pi) - torch.pi


def magnitude_phase_to_real_imag(magnitude: Tensor, phase_radians: Tensor) -> tuple[Tensor, Tensor]:
    """Convert magnitude and phase in radians to real and imaginary components."""
    return magnitude * torch.cos(phase_radians), magnitude * torch.sin(phase_radians)


def real_imag_to_magnitude_phase(real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
    """Convert real and imaginary components to magnitude and phase in radians."""
    return torch.hypot(real, imag), torch.atan2(imag, real)
