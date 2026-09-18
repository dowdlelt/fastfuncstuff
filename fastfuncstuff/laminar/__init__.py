"""
Laminar BOLD generative model (LBR-DCM).

A depth-resolved physiological forward model -- neuronal activity at N cortical
depths, through neurovascular coupling, into a venule + ascending-vein
compartment network at K vascular depths, out as laminar BOLD -- plus the
machinery to invert it.

The point is model-based deveining: recovering laminar *neuronal* activity from
depth-sampled GE-BOLD by inverting physiology, rather than by static spatial
deconvolution or vein masking. The ascending vein carries dHb and CBV changes
unidirectionally toward the pial surface, so every superficial depth's signal
contains everything below it; the leakage is activity- and physiology-dependent,
which is precisely what a fixed PSF cannot capture.

This module is shaped unlike the rest of the toolbox: ~10 depth bins x ~50 time
points against 40-60 coupled ODE states. It is float64 and CPU by default, and
batch-first throughout, because the GPU pays off only across the combinatorics
(models x vascular resolutions x subjects x restarts), never within one fit.

See ``../fmri_wiki/concepts/Laminar generative model.md``.
"""

from fastfuncstuff.laminar.forward import (
    apply_depth_psf,
    baseline_hemodynamics,
    f_ode,
    g_obs,
    neuronal_to_vascular,
)
from fastfuncstuff.laminar.integrate import build_input, integrate
from fastfuncstuff.laminar.params import P0, ModelSpec, zero_params

__all__ = [
    "P0",
    "ModelSpec",
    "apply_depth_psf",
    "baseline_hemodynamics",
    "build_input",
    "f_ode",
    "g_obs",
    "integrate",
    "neuronal_to_vascular",
    "zero_params",
]
