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

from fastfuncstuff.laminar.experiment import (
    bayesian_parameter_average,
    faes_priors,
    fit_model_space,
    layer_model_names,
    layer_model_targets,
    posterior_model_probabilities,
)
from fastfuncstuff.laminar.forward import (
    apply_depth_psf,
    baseline_hemodynamics,
    f_ode,
    g_obs,
    neuronal_to_vascular,
)
from fastfuncstuff.laminar.integrate import build_input, integrate
from fastfuncstuff.laminar.inversion import Priors, VLResult, variational_laplace
from fastfuncstuff.laminar.layers import (
    depth_bin_centres,
    estimate_depth_psf,
    label_mean,
    voxels_to_layers,
)
from fastfuncstuff.laminar.params import P0, ModelSpec, zero_params

__all__ = [
    "P0",
    "ModelSpec",
    "Priors",
    "VLResult",
    "bayesian_parameter_average",
    "depth_bin_centres",
    "estimate_depth_psf",
    "faes_priors",
    "fit_model_space",
    "label_mean",
    "layer_model_names",
    "layer_model_targets",
    "posterior_model_probabilities",
    "variational_laplace",
    "voxels_to_layers",
    "apply_depth_psf",
    "baseline_hemodynamics",
    "build_input",
    "f_ode",
    "g_obs",
    "integrate",
    "neuronal_to_vascular",
    "zero_params",
]
