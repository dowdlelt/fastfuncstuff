"""Dynamic brain-state modelling: BSDS and friends.

This subpackage ports the Bayesian Switching Dynamical Systems (BSDS) model of
Taghia et al. 2018 (*Nat Commun*), applied by Cai et al. 2024, from its original
MATLAB implementation to a native, GPU-respectable PyTorch model, and wraps it —
alongside optional external backends (osl-dynamics) — behind a common interface.

BSDS is a switching linear dynamical system fit by variational Bayes: an HMM over
K discrete brain states, where each state's observation model is a factor
analysis with automatic relevance determination (ARD) and the latent evolves via
an autoregressive process. It operates on **ROI time series** (D regions, not
voxels) and yields state timecourses, transition probabilities, occupancy and
mean lifetime, and per-state functional connectivity.

Data contract
-------------
A *session* (one run) is a ``(D, N)`` array — ``D`` ROIs by ``N`` timepoints,
matching the MATLAB ``D``-by-``N`` convention. A *dataset* is a ``list`` of such
sessions. For a densely-sampled individual, each run/session is one list element
(the BSDS "subjects" list); the group-level fit concatenates them along time.

See ``../fmri_wiki/concepts/BSDS.md`` for the method and rationale.
"""

from __future__ import annotations

from fastfuncstuff.dynamics.preprocess import (
    Sessions,
    concat_sessions,
    detrend_session,
    preprocess_sessions,
    standardize_session,
)

__all__ = [
    "Sessions",
    "concat_sessions",
    "detrend_session",
    "preprocess_sessions",
    "standardize_session",
]
