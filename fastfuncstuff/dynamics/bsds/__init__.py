"""Native BSDS: switching factor-analysis + ARD + AR, fit by variational Bayes.

Ported from the MATLAB of Taghia et al. 2018 (*Nat Commun*) — reimplemented from
the model equations rather than translated, and validated against the reference
outputs. See ``../fmri_wiki/concepts/BSDS.md`` for the full math spec.

Model (per latent state ``s``, over ROI observations ``y_t`` in R^D):

- **Augmented factor analysis.** The latent ``x_t`` in R^(k+1) has its first
  coordinate pinned to 1. The loading matrix ``L_s`` in R^(D x (k+1)) then folds
  the state *mean* into column 0 and the factor *loadings* ``Lambda_s`` into
  columns 1..k. Observation model: ``y_t = L_s x_t + noise``, diagonal noise
  precision ``psi`` (shared across states). The state covariance reported as
  dynamic FC is ``Lambda_s Lambda_s^T + diag(1/psi)``.
- **ARD.** A Gamma posterior ``(a, b_s)`` on each loading column's precision
  prunes unneeded factors per state; a Normal hyperprior ``(mean_mcl, nu_mcl)``
  pools the state means across states.
- **AR latent dynamics.** A VB vector-autoregression fit to the per-state latent
  trajectories yields a transition matrix ``B_s`` per state.
- **HMM switching.** A first-order Markov chain over states with Dirichlet
  posteriors ``(Wa, Wpi)`` on transitions/initial, coupled to the observation
  model through the responsibilities ``q(z)``.

Inference is mean-field variational Bayes: cyclic coordinate ascent over the
factors, an HMM forward-backward E-step, and an ELBO for convergence and
restart selection.
"""

from __future__ import annotations
