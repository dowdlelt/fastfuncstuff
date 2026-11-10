# Proposed correction to response curve guidance

Replace the universal recipe and broad safety claims in `Response curves for the laminar model.md` with the following text. Preserve its older numerical tables as measurements of their particular designs, not general guarantees.

## TR-spaced TENT supports fractional onsets

TENT evaluates its basis at the actual acquisition-minus-onset lags. Non-TR-locked events do not require data upsampling or finer-than-TR knots. TR spacing can remain the analysis target. Distinguish spacing from phase: a shifted TR-spaced window is an exact FIR at the observed lags only when all events share the corresponding phase. Mixed phases cannot all be aligned by one shift.

Check the actual joint design after nuisance projection. One shared off-grid phase can leave the AFNI half-tent design rank-deficient. Multiple phases can restore rank while remaining ill-conditioned. The statements that two phases automatically solve the problem or that global smoothing never harms physical curve recovery are too strong. A penalty stabilizes estimation by supplying assumptions in weak directions; it does not create observations there.

Global lambda gives the same regularizer to each voxel/depth, but its selection and signal-voxel eligibility depend on the data. Application with fixed selected settings is linear; the complete adaptive procedure is not exactly linear. Regularization can change peak, timing and unobserved edge values. The 2026-11-10 tapered analytic audit found both substantial variance reduction near half-TR and harmful extrapolation for a single phase-0.1 design. Neither the older successful cases nor the new counterexamples establish a universal method ranking.

## Laminar must predict the estimated curve

A fitted TENT coefficient is generally a design-dependent projection of the physical response, rather than its exact value at the knot. The production laminar adapter currently compares it with raw model samples. Preserve TR-spaced knots by applying the same schedule, nuisance projection, whitening and penalty to model predictions before comparison. Carry the resulting covariance and restrict the likelihood to estimable directions. A noise-free one-parameter demonstrator recovered sigma 3 Hz with this matching operator across all tested phase patterns; the direct sample comparison shifted it to 2.715–3.316 Hz. This correction is demonstrated but not yet production-wired or calibrated for full DCM inference.

## REML is usable, with its operator recorded

`ffs_reml` estimates temporal noise for GLS; `-tent-smooth reml` chooses a roughness penalty. Neither repairs missing timing information. Current model parsing supports TENTzero, so the categorical statement that REML lacks that model should be removed. GLS and smoothing change the projection target when a coarse basis misses real curvature. The question is whether the downstream model uses that same estimator and uncertainty, not whether REML is categorically allowed or forbidden.

See `reports/Laminar timing and model validation.md` and its research notes for verified measurements, source links, production fixes `edb16fc3` and `f06340fb`, and remaining implementation requirements.
