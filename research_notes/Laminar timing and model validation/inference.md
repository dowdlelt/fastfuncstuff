# Laminar curve observation and inference

## Do TR-spaced TENT coefficients represent samples of the physiological response?

### Takeaway
Not in general. TR-spaced TENT works as a regression model with fractional onsets, but a fitted coefficient is generally a design-dependent projection of the continuous response. The scientifically coherent downstream comparison retains the requested TR-spaced knots and passes each model prediction through the same estimator; the current production adapter does not do this.

### Cited Findings
- `DepthCurves` carries times, values, ROI identity and depth coverage; `LaminarData` carries curve arrays, input and row mask. Neither carries the deconvolution design, the mapping from acquired samples to fitted coefficients, nor their covariance. `two_condition_batch` creates one isolated event per condition and the inversion compares the resulting model samples directly to the coefficient values. [Current adapter](/home/logan/Dropbox/Resources/code/fastfuncstuff/fastfuncstuff/laminar/io.py), [current inversion](/home/logan/Dropbox/Resources/code/fastfuncstuff/fastfuncstuff/laminar/inversion.py).
- I ran a new reproducible CPU experiment with production `make_tent_design`, TR=2 s, **17 knots spaced exactly 2 s over 0–32 s**, 95 events, 600 acquired samples, second-degree Legendre nuisance, and seven-depth laminar-model isolated responses. Fine truth was integrated at 0.05 s and sampled at each actual event-to-acquisition lag, independently of the TENT estimator. No acquired data were upsampled. Sigma truth was 3 Hz; only sigma varied in the recovery comparison, over 41 log deviations from -0.5 to 0.5; all other parameters were fixed at their known generating values. [Script](/home/logan/Dropbox/Resources/code/fastfuncstuff/scripts/laminar_curve_review/observation_operator.py), [all numeric results](/home/logan/Dropbox/Resources/code/fastfuncstuff/research_notes/Laminar%20timing%20and%20model%20validation/observation_operator.jsonl).

| Onset phases | rank / 17 | direct sample RMSE at truth, OLS / fixed penalty | sigma fitted naively, OLS / fixed penalty | sigma fitted with same estimator |
|---|---:|---:|---:|---:|
| all on TR | 17 | <1e-15 / 0.0796 | 3.000 / 3.316 Hz | 3.000 / 3.000 Hz |
| all half-TR, unshifted knots | 16 | 0.2153 / 0.1741 | 2.715 / 2.854 Hz | 3.000 / 3.000 Hz |
| balanced 0 and half-TR | 17 | 0.0336 / 0.0711 | 2.926 / 3.154 Hz | 3.000 / 3.000 Hz |
| 95% half-TR, remainder on TR | 17 | 0.1441 / 0.1430 | 2.715 / 2.854 Hz | 3.000 / 3.000 Hz |
| quarter phases | 17 | 0.0558 / 0.0754 | 2.854 / 3.076 Hz | 3.000 / 3.000 Hz |
| arbitrary phases on 0.05 s clock | 17 | 0.0689 / 0.0824 | 2.854 / 3.000 Hz | 3.000 / 3.000 Hz |

- The fixed penalty is diff2 with relative lambda 1, scaled by trace(Gram)/trace(penalty); **it is not the data-selected global rule**, and this experiment makes no claim about global lambda's selected value. Model predictions after the same estimator match the noise-free fitted coefficients to <1.3e-16. The singular case used a Moore–Penrose solution for OLS and a penalized solve for lambda 1; no OLS-invertibility assumption was needed. [Script and results linked above].

### Inferences
- These measurements demonstrate the specific observation-model correction, not full DCM recovery or general statistical calibration. The algebra guarantees exact agreement at the generating parameter when the same forward assumptions are used. Recovering sigma here does not establish that simultaneously freeing all neuronal and vascular parameters will be identifiable at real SNR.
- There is no need to upsample the acquired data. Evaluate the candidate continuous/model response at the actual lags, create its predicted acquired series, and apply the stored fitting operator to that prediction.
- Let W whiten acquired noise, N be nuisance, M the projector removing WN, X the task design and P the roughness penalty. For fixed settings, L = (Xw' M Xw + lambda P)^+ Xw' M W, where Xw=WX; then beta_hat=L y. Compare beta_hat with **L g_theta(actual schedule)**, using the entire multi-condition design jointly. **L Sigma L' is the repeated-sampling covariance for externally fixed settings.** Plugging in a data-selected lambda treats selection as fixed; it is not automatically the true covariance conditional on that selection, because selection also changes the noise distribution. Selection uncertainty requires independent selection, cross-fitting, or appropriate full-procedure resampling. A faster isolated-response approximation uses g_theta=E h_theta; E evaluates and sums the continuous responses at actual onset lags. For rapid nonlinear trains, g_theta must simulate the complete input train rather than assume superposition.
- Off-grid singular directions remain missing information. Applying the same operator avoids pretending that a prior-defined or minimum-norm coefficient is a direct observation; it cannot make that direction data-identified.

### Gaps
- This operator is **not wired into ffs_laminar**. The present curve/PSF tables cannot reconstruct it. A robust handoff needs original sample clock/TR/t0, full event schedule and durations, censoring, preprocessing/scaling convention, full nuisance and joint task design, basis/window/edge constraints, actual penalty and selected lambda, whitening operator, and covariance or sufficient information to derive it. A serialized operator can avoid reimplementing the estimator.
- The experiment deliberately used isolated-response superposition. It does not independently quantify rapid-train nonlinear bias, B recovery or layer-model error rates. Earlier wiki numbers on those questions were not rerun in this subtask.
- These new measurements use 17 knots spaced at TR=2 s over 32 s. They are a different experiment from the earlier wiki's nine knots at TR=1.857 s over roughly 15 s and must not be presented as a direct numerical reproduction of that table.

## Are global smoothing and REML enough to make the laminar likelihood valid?

### Takeaway
No. Shared smoothing can avoid giving each depth a different regularizer, but it still changes the expected coefficient curve and induces covariance. ARMA REML and smoothing-parameter REML solve different problems; neither resolves missing design information or makes TENT coefficients independent samples.

### Cited Findings
- The global rule selects eligible signal voxels using OLS or fold-local REML/GCV held-out R2, then optimizes their median held-out R2 over lambda. Both eligibility and lambda depend on observed data. Its application is linear **conditional on fixed selected settings**; the overall adaptive rule is not an exactly linear mapping. A small global selection effect is plausible when many independent signal voxels contribute, but the implementation alone does not establish it as negligible, especially for a small correlated set of ROI-depth curves. [Global selector](/home/logan/Dropbox/Resources/code/fastfuncstuff/fastfuncstuff/glm/smooth_basis.py:552).
- The inversion uses one log precision per depth, multiplying identity over every kept temporal sample; it has no knot or between-condition covariance input. The default hyperprior mean uses the variance of the whole padded series and its variance is exp(-4). `-hyperprior rows` instead uses kept data. This is reference parity, not a deconvolution uncertainty model. [Inversion](/home/logan/Dropbox/Resources/code/fastfuncstuff/fastfuncstuff/laminar/inversion.py:255), [CLI](/home/logan/Dropbox/Resources/code/fastfuncstuff/fastfuncstuff/cli/laminar.py).
- Even under **white acquired noise**, this experiment produced maximum absolute off-diagonal coefficient-noise correlations of 0.445–0.809 for OLS and 0.818–0.862 for fixed diff2 smoothing. With overlapping responses and nuisance, even the TR-locked OLS coefficients had correlation up to 0.602. Independence is not a consequence of good timing. [Numeric results linked above].
- The current Bayesian parameter averaging combines posterior precisions across K, but its default comparison F is the first K's F, matching the SPM reference behavior. Summing F across K is not a valid independent-data likelihood because the depth grids reuse the same voxels. [Implementation and explicit documentation](/home/logan/Dropbox/Resources/code/fastfuncstuff/fastfuncstuff/laminar/experiment.py:213).
- Faes used a slow event-related design with 1.6 s quartets and mean inter-trial interval six TRs (range five–seven); the fourth-tone modulation timing has a concrete experimental meaning. The implementation's `modulation="reference"` pulse should therefore be reserved for matching that timing, whereas `event` assumes the entire second event is modulated. [Faes 2026 Methods](https://www.nature.com/articles/s41467-026-73540-z), [reference code](https://github.com/lonikefaes/predictive_tones).

### Inferences
- `ffs_reml` estimates acquired-series noise covariance and performs GLS. `-tent-smooth reml` selects the roughness penalty strength by restricted likelihood. Both can be useful; neither is a categorical reason to reject REML for curves. For laminar work, record the actual GLS and smoothing operators and transform physiological predictions by them. GLS under a misspecified coarse basis has a different projection target than OLS, so a more efficient acquired-series fit can move the coefficients farther from literal response samples.
- Different depth SNR can change which depths dominate global lambda selection. A shared scalar is helpful for comparisons but is not proof of unbiased depth amplitudes. Quantify it with synthetic ground truth spanning depth-dependent noise and timing, and preserve or cross-fit selection when estimating its uncertainty.
- Do not compare raw F/probabilities across knot counts, penalty choices or different effective ranks as if the observation likelihood were unchanged. With full covariance, singular coefficient directions must be removed or handled on their supported subspace; adding interpolated knots must not increase information.
- Proper empirical calibration should refit the complete estimator and model selection under a justified null, preserving autocorrelation/run structure and repeating adaptive lambda/eligibility choices. A permutation null is not automatically valid for every arbitrary event reassignment.

### Gaps
- Neither calibrated posterior layer probabilities nor nominal false-positive error rates have been demonstrated by this subtask. A successful reference port and a well-conditioned TENT fit are necessary but insufficient.
- Cross-depth covariance from shared voxels/PSF and uncertainty in the depth PSF also remain outside the current likelihood.

## What adapter behavior was checked and fixed?

### Takeaway
The existing timing adapter checks pass, but I found and fixed an unequal-window reporting bug. The physiological model was fit on the correct row mask; some reported peak quantities then used the first condition's length for both conditions.

### Cited Findings
- `two_condition_batch` supports different condition-window lengths, but `_condition_slices(meta)` assumed equal block lengths. I changed its production callers to derive slices from the fitted row mask, reusing existing `condition_blocks`. The measured/deveined profile, fitted profile export, and per-depth model-local peak now include the second condition's full window. [Patch](/home/logan/Dropbox/Resources/code/fastfuncstuff/fastfuncstuff/laminar/pipeline.py), [regression](/home/logan/Dropbox/Resources/code/fastfuncstuff/tests/test_laminar_plots.py).
- The regression places the test condition's actual peak beyond the shorter baseline window and checks both `profile_pair` and exported `measured_pct`/`fit_pct`. It would fail before the fix. `tests/test_laminar_plots.py -n 0`: **8 passed in 7.97 s**. [Test file linked above].
- `tests/test_laminar_io.py -n 0`: **17 passed in 29.24 s**. That exercises zero/offset start times, uniform spacing, arbitrary 0.67 s spacing, gap duration, different event lengths/windows, modulation timing and PSF mixing. [Tests](/home/logan/Dropbox/Resources/code/fastfuncstuff/tests/test_laminar_io.py).
- Focused ruff check passes for pipeline.py, test_laminar_plots.py and the new observation_operator.py. Full fast-tier testing is coordinated by the parent agent.

### Inferences
- This patch fixes a real output error but does not change fitting, solve estimator mismatch, or calibrate evidence.

### Gaps
- No GPU tests were run. No source-curve files or scientific conclusions were automatically regenerated.

Commands used:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-project --no-managed-python /home/logan/miniconda3/envs/py314_ffs/bin/python -m pytest tests/test_laminar_plots.py -n 0
UV_CACHE_DIR=/tmp/uv-cache uv run --no-project --no-managed-python /home/logan/miniconda3/envs/py314_ffs/bin/python -m pytest tests/test_laminar_io.py -n 0
PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache uv run --no-project --no-managed-python /home/logan/miniconda3/envs/py314_ffs/bin/python scripts/laminar_curve_review/observation_operator.py
UV_CACHE_DIR=/tmp/uv-cache uv run --no-project --no-managed-python /home/logan/miniconda3/envs/py314_ffs/bin/python -m ruff check fastfuncstuff/laminar/pipeline.py tests/test_laminar_plots.py scripts/laminar_curve_review/observation_operator.py
```
