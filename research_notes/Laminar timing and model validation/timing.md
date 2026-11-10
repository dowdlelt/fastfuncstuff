# TR spaced TENT timing and smoothing validation

## Can TR spaced TENT work with non TR locked onsets?

### Takeaway
Yes: TENT evaluates the continuous piecewise-linear model at every actual sample-to-onset lag. No data upsampling and no rounding of onsets are required. But the experiment must identify the requested coefficients, and fitted TR-spaced coefficients equal the true HRF sampled at knots only if the response is representable by that basis. No method can guarantee that for all onset patterns and arbitrary true HRFs. This is an information and observation-model issue, not evidence that irregular onsets are intrinsically invalid.

### Cited Findings
- AFNI defines TENT(b,c,n) as n cardinal piecewise-linear parameters at spacing (c-b)/(n-1), and defines zero models by deleting first/last columns. This permits arbitrary event times; the coefficients are samples of the *fitted HRF model*. It does not guarantee they equal samples of an arbitrary true response — [AFNI 3dDeconvolve documentation](https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dDeconvolve.html).
- Current FFS evaluates exactly those lag-dependent fractional weights, not a microtime approximation; global sampling clock t0 enters sample n as n*TR+t0 — [implementation](/home/logan/Dropbox/Resources/code/fastfuncstuff/fastfuncstuff/design/matrices.py).
- Independent CPU audit: TR=1.857 s, 3 runs x240 samples, 32 events/run, 3 depth-varying analytic gamma HRFs with different timing/amplitudes, 128 noise draws/depth, iid noise SD .15 in signal units, 9 TR-spaced knots. Truth has a smooth taper to zero at 8TR, eliminating omitted-tail confounding. Designs compared to an independent float64 NumPy half-tent implementation; errors <=3.83e-7 across tested phases — [reproducible script](/home/logan/Dropbox/Resources/code/fastfuncstuff/scripts/laminar_curve_review/tr_knots_audit.py); [numeric outputs](/home/logan/Dropbox/Resources/code/fastfuncstuff/research_notes/Laminar%20timing%20and%20model%20validation/tr_knots_audit.json).
- Every event at one shared off-grid phase .1, .3, .5, or .9, with a zero-start TR-spaced 9-knot window, gives rank8. The aligned locked case gives rank9. Balanced locked/half mixtures give rank9, condition1.31; 95%half/5%locked rank9 condition5.01; uniform phases rank9 condition2.15; phases uniformly spread only .49..51 rank9 **condition98.47**. Two distinct phases do not guarantee adequate conditioning — [numeric outputs](/home/logan/Dropbox/Resources/code/fastfuncstuff/research_notes/Laminar%20timing%20and%20model%20validation/tr_knots_audit.json).
- A two-condition design with condition A locked and B exclusively half-TR has pooled phase diversity, but ranks A=9, B=8, joint17/18. Timing diversity must be evaluated for the actual joint condition design; diversity pooled across conditions cannot supply the missing coefficient direction of B — [audit rank_checks](/home/logan/Dropbox/Resources/code/fastfuncstuff/research_notes/Laminar%20timing%20and%20model%20validation/tr_knots_audit.json).
- The requested `-window 0.9285 15.7845` at TR1.857 has span14.856=8TR, so the default gives 9 knots with precisely TR spacing. If the sample clock is t0=0 and every event has half-TR phase, these coincide with observed post-event lags and give an exact FIR. It is not one alignment that can simultaneously align arbitrary mixed phases. For a common onset phase p relative to t0, the first nonnegative aligned lag is ((1-p) mod1)*TR — [implementation and tests](/home/logan/Dropbox/Resources/code/fastfuncstuff/fastfuncstuff/design/matrices.py); [timing regressions](/home/logan/Dropbox/Resources/code/fastfuncstuff/tests/test_tent_timing.py).
- AFNI executable referee `/opt/mrisoftware/abin/3dDeconvolve` was run with `-nodata 400 1.857 -polort -1 -stim_times ... TENT(...) -x1D_stop`. FFS-vs-AFNI max differences: locked0, phase.3=1.24e-5, alignedhalf=2.57e-7. The independent float64 referee itself differs from AFNI by1.22e-5 at phase.3, so that residual is AFNI event-time/clock precision, not the new FFS arithmetic — [audit afni_referee](/home/logan/Dropbox/Resources/code/fastfuncstuff/research_notes/Laminar%20timing%20and%20model%20validation/tr_knots_audit.json).

### Inferences
- TR spaced means knot-to-knot spacing, not onset quantisation: one can keep TR-spaced lag knots while evaluating every irregular onset exactly.
- The earlier wiki phrase “As soon as two phases are present nothing needs aligning” is too broad. Full rank does not establish low variance or absence of basis bias; .49..51 gives condition98.47.
- For a smooth but non-piecewise-linear true response h, beta_hat is a least-squares projection determined jointly by the actual event train and the basis. It is not generally h(t_k). This distinction matters when laminar depths have different curve shapes.

### Gaps
- This audit is a controlled linear superposition experiment with iid noise; it does not validate every duration, event overlap, temporal filtering, censoring pattern, ARMA parameter, or nonlinear laminar train. The inference/physiology work covers the subsequent model separately.
- No finite set of simulations proves arbitrary non-TR designs identifiable. The actual condition-wise/joint matrix, nuisance, and censoring must be examined.

## Does global smoothing damage the curves and can REML help?

### Takeaway
Global smoothing stabilizes many mixed-phase designs but cannot restore unobserved information or remove basis misspecification. It has a bias–variance tradeoff, and the existing claim that it never damages curves is false in general. Two meanings of REML must be separated: choosing a roughness penalty versus fitting temporal noise covariance.

### Cited Findings
The completed tapered-truth audit has these **RMSE per knot**, in signal units. OLS here uses a rank-aware pseudoinverse, so an apparently good singular-case RMSE does not make the unobserved coefficient direction identifiable. It merely chooses another completion of that direction — [script and numeric outputs](/home/logan/Dropbox/Resources/code/fastfuncstuff/scripts/laminar_curve_review/tr_knots_audit.py).

| onset pattern | rank | OLS | global lambda | per-voxel smoothing REML |
|---|---:|---:|---:|---:|
| locked |9|.0167|.0167|.0166|
| constant .1|8|.0253|.1724|.1753|
| constant .3|8|.0540|.1322|.1312|
| constant .5|8|.0795|.0725|.0775|
| constant .9|8|.0244|.0521|.0522|
| half, aligned TR knots|9|.0166|.0166|.0164|
| mixed locked/half|9|.0258|.0237|.0250|
|95% half|9|.0638|.0604|.0601|
| uniform phases|9|.0419|.0401|.0411|
| phases .49..51|9|.5274|.0705|.0766|

- Uniform phases are well-conditioned, but global mean peak estimates remain 1.053/1.051/1.065 of true sampled peaks at the three depths. 95%half yields 1.086/1.080/1.105. Balanced halves yield1.027/1.025/1.026. Shared smoothing does not remove depth-dependent basis projection — [numeric outputs](/home/logan/Dropbox/Resources/code/fastfuncstuff/research_notes/Laminar%20timing%20and%20model%20validation/tr_knots_audit.json).
- Global selects lambda by leave-one-run-out prediction over signal voxels, then applies one scalar. The overall selection is data-dependent; only **conditional on lambda and the design** is the fitted mapping linear. The docstring now states this and removes the unsupported negligible-selection-bias assertion — [choose_global_lambda](/home/logan/Dropbox/Resources/code/fastfuncstuff/fastfuncstuff/glm/smooth_basis.py).
- `-tent-smooth reml` selects the roughness penalty using a restricted-likelihood criterion. With `-smooth-noise arma`, the implementation first estimates within-run residual correlation, groups ARMA bins, whitens and refits. That moments-based ARMA selection is distinct from a full ARMA REML likelihood search — [implementation](/home/logan/Dropbox/Resources/code/fastfuncstuff/fastfuncstuff/glm/smooth_basis.py).
- AFNI's 3dREMLfit uses REML to estimate ARMA(1,1) noise correlation and fits beta by generalized least squares; it explicitly distinguishes this from a residual-moments estimate — [AFNI 3dREMLfit documentation](https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dREMLfit.html).
- Whitening the rank17/18 two-condition design with an exactly known AR(.5) covariance leaves rank17: invertible whitening cannot manufacture design rank — [audit rank_checks](/home/logan/Dropbox/Resources/code/fastfuncstuff/research_notes/Laminar%20timing%20and%20model%20validation/tr_knots_audit.json).
- The current FFS HRF parser and shared builder support TENTZERO; today's wiki claim that ffs_reml has no zero-pinned model is stale — [parse_hrf_model_args and shared task builder](/home/logan/Dropbox/Resources/code/fastfuncstuff/fastfuncstuff/cli_utils.py).
- Ciuciu et al. explicitly address asynchronous event-related paradigms and formulate the estimate as a compromise between data and temporal physiological prior. Their method does not claim irregular timing alone guarantees identification — [Ciuciu 2003](https://pubmed.ncbi.nlm.nih.gov/14552578/); [author-hosted paper](https://www.lib.upmc.fr/~marrelec/Publis/Ciuciu-2003.pdf).

### Inferences
- GLS/ARMA REML is valid and potentially more efficient for HRF curves when the mean model is appropriate and the design identifies it. It is not intrinsically unsuitable for laminar analysis. It also cannot fix a wrong observation model or missing rank; it changes the weighting of basis approximation error when the truth lies outside the basis.
- A common fixed lambda avoids choosing a separate shrinkage strength solely according to each depth's noise level. It does not preserve arbitrary depth ratios because depths may have different true temporal shapes, and the common estimator maps those shapes differently.
- The claim global never hurts is especially untenable in singular cases: lambda can select a smooth completion of an unobserved lag-zero direction that is very different from the true initial response. Held-out time-series prediction is blind to exactly the same unobserved direction.
- Pinning endpoints is an explicit physiological constraint, useful only when the model truly starts/ends at zero there. TENTzero at a shifted window start TR/2 would incorrectly pin a potentially nonzero response at TR/2. Endpoint constraints cannot be applied mechanically.

### Gaps
- The audit's global lambda is selected from the same simulated data, then held fixed to derive the observation operator; expectation conditional on that selection is not mathematically equal to an independently fixed-lambda experiment. Production-noiseless versus dense-operator agreement establishes implementation parity, while Monte Carlo agreement is an empirical check, not a proof that selection is unbiased.
- Numerical RMSE improvements for this specific truth cannot establish universal ordering of OLS/global/REML. Preliminary untapered-truth numbers were superseded and **must not be cited**; only the saved JSON is final.

## What was fixed and what connects these estimates to laminar inference?

### Takeaway
A real timing precision bug was fixed and committed. The conceptual requirement for TR-spaced mixed-onset laminar analysis is to predict the **same estimated coefficients**, using the same event train and estimator, instead of assuming every coefficient is a literal noiseless HRF sample.

### Cited Findings
- Before this patch, TENT and CSPLIN constructed sample and onset absolute times in float32, then subtracted. At TR1.857 with genuinely aligned half-phase events and window.9285..15.7845, narrow AFNI edge tents were attenuated by up to .00759 at200 scans, .02239 at1000, and **.13456 at4000**. This is numerical error, independent of regularisation and model approximation — [regressions](/home/logan/Dropbox/Resources/code/fastfuncstuff/tests/test_tent_timing.py).
- Commit **edb16fc3**, “Preserve off-grid spline timing precision through long runs”, splits each event+window-start clock into a nearest integer sample anchor and a small phase using host float64, subtracts integer sample indices on the requested device, and evaluates the existing basis in dimensionless knot coordinates. No time series is upsampled, no phase is rounded, and the GPU/MPS float32 basis path stays on-device. Both TENT/CSPLIN use the shared helper. No epsilon snapping of real off-edge events was added — [fixed implementation](/home/logan/Dropbox/Resources/code/fastfuncstuff/fastfuncstuff/design/matrices.py).
- Six regressions cover both basis families, long-run aligned exact-FIR behavior with sample clocks0/.37, and long-run translation invariance for arbitrary phases with sample clock.23. The requested window auto-creates9 knots. Targeted CPU validation: **126 passed in8.93s** across `tests/test_tent_timing.py tests/test_design_matrices_extended.py tests/test_deconvolve_smooth.py tests/test_smooth_basis.py`, `-n 0`; ruff lint/format clean; focused ty clean — [tests](/home/logan/Dropbox/Resources/code/fastfuncstuff/tests/test_tent_timing.py).
- For each audit design, build M=I-N(N'N)^-1N', A=X'MX, P normalized to trace(A), and S=(A+lambda P)^-1 X'M. The production noiseless fixed-lambda fit agrees with dense `S @ noiseless_event_train` to <=**2.84e-6** across all10 cases. It can differ from simply sampling the true curve by RMS .1707 at constant.1, .1293 atconstant.3, .0333 foruniform phases — [numeric outputs](/home/logan/Dropbox/Resources/code/fastfuncstuff/research_notes/Laminar%20timing%20and%20model%20validation/tr_knots_audit.json).

### Inferences
- Keep TR-spaced knots and original weird onset times. Let the physiological model generate its predicted time series at actual acquisition times for the actual event train, then apply the *same* S used to estimate the measured curves. Compare beta_hat to `S y_model(theta)`, not to isolated `h_model(t_k)`. This treats basis projection and regularisation identically on prediction and measurement.
- For nonlinear haemodynamics, generating the full train matters: summing isolated responses is a different assumption. For GLS, S must include the same whitening; for censoring, the same kept rows; for PSC/ROI extraction and nuisance, the same transforms. The curve covariance is S Sigma S'; a diagonal precision per depth generally cannot represent it.
- If rank is deficient, infer only supported combinations or pass uncertainty/prior dependence explicitly. The observation-space approach can still test an identifiable low-dimensional physiological hypothesis, but it cannot claim data identified each coefficient independently.
- Applying the estimator to model predictions preserves the user's TR-spaced output grid. Finer knots may diagnose approximation error, but they are not a prerequisite for irregular-onset TENT to be mathematically valid.

### Gaps
- This audit validates the coefficient observation operator numerically; it does **not** implement a new production train-through-estimator laminar adapter. Current laminar production support and remaining gaps are covered in the inference researcher notes.
- GPU/MPS execution was not tested; runtime reports CUDA unavailable and MPS unavailable. No GPU test was run without user approval.

Reproduction (installed Linux Python; project `.venv` was initially a dangling macOS link):

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-project --no-managed-python python \
  scripts/laminar_curve_review/tr_knots_audit.py \
  --output 'research_notes/Laminar timing and model validation/tr_knots_audit.json'
UV_CACHE_DIR=/tmp/uv-cache uv run --no-project --no-managed-python python -m pytest \
  tests/test_tent_timing.py tests/test_design_matrices_extended.py \
  tests/test_deconvolve_smooth.py tests/test_smooth_basis.py -n 0 -q
ruff check fastfuncstuff/design/matrices.py fastfuncstuff/glm/smooth_basis.py \
  tests/test_tent_timing.py scripts/laminar_curve_review/tr_knots_audit.py
ruff format --check fastfuncstuff/design/matrices.py fastfuncstuff/glm/smooth_basis.py \
  tests/test_tent_timing.py scripts/laminar_curve_review/tr_knots_audit.py
ty check fastfuncstuff/design/matrices.py fastfuncstuff/glm/smooth_basis.py
```

### Repository benchmark gate supplement

The existing **`ffs_benchmark -stages glm_tent` passed** on the actual ds005165 localizer dataset, five runs x312 volumes, 67,082 masked voxels, five conditions, TENT(0,15,11), TR1.5. No upstream stages or GPU work ran. Read-only symlinks to the existing source runs/timings/mask were placed in `/tmp/laminar_glm_tent_benchmark_le7b_m2r`; all generated outputs and caches stayed there. AFNI automatically chose polort4, matching the FFS stage's explicit polort4, and sampled its iresp at the same TR1.5, producing11 knots in both tools — [raw repository benchmark result](/home/logan/Dropbox/Resources/code/fastfuncstuff/research_notes/Laminar%20timing%20and%20model%20validation/glm_tent_benchmark.json).

- Minimum condition temporal median correlation: **0.9999998212**, threshold0.95.
- Minimum condition spatial middle-knot correlation: **0.9999999749**, threshold0.95.
- AFNI26.96s, FFS9.48s, observed2.84x. These are one-run correctness-check timings with concurrent CPU tests; not a new performance claim.
- Source checkout confirmed in benchmark stdout; stage command explicitly included `-device cpu,4`. The report architecture's generic `requested_device` field says`auto` despite the explicit stage command; `ffs_arch_id=cpu` and the invocation confirm CPU execution. That reporting discrepancy was left untouched.

Exact invocation:

```bash
PYTHONPATH=/home/logan/Dropbox/Resources/code/fastfuncstuff \
  FFS_NUM_THREADS=4 OMP_NUM_THREADS=4 UV_CACHE_DIR=/tmp/uv-cache \
  uv run --no-project --no-managed-python python -m fastfuncstuff.cli.benchmark \
  -data-dir /tmp/laminar_glm_tent_benchmark_le7b_m2r \
  -stages glm_tent -device cpu,4 \
  -json /tmp/laminar_glm_tent_benchmark_le7b_m2r/results.json -verb 1
```
