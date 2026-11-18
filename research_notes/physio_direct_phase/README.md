# Continuous cardiac phase refinement: TASKFORCE pilot

2026-11-18. Branch: `research/physio-direct-phase`. Exploratory research only;
`ffs_physio` and autoproc behaviour are unchanged. NORDIC was outside this experiment.

## Question

Can a continuous phase fitted directly to the voxel observations improve on the
current detector's phase, without a spatial template, pulse-shape template, learned
network, or access to cardiac recordings during estimation?

The previous long-TR investigation found that the recorded phase passed the detector
in 17/18 runs while the estimated phase was much weaker. This pilot starts with the
local TASKFORCE images and reviewed pulse peaks to test the phase-refinement idea
before returning to those difficult acquisitions.

## Method and scope

- Current image-derived candidate phase supplies the alias and initial basin.
- Selected training voxels fit `a*cos(phi(t)) + b*sin(phi(t)) + intercept` at their
  actual excitation times. Every voxel has free coefficients: signs and delays are
  learned from that run. A shared cubic spline represents phase around a carrier.
- Variable projection solves the linear voxel coefficients inside L-BFGS phase
  optimisation. Natural cubic splines use 2, 4, or 8 s knot spacing. Smoothing the
  initial phase without optimising against voxels is an explicit control.
- Training voxels have even x indices; prediction voxels have odd x indices. Each
  pool keeps its own best 1% (at least 10 per excitation), selected using the common
  initial phase. Training series are scaled by their initial residual noise.
- Prediction coefficients are fitted on alternate 20 s blocks and applied to the
  other blocks. Candidate selection uses that image prediction score, including
  the unmodified phase as a candidate. It never uses pulse agreement.
- Reviewed peaks are converted using each run's `info.json` sampling frequency and
  start time. Recordings are loaded only after the image-derived cache exists and
  are never arguments to the phase fitter.
- Evaluation is circular phase locking `abs(mean(exp(i*(phi_est-phi_recorded))))`,
  excluding the first/last 10 s and samples outside recording coverage. A constant
  pulse-to-brain phase offset is therefore absorbed. Harmonic locking uses twice
  the phase difference; it is **not** the detector's harmonic-alignment statistic.

The baseline is the **crossed candidate phase judged by the current detector**,
not a direct comparison of the emitted nuisance columns. The first pilot run uses
the full default ICA search. Other runs reproduce the current mean-start baseline
with `ica=False`; they do not establish equivalence to the default multi-start
selection on every run. Baseline detector status in the results is unchanged by
the new fitting experiment.

This is conditional validation of phase refinement: initialisation and voxel
selection already use the whole run. Adjacent even/odd voxels can also share noise.
It is not an independent test of the complete cardiac detector. All runs are one
participant, TR 1 s; there is no demonstrated generalisation to long TR, new
participants, or new acquisitions.

## Results

Ten valid cardiac starts improve in pulse phase locking. Their median rises from
0.9672 to 0.9734; the median paired improvement is 0.0067. The largest paired gains
are 0.0509 and 0.0597. The wrong-start check stays unrelated to the pulse. Image
prediction selects the 2 s direct fit in all 11 runs, although smoothing alone
has slightly better pulse agreement in three of the ten valid starts.

| Run | Current detector phase | Smooth only, 2 s | Direct fit, 2 s |
|---|---:|---:|---:|
| s01 mvpsA 01 | 0.9869 | 0.9895 | 0.9886 |
| s01 rest 01 | 0.9647 | 0.9690 | 0.9731 |
| s02 mvpsA 02 | 0.9223 | 0.9385 | 0.9731 |
| s03 rest 01 | 0.9145 | 0.9238 | 0.9742 |
| s04 mvpsA 02 | 0.9527 | 0.9572 | 0.9607 |
| s05 mvpsB 01 | 0.9696 | 0.9732 | 0.9732 |
| s06 mvpsB 04 | 0.9154 | 0.9186 | 0.9209 |
| s07 mdtb 07 | 0.9728 | 0.9751 | 0.9737 |
| s08 mvpsB 01 | 0.9799 | 0.9832 | 0.9884 |
| s10 mdtb 01 | 0.9755 | 0.9789 | 0.9785 |
| s02 mvpsA 04, wrong start | 0.0129 | 0.0042 | 0.0060 |

On s02 mvpsA/2, harmonic locking to the recording rises from 0.7458 to 0.9065;
on s03 rest/1 it rises from 0.7165 to 0.9154. These are phase-agreement gains,
not measured changes in the detector's acceptance rate. Every selected fit
converged, and all ten valid-start fitted trajectories stayed within the cardiac
rate band at the sampled times.

An additional [harmonic diagnostic](harmonic_diagnostic.json) applies the existing
second-harmonic alignment statistic and non-harmonic controls to the **selected
prediction voxel pool**, keeping that pool fixed across variants. The harmonic
margin increases after direct fitting in all ten valid cases, while the wrong
candidate remains below its controls (-0.0585 -> -0.0316). The recorded phase has a
strong positive margin (0.6696) even in that wrong-start pool: the cardiac signal
is available there, but refining the wrong oscillator does not recover it.
Smoothing alone often gives a larger harmonic margin than direct fitting despite
less accurate agreement with the pulse. These conditional subset scores are not
full-detector decisions, calibrated significance levels, or a sensitivity estimate.

See [results.json](results.json) for per-run measurements and
[comparison.png](comparison.png) for the comparison. The five pilot runs were
sessions 1 mvpsA/1, 1 rest/1, 2 mvpsA/2, 3 rest/1, and 4 mvpsA/2. Five additional
checks span sessions 5–10; session 2 mvpsA/4 is a deliberately difficult wrong-start
check. These are exploratory checks, not a locked independent validation cohort.

## Findings that constrain the next experiment

1. Image prediction does not identify cardiac physiology. On session 2 mvpsA/4,
   prediction improves while the estimated phase remains unrelated to the pulse.
   Free voxel coefficients absorb a change of alias; a dedicated regression test
   demonstrates that ambiguity. Keep a separate cardiac identity test.
2. Several weaker valid starts improve beyond smoothing alone. This supports
   testing direct phase refinement on the existing long-TR cases, but does not
   establish improved sensitivity or false-detection control.
3. Simple smoothing can be competitive on already excellent starts. Do not claim
   that optimisation is uniformly better than smoothing or choose a method by its
   pulse agreement.
4. Independent crossed tracks can differ by multiples of `2*pi`. Direct spline
   projection of their numerical values invented half-cycle shifts on session 6
   mvpsB/4. The implementation first lifts the demodulated circular phases around
   one common trajectory. The regression test checks invariance to phase origins.
5. Reconstruction conditioning is a hypothesis to measure, not an established
   explanation. Selected valid-start matrices typically have condition numbers
   around 2–3 in this pilot; the wrong-start check is much worse.

Before promotion: test the 18 genuine long-TR cases; fit from training groups only
from initialisation onward; assess the existing slice-time and harmonic evidence
after phase refinement; calibrate any optimised acceptance statistic against a
selection-aware null; and preserve rejection of known impostors. No acceptance
thresholds have been loosened in production.

## Reproduction

From the repository root, using the existing environment:

```bash
PHYSIO_PY=/home/logan/miniconda3/envs/py314_ffs/bin/python
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 "$PHYSIO_PY" -m scripts.physio_phase_review.run_taskforce --output /tmp/ffs_physio_direct_phase --limit 1
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 "$PHYSIO_PY" -m scripts.physio_phase_review.run_taskforce --output /tmp/ffs_physio_direct_phase --mean-only
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 "$PHYSIO_PY" -m scripts.physio_phase_review.run_taskforce --output /tmp/ffs_physio_direct_phase --mean-only --check
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 "$PHYSIO_PY" -m scripts.physio_phase_review.run_taskforce --output /tmp/ffs_physio_direct_phase --mean-only --run 2 mvpsA 4
MPLCONFIGDIR=/tmp/ffs_matplotlib "$PHYSIO_PY" -m scripts.physio_phase_review.summarize --input /tmp/ffs_physio_direct_phase --output research_notes/physio_direct_phase
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PHYSIO_PY" -m scripts.physio_phase_review.diagnose_harmonic --cache /tmp/ffs_physio_direct_phase --output research_notes/physio_direct_phase/harmonic_diagnostic.json
```

Image root: `/mnt/belegost/Data/nii_data/PROJECT_TASKFORCE/subjects/sub-pilot01`.
Reviewed peaks: `/mnt/belegost/Projects/TASKFORCE/derivatives/sub-pilot01/ses-XX/all_physio/`.
Small selected-voxel caches stay in `/tmp/ffs_physio_direct_phase`; no raw scans or
recordings are duplicated in the repository. The JSON `seconds` field measures the
latest invocation, which may reuse a cache; it is not an end-to-end runtime benchmark.

The fitter is a small CPU research prototype. A production implementation would
need the repository's device policy and memory planning rather than adopting this
selected-voxel NumPy path as a general processing primitive.

## Validation status

Focused physio tests passed, and all four research regression tests passed after
the phase-origin fix. Ruff lint/format and ty checks passed. The harmonic diagnostic
ran successfully on all 11 cached cases.

The broader CPU fast tier was **stopped at the user's request** after 34 minutes;
testing and failure triage are deferred until the end of the research work. Its
interrupted summary reports 8793 passed, 117 skipped, 4 xfailed, and 5 failures:

- `test_laminar_cli.py::test_run_jobs_returns_each_roi_its_own_inversions_in_order[2]`
- `test_laminar_cli.py::test_run_jobs_raises_a_worker_failure`
- `test_laminar_cli.py::test_run_jobs_survives_the_default_file_limit`
- `test_simulation_explore.py::test_worker_pool_scores_exactly_like_the_serial_loop`
- `viewer/test_player.py::test_a_canvas_that_never_draws_does_not_stall_playback`

These failures are untriaged; interruption may affect the worker-related results.
The complete local log is `/tmp/ffs_physio_direct_phase/test-fast.log`. The broad
suite is not claimed green, and no further tests were started after the stop request.

## Follow-up: recover the missed cases

The first pilot did not establish missed-run recovery. The follow-up therefore
audits the five rejected TASKFORCE runs in the earlier default-library evaluation
([miss_audit.json](miss_audit.json)) and reruns actual detection decisions.
Four already generated a useful cardiac candidate (pulse locking 0.91–0.97) but
failed the harmonic gate. The fifth, s02 mvpsA/4, had no useful candidate: its
recorded median 96.77/min falls in the respiratory exclusion 97.07 +/- 1.8/min.

Two actual recoveries so far, using the existing acceptance rule:

| Run | Change | Before | After | Pulse locking after |
|---|---|---|---|---:|
| s01 rest/1 | Refine every crossed candidate with the direct phase fit before judging | Not found | Found | 0.9731 |
| s02 mvpsA/4 | Admit respiratory-sideband frequencies; keep the current phase estimator | Not found | Found | 0.9682 |

The second run estimates 97.15/min against 96.77/min recorded. This is a concrete
discovery failure caused by a forbidden frequency, not absent cardiac information.
It does **not** establish that removing respiratory exclusions is a safe default:
earlier long-TR experiments produced false finds when they were removed. Neither
the new fitter's false-find rate nor its shuffled null calibration is established.

Reproduction: `scripts/physio_phase_review/rescue_misses.py`. These are real-data
research experiments, not restarts of the stopped regression suite. Results are in
the [rescues](rescues/) directory; candidate phases remain in the local cache.
The next cases are the remaining TASKFORCE misses, genuine long-TR misses, and
previously observed impostors. Production behaviour is still unchanged.

The remaining three TASKFORCE confirmation failures also pass after direct phase
refinement: s01 mvpsA/7 (locking 0.9777), s09 mdtb/11 (0.9127), and s09 mdtb/5
(0.9423). Thus each of the five previously missed runs has been recovered by one
of the targeted ablations. This is **not** a single-configuration 19/19 evaluation:
four runs retain the respiratory exclusions and use direct phase fitting; one
uses the original phase fitter with those exclusions removed.

The complete fit rerun with deranged slice times rejects s01 rest/1 (alignment
0.1349 vs its initial shuffled floor 0.2298; harmonic 0.0572 vs controls 0.1210).
This is one falsification control, not false-positive-rate calibration.

Rockland A00018030 FLU1 CAP, a previously observed false-find case, is also
rejected with direct fitting plus respiratory exclusions removed. Its selected
candidate is now cardiac (84.82/min vs 85.23/min recorded, locking 0.8541), but
alignment 0.3597 falls below the final multiple-answer shuffle floor 0.3704.
Its harmonic passes (0.2971 vs controls 0.1065). This is improved discovery with
confirmation still unresolved, not a recovered detection.

Baseline provenance matters: [baseline_long_tr.json](baseline_long_tr.json)
records the prior **default-library** statuses. ds002547 sub-02 and sub-07 were
already found (locking 0.8425 and 0.9597); their new results, 0.8894 and 0.9684,
are positive controls and must not be counted as rescued long-TR misses. Their
earlier constant-rate grid experiment had failed, which initially led to a wrong
comparison in the progress report. The actual long-TR miss experiments use sub-08
and sub-09.

### Genuine long-TR recoveries and a rejected variant

- ds002547 sub-08: default-library **not found**, direct phase fitting **found**,
  locking 0.9384 (recorded median 78.52/min, estimate 78.12/min).
- ds002547 sub-09: default-library **not found**, direct fitting plus respiratory
  exclusions removed **found**, locking 0.8610 (55.42/min vs estimate 55.22/min).
- Rockland A00018030 FLU1 CAP: fitting the fundamental and harmonic jointly
  improves locking 0.8541 -> 0.8964 and passes the existing acceptance rule.
- **The joint variant is not trustworthy as a detector.** On Rockland A00027167
  FLU1 CAP it falsely reports 40.29/min against 87.21/min recorded, locking 0.0038.
  Fundamental-only refinement rejects this case. Complete alias enumeration does
  not prevent the joint variant's false acceptance. Joint fitting uses the second
  harmonic during estimation, so its post-fit harmonic confirmation especially
  needs independent observations and selection-aware calibration.
- The full shuffled-timing control is rejected with both fundamental-only and
  joint refinement. One passed falsification control does not establish safety.

### Remaining hard discovery failures

ds002547 sub-04 and sub-06 remain missed after admitting respiratory-sideband
frequencies, extra harmonic-derived ICA starts, complete aliases, and (sub-04)
80 ICA components. Extra starts have not solved these cases. Their recordings
are not fed to any discovery method.

The evaluation-only oracle confirms substantial signal under current preprocessing:

| Run | Recorded-phase alignment / shuffle max | Harmonic / controls | Voxels with recorded-phase R2 > 0.1 |
|---|---|---|---:|
| sub-04 | 0.3602 / 0.2291 | 0.3010 / 0.1208 | 7.95% |
| sub-06 | 0.4043 / 0.2451 | 0.4374 / 0.1062 | 4.33% |

Removing respiratory projection or high-pass filtering does not reveal a missing
oracle signal: it was already retained. The recorded clock draws useful signal
from the images; blind discovery has not found that clock. `oracle_audit.py` and
the corresponding JSON records separate this estimation failure from absent signal.

The fixed `k=-3..3` alias enumeration is mathematically incomplete at long TR:
at TR 2 s, a candidate at 0.8 Hz has a mirrored 1.2 Hz clock at `-0.8 + 4/2`.
Frequency-only dictionary keys can also discard opposite rate-fluctuation paths
with the same mean rate. The research `--full-aliases` option enumerates the
complete band and preserves transformations separately. Added regression tests
are written but deferred, as requested. This corrects search coverage; it has
not yet rescued the remaining hard cases.

The next discovery experiment uses split-time multivoxel spectral evidence in
short windows, rather than depending on the brain mean or ICA to supply a clean
initial trace. It remains experimental and has no acceptance decision of its own.
