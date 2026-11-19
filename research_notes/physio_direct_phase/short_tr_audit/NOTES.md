# Short-TR cardiac rejection and magnitude respiration audit — 2026-11-19

Branch: `research/physio-short-tr-audit`, starting at `6e9490b8`.
Production detection is unchanged. All new experiments used CPU, two threads.
Recordings were opened only after image-only candidates/variants were made.

## Cardiac: the short-TR candidates are correct; the judge fails even on the recording

Recomputed constant-modulus starts, respiratory projection and slice-alignment diagnostics
from raw ds004645 sub-02 and sub-04. Each run has 2247 volumes, TR 0.227 s,
five excitation groups spanning 0.172 s. Volume Nyquist is 2.203 Hz: the complete
configured cardiac band, 0.6–2 Hz, fits below it. Only one of the three CM starts
has an in-band alias in either run; it is the heartbeat.

| Run | Estimated bpm | Pulse locking | True-timing statistic | Maximum over all 44 derangements | Harmonic / control maximum |
|---|---:|---:|---:|---:|---:|
| sub-02, CM | 68.068 | 0.920 | 0.312 | 1.033 | 0.148 / 0.094 |
| sub-02, recorded phase | — | 1 by construction | 0.268 | 1.032 | 0.060 / 0.094 |
| sub-04, CM | 66.848 | 0.948 | 0.128 | 0.867 | 0.043 / 0.070 |
| sub-04, recorded phase | — | 1 by construction | 0.134 | 0.992 | 0.052 / 0.096 |

Statistic = fundamental alignment + second-harmonic alignment. This is **not merely
insufficient Monte Carlo resolution**: all possible derangements were enumerated.
41/44 and 43/44 scrambled timings score at least as well as the CM clock's true timing;
42/44 and 43/44 do so for the recorded clock. `derangement_tail_fraction` in the JSON
includes the identity in numerator and denominator; it is a diagnostic rank fraction,
**not a calibrated permutation p-value** (derangements alone are not a permutation group).
Additional random draws cannot make these clocks pass.

The test requires common fitted vessel phases across excitations. That spatial phase
alignment is a stronger requirement than a recoverable unaliased temporal heartbeat,
and it fails here even with the true clock. The experiment does not identify whether
spatial delay differences, voxel selection, acquisition timing metadata, or a mixture
causes the failure. Do not label the heartbeat absent or say the acquisition cannot
resolve its frequency. Nor would bypassing only the derangement test rescue sub-04:
its harmonic alignment fails too, as does that of the recorded pulse in both runs.

Next implementation experiment: when the cardiac band is unaliased, confirm temporal
cardiac structure without requiring pooled spatial phase alignment. For example, test
harmonic prediction in withheld data with voxel-specific delays, and calibrate against
noise, volume-locked artefacts and respiratory/task impostors. This is a proposed
experiment, not an implemented or validated replacement. A shorter TR alone does not
establish that any in-band component is cardiac.

## Follow-up review: the ds004645 rejection is the sidecar's slice order

The recorded clock ranked near the *bottom* of the derangements (tail 0.93-0.98), which
is anti-alignment, not absence of structure. Scoring it under all 120 column orders of the
five excitation groups (not only the 44 derangements): sidecar timing ranks 113/120
(sub-02, stat 0.27) and 118/120 (sub-04, 0.13); the time-reversed order ranks 2 and 1
(stat 1.13, 1.15; second harmonic 0.76/0.65 against controls ~0.09). The image is LPS
(dcm2niix, syngo B17 investigational 7T), so the sidecar's interleave runs opposite to the
array's slice axis. The judge is behaving correctly; the "confirm without spatial
alignment" experiment proposed above is not needed for these runs and is dropped.
Not pursued further (one dataset): no slice-order search in production.

## Which other misses remain?

`remaining_misses.json` is extracted from the previous **final production** evaluations
(`/tmp/ffs_physio_direct_phase/cmrun/final_{local,remote}.jsonl`, 190 runs total), not a new
cohort rerun. Do not use the older `cm_subspace/cohort_decisions.tsv` to identify the
final misses; some decisions changed after the last unwrap fix.

Excluding the prefrontal slab there are eight misses:

- Correct candidate, rejected (locking 0.874–0.953): the two ds004645 runs above;
  ds002603 sub-07 sequential, sub-04 sequential, sub-05 TR700.
- Discovery/selection failure (kept locking 0.001–0.065): ds002603 sub-02 sequential,
  sub-08 sequential, and TASKFORCE ses-06 breathhold1. This classification concerns
  the **kept** candidate, and does not prove every discarded candidate was wrong.

The other datasets' current locations were not all retained locally. Their diagnoses
above come from the saved production decisions; only the two short-TR cardiac runs
were re-read and tested against their recorded clocks in this audit.

## Respiration: distinguish short TR from long TR, and rate from identity

Reran all ten local ds002547 runs, TR 2 s, and all four available ds004645 runs,
TR 0.227 s. Estimates are magnitude-only; belts score afterwards.

At TR 2 s, volume Nyquist is **15 breaths/min**. The belts' dominant rates are
**16–23/min**, so breathing actually IS aliased in every individual voxel/group's
volume-rate series. Excitation timing is required to recover it. The current
stitched trace reads roughly **31–36/min**, around the TR comb; a narrow notch does
not reliably fix it. This is consistent with the previously parked failure.

New diagnostic variants: interpolate each excitation's volume-rate series to a common
time grid, then either average, or take PC1 after respiratory bandpass. These remove
the direct interleaving of different excitation gains but cannot restore the lost
above-Nyquist clock. Neither reliably recovers breathing on the ten runs. Occasional
partial improvements are not a replacement algorithm. An alias-aware spatial model
at the actual excitation times is needed for a serious next experiment.

At TR 0.227 s:

| Run | Belt dominant rate, /min | Current estimated median, /min | Current phase locking | Aligned-PC locking |
|---|---:|---:|---:|---:|
| sub-02 | 16 | 14.52 | 0.659 | 0.679 |
| sub-03 | 18 | 16.12 | 0.511 | 0.522 |
| sub-04 | 20 | 18.75 | 0.714 | 0.738 |
| sub-05 | 5 | 6.00 | 0.134 | 0.041 |

Belt rate is a whole-run Welch peak; estimated rate is the median of breath intervals,
so their difference is **not a matched breath-rate error metric**. Locking is more
informative here. This spectral/Hilbert belt referee is inherited from `resp_probe.py`,
not manually reviewed breath peaks. sub-05's dominant belt peak is below the detector's
0.1 Hz (6/min) lower band edge, and its estimate lands on that edge. It needs recording
and signal review before interpreting that as true slow breathing or changing the band.
At short TR the TR comb is outside the respiratory band; notching changes nothing.

These experiments establish that removing aliasing does not guarantee a good global
magnitude respiratory trace: averaging can cancel differing spatial gains/delays, and
motion/task signals can dominate. The simple spatial reweighting did not solve that.

## Reproduce and checks

Runner: `scripts/physio_phase_review/short_tr_audit.py`.
Uses the existing conda Python, `load_nifti`, excitation traces, CM start, voxel storage,
alignment and belt scoring primitives. No new library primitive or CLI flag.

```bash
PHYSIO_PY=/home/logan/miniconda3/envs/py314_ffs/bin/python
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 "$PHYSIO_PY" -m scripts.physio_phase_review.short_tr_audit \
  --mode cardiac --list /tmp/ffs_physio_direct_phase/cmrun/local.list \
  --output /tmp/ffs_short_tr_repeat/cardiac.jsonl
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 "$PHYSIO_PY" -m scripts.physio_phase_review.short_tr_audit \
  --mode respiration --root /mnt/belegost/Data/nii_data/openneuro_physio/ds002547 \
  --output /tmp/ffs_short_tr_repeat/respiration.jsonl
```

Respiratory excitation traces are cached alongside the temporary output, avoiding
repeat NIfTI reads. Outputs append: use a fresh output path for a clean evaluation.
Saved JSONL files contain all diagnostics, not only selected successes.

Verification: both cardiac cases and all 14 respiratory cases completed; Ruff lint
and formatting checks and runner `--help` passed. No GPU work, broad regression suite,
benchmark or production-detector change was required or performed.
