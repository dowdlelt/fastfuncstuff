# Constant-modulus subspace discovery — 2026-11-18

`scripts/physio_phase_review/cm_subspace.py`. Frequency-free seed: the volume-rate
heartbeat c_n = exp(i phi(n TR)) has aliased, HRV-smeared phase (folds at 0 and
Nyquist: sub-03/04 spend ~25% of beats within 0.03 Hz of alias 0, sub-06 40% near
Nyquist) but constant modulus. Whole-brain temporal PCA (Legendre deg 4, voxel z-score),
whiten top K, find orthonormal W (K x 2) minimising E[r^4]/E[r^2]^2 with r = |W^T x|
(2 = Gaussian pair, 1 = constant modulus). 60 random restarts, rank by that ratio.
No band, rate, template or recording used in discovery.

Evaluation only: volume-rate lock |mean exp(i(arg c_n -/+ phi_rec(n TR + s)))|, best of
unique slice times x sign (oracle nuisance; random-pair max given as floor).

| run | baseline | K=20 rank-0 cm ratio / lock | next subspace | random max |
|---|---|---|---|---|
| ds002547 sub-03 | miss | 1.083 / 0.96 | 1.119 / 0.26 | 0.44 |
| ds002547 sub-04 | miss | 1.058 / 0.98 | 1.271 / 0.52 | 0.52 |
| ds002547 sub-06 | miss | 1.126 / 0.93 | 1.302 / 0.48 | 0.66 |
| ds002547 sub-07 | found | 1.027 / 0.99 | 1.211 / 0.05 | 0.38 |
| Rockland A00018030 CAP | rejected | 1.027 / 0.96 | (near-duplicates) | 0.55 |
| Rockland A00027167 CAP | joint false-accept | 1.043 / 0.98 | 1.120 / 0.07 | 0.39 |

K matters: K=40 overfits at nt=120 (Rockland: many ~1.03 non-cardiac subspaces);
K=12 misses sub-06 (cardiac not in top 12 PCs). K=20 worked on all six. Needs a
principled K(nt) rule, not this one-off choice.

NOT done: this is volume-rate (aliased) phase only. Still needed: (1) unalias to an
excitation-time clock (per-slice projection of the CM maps, then existing alias
selection / direct_phase refinement), (2) identity — rank 0 was cardiac in all six but
respiration is also near-CM and could win elsewhere, (3) the existing acceptance
tests on the result, (4) TASKFORCE / short-TR / multiband runs and a fresh cohort.
The `ramp_rate_bpm` field in the JSONs is a broken estimator; ignore it.

## 2026-11-19: cohort validation of the production path (`physio_cm.detect`)

The "NOT done" items above are done in `fastfuncstuff/processing/physio_cm.py` (commit
aff98498) and wired first into `refine_cardiac`. In brief: the start is the CM pair
interpolated to excitation times, with the unwrap lifted around a 60 s local mean step
(the reference is itself unwrapped; the unwrapped-reference bug slipped cycles, so
`*_buggyunwrap.jsonl` cohorts are invalid). The alias is the one maximising the slice
alignment of phi plus that of 2 phi; the fundamental alone could not separate a
heartbeat's aliases once voxel delays scatter it. The phase is used unrefined:
re-tracking dragged correct starts off. K = min(20, max(4, nt // 6)).

Cohorts: image-only through the decision (`scripts/physio_phase_review/cm_detect.py`),
then scored against the recordings. Held-out = 66 runs (ds002603, ds002547, Rockland)
never used for any tuning; local = 105 (the 14 original + 91 TASKFORCE). Good = lock > 0.5.
Per-run decisions are in `cohort_decisions.tsv`.

| cohort | old path | CM, cross-start floor | CM, kept-start floor |
|---|---|---|---|
| held-out 66 | 26 right / 0 wrong | 41 / 0 | **61 / 0** (3 good missed, 2 wrong picks rejected) |
| local 105 | – | 57 / 0 | **101 / 0** (2 good missed, 2 runs with no start) |
| paired (33 dev runs in both) | 19 found | – | 31 found, none lost |

Lock of the finds: median 0.964 on held-out and 0.985 on local, minimum 0.61. Median
30 s window lock is 0.98–0.99.

The selection is the strong part: the chosen alias is right on 64/66 held-out runs. What
lost finds was the floor. It took the maximum over every start's own derangement null,
and starts that are no heartbeat (signal the same in every slice, drift) survive
derangement and set a high floor (up to 1.2 against true statistics of 0.5–0.8). Fix
(7fe612c3): draw the Bonferroni extra 30·(n−1) derangements for the KEPT start, as
the old path did. The table's last column is the offline 30-draw version. The production
run (90 draws for 3 starts) was re-checked on the 8 hardest local runs: all agree. They
include breathhold1, which is a wrong pick that the harmonic-only rule accepts, and the
production rule rejects it. Derangement has little power at short TR with multiband
(G ≈ 9–14 close slice times), so the harmonic test carries more weight there.

Still open: a principled K(nt); respiration winning the CM ranking (not seen yet);
ds001168 fresh cohort not run.
