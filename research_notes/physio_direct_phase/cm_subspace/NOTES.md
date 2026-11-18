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
