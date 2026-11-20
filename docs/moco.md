# Motion correction: what was tested, and what held up

`ffs_moco` estimates rigid head motion the way `3dvolreg` does (Gauss-Newton on a
weighted least-squares cost, rotations applied as four 1-D shears), batched over
volumes on the GPU. The ten-run moco benchmark stage takes 11.6 s against 3dvolreg's
169.5 s.

This page records a study of its accuracy. The short version: several real bugs were
found and fixed, and every attempt to beat the heptic/wsinc5 defaults with a cleverer
interpolator failed on real data. The defaults are unchanged; the attempts remain as
**experimental** options.

## What counts as "better"

Motion has no ground truth, so two kinds of evidence were used, and they measure
different things:

1. **Simulated truth: how close are the motion *parameters*?** Real high-resolution
   anatomy (a 0.35 mm T2\*-weighted GRE) moved by known motion (two replayed real
   traces, a 7.6° high-motion trace, and sub-voxel sweeps), sampled to 2.1 mm.
2. **Real data: how clean is the *corrected series*?** On 10 real 2.5 mm runs and one
   real high-motion run (mean FD 0.5 mm), with a common brain mask: detrended tSNR, and
   the variance explained by one common set of 12 motion regressors (lower = less motion
   left in the data).

**Real data decides defaults.** The simulations turned out to reward whatever
matched their own assumptions (below), so they are used for finding bugs and
mechanisms, not for choosing defaults.

## What was fixed (measured, and kept)

| fix | evidence |
|---|---|
| **Rotations in millimetres.** Rotations were built in voxel-index space; on non-cubic voxels (most EPI) that is a rotation plus a small stretch and shear | the "rigid" matrices on a 2.5×2.5×2.75 mm run had singular values 0.9989–1.0011 (3dvolreg's: exactly 1). Benchmark agreement with 3dvolreg doubled: max rotation difference 0.0020 → 0.0009, time-series nRMSD 0.0049 → 0.0025 |
| **Shear plan used Aᵀ for A⁻¹.** Only true for a pure rotation; the millimetre fix exposed it | non-orthogonal plans were 0.2 voxel off in the shift, inside a guard that checked only the linear part. The guard now checks both |
| **`-twopass` ran out of GPU memory** on large runs (also on `main`) | heptic `-twopass` used 1.47x the planned memory and crashed on a 130×130×85×391 run. Measured/predicted is now 0.75–0.92x across all modes |
| **At least two Gauss-Newton steps.** A motion below the convergence threshold stopped after one linearized step | on simulated truth, 0.005–0.015° rotations came back 2–8% too large; fixed |

Together, against `main` on real data: tSNR equal or higher on every run (by up to
0.37), motion-regressor variance within ±0.001 (high-motion run: +0.005, with higher
tSNR). Speed is essentially unchanged:

| | `main` | this version |
|---|---|---|
| CUDA estimation, 76×76×54×212 | 0.35 s | 0.38 s |
| CUDA estimation, 160×160×114×225 | 0.97 s | 0.98 s |
| CPU estimation, 76×76×54×212 | 7.1 s | 7.3 s |
| output resample | unchanged | unchanged |

## What was tried and did not hold up (kept as experimental options)

**Fourier interpolation (`-interp fourier`, `-final fourier`).** Polynomial kernels
smooth by an amount that depends on the fractional shift, which biases sub-voxel
estimates toward the half voxel (±1.6% of a voxel). On simulated truth sampled as an
exactly band-limited image, FFT shears removed that bias and looked 6–10x more
accurate; real data does show the predicted heptic-vs-Fourier difference (r = 0.98
against the predicted sine across 10 runs). But a scanner image is the *magnitude* of
a complex image, which is not band-limited. With the simulation fixed to take the
magnitude after reconstruction, Fourier estimation was no better than heptic overall
(worse for in-plane translation), and Fourier output was the worst kernel. On real
runs Fourier output left 3–4% more motion-locked variance than wsinc5, at matched
smoothness.

**Estimation-only smoothing (`-blur 1.5`, `-slice_lowpass 1`).** On the
complex-magnitude truth this made the motion *parameters* 5–6x more accurate (heptic
0.024 → 0.004 mm mean error, in 2D, 3D and on non-cubic voxels, at 2% and 5% noise),
and held lock on the 7.6° trace without `-twopass` where plain heptic lost it. On real
runs, though, the corrected data kept 1.5–3% more motion-locked variance and about
0.3 lower tSNR than the default full-resolution fit, whichever estimate supplied the
regressors.

The likely reason both fail on real data: the default estimator minimizes the actual
full-resolution residual, including motion-locked effects a rigid model cannot
represent (spin history, motion-dependent distortion, motion within a volume), and
that is what leaves the corrected series cleanest. The simulation contains none of
those effects, so it rewards parameter accuracy alone. The options may still be the
better choice when the parameters themselves are the product (FD, censoring,
regressors), but that is not established on real data.

**Output kernel.** On the complex-magnitude truth heptic, quintic and wsinc5 were
within a few percent of each other (cubic and Fourier worst); on real runs wsinc5 left
the least motion-locked variance. wsinc5 stays the default.

## Why ffs_moco and 3dvolreg differ

Mostly the kernel, and it does not matter downstream. On real runs 3dvolreg's own
`-heptic` and `-Fourier` estimates differ by 0.034–0.049 mm, more than either differs
from `ffs_moco`. On the real-data cleanliness measures, 3dvolreg `-heptic` and
`ffs_moco` are within the same few percent; 3dvolreg's heptic output gives a higher
tSNR because heptic smooths more than wsinc5.

## Limits

- One anatomy source for the simulated truth; complex-magnitude truth from one echo's
  phase, 2D only.
- Real-data measures show how clean the corrected data is, not how accurate the
  parameters are; nothing here measures parameter accuracy on real data.
- The remaining motion-locked variance in real data is dominated by effects outside the
  rigid model; further gains are more likely from within-volume and distortion-aware
  correction than from interpolation.

## Reproducing

[`scripts/moco_truth/`](../scripts/moco_truth/) has the truth generator (`gen_truth.py`:
`-phase` for complex-magnitude sampling, `-battery`, `-large`, `-trace`, `-coilfield`),
the parameter scorer (`score_motion.py`), the output-kernel scorer
(`score_resample.py`) and a figure script.
