# Motion correction: how accurate is `ffs_moco`?

`ffs_moco` estimates rigid head motion the way `3dvolreg` does (Gauss-Newton on a
weighted least-squares cost, rotations applied as four 1-D shears), batched over
volumes on the GPU. It is fast: the ten-run moco benchmark stage takes 11.4 s
against 3dvolreg's 169.5 s. Speed is only worth having if the motion is right, and
motion has no ground truth: every motion trace is an estimate. This page builds a
ground truth, uses it to answer three questions, and records what changed because
of the answers.

1. **Can it miss sub-voxel motion?** No. But a polynomial interpolator biases
   sub-voxel estimates in a smooth, predictable way, and that bias shows up in real
   data too.
2. **Can it do better?** Yes. Fourier shears in-plane, plus a one-slice low-pass
   along the slice axis for estimation, cut the error about 8x. That is now the
   default for both estimation and the output resample.
3. **Why does it disagree with 3dvolreg?** Mostly the kernel. 3dvolreg's own
   `-heptic` and `-Fourier` runs disagree with each other more than either disagrees
   with `ffs_moco`. One real ffs bug turned up along the way (rotations on
   non-cubic voxels) and is fixed.

## At a glance

Error is the displacement-field error over the brain in mm, mean / max, at 2% noise.
The truth is real 0.35 mm anatomy moved by known motion and sampled to 2.1 mm
(below).

| | 2D, cubic voxels | 3D, cubic voxels | 2D, 2.1×2.1×2.45 mm |
|---|---|---|---|
| **`ffs_moco` (default: `-interp fourier`)** | **0.003 / 0.010** | **0.003 / 0.009** | **0.003 / 0.011** |
| `ffs_moco -interp heptic` | 0.028 / 0.075 | 0.030 / 0.100 | 0.031 / 0.099 |
| `3dvolreg -heptic` | 0.026 / 0.078 | 0.026 / 0.089 | 0.029 / 0.093 |
| `3dvolreg -Fourier` | 0.036 / 0.144 | 0.003 / 0.013 | 0.050 / 0.194 |

(Random 6-DOF motion, σ = 0.4 voxel and 0.4°. Small motion, σ = 0.1, and real
motion traces give the same picture; see the tables below.)

| | result |
|---|---|
| Sub-voxel bias | heptic: a ±1.6% of a voxel sinusoid over each voxel, small motions read ~10% large. Fourier: flat within ±0.2% |
| Same bias in real data | heptic minus Fourier follows sin(2π·position) at r = 0.98 on 10 real runs, amplitude 1.4% of a voxel |
| Output resample (undoing the true motion) | Fourier injects 0.47% spurious temporal std, wsinc5 and heptic 0.66%; less error outside the brain too (no added ringing) |
| Non-cubic voxels | before the fix, the old default was off by up to 0.22 mm on real motion; now 0.009 mm |
| Cost of Fourier | CUDA +0.03 s (small run) / +0.2 s (large run) of estimation; CPU slightly faster than heptic |

## Building a ground truth: real anatomy, synthetic motion

A simulation is only as good as its realism, and testing an interpolator on data
made with the same interpolator flatters it (the "inverse crime"). So the truth is
built from real anatomy, moved more accurately than any estimator could, and sampled
the way a scanner samples a moving head.

**Anatomy.** A T2\*-weighted multi-echo GRE echo mean at **0.35 mm isotropic**,
whole brain (572×572×380), and for replication a 1.25 mm whole-brain 3D EPI
(echo- and time-averaged after motion correction). Multi-echo series are used only
as mean anatomy, never as test series.

**Moving it exactly.** A 3D GRE image is a discrete Fourier reconstruction on all
three axes, so it is a trigonometric polynomial, and FFT phase ramps move it
exactly: a translation, then each rotation as three shears (Paeth). This is done at
the native 0.35 mm on the GPU, in code written independently of `ffs_moco`. It is
checked where it matters: at the full 0.35 mm grid the rotation is not
self-consistent (applying it in two halves differs by 0.7%), because white noise
reaches Nyquist and the middle shear aliases it. That aliasing lands near Nyquist,
which the reduction to 2.1 mm throws away. At the 2.1 mm band, halves-vs-once agree
to 1.5e-6 and a rotation followed by its inverse returns the original to 2.2e-6.

**Sampling it like an EPI.** k-space truncation to 2.1 mm in-plane (exactly what a
DFT reconstruction does); along z, either a box slice profile six fine slices wide
and centred on each sample (2D multislice) or truncation (3D). A non-cubic version
uses 2.45 mm slices. Gaussian noise at 0, 2% and 5% of the brain mean.

**The motion.**
- *Real motion traces:* two `ffs_moco` traces replayed onto the anatomy, a 2.5 mm
  3T resting-state run (212 volumes, up to 0.92°, ~0.7 mm) and a 1.25 mm 7T run
  (225 volumes, up to 0.38°).
- *A sub-voxel battery:* translations from 0.025 to 1 voxel in 0.025 steps along
  each axis, rotations from 0.005° to 0.5° about each axis (log spaced), and random
  6-DOF motion at three scales: jitter (σ 0.02 voxel / 0.02°), small (σ 0.1) and
  moderate (σ 0.4).

**Scoring.** Each estimate is compared with the truth as the displacement field it
implies over the brain, in mm. That is independent of how a tool parametrizes or
pivots its rotations, which is where most "disagreements" between tools live.

The generator and scorers are in [`scripts/moco_truth/`](../scripts/moco_truth/).

## Question 1: sub-voxel motion and interpolation bias

Sub-voxel motion is never missed. Across the dense translation sweep there is no
staircase and no dead zone. But with a polynomial kernel the error is a smooth
sinusoid with a period of one voxel: zero at 0, ½ and 1 voxel, peaking at ±1.6% of a
voxel near ¼ and ¾, always pulling the estimate toward the half voxel. Near zero
that bias is proportional to the motion, so **small motions read about 10% too
large**, which inflates FD.

The reason: a polynomial interpolator smooths the image by an amount that depends
on the fractional shift (none at a whole voxel, most at half a voxel). Gauss-Newton
trades alignment against that changing smoothness and lands slightly off.

Real data shows the same thing. On ten real 2.5 mm runs, the difference between
heptic and Fourier estimates follows sin(2π × fractional position) at r = 0.98
(amplitude 1.4% of a voxel along one in-plane axis, 1.1% along the other), with
small motions 7-9% larger under heptic. That is the predicted curve, measured
without any simulation.

| translation sweep, 2D, noise-free | error, % of a voxel |
|---|---|
| heptic, in-plane | sinusoid, ±1.6 (x), ±1.4 (y) |
| heptic, slice axis | −0.6 at most |
| **fourier, every axis** | **within ±0.2** |

## Question 2: doing better

**In-plane: Fourier shears.** An EPI row is a DFT reconstruction, so a phase ramp
shifts it exactly. Each of the four shears is a 1-D row shift, so each in-plane step
becomes an FFT, a ramp and an inverse FFT. Rows are treated as periodic and are not
padded: the image is periodic in-plane anyway (EPI wrap-around is periodic), and
zero-padding breaks that at the edges. 3dvolreg's `-Fourier` pads, which is why its
slice-axis errors are so large on 2D and slab data in the tables.

**The slice axis: a polynomial, plus a one-slice low-pass.** A 2D slice is a
discrete excitation through a slice profile, not a Fourier sample, so a phase ramp
along it is wrong (it tripled the slice-axis error). The slice axis keeps heptic.
Once in-plane is exact, though, heptic's slice-axis bias is what is left, and it
showed up as a 2-4% gain error on pitch and roll (with heptic in-plane the two
biases had partly cancelled). A Gaussian of one slice along z on both images, for
estimation only, removes it. The result is flat for σ between 0.8 and 1.3 slices at
2% and 5% noise; the corrected output is never blurred.

**3D EPI.** The slab axis is Fourier-encoded too, and `-interp fourier3d` shifts it
by FFT. In practice the default `fourier` is just as good on whole-brain 3D data and
needs no knowledge of how the data were acquired; `fourier3d` wins only on a thin 3D
slab (17 slices: 0.005 / 0.021 against 0.010 / 0.038 mm).

**At least two Gauss-Newton steps.** A motion below the convergence threshold (0.01
voxel / 0.02°, 3dvolreg's values) made the first step read as converged, leaving the
one-step linearization's gain error: rotations of 0.005-0.015° came back 2-8% large,
for every kernel. Requiring a second step fixes it at no measurable cost. For mixed
sub-threshold jitter the thresholds themselves are the limit; `-workhard` halves that
error (0.0028 → 0.0014 mm at 2% noise) for about a third more CPU time.

Estimation error, mean / max mm, 2% noise:

| motion | heptic | **fourier** | 3dvolreg `-heptic` | 3dvolreg `-Fourier` |
|---|---|---|---|---|
| 2D cubic, jitter | 0.005 / 0.018 | **0.003 / 0.006** | 0.004 / 0.020 | 0.006 / 0.027 |
| 2D cubic, small | 0.024 / 0.058 | **0.003 / 0.008** | 0.022 / 0.054 | 0.023 / 0.097 |
| 2D cubic, moderate | 0.028 / 0.075 | **0.003 / 0.010** | 0.026 / 0.078 | 0.036 / 0.144 |
| 2D cubic, real trace (3T) | 0.023 / 0.054 | **0.004 / 0.010** | 0.019 / 0.045 | 0.026 / 0.094 |
| 2D cubic, real trace (7T) | 0.019 / 0.040 | **0.003 / 0.009** | 0.017 / 0.034 | 0.030 / 0.099 |
| 3D cubic, moderate | 0.030 / 0.100 | **0.003 / 0.009** | 0.026 / 0.089 | 0.003 / 0.013 |
| 2D 2.1×2.1×2.45, moderate | 0.031 / 0.099 | **0.003 / 0.011** | 0.029 / 0.093 | 0.050 / 0.194 |
| 2D 2.1×2.1×2.45, real trace (3T) | 0.023 / 0.057 | **0.003 / 0.009** | 0.020 / 0.056 | 0.037 / 0.143 |

At 5% noise `fourier` stays at 0.004-0.005 mm mean (max 0.013-0.020).

## A bug the truth found: rotations on non-cubic voxels

Replaying a real trace exposed it: every estimator, 3dvolreg included, sat at the
same error floor on one trace, so the truth itself was suspect. That trace's
"rigid" matrices were not rigid (singular values 0.9989 to 1.0011). `ffs_moco` built
its rotations in voxel-index space, and on 2.5×2.5×2.75 mm voxels a rotation in
index space is a rotation plus a small stretch and shear in millimetres, about
angle × anisotropy. A head turns rigidly in millimetres; 3dvolreg's matrices are
exactly rigid.

Rotations are now built in millimetres and mapped into index space by the voxel
sizes, as 3dvolreg does. That exposed a second, latent bug: the shear factorization
took the transpose of the pull matrix as its inverse, which is only true for a pure
rotation. Its plans were 0.2 voxel off in the shift, and the safety check, which
looked only at the linear part, let them through; it now checks the shift too.

On the 2.1×2.1×2.45 mm truth with a real 3T trace, the old default (heptic,
index-space rotations) was off by 0.047 / 0.222 mm. Fixed heptic gives 0.023 / 0.057,
matching 3dvolreg's 0.020 / 0.056, and `fourier` gives 0.003 / 0.009. The fix also
halved the moco benchmark's disagreement with 3dvolreg (maximum rotation difference
0.0020 → 0.0009; time-series nRMSD 0.0049 → 0.0025).

## The output resample, and ringing

The same argument applies to the final resample, where fraction-dependent smoothing
adds spurious, motion-locked variance to the corrected time series. To measure only
the kernel, the *true* motion was undone with each 1-D kernel through the same shear
plan, on noise-free data:

| 2D, real 3T trace | brain residual | spurious temporal std (median voxel) | error just outside the brain |
|---|---|---|---|
| heptic | 3.36% | 0.66% | 2.57% |
| wsinc5 (the previous `-final` default) | 3.14% | 0.66% | 2.41% |
| **fourier** | **2.74%** | **0.47%** | **2.23%** |

(% of the brain mean.) On the 7T trace the injected std is 0.61 / 0.56 / **0.35%**;
for tiny in-plane shifts Fourier injects 4x less. On 3D data `fourier3d` halves it
again (0.22%).

The residual is not zero for any kernel. A 2D slice profile cannot be un-moved
exactly, and rotation moves k-space corners in and out of the acquired band. The
comparison between kernels is what matters.

**Ringing.** A Fourier shift of a band-limited image adds no ringing; it carries the
image's own Gibbs pattern along. The shell outside the brain, where added ringing
would show, has the least error with Fourier. The one trade-off is at the edge of the
field of view: rows wrap around rather than zero-fill. The RMS error in the outer
four voxels is still lower (0.75% against 0.94% for wsinc5), but on a tight
field-of-view dataset a few wrapped edge voxels outside the head reached 39% of the
brain mean, against 32% for wsinc5.

## Why `ffs_moco` and 3dvolreg disagree

Mostly the kernel. On real 2.5 mm runs `ffs_moco` heptic is closest to 3dvolreg
`-heptic` (0.016-0.029 mm apart), `ffs_moco` Fourier is closest to 3dvolreg
`-Fourier` (0.020-0.026 mm), and **3dvolreg's own two kernels are 0.034-0.049 mm
apart**, more than either cross-tool gap. The rest is 3dvolreg's zero-padded Fourier
along the slice axis. On the truth, 3dvolreg `-Fourier` is as good as `ffs_moco`
in-plane and on whole-brain 3D data, and 0.03-0.15 mm off along z on 2D and slab
data.

The benchmark (`ffs_benchmark -stages moco`) still compares `ffs_moco -interp heptic
-final heptic` with `3dvolreg -heptic`, as a parity check against the reference. It
is not an accuracy test; this page is.

## Speed

Estimation and output resample, seconds (76×76×54×212 and 160×160×114×225):

| | estimation, heptic | **estimation, fourier** | output, wsinc5 | **output, fourier** |
|---|---|---|---|---|
| CUDA, 2.5 mm run | 0.37 | **0.40** | 0.22 | **0.22** |
| CUDA, 1.25 mm run | 0.98 | **1.18** | 1.26-1.31 | **1.20-1.29** |
| CPU, 2.5 mm run (whole correction) | 9.9 | **9.6** | 3.1 | **≤ 2.5** |

On the GPU the whole batch shares one shear axis order where it is well conditioned
(volumes it would condition badly form a second group with their own order), so each
step is one batched FFT or one Triton pass; the phase ramp is a single fused kernel.
Load and save dominate wall time for large runs either way.

## Options

| flag | default | what it does |
|---|---|---|
| `-interp` | `fourier` | estimation kernel. `fourier`: FFT in-plane, heptic along the slice axis, with the slice low-pass. `fourier3d`: FFT on all three axes (thin 3D slabs). Polynomials: `heptic`, `quintic`, `cubic`, `linear`, `wsinc5` |
| `-final` | `fourier` | output kernel; same choices |
| `-slice_lowpass SIGMA` | 1.0 with `fourier`, else 0 | estimation-only Gaussian along the slice axis, in slices |
| `-workhard` | off | 5x stricter convergence and twice the iterations; halves sub-threshold jitter error |

Paths without the shear estimator (`-cost lpa` / `quad`, `-no_shear`, `-chain_init`
on CUDA) and the space-time (`-tpattern`) output warp have no Fourier kernel; they
use heptic and print a note saying so.

## Limits

- **Band-limited data assumed.** True of DFT reconstruction, zero-filled partial
  Fourier and k-space filters. Not strictly true after image-space resampling before
  moco (scanner online distortion correction, regridding) or after denoising;
  untested how much of the gain survives there.
- **What the truth leaves out.** Spin history, motion-dependent B0 distortion and
  motion within a volume are not simulated: this tests registration accuracy, not
  every way motion corrupts data.
- **Texture to the slab edges.** On a synthetic volume filled with texture right up
  to the ends of a 24-slice slab, the shear-based estimators mis-estimated pitch on
  CPU (5% at 1°) while a 3-D gather did not. Realistic head data showed no such
  effect and CPU matched CUDA exactly; recorded as a caveat.
- **Apple Silicon** (MPS) runs the Fourier path but has not been checked against
  these numbers.
