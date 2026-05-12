"""Voxelwise diff of ffs_phasereg vs phaseprep_ref outputs, over mask.nii.gz.

Adds range and quartile breakdown for slope, R², and corrected-mag timeseries.
The "ours has lots of zeros that phaseprep doesn't" symptom comes from our
core.py:565 step that zeros the slope wherever obs-R² goes negative —
phaseprep keeps every fitted voxel. So we explicitly count zeros below.
"""

import os
import numpy as np
import nibabel as nb

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    return nb.load(os.path.join(HERE, path)).get_fdata()


# ── load mask + outputs ─────────────────────────────────────────────────────
mask = (
    load("mask.nii.gz").astype(bool)
    if os.path.exists(os.path.join(HERE, "mask.nii.gz"))
    else load("mask.nii").astype(bool)
)
n = int(mask.sum())
print(f"Mask voxels: {n:,}\n")

pp_r2 = load("phaseprep_ref_r2.nii.gz")
pp_slope = load("phaseprep_ref_slope.nii.gz")
pp_inter = load("phaseprep_ref_intercept.nii.gz")
pp_corr = load("phaseprep_ref_corrected.nii.gz")

ffs_r2 = load("test_new_nosgf_nwarp_pol2_r2.nii.gz")
ffs_slope = load("test_new_nosgf_nwarp_pol2_slope.nii.gz")
ffs_corr = load("test_new_nosgf_nwarp_pol2_corrected.nii.gz")
phi = load("test_new_nosgf_nwarp_pol2_phi.nii.gz")


# ── helper: full distribution print ─────────────────────────────────────────
def describe(name, arr, m):
    v = arr[m]
    qs = np.quantile(v, [0.0, 0.05, 0.25, 0.50, 0.75, 0.95, 1.0])
    n_zero = int((v == 0).sum())
    print(f"{name}:")
    print(f"  n={v.size:,}  zeros={n_zero:,} ({100.0 * n_zero / v.size:.2f}%)")
    print(
        f"  min={qs[0]:+.4f}  p05={qs[1]:+.4f}  p25={qs[2]:+.4f}  "
        f"med={qs[3]:+.4f}  p75={qs[4]:+.4f}  p95={qs[5]:+.4f}  max={qs[6]:+.4f}"
    )
    print(f"  mean={v.mean():+.4f}  std={v.std():.4f}")


def compare_pair(label, a, b, m):
    print(f"═══ {label} ═══")
    describe("  phaseprep", a, m)
    describe("  ours     ", b, m)
    # voxelwise stats over mask
    av, bv = a[m], b[m]
    diff = av - bv
    both_nonzero = (av != 0) & (bv != 0)
    if both_nonzero.sum() > 10:
        r = float(np.corrcoef(av[both_nonzero], bv[both_nonzero])[0, 1])
    else:
        r = float("nan")
    print(f"  pearson r (both nonzero, n={int(both_nonzero.sum()):,}): {r:.6f}")
    print(f"  voxels where ours==0 but phaseprep!=0: {int(((bv == 0) & (av != 0)).sum()):,}")
    print(f"  voxels where phaseprep==0 but ours!=0: {int(((av == 0) & (bv != 0)).sum()):,}")
    print(f"  median |a-b|: {np.median(np.abs(diff)):.4f}")
    print()


compare_pair("SLOPE", pp_slope, ffs_slope, mask)
compare_pair("R²", pp_r2, ffs_r2, mask)

# ── ODR-inflation factor (predicts phaseprep R² from ours) ──────────────────
# eps_ODR = r_OLS / (1 + A²/φ),  so R²_ODR = 1 - (1 - R²_obs)/(1+A²/φ)²
A = ffs_slope
infl = 1.0 + (A**2) / np.maximum(phi, 1e-9)
print("═══ ODR INFLATION FACTOR  (1 + A²/φ) ═══")
describe("  factor", infl, mask)
print(
    f"  voxels with factor > 2:  {int((infl[mask] > 2).sum()):,} "
    f"({100 * (infl[mask] > 2).mean():.2f}%)"
)
print(
    f"  voxels with factor > 10: {int((infl[mask] > 10).sum()):,} "
    f"({100 * (infl[mask] > 10).mean():.2f}%)"
)
print()

# ── 4D corrected-mag timeseries comparison ──────────────────────────────────
print("═══ CORRECTED MAG (per-voxel timeseries correlation) ═══")
pp_c = pp_corr.reshape(-1, pp_corr.shape[-1])
ff_c = ffs_corr.reshape(-1, ffs_corr.shape[-1])
m_flat = mask.reshape(-1)
all_idx = np.where(m_flat)[0]
rng = np.random.RandomState(0)
idx = rng.choice(all_idx, size=min(10000, all_idx.size), replace=False)
corrs = np.full(len(idx), np.nan)
rms_diff = np.zeros(len(idx))
for k, i in enumerate(idx):
    a = pp_c[i] - pp_c[i].mean()
    b = ff_c[i] - ff_c[i].mean()
    if a.std() > 0 and b.std() > 0:
        corrs[k] = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))
    rms_diff[k] = float(np.sqrt(((pp_c[i] - ff_c[i]) ** 2).mean()))
ok = ~np.isnan(corrs)
print(f"  sample size: {ok.sum():,} voxels")
qc = np.quantile(corrs[ok], [0.0, 0.05, 0.25, 0.50, 0.75, 0.95, 1.0])
qr = np.quantile(rms_diff, [0.0, 0.05, 0.25, 0.50, 0.75, 0.95, 1.0])
print(
    f"  ts correlation: min={qc[0]:.4f}  p05={qc[1]:.4f}  p25={qc[2]:.4f}  "
    f"med={qc[3]:.4f}  p75={qc[4]:.4f}  p95={qc[5]:.4f}  max={qc[6]:.4f}"
)
print(
    f"  RMS(a-b):       min={qr[0]:.2e}  p05={qr[1]:.2e}  p25={qr[2]:.2e}  "
    f"med={qr[3]:.2e}  p75={qr[4]:.2e}  p95={qr[5]:.2e}  max={qr[6]:.2e}"
)
print()

# ── how many of our zeros are voxels phaseprep DID fit ──────────────────────
ours_zero_in_mask = (ffs_slope == 0) & mask
pp_nonzero_at_those = (pp_slope != 0) & ours_zero_in_mask
print("═══ ZERO-SLOPE FOOTPRINT ═══")
print(
    f"  ours has slope==0 inside mask: {int(ours_zero_in_mask.sum()):,} "
    f"({100 * ours_zero_in_mask.sum() / n:.2f}%)"
)
print(
    f"  ...of those, phaseprep fit a nonzero slope: "
    f"{int(pp_nonzero_at_those.sum()):,} "
    f"({100 * pp_nonzero_at_those.sum() / max(ours_zero_in_mask.sum(), 1):.2f}% of our zeros)"
)
print(f"  This is the population zeroed by core.py:565 ('R² < 0 → slope=0').")
