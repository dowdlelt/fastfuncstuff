"""Run phaseprep PhaseFitOdr directly on the already-MC'd, already-unwrapped data.

Skips the BIDS/nipype workflow and the heavy preprocessing wrapper — calls into
phaseprep's actual ODR math per voxel. Output naming mirrors phaseprep so its
R²/slope maps can be compared voxelwise against ffs_phasereg outputs.

Run in the py37_phaseprep env:
    conda run -n py37_phaseprep python run_phaseprep_reference.py
"""

import os
import sys
import time
import multiprocessing as mp
import numpy as np
import nibabel as nb
from scipy import odr
from tqdm import tqdm

N_WORKERS = 6

HERE = os.path.dirname(os.path.abspath(__file__))
MAG_PATH = os.path.join(HERE, "method_direct_aligned.nii.gz")
PHASE_PATH = os.path.join(HERE, "method_direct_aligned_phase.nii.gz")
MASK_PATH = os.path.join(HERE, "mask.nii.gz")
OUT_PREFIX = os.path.join(HERE, "phaseprep_ref")
NOISE_LB = 0.15  # Hz, Stanley §2.2.3 / phaseprep default

# ── load ────────────────────────────────────────────────────────────────────
mag_img = nb.load(MAG_PATH)
pha_img = nb.load(PHASE_PATH)
TR = float(mag_img.header.get_zooms()[-1])
print(f"TR={TR}s  shape={mag_img.shape}")

mag = np.asarray(mag_img.get_fdata(), dtype=np.float64)
pha = np.asarray(pha_img.get_fdata(), dtype=np.float64)
nx, ny, nz, nt = mag.shape

if os.path.exists(MASK_PATH):
    mask3d = nb.load(MASK_PATH).get_fdata().astype(bool)
    print(f"Using mask: {mask3d.sum():,} voxels")
else:
    mm = mag.mean(axis=-1)
    mask3d = mm > 0.03 * mm.max()
    print(f"Auto signal-threshold mask: {mask3d.sum():,} voxels")

mag = mag.reshape(-1, nt)
pha = pha.reshape(-1, nt)
mask = mask3d.reshape(-1)
nv = mag.shape[0]

# ── linear detrend (mimics DetrendMag + PreprocessPhase final step) ─────────
# data - polyfit(linear) + mean(data)
xval = np.arange(nt, dtype=np.float64)


def linear_detrend(ts):
    p = np.polyfit(xval, ts, 1)
    return ts - np.polyval(p, xval) + ts.mean()


# ── PhaseFitOdr math (lifted from phaseprep/interfaces/PhaseFitOdr.py) ──────
def get_noise(ts):
    fft_freqs = np.abs(np.fft.fftfreq(len(ts), TR))
    noise = np.fft.ifft((fft_freqs > NOISE_LB) * np.fft.fft(ts - ts.mean()))
    return float(np.std(noise.real))


def multiplelinear(beta, x):
    return beta[0] * x[0] + beta[1] * x[1]


linear_model = odr.Model(multiplelinear)

# ── allocate outputs ────────────────────────────────────────────────────────
beta_A = np.zeros(nv)  # phase coefficient
beta_B = np.zeros(nv)  # intercept
r2 = np.zeros(nv)
stdm_arr = np.zeros(nv)
stdp_arr = np.zeros(nv)
sim = np.zeros((nv, nt))  # phaseprep's "sim" = predicted mag
filt = np.zeros((nv, nt))  # mag - sim + mean(mag)

# ── per-voxel fit (parallel via fork-inherited globals) ─────────────────────
# Skip codes: 1 = bad noise std, 2 = ODR exception, 3 = zero ss_total
SKIP_STD = 1
SKIP_ODR = 2
SKIP_SSTOT = 3


def fit_voxel(idx):
    """Worker: returns (idx, r2, A, B, stdm, stdp, sim_ts, filt_ts) or (idx, skip_code, None...).

    Relies on `mag`, `pha`, `linear_detrend`, `get_noise`, `linear_model` being
    inherited from the parent process via fork.
    """
    m = linear_detrend(mag[idx])
    p = linear_detrend(pha[idx])

    stdm = get_noise(m)
    stdp = get_noise(p)
    if stdm <= 0 or stdp <= 0 or not np.isfinite(stdm) or not np.isfinite(stdp):
        return idx, SKIP_STD, None

    design = np.vstack([p, np.ones_like(p)])
    ests = [
        m.std() / p.std() if p.std() > 0 else 0.0,
        m.mean() / p.mean() if abs(p.mean()) > 1e-12 else m.mean(),
    ]
    data = odr.RealData(design, m, sx=np.hstack([stdp, np.finfo(float).eps]), sy=stdm)
    try:
        res = odr.ODR(data, linear_model, beta0=ests, maxit=400).run()
    except Exception:
        return idx, SKIP_ODR, None

    est = res.y
    mm_x = m.mean()
    ss_tot = float(((m - mm_x) ** 2).sum())
    if ss_tot <= 0:
        return idx, SKIP_SSTOT, None

    r2_v = 1.0 - float(((m - est) ** 2).sum()) / ss_tot
    return idx, 0, (
        r2_v,
        float(res.beta[0]),
        float(res.beta[1]),
        stdm,
        stdp,
        est.astype(np.float32),
        (m - est + mm_x).astype(np.float32),
    )


if __name__ == "__main__":
    voxel_ids = np.where(mask)[0]
    total = len(voxel_ids)
    print(f"Fitting {total:,} masked voxels with scipy.odr on {N_WORKERS} workers...", flush=True)

    n_ok = 0
    n_skip_std = 0
    n_skip_odr = 0
    n_skip_sstot = 0
    running_max_r2 = -np.inf
    t0 = time.time()

    # Chunksize: larger = less IPC overhead, smaller = smoother progress.
    # ~5–15 ms per ODR fit, so 256 voxels per chunk ≈ 1–4 s of work — fine.
    chunksize = 256

    # Force fork on linux so workers inherit `mag`/`pha`/`linear_model` cheaply.
    ctx = mp.get_context("fork")
    with ctx.Pool(N_WORKERS) as pool:
        pbar = tqdm(total=total, mininterval=0.5, smoothing=0.05,
                    unit="vox", dynamic_ncols=True)
        for idx, code, payload in pool.imap_unordered(fit_voxel, voxel_ids, chunksize=chunksize):
            if code == 0:
                r2_v, A, B, stdm, stdp, sim_ts, filt_ts = payload
                r2[idx] = r2_v
                beta_A[idx] = A
                beta_B[idx] = B
                stdm_arr[idx] = stdm
                stdp_arr[idx] = stdp
                sim[idx] = sim_ts
                filt[idx] = filt_ts
                n_ok += 1
                if r2_v > running_max_r2:
                    running_max_r2 = r2_v
            elif code == SKIP_STD:
                n_skip_std += 1
            elif code == SKIP_ODR:
                n_skip_odr += 1
            elif code == SKIP_SSTOT:
                n_skip_sstot += 1

            pbar.update(1)
            if (pbar.n & 4095) == 0:
                pbar.set_postfix(
                    ok=n_ok,
                    max_r2=f"{running_max_r2:.3f}",
                    skip=n_skip_std + n_skip_odr + n_skip_sstot,
                    refresh=False,
                )
        pbar.close()

    print(
        f"\nFit summary: ok={n_ok:,}  skipped_std={n_skip_std:,}  "
        f"skipped_odr={n_skip_odr:,}  skipped_zerovar={n_skip_sstot:,}",
        flush=True,
    )
    print(f"Total time: {time.time() - t0:.1f}s", flush=True)


# ── save outputs ────────────────────────────────────────────────────────────
def save3d(arr, suffix):
    img = nb.Nifti1Image(arr.reshape(nx, ny, nz).astype(np.float32), mag_img.affine, mag_img.header)
    path = f"{OUT_PREFIX}_{suffix}.nii.gz"
    nb.save(img, path)
    print(f"  wrote {path}")


def save4d(arr, suffix):
    img = nb.Nifti1Image(
        arr.reshape(nx, ny, nz, nt).astype(np.float32), mag_img.affine, mag_img.header
    )
    path = f"{OUT_PREFIX}_{suffix}.nii.gz"
    nb.save(img, path)
    print(f"  wrote {path}")


if __name__ == "__main__":
    save3d(r2, "r2")
    save3d(beta_A, "slope")  # the A in M = A·φ + B
    save3d(beta_B, "intercept")
    save3d(stdm_arr, "stdm")
    save3d(stdp_arr, "stdp")
    save4d(sim, "macro")  # equivalent to ffs_phasereg's _macro
    save4d(filt, "corrected")  # equivalent to ffs_phasereg's _corrected

    # ── histogram ──────────────────────────────────────────────────────────
    r2_good = r2[mask]
    print(f"\nR² over {mask.sum():,} masked voxels:")
    print(f"  median={np.median(r2_good):.4f}  mean={r2_good.mean():.4f}  max={r2_good.max():.4f}")
    for thresh in (0.01, 0.05, 0.10, 0.20, 0.30, 0.50):
        n = int((r2_good > thresh).sum())
        pct = 100.0 * n / len(r2_good)
        print(f"  R² > {thresh:.2f}:  {n:>8,} ({pct:5.2f}%)")
