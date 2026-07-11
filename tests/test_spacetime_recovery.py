"""End-to-end recovery test for ffs_nwarp joint space-time realignment.

A synthetic 4-D series is corrupted by known in-plane motion + a constant shift +
multiband-2 interleaved slice timing, with a distinct sinusoid frequency per
3-slice block (slow -> near-Nyquist). The corruption is applied ANALYTICALLY
(look up the tissue at the motion-mapped coord, sample the world at the scanner
slice's acquisition time), so there is no inverse crime: the only errors
ffs_nwarp can be blamed for are its own interpolation.

We then run ffs_nwarp two ways on the SAME known warp chain:
  * joint  -> with -tpattern (motion + slice timing in one resample)
  * moco   -> no -tpattern (motion only)
and assert that the joint resample recovers the underlying signal, and that it
beats motion-only by a widening margin as the signal frequency rises -- exactly
where uncorrected slice timing corrupts the phase. This would have caught a sign
error in the temporal coordinate, a scanner/anatomical slice mix-up, or a broken
temporal kernel.

The fuller two-scenario demo (adds through-plane motion and a tshift-then-motion
baseline) lives in scripts/spacetime_recovery_demo.py.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")


# --- tiny, fast synthetic (kept small so the test runs in a couple of seconds) ---
SHAPE = (20, 20, 12)
NFRAMES = 36
TR = 1.0


def _struct(coords):
    xi, xj = coords[0], coords[1]
    cx, cy, rx, ry = 9.5, 9.5, 8.0, 8.0
    checker = 1.0 + 0.2 * np.cos(2 * np.pi * xi / 6.0) * np.cos(2 * np.pi * xj / 6.0)
    r = np.sqrt(((xi - cx) / rx) ** 2 + ((xj - cy) / ry) ** 2)
    env = 0.5 * (1.0 + np.cos(np.pi * np.clip(r, 0.0, 1.0)))
    return checker * env


def _temporal(nz, block=3, f_lo=0.02, f_hi=0.45, amp=0.5, seed=0):
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(nz / block))
    bnu = np.linspace(f_lo, f_hi, n_blocks)
    bph = rng.uniform(0, 2 * np.pi, n_blocks)
    blk = np.minimum(np.arange(nz) // block, n_blocks - 1)
    nu, phi = bnu[blk], bph[blk]

    def mod(coords, tau):
        k0 = np.clip(np.round(coords[2]).astype(int), 0, nz - 1)
        return 1.0 + amp * np.sin(2 * np.pi * nu[k0] * tau + phi[k0])

    return nu, mod


def _slice_times(nz, tr, mb=2):
    n_groups = nz // mb
    order = list(range(0, n_groups, 2)) + list(range(1, n_groups, 2))
    t = np.zeros(nz)
    for pos, g in enumerate(order):
        for m in range(mb):
            t[g + m * n_groups] = (pos / n_groups) * tr
    return t


def _motion(nframes, shape, seed=1):
    """In-plane only: z-rotation + x/y translation (deterministic)."""
    rng = np.random.default_rng(seed)
    tt = np.arange(nframes)
    az = np.deg2rad(2.0 * np.sin(2 * np.pi * tt / nframes) + 0.3 * rng.standard_normal(nframes))
    tx = 1.5 * np.sin(2 * np.pi * tt / nframes + 1.0)
    ty = 1.2 * np.cos(2 * np.pi * tt / nframes)
    nx, ny, nz = shape
    c = np.array([(nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2])
    mats = []
    for f in range(nframes):
        R = np.array(
            [[np.cos(az[f]), -np.sin(az[f]), 0], [np.sin(az[f]), np.cos(az[f]), 0], [0, 0, 1]]
        )
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = c - R @ c + np.array([tx[f], ty[f], 0.0])
        mats.append(M)
    S = np.eye(4)
    S[:3, 3] = [1.0, -0.8, 0.0]
    return mats, S


def _grid(shape):
    nx, ny, nz = shape
    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    return np.stack([ii.reshape(-1), jj.reshape(-1), kk.reshape(-1), np.ones(ii.size)], 0)


def _tcorr(a, b, mask):
    A, B = a[mask], b[mask]
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    return (A * B).sum(1) / (np.sqrt((A**2).sum(1) * (B**2).sum(1)) + 1e-12)


def test_joint_spacetime_recovers_and_beats_motion_only():
    from fastfuncstuff.cli.nwarp import main as nwarp_main

    nz = SHAPE[2]
    aff = np.diag([-1.0, -1.0, 1.0, 1.0])  # aff12 content == voxel matrix
    nu, mod = _temporal(nz)
    mats, S = _motion(NFRAMES, SHAPE)
    st = _slice_times(nz, TR, mb=2)
    tzero = float(st.mean())
    grid = _grid(SHAPE)
    kidx = grid[2].astype(int)

    # Analytic forward corruption + ideal target.
    acquired = np.zeros((*SHAPE, NFRAMES), np.float32)
    ideal = np.zeros((*SHAPE, NFRAMES), np.float32)
    base = _struct(grid)
    for f in range(NFRAMES):
        Tf = mats[f] @ S
        x = np.linalg.inv(Tf) @ grid
        tau = f * TR + st[kidx]
        acquired[..., f] = (_struct(x) * mod(x, tau)).reshape(SHAPE)
        ideal[..., f] = (base * mod(grid, f * TR + tzero)).reshape(SHAPE)

    work = tempfile.mkdtemp()
    src = os.path.join(work, "acq.nii.gz")
    nib.save(nib.Nifti1Image(acquired, aff), src)
    shift_p = os.path.join(work, "s.1D")
    motion_p = os.path.join(work, "m.1D")
    with open(shift_p, "w") as f:
        f.write(" ".join(f"{v:.8f}" for v in S[:3, :].reshape(-1)) + "\n")
    with open(motion_p, "w") as f:
        for M in mats:
            f.write(" ".join(f"{v:.8f}" for v in M[:3, :].reshape(-1)) + "\n")
    st_p = os.path.join(work, "st.1D")
    np.savetxt(st_p, st)
    chain = f"{shift_p} {motion_p}"

    joint_p = os.path.join(work, "joint.nii.gz")
    moco_p = os.path.join(work, "moco.nii.gz")
    nwarp_main(
        [
            "-source",
            src,
            "-nwarp",
            chain,
            "-prefix",
            joint_p,
            "-tpattern",
            st_p,
            "-TR",
            str(TR),
            "-tzero",
            str(tzero),
            "-tinterp",
            "wsinc5",
            "-interp",
            "wsinc5",
            "-master",
            src,
            "-device",
            "cpu",
            "-verb",
            "0",
        ]
    )
    nwarp_main(
        [
            "-source",
            src,
            "-nwarp",
            chain,
            "-prefix",
            moco_p,
            "-interp",
            "wsinc5",
            "-master",
            src,
            "-device",
            "cpu",
            "-verb",
            "0",
        ]
    )

    joint = np.asarray(nib.load(joint_p).dataobj).astype(np.float32)
    moco = np.asarray(nib.load(moco_p).dataobj).astype(np.float32)

    # Interior mask (in-plane erosion; every slice kept).
    from scipy.ndimage import binary_erosion

    m0 = _struct(grid).reshape(SHAPE) > 0.2
    fp = np.zeros((3, 3, 1), bool)
    fp[:, 1, 0] = fp[1, :, 0] = True
    mask = binary_erosion(m0, structure=fp, iterations=2)

    cj = _tcorr(joint, ideal, mask)
    cm = _tcorr(moco, ideal, mask)
    kmap = grid[2].reshape(SHAPE).astype(int)
    freq = nu[kmap[mask]]
    fast = freq >= np.quantile(freq, 2 / 3)

    # 1. Joint recovers the signal almost perfectly overall.
    assert cj.mean() > 0.95, f"joint overall corr too low: {cj.mean():.3f}"
    # 2. Even the fastest signals are well recovered by the joint resample.
    assert cj[fast].mean() > 0.90, f"joint fast-band corr too low: {cj[fast].mean():.3f}"
    # 3. Slice timing matters: joint clearly beats motion-only on the fast band.
    assert cj[fast].mean() - cm[fast].mean() > 0.08, (
        f"joint should beat motion-only on fast band: "
        f"{cj[fast].mean():.3f} vs {cm[fast].mean():.3f}"
    )
