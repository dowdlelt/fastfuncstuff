"""Strong end-to-end recovery test for ffs_nwarp joint space-time realignment.

Slice timing AND head motion jointly corrupt an fMRI series. ffs_nwarp
(``processing/spacetime.py``, after Roche 2011) can undo both in a SINGLE resample
by folding the slice-timing shift into the motion warp. This script builds a
synthetic series where the truth is known exactly and checks how well that single
fix recovers it, against two references.

Design (fully synthetic, no inverse crime on the headline claim)
----------------------------------------------------------------
A *continuous* world  I(x, tau) = struct(x) * (1 + a*sin(2*pi*nu(x_k)*tau + phi)):
  * struct(x): a gentle checkerboard under a smooth in-plane envelope -- spatial
    structure so motion is visible (and so a demo looks brain-ish).
  * nu(x_k): the temporal frequency is a property of the *tissue*, replicated
    across blocks of 3 neighbouring anatomical slices (a single-slice time course
    is unphysical), ramping slow -> near-Nyquist across blocks.
  * Delta(k): the acquisition offset is a property of the *scanner slice* --
    MB2, interleaved (the realistic worst case for motion).

Forward corruption is evaluated ANALYTICALLY: for scanner voxel v in frame f we
look up the tissue at x = T_f^{-1}(v) and sample the world at time f*TR + Delta(k),
where T_f = (per-frame motion) o (one constant shift). No grid interpolation is
used to *make* the data, so the only errors ffs_nwarp can be blamed for are its
own (spatial + temporal interpolation, and the paper's slow-motion assumption).

Target: I(u, j*TR + tzero) -- every anatomical voxel, all slices realigned to a
common within-TR reference time. Four corrections are scored against it:
  * follow  -- ffs_nwarp -tpattern (tissue-following joint, the DEFAULT: each
               temporal tap sampled at its own frame's pose)
  * joint   -- ffs_nwarp -tpattern -frozen (frozen-pose joint: one pose per frame)
  * tshift  -- ffs_slicetime (static 3dTshift) THEN ffs_nwarp motion-only
  * moco    -- ffs_nwarp, motion only (no slice-timing correction)
run for two motion regimes: in-plane and (harder) through-plane.

What it establishes: for slow/moderate motion the frozen-pose joint reproduces the
correct two-step (tshift-then-motion) to ~1% while both crush motion-only -- a
correctness validation. They agree because a sample's acquisition time is a
property of the scanner slice both assign correctly. But when motion sweeps tissue
between scanner locations within the temporal window (see the oscillating-edge
experiment; -tfollow), the frozen-pose joint and tshift both mix the WRONG tissue
into the temporal interpolation, and only the tissue-following joint recovers the
signal.

Substituting real data: struct() is a callable -- swap it for an interpolator over
a cropped real volume. make_motion() returns 4x4 voxel-space matrices -- swap it
for real motion params converted to this grid's voxel frame. Everything else is
unchanged.

Run:  python scripts/spacetime_recovery_demo.py  [-o OUTDIR] [--device cpu]
"""

from __future__ import annotations

import argparse
import os
import tempfile

import nibabel as nib
import numpy as np


# --------------------------------------------------------------------------- #
# World model: struct(x) and the per-slice temporal signal.
# --------------------------------------------------------------------------- #
def make_struct_analytic(shape, checker_period=12.0, checker_contrast=0.2, envelope_margin=0.1):
    """Continuous spatial structure: gentle checkerboard * smooth in-plane envelope.

    Returns a callable struct(coords) where coords is (3, N) voxel coords (i,j,k).

    The envelope is a function of (i, j) only -- every slice is a full disk, so the
    high-frequency edge slices are measured just as well as the middle ones. The
    checkerboard is band-limited (cosine, not sign) and kept low-contrast on
    purpose: it gives motion visible structure without injecting the frame-varying
    resampling error that would otherwise mask the (temporal) slice-timing effect.
    Turn the contrast up for a punchier demo image; leave it low for the metric.
    """
    nx, ny, nz = shape
    cx, cy = (nx - 1) / 2, (ny - 1) / 2
    rx, ry = nx * (0.5 - envelope_margin), ny * (0.5 - envelope_margin)

    def struct(coords):
        xi, xj = coords[0], coords[1]
        checker = 1.0 + checker_contrast * np.cos(2 * np.pi * xi / checker_period) * np.cos(
            2 * np.pi * xj / checker_period
        )
        r = np.sqrt(((xi - cx) / rx) ** 2 + ((xj - cy) / ry) ** 2)
        # Raised-cosine (Tukey) disk: C1, flat-topped, zero slope at centre AND
        # edge -> no cusp for wsinc5 to ring on, so the spatial resample under
        # motion is near-lossless and doesn't cap the recovery metric.
        env = 0.5 * (1.0 + np.cos(np.pi * np.clip(r, 0.0, 1.0)))
        return checker * env

    return struct


def make_temporal(shape, slice_block=3, f_lo=0.02, f_hi=0.45, amp=0.5, seed=0):
    """Per-slice sinusoid, but replicated across blocks of ``slice_block`` slices.

    A single-slice time course is unphysical: real signal has through-slice extent,
    so neighbouring slices carry the *same* anatomical signal. We give each block of
    ``slice_block`` consecutive anatomical slices one shared (nu, phi). Under
    interleaved multiband timing those neighbours are sampled at very different
    times -- so a correct fix must make the block agree again, and through-plane
    motion (tissue drifting within a block) keeps the frequency but changes the
    timing, which is exactly what the joint resample is meant to handle.

    Returns per-slice (nu, phi) arrays (block-piecewise-constant) and a
    modulation(coords, tau) callable.
    """
    nz = shape[2]
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(nz / slice_block))
    block_nu = np.linspace(f_lo, f_hi, n_blocks)
    block_phi = rng.uniform(0, 2 * np.pi, n_blocks)
    blk = np.minimum(np.arange(nz) // slice_block, n_blocks - 1)
    nu = block_nu[blk]
    phi = block_phi[blk]

    def modulation(coords, tau):
        xk = coords[2]
        k0 = np.clip(np.round(xk).astype(int), 0, nz - 1)
        return 1.0 + amp * np.sin(2 * np.pi * nu[k0] * tau + phi[k0])

    return nu, phi, modulation


# --------------------------------------------------------------------------- #
# Motion: per-frame affine (voxel space) + one constant shift.
# --------------------------------------------------------------------------- #
def _rot_about_center(shape, ax, ay, az):
    """4x4 voxel-space rotation (radians) about the volume centre."""
    nx, ny, nz = shape
    c = np.array([(nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2])
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = c - R @ c
    return M


def make_motion(nframes, shape, seed=1, amp_deg=2.0, amp_trans=1.5, through_plane=False):
    """Realistic-ish motion: slow drift + a couple of sharp events, per frame.

    Returns (M_list, S) where M_list[f] is the 4x4 per-frame motion and S is a
    single constant shift; the transform ffs_nwarp applies is T_f = M_f @ S.

    ``through_plane=False`` (default) keeps motion IN-PLANE (z-rotation + x/y
    translation only). Then every anatomical voxel stays in its own scanner slice,
    so its acquisition time is a clean function of that slice and the recovery
    ceiling is high. ``through_plane=True`` is the harder regime: tissue crosses
    slice boundaries, so any method must interpolate a fixed location's timeseries
    across frames (mixing tissue) and every method's ceiling drops -- including the
    joint pass, whose one-pose-per-frame slow-motion assumption gives up a little
    here (this is where a joint *estimation*, not just application, would pay off).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(nframes)
    # Smooth drift (low-freq) in each of 6 DOF.
    drift = lambda ph, sc: sc * np.sin(2 * np.pi * (t / nframes) + ph)  # noqa: E731
    az = np.deg2rad(drift(rng.uniform(0, 6), amp_deg) + 0.4 * rng.standard_normal(nframes))
    tx = drift(rng.uniform(0, 6), amp_trans) + 0.3 * rng.standard_normal(nframes)
    ty = drift(rng.uniform(0, 6), amp_trans) + 0.3 * rng.standard_normal(nframes)
    if through_plane:
        # Enough through-plane motion that tissue genuinely crosses slice
        # boundaries during the run (rotation tilts + a z drift of ~1-2 slices),
        # which is where a static 3dTshift applies the wrong offset.
        ax = np.deg2rad(
            1.5 * drift(rng.uniform(0, 6), amp_deg) + 0.4 * rng.standard_normal(nframes)
        )
        ay = np.deg2rad(drift(rng.uniform(0, 6), amp_deg))
        tz = 1.2 * drift(rng.uniform(0, 6), amp_trans)
    else:
        ax = ay = tz = np.zeros(nframes)
    # A couple of abrupt motion "spikes" -- the hard case for slice timing.
    for spike in (nframes // 3, 2 * nframes // 3):
        az[spike:] += np.deg2rad(amp_deg)
        ty[spike:] += amp_trans
        if through_plane:
            tz[spike:] += 0.8 * amp_trans

    M_list = []
    for f in range(nframes):
        M = _rot_about_center(shape, ax[f], ay[f], az[f])
        M[:3, 3] += np.array([tx[f], ty[f], tz[f]])
        M_list.append(M)
    S = np.eye(4)
    S[:3, 3] = np.array([1.5, -1.0, 0.0 if not through_plane else 0.7])
    return M_list, S


def make_slice_times(nz, tr, multiband=2, order="interleaved"):
    """Per-slice acquisition offsets in [0, TR) for a multiband, interleaved scan.

    Multiband ``mb`` excites ``mb`` slices simultaneously, separated by ``nz/mb``.
    So there are only ``nz/mb`` distinct acquisition times, and slices k and
    k+nz/mb share one. The ``nz/mb`` groups are fired in interleaved order (the
    common Siemens/CMRR default) -- adjacent scanner slices land far apart in time,
    the worst case for motion. This is a realistic MB2 interleaved timing.
    """
    if nz % multiband != 0:
        raise ValueError(f"nz={nz} not divisible by multiband={multiband}")
    n_groups = nz // multiband
    if order == "interleaved":
        group_order = list(range(0, n_groups, 2)) + list(range(1, n_groups, 2))
    else:  # ascending
        group_order = list(range(n_groups))
    times = np.zeros(nz)
    for acq_pos, g in enumerate(group_order):
        t = (acq_pos / n_groups) * tr
        for m in range(multiband):
            times[g + m * n_groups] = t
    return times


# --------------------------------------------------------------------------- #
# Forward acquisition (analytic) and ideal target.
# --------------------------------------------------------------------------- #
def _voxel_grid(shape):
    nx, ny, nz = shape
    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    ones = np.ones(ii.size)
    return np.stack([ii.reshape(-1), jj.reshape(-1), kk.reshape(-1), ones], 0), (nx, ny, nz)


def generate_acquired(shape, struct, modulation, M_list, S, slice_times, tr, tzero):
    """Analytic forward model: what the scanner records (nx,ny,nz,nframes)."""
    grid, (nx, ny, nz) = _voxel_grid(shape)
    nframes = len(M_list)
    kidx = grid[2].astype(int)  # scanner slice of each voxel -> timing offset
    out = np.zeros((nx, ny, nz, nframes), np.float32)
    for f in range(nframes):
        Tf = M_list[f] @ S  # output(anat) -> source(scanner), same as ffs applies
        x = np.linalg.inv(Tf) @ grid  # tissue coord seen by each scanner voxel
        tau = f * tr + slice_times[kidx]  # scanner-slice acquisition time
        vals = struct(x) * modulation(x, tau)
        out[..., f] = vals.reshape(nx, ny, nz)
    return out


def generate_ideal(shape, struct, modulation, nframes, tr, tzero):
    """The motion- and slice-timing-corrected truth: I(u, j*TR + tzero)."""
    grid, (nx, ny, nz) = _voxel_grid(shape)
    out = np.zeros((nx, ny, nz, nframes), np.float32)
    base = struct(grid)
    for j in range(nframes):
        tau = j * tr + tzero
        out[..., j] = (base * modulation(grid, tau)).reshape(nx, ny, nz)
    return out


def interior_mask(shape, struct, erode=3, z_erode=0):
    """Voxels safely inside the brain envelope (avoid FOV/edge contamination).

    In-plane erosion keeps every slice (hence every signal frequency) alive; the
    envelope is full in z. ``z_erode`` trims the top/bottom slices only when
    through-plane motion can push them out of the acquired FOV.
    """
    grid, (nx, ny, nz) = _voxel_grid(shape)
    m = struct(grid).reshape(nx, ny, nz) > 0.2
    from scipy.ndimage import binary_erosion

    footprint = np.zeros((3, 3, 1), bool)
    footprint[:, 1, 0] = True
    footprint[1, :, 0] = True
    m = binary_erosion(m, structure=footprint, iterations=erode)
    if z_erode:
        m[:, :, :z_erode] = False
        m[:, :, nz - z_erode :] = False
    return m


# --------------------------------------------------------------------------- #
# Metrics.
# --------------------------------------------------------------------------- #
def temporal_corr(a, b, mask):
    """Per-voxel Pearson correlation over time, within mask -> (mask,) vector."""
    A = a[mask]  # (Nvox, T)
    B = b[mask]
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    num = (A * B).sum(1)
    den = np.sqrt((A**2).sum(1) * (B**2).sum(1)) + 1e-12
    return num / den


def amplitude_slope(rec, ideal, mask):
    """Per-voxel regression slope rec ~ ideal (1.0 = perfect amplitude recovery)."""
    R = rec[mask]
    I = ideal[mask]
    R = R - R.mean(1, keepdims=True)
    I = I - I.mean(1, keepdims=True)
    return (R * I).sum(1) / ((I**2).sum(1) + 1e-12)


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #
def write_aff12_single(path, M):
    with open(path, "w") as f:
        f.write(" ".join(f"{v:.8f}" for v in M[:3, :].reshape(-1)) + "\n")


def write_aff12_series(path, M_list):
    with open(path, "w") as f:
        for M in M_list:
            f.write(" ".join(f"{v:.8f}" for v in M[:3, :].reshape(-1)) + "\n")


def run_scenario(through_plane, work, device="cpu", shape=(48, 48, 20), nframes=64, tr=1.0):
    """One scenario: build corrupted data, run all three corrections, score them.

    The three corrections compared:
      * joint   -- ffs_nwarp with -tpattern (motion + slice timing, one resample)
      * moco    -- ffs_nwarp, no slice timing (motion only)
      * tshift  -- ffs_slicetime (static 3dTshift) THEN ffs_nwarp motion-only:
                   the naive two-step. Correct when the head is still; wrong when
                   through-plane motion carries tissue across scanner slices.
    """
    from fastfuncstuff.cli.nwarp import main as nwarp_main
    from fastfuncstuff.cli.slicetime import main as slicetime_main

    aff = np.diag([-1.0, -1.0, 1.0, 1.0])  # 1mm LPI: aff12 content == voxel matrix
    tag = "through-plane" if through_plane else "in-plane"

    struct = make_struct_analytic(shape)
    nu, phi, modulation = make_temporal(shape)
    M_list, S = make_motion(nframes, shape, through_plane=through_plane)
    slice_times = make_slice_times(shape[2], tr, multiband=2, order="interleaved")
    tzero = float(slice_times.mean())

    acquired = generate_acquired(shape, struct, modulation, M_list, S, slice_times, tr, tzero)
    ideal = generate_ideal(shape, struct, modulation, nframes, tr, tzero)

    src = os.path.join(work, f"acquired_{tag}.nii.gz")
    nib.save(nib.Nifti1Image(acquired, aff), src)

    shift_p = os.path.join(work, f"shift_{tag}.1D")
    motion_p = os.path.join(work, f"motion_{tag}.1D")
    write_aff12_single(shift_p, S)
    write_aff12_series(motion_p, M_list)
    chain = f"{shift_p} {motion_p}"  # T_f = M_f @ S
    st_p = os.path.join(work, f"slicetimes_{tag}.1D")
    np.savetxt(st_p, slice_times)

    joint_p = os.path.join(work, f"joint_{tag}.nii.gz")
    follow_p = os.path.join(work, f"follow_{tag}.nii.gz")
    moco_p = os.path.join(work, f"moco_{tag}.nii.gz")
    tshift_p = os.path.join(work, f"tshift_{tag}.nii.gz")
    tshiftmoco_p = os.path.join(work, f"tshiftmoco_{tag}.nii.gz")

    joint_args = [
        "-source",
        src,
        "-nwarp",
        chain,
        "-tpattern",
        st_p,
        "-TR",
        str(tr),
        "-tzero",
        str(tzero),
        "-tinterp",
        "wsinc5",
        "-interp",
        "wsinc5",
        "-master",
        src,
        "-device",
        device,
        "-verb",
        "0",
    ]
    # Tissue-following is the default; -frozen forces the old slow-motion path.
    print(f"[{tag}] joint space-time (frozen pose) ...")
    nwarp_main([*joint_args, "-prefix", joint_p, "-frozen"])
    print(f"[{tag}] joint space-time (tissue-following) ...")
    nwarp_main([*joint_args, "-prefix", follow_p, "-tfollow"])
    print(f"[{tag}] motion-only ...")
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
            device,
            "-verb",
            "0",
        ]
    )
    print(f"[{tag}] tshift-then-motion (naive two-step) ...")
    slicetime_main(
        [
            "-input",
            src,
            "-prefix",
            tshift_p,
            "-tpattern",
            st_p,
            "-TR",
            str(tr),
            "-tzero",
            str(tzero),
            "-wsinc5",
            "-device",
            device,
        ]
    )
    nwarp_main(
        [
            "-source",
            tshift_p,
            "-nwarp",
            chain,
            "-prefix",
            tshiftmoco_p,
            "-interp",
            "wsinc5",
            "-master",
            src,
            "-device",
            device,
            "-verb",
            "0",
        ]
    )

    rec = {
        "follow": np.asarray(nib.load(follow_p).dataobj).astype(np.float32),
        "joint": np.asarray(nib.load(joint_p).dataobj).astype(np.float32),
        "tshift": np.asarray(nib.load(tshiftmoco_p).dataobj).astype(np.float32),
        "moco": np.asarray(nib.load(moco_p).dataobj).astype(np.float32),
    }

    mask = interior_mask(shape, struct, z_erode=2 if through_plane else 0)
    grid, sh = _voxel_grid(shape)
    kmap = grid[2].reshape(sh).astype(int)
    freq = nu[kmap[mask]]

    corr = {m: temporal_corr(v, ideal, mask) for m, v in rec.items()}
    ampl = {m: amplitude_slope(v, ideal, mask) for m, v in rec.items()}

    methods = ["follow", "joint", "tshift", "moco"]
    print(f"\n=== {tag} motion | MB2 interleaved | {int(mask.sum())} interior voxels ===")
    print("  band              | " + " ".join(f"{m:>7}" for m in methods) + "   (corr)")
    print("  ------------------+" + "-" * 34)
    edges = np.quantile(freq, [0, 1 / 3, 2 / 3, 1.0])
    for lo, hi, name in [
        (edges[0], edges[1], "slow"),
        (edges[1], edges[2], "mid "),
        (edges[2], edges[3] + 1e-9, "fast"),
    ]:
        s = (freq >= lo) & (freq < hi)
        print(
            f"  {name} {lo:.2f}-{hi:.2f} Hz    | "
            + " ".join(f"{corr[m][s].mean():7.3f}" for m in methods)
        )
    print("  OVERALL           | " + " ".join(f"{corr[m].mean():7.3f}" for m in methods))
    print("  amplitude (mean)  | " + " ".join(f"{ampl[m].mean():7.3f}" for m in methods))

    return dict(
        tag=tag,
        shape=shape,
        mask=mask,
        kmap=kmap,
        nu=nu,
        freq=freq,
        ideal=ideal,
        rec=rec,
        corr=corr,
    )


def run(outdir, device="cpu", nframes=64, tr=1.0):
    os.makedirs(outdir, exist_ok=True)
    work = tempfile.mkdtemp()
    print(
        f"frames={nframes} TR={tr}s  Nyquist={0.5 / tr:.3f} Hz  "
        f"signal replicated across 3-slice blocks\n"
    )
    results = [
        run_scenario(False, work, device=device, nframes=nframes, tr=tr),
        run_scenario(True, work, device=device, nframes=nframes, tr=tr),
    ]
    _figure(outdir, results)
    return results


_STYLE = {
    "follow": ("C4", "D-", "joint (tissue-following)"),
    "joint": ("C0", "o-", "joint (frozen pose)"),
    "tshift": ("C2", "^-", "tshift-then-motion"),
    "moco": ("C3", "s--", "motion-only"),
}


def _figure(outdir, results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(results), 2, figsize=(13, 4.2 * len(results)))
    if len(results) == 1:
        axes = axes[None, :]
    for row, r in enumerate(results):
        nu, mask, kmap, freq = r["nu"], r["mask"], r["kmap"], r["freq"]
        # Legible mid-high slice for the timeseries panel.
        kdemo = min(range(len(nu)), key=lambda k: abs(nu[k] - 0.29))
        idx = np.argwhere(mask & (kmap == kdemo))
        vx = tuple(int(c) for c in idx[len(idx) // 2])

        ax = axes[row, 0]
        ax.plot(r["ideal"][vx], "k-", lw=2.2, label="ideal (truth)")
        for m, (c, _, lab) in _STYLE.items():
            ax.plot(r["rec"][m][vx], c + ".-", lw=1.2, alpha=0.9, label=lab)
        ax.set_title(f"{r['tag']}: slice k={kdemo} ({nu[kdemo]:.2f} Hz), voxel {vx}")
        ax.set_xlabel("frame")
        ax.set_ylabel("signal")
        ax.legend(fontsize=8)

        ax = axes[row, 1]
        fb = np.linspace(freq.min(), freq.max(), 12)
        for m, (c, mk, lab) in _STYLE.items():
            fc, cc = [], []
            for a, b in zip(fb[:-1], fb[1:], strict=False):
                s = (freq >= a) & (freq < b)
                if s.sum() > 5:
                    fc.append((a + b) / 2)
                    cc.append(r["corr"][m][s].mean())
            ax.plot(fc, cc, c + mk, label=lab)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{r['tag']}: recovery vs signal frequency")
        ax.set_xlabel("slice signal frequency (Hz)")
        ax.set_ylabel("corr with truth")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    p = os.path.join(outdir, "spacetime_recovery.png")
    fig.savefig(p, dpi=110)
    print(f"\nsaved figure -> {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--outdir", default="scratch_spacetime")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run(args.outdir, device=args.device)
