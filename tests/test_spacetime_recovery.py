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


@pytest.mark.slow
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


def _edge_struct(coords, shape, R=15.0, taper=2.0):
    """A disk with a sharp-ish brain/air edge (smoothstep over `taper` voxels)."""
    cx, cy = (shape[0] - 1) / 2, (shape[1] - 1) / 2
    r = np.sqrt((coords[0] - cx) ** 2 + (coords[1] - cy) ** 2)
    t = np.clip((R - r) / taper, 0, 1)
    return 1.0 + 0.2 * np.cos(2 * np.pi * coords[0] / 8.0) * np.cos(
        2 * np.pi * coords[1] / 8.0
    ), t * t * (3 - 2 * t)


def test_tissue_following_beats_frozen_pose_on_fast_motion():
    """The whole reason -tfollow exists: when motion sweeps tissue between scanner
    locations frame to frame (here +/-A vox every frame, a brain edge going in and
    out of a voxel), the frozen-pose joint mixes the wrong tissue across the
    temporal window -- exactly like a static 3dTshift. Following the tissue (sample
    each neighbour at its own pose) recovers it. Guard both facts: following is
    near-perfect AND clearly beats the frozen-pose joint."""
    from fastfuncstuff.cli.nwarp import main as nwarp_main

    shape = (32, 32, 12)
    nf, tr, A = 40, 1.0, 4.0
    nz = shape[2]
    aff = np.diag([-1.0, -1.0, 1.0, 1.0])
    nu, mod = _temporal(nz)
    st = _slice_times(nz, tr, mb=2)
    tzero = float(st.mean())
    g = _grid(shape)
    kidx = g[2].astype(int)

    def sedge(c):
        checker, env = _edge_struct(c, shape)
        return checker * env

    mats = []
    for f in range(nf):
        M = np.eye(4)
        M[0, 3] = A * (-1) ** f  # alternate +/-A in x every frame
        mats.append(M)
    S = np.eye(4)

    acquired = np.zeros((*shape, nf), np.float32)
    ideal = np.zeros((*shape, nf), np.float32)
    base = sedge(g)
    for f in range(nf):
        x = np.linalg.inv(mats[f] @ S) @ g
        acquired[..., f] = (sedge(x) * mod(x, f * tr + st[kidx])).reshape(shape)
        ideal[..., f] = (base * mod(g, f * tr + tzero)).reshape(shape)

    work = tempfile.mkdtemp()
    src = os.path.join(work, "acq.nii.gz")
    nib.save(nib.Nifti1Image(acquired, aff), src)
    sp, mp, stp = (os.path.join(work, n) for n in ("s.1D", "m.1D", "st.1D"))
    with open(sp, "w") as f:
        f.write(" ".join(f"{v:.8f}" for v in S[:3, :].reshape(-1)) + "\n")
    with open(mp, "w") as f:
        for M in mats:
            f.write(" ".join(f"{v:.8f}" for v in M[:3, :].reshape(-1)) + "\n")
    np.savetxt(stp, st)
    chain = f"{sp} {mp}"
    base_args = [
        "-source",
        src,
        "-nwarp",
        chain,
        "-tpattern",
        stp,
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
        "cpu",
        "-verb",
        "0",
    ]
    frozen_p = os.path.join(work, "frozen.nii.gz")
    follow_p = os.path.join(work, "follow.nii.gz")
    # Tissue-following is the default now, so force the frozen path explicitly.
    nwarp_main([*base_args, "-prefix", frozen_p, "-frozen"])
    nwarp_main([*base_args, "-prefix", follow_p, "-tfollow"])

    frozen = np.asarray(nib.load(frozen_p).dataobj).astype(np.float32)
    follow = np.asarray(nib.load(follow_p).dataobj).astype(np.float32)

    from scipy.ndimage import binary_erosion

    fp = np.zeros((3, 3, 1), bool)
    fp[:, 1, 0] = fp[1, :, 0] = True
    mask = binary_erosion(sedge(g).reshape(shape) > 0.3, structure=fp, iterations=2)

    cfrozen = _tcorr(frozen, ideal, mask).mean()
    cfollow = _tcorr(follow, ideal, mask).mean()
    assert cfollow > 0.97, f"tissue-following should be near-perfect: {cfollow:.3f}"
    assert cfollow - cfrozen > 0.03, (
        f"tissue-following should clearly beat frozen-pose under fast motion: "
        f"follow={cfollow:.3f} frozen={cfrozen:.3f}"
    )


def _interleaved_slice_times(nz, tr):
    """3dTshift 'alt+z': adjacent slices are ~TR/2 apart in acquisition time. This
    is what turns one slice of through-plane motion into a ~TR/2 jump in a voxel's
    tap times -- the regime that exposes a non-uniform-tap bug."""
    order = list(range(0, nz, 2)) + list(range(1, nz, 2))
    st = np.zeros(nz)
    for n, k in enumerate(order):
        st[k] = n * tr / nz
    return st


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("step", [1, 2])
def test_tissue_following_survives_through_plane_motion(interleaved, step):
    """Through-plane motion makes each voxel's temporal taps land on a NON-uniform
    time grid: Delta is read at a different scanner slice every frame. Combining
    them with a plain normalised kernel average is only zeroth-order accurate, and
    with an interleaved order + an odd-slice step (taps ~TR/2 apart) it was several
    times WORSE than the frozen path it exists to improve on. The sampler now slides
    each tap onto the output frame's nominal grid with a local time derivative.

    Structure is smooth in z and uniform in-plane so spatial interpolation is near
    exact -- what is measured here is the temporal combination alone. The repo's
    other tissue-following test translates in x only, where Delta never varies.
    """
    from fastfuncstuff.cli.nwarp import main as nwarp_main

    shape, nf, tr, nz = (8, 8, 24), 40, 1.0, 24
    st = _interleaved_slice_times(nz, tr) if interleaved else np.arange(nz) * tr / nz
    tzero = float(st.mean())
    disp = [step * (f % 2) for f in range(nf)]  # 0 / step voxels in z
    sig = lambda z, t: 1.0 + 0.3 * np.sin(2 * np.pi * t / 12.0 + 0.2 * np.asarray(z))  # noqa: E731

    acq = np.zeros((*shape, nf), np.float32)
    ideal = np.zeros((*shape, nf), np.float32)
    for f in range(nf):
        for k in range(nz):
            acq[:, :, k, f] = sig(k - disp[f], f * tr + st[k])
            ideal[:, :, k, f] = sig(k, f * tr + tzero)

    work = tempfile.mkdtemp()
    src = os.path.join(work, "acq.nii.gz")
    nib.save(nib.Nifti1Image(acq, np.eye(4)), src)
    sp, mp, stp = (os.path.join(work, n) for n in ("s.1D", "m.1D", "st.1D"))
    with open(sp, "w") as fh:
        fh.write(" ".join(f"{v:.8f}" for v in np.eye(4)[:3, :].reshape(-1)) + "\n")
    with open(mp, "w") as fh:
        for f in range(nf):
            M = np.eye(4)
            M[2, 3] = disp[f]
            fh.write(" ".join(f"{v:.8f}" for v in M[:3, :].reshape(-1)) + "\n")
    np.savetxt(stp, st)

    args = [
        "-source",
        src,
        "-nwarp",
        f"{sp} {mp}",
        "-tpattern",
        stp,
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
        "cpu",
        "-verb",
        "0",
    ]
    out = {}
    for name, flag in (("frozen", "-frozen"), ("follow", "-tfollow")):
        p = os.path.join(work, f"{name}.nii.gz")
        nwarp_main([*args, "-prefix", p, flag])
        out[name] = np.asarray(nib.load(p).dataobj).astype(np.float32)

    sl = (slice(None), slice(None), slice(4, nz - 4), slice(6, nf - 6))
    den = ideal[sl].std()
    err = {k: float(np.sqrt(((v[sl] - ideal[sl]) ** 2).mean()) / den) for k, v in out.items()}

    # Tissue-following must beat the frozen path in EVERY through-plane regime --
    # the uncorrected sampler failed this at (interleaved, step=1) by ~8x.
    assert err["follow"] < err["frozen"], (
        f"tissue-following worse than frozen (interleaved={interleaved}, step={step}): "
        f"follow={err['follow']:.4f} frozen={err['frozen']:.4f}"
    )
    assert err["follow"] < 0.02, f"tissue-following residual too large: {err['follow']:.4f}"


def test_tap_slide_correction_vanishes_without_through_plane_motion():
    """The tap-slide correction is scaled by (Delta_f - Delta_j). With in-plane-only
    motion every tap reads the SAME scanner slice, so that factor is identically
    zero and the sampler must reduce to the plain weighted tap average it used
    before -- i.e. the fix cannot perturb data that never needed it.

    Checked at the sampler level, where the tap poses can be held to pure in-plane
    shifts. (End-to-end, `follow` and `frozen` legitimately differ under any motion:
    frozen warps every tap at frame j's pose, which is the thing being fixed.)
    """
    import torch

    from fastfuncstuff.processing.interp import warp_image_multi
    from fastfuncstuff.processing.spacetime import (
        TissueFollowingSampler,
        interp_slice_times,
        temporal_kernel_weights,
    )

    dev = torch.device("cpu")
    nt, shape, nz = 24, (10, 10, 12), 12
    tr, tinterp = 1.0, "cubic"
    rng = np.random.default_rng(0)
    src = torch.tensor(rng.normal(size=(nt, *shape[::-1])).astype(np.float32))
    st = torch.tensor(_interleaved_slice_times(nz, tr).astype(np.float32))
    tzero = float(st.mean())

    kk, jj, ii = torch.meshgrid(
        *[torch.arange(n, dtype=torch.float32) for n in (shape[2], shape[1], shape[0])],
        indexing="ij",
    )

    def coords_fn(f):  # pure in-plane shift -> sz identical for every frame
        return ii + 0.7 * np.sin(f / 3.0), jj + 0.5 * np.cos(f / 4.0), kk.clone()

    sampler = TissueFollowingSampler(
        src, coords_fn, (shape[2], shape[1], shape[0]), tr, tzero, st, dev, tinterp=tinterp
    )
    got = sampler.sample(nt // 2)

    # Reference: plain normalised kernel average of the same tissue-following taps,
    # with the sampler's edge rule -- a tap that samples outside the volume is not
    # data and drops out, and a voxel is covered where frame j's own pose is.
    j, half = nt // 2, 2

    def inside(sx, sy, sz):
        nz_, ny_, nx_ = src.shape[-3:]
        return (  # the resampler's coverage, which the sampler shares
            (sx >= -0.5)
            & (sx <= nx_ - 0.5)
            & (sy >= -0.5)
            & (sy <= ny_ - 0.5)
            & (sz >= -0.5)
            & (sz <= nz_ - 0.5)
        ).float()

    num = torch.zeros_like(ii)
    den = torch.zeros_like(ii)
    delta_ref = None
    for f in range(j - (half + 1), j + half + 2):
        sx, sy, sz = coords_fn(f)
        delta = interp_slice_times(sz, st)
        if delta_ref is None or f == j:
            delta_ref = interp_slice_times(coords_fn(j)[2], st)
        w = temporal_kernel_weights((j - f) + (tzero - delta) / tr, tinterp) * inside(sx, sy, sz)
        tap = warp_image_multi([src[f]], sx - ii, sy - jj, sz - kk, mode="wsinc5")[0]
        num = num + w * tap
        den = den + w
    want = num / den.clamp_min(1e-8) * inside(*coords_fn(j))

    d = float((got - want).abs().max())
    assert d < 1e-4, f"correction fired without through-plane motion: {d:.2e}"


def test_taps_outside_the_volume_drop_out_instead_of_reading_zero():
    """A constant series with motion that carries edge voxels out of the volume in some
    frames: wherever the output frame's own pose is inside, every tap that is inside
    reads the constant, so the result must be the constant -- not a blend with the zero
    padding (which dims a magnitude and biases a directly interpolated phase)."""
    import torch

    from fastfuncstuff.processing.spacetime import TissueFollowingSampler

    dev = torch.device("cpu")
    nt, (nx, ny, nz) = 20, (12, 10, 8)
    src = torch.full((nt, nz, ny, nx), 0.7)
    st = torch.tensor(_interleaved_slice_times(nz, 1.0).astype(np.float32))
    kk, jj, ii = torch.meshgrid(
        *[torch.arange(n, dtype=torch.float32) for n in (nz, ny, nx)], indexing="ij"
    )

    def coords_fn(f):  # +-1.5 voxel x-motion: the x edges leave the volume in some frames
        return ii + 1.5 * np.sin(f / 2.0), jj, kk + 0.3 * np.cos(f / 3.0)

    sampler = TissueFollowingSampler(
        src, coords_fn, (nz, ny, nx), 1.0, float(st.mean()), st, dev, tinterp="wsinc5"
    )
    j = nt // 2
    got = sampler.sample(j)
    sx, sy, sz = coords_fn(j)
    own = (
        (sx >= -0.5)
        & (sx <= nx - 0.5)
        & (sy >= -0.5)
        & (sy <= ny - 0.5)
        & (sz >= -0.5)
        & (sz <= nz - 0.5)
    )
    assert own.any() and (~own).any()
    np.testing.assert_allclose(got[own].numpy(), 0.7, atol=1e-5)
    assert float(got[~own].abs().max()) == 0.0


def _physical_case(st, tzero, pose, modes, nf=48, shape=(8, 8, 24), interp="wsinc5"):
    """nRMSE of ffs_nwarp against analytic truth when the head moves DURING a volume.

    ``pose(t)`` is the through-plane position (slices) at continuous time ``t`` (TR
    units). Slice ``k`` of volume ``f`` is acquired at ``f + st[k]`` and images the
    tissue that was under it *at that moment* -- not at a pose frozen for the whole
    volume, which is what the older tests in this file build, and which cannot
    happen in a gapless acquisition. The matrices handed to ffs_nwarp are the true
    pose at the mean slice time: a perfect per-volume fit. ``modes`` maps a name to
    the extra flags for that run.
    """
    from fastfuncstuff.cli.nwarp import main as nwarp_main

    nz = shape[2]

    def sig(z, t):
        z = np.asarray(z, dtype=float)
        return (1.0 + 0.3 * np.cos(z / 3.0)) * (1.0 + 0.3 * np.sin(2 * np.pi * t / 12.0 + 0.2 * z))

    acq = np.zeros((*shape, nf), np.float32)
    ideal = np.zeros((*shape, nf), np.float32)
    for f in range(nf):
        for k in range(nz):
            t = f + st[k]
            acq[:, :, k, f] = sig(k - pose(t), t)
            ideal[:, :, k, f] = sig(k, f + tzero)

    work = tempfile.mkdtemp()
    src, mp, stp = (os.path.join(work, n) for n in ("acq.nii.gz", "m.1D", "st.1D"))
    nib.save(nib.Nifti1Image(acq, np.eye(4)), src)
    with open(mp, "w") as fh:
        for f in range(nf):
            M = np.eye(4)
            M[2, 3] = pose(f + float(np.mean(st)))
            fh.write(" ".join(f"{v:.8f}" for v in M[:3, :].reshape(-1)) + "\n")
    np.savetxt(stp, st)
    err = {}
    for name, flags in modes.items():
        p = os.path.join(work, f"{name}.nii.gz")
        nwarp_main(
            ["-source", src, "-nwarp", mp, "-tpattern", stp, "-TR", "1.0", "-tzero", str(tzero),
             "-interp", interp, "-master", src, "-device", "cpu", "-verb", "0", "-prefix", p, *flags]
        )  # fmt: skip
        out = np.asarray(nib.load(p).dataobj).astype(np.float32)
        sl = (slice(None), slice(None), slice(6, nz - 6), slice(8, nf - 8))
        err[name] = float(np.sqrt(((out[sl] - ideal[sl]) ** 2).mean()) / ideal[sl].std())
    return err


_POSES = {"slice": ["-tfollow"], "volume": ["-tfollow", "-volume_pose"]}


@pytest.mark.parametrize("tzero", [None, 0.0])
def test_slice_pose_corrects_motion_within_a_volume(tzero):
    """The head drifts while each volume is acquired (measured on real multiband
    data: late-minus-early pose is ~0.6-0.75 of the between-volume velocity). One
    pose per volume then misplaces every slice but the one at the pose's own time.
    Reading each slice's pose at its own acquisition time removes that, and does
    not care where -tzero sits relative to the time the poses describe.
    Measured: slice pose 0.0003, volume pose 0.008.
    """
    st = np.arange(24) / 24
    tz = float(st.mean()) if tzero is None else tzero
    pose = lambda t: 1.5 * np.sin(2 * np.pi * np.asarray(t) / 40.0)  # noqa: E731
    err = _physical_case(st, tz, pose, _POSES)
    assert err["slice"] < 0.002, err
    assert err["slice"] < 0.2 * err["volume"], err


def test_slice_pose_leaves_volumes_beside_an_abrupt_move_alone():
    """Quick jumps are common and large. The head's rate of motion is slope-limited,
    so a still volume next to a jump keeps its own pose rather than having half the
    jump extrapolated into it -- which an unlimited interpolation did, measured
    1.7x worse than one pose per volume around the move in the 3-D simulation.
    """
    st = np.arange(24) / 24
    pose = lambda t: 2.0 / (1 + np.exp(-(np.asarray(t) - 24.4) / 0.04))  # noqa: E731
    err = _physical_case(st, float(st.mean()), pose, _POSES)
    assert err["slice"] <= 1.05 * err["volume"], err


def test_slice_pose_is_a_no_op_without_motion():
    """No motion: every slice pose equals the volume pose, bit for bit in effect."""
    st = _interleaved_slice_times(24, 1.0)
    err = _physical_case(st, float(st.mean()), lambda t: 0.0 * np.asarray(t), _POSES)
    assert abs(err["slice"] - err["volume"]) < 1e-6, err
