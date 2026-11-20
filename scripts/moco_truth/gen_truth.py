"""Native-resolution moco ground truth on the GPU, independent of ffs code.

A 3D GRE image is a DFT reconstruction on all three axes, i.e. an exact
trigonometric polynomial, so moving it with FFT phase ramps (translation, then
three Paeth plane rotations of three shears each) is exact up to the spectral
corners a rotation pushes past the grid. The moved 0.35 mm volume is then
sampled the way an EPI would be:
  2D: k-space truncation in-plane, a 6-slice box slice profile CENTRED on each
      coarse sample along z;
  3D: k-space truncation on all axes (after cropping a margin of slices).
Writes AFNI-convention truth (RAI mm, base -> source): {out}_truth.aff12.1D,
{out}_{2d,3d}_n{noise}.nii series, {out}_mask.nii.gz and {out}_meta.json (tags).
Score with score_motion.py (estimates) and score_resample.py (output kernels).

The FFT rotation aliases noise near the fine grid's Nyquist, which the
reduction to the coarse grid discards: validate it at the OUTPUT band
(halves-vs-once 1.5e-6 there), not at the fine grid (0.7%).
"""

import argparse
import json
import math

import nibabel as nb
import numpy as np
import torch
from scipy import ndimage
from scipy.spatial.transform import Rotation

FLIP = np.diag([-1.0, -1.0, 1.0, 1.0])
DEV = torch.device("cuda")


def shift_axis(vol, axis, s):
    """Pull along axis: out(u) = vol(u + s); s broadcastable, size 1 along axis."""
    n = vol.shape[axis]
    spec = torch.fft.rfft(vol, dim=axis)
    k = torch.fft.rfftfreq(n, device=vol.device, dtype=torch.float64).to(torch.float32) * n
    shape = [1, 1, 1]
    shape[axis] = k.numel()
    spec *= torch.exp((s * k.reshape(shape)) * (2j * math.pi / n))
    return torch.fft.irfft(spec, n=n, dim=axis)


def coord(vol, axis, c):
    shape = [1, 1, 1]
    shape[axis] = vol.shape[axis]
    return (
        torch.arange(vol.shape[axis], device=vol.device, dtype=torch.float32) - c[axis]
    ).reshape(shape)


def shear_pull(vol, p, q, alpha, c):
    """out(u) = vol(u + alpha * (u_q - c_q) e_p)."""
    return shift_axis(vol, p, alpha * coord(vol, q, c))


def rot_pull(vol, k, theta, c):
    """out(u) = vol(c + R_k(theta) (u - c)), Paeth three-shear."""
    if abs(theta) < 1e-12:
        return vol
    p, q = [(1, 2), (2, 0), (0, 1)][k]  # right-handed plane for rotation about k
    a, b = -math.tan(theta / 2), math.sin(theta)
    vol = shear_pull(vol, p, q, a, c)
    vol = shear_pull(vol, q, p, b, c)
    return shear_pull(vol, p, q, a, c)


def rigid_pull(vol, L, d, c):
    """out(u) = vol(c + L (u - c) + d) for a proper rotation L."""
    for ax in range(3):
        if abs(d[ax]) > 0:
            vol = shift_axis(vol, ax, float(d[ax]))
    ex, ey, ez = Rotation.from_matrix(L).as_euler("xyz")  # L = Rz Ry Rx
    for k, th in ((2, ez), (1, ey), (0, ex)):
        vol = rot_pull(vol, k, float(th), c)
    return vol


def truncate(vol, axis, nc):
    """k-space truncation of a real volume to nc samples (Nyquist split)."""
    n = vol.shape[axis]
    X = torch.fft.fft(vol, dim=axis)
    h = nc // 2
    idx = list(range(h)) + list(range(n - h + 1 if nc % 2 == 0 else n - h, n))
    parts = [X.index_select(axis, torch.tensor(list(range(h)), device=vol.device))]
    if nc % 2 == 0:
        nyq = 0.5 * (
            X.index_select(axis, torch.tensor([h], device=vol.device))
            + X.index_select(axis, torch.tensor([n - h], device=vol.device))
        )
        parts.append(nyq)
        parts.append(
            X.index_select(axis, torch.tensor(list(range(n - h + 1, n)), device=vol.device))
        )
    else:
        parts.append(X.index_select(axis, torch.tensor(list(range(n - h, n)), device=vol.device)))
    del idx
    C = torch.cat(parts, dim=axis) * (nc / n)
    return torch.fft.ifft(C, dim=axis).real


def rigid(rx, ry, rz, tx, ty, tz):
    a, b, c = np.deg2rad([rx, ry, rz])
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    M = np.eye(4)
    M[:3, :3] = Rz @ Ry @ Rx
    M[:3, 3] = [tx, ty, tz]
    return M


def trace_motions(path, ref_img, tag):
    """Real aff12 trace (RAI, base->source) -> RAS [R, t] about the grid centre."""
    img = nb.load(ref_img)
    cen = FLIP @ img.affine @ np.r_[(np.array(img.shape[:3]) - 1) / 2, 1.0]
    out, tags = [], []
    for i, row in enumerate(np.loadtxt(path).reshape(-1, 3, 4)):
        M = np.eye(4)
        M[:3] = row
        t = (M @ cen)[:3] - cen[:3]
        T = np.eye(4)
        # ffs_moco traces from anisotropic voxels are not exactly rigid (voxel-space
        # rotation); use the nearest rotation so the truth is what gets applied.
        U, _, Vt = np.linalg.svd(M[:3, :3])
        T[:3, :3] = FLIP[:3, :3] @ (U @ Vt) @ FLIP[:3, :3]
        T[:3, 3] = FLIP[:3, :3] @ t
        out.append(T)
        tags.append(f"{tag}_{i:04d}")
    return out[1:], tags[1:]  # drop the trace's own base (identity)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fine")
    ap.add_argument("out")
    ap.add_argument("-factor", type=int, nargs=3, default=[6, 6, 6])
    ap.add_argument("-ncoarse", type=int, nargs=3, default=[94, 94, 56])
    ap.add_argument("-trace", nargs=3, action="append", default=[], metavar=("AFF12", "REF", "TAG"))
    ap.add_argument("-noise", type=float, nargs="+", default=[0.0, 0.02, 0.05])
    ap.add_argument("-seed", type=int, default=0)
    ap.add_argument(
        "-battery", action="store_true", help="dense sub-voxel battery instead of tiny set"
    )
    a = ap.parse_args()

    img = nb.load(a.fine)
    base = torch.tensor(np.asarray(img.dataobj, dtype=np.float32), device=DEV)
    h = [float(z) for z in img.header.get_zooms()[:3]]
    N = base.shape
    F = a.factor
    nc = a.ncoarse
    o = [(N[i] - F[i] * nc[i]) // 2 for i in range(3)]
    assert all(oo >= 4 for oo in o), o
    aff_f = np.diag([-h[0], h[1], h[2], 1.0])
    aff_f[:3, 3] = [h[0] * N[0] / 2, -h[1] * N[1] / 2, -h[2] * N[2] / 2]
    aff_c = aff_f @ np.diag([*F, 1.0])
    aff_c[:, 3] = aff_f @ np.r_[o, 1.0]
    cf = [o[i] + F[i] * (nc[i] - 1) / 2 for i in range(3)]  # coarse centre in fine index
    centre = (aff_f @ np.r_[cf, 1.0])[:3]

    vox_c = F[0] * h[0]
    Ts, tags = [np.eye(4)], ["base"]
    for path, ref, tag in a.trace:
        t, g = trace_motions(path, ref, tag)
        Ts += t
        tags += g
    if a.battery:
        rng = np.random.default_rng(11)
        for ax, nm in enumerate("xyz"):
            vox_a = F[ax] * h[ax]
            for frac in np.round(np.arange(0.025, 1.0001, 0.025), 3):
                t = [0.0, 0.0, 0.0]
                t[ax] = frac * vox_a
                Ts.append(rigid(0, 0, 0, *t))
                tags.append(f"sweep_t{nm}_{frac}")
        for ax, nm in enumerate("xyz"):
            for deg in np.round(np.geomspace(0.005, 0.5, 12), 4):
                r = [0.0, 0.0, 0.0]
                r[ax] = deg
                Ts.append(rigid(*r, 0, 0, 0))
                tags.append(f"sweep_r{nm}_{deg}")
        for name, sd in (("jitter", 0.02), ("small", 0.1), ("moderate", 0.4)):
            for i in range(40):
                r = rng.normal(0, sd, 3)
                t = rng.normal(0, sd, 3) * vox_c
                Ts.append(rigid(*r, *t))
                tags.append(f"{name}_{i:02d}")
    for frac in [] if a.battery else [0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1]:
        Ts.append(rigid(0, 0, 0, frac * vox_c, 0, 0))
        tags.append(f"tiny_tx_{frac}")
        Ts.append(rigid(0, 0, 0, 0, 0, frac * vox_c))
        tags.append(f"tiny_tz_{frac}")
    for deg in [] if a.battery else [0.004, 0.01, 0.02, 0.04, 0.06, 0.1, 0.15, 0.2]:
        Ts.append(rigid(deg, 0, 0, 0, 0, 0))
        tags.append(f"tiny_rx_{deg}")
        Ts.append(rigid(0, 0, deg, 0, 0, 0))
        tags.append(f"tiny_rz_{deg}")

    Ainv = np.linalg.inv(aff_f)
    C = np.eye(4)
    C[:3, 3] = centre
    fz = F[2]
    # Box slice profile of fz fine slices centred on each coarse sample.
    taps = [1.0] * fz if fz % 2 else [0.5] + [1.0] * (fz - 1) + [0.5]
    box = torch.tensor(taps, device=DEV) / fz
    half = len(taps) // 2
    zi = torch.tensor([o[2] + fz * j for j in range(nc[2])], device=DEV)
    sl = [slice(o[i], o[i] + F[i] * nc[i]) for i in range(3)]
    out2d = np.zeros((*nc, len(Ts)), np.float32)
    out3d = np.zeros((*nc, len(Ts)), np.float32)
    rows = []
    for t, T in enumerate(Ts):
        Tc = C @ T @ np.linalg.inv(C)
        M = Ainv @ np.linalg.inv(Tc) @ aff_f  # source fine voxel -> base fine voxel
        L, m = M[:3, :3], M[:3, 3]
        d = L @ np.array(cf) + m - np.array(cf)
        with torch.no_grad():
            moved = rigid_pull(base, L, d, cf) if t else base
            ip = truncate(truncate(moved[sl[0], sl[1], :], 0, nc[0]), 1, nc[1])
            v2 = sum(w * ip.index_select(2, zi + (k - half)) for k, w in enumerate(box))
            v3 = truncate(ip[:, :, sl[2]], 2, nc[2])
        out2d[..., t] = v2.cpu().numpy()
        out3d[..., t] = v3.cpu().numpy()
        rows.append((FLIP @ Tc @ FLIP)[:3].ravel())
        if t % 50 == 0:
            print(t, len(Ts), flush=True)

    b2 = out2d[..., 0]
    mask = ndimage.binary_opening(b2 > 0.15 * np.percentile(b2, 98), iterations=2)
    lab, n = ndimage.label(mask)
    if n > 1:
        mask = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    mask = ndimage.binary_fill_holes(mask)
    nb.save(nb.Nifti1Image(mask.astype(np.uint8), aff_c), f"{a.out}_mask.nii.gz")
    np.savetxt(f"{a.out}_truth.aff12.1D", np.array(rows), fmt="%.10f")
    rng = np.random.default_rng(a.seed)
    for kind, arr in (("2d", out2d), ("3d", out3d)):
        bm = float(arr[..., 0][mask].mean())
        for s in a.noise:
            ser = arr + (rng.normal(0, s * bm, arr.shape).astype(np.float32) if s else 0)
            im = nb.Nifti1Image(ser, aff_c)
            im.header.set_xyzt_units("mm", "sec")
            im.header["pixdim"][4] = 1.0
            nb.save(im, f"{a.out}_{kind}_n{int(round(s * 1000)):03d}.nii")
    json.dump(
        {
            "tags": tags,
            "voxel_mm": [F[i] * h[i] for i in range(3)],
            "shape": nc,
            "centre_ras": centre.tolist(),
        },
        open(f"{a.out}_meta.json", "w"),
        indent=1,
    )
    print("done", nc, len(Ts), int(mask.sum()))


if __name__ == "__main__":
    main()
