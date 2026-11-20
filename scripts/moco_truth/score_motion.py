"""Score estimated aff12 series against truth as a displacement field (mm) over the mask."""

import argparse
import json
import re
from collections import defaultdict

import nibabel as nb
import numpy as np

FLIP = np.diag([-1.0, -1.0, 1.0, 1.0])


def load_aff12(path):
    m = np.loadtxt(path, comments="#").reshape(-1, 12)
    out = np.tile(np.eye(4), (len(m), 1, 1))
    out[:, :3, :] = m.reshape(-1, 3, 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("est", nargs="+")
    ap.add_argument("-invert", action="store_true")
    ap.add_argument("-sweeps", action="store_true")
    ap.add_argument("-gain", action="store_true")
    a = ap.parse_args()
    meta = json.load(open(f"{a.prefix}_meta.json"))
    tags = meta["tags"]
    mimg = nb.load(f"{a.prefix}_mask.nii.gz")
    mask = np.asarray(mimg.dataobj) > 0
    ijk = np.c_[np.argwhere(mask), np.ones(mask.sum())].T
    xyz = FLIP @ mimg.affine @ ijk  # RAI mm
    truth = load_aff12(f"{a.prefix}_truth.aff12.1D")
    groups = defaultdict(list)
    for i, t in enumerate(tags):
        groups[re.sub(r"_[-\d.]+$", "", t)].append(i)

    print(
        f"{'estimate':42s} "
        + " ".join(f"{g:>14s}" for g in groups)
        + "   (mean|max displacement error, mm)"
    )
    for path in a.est:
        est = load_aff12(path)
        if a.invert:
            est = np.linalg.inv(est)
        err = np.linalg.norm(((est - truth) @ xyz)[:, :3], axis=1)  # (nt, nvox)
        cells = []
        for idx in groups.values():
            e = err[idx]
            cells.append(f"{e.mean():6.3f}|{e.max():6.3f}")
        print(f"{path:42s} " + " ".join(f"{c:>14s}" for c in cells))
        if a.gain:
            for g, idx in groups.items():
                if not (g.startswith("tiny") or g.startswith("sweep_r")):
                    continue
                dt = np.linalg.norm(((truth[idx] - np.eye(4)) @ xyz)[:, :3], axis=1).mean(1)
                de = np.linalg.norm(((est[idx] - np.eye(4)) @ xyz)[:, :3], axis=1).mean(1)
                print(f"   {g}: true mm " + " ".join(f"{t:.4f}" for t in dt))
                print(
                    f"   {g}: est/true "
                    + " ".join(f"{e / t:.3f}" for t, e in zip(dt, de, strict=True))
                )
        if a.gain:
            for g, idx in groups.items():
                if g not in ("jitter", "small", "moderate"):
                    continue
                dt = np.linalg.norm(((truth[idx] - np.eye(4)) @ xyz)[:, :3], axis=1).mean(1)
                e = err[idx].mean(1)
                print(
                    f"   {g}: mean true disp {dt.mean():.4f} mm, mean err {e.mean():.4f} mm, err/disp {np.median(e / dt):.3f}"
                )
        if a.sweeps:
            for g, idx in groups.items():
                if not g.startswith("sweep_t"):
                    continue
                ax = "xyz".index(g[-1])
                # mean displacement of the brain along the sweep axis (RAS axis -> RAI sign)
                sgn = -1 if ax < 2 else 1
                tr = (((truth[idx] - np.eye(4)) @ xyz)[:, ax].mean(1)) * sgn
                es = (((est[idx] - np.eye(4)) @ xyz)[:, ax].mean(1)) * sgn
                vox = meta["voxel_mm"][ax]
                print(f"   {g}: frac  " + " ".join(f"{t / vox:.2f}" for t in tr))
                print(
                    f"   {g}: err%  "
                    + " ".join(f"{100 * (e - t) / vox:+.1f}" for t, e in zip(tr, es, strict=True))
                )


if __name__ == "__main__":
    main()
