"""Output-resampler test: undo the TRUE motion with each 1D shear kernel.

Same shear plan for every kernel, so only the 1D interpolator differs. On the
noise-free series the residual vs the base is pure resampling error (plus, for
2D, the irreducible slice-profile mismatch). Reports per motion group:
  rms%   : brain RMS residual, % of brain mean
  tstd%  : median over brain voxels of the residual's temporal std, % of mean
           (spurious signal the resampler injects into the time series)
  ring%  : RMS residual in a 2-6 voxel shell outside the brain, % of brain mean
"""

import json
import re
import sys
from collections import defaultdict

import nibabel as nb
import numpy as np
import torch
from scipy import ndimage

from fastfuncstuff.processing.shear import shear_resample
from fastfuncstuff.processing.shear_triton import shear_resample_triton

FLIP = np.diag([-1.0, -1.0, 1.0, 1.0])


def main():
    prefix, series = sys.argv[1], sys.argv[2]
    modes = (
        sys.argv[3].split(",")
        if len(sys.argv) > 3
        else ["heptic", "wsinc5", "fourier", "fourier3d"]
    )
    meta = json.load(open(f"{prefix}_meta.json"))
    tags = meta["tags"]
    img = nb.load(series)
    data = np.asarray(img.dataobj, dtype=np.float32)  # (x,y,z,t)
    mask = np.asarray(nb.load(f"{prefix}_mask.nii.gz").dataobj) > 0
    shell = ndimage.binary_dilation(mask, iterations=6) & ~ndimage.binary_dilation(
        mask, iterations=2
    )
    A = FLIP @ img.affine
    rows = np.loadtxt(f"{prefix}_truth.aff12.1D").reshape(-1, 3, 4)
    mats = np.tile(np.eye(4), (len(rows), 1, 1))
    mats[:, :3] = rows
    vox = np.linalg.inv(A) @ mats @ A  # base voxel -> source voxel, (i,j,k)
    src = torch.tensor(np.ascontiguousarray(data.transpose(3, 2, 1, 0)), device="cuda")  # (t,z,y,x)
    shape = tuple(src.shape[1:])
    base = data[..., 0]
    bm = float(base[mask].mean())
    groups = defaultdict(list)
    for i, t in enumerate(tags[1:], start=1):
        groups[re.sub(r"_[-\d.]+$", "", t)].append(i)
    print(f"{'kernel':10s} " + " ".join(f"{g:>22s}" for g in groups) + "   rms% | tstd% | ring%")
    for mode in modes:
        out = []
        for c in range(0, src.shape[0], 64):
            m = torch.tensor(vox[c : c + 64], device="cuda", dtype=torch.float32)
            w, valid = shear_resample_triton(src[c : c + 64].contiguous(), m, shape, mode)
            for j in (~valid).nonzero(as_tuple=True)[0].tolist():  # exact single-axis: Paeth path
                w[j] = shear_resample(src[c + j], m[j], shape, mode)
            out.append(w.cpu().numpy())
        corr = np.concatenate(out).transpose(3, 2, 1, 0)  # (x,y,z,t)
        res = corr - base[..., None]
        cells = []
        for idx in groups.values():
            r = res[..., idx]
            rms = np.sqrt((r[mask] ** 2).mean()) / bm * 100
            tstd = np.median(r[mask].std(-1)) / bm * 100
            ring = np.sqrt((r[shell] ** 2).mean()) / bm * 100
            cells.append(f"{rms:6.3f}|{tstd:6.3f}|{ring:6.3f}")
        print(f"{mode:10s} " + " ".join(f"{c:>22s}" for c in cells))


if __name__ == "__main__":
    main()
