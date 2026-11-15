"""Write the analyzePRF vertex set as an ffs surface mask (mask.{lh,rh}.shape.gii),
so the parity fit runs on exactly the vertices analyzePRF saw."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.io as sio

from fastfuncstuff.io.gifti import save_gifti_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("apr_input", type=Path)
    args = parser.parse_args()
    m = sio.loadmat(args.apr_input, variable_names=["lh_index", "rh_index"])
    meta = json.loads((args.work / "meta.json").read_text())
    for hemi in ("lh", "rh"):
        mask = np.zeros(meta["n_vertices"][hemi], np.float32)
        mask[m[f"{hemi}_index"].ravel() - 1] = 1
        run_meta = dict(nib.load(args.work / "runs" / f"run-01.{hemi}.func.gii").meta)
        keep = {k: run_meta[k] for k in ("mesh_fingerprint", "geometry") if k in run_meta}
        save_gifti_data(args.work / "runs" / f"aprmask.{hemi}.shape.gii", mask, meta=keep)
        print(hemi, int(mask.sum()))


if __name__ == "__main__":
    main()
