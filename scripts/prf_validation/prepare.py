"""Turn the downloaded ds003787 files into what ffs_pyrf and analyzePRF both read.

Inputs (from ``run.sh``'s download): fmriprep fsnative ``.func.mgz`` runs, the
stimulus ``.mat`` apertures, the V1-V4 ROI labels and the vistasoft fits.

Outputs in WORK:
  runs/run-NN.{lh,rh}.func.gii   per-hemisphere series for ffs_pyrf (-hemi_pair auto)
  aperture.npy                   row x col x time, exactly the array MATLAB loads
  meta.json                      vertex counts per hemisphere, TR, run list

Both tools see the same numbers: the GIfTI holds the mgz values as float32, and
run_analyzeprf.m reads those GIfTI files back rather than the mgz.
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.io as sio

from fastfuncstuff.io.gifti import mesh_fingerprint, save_gifti_data, save_gifti_surface

SUB = "sub-wlsubj001"
SES = "ses-nyu3t01"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="ds003787-download root")
    parser.add_argument("work", type=Path, help="output directory")
    parser.add_argument("--runs", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    args = parser.parse_args()

    func = args.data / "derivatives/fmriprep" / SUB / SES / "func"
    aper = args.data / "derivatives/stimulus_apertures" / SUB / SES
    runs_dir = args.work / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # ffs keeps surface data tied to its mesh (a vertex count alone can collide
    # with an unrelated mesh), so each run names the midthickness it lives on.
    fs_surf = args.data / "derivatives/freesurfer" / SUB / "surf"
    mesh_meta = {}
    for hemi in ("lh", "rh"):
        white, faces = nib.freesurfer.read_geometry(fs_surf / f"{hemi}.white")
        pial, _ = nib.freesurfer.read_geometry(fs_surf / f"{hemi}.pial")
        geom = runs_dir / f"{hemi}.midthickness.surf.gii"
        save_gifti_surface(geom, ((white + pial) / 2).astype(np.float32), faces.astype(np.int32))
        mesh_meta[hemi] = {
            "mesh_fingerprint": mesh_fingerprint(faces, len(white)),
            "geometry": geom.name,
        }

    tr = json.loads(
        (args.data / f"{SUB}/{SES}/func/{SUB}_{SES}_task-prf_acq-PA_run-01_bold.json").read_text()
    )["RepetitionTime"]
    n_vertices = {}
    for run in args.runs:
        for hemi, letter in (("lh", "L"), ("rh", "R")):
            src = (
                func
                / f"{SUB}_{SES}_task-prf_acq-PA_run-{run:02d}_space-fsnative_hemi-{letter}.func.mgz"
            )
            dst = runs_dir / f"run-{run:02d}.{hemi}.func.gii"
            data = np.asarray(nib.load(src).dataobj, dtype=np.float32)
            data = data.reshape(data.shape[0], -1)  # (V, 1, 1, T) -> (V, T)
            n_vertices[hemi] = int(data.shape[0])
            if not dst.exists():
                save_gifti_data(
                    dst, data, meta={"source": src.name, "TR_seconds": str(tr), **mesh_meta[hemi]}
                )
            print(f"{dst.name}: {data.shape}")

    apertures = [
        sio.loadmat(aper / f"{SUB}_{SES}_task-prf_run-{run:02d}_aperture.mat")["stimulus"]
        for run in args.runs
    ]
    # Every run of this protocol is the same sweep; say so rather than assume it.
    assert all(np.array_equal(apertures[0], a) for a in apertures[1:])
    np.save(args.work / "aperture.npy", apertures[0].astype(np.float32))

    meta = {
        "runs": args.runs,
        "n_vertices": n_vertices,
        "tr": tr,
        "n_frames": apertures[0].shape[-1],
    }
    (args.work / "meta.json").write_text(json.dumps(meta, indent=2))
    print(meta)


if __name__ == "__main__":
    main()
