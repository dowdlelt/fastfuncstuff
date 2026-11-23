"""ffs_fc validation: parity with AFNI and nilearn, run cross-validation, speed.

Produces every number docs/fc.md reports, into docs/fc/numbers.json (merged per
stage, so a stage can be rerun alone) and the figures into docs/fc/.

    python scripts/fc_validation/validate.py [--stages a,b,...] [--work DIR]

Stages:
  prepare      Glasser (HCPMMP1) labels + brain mask on the TASKFORCE grid; censor files
  netcorr      3dNetCorr parity (ds005165 rest, 345 box ROIs): CC/FZ, maps, PC case
  tproject     3dTproject + 3dNetCorr parity for cleaning (polort, per-run ort,
               passband, censoring), TASKFORCE 2 runs, Glasser
  tcorr1d      3dTcorr1D parity for -seed_1D maps
  nilearn      NiftiLabelsMasker + signal.clean + corrcoef parity and speed
  fft          FFT cleaning == regression cleaning on real data; speed of each
  xval         10 TASKFORCE rest runs: LORO reproducibility per cleaning variant,
               CV shrinkage vs Ledoit-Wolf (dof / TR) by held-out likelihood
  loro_speed   leave-one-run-out: one ffs_fc pass vs a fold loop (in-memory and per
               tool call) vs 3dTproject + 3dNetCorr per fold
  speed        ffs_fc CPU vs GPU on the 10-run Glasser problem, plus the read alone
  edge         edge-series identities and RSS on real data
  efc          edge communities vs the Faskowitz et al. (2020) MATLAB reference on its
               example data (needs MATLAB and $EFC_DEMO); timing and a figure on
               TASKFORCE Glasser 360
  figures      docs figures from the saved outputs

Reference runs are slow and need not be repeated: netcorr, tproject, tcorr1d,
nilearn and loro_speed (~45 min, most of it the per-fold AFNI chain, which writes
~26 GB to --work). Rerun them only after a change to the cleaning or the matrix code;
xval, edge, fft, speed and figures are ffs-only and take a few minutes.

Data: $FFS_BENCHMARK_DATA_DIR (ds005165) and the TASKFORCE derivatives (anat space,
1.6 mm, TR 1 s, 10 rest runs over 7 sessions) with the FreeSurfer subject
TASKFORCE_sub-01_presurfer in $SUBJECTS_DIR.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs" / "fc"
NUMBERS = DOCS / "numbers.json"
PY = sys.executable
AFNI = Path(os.environ.get("AFNI_BIN", "/opt/mrisoftware/abin"))
TF = Path("/mnt/belegost/Projects/TASKFORCE/derivatives/global_proc")
TF_RUNS = [
    ("01", "01"), ("01", "02"), ("02", "01"), ("02", "02"), ("03", "01"),
    ("04", "01"), ("05", "01"), ("06", "01"), ("06", "02"), ("07", "01"),
]  # fmt: skip
FS_SUBJ = "TASKFORCE_sub-01_presurfer"
DS5165 = Path(os.environ.get("FFS_BENCHMARK_DATA_DIR", "")) / "ds005165-download/processing"
DEVICE = os.environ.get("DEVICE", "cuda")
CENSOR_MM = 0.3


def tf_run(ses: str, run: str) -> Path:
    return TF / f"ses-{ses}_task-rest_run{run}_final.nii.gz"


def tf_motion(ses: str, run: str) -> Path:
    return TF / f"step02_mot_ses-{ses}_task-rest_run{run}_demean.1D"


# -- plumbing ---------------------------------------------------------------------


def save_numbers(stage: str, values: dict) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    data = json.loads(NUMBERS.read_text()) if NUMBERS.exists() else {}
    data[stage] = values
    NUMBERS.write_text(json.dumps(data, indent=2, default=_jsonable) + "\n")
    print(f"[{stage}] " + json.dumps(values, default=_jsonable)[:600])


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def run(cmd: list, quiet: bool = True, env: dict | None = None) -> float:
    """Run a command, return wall seconds; raise with its tail on failure."""
    t0 = time.perf_counter()
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=env)
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{(p.stdout + p.stderr)[-3000:]}")
    if not quiet:
        print(p.stdout[-2000:])
    return dt


def ffs_fc(args: list, device: str = DEVICE) -> float:
    return run([PY, "-m", "fastfuncstuff.cli.fc", *args, "-device", device, "-quiet"])


def upper(m: np.ndarray) -> np.ndarray:
    return m[np.triu_indices(m.shape[0], 1)]


def matrix_agreement(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(upper(a) - upper(b))
    return {
        "max_abs_diff": float(d.max()),
        "median_abs_diff": float(np.median(d)),
        "r": float(np.corrcoef(upper(a), upper(b))[0, 1]),
    }


def gpu_free() -> bool:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout  # fmt: skip
        return int(out.split()[0]) < 20
    except Exception:
        return False


# -- stages -----------------------------------------------------------------------


def stage_prepare(work: Path) -> None:
    """Glasser labels and a brain mask on the TASKFORCE grid, plus censor files."""
    import nibabel as nib

    from fastfuncstuff.processing.censor import motion_enorm

    sd = os.environ.get("SUBJECTS_DIR", "/mnt/belegost/Data/fs_subjects")
    env = {**os.environ, "SUBJECTS_DIR": sd}
    fs_lab, fs_mask = work / "glasser_fs.nii.gz", work / "brainmask_fs.nii.gz"
    if not (work / "glasser.nii.gz").exists():
        run(["mri_aparc2aseg", "--s", FS_SUBJ, "--annot", "HCPMMP1", "--o", work / "hcp_fs.nii.gz"],
            env=env)  # fmt: skip
        img = nib.load(work / "hcp_fs.nii.gz")
        a = np.asarray(img.dataobj).astype(np.int32)
        ctx = np.unique(a)
        ctx = ctx[(ctx > 1000) & (ctx < 3000) & (ctx % 1000 != 0)]
        nib.save(nib.Nifti1Image(np.where(np.isin(a, ctx), a, 0).astype(np.int32), img.affine,
                                 img.header), fs_lab)  # fmt: skip
        run(["mri_convert", Path(sd) / FS_SUBJ / "mri/brainmask.mgz", fs_mask], env=env)
        master = f"{tf_run(*TF_RUNS[0])}[0]"
        for src, dst in ((fs_lab, "glasser.nii.gz"), (fs_mask, "brainmask.nii.gz")):
            run([PY, "-m", "fastfuncstuff.cli.util_resample", "-input", src, "-master", master,
                 "-rmode", "NN", "-prefix", work / dst])  # fmt: skip
    keep_all, frac = [], []
    for ses, r in TF_RUNS:
        mot = np.loadtxt(tf_motion(ses, r))
        keep = (motion_enorm(mot) <= CENSOR_MM).astype(int)
        np.savetxt(work / f"censor_{ses}_{r}.1D", keep, fmt="%d")
        keep_all.append(keep)
        frac.append(1 - keep.mean())
    lab = np.asarray(nib.load(work / "glasser.nii.gz").dataobj)
    save_numbers("prepare", {
        "n_runs": len(TF_RUNS), "n_rois": int(len(np.unique(lab)) - 1),
        "censor_threshold_mm": CENSOR_MM, "censored_fraction_per_run": frac,
        "trs_per_run": int(len(keep_all[0])),
    })  # fmt: skip


def stage_netcorr(work: Path) -> None:
    """3dNetCorr parity on a public run: one run, raw means, no detrend."""
    import nibabel as nib

    from fastfuncstuff.connectivity.netcc import read_netcc

    d = work / "netcorr"
    d.mkdir(exist_ok=True)
    src = DS5165 / "ffs_mni_task-rest_run-1.nii.gz"
    img = nib.load(src)
    if not (d / "atlas.nii.gz").exists():
        x = np.asarray(img.dataobj[..., :20], dtype=np.float32)
        m = x.std(-1) > 0
        m &= x.mean(-1) > 0.2 * np.percentile(x.mean(-1)[m], 90)
        nib.save(nib.Nifti1Image(m.astype(np.uint8), img.affine), d / "mask.nii.gz")
        ii, jj, kk = np.indices(m.shape)
        lab = (ii // 8) * 10000 + (jj // 8) * 100 + (kk // 8) + 1
        _, inv = np.unique(np.where(m, lab, 0), return_inverse=True)
        nib.save(nib.Nifti1Image(inv.reshape(m.shape).astype(np.int32), img.affine),
                 d / "atlas.nii.gz")  # fmt: skip
    for f in d.glob("afni_000*"):
        shutil.rmtree(f) if f.is_dir() else f.unlink()
    t_afni = run([AFNI / "3dNetCorr", "-inset", src, "-in_rois", d / "atlas.nii.gz", "-mask",
                  d / "mask.nii.gz", "-fish_z", "-part_corr", "-ts_wb_corr", "-nifti", "-prefix",
                  d / "afni"])  # fmt: skip
    common = ["-input", src, "-atlas", d / "atlas.nii.gz", "-mask", d / "mask.nii.gz",
              "-polort", "-1", "-no_scale", "-wb_corr", "-prefix"]  # fmt: skip
    t_cpu = ffs_fc([*common, d / "ffs_cpu"], device="cpu")
    t_gpu = ffs_fc([*common, d / "ffs_gpu"], device="cuda") if gpu_free() else None
    A, aid, _ = read_netcc(d / "afni_000.netcc")
    F, fid, _ = read_netcc(d / "ffs_cpu.netcc")
    full = np.load(d / "ffs_cpu_fc.npz")
    wb = np.asarray(nib.load(d / "ffs_cpu_wb_r.nii.gz").dataobj)
    mask = np.asarray(nib.load(d / "mask.nii.gz").dataobj) > 0
    files = sorted((d / "afni_000_INDIV").glob("WB_CORR_ROI_*.nii.gz"))
    worst = 0.0
    for i in range(len(files)):
        a = np.asarray(nib.load(files[i]).dataobj)
        worst = max(worst, float(np.abs(a[mask] - wb[..., i][mask]).max()))
    save_numbers("netcorr", {
        "data": "ds005165 rest run 1 (MNI 3 mm, 212 TRs)", "n_rois": len(aid),
        "ids_equal": aid == fid,
        "cc_max_abs_diff_netcc": float(np.abs(A["CC"] - F["CC"]).max()),
        "cc_max_abs_diff_vs_full_precision": float(np.abs(A["CC"] - full["CC"]).max()),
        "fz_max_abs_diff": float(np.abs(A["FZ"] - F["FZ"]).max()),
        "wb_maps_max_abs_diff_all_rois": worst,
        "afni_pc_block_all_zero": bool(np.all(A["PC"] == 0)),
        "time_3dnetcorr_s": t_afni, "time_ffs_cpu_s": t_cpu, "time_ffs_gpu_s": t_gpu,
    })  # fmt: skip


def _per_run_ort(work: Path, runs: list) -> Path:
    """Motion as AFNI wants it for per-run nuisance: zero-padded per-run columns."""
    mots = [np.loadtxt(tf_motion(*r)) for r in runs]
    n = sum(len(m) for m in mots)
    out = np.zeros((n, 6 * len(runs)))
    pos = 0
    for k, m in enumerate(mots):
        out[pos : pos + len(m), 6 * k : 6 * k + 6] = m - m.mean(0)
        pos += len(m)
    path = work / f"ort_{len(runs)}runs.1D"
    np.savetxt(path, out, fmt="%.6f")
    return path


def _ffs_ortvec_run(runs: list) -> list:
    out = []
    for k, r in enumerate(runs):
        out += ["-ortvec_run", tf_motion(*r), "motion", str(k + 1)]
    return out


def stage_tproject(work: Path) -> None:
    """Cleaning parity: 3dTproject then 3dNetCorr vs one ffs_fc call."""
    from fastfuncstuff.connectivity.netcc import read_netcc

    d = work / "tproject"
    d.mkdir(exist_ok=True)
    runs = TF_RUNS[:2]
    cat = d / "cat.nii.gz"
    if not cat.exists():
        run([AFNI / "3dTcat", "-prefix", cat, *[tf_run(*r) for r in runs]])
    ort = _per_run_ort(d, runs)
    censor = np.concatenate([np.loadtxt(work / f"censor_{s}_{r}.1D") for s, r in runs])
    # Motion censors <1% here; a fixed extra pattern (every 17th TR) exercises the
    # censored-gap path properly. Parity is the point, not the motion.
    censor[::17] = 0
    np.savetxt(d / "censor.1D", censor, fmt="%d")
    (d / "concat.1D").write_text("0 360\n")
    results = {}
    variants = {
        "polort2": ([], []),
        "polort2_motion": (["-ort", ort], _ffs_ortvec_run(runs)),
        "polort2_motion_passband": (["-ort", ort, "-passband", "0.01", "0.1"],
                                    [*_ffs_ortvec_run(runs), "-passband", "0.01", "0.1"]),
        "polort2_motion_highpass_censor": (
            ["-ort", ort, "-passband", "0.01", "99", "-censor", d / "censor.1D", "-cenmode",
             "KILL"],
            # "0.01 99" on both sides: 3dTproject's clamped upper stopband still rounds
            # onto the Nyquist bin, and the shared edge rule reproduces that.
            [*_ffs_ortvec_run(runs), "-passband", "0.01", "99", "-censor", d / "censor.1D"]),
    }  # fmt: skip
    for name, (afni_extra, ffs_extra) in variants.items():
        clean = d / f"tp_{name}.nii.gz"
        for f in d.glob(f"afni_{name}_000*"):
            shutil.rmtree(f) if f.is_dir() else f.unlink()
        t_tp = run([AFNI / "3dTproject", "-overwrite", "-input", cat, "-concat", d / "concat.1D",
                    "-polort", "2", "-mask", work / "brainmask.nii.gz", "-prefix", clean,
                    *afni_extra])  # fmt: skip
        t_nc = run([AFNI / "3dNetCorr", "-inset", clean, "-in_rois", work / "glasser.nii.gz",
                    "-mask", work / "brainmask.nii.gz", "-prefix", d / f"afni_{name}"])  # fmt: skip
        t_ffs = ffs_fc(["-input", *[tf_run(*r) for r in runs], "-atlas", work / "glasser.nii.gz",
                        "-mask", work / "brainmask.nii.gz", "-polort", "2", "-no_scale",
                        *ffs_extra, "-prefix", d / f"ffs_{name}"])  # fmt: skip
        A, _, _ = read_netcc(d / f"afni_{name}_000.netcc")
        F = np.load(d / f"ffs_{name}_fc.npz")["CC"]
        meta = json.loads((d / f"ffs_{name}.json").read_text())
        results[name] = {**matrix_agreement(A["CC"], F), "dof": meta["dof"],
                         "time_afni_s": t_tp + t_nc, "time_ffs_s": t_ffs}  # fmt: skip
    save_numbers("tproject", results)


def stage_tcorr1d(work: Path) -> None:
    """-seed_1D maps vs 3dTcorr1D on the same cleaned series."""
    import nibabel as nib

    d = work / "tcorr1d"
    d.mkdir(exist_ok=True)
    clean = work / "tproject" / "tp_polort2_motion.nii.gz"
    lab = np.asarray(nib.load(work / "glasser.nii.gz").dataobj)
    data = np.asarray(nib.load(clean).dataobj, dtype=np.float64)
    seeds = np.stack([data[lab == v].mean(0) for v in (1001, 1004, 2023)], 1)  # V1, V4-ish, ...
    np.savetxt(d / "seeds.1D", seeds)
    t_afni = run([AFNI / "3dTcorr1D", "-overwrite", "-pearson", "-mask",
                  work / "brainmask.nii.gz", "-prefix", d / "afni.nii.gz", clean,
                  d / "seeds.1D"])  # fmt: skip
    t_ffs = ffs_fc(["-input", clean, "-seed_1D", d / "seeds.1D", "-mask",
                    work / "brainmask.nii.gz", "-polort", "-1", "-no_scale", "-wb_corr",
                    "-prefix", d / "ffs"])  # fmt: skip
    a = np.asarray(nib.load(d / "afni.nii.gz").dataobj).squeeze()
    f = np.asarray(nib.load(d / "ffs_wb_r.nii.gz").dataobj)
    m = (np.asarray(nib.load(work / "brainmask.nii.gz").dataobj) > 0) & (data.std(-1) > 0)
    save_numbers("tcorr1d", {
        "n_seeds": 3, "max_abs_diff": float(np.abs(a[m] - f[m]).max()),
        "time_3dtcorr1d_s": t_afni, "time_ffs_s": t_ffs,
    })  # fmt: skip


def stage_nilearn(work: Path) -> None:
    """nilearn's standard route, matched setting: labels mean, linear detrend + motion."""
    import nibabel as nib
    from nilearn.maskers import NiftiLabelsMasker
    from nilearn.signal import clean

    d = work / "nilearn"
    d.mkdir(exist_ok=True)
    runs = TF_RUNS[:2]
    t0 = time.perf_counter()
    masker = NiftiLabelsMasker(str(work / "glasser.nii.gz"), mask_img=str(work /
                               "brainmask.nii.gz"), standardize=False, resampling_target=None)  # fmt: skip
    parts = []
    for r in runs:
        ts = masker.fit_transform(str(tf_run(*r)))
        parts.append(clean(ts, detrend=True, confounds=np.loadtxt(tf_motion(*r)),
                           standardize=False, standardize_confounds=False, t_r=1.0))  # fmt: skip
    r_nl = np.corrcoef(np.concatenate(parts, 0).T)
    t_nl = time.perf_counter() - t0
    t_ffs = ffs_fc(["-input", *[tf_run(*r) for r in runs], "-atlas", work / "glasser.nii.gz",
                    "-mask", work / "brainmask.nii.gz", "-polort", "1", "-no_scale",
                    *_ffs_ortvec_run(runs), "-prefix", d / "ffs"])  # fmt: skip
    f = np.load(d / "ffs_fc.npz")
    labels_nl = np.asarray(masker.labels_[1:] if len(masker.labels_) > r_nl.shape[0]
                           else masker.labels_, dtype=int)  # fmt: skip
    order = np.searchsorted(labels_nl, f["ids"])
    save_numbers("nilearn", {
        "version": __import__("nilearn").__version__,
        **matrix_agreement(r_nl[np.ix_(order, order)], f["CC"]),
        "time_nilearn_s": t_nl, "time_ffs_s": t_ffs,
        "n_rois": int(r_nl.shape[0]), "n_runs": len(runs),
    })  # fmt: skip
    del nib


def stage_fft(work: Path) -> None:
    """The two cleaning algorithms on real ROI series: same residual, same dof."""
    import nibabel as nib
    import torch

    from fastfuncstuff.cli_utils import assemble_per_run_nuisance, load_and_preprocess_runs
    from fastfuncstuff.connectivity.prep import build_cleaner
    from fastfuncstuff.processing.roi_collapse import build_roi_set

    runs = TF_RUNS
    mask = np.asarray(nib.load(work / "brainmask.nii.gz").dataobj) > 0
    rs = build_roi_set(str(work / "glasser.nii.gz"), mask.shape, mask=mask, restrict_to_rois=False)
    dev = torch.device(DEVICE if gpu_free() or DEVICE == "cpu" else "cpu")
    lr = load_and_preprocess_runs([str(tf_run(*r)) for r in runs], mask_array=rs.rows,
                                  device=dev, verbose=False)  # fmt: skip
    vox = lr.data
    nuis = assemble_per_run_nuisance([], lr.run_starts, lr.n_timepoints, 2, torch.device("cpu"))
    out = {}
    for method in ("regression", "fft"):
        cl = build_cleaner(nuis.per_run, lr.run_starts, lr.n_timepoints, tr=1.0,
                           passband=(0.01, 0.1), method=method)  # fmt: skip
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = cl.apply(vox, device=dev, out_device=torch.device("cpu"))
        if dev.type == "cuda":
            torch.cuda.synchronize()
        out[method] = (res, cl.dof, time.perf_counter() - t0)
    a, b = out["regression"][0], out["fft"][0]
    rel = float((a - b).norm() / a.norm())
    save_numbers("fft", {
        "device": dev.type, "n_voxels": int(vox.shape[0]), "n_timepoints": int(lr.n_timepoints),
        "dof_regression": out["regression"][1], "dof_fft": out["fft"][1],
        "relative_residual_difference": rel,
        "time_regression_s": out["regression"][2], "time_fft_s": out["fft"][2],
    })  # fmt: skip


XVAL_VARIANTS = {
    "drift": [],
    "drift+motion": ["MOTION"],
    "drift+motion+highpass": ["MOTION", "-passband", "0.01", "0"],
    "drift+motion+bandpass": ["MOTION", "-passband", "0.01", "0.1"],
    "drift+motion+highpass+censor": ["MOTION", "-passband", "0.01", "0", "CENSOR"],
}


def _xval_args(work: Path, spec: list, out: Path) -> list:
    runs = TF_RUNS
    args = ["-input", *[tf_run(*r) for r in runs], "-atlas", work / "glasser.nii.gz", "-mask",
            work / "brainmask.nii.gz", "-loro", "-per_run", "-shrink", "cv", "-prefix", out]  # fmt: skip
    for tok in spec:
        if tok == "MOTION":
            args += _ffs_ortvec_run(runs)
        elif tok == "CENSOR":
            c = np.concatenate([np.loadtxt(work / f"censor_{s}_{r}.1D") for s, r in runs])
            np.savetxt(work / "censor_all.1D", c, fmt="%d")
            args += ["-censor", work / "censor_all.1D"]
        else:
            args.append(tok)
    return args


def stage_xval(work: Path) -> None:
    d = work / "xval"
    d.mkdir(exist_ok=True)
    res = {}
    for name, spec in XVAL_VARIANTS.items():
        out = d / name.replace("+", "_")
        t = ffs_fc(_xval_args(work, spec, out))
        meta = json.loads(Path(f"{out}.json").read_text())
        rep = np.loadtxt(f"{out}_loro.tsv", skiprows=1, usecols=3)
        res[name] = {
            "dof": meta["dof"], "n_kept": meta["n_kept"],
            "loro_mean": meta["loro_mean"], "loro_min": float(rep.min()),
            "loro_max": float(rep.max()), "shrink_cv": meta["shrink_cv"], "time_s": t,
        }  # fmt: skip
    save_numbers("xval", res)


def stage_loro_speed(work: Path) -> None:
    """One pass with -loro against the fold loops people actually run."""
    import torch

    d = work / "loro_speed"
    d.mkdir(exist_ok=True)
    runs = TF_RUNS
    base = ["-atlas", work / "glasser.nii.gz", "-mask", work / "brainmask.nii.gz",
            "-passband", "0.01", "0"]  # fmt: skip
    t_one = ffs_fc(["-input", *[tf_run(*r) for r in runs], *base, *_ffs_ortvec_run(runs),
                    "-loro", "-prefix", d / "one"])  # fmt: skip
    # per-fold tool calls: train on 9 runs, and the held-out run alone (what a script does)
    t_tool = 0.0
    for k in range(len(runs)):
        train = [r for i, r in enumerate(runs) if i != k]
        t_tool += ffs_fc(["-input", *[tf_run(*r) for r in train], *base,
                          *_ffs_ortvec_run(train), "-prefix", d / f"fold{k}"])  # fmt: skip
        t_tool += ffs_fc(["-input", tf_run(*runs[k]), *base, *_ffs_ortvec_run([runs[k]]),
                          "-prefix", d / f"held{k}"])  # fmt: skip
    # equivalence of the folds
    one = np.load(d / "one_fc.npz") if (d / "one_fc.npz").exists() else None
    del one
    # in-memory: ROI series cleaned per run once, folds by re-concatenation vs sufficient stats
    from fastfuncstuff.connectivity.matrix import correlation_matrix
    from fastfuncstuff.connectivity.xval import cross_products, loro_reproducibility

    ts = np.loadtxt(d / "ts.tsv", skiprows=1)[:, 2:].T if (d / "ts.tsv").exists() else None
    if ts is None:
        ffs_fc(["-input", *[tf_run(*r) for r in runs], *base, *_ffs_ortvec_run(runs), "-ts_out",
                "-prefix", d / "ts_run"])  # fmt: skip
        ts = np.loadtxt(f"{d}/ts_run_ts.tsv", skiprows=1)[:, 2:].T
    x = torch.as_tensor(ts)
    starts = list(range(0, x.shape[1], 360))
    t0 = time.perf_counter()
    for _ in range(20):
        loro_reproducibility(cross_products(x, starts))
    t_suff = (time.perf_counter() - t0) / 20
    t0 = time.perf_counter()
    for _ in range(20):
        for k in range(len(starts)):
            keep = np.ones(x.shape[1], bool)
            keep[starts[k] : starts[k] + 360] = False
            correlation_matrix(x[:, keep])
            correlation_matrix(x[:, ~keep])
    t_concat = (time.perf_counter() - t0) / 20
    # the AFNI way: 3dTproject + 3dNetCorr per fold (train), measured on 2 folds, scaled
    t_afni_fold = []
    for k in range(2):
        train = [r for i, r in enumerate(runs) if i != k]
        cat = d / f"afni_cat{k}.nii.gz"
        run([AFNI / "3dTcat", "-overwrite", "-prefix", cat, *[tf_run(*r) for r in train]])
        ort = _per_run_ort(d, train)
        starts_t = " ".join(str(360 * i) for i in range(len(train)))
        (d / "concat9.1D").write_text(starts_t + "\n")
        t = run([AFNI / "3dTproject", "-overwrite", "-input", cat, "-concat", d / "concat9.1D",
                 "-polort", "2", "-ort", ort, "-passband", "0.01", "99", "-mask",
                 work / "brainmask.nii.gz", "-prefix", d / f"afni_clean{k}.nii.gz"])  # fmt: skip
        t += run([AFNI / "3dNetCorr", "-overwrite", "-inset", d / f"afni_clean{k}.nii.gz",
                  "-in_rois", work / "glasser.nii.gz", "-mask", work / "brainmask.nii.gz",
                  "-prefix", d / f"afni_nc{k}"])  # fmt: skip
        t_afni_fold.append(t)
    save_numbers("loro_speed", {
        "n_runs": len(runs), "time_one_pass_s": t_one, "time_tool_per_fold_s": t_tool,
        "speedup_vs_tool_loop": t_tool / t_one,
        "in_memory_sufficient_stats_s": t_suff, "in_memory_reconcatenate_s": t_concat,
        "afni_train_fold_s_mean": float(np.mean(t_afni_fold)),
        "afni_train_folds_extrapolated_s": float(np.mean(t_afni_fold)) * len(runs),
        "afni_note": "train folds only (2 measured, scaled to 10); held-out runs not counted",
    })  # fmt: skip


def stage_speed(work: Path) -> None:
    d = work / "speed"
    d.mkdir(exist_ok=True)
    runs = TF_RUNS
    base = ["-input", *[tf_run(*r) for r in runs], "-atlas", work / "glasser.nii.gz", "-mask",
            work / "brainmask.nii.gz", *_ffs_ortvec_run(runs), "-passband", "0.01", "0"]  # fmt: skip
    cases = {"matrix": [], "matrix+maps": ["-wb_z"], "matrix+maps+edges": ["-wb_z", "-edge_ts"]}
    out = {}
    devices = ["cpu"] + (["cuda"] if gpu_free() else [])
    for name, extra in cases.items():
        for dev in devices:
            reps = [ffs_fc([*base, *extra, "-prefix", d / f"{name}_{dev}"], device=dev)
                    for _ in range(2)]  # fmt: skip
            out[f"{name}_{dev}_s"] = min(reps)
    # decompression alone, the floor every tool pays
    import nibabel as nib

    t0 = time.perf_counter()
    for r in runs:
        np.asarray(nib.load(tf_run(*r)).dataobj, dtype=np.float32)
    out["load_only_s"] = time.perf_counter() - t0
    save_numbers("speed", out)


def stage_edge(work: Path) -> None:
    d = work / "edge"
    d.mkdir(exist_ok=True)
    runs = TF_RUNS[:2]
    ffs_fc(["-input", *[tf_run(*r) for r in runs], "-atlas", work / "glasser.nii.gz", "-mask",
            work / "brainmask.nii.gz", *_ffs_ortvec_run(runs), "-passband", "0.01", "0",
            "-edge_ts", "-plot", "-prefix", d / "e"])  # fmt: skip
    cc = np.load(d / "e_fc.npz")["CC"]
    e = np.load(d / "e_edge_ts.npy", mmap_mode="r")
    iu = np.triu_indices(cc.shape[0], 1)
    sel = np.linspace(0, e.shape[0] - 1, 5000).astype(int)
    rss = np.loadtxt(d / "e_rss.tsv", skiprows=1)[:, 2]
    top = np.argsort(rss)[-int(0.05 * rss.size) :]
    r_top = np.asarray(e[sel][:, top]).mean(1)
    save_numbers("edge", {
        "edge_ts_shape": list(e.shape), "edge_ts_gb": e.nbytes / 1e9,
        "mean_edge_equals_r_max_abs_diff": float(np.abs(np.asarray(e[sel]).mean(1)
                                                        - cc[iu][sel]).max()),
        "rss2_share_top5pct_frames": float(np.sort(rss**2)[-top.size:].sum() / (rss**2).sum()),
        "r_top5pct_vs_full_r": float(np.corrcoef(r_top, cc[iu][sel])[0, 1]),
    })  # fmt: skip


EFC_DEMO = Path(os.environ.get("EFC_DEMO", Path.home() / "local_code/edge-centric_demo"))


def stage_efc(work: Path) -> None:
    import scipy.io as sio
    import torch
    from sklearn.metrics import adjusted_rand_score as ari

    from fastfuncstuff.connectivity.edge import EdgeCommunities, edge_communities, zscore_rows

    d = work / "efc"
    d.mkdir(exist_ok=True)
    ref_mat = d / "reference.mat"
    script = Path(__file__).with_name("efc_reference.m")
    run(["matlab", "-batch",
         f"demo='{EFC_DEMO}'; out_mat='{ref_mat}'; run('{script}')"])  # fmt: skip
    ref = sio.loadmat(ref_mat)
    ts = sio.loadmat(EFC_DEMO / "example_data.mat")["ts"].astype(np.float64)
    n_t, n = ts.shape
    iu, ju = np.triu_indices(n, 1)
    pos = {(a, b): e for e, (a, b) in enumerate(zip(iu, ju, strict=True))}
    to_ours = np.array([pos[(a - 1, b - 1)] for a, b in zip(ref["u"].ravel(), ref["v"].ravel(),
                                                            strict=True)])  # fmt: skip
    ci = np.zeros_like(ref["CI"])
    ci[to_ours] = ref["CI"]  # MATLAB's column-major edge order -> triu_indices order
    best = ci[:, 5]
    scale = ((n_t - 1) / n_t) ** 2  # MATLAB zscore (T-1): every point x (T-1)/T
    out = {
        "data": f"edge-centric_demo example ({n} ROIs, {n_t} TRs), k = 10",
        "matlab_inertia_single": ref["SUMD"].ravel()[:5].tolist(),
        "matlab_inertia_best_of_10": float(ref["SUMD"].ravel()[5]),
        "matlab_single_vs_best_ari": [float(ari(ci[:, s], best)) for s in range(5)],
        "matlab_time_single_s": float(ref["t_single"].item()),
        "matlab_time_best_of_10_s": float(ref["t_rep"].item()),
    }
    devices = ["cpu"] + (["cuda"] if gpu_free() else [])
    for dev in devices:
        z = zscore_rows(torch.as_tensor(ts.T).to(dev))
        t0 = time.perf_counter()
        com = edge_communities(z, 10)
        if dev == "cuda":
            torch.cuda.synchronize()
        out[f"ffs_{dev}_time_best_of_10_s"] = time.perf_counter() - t0
        out[f"ffs_{dev}_inertia_matlab_scale"] = com.inertia * scale
        out[f"ffs_{dev}_vs_matlab_best_ari"] = float(ari(com.labels, best))
    mine = EdgeCommunities(labels=best - 1, k=10, inertia=0.0)
    out["entropy_max_abs_diff"] = float(np.abs(mine.node_profile(n)[1] - ref["enorm"].ravel()).max())
    out["profilesim_max_abs_diff"] = float(np.abs(mine.profile_similarity(n) - ref["S"]).max())
    # real data: the whole ffs_fc job with -efc on two TASKFORCE runs, and the figure
    runs = TF_RUNS[:2]
    base = ["-input", *[tf_run(*r) for r in runs], "-atlas", work / "glasser.nii.gz", "-mask",
            work / "brainmask.nii.gz", *_ffs_ortvec_run(runs), "-passband", "0.01", "0",
            "-efc", "10"]  # fmt: skip
    for dev in devices:
        out[f"taskforce_job_{dev}_s"] = ffs_fc([*base, "-prefix", d / f"g360_{dev}"], device=dev)
    ffs_fc([*base, "-plot", "-prefix", d / "g360_plot"], device=devices[-1])
    shutil.copy(d / "g360_plot_efc.png", DOCS / "efc.png")
    meta = json.loads((d / "g360_plot.json").read_text())
    out["taskforce"] = {"n_rois": 360, "n_trs": 720, **meta["efc"]}
    save_numbers("efc", out)


def stage_figures(work: Path) -> None:
    """Figures for the docs page."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    from fastfuncstuff.connectivity.plots import plot_fc
    from fastfuncstuff.dynamics.plots import _INK, _INK2, _SURFACE, _style_axes

    DOCS.mkdir(parents=True, exist_ok=True)
    e = work / "edge"
    fc = np.load(e / "e_fc.npz")
    rss = np.loadtxt(e / "e_rss.tsv", skiprows=1)
    plot_fc(DOCS / "matrix_rss.png", fc["CC"], list(fc["names"]), rss=rss[:, 2], times=rss[:, 1],
            title="TASKFORCE rest, 2 runs, Glasser 360 (highpass 0.01 Hz, motion)")  # fmt: skip

    nums = json.loads(NUMBERS.read_text())
    xv = nums.get("xval", {})
    if xv:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=_SURFACE)
        names = list(xv)
        ax = axes[0]
        for i, n in enumerate(names):
            rep = np.loadtxt(work / "xval" / f"{n.replace('+', '_')}_loro.tsv", skiprows=1,
                             usecols=3)  # fmt: skip
            ax.scatter(np.full(rep.size, i) + np.linspace(-0.15, 0.15, rep.size), rep, s=14,
                       color="#2a78d6")  # fmt: skip
            ax.plot([i - 0.25, i + 0.25], [rep.mean()] * 2, color=_INK, lw=1.5)
        ax.set_xticks(range(len(names)), [n.replace("+", "\n+") for n in names], fontsize=7)
        ax.set_ylabel("held-out run vs rest, r (Fisher z)", fontsize=8)
        ax.set_title("Leave-one-run-out reproducibility (10 runs)", fontsize=10, color=_INK)
        _style_axes(ax)
        ax = axes[1]
        spans = []
        for n, col in (("drift+motion+highpass", "#2a78d6"), ("drift+motion+bandpass", "#e34948")):
            curve = np.loadtxt(work / "xval" / f"{n.replace('+', '_')}_shrink_cv.tsv",
                               skiprows=1, usecols=(0, 1))  # fmt: skip
            a, ll = curve[:, 0], curve[:, 1] - curve[:, 1].max()
            spans.append(ll)
            info = xv[n]["shrink_cv"]
            label = n.split("+")[-1]
            ax.plot(a, ll, color=col, lw=1.6, label=f"{label}: CV {info['best']:.2f}")
            ax.plot(info["best"], 0.0, "o", color=col, ms=6)
            for key, mk in (("lw_dof", "v"), ("lw_tr", "x")):
                lw_a = info[key]["shrinkage"]
                ax.plot(lw_a, np.interp(lw_a, a, ll), mk, color=col, ms=6, mfc="none")
        ax.plot([], [], "kv", mfc="none", label="Ledoit-Wolf, n = dof")
        ax.plot([], [], "kx", label="Ledoit-Wolf, n = TRs")
        ax.set_xlim(0, 0.3)
        lo = min(float(np.ptp(c[a <= 0.3])) for c in spans)
        ax.set_ylim(-1.1 * lo, 0.05 * lo)
        ax.set_xlabel("shrinkage toward identity", fontsize=8)
        ax.set_ylabel("held-out log-likelihood per TR\n(relative to the best)", fontsize=8)
        ax.set_title("Shrinkage chosen by held-out runs", fontsize=10, color=_INK)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        ax.tick_params(colors=_INK2)
        _style_axes(ax)
        fig.tight_layout()
        fig.savefig(DOCS / "xval.png", dpi=150, facecolor=_SURFACE)
        plt.close(fig)


STAGES = {
    "prepare": stage_prepare, "netcorr": stage_netcorr, "tproject": stage_tproject,
    "tcorr1d": stage_tcorr1d, "nilearn": stage_nilearn, "fft": stage_fft, "xval": stage_xval,
    "loro_speed": stage_loro_speed, "speed": stage_speed, "edge": stage_edge,
    "efc": stage_efc, "figures": stage_figures,
}  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--work", default=os.environ.get("FC_WORK", str(REPO / "scratch_fc")))
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    for name in a.stages.split(","):
        t0 = time.perf_counter()
        STAGES[name](work)
        print(f"== {name} done in {time.perf_counter() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
